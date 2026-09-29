import os
import torch
import pandas as pd
import numpy as np
import re
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import BatchEncoding
import random
from typing import Tuple, Dict, Any, List
import argparse
import warnings
warnings.filterwarnings('ignore')

from Mol_LLaMA.models.mol_llama import MolLLaMA
from torch.nn import functional as F
from Mol_LLaMA.models.mol_llama import get_mol_graphs_from_preprocessed_mols

from transformers import EsmModel, LlamaForCausalLM
from Prot2Text_V2.models import (
    ModalityAdapter,
    ModalityAdapterConfig,
    Esm2LlamaInstructForCausalLM
)

device = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 330


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Utilities
# =========================================================

def normalize_key(k: str) -> str:
    k = k.lower()
    prefixes_to_remove = [
        'base_model.model.base_model.model.model.',
        'base_model.model.base_model.model.',
        'base_model.model.llm.base_model.model.',
        'base_model.model.llm.model.',
        'llm.base_model.model.model.',
        'llm.base_model.model.',
        'base_model.model.llama_decoder.model.',
        'llama_decoder.model.',
        'base_model.model.',
        'llm.model.',
        'llm.',
        'model.',
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes_to_remove:
            if k.startswith(prefix):
                k = k[len(prefix):]
                changed = True
                break
    k = k.replace('.default.weight', '.weight')
    k = k.replace('.default', '')
    k = k.replace('.lora_A.', '.lora_a.')
    k = k.replace('.lora_B.', '.lora_b.')
    return k


def mean_std_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sigma = max(float(np.nanstd(x)), eps)
    return (x - mu) / sigma


def softmax_3class(a, b, c, tau=0.5, eps=1e-12):
    tau = max(float(tau), 1e-8)
    a = np.asarray(a, dtype=float) / tau
    b = np.asarray(b, dtype=float) / tau
    c = np.asarray(c, dtype=float) / tau
    m = np.maximum(np.maximum(a, b), c)
    ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
    denom = ea + eb + ec + eps
    return ea / denom, eb / denom, ec / denom


def is_lora_param(name: str) -> bool:
    return "lora_" in name.lower()


def get_lora_named_params(model):
    return [(n, p) for n, p in model.named_parameters() if is_lora_param(n)]


def init_score_buffers(model) -> Dict[str, torch.Tensor]:
    buf = {}
    for n, p in get_lora_named_params(model):
        buf[normalize_key(n)] = torch.zeros_like(p, dtype=torch.float32).cpu()
    return buf


def parse_layer_range(s: str) -> List[int]:
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        raise ValueError("layers must be like '24-31' or '0-31'")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError("invalid layers range")
    return list(range(a, b + 1))


def map_layer_idx(hidden_states, layer, n_layers_expected=32):
    return layer + 1 if len(hidden_states) == n_layers_expected + 1 else layer


def _ensure_bfloat16(module):
    for param in module.parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    for buffer in module.buffers():
        if buffer.dtype != torch.bfloat16:
            buffer.data = buffer.data.to(torch.bfloat16)


def set_lora_requires_grad(llm_model, requires_grad: bool):
    for n, p in llm_model.named_parameters():
        if is_lora_param(n):
            p.requires_grad_(requires_grad)


def build_norm_to_actual_map(model) -> Dict[str, str]:
    m = {}
    for n, _ in model.named_parameters():
        if is_lora_param(n):
            nk = normalize_key(n)
            if nk not in m:
                m[nk] = n
    return m


def batch_iterator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# =========================================================
# Dataset Loading
# =========================================================

def load_all_data(data_dir):
    with open(os.path.join(data_dir, "protein.txt"), 'r') as f:
        proteins = [line.strip() for line in f if line.strip()]
    with open(os.path.join(data_dir, "drug.txt"), 'r') as f:
        molecules = [line.strip() for line in f if line.strip() and not line.startswith('===')]
    with open(os.path.join(data_dir, "cell_line.txt"), 'r') as f:
        cells = [line.strip() for line in f if line.strip()]
    with open(os.path.join(data_dir, "drug_3d.json"), 'r') as f:
        molecule_3d_dict = json.load(f)

    molecule_dict = {}
    for smiles, data in molecule_3d_dict.items():
        coords = data.get('coordinates', [])
        if coords and isinstance(coords, list):
            try:
                coords = np.array(coords, dtype=np.float32)
            except:
                coords = np.array([])
        molecule_dict[smiles] = {
            'smiles': smiles,
            'atoms': data.get('atoms', []),
            'coordinates': coords
        }

    print(f"Loaded {len(proteins)} proteins, {len(molecules)} molecules, {len(cells)} cell lines, {len(molecule_dict)} 3D structures")
    return proteins, molecules, cells, molecule_dict


def create_330_samples(proteins, molecules, cells):
    num_samples = min(330, len(proteins), len(molecules), len(cells))
    samples = [
        {'Protein': proteins[i], 'Drug': molecules[i], 'Cell': cells[i], 'sample_idx': i}
        for i in range(num_samples)
    ]
    print(f"Created {len(samples)} samples")
    return samples


# =========================================================
# Tokenizers & Model Loaders
# =========================================================

def setup_tokenizers(base_model_path, esm_path):
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    llama_tokenizer.bos_token = "<|begin_of_text|>"
    llama_tokenizer.eos_token = "<|eot_id|>"
    llama_tokenizer.pad_token = "<|eot_id|>"
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
    if '<mol>' not in llama_tokenizer.get_vocab():
        llama_tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
    llama_tokenizer.mol_token_id = llama_tokenizer.convert_tokens_to_ids('<mol>')
    llama_tokenizer.protein_placeholder_token_id = 128003
    terminators = [llama_tokenizer.eos_token_id, 128009]
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_path)
    return esm_tokenizer, llama_tokenizer, terminators


def load_prot2text_with_lora(esm_path, llama_path, base_checkpoint_path, lora_checkpoint_path):
    esm_encoder = EsmModel.from_pretrained(
        esm_path, add_pooling_layer=False, torch_dtype=torch.bfloat16, device_map=device
    )
    llama_decoder = LlamaForCausalLM.from_pretrained(
        llama_path, torch_dtype=torch.bfloat16, device_map=device
    )
    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size, intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config)
    adapter.to(torch.bfloat16).to(device)

    prot2text_model = Esm2LlamaInstructForCausalLM(
        esm_encoder=esm_encoder, adapter=adapter, llama_decoder=llama_decoder,
    )
    if os.path.exists(base_checkpoint_path):
        model_state_dict = torch.load(base_checkpoint_path, weights_only=True, map_location=device)
        prot2text_model.load_state_dict(model_state_dict)
    if os.path.exists(lora_checkpoint_path):
        prot2text_with_lora = PeftModel.from_pretrained(prot2text_model, lora_checkpoint_path, is_trainable=False)
    else:
        lora_config = LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.1, bias="none", init_lora_weights=True,
            target_modules=[
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
            ],
            modules_to_save=["adapter.fc1", "adapter.fc2"]
        )
        prot2text_with_lora = get_peft_model(prot2text_model, lora_config)
    return prot2text_with_lora


def load_cell_lora(cell_lora_path, base_llama_path):
    base_model = LlamaForCausalLM.from_pretrained(base_llama_path, torch_dtype=torch.bfloat16, device_map=device)
    cell_model = PeftModel.from_pretrained(base_model, cell_lora_path, torch_dtype=torch.bfloat16)
    cell_model.eval()
    return cell_model


def get_noLoRA_models(paths):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])

    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))

    mol_state = mol_llama.state_dict()
    for k in [k for k in mol_state if "lora_" in k.lower()]:
        mol_state[k] = torch.zeros_like(mol_state[k])
    mol_llama.load_state_dict(mol_state, strict=True)

    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora']
    )
    prot_state = temp_prot2text.state_dict()
    for k in [k for k in prot_state if "lora_" in k.lower()]:
        prot_state[k] = torch.zeros_like(prot_state[k])
    temp_prot2text.load_state_dict(prot_state, strict=True)

    esm_encoder = temp_prot2text.esm_encoder
    adapter = temp_prot2text.adapter

    del temp_prot2text, prot_state
    torch.cuda.empty_cache()

    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators


def merge_mol_only(paths, mol_scale=1.0):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])

    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))

    if mol_scale != 1.0:
        sd = mol_llama.state_dict()
        for k in [k for k in sd if "lora_" in k.lower()]:
            sd[k] = sd[k] * mol_scale
        mol_llama.load_state_dict(sd, strict=True)

    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora']
    )
    esm_encoder = temp_prot2text.esm_encoder.to(device)
    adapter = temp_prot2text.adapter.to(device)
    del temp_prot2text; torch.cuda.empty_cache()

    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators


def merge_prot_only(paths, prot_scale=1.0):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])

    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))

    mol_state_dict = mol_llama.state_dict()
    mol_lora_keys = [k for k in mol_state_dict if "lora_" in k.lower()]
    for k in mol_lora_keys:
        mol_state_dict[k] = torch.zeros_like(mol_state_dict[k])
    mol_llama.load_state_dict(mol_state_dict, strict=True)

    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora']
    )
    if prot_scale != 1.0:
        sd = temp_prot2text.state_dict()
        for k in [k for k in sd if "lora_" in k.lower()]:
            sd[k] = sd[k] * prot_scale
        temp_prot2text.load_state_dict(sd, strict=True)

    prot_state_dict = temp_prot2text.state_dict()
    prot_lora_keys = [k for k in prot_state_dict if "lora_" in k.lower()]

    mol_norm_to_orig = {normalize_key(k): k for k in mol_lora_keys}
    prot_norm_to_orig = {normalize_key(k): k for k in prot_lora_keys}

    for nk in set(mol_norm_to_orig) & set(prot_norm_to_orig):
        mk, pk = mol_norm_to_orig[nk], prot_norm_to_orig[nk]
        t = prot_state_dict[pk].to(device, dtype=torch.bfloat16)
        if mol_state_dict[mk].shape == t.shape:
            mol_state_dict[mk] = t
    mol_llama.load_state_dict(mol_state_dict, strict=True)

    esm_encoder = temp_prot2text.esm_encoder.to(device)
    adapter = temp_prot2text.adapter.to(device)
    del temp_prot2text; torch.cuda.empty_cache()

    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators


def merge_cell_only(paths, cell_scale=1.0):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])

    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))

    mol_state_dict = mol_llama.state_dict()
    mol_lora_keys = [k for k in mol_state_dict if "lora_" in k.lower()]
    for k in mol_lora_keys:
        mol_state_dict[k] = torch.zeros_like(mol_state_dict[k])
    mol_llama.load_state_dict(mol_state_dict, strict=True)

    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora']
    )
    temp_cell_model = load_cell_lora(paths['cell_lora'], paths['base_model'])

    if cell_scale != 1.0:
        cell_sd = temp_cell_model.state_dict()
        for k in [k for k in cell_sd if "lora_" in k.lower()]:
            cell_sd[k] = cell_sd[k] * cell_scale
        temp_cell_model.load_state_dict(cell_sd, strict=True)

    cell_state_dict = temp_cell_model.state_dict()
    cell_lora_keys = [k for k in cell_state_dict if "lora_" in k.lower()]

    mol_norm_to_orig = {normalize_key(k): k for k in mol_lora_keys}
    cell_norm_to_orig = {normalize_key(k): k for k in cell_lora_keys}

    for nk in set(mol_norm_to_orig) & set(cell_norm_to_orig):
        mk, ck = mol_norm_to_orig[nk], cell_norm_to_orig[nk]
        t = cell_state_dict[ck].to(device, dtype=torch.bfloat16)
        if mol_state_dict[mk].shape == t.shape:
            mol_state_dict[mk] = t
    mol_llama.load_state_dict(mol_state_dict, strict=True)

    esm_encoder = temp_prot2text.esm_encoder.to(device)
    adapter = temp_prot2text.adapter.to(device)
    del temp_prot2text, temp_cell_model; torch.cuda.empty_cache()

    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators


# =========================================================
# Encoding Utilities
# =========================================================

def prepare_protein_sequence_inputs_batch(esm_tokenizer, protein_sequences, max_length=1024):
    batch_inputs = []
    for sequence in protein_sequences:
        try:
            inputs = esm_tokenizer(
                sequence, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_length, add_special_tokens=True
            )
            batch_inputs.append({
                "input_ids": inputs.input_ids.squeeze(0),
                "attention_mask": inputs.attention_mask.squeeze(0)
            })
        except Exception:
            batch_inputs.append({
                "input_ids": torch.zeros(max_length, dtype=torch.long),
                "attention_mask": torch.zeros(max_length, dtype=torch.long)
            })
    return {
        "input_ids": torch.stack([inp["input_ids"] for inp in batch_inputs]).to(device),
        "attention_mask": torch.stack([inp["attention_mask"] for inp in batch_inputs]).to(device)
    }


def prepare_molecule_embeddings_batch(mol_llama, preprocessed_mols):
    batch_embeddings = []
    for mol_data in preprocessed_mols:
        try:
            if not mol_data.get('atoms') or not hasattr(mol_data.get('coordinates'), 'shape'):
                batch_embeddings.append(torch.zeros((1, 32, mol_llama.llm_proj.out_features), device=device))
                continue
            graph_batch = get_mol_graphs_from_preprocessed_mols(
                [mol_data], mol_llama.encoder.unimol_dictionary, device
            )
            with torch.no_grad():
                _, _, query_output = mol_llama.encoder.graph_forward(graph_batch)
                mol_embedding = mol_llama.llm_proj(query_output.last_hidden_state)
            current_len = mol_embedding.size(1)
            if current_len < 32:
                padding = torch.zeros(
                    (mol_embedding.size(0), 32 - current_len, mol_embedding.size(2)),
                    device=device, dtype=mol_embedding.dtype
                )
                mol_embedding = torch.cat([mol_embedding, padding], dim=1)
            elif current_len > 32:
                mol_embedding = mol_embedding[:, :32, :]
            batch_embeddings.append(mol_embedding)
        except Exception:
            batch_embeddings.append(torch.zeros((1, 32, mol_llama.llm_proj.out_features), device=device))
    return torch.cat(batch_embeddings, dim=0)


def tokenize_batch_messages_llama3(batch_messages, tokenizer, padding_side='left'):
    all_input_ids, all_attention_masks = [], []
    all_mol_token_flags, all_protein_placeholder_flags = [], []
    max_length = 0

    for messages in batch_messages:
        tokenized = {"input_ids": [], "attention_mask": []}
        has_system = messages and len(messages) > 0 and messages[0]['role'] == 'system'
        for idx, m in enumerate(messages):
            if m['role'] == 'system':
                text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + m['content'] + "<|eot_id|>"
            elif m['role'] == 'user':
                if idx == 0 and not has_system:
                    text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + m['content'] + "<|eot_id|>"
                else:
                    text = "<|start_header_id|>user<|end_header_id|>\n\n" + m['content'] + "<|eot_id|>"
                text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                continue
            tokenized_ = tokenizer(text, add_special_tokens=False)
            tokenized["input_ids"].extend(tokenized_['input_ids'])
            tokenized["attention_mask"].extend(tokenized_['attention_mask'])
        max_length = max(max_length, len(tokenized["input_ids"]))
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
        all_mol_token_flags.append([bool(t == tokenizer.mol_token_id) for t in tokenized['input_ids']])
        all_protein_placeholder_flags.append([bool(t == tokenizer.protein_placeholder_token_id) for t in tokenized['input_ids']])

    padded_input_ids, padded_attention_masks = [], []
    padded_mol_token_flags, padded_protein_placeholder_flags = [], []
    for i in range(len(all_input_ids)):
        pad_len = max_length - len(all_input_ids[i])
        if padding_side == 'right':
            padded_input_ids.append(all_input_ids[i] + [tokenizer.pad_token_id] * pad_len)
            padded_attention_masks.append(all_attention_masks[i] + [0] * pad_len)
            padded_mol_token_flags.append(all_mol_token_flags[i] + [False] * pad_len)
            padded_protein_placeholder_flags.append(all_protein_placeholder_flags[i] + [False] * pad_len)
        else:
            padded_input_ids.append([tokenizer.pad_token_id] * pad_len + all_input_ids[i])
            padded_attention_masks.append([0] * pad_len + all_attention_masks[i])
            padded_mol_token_flags.append([False] * pad_len + all_mol_token_flags[i])
            padded_protein_placeholder_flags.append([False] * pad_len + all_protein_placeholder_flags[i])
    return BatchEncoding({
        "input_ids": torch.tensor(padded_input_ids, device=device),
        "attention_mask": torch.tensor(padded_attention_masks, device=device),
        "mol_token_flag": torch.tensor(padded_mol_token_flags, device=device),
        "protein_placeholder_flag": torch.tensor(padded_protein_placeholder_flags, device=device),
    })


def create_cell_mask(input_ids, tokenizer, cell_gene_lists):
    batch_size, seq_len = input_ids.shape
    cell_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    cell_patterns = [
        tokenizer.encode("Cell:", add_special_tokens=False),
        tokenizer.encode("Cell :", add_special_tokens=False),
        tokenizer.encode("\nCell:", add_special_tokens=False),
    ]
    end_patterns = [
        tokenizer.encode("<|eot_id|>", add_special_tokens=False),
        tokenizer.encode("<|start_header_id|>", add_special_tokens=False),
        tokenizer.encode("assistant", add_special_tokens=False),
        tokenizer.encode("<|end_header_id|>", add_special_tokens=False),
        tokenizer.encode("\n\n", add_special_tokens=False),
    ]
    pad_token_id = tokenizer.pad_token_id
    newline_tokens = set(tokenizer.encode('\n', add_special_tokens=False) or [])
    space_tokens = set(tokenizer.encode(' ', add_special_tokens=False) or [])
    exclude_tokens = newline_tokens | space_tokens

    for b in range(batch_size):
        all_cell_positions = []
        for pattern_ids in cell_patterns:
            if not pattern_ids:
                continue
            for i in range(len(input_ids[b]) - len(pattern_ids) + 1):
                if all(input_ids[b, i+j].item() == pattern_ids[j] for j in range(len(pattern_ids))):
                    all_cell_positions.append((i, len(pattern_ids)))
        if not all_cell_positions:
            continue

        last_cell_pos, pattern_len = all_cell_positions[-1]
        start_idx = last_cell_pos + pattern_len
        earliest_marker_pos = seq_len
        for end_pattern in end_patterns:
            if not end_pattern:
                continue
            for i in range(start_idx, len(input_ids[b]) - len(end_pattern) + 1):
                if all(input_ids[b, i+j].item() == end_pattern[j] for j in range(len(end_pattern))):
                    if i < earliest_marker_pos:
                        earliest_marker_pos = i
                    break
        end_idx = earliest_marker_pos
        while end_idx > start_idx and input_ids[b, end_idx - 1].item() == pad_token_id:
            end_idx -= 1
        while end_idx > start_idx and input_ids[b, end_idx - 1].item() in exclude_tokens:
            end_idx -= 1
        if end_idx > start_idx:
            cell_mask[b, start_idx:end_idx] = True
    return cell_mask


def build_unified_embedding_batch(
    mol_llama, esm_encoder, adapter, esm_tokenizer, llama_tokenizer,
    molecule_dict, batch_samples, system_prompt, user_prompt,
    esm_max_len=1024, padding_side="left",
):
    batch_messages = []
    MOL_PLACEHOLDER_STR = '<mol>' * 8
    PROT_PLACEHOLDER_TOKEN_STR = llama_tokenizer.decode([llama_tokenizer.protein_placeholder_token_id])
    drugs, prots, cells = [], [], []

    for sample in batch_samples:
        drug, prot, cell = sample["Drug"], sample["Protein"], sample["Cell"]
        user_content = user_prompt
        user_content = user_content.replace('<protein>', PROT_PLACEHOLDER_TOKEN_STR * (len(prot) + 2), 1)
        prots.append(prot)
        user_content = user_content.replace('<mol>', MOL_PLACEHOLDER_STR, 1)
        drugs.append(drug)
        user_content = user_content.replace('<gene>', cell, 1)
        cells.append(cell)

        if system_prompt.strip():
            batch_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])
        else:
            batch_messages.append([{"role": "user", "content": user_content}])

    inputs = tokenize_batch_messages_llama3(batch_messages, llama_tokenizer, padding_side=padding_side)

    with torch.no_grad():
        text_embeds = mol_llama.llm.get_input_embeddings()(inputs.input_ids).to(dtype=torch.bfloat16)

    preprocessed = []
    for d in drugs:
        preprocessed.append(molecule_dict.get(d, {"smiles": d, "atoms": [], "coordinates": np.array([])}))
    mol_embeds = prepare_molecule_embeddings_batch(mol_llama, preprocessed)

    batch_size = len(batch_samples)
    mol_embed_counter = 0
    for b in range(batch_size):
        mol_indices = torch.where(inputs.mol_token_flag[b])[0]
        num_drugs_to_inject = min(1, len(mol_indices) // 8)
        for drug_idx in range(num_drugs_to_inject):
            mol_embed = mol_embeds[mol_embed_counter]
            if mol_embed.dim() == 3:
                mol_embed = mol_embed.squeeze(0)
            drug_indices = mol_indices[drug_idx * 8 : drug_idx * 8 + 8]
            for j in range(min(len(drug_indices), 8)):
                text_embeds[b, drug_indices[j]] = mol_embed[j]
            mol_embed_counter += 1

    protein_inputs = prepare_protein_sequence_inputs_batch(esm_tokenizer, prots, max_length=esm_max_len)
    with torch.no_grad():
        esm_outputs = esm_encoder(**protein_inputs)
        all_prot_embeddings = adapter(esm_outputs.last_hidden_state)

    prot_embed_counter = 0
    for b in range(batch_size):
        prot_indices = torch.where(inputs.protein_placeholder_flag[b])[0]
        if prot_embed_counter >= len(all_prot_embeddings):
            break
        prot_embed = all_prot_embeddings[prot_embed_counter]
        valid_len = protein_inputs['attention_mask'][prot_embed_counter].sum().item()
        valid_embeds = prot_embed[:valid_len]
        indices_to_replace = prot_indices[:valid_len]
        for j in range(min(len(indices_to_replace), len(valid_embeds))):
            text_embeds[b, indices_to_replace[j]] = valid_embeds[j]
        prot_embed_counter += 1

    target_cell_mask = create_cell_mask(inputs.input_ids, llama_tokenizer, cells)

    target_mol_mask = torch.zeros_like(inputs.mol_token_flag, dtype=torch.bool)
    for b in range(batch_size):
        mol_indices = torch.where(inputs.mol_token_flag[b])[0]
        if len(mol_indices) >= 8:
            target_mol_mask[b, mol_indices[-8:]] = True
        elif len(mol_indices) > 0:
            target_mol_mask[b, mol_indices] = True

    target_prot_mask = torch.zeros_like(inputs.protein_placeholder_flag, dtype=torch.bool)
    for b in range(batch_size):
        prot_indices = torch.where(inputs.protein_placeholder_flag[b])[0]
        if len(prot_indices) > 0:
            target_prot_mask[b, prot_indices] = True

    return text_embeds.detach(), inputs.attention_mask, target_mol_mask, target_prot_mask, target_cell_mask


# =========================================================
# Hidden States & Scoring
# =========================================================

def get_hidden_states(llm_model, inputs_embeds, attention_mask):
    out = llm_model(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False
    )
    return out.hidden_states


def layerwise_paramwise_scores(
    base_llm, spec_llm, inputs_embeds, attention_mask, token_mask,
    layers_to_use, score_buf, score_type="grad_abs",
    n_layers_expected=32,
):
    with torch.no_grad():
        hs_base = get_hidden_states(base_llm, inputs_embeds, attention_mask)
    hs_spec = get_hidden_states(spec_llm, inputs_embeds, attention_mask)

    for l in sorted(layers_to_use, reverse=True):
        bi = map_layer_idx(hs_base, l, n_layers_expected)
        si = map_layer_idx(hs_spec, l, n_layers_expected)
        if not token_mask.any():
            continue

        h_spec = hs_spec[si][token_mask]
        h_base = hs_base[bi][token_mask].detach()
        if h_spec.numel() == 0:
            continue

        L_l = (h_spec - h_base).pow(2).mean()

        # Score each layer's LoRA parameters using that layer's output difference.
        layer_params = [
            p for n, p in spec_llm.named_parameters()
            if f"layers.{l}." in n.lower() and is_lora_param(n) and p.requires_grad
        ]
        if not layer_params:
            continue

        grads = torch.autograd.grad(outputs=L_l, inputs=layer_params, retain_graph=True, allow_unused=True)
        param_names = [
            n for n, p in spec_llm.named_parameters()
            if f"layers.{l}." in n.lower() and is_lora_param(n) and p.requires_grad
        ]
        for name, grad in zip(param_names, grads):
            if grad is not None:
                nk = normalize_key(name)
                g = grad.detach().float().cpu()
                if score_type == "grad2":
                    score_buf[nk] += g.pow(2)
                elif score_type == "grad_abs":
                    score_buf[nk] += g.abs()


def finalize_coefficients_3way(score_mol, score_prot, score_cell, tau):
    common = sorted(set(score_mol) & set(score_prot) & set(score_cell))
    if not common:
        raise RuntimeError("No common LoRA keys between mol/prot/cell")

    coef_mol, coef_prot, coef_cell = {}, {}, {}
    for k in common:
        a = np.nan_to_num(score_mol[k].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(score_prot[k].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        c = np.nan_to_num(score_cell[k].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        pa, pb, pc = softmax_3class(mean_std_norm(a), mean_std_norm(b), mean_std_norm(c), tau=tau)
        coef_mol[k] = torch.from_numpy(pa).float().cpu()
        coef_prot[k] = torch.from_numpy(pb).float().cpu()
        coef_cell[k] = torch.from_numpy(pc).float().cpu()
    return coef_mol, coef_prot, coef_cell


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser("element-wise coefficient computation")

    # Model paths
    parser.add_argument('--base_model_path', type=str, default='./checkpoints/Llama-3.1-8B-Instruct')
    parser.add_argument('--molecule_model_path', type=str, default='./checkpoints/Mol-Llama-3.1-8B-Instruct')
    parser.add_argument('--esm_path', type=str, default='./checkpoints/esm2_t36_3B_UR50D')
    parser.add_argument('--prot2text_base_checkpoint', type=str, default='./checkpoints/Prot2Text-V2/model_checkpoint.pt')
    parser.add_argument('--prot2text_lora_path', type=str, default='./checkpoints/Prot2Text-V2/adapter_checkpoint')
    parser.add_argument('--cell_lora_path', type=str, default='./checkpoints/cell-o1/lora_checkpoint')
    parser.add_argument('--data_dir', type=str, default='./coefficient_data')
    parser.add_argument('--output_dir', type=str, default='./coefficients/Element_Wise')

    # Experiment params
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = full pass")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--score_type", choices=["grad2", "grad_abs"], default="grad_abs")
    parser.add_argument("--esm_max_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    layers_to_use = parse_layer_range(args.layers)

    paths = {
        'base_model': args.base_model_path,
        'molecule_model': args.molecule_model_path,
        'esm': args.esm_path,
        'prot2text_base_checkpoint': args.prot2text_base_checkpoint,
        'prot2text_lora': args.prot2text_lora_path,
        'cell_lora': args.cell_lora_path,
    }

    output_dir = os.path.join(args.output_dir, "mse", args.score_type)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "element_wise_coef.pt")

    system_prompt = ""
    user_prompt = """Molecule: <mol>
Protein: <protein>
Cell: <gene>"""

    proteins, molecules, cells, molecule_dict = load_all_data(args.data_dir)
    samples = create_330_samples(proteins, molecules, cells)

    # Load models
    mol_llama_base, (esm_base, adapter_base), esm_tok, llama_tok, _ = get_noLoRA_models(paths)
    base_llm = mol_llama_base.llm

    mol_llama_mol, _, _, _, _ = merge_mol_only(paths, mol_scale=1.0)
    mol_llm = mol_llama_mol.llm

    mol_llama_prot, _, _, _, _ = merge_prot_only(paths, prot_scale=1.0)
    prot_llm = mol_llama_prot.llm

    mol_llama_cell, _, _, _, _ = merge_cell_only(paths, cell_scale=1.0)
    cell_llm = mol_llama_cell.llm

    base_llm.eval(); mol_llm.eval(); prot_llm.eval(); cell_llm.eval()

    for p in base_llm.parameters():
        p.requires_grad_(False)
    set_lora_requires_grad(mol_llm, True)
    set_lora_requires_grad(prot_llm, True)
    set_lora_requires_grad(cell_llm, True)

    score_mol = init_score_buffers(mol_llm)
    score_prot = init_score_buffers(prot_llm)
    score_cell = init_score_buffers(cell_llm)

    norm2actual_mol = build_norm_to_actual_map(mol_llm)
    norm2actual_prot = build_norm_to_actual_map(prot_llm)
    norm2actual_cell = build_norm_to_actual_map(cell_llm)

    print(f"Samples: {len(samples)}, Batch size: {args.batch_size}, Layers: {args.layers}, Loss: mse")

    step = 0
    for batch_samples in tqdm(batch_iterator(samples, args.batch_size), desc="Processing"):
        step += 1
        if args.max_steps and step > args.max_steps:
            break

        inputs_embeds, attn, mol_mask, prot_mask, cell_mask = build_unified_embedding_batch(
            mol_llama=mol_llama_base, esm_encoder=esm_base, adapter=adapter_base,
            esm_tokenizer=esm_tok, llama_tokenizer=llama_tok,
            molecule_dict=molecule_dict, batch_samples=batch_samples,
            system_prompt=system_prompt, user_prompt=user_prompt,
            esm_max_len=args.esm_max_len, padding_side="left",
        )

        score_kwargs = dict(
            layers_to_use=layers_to_use, score_type=args.score_type,
        )

        if mol_mask.any():
            layerwise_paramwise_scores(base_llm, mol_llm, inputs_embeds, attn, mol_mask, score_buf=score_mol, **score_kwargs)
        if prot_mask.any():
            layerwise_paramwise_scores(base_llm, prot_llm, inputs_embeds, attn, prot_mask, score_buf=score_prot, **score_kwargs)
        if cell_mask.any():
            layerwise_paramwise_scores(base_llm, cell_llm, inputs_embeds, attn, cell_mask, score_buf=score_cell, **score_kwargs)

    coef_mol, coef_prot, coef_cell = finalize_coefficients_3way(score_mol, score_prot, score_cell, tau=args.tau)

    payload = {
        "meta": {
            "layers": args.layers, "batch_size": args.batch_size,
            "max_steps": args.max_steps, "total_steps": step,
            "score_type": args.score_type, "loss_metric": "mse",
            "tau": args.tau, "esm_max_len": args.esm_max_len, "seed": args.seed,
        },
        "norm2actual_mol": norm2actual_mol, "norm2actual_prot": norm2actual_prot, "norm2actual_cell": norm2actual_cell,
        "score_mol": score_mol, "score_prot": score_prot, "score_cell": score_cell,
        "coef_mol": coef_mol, "coef_prot": coef_prot, "coef_cell": coef_cell,
    }
    torch.save(payload, output_path)

    common = len(set(score_mol) & set(score_prot) & set(score_cell))
    print(f"Matched LoRA tensors: {common}, Total steps: {step}, Output: {output_path}")


if __name__ == "__main__":
    main()