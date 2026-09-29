import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from transformers import BatchEncoding
import random
from typing import Tuple, Dict, Any, List
import math
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from Mol_LLaMA.models.mol_llama import MolLLaMA
from torch.nn import functional as F
from Mol_LLaMA.models.mol_llama import get_mol_graphs_from_preprocessed_mols, gen_3d_conformation_from_libraries

from transformers import EsmModel, LlamaForCausalLM
from Prot2Text_V2.models import (
    ModalityAdapter,
    ModalityAdapterConfig, 
    Esm2LlamaInstructForCausalLM
)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 330

comprehensive_hidden_states = {}
global_sample_order = []


class ComprehensiveHiddenStateCapture:
    
    def __init__(self, model_name, tokenizer):
        self.model_name = model_name
        self.tokenizer = tokenizer
        
        self.mol_layer_states = {}
        self.prot_layer_states = {}
        self.cell_layer_states = {}
        
        self.mol_lm_head_states = []
        self.prot_lm_head_states = []
        self.cell_lm_head_states = []
        
        self.mol_final_output_states = []
        self.prot_final_output_states = []
        self.cell_final_output_states = []
        
        self.output_token_embeddings = []
        self.hooks = []
        
        self.current_mol_mask = None
        self.current_prot_mask = None
        self.current_cell_mask = None
        self.current_input_ids = None
        
        self.current_sample_idx = 0
        self.batch_count = 0
        self.total_samples_processed = 0
    
    def set_token_masks(self, mol_mask, prot_mask, cell_mask, input_ids, cell_gene_lists=None):
        self.current_mol_mask = mol_mask
        self.current_prot_mask = prot_mask
        self.current_cell_mask = cell_mask
        self.current_input_ids = input_ids
        self.current_cell_gene_lists = cell_gene_lists
    
    def create_cell_mask(self, input_ids):
        batch_size, seq_len = input_ids.shape
        cell_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        tokenizer = self.tokenizer
        
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
                    match = all(
                        input_ids[b, i+j].item() == pattern_ids[j] 
                        for j in range(len(pattern_ids))
                    )
                    if match:
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
                    match = all(
                        input_ids[b, i+j].item() == end_pattern[j]
                        for j in range(len(end_pattern))
                    )
                    if match:
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
    
    def register_hooks(self, model):
        llm_model = None
        
        if hasattr(model, 'llm'):
            if hasattr(model.llm, 'base_model'):
                if hasattr(model.llm.base_model, 'model'):
                    base_llama = model.llm.base_model.model
                    llm_model = base_llama.model if hasattr(base_llama, 'model') else base_llama
                else:
                    llm_model = model.llm.base_model
            else:
                llm_model = model.llm.model if hasattr(model.llm, 'model') else model.llm
        elif hasattr(model, 'llama_decoder'):
            llm_model = model.llama_decoder.model if hasattr(model.llama_decoder, 'model') else model.llama_decoder
        else:
            llm_model = model.model if hasattr(model, 'model') else model
        
        layers = None
        if hasattr(llm_model, 'layers'):
            layers = llm_model.layers
        elif hasattr(llm_model, 'model') and hasattr(llm_model.model, 'layers'):
            layers = llm_model.model.layers
        else:
            raise RuntimeError("Cannot locate transformer layers")
        
        for layer_idx, layer in enumerate(layers):
            self.hooks.append(layer.register_forward_hook(self._create_layer_hook_fn(layer_idx)))
        
        self.hooks.append(layers[-1].register_forward_hook(self._create_final_output_hook_fn()))
        
        if hasattr(llm_model, 'lm_head'):
            self.hooks.append(llm_model.lm_head.register_forward_hook(self._create_lm_head_hook_fn()))
    
    def _create_layer_hook_fn(self, layer_idx):
        def hook_fn(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            
            if self.current_mol_mask is None or self.current_prot_mask is None or self.current_input_ids is None:
                return
            if hidden_state.shape[1] != self.current_mol_mask.shape[1]:
                return
            
            batch_size = hidden_state.shape[0]
            current_total = sum(len(bd) for bd in self.mol_layer_states.get(layer_idx, []))
            if current_total >= SAMPLE_SIZE:
                return
            
            batch_mol_means = []
            for b in range(batch_size):
                mol_indices = torch.where(self.current_mol_mask[b])[0]
                if len(mol_indices) > 0:
                    batch_mol_means.append(hidden_state[b, mol_indices, :].mean(dim=0).detach())
                else:
                    batch_mol_means.append(None)
            
            batch_prot_means = []
            for b in range(batch_size):
                prot_indices = torch.where(self.current_prot_mask[b])[0]
                if len(prot_indices) > 0:
                    batch_prot_means.append(hidden_state[b, prot_indices, :].mean(dim=0).detach())
                else:
                    batch_prot_means.append(None)
            
            batch_cell_means = []
            cell_mask = self.create_cell_mask(self.current_input_ids)
            for b in range(batch_size):
                cell_indices = torch.where(cell_mask[b])[0]
                if len(cell_indices) > 0:
                    batch_cell_means.append(hidden_state[b, cell_indices, :].mean(dim=0).detach())
                else:
                    batch_cell_means.append(None)
            
            self.mol_layer_states.setdefault(layer_idx, []).append(batch_mol_means)
            self.prot_layer_states.setdefault(layer_idx, []).append(batch_prot_means)
            self.cell_layer_states.setdefault(layer_idx, []).append(batch_cell_means)
        
        return hook_fn
    
    def _create_lm_head_hook_fn(self):
        def lm_head_hook_fn(module, input, output):
            final_hidden_state = input[0] if isinstance(input, tuple) else input
            
            if self.current_mol_mask is None or self.current_prot_mask is None or self.current_input_ids is None:
                return
            if len(self.mol_lm_head_states) >= SAMPLE_SIZE:
                return
            
            batch_size = final_hidden_state.shape[0]
            
            for b in range(batch_size):
                mol_indices = torch.where(self.current_mol_mask[b])[0]
                if len(mol_indices) > 0:
                    self.mol_lm_head_states.append(final_hidden_state[b, mol_indices, :].mean(dim=0).detach())
                else:
                    self.mol_lm_head_states.append(None)
            
            for b in range(batch_size):
                prot_indices = torch.where(self.current_prot_mask[b])[0]
                if len(prot_indices) > 0:
                    self.prot_lm_head_states.append(final_hidden_state[b, prot_indices, :].mean(dim=0).detach())
                else:
                    self.prot_lm_head_states.append(None)
                    
            cell_mask = self.create_cell_mask(self.current_input_ids)
            for b in range(batch_size):
                cell_indices = torch.where(cell_mask[b])[0]
                if len(cell_indices) > 0:
                    self.cell_lm_head_states.append(final_hidden_state[b, cell_indices, :].mean(dim=0).detach())
                else:
                    self.cell_lm_head_states.append(None)
        
        return lm_head_hook_fn
    
    def _create_final_output_hook_fn(self):
        def final_output_hook_fn(module, input, output):
            hidden_state = output[0] if isinstance(output, tuple) else output
            
            if self.current_mol_mask is None or self.current_prot_mask is None or self.current_input_ids is None:
                return
            if hidden_state.shape[1] != self.current_mol_mask.shape[1]:
                return
            if len(self.mol_final_output_states) >= SAMPLE_SIZE:
                return
            
            batch_size = hidden_state.shape[0]
            
            for b in range(batch_size):
                mol_indices = torch.where(self.current_mol_mask[b])[0]
                if len(mol_indices) > 0:
                    self.mol_final_output_states.append(hidden_state[b, mol_indices, :].mean(dim=0).detach())
                else:
                    self.mol_final_output_states.append(None)
            
            for b in range(batch_size):
                prot_indices = torch.where(self.current_prot_mask[b])[0]
                if len(prot_indices) > 0:
                    self.prot_final_output_states.append(hidden_state[b, prot_indices, :].mean(dim=0).detach())
                else:
                    self.prot_final_output_states.append(None)
                    
            cell_mask = self.create_cell_mask(self.current_input_ids)
            for b in range(batch_size):
                cell_indices = torch.where(cell_mask[b])[0]
                if len(cell_indices) > 0:
                    self.cell_final_output_states.append(hidden_state[b, cell_indices, :].mean(dim=0).detach())
                else:
                    self.cell_final_output_states.append(None)
        
        return final_output_hook_fn
    
    def capture_output_token_embeddings(self, tokenizer, generated_ids, input_length):
        current_total = sum(len(bd) for bd in self.output_token_embeddings)
        if current_total >= SAMPLE_SIZE:
            return
            
        try:
            new_token_ids = generated_ids[:, input_length:]
            vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else tokenizer.vocab_size
            
            batch_size = new_token_ids.shape[0]
            seq_len = new_token_ids.shape[1]
            dev = new_token_ids.device
            batch_embeddings = []
            
            for i in range(batch_size):
                if seq_len > 0:
                    avg_token_id = new_token_ids[i].float().mean().item()
                    token_embedding = torch.tensor([avg_token_id / vocab_size], dtype=torch.float32, device=dev)
                else:
                    token_embedding = torch.tensor([0.0], dtype=torch.float32, device=dev)
                batch_embeddings.append(token_embedding)
            
            self.output_token_embeddings.append(torch.stack(batch_embeddings))
            
        except Exception:
            batch_size = generated_ids.shape[0]
            self.output_token_embeddings.append(
                torch.zeros(batch_size, 1, dtype=torch.float32, device=generated_ids.device)
            )
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_organized_states_by_sample(self, batch_size, num_batches):
        organized_states = {}
        max_samples = SAMPLE_SIZE
        
        def _organize_layer_states(layer_states_dict, key_prefix):
            for layer_idx, layer_states in layer_states_dict.items():
                sample_count = 0
                for batch_state in layer_states:
                    for sample_in_batch in range(len(batch_state)):
                        if sample_count >= max_samples:
                            break
                        if sample_count not in organized_states:
                            organized_states[sample_count] = {}
                        if batch_state[sample_in_batch] is not None:
                            organized_states[sample_count][f'{key_prefix}_layer_{layer_idx}'] = batch_state[sample_in_batch]
                        sample_count += 1
                    if sample_count >= max_samples:
                        break
        
        _organize_layer_states(self.mol_layer_states, 'mol')
        _organize_layer_states(self.prot_layer_states, 'prot')
        _organize_layer_states(self.cell_layer_states, 'cell')
        
        def _organize_flat_states(states_list, key_name):
            for sample_idx, state in enumerate(states_list):
                if sample_idx >= max_samples:
                    break
                if sample_idx not in organized_states:
                    organized_states[sample_idx] = {}
                if state is not None:
                    organized_states[sample_idx][key_name] = state
        
        _organize_flat_states(self.mol_lm_head_states, 'mol_lm_head')
        _organize_flat_states(self.prot_lm_head_states, 'prot_lm_head')
        _organize_flat_states(self.cell_lm_head_states, 'cell_lm_head')
        _organize_flat_states(self.mol_final_output_states, 'mol_final_output')
        _organize_flat_states(self.prot_final_output_states, 'prot_final_output')
        _organize_flat_states(self.cell_final_output_states, 'cell_final_output')
        
        sample_count = 0
        for batch_output_embeddings in self.output_token_embeddings:
            for sample_in_batch in range(batch_output_embeddings.shape[0]):
                if sample_count >= max_samples:
                    break
                if sample_count not in organized_states:
                    organized_states[sample_count] = {}
                output_token = batch_output_embeddings[sample_in_batch]
                if output_token.device.type != 'cpu':
                    output_token = output_token.cpu().float()
                organized_states[sample_count]['output_tokens'] = output_token
                sample_count += 1
            if sample_count >= max_samples:
                break
        
        return organized_states
    
    def clear_states(self):
        self.mol_layer_states = {}
        self.prot_layer_states = {}
        self.cell_layer_states = {}
        self.mol_lm_head_states = []
        self.prot_lm_head_states = []
        self.cell_lm_head_states = []
        self.mol_final_output_states = []
        self.prot_final_output_states = []
        self.cell_final_output_states = []
        self.output_token_embeddings = []
        self.current_mol_mask = None
        self.current_prot_mask = None
        self.current_cell_mask = None
        self.batch_count = 0
        self.total_samples_processed = 0


def load_dataset_related(data_dir):
    system_prompt = ""
    user_prompt = """
Molecule: <mol>
Protein: <protein>
Cell: <gene>
"""
    protein_dataset_path = os.path.join(data_dir, "protein.txt")
    molecule_dataset_path = os.path.join(data_dir, "drug.txt")
    molecule_dataset_3d_path = os.path.join(data_dir, "drug_3d.json")
    cell_dataset_path = os.path.join(data_dir, "cell_line.txt")
    
    return (protein_dataset_path, molecule_dataset_path, molecule_dataset_3d_path, 
            cell_dataset_path, system_prompt, user_prompt)


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


def setup_tokenizers(base_model_path, esm_path):
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    llama_tokenizer.pad_token = '<|reserved_special_token_0|>'
    llama_tokenizer.bos_token = "<|begin_of_text|>"
    llama_tokenizer.eos_token = "<|eot_id|>"
    llama_tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
    llama_tokenizer.mol_token_id = llama_tokenizer("<mol>", add_special_tokens=False).input_ids[0]
    llama_tokenizer.pad_token = "<|eot_id|>"
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
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
        input_dim=esm_encoder.config.hidden_size,
        intermediate_dim=2048,
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
        prot2text_with_lora = PeftModel.from_pretrained(
            prot2text_model, lora_checkpoint_path, is_trainable=False
        )
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
    base_model = LlamaForCausalLM.from_pretrained(
        base_llama_path, torch_dtype=torch.bfloat16, device_map=device
    )
    cell_model = PeftModel.from_pretrained(base_model, cell_lora_path, torch_dtype=torch.bfloat16)
    cell_model.eval()
    return cell_model


def _ensure_bfloat16(module):
    for param in module.parameters():
        if param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    for buffer in module.buffers():
        if buffer.dtype != torch.bfloat16:
            buffer.data = buffer.data.to(torch.bfloat16)


def _normalize_lora_key(key):
    normalized = key.lower()
    for prefix in [
        'llm.base_model.model.model.', 'base_model.model.llama_decoder.model.',
        'base_model.model.model.', 'llm.model.', 'model.',
    ]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    normalized = normalized.replace('.default.weight', '.weight')
    normalized = normalized.replace('.lora_a.weight', '.lora_A.weight')
    normalized = normalized.replace('.lora_b.weight', '.lora_B.weight')
    return normalized


def merge_mol_only(mol_scale, paths):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])
    
    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))
    
    if mol_scale != 1.0:
        mol_state_dict = mol_llama.state_dict()
        for key in [k for k in mol_state_dict if "lora_" in k.lower()]:
            mol_state_dict[key] = mol_state_dict[key] * mol_scale
        mol_llama.load_state_dict(mol_state_dict, strict=True)
    
    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora'],
    )
    esm_encoder = temp_prot2text.esm_encoder.to(device)
    adapter = temp_prot2text.adapter.to(device)
    
    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    
    del temp_prot2text
    torch.cuda.empty_cache()
    
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators


def merge_prot_only(prot_scale, paths):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])
    
    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))
    
    mol_state_dict = mol_llama.state_dict()
    mol_lora_keys = [k for k in mol_state_dict if "lora_" in k.lower()]
    for key in mol_lora_keys:
        mol_state_dict[key] = torch.zeros_like(mol_state_dict[key])
    mol_llama.load_state_dict(mol_state_dict, strict=True)
    
    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora'],
    )
    
    if prot_scale != 1.0:
        prot_state_dict = temp_prot2text.state_dict()
        for key in [k for k in prot_state_dict if "lora_" in k.lower()]:
            prot_state_dict[key] = prot_state_dict[key] * prot_scale
        temp_prot2text.load_state_dict(prot_state_dict, strict=True)
    
    prot_state_dict = temp_prot2text.state_dict()
    prot_lora_keys = [k for k in prot_state_dict if "lora_" in k.lower()]
    
    mol_norm_to_orig = {_normalize_lora_key(k): k for k in mol_lora_keys}
    prot_norm_to_orig = {_normalize_lora_key(k): k for k in prot_lora_keys}
    
    for norm_key in set(mol_norm_to_orig) & set(prot_norm_to_orig):
        prot_tensor = prot_state_dict[prot_norm_to_orig[norm_key]].to(device, dtype=torch.bfloat16)
        mol_key = mol_norm_to_orig[norm_key]
        if mol_state_dict[mol_key].shape == prot_tensor.shape:
            mol_state_dict[mol_key] = prot_tensor
    
    mol_llama.load_state_dict(mol_state_dict, strict=True)
    esm_encoder = temp_prot2text.esm_encoder.to(device)
    adapter = temp_prot2text.adapter.to(device)
    
    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    
    del temp_prot2text
    torch.cuda.empty_cache()
    
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators

def merge_cell_only(cell_scale, paths):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])
    
    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))
    
    mol_state_dict = mol_llama.state_dict()
    mol_lora_keys = [k for k in mol_state_dict if "lora_" in k.lower()]
    for key in mol_lora_keys:
        mol_state_dict[key] = torch.zeros_like(mol_state_dict[key])
    mol_llama.load_state_dict(mol_state_dict, strict=True)
    
    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora'],
    )
    temp_cell_model = load_cell_lora(paths['cell_lora'], paths['base_model'])
     
    if cell_scale != 1.0:
        cell_state_dict = temp_cell_model.state_dict()
        for key in [k for k in cell_state_dict if "lora_" in k.lower()]:
            cell_state_dict[key] = cell_state_dict[key] * cell_scale
        temp_cell_model.load_state_dict(cell_state_dict, strict=True)
    
    cell_state_dict = temp_cell_model.state_dict()
    cell_lora_keys = [k for k in cell_state_dict if "lora_" in k.lower()]
    
    mol_norm_to_orig = {_normalize_lora_key(k): k for k in mol_lora_keys}
    cell_norm_to_orig = {_normalize_lora_key(k): k for k in cell_lora_keys}
    
    for norm_key in set(mol_norm_to_orig) & set(cell_norm_to_orig):
        cell_tensor = cell_state_dict[cell_norm_to_orig[norm_key]].to(device, dtype=torch.bfloat16)
        mol_key = mol_norm_to_orig[norm_key]
        if mol_state_dict[mol_key].shape == cell_tensor.shape:
            mol_state_dict[mol_key] = cell_tensor
    
    mol_llama.load_state_dict(mol_state_dict, strict=True)
    esm_encoder = temp_prot2text.esm_encoder.to(device)
    adapter = temp_prot2text.adapter.to(device)
    
    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    
    del temp_prot2text, temp_cell_model
    torch.cuda.empty_cache()
    
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators

def merge_withoutlora(paths):
    torch.cuda.empty_cache()
    esm_tokenizer, llama_tokenizer, terminators = setup_tokenizers(paths['base_model'], paths['esm'])
    
    mol_llama = MolLLaMA.from_pretrained(
        paths['molecule_model'], torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tokenizer), enable_flash=False
    ).to("cpu")
    mol_llama.llm.resize_token_embeddings(len(llama_tokenizer))
    
    mol_state = mol_llama.state_dict()
    for key in [k for k in mol_state if 'lora_' in k.lower()]:
        mol_state[key] = torch.zeros_like(mol_state[key])
    mol_llama.load_state_dict(mol_state, strict=True)
    
    temp_prot2text = load_prot2text_with_lora(
        paths['esm'], paths['base_model'], paths['prot2text_base_checkpoint'], paths['prot2text_lora'],
    )
    
    prot_state = temp_prot2text.state_dict()
    for key in [k for k in prot_state if 'lora_' in k.lower()]:
        prot_state[key] = torch.zeros_like(prot_state[key])
    temp_prot2text.load_state_dict(prot_state, strict=True)
    
    esm_encoder = temp_prot2text.esm_encoder
    adapter = temp_prot2text.adapter
    
    del temp_prot2text, prot_state
    torch.cuda.empty_cache()
    
    mol_llama = mol_llama.to(device)
    
    for m in [mol_llama, esm_encoder, adapter]:
        _ensure_bfloat16(m)
    
    mol_llama.eval(); esm_encoder.eval(); adapter.eval()
    return mol_llama, (esm_encoder, adapter), esm_tokenizer, llama_tokenizer, terminators

def get_merged_models(mol_scale, prot_scale, cell_scale, paths):
    if mol_scale > 0 and prot_scale == 0 and cell_scale == 0:
        return merge_mol_only(mol_scale, paths)
    elif mol_scale == 0 and prot_scale > 0 and cell_scale == 0:
        return merge_prot_only(prot_scale, paths)
    elif mol_scale == 0 and prot_scale == 0 and cell_scale > 0:
        return merge_cell_only(cell_scale, paths)
    elif mol_scale == 0 and prot_scale == 0 and cell_scale == 0:
        return merge_withoutlora(paths)
    else:
        raise ValueError("Invalid scale configuration")
    
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
            coords = mol_data.get('coordinates', [])
            atoms = mol_data.get('atoms', [])
            has_coords = (isinstance(coords, np.ndarray) and coords.size > 0) or (isinstance(coords, list) and len(coords) > 0)
            
            if not (atoms and len(atoms) > 0) or not has_coords:
                batch_embeddings.append(torch.zeros((1, 32, mol_llama.llm_proj.out_features), device=device))
                continue
            
            graph_batch = get_mol_graphs_from_preprocessed_mols([mol_data], mol_llama.encoder.unimol_dictionary, device)
            with torch.no_grad():
                _, _, query_output = mol_llama.encoder.graph_forward(graph_batch)
                batch_embeddings.append(mol_llama.llm_proj(query_output.last_hidden_state))
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


def comprehensive_predict_with_hidden_state_capture(
    merged_mol_model, merged_prot_model_or_components,
    esm_tokenizer, llama_tokenizer, terminators,
    preprocessed_mols, protein_sequences, cell_gene_lists,
    system_prompt, user_prompt, hidden_state_capture, max_length=512,
):
    batch_size = len(preprocessed_mols)
    model_name = hidden_state_capture.model_name

    batch_messages, mols_to_embed, prots_to_embed = [], [], []
    MOL_PLACEHOLDER_STR = '<mol>' * 8
    PROT_PLACEHOLDER_TOKEN_STR = llama_tokenizer.decode([llama_tokenizer.protein_placeholder_token_id])

    for i in range(batch_size):
        user_content = user_prompt
        user_content = user_content.replace('<protein>', PROT_PLACEHOLDER_TOKEN_STR * (len(protein_sequences[i]) + 2), 1)
        prots_to_embed.append(protein_sequences[i])
        user_content = user_content.replace('<mol>', MOL_PLACEHOLDER_STR, 1)
        mols_to_embed.append(preprocessed_mols[i])
        user_content = user_content.replace('<gene>', cell_gene_lists[i], 1)
        
        if system_prompt.strip():
            batch_messages.append([
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_content}
            ])
        else:
            batch_messages.append([{"role": "user", "content": user_content}])
    
    inputs = tokenize_batch_messages_llama3(batch_messages, llama_tokenizer)

    target_mol_mask = torch.zeros_like(inputs.mol_token_flag, dtype=torch.bool)
    target_prot_mask = torch.zeros_like(inputs.protein_placeholder_flag, dtype=torch.bool)
    target_cell_mask = torch.zeros_like(inputs.input_ids, dtype=torch.bool)

    for b in range(batch_size):
        mol_indices = torch.where(inputs.mol_token_flag[b])[0]
        if len(mol_indices) >= 8:
            target_mol_mask[b, mol_indices[-8:]] = True
        elif len(mol_indices) > 0:
            target_mol_mask[b, mol_indices] = True
        
        prot_indices = torch.where(inputs.protein_placeholder_flag[b])[0]
        if len(prot_indices) > 0:
            target_prot_mask[b, prot_indices] = True

    hidden_state_capture.set_token_masks(
        mol_mask=target_mol_mask, cell_mask=target_cell_mask,
        prot_mask=target_prot_mask, input_ids=inputs.input_ids,
        cell_gene_lists=cell_gene_lists
    )

    with torch.no_grad():
        text_embeds = merged_mol_model.llm.get_input_embeddings()(inputs.input_ids).to(dtype=torch.bfloat16)

        if mols_to_embed and inputs.mol_token_flag.any():
            all_mol_embeddings = prepare_molecule_embeddings_batch(merged_mol_model, mols_to_embed)
            mol_embed_counter = 0
            for b in range(batch_size):
                mol_indices = torch.where(inputs.mol_token_flag[b])[0]
                for i in range(len(mol_indices) // 8):
                    if mol_embed_counter >= len(all_mol_embeddings): 
                        break
                    mol_embed = all_mol_embeddings[mol_embed_counter]
                    indices_to_replace = mol_indices[i * 8 : i * 8 + 8]
                    for j in range(min(len(indices_to_replace), mol_embed.size(0))):
                        text_embeds[b, indices_to_replace[j]] = mol_embed[j]
                    mol_embed_counter += 1

        if prots_to_embed:
            esm_encoder, adapter = merged_prot_model_or_components
            protein_inputs = prepare_protein_sequence_inputs_batch(esm_tokenizer, prots_to_embed, 1024)
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
                indices_to_replace = prot_indices[:len(valid_embeds)]
                for j in range(min(len(indices_to_replace), len(valid_embeds))):
                    text_embeds[b, indices_to_replace[j]] = valid_embeds[j]
                prot_embed_counter += 1

        outputs_ids = merged_mol_model.llm.generate(
            inputs_embeds=text_embeds, attention_mask=inputs.attention_mask,
            max_new_tokens=max_length, do_sample=False,
            pad_token_id=llama_tokenizer.eos_token_id, eos_token_id=terminators
        )

    if outputs_ids is None:
        raise RuntimeError(f"Text generation failed for model: {model_name}")
    
    original_input_ids = outputs_ids[:, :inputs.input_ids.shape[1]] if outputs_ids.shape[1] >= inputs.input_ids.shape[1] else inputs.input_ids
    for i in range(batch_size):
        hidden_state_capture.capture_output_token_embeddings(
            llama_tokenizer, outputs_ids[i:i+1], inputs.input_ids.shape[1]
        )
    
    generated_texts = llama_tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
    original_texts = llama_tokenizer.batch_decode(original_input_ids, skip_special_tokens=False)
    
    return generated_texts, original_texts


def run_comprehensive_cross_model_experiment(samples, molecule_dict, system_prompt, user_prompt, output_generated_path, paths, batch_size):
    global global_sample_order, comprehensive_hidden_states
    global_sample_order = samples
    
    model_configs = [
        (1.0, 0.0, 0.0, 'mol_only'),
        (0.0, 1.0, 0.0, 'prot_only'),
        (0.0, 0.0, 1.0, 'cell_only'),
        (0.0, 0.0, 0.0, 'withoutlora')
    ]
    
    all_results = {}
    comprehensive_hidden_states = {name: {} for _, _, _, name in model_configs}

    for model_idx, (mol_scale, prot_scale, cell_scale, model_name) in enumerate(model_configs):
        print(f"\nProcessing {model_name} ({model_idx+1}/{len(model_configs)})")
        torch.cuda.empty_cache()
        
        merged_mol_model, merged_prot_model_or_components, esm_tokenizer, llama_tokenizer, terminators = get_merged_models(
            mol_scale, prot_scale, cell_scale, paths
        )
        
        hidden_capture = ComprehensiveHiddenStateCapture(model_name, llama_tokenizer)
        hidden_capture.clear_states()
        
        if merged_mol_model is None:
            raise ValueError(f"Model loading failed for {model_name}!")
        hidden_capture.register_hooks(merged_mol_model)
        
        model_results = []
        
        for i in tqdm(range(0, len(global_sample_order), batch_size), desc=f"{model_name}"):
            if len(model_results) >= SAMPLE_SIZE:
                break
            
            batch_data = global_sample_order[i:min(i + batch_size, len(global_sample_order))]
            
            preprocessed_mols, protein_sequences, cell_gene_lists = [], [], []
            for row in batch_data:
                protein_sequences.append(row['Protein'])
                cell_gene_lists.append(row['Cell'])
                smiles = row['Drug']
                if smiles in molecule_dict:
                    preprocessed_mols.append(molecule_dict[smiles])
                else:
                    preprocessed_mols.append({'smiles': smiles, 'atoms': [], 'coordinates': np.array([])})
            
            generated_texts, original_texts = comprehensive_predict_with_hidden_state_capture(
                merged_mol_model, merged_prot_model_or_components,
                esm_tokenizer, llama_tokenizer, terminators,
                preprocessed_mols, protein_sequences, cell_gene_lists,
                system_prompt, user_prompt, hidden_capture
            )
            
            for j, row in enumerate(batch_data):
                if len(model_results) >= SAMPLE_SIZE:
                    break
                model_results.append({
                    'Response': generated_texts[j],
                    'Original_Response': original_texts[j],
                    'method': model_name,
                    'sample_idx': len(model_results),
                    'Protein': row['Protein'],
                    'Drug': row['Drug'],
                    'Cell': row['Cell']
                })
            
            del generated_texts, original_texts
            torch.cuda.empty_cache()
        
        total_batches = math.ceil(min(len(global_sample_order), SAMPLE_SIZE) / batch_size)
        organized_states = hidden_capture.get_organized_states_by_sample(batch_size, total_batches)
        comprehensive_hidden_states[model_name] = organized_states
        all_results[model_name] = model_results
        
        pd.DataFrame(model_results).to_csv(
            os.path.join(output_generated_path, f"results_{model_name}.csv"), index=False
        )
        print(f"  {model_name}: {len(model_results)} results, {len(organized_states)} hidden states")
        
        hidden_capture.remove_hooks()
        hidden_capture.clear_states()
        del hidden_capture, merged_mol_model, merged_prot_model_or_components, esm_tokenizer, llama_tokenizer
        torch.cuda.empty_cache()
    
    return all_results, comprehensive_hidden_states


def save_comprehensive_hidden_states(comprehensive_hidden_states, output_path):
    save_path = os.path.join(output_path, "embedding.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(comprehensive_hidden_states, f)
    total = sum(len(d) for d in comprehensive_hidden_states.values())
    print(f"Hidden states saved to {save_path} ({total} total sample records)")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive cross-model analysis - Save hidden states only')
    
    parser.add_argument('--base_model_path', type=str, default='./checkpoints/Llama-3.1-8B-Instruct')
    parser.add_argument('--molecule_model_path', type=str, default='./checkpoints/Mol-Llama-3.1-8B-Instruct')
    parser.add_argument('--esm_path', type=str, default='./checkpoints/esm2_t36_3B_UR50D')
    parser.add_argument('--prot2text_base_checkpoint', type=str, default='./checkpoints/Prot2Text-V2/model_checkpoint.pt')
    parser.add_argument('--prot2text_lora_path', type=str, default='./checkpoints/Prot2Text-V2/adapter_checkpoint')
    parser.add_argument('--cell_lora_path', type=str, default='./checkpoints/cell-o1/lora_checkpoint')
    parser.add_argument('--data_dir', type=str, default='./coefficient_data')
    parser.add_argument('--output_path', type=str, default='./Embedding/')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    paths = {
        'base_model': args.base_model_path,
        'molecule_model': args.molecule_model_path,
        'esm': args.esm_path,
        'prot2text_base_checkpoint': args.prot2text_base_checkpoint,
        'prot2text_lora': args.prot2text_lora_path,
        'cell_lora': args.cell_lora_path,
    }
    
    set_seed(42)
    print(f"Sample size: {SAMPLE_SIZE}, Batch size: {args.batch_size}, Device: {device}")
    
    result = load_dataset_related(args.data_dir)
    proteins, molecules, cells, molecule_dict = load_all_data(args.data_dir)
    samples = create_330_samples(proteins, molecules, cells)
    (_, _, _, _, system_prompt, user_prompt) = result
        
    try:
        all_results, comprehensive_hidden_states = run_comprehensive_cross_model_experiment(
            samples, molecule_dict, system_prompt, user_prompt, args.output_path, paths, args.batch_size
        )
        save_comprehensive_hidden_states(comprehensive_hidden_states, args.output_path)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    
if __name__ == "__main__":
    main()