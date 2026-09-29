import os
import torch
import pandas as pd
import numpy as np
import re
import json
import math
import random
import pickle
import argparse
import warnings
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, Dict, Any, List

from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from transformers import EsmModel, LlamaForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from Mol_LLaMA.models.mol_llama import MolLLaMA, get_mol_graphs_from_preprocessed_mols
from Prot2Text_V2.models import ModalityAdapter, ModalityAdapterConfig, Esm2LlamaInstructForCausalLM

from es_merging import (
    normalize_key_layer, normalize_key_element, extract_layer_index,
    load_layerwise_coefficients, load_elementwise_coefficients,
    _get_common_lora_keys, merge_layerwise, merge_elementwise, merge_es,
    merge_lora_state_dicts,
)

warnings.filterwarnings('ignore')


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_task_type(dataset_name: str) -> str:
    """Detect task type from dataset name."""
    name = dataset_name.lower()
    if "cyp" in name:
        return "cyp"
    elif any(x in name for x in ["bindingdb", "biosnap", "human"]):
        return "dti"
    elif any(x in name for x in ["gdsc", "drugcomb"]):
        return "cell"
    else:
        raise ValueError(f"Cannot detect task type from dataset name: {dataset_name}")


def detect_cyp_subtype(dataset_name: str) -> str:
    name = dataset_name.lower()
    if "inhibition" in name:
        return "inhibition"
    elif "substrate" in name:
        return "substrate"
    raise ValueError(f"Cannot detect CYP subtype from: {dataset_name}")


# ============================================================================
# MODEL LOADING
# ============================================================================

def setup_tokenizers(base_model_path, esm_path, task_type):
    llama_tok = AutoTokenizer.from_pretrained(base_model_path)
    llama_tok.pad_token = '<|reserved_special_token_0|>'
    llama_tok.add_special_tokens({'additional_special_tokens': ['<mol>']})
    llama_tok.mol_token_id = llama_tok("<mol>", add_special_tokens=False).input_ids[0]
    if task_type == "cell":
        llama_tok.bos_token = "<|begin_of_text|>"
        llama_tok.eos_token = "<|eot_id|>"
        llama_tok.pad_token = "<|eot_id|>"
        llama_tok.pad_token_id = llama_tok.eos_token_id
    else:
        llama_tok.pad_token_id = 128002
    llama_tok.protein_placeholder_token_id = 128003
    terminators = [llama_tok.eos_token_id, 128009]
    esm_tok = AutoTokenizer.from_pretrained(esm_path)
    return esm_tok, llama_tok, terminators


def load_prot2text(esm_path, llama_path, base_ckpt, lora_path, device):
    esm_encoder = EsmModel.from_pretrained(esm_path, add_pooling_layer=False, torch_dtype=torch.bfloat16, device_map=device)
    llama_decoder = LlamaForCausalLM.from_pretrained(llama_path, torch_dtype=torch.bfloat16, device_map=device)
    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.config.hidden_size, intermediate_dim=2048,
        output_dim=llama_decoder.config.hidden_size)
    adapter = ModalityAdapter(adapter_config).to(torch.bfloat16).to(device)
    model = Esm2LlamaInstructForCausalLM(esm_encoder=esm_encoder, adapter=adapter, llama_decoder=llama_decoder)
    if os.path.exists(base_ckpt):
        model.load_state_dict(torch.load(base_ckpt, weights_only=True, map_location=device))
    if os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    return model


def load_cell_lora(cell_lora_path, base_llama_path, device):
    base = LlamaForCausalLM.from_pretrained(base_llama_path, torch_dtype=torch.bfloat16, device_map=device)
    cell_model = PeftModel.from_pretrained(base, cell_lora_path, torch_dtype=torch.bfloat16)
    cell_model.eval()
    return cell_model


def load_and_merge_models(args, device, task_type):
    """Load base models, merge LoRA weights, return ready-to-use models."""
    esm_tok, llama_tok, terminators = setup_tokenizers(args.base_model_path, args.esm_path, task_type)

    # Load MolLLaMA
    mol_llama = MolLLaMA.from_pretrained(
        args.molecule_model_path, torch_dtype=torch.bfloat16,
        vocab_size=len(llama_tok), enable_flash=False).to(device)
    mol_llama.llm.resize_token_embeddings(len(llama_tok))

    # Load Prot2Text + Cell LoRA (for weight extraction)
    prot2text = load_prot2text(args.esm_path, args.base_model_path,
                               args.prot2text_base_path, args.prot2text_lora_path, device).to(device)
    cell_model = load_cell_lora(args.cell_lora_path, args.base_model_path, device)

    # Merging
    mol_sd = mol_llama.state_dict()
    prot_sd = prot2text.state_dict()
    cell_sd = cell_model.state_dict()

    mol_sd = merge_lora_state_dicts(
        mol_sd, prot_sd, cell_sd,
        method=args.merging_method,
        layer_coef_path=args.layer_coef_path,
        element_coef_path=args.element_coef_path,
        device=device,
    )

    mol_llama.load_state_dict(mol_sd, strict=True)
    esm_encoder = prot2text.esm_encoder
    adapter = prot2text.adapter

    del prot2text, cell_model
    torch.cuda.empty_cache()

    for m in [mol_llama, esm_encoder, adapter]:
        for p in m.parameters():
            if p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        m.eval()

    return mol_llama, (esm_encoder, adapter), esm_tok, llama_tok, terminators


# ============================================================================
# DATA LOADING
# ============================================================================

def load_preprocessed_train_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    from rdkit import DataStructs as DS
    for item in data:
        fp_list = item.get('fingerprint')
        if fp_list and isinstance(fp_list, list):
            try:
                fp = DS.ExplicitBitVect(2048)
                for i, bit in enumerate(''.join(fp_list)):
                    if bit == '1':
                        fp.SetBit(i)
                item['fingerprint_obj'] = fp
            except:
                item['fingerprint_obj'] = None
        else:
            item['fingerprint_obj'] = None
        for coord_key in ['coordinates', 'Drug1_coordinates', 'Drug2_coordinates']:
            if coord_key in item and item[coord_key]:
                try:
                    item[coord_key] = np.array(item[coord_key], dtype=np.float32)
                except:
                    pass
    print(f"Loaded {len(data)} train samples from {path}")
    return data


def load_molecule_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    mol_dict = {}
    for item in data:
        if 'Drug' in item:
            coords = item.get('coordinates', [])
            if coords and isinstance(coords, list):
                try: coords = np.array(coords, dtype=np.float32)
                except: pass
            entry = {'smiles': item.get('Drug', ''), 'atoms': item.get('atoms', []), 'coordinates': coords}
            if item.get('Drug'):
                mol_dict[item['Drug']] = entry
        elif 'Drug1' in item and 'Drug2' in item:
            for suffix in ['1', '2']:
                ckey = f'Drug{suffix}_coordinates'
                c = item.get(ckey, [])
                if c and isinstance(c, list):
                    try: item[ckey] = np.array(c, dtype=np.float32)
                    except: pass
            key = (item.get('Drug1', ''), item.get('Drug2', ''))
            mol_dict[key] = {
                'smiles1': item.get('Drug1', ''), 'smiles2': item.get('Drug2', ''),
                'atoms1': item.get('Drug1_atoms', []), 'coordinates1': item.get('Drug1_coordinates', []),
                'atoms2': item.get('Drug2_atoms', []), 'coordinates2': item.get('Drug2_coordinates', []),
            }
    print(f"Loaded {len(mol_dict)} molecules from {path}")
    return mol_dict


def load_protein_embeddings(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return {item['Protein']: item['embedding'] for item in data['data']}


# ============================================================================
# ICL EXAMPLE SELECTION
# ============================================================================

def find_icl_cyp(target_smiles, train_data, top_k=5):
    """Tanimoto-based ICL for CYP tasks."""
    try:
        mol = Chem.MolFromSmiles(target_smiles)
        if not mol: return []
        target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except:
        return []
    sims = []
    for item in train_data:
        fp = item.get('fingerprint_obj')
        if fp:
            try: sims.append((TanimotoSimilarity(target_fp, fp), item))
            except: continue
    sims.sort(key=lambda x: x[0], reverse=True)
    return [{
        'smiles': it['Drug'], 'protein': it['Target'],
        'label': f" {it['Y']}", 'atoms': it['atoms'], 'coordinates': it['coordinates']
    } for _, it in sims[:top_k]]


def find_icl_dti(target_protein, target_smiles, dataset_name, train_data,
                 train_prot_emb, test_prot_emb, top_k=5):

    drug_key = 'ligand' if "biosnap" in dataset_name.lower() else 'Drug'
    prot_key = 'Target'

    try:
        mol = Chem.MolFromSmiles(target_smiles)
        target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None
    except:
        target_fp = None

    # Exact protein matches
    exact = [it for it in train_data if it.get(prot_key) == target_protein]

    if len(exact) == top_k:
        final = exact[:top_k]
    elif len(exact) > top_k:
        if target_fp:
            scored = []
            for it in exact:
                fp = it.get('fingerprint_obj')
                sim = TanimotoSimilarity(target_fp, fp) if fp else 0.0
                scored.append((sim, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            final = [it for _, it in scored[:top_k]]
        else:
            final = exact[:top_k]
    else:
        final = list(exact)
        needed = top_k - len(final)
        # Compare record identity; coordinate arrays do not support scalar equality.
        exact_ids = {id(it) for it in exact}
        if train_prot_emb and test_prot_emb and target_protein in test_prot_emb:
            t_emb = test_prot_emb[target_protein]
            prot_sims = []
            for it in train_data:
                if id(it) in exact_ids: continue
                p = it.get(prot_key)
                if p in train_prot_emb:
                    sim = np.dot(t_emb, train_prot_emb[p]) / (np.linalg.norm(t_emb) * np.linalg.norm(train_prot_emb[p]) + 1e-12)
                    prot_sims.append((sim, it))
            prot_sims.sort(key=lambda x: x[0], reverse=True)
            for _, it in prot_sims[:needed]:
                final.append(it)
        else:
            remaining = [it for it in train_data if id(it) not in exact_ids]
            final.extend(remaining[:needed])

    def label_text(y):
        return "Interacts" if y in [1, 1.0, "1", "1.0"] else "Does not interact"

    return [{
        'smiles': it.get(drug_key, ''), 'protein': it.get(prot_key, ''),
        'label': label_text(it['Y']), 'atoms': it.get('atoms', []),
        'coordinates': it.get('coordinates', [])
    } for it in final[:top_k]]


def precompute_fingerprints(train_data, test_df, columns, dataset_name):

    fp_cache = {}
    is_combo = "drugcomb" in dataset_name.lower()
    for item in (train_data or []):
        for dk in (['Drug1', 'Drug2'] if 'Drug1' in item else ['Drug']):
            s = item.get(dk)
            if s and s not in fp_cache:
                try:
                    m = Chem.MolFromSmiles(s)
                    if m: fp_cache[s] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                except: pass
    if test_df is not None:
        drug_cols = [columns.get("Drug1"), columns.get("Drug2")] if is_combo else [columns.get("Drug")]
        for _, row in test_df.iterrows():
            for dc in drug_cols:
                if dc:
                    s = row.get(dc)
                    if s and s not in fp_cache:
                        try:
                            m = Chem.MolFromSmiles(s)
                            if m: fp_cache[s] = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
                        except: pass
    return fp_cache


def jaccard_genes(g1, g2):
    s1, s2 = set(g1.split(',')), set(g2.split(','))
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0


def find_icl_cell_single(target_smiles, target_genes, train_data, fp_cache, top_k=5):

    target_fp = fp_cache.get(target_smiles)
    if not target_fp: return []
    exact = [it for it in train_data if it['Top_50_Genes'] == target_genes]
    if len(exact) == top_k:
        final = exact
    elif len(exact) > top_k:
        scored = [(TanimotoSimilarity(target_fp, fp_cache.get(it['Drug'], target_fp)), it) for it in exact]
        scored.sort(key=lambda x: x[0], reverse=True)
        final = [it for _, it in scored[:top_k]]
    else:
        final = list(exact)
        needed = top_k - len(final)
        others = [(jaccard_genes(target_genes, it['Top_50_Genes']), it) for it in train_data if it not in exact]
        others.sort(key=lambda x: x[0], reverse=True)
        for _, it in others[:needed]:
            final.append(it)
    return [{
        'smiles': it['Drug'], 'genes': it['Top_50_Genes'],
        'label': f" {it['Y']}", 'atoms': it['atoms'], 'coordinates': it['coordinates']
    } for it in final[:top_k]]


def find_icl_cell_combo(target_s1, target_s2, target_genes, train_data, fp_cache, top_k=5):

    fp1, fp2 = fp_cache.get(target_s1), fp_cache.get(target_s2)
    if not fp1 or not fp2: return []
    exact = [it for it in train_data if it['Top_50_Genes'] == target_genes]
    def combo_sim(it):
        f1, f2 = fp_cache.get(it['Drug1']), fp_cache.get(it['Drug2'])
        if f1 and f2:
            return (TanimotoSimilarity(fp1, f1) + TanimotoSimilarity(fp2, f2)) / 2
        return 0.0
    if len(exact) == top_k:
        final = exact
    elif len(exact) > top_k:
        scored = sorted(exact, key=combo_sim, reverse=True)
        final = scored[:top_k]
    else:
        final = list(exact)
        needed = top_k - len(final)
        others = [(jaccard_genes(target_genes, it['Top_50_Genes']), it) for it in train_data if it not in exact]
        others.sort(key=lambda x: x[0], reverse=True)
        for _, it in others[:needed]:
            final.append(it)
    return [{
        'smiles1': it['Drug1'], 'smiles2': it['Drug2'], 'genes': it['Top_50_Genes'],
        'label': f" {it['Y']}", 'atoms1': it['Drug1_atoms'], 'coordinates1': it['Drug1_coordinates'],
        'atoms2': it['Drug2_atoms'], 'coordinates2': it['Drug2_coordinates']
    } for it in final[:top_k]]


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def get_prompts(dataset_name, task_type):
    if task_type == "cyp":
        sub = detect_cyp_subtype(dataset_name)
        sys_p = ("You are an expert specialized in drug discovery and molecular biology. "
                 "You will be given a protein and a molecule. Your task is to determine whether "
                 f"a given molecule {'inhibits a specific protein' if sub == 'inhibition' else 'is a substrate of a specific protein'}.")
        if sub == "inhibition":
            usr_p = ("Determine whether the given molecule inhibits the protein by following the examples.\n"
                     "    Examples:\n    {examples_placeholder}\n\n"
                     "    Protein: <protein>\n    Molecule: <mol>\n\n"
                     "Your final answer should be formatted as either: "
                     "'Final answer: Inhibits' or 'Final answer: Does not inhibit'.")
        else:
            usr_p = ("Determine whether the given molecule is a substrate of the protein by following the examples.\n"
                     "    Examples:\n    {examples_placeholder}\n\n"
                     "    Protein: <protein>\n    Molecule: <mol>\n\n"
                     "Your final answer should be formatted as either: "
                     "'Final answer: Substrate' or 'Final answer: Not a substrate'.")
    elif task_type == "dti":
        sys_p = ("You are an expert specialized in drug discovery and molecular biology. "
                 "You will be given a protein and a molecule. Your task is to determine whether "
                 "a given molecule interacts with a specific protein.")
        usr_p = ("Determine whether the given molecule interacts with the protein by following the examples.\n"
                 "    Examples:\n    {examples_placeholder}\n\n"
                 "    Protein: <protein>\n    Molecule: <mol>\n\n"
                 "Your final answer must be exactly one of the following:\n"
                 "- 'Final answer: Interacts'\n- 'Final answer: Does not interact'.")
    elif task_type == "cell":
        is_combo = "drugcomb" in dataset_name.lower()
        if is_combo:
            sys_p = ("You are an expert specialized in cancer pharmacogenomics and anticancer drug-combination response prediction. "
                     "You will be given two anticancer molecules and a cancer cell represented as a ranked list of gene names. "
                     "Predict the binary interaction outcome: Synergistic or Antagonistic.")
            usr_p = ("Predict whether the drug pair is Synergistic or Antagonistic.\n"
                     "Examples:\n{examples_placeholder}\n\n"
                     "Molecule A: <mol>\nMolecule B: <mol>\nCell: <gene>\n\n"
                     "Your final answer should be formatted as either: "
                     "'Final answer: Synergistic' or 'Final answer: Antagonistic'.")
        else:
            sys_p = ("You are an expert specialized in cancer pharmacogenomics and drug response prediction. "
                     "You will be given an anticancer molecule and a cancer cell represented as a list of gene names "
                     "ordered by expression. Predict whether the molecule will suppress the cancer cell.")
            usr_p = ("Determine whether the given cell is sensitive or resistant to the given molecule.\n"
                     "Examples:\n{examples_placeholder}\n\n"
                     "Cell: <gene>\nDrug: <mol>\n\n"
                     "Your final answer should be formatted as either: "
                     "'Final answer: Sensitive' or 'Final answer: Resistant'.")
    return sys_p, usr_p


# ============================================================================
# TOKENIZATION & EMBEDDING
# ============================================================================

def tokenize_batch(batch_messages, tokenizer, device, padding_side='left'):
    all_ids, all_masks, all_mol_flags, all_prot_flags = [], [], [], []
    max_len = 0
    for messages in batch_messages:
        ids, masks = [], []
        has_sys = messages and messages[0]['role'] == 'system'
        for idx, m in enumerate(messages):
            if m['role'] == 'system':
                t = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + m['content'] + "<|eot_id|>"
            elif m['role'] == 'user':
                prefix = "<|begin_of_text|>" if (idx == 0 and not has_sys) else ""
                t = prefix + "<|start_header_id|>user<|end_header_id|>\n\n" + m['content'] + "<|eot_id|>"
                t += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            tok = tokenizer(t, add_special_tokens=False)
            ids.extend(tok['input_ids'])
            masks.extend(tok['attention_mask'])
        max_len = max(max_len, len(ids))
        mol_flag = [t == tokenizer.mol_token_id for t in ids]
        prot_flag = [t == tokenizer.protein_placeholder_token_id for t in ids]
        all_ids.append(ids); all_masks.append(masks)
        all_mol_flags.append(mol_flag); all_prot_flags.append(prot_flag)

    pad_id = tokenizer.pad_token_id
    padded = {'input_ids': [], 'attention_mask': [], 'mol_token_flag': [], 'protein_placeholder_flag': []}
    for ids, masks, mf, pf in zip(all_ids, all_masks, all_mol_flags, all_prot_flags):
        pad_n = max_len - len(ids)
        if padding_side == 'left':
            padded['input_ids'].append([pad_id]*pad_n + ids)
            padded['attention_mask'].append([0]*pad_n + masks)
            padded['mol_token_flag'].append([False]*pad_n + mf)
            padded['protein_placeholder_flag'].append([False]*pad_n + pf)
        else:
            padded['input_ids'].append(ids + [pad_id]*pad_n)
            padded['attention_mask'].append(masks + [0]*pad_n)
            padded['mol_token_flag'].append(mf + [False]*pad_n)
            padded['protein_placeholder_flag'].append(pf + [False]*pad_n)

    return BatchEncoding({k: torch.tensor(v, device=device) for k, v in padded.items()})


def prepare_mol_embeddings(mol_llama, mol_data_list, device):
    embeds = []
    for md in mol_data_list:
        try:
            coords = md.get('coordinates', [])
            has_data = (isinstance(coords, np.ndarray) and coords.size > 0) or (isinstance(coords, list) and len(coords) > 0)
            if not md.get('atoms') or not has_data:
                embeds.append(torch.zeros((1, 32, mol_llama.llm_proj.out_features), device=device))
                continue
            graph = get_mol_graphs_from_preprocessed_mols([md], mol_llama.encoder.unimol_dictionary, device)
            with torch.no_grad():
                _, _, qo = mol_llama.encoder.graph_forward(graph)
                embeds.append(mol_llama.llm_proj(qo.last_hidden_state))
        except Exception as e:
            print(f"  Mol embed error: {e}")
            embeds.append(torch.zeros((1, 32, mol_llama.llm_proj.out_features), device=device))
    return torch.cat(embeds, dim=0)


def prepare_prot_inputs(esm_tok, seqs, device, max_len=1024):
    batch = []
    for s in seqs:
        inp = esm_tok(s, return_tensors="pt", padding="max_length", truncation=True,
                      max_length=max_len, add_special_tokens=True)
        batch.append({'input_ids': inp.input_ids.squeeze(0), 'attention_mask': inp.attention_mask.squeeze(0)})
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]).to(device),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]).to(device)
    }


# ============================================================================
# PREDICTION EXTRACTION & EVALUATION
# ============================================================================

def extract_prediction(response, dataset_name, task_type):
    r = response.strip().lower()
    match = re.search(r"final\s*answer(?:\s+is)?\s*:?\s*(.+)", r)
    part = match.group(1).strip() if match else r

    if task_type == "cyp":
        sub = detect_cyp_subtype(dataset_name)
        if sub == "inhibition":
            if "does not inhibit" in part or "not inhibit" in part: return "Does not inhibit"
            if "inhibits" in part: return "Inhibits"
        else:
            if "not a substrate" in part or "not substrate" in part: return "Not a substrate"
            if "substrate" in part: return "Substrate"
    elif task_type == "dti":
        if "does not interact" in part or "not interact" in part: return "Does not interact"
        if "interacts" in part or "interact" in part: return "Interacts"
    elif task_type == "cell":
        is_combo = "drugcomb" in dataset_name.lower()
        if is_combo:
            if "antagonistic" in part: return "Antagonistic"
            if "synergistic" in part: return "Synergistic"
        else:
            if "resistant" in part: return "Resistant"
            if "sensitive" in part: return "Sensitive"
    return None


def normalize_ground_truth(gt, dataset_name, task_type):
    if task_type == "dti":
        if gt in [1, 1.0, "1", "1.0"]: return "Interacts"
        if gt in [0, 0.0, "0", "0.0"]: return "Does not interact"
    return str(gt).strip()


def compute_metrics(results, dataset_name, task_type, y_col):
    total, valid, correct = 0, 0, 0
    preds, gts = [], []
    for r in results:
        gt = r.get(y_col)
        if gt is None or gt == 'N/A': continue
        total += 1
        gt_norm = normalize_ground_truth(gt, dataset_name, task_type)
        pred = r.get('Predicted_Value')
        if pred is None:
            continue
        valid += 1
        preds.append(pred); gts.append(gt_norm)
        if pred.lower() == gt_norm.lower():
            correct += 1

    # F1-macro
    from collections import Counter
    classes = set(gts) | set(preds)
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gts) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    f1_macro = sum(f1s) / len(f1s) if f1s else 0

    return {
        'total': total, 'valid': valid, 'correct': correct,
        'invalid': total - valid,
        'accuracy': correct / valid if valid > 0 else 0,
        'total_accuracy': correct / total if total > 0 else 0,
        'f1_macro': f1_macro,
        'error_rate': (total - valid) / total if total > 0 else 0,
    }


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

def run_inference(mol_model, prot_components, esm_tok, llama_tok, terminators,
                  df, mol_dict, columns, dataset_name, task_type,
                  system_prompt, user_prompt,
                  train_data=None, train_prot_emb=None, test_prot_emb=None,
                  fp_cache=None, batch_size=32, device='cuda'):
    
    esm_encoder, adapter = prot_components
    is_combo = "drugcomb" in dataset_name.lower()
    samples = df.to_dict('records')
    results = []

    MOL_PH = '<mol>' * 8
    PROT_PH_TOK = llama_tok.decode([llama_tok.protein_placeholder_token_id])

    for start in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        batch = samples[start:start + batch_size]
        bs = len(batch)

        batch_msgs, mols_to_embed, prots_to_embed = [], [], []
        per_sample_prot_count = []  # track protein count per sample

        for i, row in enumerate(batch):
            # Expand target molecules before inserting already-expanded ICL examples.
            usr = user_prompt.replace('<mol>', MOL_PH)
            ex_str = ""
            n_prots_this_sample = 0

            # ICL examples
            icl_ex = []
            if train_data:
                if task_type == "cyp":
                    icl_ex = find_icl_cyp(row[columns["Drug"]], train_data)
                elif task_type == "dti":
                    icl_ex = find_icl_dti(row[columns["Protein"]], row[columns["Drug"]],
                                          dataset_name, train_data, train_prot_emb, test_prot_emb)
                elif task_type == "cell":
                    if is_combo:
                        icl_ex = find_icl_cell_combo(row[columns["Drug1"]], row[columns["Drug2"]],
                                                     row[columns["Genes"]], train_data, fp_cache)
                    else:
                        icl_ex = find_icl_cell_single(row[columns["Drug"]], row[columns["Genes"]],
                                                      train_data, fp_cache)

            # Build examples string
            for eidx, ex in enumerate(icl_ex):
                ex_str += f"    Example {eidx+1}:\n"
                if task_type in ["cyp", "dti"]:
                    prot_len = len(ex['protein']) + 2
                    ex_str += f"        Protein: {PROT_PH_TOK * prot_len}\n"
                    prots_to_embed.append(ex['protein'])
                    n_prots_this_sample += 1
                    ex_str += f"        Molecule: {MOL_PH}\n"
                    mols_to_embed.append({'smiles': ex['smiles'], 'atoms': ex['atoms'], 'coordinates': ex['coordinates']})
                elif task_type == "cell":
                    if is_combo:
                        ex_str += f"        Molecule A: {MOL_PH}\n"
                        mols_to_embed.append({'smiles': ex['smiles1'], 'atoms': ex['atoms1'], 'coordinates': ex['coordinates1']})
                        ex_str += f"        Molecule B: {MOL_PH}\n"
                        mols_to_embed.append({'smiles': ex['smiles2'], 'atoms': ex['atoms2'], 'coordinates': ex['coordinates2']})
                        ex_str += f"        Cell: {ex['genes']}\n"
                    else:
                        ex_str += f"        Drug: {MOL_PH}\n"
                        mols_to_embed.append({'smiles': ex['smiles'], 'atoms': ex['atoms'], 'coordinates': ex['coordinates']})
                        ex_str += f"        Cell: {ex['genes']}\n"
                ex_str += f"        Final answer: {ex['label']}\n"

            usr = usr.format(examples_placeholder=ex_str)

            # Target tokens
            if task_type in ["cyp", "dti"]:
                prot_len = len(row[columns["Protein"]]) + 2
                usr = usr.replace('<protein>', PROT_PH_TOK * prot_len, 1)
                prots_to_embed.append(row[columns["Protein"]])
                n_prots_this_sample += 1
                mols_to_embed.append(mol_dict.get(row[columns["Drug"]], {'smiles': row[columns["Drug"]], 'atoms': [], 'coordinates': []}))
            elif task_type == "cell":
                usr = usr.replace('<gene>', row[columns["Genes"]], 1)
                if is_combo:
                    key = (row[columns["Drug1"]], row[columns["Drug2"]])
                    combo = mol_dict.get(key, {'smiles1': '', 'smiles2': '', 'atoms1': [], 'coordinates1': [], 'atoms2': [], 'coordinates2': []})
                    mols_to_embed.append({'smiles': combo.get('smiles1',''), 'atoms': combo.get('atoms1',[]), 'coordinates': combo.get('coordinates1',[])})
                    mols_to_embed.append({'smiles': combo.get('smiles2',''), 'atoms': combo.get('atoms2',[]), 'coordinates': combo.get('coordinates2',[])})
                else:
                    mols_to_embed.append(mol_dict.get(row[columns["Drug"]], {'smiles': row[columns["Drug"]], 'atoms': [], 'coordinates': []}))

            per_sample_prot_count.append(n_prots_this_sample)
            batch_msgs.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": usr}])

        # Tokenize
        inputs = tokenize_batch(batch_msgs, llama_tok, device)

        with torch.no_grad():
            text_emb = mol_model.llm.get_input_embeddings()(inputs.input_ids).to(dtype=torch.bfloat16)

            # Inject molecule embeddings
            if mols_to_embed:
                all_mol_emb = prepare_mol_embeddings(mol_model, mols_to_embed, device)
                counter = 0
                for b in range(bs):
                    mol_idx = torch.where(inputs.mol_token_flag[b])[0]
                    n_groups = len(mol_idx) // 8
                    for g in range(n_groups):
                        if counter >= len(all_mol_emb): break
                        emb = all_mol_emb[counter]
                        for j in range(min(8, len(mol_idx) - g*8)):
                            text_emb[b, mol_idx[g*8+j]] = emb[j]
                        counter += 1

            # Inject protein embeddings (CYP/DTI only)
            if prots_to_embed and task_type in ["cyp", "dti"]:
                prot_inputs = prepare_prot_inputs(esm_tok, prots_to_embed, device)
                with torch.no_grad():
                    esm_out = esm_encoder(**prot_inputs)
                    all_prot_emb = adapter(esm_out.last_hidden_state)
                counter = 0
                for b in range(bs):
                    prot_idx = torch.where(inputs.protein_placeholder_flag[b])[0]
                    offset = 0
                    n_prots = per_sample_prot_count[b]
                    for p in range(n_prots):
                        if counter >= len(all_prot_emb): break
                        vlen = int(prot_inputs['attention_mask'][counter].sum().item())
                        for j in range(min(vlen, len(prot_idx) - offset)):
                            text_emb[b, prot_idx[offset + j]] = all_prot_emb[counter][j]
                        offset += vlen
                        counter += 1

            out_ids = mol_model.llm.generate(
                inputs_embeds=text_emb, attention_mask=inputs.attention_mask,
                max_new_tokens=512, do_sample=False,
                pad_token_id=llama_tok.eos_token_id, eos_token_id=terminators)

        texts = llama_tok.batch_decode(out_ids, skip_special_tokens=True)

        for i, row in enumerate(batch):
            pred = extract_prediction(texts[i], dataset_name, task_type)
            result = {
                'Predicted_Value': pred,
                'Response': texts[i],
                'dataset': dataset_name,
                'sample_idx': start + i,
            }
            for cn, ck in columns.items():
                if ck: result[ck] = row.get(ck, 'N/A')
            results.append(result)

        torch.cuda.empty_cache()

    return results


# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

def get_columns(dataset_name, task_type):
    if task_type == "cyp":
        return {"Drug": "Drug", "Protein": "Target", "Y": "Y"}
    elif task_type == "dti":
        return {"Drug": "Ligand", "Protein": "Protein", "Y": "classification_label"}
    elif task_type == "cell":
        if "drugcomb" in dataset_name.lower():
            return {"Drug1": "Drug1", "Drug2": "Drug2", "Genes": "Top_50_Genes", "Y": "Y"}
        else:
            return {"Drug": "Drug", "Genes": "Top_50_Genes", "Y": "Y"}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified ES-Merging Inference")

    # Dataset & Task
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (e.g., CYP1A2_inhibition, BindingDB_protein, GDSC2, DrugComb)')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--molecule_3d_path', type=str, required=True, help='Path to molecule 3D JSON')
    parser.add_argument('--train_json_path', type=str, default=None, help='Path to train JSON for ICL')

    # DTI-specific
    parser.add_argument('--train_prot_emb_path', type=str, default=None, help='Train protein embeddings pkl (DTI)')
    parser.add_argument('--test_prot_emb_path', type=str, default=None, help='Test protein embeddings pkl (DTI)')

    # Merging
    parser.add_argument('--merging_method', type=str, required=True, choices=['layerwise', 'elementwise', 'es'],
                        help='Merging method: layerwise, elementwise, or es (full ES-Merging)')
    parser.add_argument('--layer_coef_path', type=str, default=None, help='Path to layer-wise coef CSV')
    parser.add_argument('--element_coef_path', type=str, default=None, help='Path to element-wise coef .pt')

    # Model paths
    parser.add_argument('--base_model_path', type=str, required=True, help='LLaMA base model path')
    parser.add_argument('--molecule_model_path', type=str, required=True, help='Mol-LLaMA model path')
    parser.add_argument('--esm_path', type=str, required=True, help='ESM2 model path')
    parser.add_argument('--prot2text_base_path', type=str, required=True, help='Prot2Text base checkpoint')
    parser.add_argument('--prot2text_lora_path', type=str, required=True, help='Prot2Text LoRA path')
    parser.add_argument('--cell_lora_path', type=str, required=True, help='Cell LoRA path')

    # Output
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    # Misc
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect task
    task_type = detect_task_type(args.dataset_name)
    columns = get_columns(args.dataset_name, task_type)
    system_prompt, user_prompt = get_prompts(args.dataset_name, task_type)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name} | Task: {task_type} | Merge: {args.merging_method}")
    print(f"{'='*60}")

    # Load test data
    if args.test_data_path.endswith('.json'):
        with open(args.test_data_path) as f:
            df = pd.DataFrame(json.load(f))
    else:
        df = pd.read_csv(args.test_data_path)
    print(f"Test data: {len(df)} samples")

    # Balance/sample
    y_col = columns["Y"]
    if y_col in df.columns:
        df = df.dropna(subset=[y_col])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After preprocessing: {len(df)} samples")

    # Load molecule data
    mol_dict = load_molecule_data(args.molecule_3d_path)

    train_data = None
    if args.train_json_path:
        train_data = load_preprocessed_train_json(args.train_json_path)

    train_prot_emb, test_prot_emb = None, None
    if task_type == "dti" and args.train_prot_emb_path and args.test_prot_emb_path:
        train_prot_emb = load_protein_embeddings(args.train_prot_emb_path)
        test_prot_emb = load_protein_embeddings(args.test_prot_emb_path)


    fp_cache = None
    if task_type == "cell":
        fp_cache = precompute_fingerprints(train_data, df, columns, args.dataset_name)

    # Load and merge models
    print("\nLoading and merging models...")
    mol_model, prot_components, esm_tok, llama_tok, terminators = load_and_merge_models(args, device, task_type)

    # Run inference
    print("\nRunning evaluation...")
    results = run_inference(
        mol_model, prot_components, esm_tok, llama_tok, terminators,
        df, mol_dict, columns, args.dataset_name, task_type,
        system_prompt, user_prompt,
        train_data=train_data, train_prot_emb=train_prot_emb, test_prot_emb=test_prot_emb,
        fp_cache=fp_cache, batch_size=args.batch_size, device=device)

    # Evaluate
    metrics = compute_metrics(results, args.dataset_name, task_type, columns["Y"])
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.dataset_name} ({args.merging_method})")
    print(f"{'='*60}")
    print(f"  Total: {metrics['total']} | Valid: {metrics['valid']} | Correct: {metrics['correct']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-macro: {metrics['f1_macro']:.4f}")
    print(f"  Error rate: {metrics['error_rate']:.4f}")

    # Save
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"results_{args.dataset_name}_{args.merging_method}.csv")
    results_df.to_csv(csv_path, index=False)

    metrics_path = os.path.join(args.output_dir, f"metrics_{args.dataset_name}_{args.merging_method}.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'dataset': args.dataset_name, 'task_type': task_type,
            'merging_method': args.merging_method, **metrics,
        }, f, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {metrics_path}")
    print("Done!")


if __name__ == "__main__":
    main()