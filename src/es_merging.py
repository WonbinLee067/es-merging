"""ES-Merging for molecule, protein, and cell LoRA state dictionaries.

The merging functions operate on state dictionaries and do not load datasets,
tokenizers, or expert model implementations. Matching LoRA entries in the
molecule state dictionary are updated in place; its other entries are retained.
"""

import re
from typing import Dict, Tuple

import pandas as pd
import torch


# ============================================================================
# KEY NORMALIZATION
# ============================================================================

def normalize_key_layer(k: str) -> str:
    k = k.lower()
    for prefix in [
        'llm.base_model.model.model.', 'base_model.model.llama_decoder.model.',
        'base_model.model.model.', 'llm.model.', 'model.',
    ]:
        if k.startswith(prefix):
            k = k[len(prefix):]
            break
    if not k.startswith('layers.') and 'layers.' in k:
        k = k[k.index('layers.'):]
    k = k.replace('.default.weight', '.weight').replace('.default', '')
    k = k.replace('.lora_a.weight', '.lora_A.weight').replace('.lora_b.weight', '.lora_B.weight')
    return k


def normalize_key_element(k: str) -> str:
    k = k.lower()
    prefixes = [
        'base_model.model.base_model.model.model.', 'base_model.model.base_model.model.',
        'base_model.model.llm.base_model.model.', 'base_model.model.llm.model.',
        'llm.base_model.model.model.', 'llm.base_model.model.',
        'base_model.model.llama_decoder.model.', 'llama_decoder.model.',
        'base_model.model.', 'llm.model.', 'llm.', 'model.',
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
                changed = True
                break
    k = k.replace('.default.weight', '.weight').replace('.default', '')
    k = k.replace('.lora_A.', '.lora_a.').replace('.lora_B.', '.lora_b.')
    return k


def extract_layer_index(key: str):
    match = re.search(r'\.layers\.(\d+)\.', key)
    return int(match.group(1)) if match else None


# ============================================================================
# COEFFICIENT LOADING
# ============================================================================

def load_layerwise_coefficients(csv_path: str) -> Dict[int, Tuple[float, float, float]]:
    print(f"Loading layer-wise coefficients from: {csv_path}")
    df = pd.read_csv(csv_path)
    coefficients = {}
    for _, row in df.iterrows():
        layer_name = row['layer']
        if layer_name.startswith('layer_'):
            try:
                idx = int(layer_name.split('_')[1])
                if 0 <= idx <= 31:
                    coefficients[idx] = (float(row['coef_mol']), float(row['coef_prot']), float(row['coef_cell']))
            except (ValueError, IndexError):
                continue
    for i in range(32):
        if i not in coefficients:
            coefficients[i] = (1/3, 1/3, 1/3)
    print(f"  Loaded coefficients for {len(coefficients)} layers")
    return coefficients


def load_elementwise_coefficients(pt_path: str):
    print(f"Loading element-wise coefficients from: {pt_path}")
    payload = torch.load(pt_path, map_location="cpu")
    coef_mol = payload.get("coef_mol", {})
    coef_prot = payload.get("coef_prot", {})
    coef_cell = payload.get("coef_cell", {})
    if not coef_mol or not coef_prot or not coef_cell:
        raise ValueError("coef_mol/coef_prot/coef_cell not found in coefficient file")
    print(f"  Loaded coefficients for {len(coef_mol)} parameters")
    return coef_mol, coef_prot, coef_cell


# ============================================================================
# MERGING FUNCTIONS
# ============================================================================

def _get_common_lora_keys(mol_sd, prot_sd, cell_sd, normalize_fn):
    mol_keys = {normalize_fn(k): k for k in mol_sd if 'lora_' in k.lower()}
    prot_keys = {normalize_fn(k): k for k in prot_sd if 'lora_' in k.lower()}
    cell_keys = {normalize_fn(k): k for k in cell_sd if 'lora_' in k.lower()}
    common = set(mol_keys) & set(prot_keys) & set(cell_keys)
    print(f"  Common LoRA keys: {len(common)}")
    return mol_keys, prot_keys, cell_keys, common


def merge_layerwise(mol_sd, prot_sd, cell_sd, coefficients, device):
    print("\n=== Layer-wise LoRA Merging ===")
    mol_map, prot_map, cell_map, common = _get_common_lora_keys(mol_sd, prot_sd, cell_sd, normalize_key_layer)
    merged = 0
    for nk in common:
        mk, pk, ck = mol_map[nk], prot_map[nk], cell_map[nk]
        layer_idx = extract_layer_index(mk)
        if layer_idx is None or layer_idx not in coefficients:
            continue
        mt, pt, ct = mol_sd[mk].to(device), prot_sd[pk].to(device), cell_sd[ck].to(device)
        if mt.shape != pt.shape or mt.shape != ct.shape:
            continue
        cm, cp, cc = coefficients[layer_idx]
        mol_sd[mk] = cm * mt + cp * pt + cc * ct
        merged += 1
    print(f"  Merged: {merged} parameters")
    return mol_sd


def merge_elementwise(mol_sd, prot_sd, cell_sd, coef_mol, coef_prot, coef_cell, device):
    print("\n=== Element-wise LoRA Merging ===")
    mol_map, prot_map, cell_map, common = _get_common_lora_keys(mol_sd, prot_sd, cell_sd, normalize_key_element)
    merged, missing = 0, 0
    for nk in common:
        mk, pk, ck = mol_map[nk], prot_map[nk], cell_map[nk]
        mt, pt, ct = mol_sd[mk], prot_sd[pk], cell_sd[ck]
        cm, cp, cc = coef_mol.get(nk), coef_prot.get(nk), coef_cell.get(nk)
        if cm is None or cp is None or cc is None:
            missing += 1
            continue
        if mt.shape != pt.shape or mt.shape != ct.shape or cm.shape != mt.shape:
            continue
        mt, pt, ct = mt.to(device), pt.to(device), ct.to(device)
        cm = cm.to(device=device, dtype=mt.dtype)
        cp = cp.to(device=device, dtype=mt.dtype)
        cc = cc.to(device=device, dtype=mt.dtype)
        mol_sd[mk] = cm * mt + cp * pt + cc * ct
        merged += 1
    print(f"  Merged: {merged}, Coef missing: {missing}")
    return mol_sd


def merge_es(mol_sd, prot_sd, cell_sd, coef_mol_w, coef_prot_w, coef_cell_w,
             layer_coefficients, device):
    """ES-Merging: element-wise × layer-wise with 'none' mixing."""
    print("\n=== ES-Merging (Element × Layer) ===")
    mol_map, prot_map, cell_map, common = _get_common_lora_keys(mol_sd, prot_sd, cell_sd, normalize_key_element)
    merged, missing = 0, 0
    eps = 1e-12
    for nk in common:
        mk, pk, ck = mol_map[nk], prot_map[nk], cell_map[nk]
        layer_idx = extract_layer_index(mk)
        if layer_idx is None or layer_idx not in layer_coefficients:
            continue
        mt, pt, ct = mol_sd[mk].to(device), prot_sd[pk].to(device), cell_sd[ck].to(device)
        cm_w, cp_w, cc_w = coef_mol_w.get(nk), coef_prot_w.get(nk), coef_cell_w.get(nk)
        if cm_w is None or cp_w is None or cc_w is None:
            missing += 1
            continue
        if mt.shape != pt.shape or mt.shape != ct.shape or cm_w.shape != mt.shape:
            continue
        cm_w = cm_w.to(device=device, dtype=mt.dtype)
        cp_w = cp_w.to(device=device, dtype=mt.dtype)
        cc_w = cc_w.to(device=device, dtype=mt.dtype)
        cm_l, cp_l, cc_l = layer_coefficients[layer_idx]
        # none mixing: product then normalize
        w_mol = cm_w * cm_l
        w_prot = cp_w * cp_l
        w_cell = cc_w * cc_l
        denom = w_mol + w_prot + w_cell + eps
        mol_sd[mk] = (w_mol / denom) * mt + (w_prot / denom) * pt + (w_cell / denom) * ct
        merged += 1
    print(f"  Merged: {merged}, Coef missing: {missing}")
    return mol_sd


def merge_lora_state_dicts(
    mol_sd, prot_sd, cell_sd, *, method="es",
    layer_coef_path=None, element_coef_path=None, device="cpu",
):
    """Load coefficients and merge three experts into ``mol_sd`` in place.

    ``method`` is ``es``, ``layerwise``, or ``elementwise``. Full ES-Merging
    requires both coefficient files. State-dict keys may retain the experts'
    original model prefixes; each merging function normalizes them for matching.
    The returned dictionary retains the molecule model's original key names.
    """
    if method == "layerwise":
        if not layer_coef_path:
            raise ValueError("--layer_coef_path required for layerwise es-merging")
        coeffs = load_layerwise_coefficients(layer_coef_path)
        return merge_layerwise(mol_sd, prot_sd, cell_sd, coeffs, device)
    if method == "elementwise":
        if not element_coef_path:
            raise ValueError("--element_coef_path required for elementwise es-merging")
        cm, cp, cc = load_elementwise_coefficients(element_coef_path)
        return merge_elementwise(mol_sd, prot_sd, cell_sd, cm, cp, cc, device)
    if method == "es":
        if not layer_coef_path or not element_coef_path:
            raise ValueError("Both --layer_coef_path and --element_coef_path required for ES merging")
        layer_coeffs = load_layerwise_coefficients(layer_coef_path)
        cm, cp, cc = load_elementwise_coefficients(element_coef_path)
        return merge_es(mol_sd, prot_sd, cell_sd, cm, cp, cc, layer_coeffs, device)
    raise ValueError(f"Unknown merging method: {method}")
