import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict
from sklearn.preprocessing import StandardScaler


# =========================================================
# Utilities
# =========================================================

def mean_std_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sigma = max(float(np.nanstd(x)), eps)
    return (x - mu) / sigma


def softmax_3class(a: np.ndarray, b: np.ndarray, c: np.ndarray, tau: float = 0.5, eps: float = 1e-12):
    tau = max(float(tau), 1e-8)
    a = np.asarray(a, dtype=float) / tau
    b = np.asarray(b, dtype=float) / tau
    c = np.asarray(c, dtype=float) / tau

    m = np.maximum(np.maximum(a, b), c)
    ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
    denom = ea + eb + ec + eps
    return ea / denom, eb / denom, ec / denom


def normalize_embeddings(X: np.ndarray, Y: np.ndarray):
    scaler = StandardScaler()
    combined = np.vstack([X, Y])
    scaler.fit(combined)
    return scaler.transform(X), scaler.transform(Y)


def compute_sliced_wasserstein_distance(
    X: np.ndarray, Y: np.ndarray,
    num_projections: int = 50, p: float = 2.0, eps: float = 1e-12
) -> float:

    N, d = X.shape
    M, d_Y = Y.shape

    if d != d_Y:
        raise ValueError(f"Dimension mismatch: X has dim {d}, Y has dim {d_Y}")
    if N == 0 or M == 0:
        return 0.0

    n = min(N, M)
    if N > n:
        X = X[np.random.choice(N, n, replace=False)]
    if M > n:
        Y = Y[np.random.choice(M, n, replace=False)]

    directions = np.random.randn(num_projections, d)
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + eps)

    proj_X = np.sort(X @ directions.T, axis=0)
    proj_Y = np.sort(Y @ directions.T, axis=0)

    diff_p = np.power(np.abs(proj_X - proj_Y), p)
    swd = np.power(np.mean(np.mean(diff_p, axis=0)) + eps, 1.0 / p)
    return float(swd)


def load_model_block_from_pkl(pkl_path: str, block_key: str) -> Dict:
    if pkl_path.endswith('.pt'):
        data = torch.load(pkl_path, map_location='cpu')
    else:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{pkl_path} is not a dict. Got type={type(data)}")
    if block_key not in data:
        raise ValueError(f"Block '{block_key}' not found in {pkl_path}. Available: {list(data.keys())}")
    return data[block_key]


def extract_layer_embeddings(model_data: Dict, layer_key: str, token_type: str) -> np.ndarray:
    embeds = []
    key = f"{token_type}_{layer_key}"
    for idx in sorted(model_data.keys()):
        if key not in model_data[idx]:
            continue
        v = model_data[idx][key]
        if isinstance(v, torch.Tensor):
            if v.dtype in (torch.bfloat16, torch.float16):
                v = v.float()
            v = v.detach().cpu().numpy()
        embeds.append(v)
    return np.asarray(embeds)


# =========================================================
# SWD per layer
# =========================================================

def compute_anchor_swd_for_all_layers(
    base_data: Dict, other_data: Dict, token_type: str,
    num_layers: int, num_projections: int = 50, swd_p: float = 2.0,
) -> pd.DataFrame:
    layer_keys = [f"layer_{i}" for i in range(num_layers)]
    rows = []

    for layer in layer_keys:
        try:
            X = extract_layer_embeddings(base_data, layer, token_type)
            Y = extract_layer_embeddings(other_data, layer, token_type)
            Xn, Yn = normalize_embeddings(X, Y)
            swd = compute_sliced_wasserstein_distance(Xn, Yn, num_projections=num_projections, p=swd_p)
            rows.append({"layer": layer, "SWD": float(swd)})
        except Exception as e:
            print(f"[WARN] {token_type} {layer}: {e}")
            rows.append({"layer": layer, "SWD": np.nan})

    df = pd.DataFrame(rows)
    df["Delta_SWD"] = df["SWD"].diff()
    df.loc[df.index[0], "Delta_SWD"] = df.loc[df.index[0], "SWD"]
    df["Delta_SWD"] = df["Delta_SWD"].fillna(0.0)
    return df


def compute_anchor_coefficients(
    df_base_to_mol_molTok, df_base_to_prot_molTok, df_base_to_cell_molTok,
    df_base_to_mol_protTok, df_base_to_prot_protTok, df_base_to_cell_protTok,
    df_base_to_mol_cellTok, df_base_to_prot_cellTok, df_base_to_cell_cellTok,
    alpha: float, temperature: float,
) -> pd.DataFrame:

    m_mol = df_base_to_mol_molTok.set_index("layer")
    p_mol = df_base_to_prot_molTok.set_index("layer")
    c_mol = df_base_to_cell_molTok.set_index("layer")
    m_prot = df_base_to_mol_protTok.set_index("layer")
    p_prot = df_base_to_prot_protTok.set_index("layer")
    c_prot = df_base_to_cell_protTok.set_index("layer")
    m_cell = df_base_to_mol_cellTok.set_index("layer")
    p_cell = df_base_to_prot_cellTok.set_index("layer")
    c_cell = df_base_to_cell_cellTok.set_index("layer")

    all_dfs = [m_mol, p_mol, c_mol, m_prot, p_prot, c_prot, m_cell, p_cell, c_cell]
    common_layers = [ly for ly in m_mol.index if all(ly in df.index for df in all_dfs)]

    # Extract Delta_SWD signals
    s_mol_primary = m_mol.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)
    s_mol_sec_prot = p_mol.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)
    s_mol_sec_cell = c_mol.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)

    s_prot_primary = p_prot.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)
    s_prot_sec_mol = m_prot.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)
    s_prot_sec_cell = c_prot.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)

    s_cell_primary = c_cell.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)
    s_cell_sec_mol = m_cell.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)
    s_cell_sec_prot = p_cell.loc[common_layers, "Delta_SWD"].to_numpy(dtype=float)

    # NaN-safe
    signals = [s_mol_primary, s_mol_sec_prot, s_mol_sec_cell,
               s_prot_primary, s_prot_sec_mol, s_prot_sec_cell,
               s_cell_primary, s_cell_sec_mol, s_cell_sec_prot]
    signals = [np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0) for s in signals]
    (s_mol_primary, s_mol_sec_prot, s_mol_sec_cell,
     s_prot_primary, s_prot_sec_mol, s_prot_sec_cell,
     s_cell_primary, s_cell_sec_mol, s_cell_sec_prot) = signals

    # Normalize excluding lm_head and final_output
    exclude_keys = {"lm_head", "final_output"}
    norm_mask = np.array([ly not in exclude_keys for ly in common_layers])

    def masked_mean_std_norm(x):
        out = np.zeros_like(x, dtype=float)
        if norm_mask.sum() > 0:
            out[norm_mask] = mean_std_norm(x[norm_mask])
        return out

    z_mol_primary = masked_mean_std_norm(s_mol_primary)
    z_mol_sec_prot = masked_mean_std_norm(s_mol_sec_prot)
    z_mol_sec_cell = masked_mean_std_norm(s_mol_sec_cell)
    z_prot_primary = masked_mean_std_norm(s_prot_primary)
    z_prot_sec_mol = masked_mean_std_norm(s_prot_sec_mol)
    z_prot_sec_cell = masked_mean_std_norm(s_prot_sec_cell)
    z_cell_primary = masked_mean_std_norm(s_cell_primary)
    z_cell_sec_mol = masked_mean_std_norm(s_cell_sec_mol)
    z_cell_sec_prot = masked_mean_std_norm(s_cell_sec_prot)

    # Asymmetric mixing
    g_mol = z_mol_primary + alpha * z_prot_sec_mol + alpha * z_cell_sec_mol
    g_prot = z_prot_primary + alpha * z_mol_sec_prot + alpha * z_cell_sec_prot
    g_cell = z_cell_primary + alpha * z_prot_sec_cell + alpha * z_mol_sec_cell

    coef_mol, coef_prot, coef_cell = softmax_3class(g_mol, g_prot, g_cell, tau=temperature)

    return pd.DataFrame({
        "layer": common_layers,
        "s_mol_primary(base->mol|molTok)": s_mol_primary,
        "s_mol_secondary_prot(base->prot|molTok)": s_mol_sec_prot,
        "s_mol_secondary_cell(base->cell|molTok)": s_mol_sec_cell,
        "s_prot_primary(base->prot|protTok)": s_prot_primary,
        "s_prot_secondary_mol(base->mol|protTok)": s_prot_sec_mol,
        "s_prot_secondary_cell(base->cell|protTok)": s_prot_sec_cell,
        "s_cell_primary(base->cell|cellTok)": s_cell_primary,
        "s_cell_secondary_mol(base->mol|cellTok)": s_cell_sec_mol,
        "s_cell_secondary_prot(base->prot|cellTok)": s_cell_sec_prot,
        "z_mol_primary": z_mol_primary,
        "z_mol_secondary_prot": z_mol_sec_prot,
        "z_mol_secondary_cell": z_mol_sec_cell,
        "z_prot_primary": z_prot_primary,
        "z_prot_secondary_mol": z_prot_sec_mol,
        "z_prot_secondary_cell": z_prot_sec_cell,
        "z_cell_primary": z_cell_primary,
        "z_cell_secondary_mol": z_cell_sec_mol,
        "z_cell_secondary_prot": z_cell_sec_prot,
        "g_mol": g_mol, "g_prot": g_prot, "g_cell": g_cell,
        "coef_mol": coef_mol, "coef_prot": coef_prot, "coef_cell": coef_cell,
        "alpha": float(alpha), "temperature": float(temperature),
    })


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser("Anchor Sliced Wasserstein Distance - 3-way coefficient computation")

    parser.add_argument("--pkl_path", default="./Embedding/embedding.pkl")
    parser.add_argument("--output_base", default="./coefficients/Layer_Wise")
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--swd_num_projections", type=int, default=1024)
    parser.add_argument("--swd_p", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.output_base, exist_ok=True)

    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"PKL not found: {args.pkl_path}")

    base_data = load_model_block_from_pkl(args.pkl_path, "withoutlora")
    mol_data = load_model_block_from_pkl(args.pkl_path, "mol_only")
    prot_data = load_model_block_from_pkl(args.pkl_path, "prot_only")
    cell_data = load_model_block_from_pkl(args.pkl_path, "cell_only")
    print(f"Loaded blocks: base={len(base_data)}, mol={len(mol_data)}, prot={len(prot_data)}, cell={len(cell_data)} samples")

    out_dir = os.path.join(args.output_base, f"SWD_anchor_{args.alpha}", f"{args.swd_num_projections}_{args.swd_p}_{args.temperature}")
    os.makedirs(out_dir, exist_ok=True)

    # Compute SWD(base, X) for all token type x model combinations
    swd_params = dict(num_layers=args.num_layers,
                      num_projections=args.swd_num_projections, swd_p=args.swd_p)

    token_model_combos = [
        ("mol", mol_data, "mol_token_M_vs_Base_swd.csv"),
        ("mol", prot_data, "mol_token_P_vs_Base_swd.csv"),
        ("mol", cell_data, "mol_token_C_vs_Base_swd.csv"),
        ("prot", mol_data, "prot_token_M_vs_Base_swd.csv"),
        ("prot", prot_data, "prot_token_P_vs_Base_swd.csv"),
        ("prot", cell_data, "prot_token_C_vs_Base_swd.csv"),
        ("cell", mol_data, "cell_token_M_vs_Base_swd.csv"),
        ("cell", prot_data, "cell_token_P_vs_Base_swd.csv"),
        ("cell", cell_data, "cell_token_C_vs_Base_swd.csv"),
    ]

    swd_results = {}
    for token_type, other_data, filename in token_model_combos:
        print(f"  Computing SWD: {filename.replace('_swd.csv', '')}")
        df = compute_anchor_swd_for_all_layers(base_data, other_data, token_type=token_type, **swd_params)
        df.to_csv(os.path.join(out_dir, filename), index=False, float_format="%.8f")
        swd_results[filename] = df

    # Compute coefficients
    coef_df = compute_anchor_coefficients(
        df_base_to_mol_molTok=swd_results["mol_token_M_vs_Base_swd.csv"],
        df_base_to_prot_molTok=swd_results["mol_token_P_vs_Base_swd.csv"],
        df_base_to_cell_molTok=swd_results["mol_token_C_vs_Base_swd.csv"],
        df_base_to_mol_protTok=swd_results["prot_token_M_vs_Base_swd.csv"],
        df_base_to_prot_protTok=swd_results["prot_token_P_vs_Base_swd.csv"],
        df_base_to_cell_protTok=swd_results["prot_token_C_vs_Base_swd.csv"],
        df_base_to_mol_cellTok=swd_results["cell_token_M_vs_Base_swd.csv"],
        df_base_to_prot_cellTok=swd_results["cell_token_P_vs_Base_swd.csv"],
        df_base_to_cell_cellTok=swd_results["cell_token_C_vs_Base_swd.csv"],
        alpha=args.alpha,
        temperature=args.temperature,
    )
    coef_df["num_projections"] = args.swd_num_projections
    coef_df["swd_p"] = args.swd_p

    coef_path = os.path.join(out_dir, "layerwise_merging_coefficients.csv")
    coef_df.to_csv(coef_path, index=False, float_format="%.8f")

    print(f"Saved 9 SWD CSVs + coefficients to {out_dir}")


if __name__ == "__main__":
    main()