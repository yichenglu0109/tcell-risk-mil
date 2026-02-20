#!/usr/bin/env python3
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from tqdm import tqdm
from scipy.stats import median_abs_deviation
from typing import Optional, List, Dict

from src.MIL import AttentionMIL
from src.Autoencoder import Autoencoder  # 如果你 repo 裡 class 名稱不同，改這行


# -------------------------
# Utils
# -------------------------
def _to_dense(x):
    """anndata.X could be np.ndarray or scipy sparse; return dense np.ndarray."""
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            return x.toarray()
    except Exception:
        pass
    return np.asarray(x)


def parse_tfs_arg(tfs_arg: Optional[str], adata) -> List[str]:
    """
    --tfs:
      - None => ALL var_names
      - 'all' => ALL var_names
      - 'TF1,TF2' => those (must exist)
      - '@/path/to/list.txt' => one TF per line (must exist)
    """
    if tfs_arg is None:
        return list(map(str, adata.var_names.tolist()))

    s = str(tfs_arg).strip()
    if s.lower() == "all":
        return list(map(str, adata.var_names.tolist()))

    if s.startswith("@"):
        path = s[1:]
        if not os.path.exists(path):
            raise FileNotFoundError(f"TF list file not found: {path}")
        with open(path, "r") as f:
            tfs = [line.strip() for line in f if line.strip() and (not line.strip().startswith("#"))]
    else:
        tfs = [x.strip() for x in s.split(",") if x.strip()]

    # validate exists
    missing = [tf for tf in tfs if tf not in adata.var_names]
    if len(missing) > 0:
        raise KeyError(f"{len(missing)} TFs not found in adata.var_names. Examples: {missing[:10]}")
    return tfs


def parse_directions_arg(dirs_arg: str) -> List[str]:
    dirs = [d.strip().lower() for d in str(dirs_arg).split(",") if d.strip()]
    ok = {"up", "down"}
    bad = [d for d in dirs if d not in ok]
    if bad:
        raise ValueError(f"Invalid directions: {bad}. Allowed: up,down")
    return dirs


# -------------------------
# Perturbation
# -------------------------
def perturb_tf_activity_inplace(X: np.ndarray, tf_idx: int, direction: str, mad_mult: float) -> None:
    """
    X: dense array [n_cells, n_features] for ONE patient (will be copied by caller if needed)
    Perturb one TF feature by +/- mad_mult * MAD, then clip to [-1, 1].
    Assumes X already scaled to [-1,1].
    """
    x = X[:, tf_idx].reshape(-1)
    mad = median_abs_deviation(x, scale=1.0, nan_policy="omit")
    if (not np.isfinite(mad)) or mad <= 0:
        # if MAD is 0 (constant), perturb does nothing
        return

    if direction == "up":
        x_new = np.clip(x + mad_mult * mad, -1.0, 1.0)
    elif direction == "down":
        x_new = np.clip(x - mad_mult * mad, -1.0, 1.0)
    else:
        raise ValueError("direction must be 'up' or 'down'")

    X[:, tf_idx] = x_new


# -------------------------
# Loaders
# -------------------------
def load_autoencoder(ae_path: str, input_dim: int, latent_dim: int, device):
    ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    state = torch.load(ae_path, map_location=device)
    ae.load_state_dict(state, strict=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    ae.to(device)
    return ae


def load_cox_mil(model_path: str, input_dim: int, hidden_dim: int, dropout: float,
                 aggregator: str, topk: int, tau: float, device):
    _topk = int(topk) if (topk is not None and int(topk) > 0) else None
    _tau = float(tau) if (tau is not None and float(tau) > 0) else None

    model = AttentionMIL(
        input_dim=input_dim,
        num_classes=2,           # Cox 用 AttentionMIL 架構，但 forward 取 out["risk"]
        hidden_dim=hidden_dim,
        dropout=dropout,
        sample_source_dim=None,
        aggregator=aggregator,
        topk=_topk,
        tau=_tau,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model


@torch.no_grad()
def predict_risk_for_patient_X(X_patient: np.ndarray, ae, mil_model, device) -> float:
    """
    X_patient: dense np.ndarray [n_cells, n_features] in input space (scaled to [-1,1])
    return: scalar risk
    """
    x_t = torch.from_numpy(X_patient.astype(np.float32, copy=False)).to(device)

    # AE encode -> latent
    # Kristin 的 notebook/你的 AE code 通常會有 encode()；如果沒有，fallback 用 forward()
    if hasattr(ae, "encode"):
        z = ae.encode(x_t)
    else:
        out = ae(x_t)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            z = out[1]
        else:
            z = out

    out_mil = mil_model([z])
    r = out_mil["risk"].view(-1).item()
    return float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--patient_col", default="patient_id")

    # scaling
    ap.add_argument("--assume_scaled", action="store_true",
                    help="If set, assume adata.X already in [-1,1]. Otherwise apply (X-0.5)*2.")

    # AE
    ap.add_argument("--ae_path", required=True)
    ap.add_argument("--ae_latent_dim", type=int, required=True)

    # MIL models: provide multiple paths for fold-ensemble (comma-separated)
    ap.add_argument("--mil_paths", required=True,
                    help="Comma-separated list of best_model.pth (e.g., fold_01,...,fold_05 or full)")
    ap.add_argument("--mil_hidden_dim", type=int, default=128)
    ap.add_argument("--mil_dropout", type=float, default=0.25)
    ap.add_argument("--aggregator", default="attention", choices=["attention", "mean", "q90"])
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0)

    # perturb config
    ap.add_argument("--mad_mult", type=float, default=3.0)
    ap.add_argument("--tfs", default="all",
                    help="TF spec: 'all' | 'TF1,TF2' | '@/path/to/tfs.txt'. Default: all")
    ap.add_argument("--directions", default="up,down")

    # output size control
    ap.add_argument("--no_store_all", action="store_true",
                    help="If set, do NOT store baseline_risk_all / perturbed_risk_all (smaller CSV).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    adata = sc.read_h5ad(args.input_h5ad)

    # scale if needed
    if not args.assume_scaled:
        X = _to_dense(adata.X).astype(np.float32, copy=False)
        X = (X - 0.5) * 2.0
        adata.X = X

    input_dim = int(adata.X.shape[1])
    print(f"[INFO] n_cells={adata.n_obs} input_dim={input_dim}")
    print(f"[INFO] patient_col={args.patient_col}")

    # guard: pseudobulk aggregators ignore topk/tau
    if args.aggregator in ("mean", "q90"):
        if args.topk != 0 or float(args.tau) != 0.0:
            print("[WARN] mean/q90 ignores topk/tau; forcing topk=0 tau=0.0")
        args.topk = 0
        args.tau = 0.0

    # load AE
    ae = load_autoencoder(args.ae_path, input_dim=input_dim, latent_dim=args.ae_latent_dim, device=device)

    # load MIL models (fold ensemble)
    mil_paths = [p.strip() for p in str(args.mil_paths).split(",") if p.strip()]
    if len(mil_paths) == 0:
        raise ValueError("Empty --mil_paths")

    mil_models = [
        load_cox_mil(p, input_dim=args.ae_latent_dim, hidden_dim=args.mil_hidden_dim,
                     dropout=args.mil_dropout, aggregator=args.aggregator,
                     topk=args.topk, tau=args.tau, device=device)
        for p in mil_paths
    ]
    print(f"[INFO] loaded MIL models: {len(mil_models)}")

    # patients
    patients = adata.obs[args.patient_col].astype(str).unique().tolist()
    print(f"[INFO] patients={len(patients)}")

    # TFs + directions
    tfs = parse_tfs_arg(args.tfs, adata)
    directions = parse_directions_arg(args.directions)
    print(f"[INFO] TFs={len(tfs)} directions={directions} mad_mult={args.mad_mult}")

    # precompute tf indices for speed
    tf2idx: Dict[str, int] = {tf: int(adata.var_names.get_loc(tf)) for tf in tfs}

    rows = []
    for pid in tqdm(patients, desc="patients"):
        ad_p = adata[adata.obs[args.patient_col].astype(str) == pid]
        X_base = _to_dense(ad_p.X).astype(np.float32, copy=True)  # [n_cells, n_features]

        # baseline risks (per model)
        base_rs = [predict_risk_for_patient_X(X_base, ae, m, device) for m in mil_models]
        base_med = float(np.median(base_rs))

        for tf in tfs:
            tf_idx = tf2idx[tf]
            for direction in directions:
                X_pert = X_base.copy()
                perturb_tf_activity_inplace(X_pert, tf_idx=tf_idx, direction=direction, mad_mult=float(args.mad_mult))

                pert_rs = [predict_risk_for_patient_X(X_pert, ae, m, device) for m in mil_models]
                pert_med = float(np.median(pert_rs))

                row = {
                    "patient": pid,
                    "tf": tf,
                    "direction": direction,
                    "baseline_risk_median": base_med,
                    "perturbed_risk_median": pert_med,
                    "delta_risk_median": pert_med - base_med,
                }
                if not args.no_store_all:
                    row["baseline_risk_all"] = json.dumps(base_rs)
                    row["perturbed_risk_all"] = json.dumps(pert_rs)

                rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] saved: {args.out_csv}")
    print(f"[INFO] rows={len(df)} cols={list(df.columns)}")


if __name__ == "__main__":
    main()