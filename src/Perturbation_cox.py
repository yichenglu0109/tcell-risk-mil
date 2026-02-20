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

from src.MIL import AttentionMIL
from src.Autoencoder import Autoencoder  # <= 如果你 repo 裡 class 名稱不同，改這行

# -------------------------
# Perturbation 
# -------------------------
def perturb_tf_activity(adata, tf_name: str, direction: str = "up", mad_mult: float = 3.0):
    """
    Perturb one TF feature by +/- mad_mult * MAD, then clip to [-1, 1].
    Assumes adata.X is already scaled to [-1, 1].
    """
    if tf_name not in adata.var_names:
        raise KeyError(f"TF '{tf_name}' not in adata.var_names")

    pert = adata.copy()
    tf_idx = adata.var_names.get_loc(tf_name)

    x = np.asarray(adata.X[:, tf_idx]).reshape(-1)
    med = np.median(x)
    mad = median_abs_deviation(x)

    if direction == "up":
        x_new = np.clip(x + mad_mult * mad, -1.0, 1.0)
    elif direction == "down":
        x_new = np.clip(x - mad_mult * mad, -1.0, 1.0)
    else:
        raise ValueError("direction must be 'up' or 'down'")

    # write back
    pert.X[:, tf_idx] = x_new
    return pert


# -------------------------
# Loaders
# -------------------------
def load_autoencoder(ae_path: str, input_dim: int, latent_dim: int, device):
    ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    ae.to(device)
    return ae


def load_cox_mil(model_path: str, input_dim: int, hidden_dim: int, dropout: float,
                 aggregator: str, topk: int, tau: float, device):
    _topk = int(topk) if (topk is not None and int(topk) > 0) else None
    _tau  = float(tau) if (tau is not None and float(tau) > 0) else None

    model = AttentionMIL(
        input_dim=input_dim,
        num_classes=2,          # 你 Cox 用的 AttentionMIL 仍然是 num_classes=2 的架構，但取 out["risk"]
        hidden_dim=hidden_dim,
        dropout=dropout,
        sample_source_dim=None,
        aggregator=aggregator,
        topk=_topk,
        tau=_tau,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model


@torch.no_grad()
def predict_risk_for_patient(adata_patient, ae, mil_model, device):
    """
    adata_patient: AnnData of ONE patient (cells x features) in input space [-1,1]
    return: scalar risk
    """
    x = np.asarray(adata_patient.X, dtype=np.float32)
    x_t = torch.from_numpy(x).to(device)

    # AE encode -> latent
    z = ae.encode(x_t)              # [n_cells, latent_dim]
    out = mil_model([z])            # bag = [latent]
    r = out["risk"].view(-1).item() # scalar
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
    ap.add_argument("--tfs", default=None,
                    help="Comma-separated TF list. If omitted, perturb ALL var_names (can be huge).")
    ap.add_argument("--directions", default="up,down")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.input_h5ad)
    if not args.assume_scaled:
        adata.X = (adata.X - 0.5) * 2  # Kristin-style scaling

    input_dim = adata.X.shape[1]
    ae = load_autoencoder(args.ae_path, input_dim=input_dim, latent_dim=args.ae_latent_dim, device=device)

    mil_paths = [p.strip() for p in args.mil_paths.split(",") if p.strip()]
    mil_models = [
        load_cox_mil(p, input_dim=args.ae_latent_dim, hidden_dim=args.mil_hidden_dim,
                     dropout=args.mil_dropout, aggregator=args.aggregator,
                     topk=args.topk, tau=args.tau, device=device)
        for p in mil_paths
    ]

    patients = adata.obs[args.patient_col].astype(str).unique().tolist()

    if args.tfs is None:
        tfs = adata.var_names.tolist()
    else:
        tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]

    rows = []
    for pid in tqdm(patients, desc="patients"):
        ad_p = adata[adata.obs[args.patient_col].astype(str) == pid].copy()

        # baseline risks (per model)
        base_rs = [predict_risk_for_patient(ad_p, ae, m, device) for m in mil_models]
        base_med = float(np.median(base_rs))

        for tf in tfs:
            for direction in directions:
                ad_pert = perturb_tf_activity(ad_p, tf, direction=direction, mad_mult=args.mad_mult)

                pert_rs = [predict_risk_for_patient(ad_pert, ae, m, device) for m in mil_models]
                pert_med = float(np.median(pert_rs))

                rows.append({
                    "patient": pid,
                    "tf": tf,
                    "direction": direction,
                    "baseline_risk_median": base_med,
                    "perturbed_risk_median": pert_med,
                    "delta_risk_median": pert_med - base_med,
                    "baseline_risk_all": json.dumps(base_rs),
                    "perturbed_risk_all": json.dumps(pert_rs),
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] saved: {args.out_csv}")


if __name__ == "__main__":
    main()