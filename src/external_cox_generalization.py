#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from tqdm import tqdm
from scipy.stats import spearmanr
from lifelines.utils import concordance_index

from src.MIL import AttentionMIL
from src.Autoencoder import Autoencoder


# -------------------------
# utils
# -------------------------
def _to_dense(x):
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            return x.toarray()
    except Exception:
        pass
    return np.asarray(x)


def load_autoencoder(path, input_dim, latent_dim, device):
    ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(path, map_location=device))
    ae.eval()
    ae.to(device)
    return ae


def load_mil(path, latent_dim, hidden_dim, dropout,
             aggregator, topk, tau, device):

    model = AttentionMIL(
        input_dim=latent_dim,
        num_classes=2,
        hidden_dim=hidden_dim,
        dropout=dropout,
        sample_source_dim=None,
        aggregator=aggregator,
        topk=topk if topk > 0 else None,
        tau=tau if tau > 0 else None,
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def predict_risk(X_patient, ae, mil_models, device):
    x = torch.from_numpy(X_patient.astype(np.float32)).to(device)

    if hasattr(ae, "encode"):
        z = ae.encode(x)
    else:
        z = ae(x)

    risks = []
    for m in mil_models:
        out = m([z])
        r = out["risk"].view(-1).item()
        risks.append(r)

    return float(np.median(risks))


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--patient_col", default="patient_id")

    ap.add_argument("--ae_path", required=True)
    ap.add_argument("--ae_latent_dim", type=int, required=True)

    ap.add_argument("--mil_paths", required=True)
    ap.add_argument("--mil_hidden_dim", type=int, default=128)
    ap.add_argument("--mil_dropout", type=float, default=0.25)
    ap.add_argument("--aggregator", default="attention")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0)

    ap.add_argument("--filter_studies", default=None,
                    help="comma-separated study names")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.input_h5ad)

    if args.filter_studies is not None:
        keep = [x.strip() for x in args.filter_studies.split(",")]
        adata = adata[adata.obs["sample_source"].isin(keep)].copy()
        print("Filtered studies:", keep)

    X = _to_dense(adata.X).astype(np.float32)
    X = (X - 0.5) * 2.0
    adata.X = X

    input_dim = adata.shape[1]

    ae = load_autoencoder(args.ae_path, input_dim,
                          args.ae_latent_dim, device)

    mil_paths = [p.strip() for p in args.mil_paths.split(",")]
    mil_models = [
        load_mil(p, args.ae_latent_dim,
                 args.mil_hidden_dim,
                 args.mil_dropout,
                 args.aggregator,
                 args.topk,
                 args.tau,
                 device)
        for p in mil_paths
    ]

    patients = adata.obs[args.patient_col].unique().tolist()

    rows = []
    for pid in tqdm(patients):
        ad_p = adata[adata.obs[args.patient_col] == pid]
        X_p = _to_dense(ad_p.X)

        risk = predict_risk(X_p, ae, mil_models, device)

        row = {
            "patient": pid,
            "risk": risk,
            "time_to_relapse_days": ad_p.obs["time_to_relapse_days"].iloc[0],
            "follow_up_duration_months": ad_p.obs["follow_up_duration_months"].iloc[0],
            "relapse_y_n": ad_p.obs["relapse_y_n"].iloc[0],
            "study": ad_p.obs["sample_source"].iloc[0],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # survival time
    df["event"] = (df["relapse_y_n"] == "Yes").astype(int)
    df["time"] = np.where(
        df["event"] == 1,
        df["time_to_relapse_days"],
        df["follow_up_duration_months"] * 30
    )

    # Spearman
    rho, p = spearmanr(df["risk"], df["time"])
    print("Spearman rho:", rho, "p:", p)

    # C-index
    cidx = concordance_index(
        df["time"],
        -df["risk"],   # higher risk = shorter survival
        df["event"]
    )
    print("C-index:", cidx)

    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()