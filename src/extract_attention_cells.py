#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import scanpy as sc

from src.MIL import AttentionMIL
from src.Autoencoder import Autoencoder

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
    ae.load_state_dict(torch.load(path, map_location=device), strict=True)
    ae.eval().to(device)
    for p in ae.parameters():
        p.requires_grad = False
    return ae

def load_mil(path, latent_dim, hidden_dim, dropout, aggregator, topk, tau, device):
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
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def parse_mil_paths(mil_path, mil_paths):
    paths = []
    if mil_path is not None and str(mil_path).strip():
        paths.append(str(mil_path).strip())
    if mil_paths is not None and str(mil_paths).strip():
        paths.extend([p.strip() for p in str(mil_paths).split(",") if p.strip()])
    # unique while preserving order
    out = []
    seen = set()
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    if len(out) == 0:
        raise ValueError("Please provide --mil_path or --mil_paths.")
    return out


@torch.no_grad()
def forward_with_attention(adata_p, ae, mil_models, device, assume_scaled=False, attn_reduce="median"):
    X = _to_dense(adata_p.X).astype(np.float32)
    if not assume_scaled:
        X = (X - 0.5) * 2.0
    x = torch.from_numpy(X).to(device)

    # encode
    if hasattr(ae, "encode"):
        z = ae.encode(x)
    else:
        z = ae(x)

    risks = []
    attns = []
    for mil in mil_models:
        out = mil([z], return_attention=True)
        risks.append(float(out["risk"].view(-1).item()))

        attn_list = out.get("attn", None)
        if (attn_list is None) or (len(attn_list) == 0) or (attn_list[0] is None):
            raise RuntimeError(
                "No attention returned. Make sure aggregator=attention, topk=0, and model supports return_attention."
            )
        w = attn_list[0].detach().cpu().numpy().reshape(-1)  # [n_cells]
        attns.append(w)

    if len(attns) == 1:
        w_ens = attns[0]
    else:
        A = np.vstack(attns)  # [n_models, n_cells]
        if attn_reduce == "mean":
            w_ens = A.mean(axis=0)
        else:
            w_ens = np.median(A, axis=0)

    risk_ens = float(np.median(np.asarray(risks, dtype=float)))
    return risk_ens, w_ens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--ae_path", required=True)
    ap.add_argument("--ae_latent_dim", type=int, required=True)

    ap.add_argument("--mil_path", default=None, help="single MIL checkpoint path")
    ap.add_argument("--mil_paths", default=None, help="comma-separated MIL checkpoint paths (ensemble)")
    ap.add_argument("--mil_hidden_dim", type=int, default=128)
    ap.add_argument("--mil_dropout", type=float, default=0.25)
    ap.add_argument("--aggregator", default="attention")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--attn_reduce", default="median", choices=["median", "mean"],
                    help="How to combine per-model cell attention when using --mil_paths.")

    ap.add_argument("--patient_col", default="patient_id_std")
    ap.add_argument("--pid", default=None, help="single patient id to extract; if not set, run all patients")
    ap.add_argument("--study", default=None, help="optional study filter (sample_source)")
    ap.add_argument("--cell_id_col", default="cell_raw", help="which obs column to output as cell id")
    ap.add_argument("--topn", type=int, default=50,
                    help="Top-N cells per patient. Use <=0 to keep all cells.")
    ap.add_argument("--assume_scaled", action="store_true")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.input_h5ad)

    if args.study is not None:
        if "sample_source" not in adata.obs.columns:
            raise ValueError(
                "--study was provided, but 'sample_source' column is missing in adata.obs. "
                "Either remove --study or provide the correct study column in data."
            )
        adata = adata[adata.obs["sample_source"].astype(str) == str(args.study)].copy()

    input_dim = int(adata.shape[1])
    ae = load_autoencoder(args.ae_path, input_dim=input_dim, latent_dim=args.ae_latent_dim, device=device)
    mil_ckpts = parse_mil_paths(args.mil_path, args.mil_paths)
    if len(mil_ckpts) > 1 and int(args.topk) > 0:
        raise ValueError("Ensemble attention requires --topk 0 so attention vectors align to original cells.")
    mil_models = [
        load_mil(p, latent_dim=args.ae_latent_dim,
                 hidden_dim=args.mil_hidden_dim, dropout=args.mil_dropout,
                 aggregator=args.aggregator, topk=args.topk, tau=args.tau, device=device)
        for p in mil_ckpts
    ]
    print(f"[INFO] loaded MIL models: {len(mil_models)}")

    if args.pid is not None:
        pids = [str(args.pid)]
    else:
        pids = adata.obs[args.patient_col].astype(str).unique().tolist()
    if len(pids) == 0:
        raise ValueError(f"No patients found in column: {args.patient_col}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    total_rows = 0
    written_patients = 0
    first_write = True

    for pid in pids:
        mask = (adata.obs[args.patient_col].astype(str) == str(pid)).to_numpy()
        adata_p = adata[mask]
        if adata_p.n_obs == 0:
            continue

        risk, w = forward_with_attention(
            adata_p, ae, mil_models, device,
            assume_scaled=args.assume_scaled, attn_reduce=args.attn_reduce
        )

        if args.cell_id_col in adata_p.obs.columns:
            cell_ids = adata_p.obs[args.cell_id_col].astype(str).to_numpy()
        else:
            cell_ids = adata_p.obs_names.astype(str).to_numpy()

        study_val = str(adata_p.obs["sample_source"].astype(str).iloc[0]) if "sample_source" in adata_p.obs.columns else ""
        df_p = pd.DataFrame({
            "patient": str(pid),
            "cell_id": cell_ids,
            "attn": w,
            "study": study_val,
            "risk": risk,
        }).sort_values("attn", ascending=False)
        df_p["rank"] = np.arange(1, len(df_p) + 1)

        if int(args.topn) > 0:
            df_p = df_p.head(int(args.topn)).copy()
        df_p.to_csv(args.out_csv, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False
        total_rows += len(df_p)
        written_patients += 1
        print(f"[INFO] pid={pid} risk={risk:.6f} n_cells={adata_p.n_obs} kept={len(df_p)}")

    if written_patients == 0:
        raise ValueError("No rows to write. Check --patient_col/--pid/--study filters.")
    print(f"[INFO] wrote rows={total_rows} patients={written_patients}: {args.out_csv}")

if __name__ == "__main__":
    main()
