import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader

import scanpy as sc

from src.Dataset import PatientBagDataset
from src.MIL import AttentionMIL


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Simple rankdata implementation (average ranks for ties).
    Returns ranks starting at 1.
    """
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))

    a_sorted = a[sorter]
    ranks = np.zeros(len(a), dtype=float)

    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        # average rank for ties, ranks are 1-indexed
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    return ranks[inv]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    # Pearson on ranks
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = (np.sqrt((vx**2).sum()) * np.sqrt((vy**2).sum()))
    if denom == 0:
        return float("nan")
    return float((vx * vy).sum() / denom)

def _normalize_yesno_to_01(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    yes = {"YES", "Y", "1", "TRUE", "T"}
    no  = {"NO", "N", "0", "FALSE", "F"}
    na  = {"NA", "N/A", "NONE", "NULL", "NAN", ""}

    if s in yes:
        return 1.0
    if s in no:
        return 0.0
    if s in na:
        return np.nan
    return np.nan


def filter_relapse_only(adata, patient_col, relapse_col, time_col, relapse_positive_value=1):
    """
    Keep only patients with relapse == relapse_positive_value AND non-missing time_col.
    Filter at patient-level then subset cells accordingly.
    """
    obs = adata.obs.copy()

    for c in (patient_col, relapse_col, time_col):
        if c not in obs.columns:
            raise ValueError(f"Missing column in adata.obs: {c}")

    tmp = obs[[patient_col, relapse_col, time_col]].copy()
    tmp[patient_col] = tmp[patient_col].astype(str)

    def first_non_null(s):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) > 0 else np.nan

    byp = tmp.groupby(patient_col, observed=True).agg(
        {relapse_col: first_non_null, time_col: first_non_null}
    )

    # keep a raw copy for debugging
    byp_raw = byp.copy()

    byp[relapse_col] = byp[relapse_col].apply(_normalize_yesno_to_01)
    byp[time_col] = pd.to_numeric(byp[time_col], errors="coerce")

    pos = float(relapse_positive_value)

    kept_patients = byp.index[
        (byp[relapse_col] == pos) & (byp[time_col].notna())
    ].astype(str)

    if len(kept_patients) == 0:
        print("[DEBUG] relapse_col raw (patient-level):", byp_raw[relapse_col].value_counts(dropna=False).to_dict())
        print("[DEBUG] relapse_col normalized (patient-level):", byp[relapse_col].value_counts(dropna=False).to_dict())
        print("[DEBUG] time_col non-null count:", int(byp[time_col].notna().sum()), "/", int(byp.shape[0]))
        raise ValueError("After filtering relapse-only patients, nothing remains.")

    adata_sub = adata[adata.obs[patient_col].astype(str).isin(kept_patients)].copy()
    return adata_sub, kept_patients.tolist()


def train_one_fold(
    train_adata,
    input_dim,
    hidden_dim,
    dropout,
    sample_source_dim,
    device,
    num_epochs=100,
    lr=5e-4,
    weight_decay=1e-2,
    patience=10,
    use_sample_source=True,
    seed=1,
    patient_col="patient_id",
    time_col="time_to_relapse",
    sample_source_col="Sample_source",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = PatientBagDataset(
        train_adata,
        patient_col=patient_col,
        task_type="regression",
        label_col=time_col,
        drop_missing=True,
        use_sample_source=use_sample_source,
        sample_source_col=sample_source_col,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    model = AttentionMIL(
        input_dim=input_dim,
        num_classes=2,  # not used for regression, but constructor needs it
        hidden_dim=hidden_dim,
        dropout=dropout,
        sample_source_dim=sample_source_dim if use_sample_source else None,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        for batch in train_loader:
            if len(batch) == 4:
                bag, label, _patient, one_hot = batch
                one_hot = one_hot.to(device)
            else:
                bag, label, _patient = batch
                one_hot = None

            bag = bag.to(device)
            if bag.dim() == 3:            # [1, n_cells, latent_dim]
                bag = bag.squeeze(0)      # -> [n_cells, latent_dim]

            y = label.to(device).float()
            if y.dim() > 0:
                y = y.view(-1)[0]         # -> scalar tensor

            out = model([bag], sample_source=one_hot.unsqueeze(0) if one_hot is not None else None)
            pred = out["risk"].view(-1)[0]   # -> scalar
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item())

        epoch_loss = running / max(len(train_loader), 1)

        # early stop on train loss (no val set in LOOCV)
        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model


@torch.no_grad()
def predict_one_patient(model, test_adata, device, use_sample_source=True, return_attention=True, 
                        patient_col="patient_id", time_col="time_to_relapse", sample_source_col="Sample_source"):
    test_ds = PatientBagDataset(
        test_adata,
        patient_col=patient_col,
        task_type="regression",
        label_col=time_col,
        drop_missing=True,
        use_sample_source=use_sample_source,
        sample_source_col=sample_source_col,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model.eval()

    for batch in test_loader:
        if len(batch) == 4:
            bag, label, patient, one_hot = batch
            one_hot = one_hot.to(device)
        else:
            bag, label, patient = batch
            one_hot = None

        bag = bag.to(device)
        y = float(label.item())

        # MIL forward expects list of bags
        out = model(
            [bag],
            sample_source=one_hot.unsqueeze(0) if one_hot is not None else None,
            return_attention=return_attention
        )
        bag = bag.to(device)
        if bag.dim() == 3:
            bag = bag.squeeze(0)

        out = model([bag], return_attention=return_attention)
        pred = float(out["risk"].view(-1)[0].item())

        attn = None
        if return_attention:
            # out["attn"] is a list length B; each element is [num_instances, 1]
            # Here B=1
            attn = out["attn"][0].detach().cpu().numpy()

        pid = patient[0] if isinstance(patient, (list, tuple)) else str(patient)
        return pid, y, pred, attn

    raise RuntimeError("Empty test_loader (unexpected).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True, help="Path to latent_representation.h5ad (cell x latent_dim)")
    ap.add_argument("--outdir", default="results_time_regression", help="Output directory")
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--relapse_col", default="relapse")
    ap.add_argument("--time_col", default="time_to_relapse")
    ap.add_argument("--relapse_positive_value", type=int, default=1)
    ap.add_argument("--use_sample_source", action="store_true", help="Use one-hot Sample_source covariate")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    input_dim = adata.X.shape[1]

    # relapse-only filter
    adata_r, kept_patients = filter_relapse_only(
        adata,
        patient_col=args.patient_col,
        relapse_col=args.relapse_col,
        time_col=args.time_col,
        relapse_positive_value=args.relapse_positive_value,
    )
    print(f"[INFO] relapse-only patients kept: {len(kept_patients)}")

    # Determine sample_source_dim from dataset (if using)
    tmp_ds = PatientBagDataset(
        adata_r,
        patient_col=args.patient_col,
        task_type="regression",
        label_col=args.time_col,
        drop_missing=True,
        use_sample_source=args.use_sample_source,
        sample_source_col="Sample_source",
    )
    sample_source_dim = tmp_ds.sample_source_dim
    if args.use_sample_source and sample_source_dim is None:
        raise ValueError("use_sample_source=True but Sample_source missing or could not be encoded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] input_dim (latent_dim): {input_dim}")
    print(f"[INFO] use_sample_source: {args.use_sample_source} | sample_source_dim: {sample_source_dim}")

    cv_results = {
        "fold_metrics": [],
        "patient_predictions": {},
        "attention_weights": {},
        "overall_metrics": {}
    }

    # LOOCV over relapse patients
    patients = np.array(sorted(set(adata_r.obs[args.patient_col].astype(str).tolist())))
    y_true = []
    y_pred = []
    pid_list = []

    for i, test_pid in enumerate(patients, start=1):
        print(f"[LOOCV] {i}/{len(patients)} test_patient={test_pid}")

        train_adata = adata_r[adata_r.obs[args.patient_col].astype(str) != str(test_pid)].copy()
        test_adata  = adata_r[adata_r.obs[args.patient_col].astype(str) == str(test_pid)].copy()

        model = train_one_fold(
            train_adata=train_adata,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            sample_source_dim=sample_source_dim,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            use_sample_source=args.use_sample_source,
            seed=args.seed,
            patient_col=args.patient_col,
            time_col=args.time_col,
            sample_source_col="Sample_source",
        )

        pid, yt, yp, attn = predict_one_patient(
            model,
            test_adata,
            device,
            use_sample_source=args.use_sample_source,
            return_attention=True,
            patient_col=args.patient_col,
            time_col=args.time_col,
            sample_source_col="Sample_source",
        )

        y_true.append(yt)
        y_pred.append(yp)
        pid_list.append(pid)

        # store patient-level prediction (like train.py)
        cv_results["patient_predictions"][pid] = {
            "true_time": float(yt),
            "pred_time": float(yp),
            "abs_error": float(abs(yp - yt)),
            "sq_error": float((yp - yt) ** 2),
        }

        # store attention weights (optional but keeps parity with train.py)
        if attn is not None:
            cv_results["attention_weights"][pid] = attn  # numpy array [n_cells, 1]

        # fold_metrics (one patient = one fold in LOOCV)
        cv_results["fold_metrics"].append({
            "fold": int(i - 1) if isinstance(i, int) else int(i),
            "patient_id": pid,
            "true_time": float(yt),
            "pred_time": float(yp),
            "abs_error": float(abs(yp - yt)),
            "sq_error": float((yp - yt) ** 2),
        })


    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    spr = spearman_corr(y_true, y_pred)

    print("\n===== Time Regression (relapse-only) Results =====")
    print(f"MAE (days):  {mae:.3f}")
    print(f"RMSE (days): {rmse:.3f}")
    print(f"Spearman:    {spr:.3f}")

    cv_results["overall_metrics"] = {
        "mae_days": float(mae),
        "rmse_days": float(rmse),
        "spearman": float(spr),
        "n_patients": int(len(pid_list)),
        "use_sample_source": bool(args.use_sample_source),
    }

    # Save pkl (same style as train.py)
    out_pkl = os.path.join(args.outdir, "loocv_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"[INFO] saved: {out_pkl}")

    # Save predictions
    out_csv = os.path.join(args.outdir, "predictions.csv")
    with open(out_csv, "w") as f:
        f.write("patient_id,true_time_days,pred_time_days\n")
        for pid, yt, yp in zip(pid_list, y_true, y_pred):
            f.write(f"{pid},{yt},{yp}\n")
    print(f"[INFO] saved: {out_csv}")

    # Save summary
    out_txt = os.path.join(args.outdir, "summary.txt")
    with open(out_txt, "w") as f:
        f.write(f"MAE_days\t{mae}\n")
        f.write(f"RMSE_days\t{rmse}\n")
        f.write(f"Spearman\t{spr}\n")
        f.write(f"N_patients\t{len(pid_list)}\n")
        f.write(f"use_sample_source\t{args.use_sample_source}\n")
    print(f"[INFO] saved: {out_txt}")


if __name__ == "__main__":
    main()
