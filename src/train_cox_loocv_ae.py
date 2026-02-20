#!/usr/bin/env python3
import os
import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import scanpy as sc

from src.MIL import AttentionMIL
from src.Dataset import PatientBagSurvivalDataset, collate_survival, DAYS_PER_MONTH

# === NEW: AE imports (match train.py) ===
from src.Dataset import preprocess_data
from src.Autoencoder import train_autoencoder, evaluate_autoencoder


class CoxPHLoss(nn.Module):
    """Negative partial log-likelihood (Breslow ties)."""
    def __init__(self):
        super().__init__()

    def forward(self, risk, time, event):
        # risk: [B] (higher = higher hazard)
        # time: [B]
        # event: [B] float 0/1

        # no-event guard (keeps graph)
        if (event > 0).sum() == 0:
            return risk.sum() * 0.0

        device = risk.device

        # sort by time descending
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        time = time[order]
        event = event[order]

        uniq_times = torch.unique(time)
        nll = torch.tensor(0.0, device=device)

        for t in uniq_times:
            ix = (time == t)
            d = event[ix].sum()
            if d <= 0:
                continue

            sum_risk_events = risk[ix][event[ix] > 0].sum()
            at_risk = (time >= t)
            log_denom = torch.logsumexp(risk[at_risk], dim=0)
            nll = nll - (sum_risk_events - d * log_denom)

        return nll


@torch.no_grad()
def predict_risk(model, bag, device):
    model.eval()
    b = bag.to(device)
    out = model([b])
    r = out["risk"].squeeze(0)  # scalar
    return float(r.item())


def c_index(time, event, risk):
    """Simple concordance index (higher risk -> earlier event)."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)

    n = 0
    conc = 0.0
    for i in range(len(time)):
        for j in range(len(time)):
            if i == j:
                continue
            if event[i] == 1 and time[i] < time[j]:
                n += 1
                if risk[i] > risk[j]:
                    conc += 1
                elif risk[i] == risk[j]:
                    conc += 0.5
    return float(conc / n) if n > 0 else float("nan")


def train_one_fold(
    train_ds,
    input_dim,
    hidden_dim,
    dropout,
    device,
    epochs=80,
    lr=5e-4,
    weight_decay=1e-2,
    patience=10,
    seed=1,
    aggregator="attention",
    topk=0,
    tau=0.0,
    fold_save_path=None,   # NEW: where to save best_model.pth
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # NOTE: keep your original behavior: full-batch on patient-bags
    loader = DataLoader(
        train_ds,
        batch_size=len(train_ds),
        shuffle=False,
        collate_fn=collate_survival
    )

    _topk = int(topk) if (topk is not None and int(topk) > 0) else None
    _tau  = float(tau) if (tau is not None and float(tau) > 0) else None

    model = AttentionMIL(
        input_dim=input_dim,
        num_classes=2,
        hidden_dim=hidden_dim,
        dropout=dropout,
        sample_source_dim=None,
        aggregator=aggregator,
        topk=_topk,
        tau=_tau,
    ).to(device)

    loss_fn = CoxPHLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")
    best_state = None
    no_imp = 0

    for ep in range(epochs):
        model.train()
        running = 0.0
        steps = 0

        for bags, times, events, _pids in loader:
            bags = [b.to(device) for b in bags]
            out = model(bags)
            risk = out["risk"].view(-1)

            times = times.to(device)
            events = events.to(device)

            loss = loss_fn(risk, times, events)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            steps += 1

        epoch_loss = running / max(steps, 1)
        print(f"[train] ep={ep:03d} loss={epoch_loss:.6f} best={best:.6f} no_imp={no_imp}")

        if epoch_loss < best - 1e-8:
            best = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0

            # NEW: write checkpoint immediately (like train.py fold/best_model.pth)
            if fold_save_path is not None:
                os.makedirs(fold_save_path, exist_ok=True)
                torch.save(best_state, os.path.join(fold_save_path, "best_model.pth"))
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--outdir", default="results_cox")

    # survival columns
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--relapse_col", default="relapse_y_n")
    ap.add_argument("--ttr_col", default="time_to_relapse_days")
    ap.add_argument("--fu_col", default="follow_up_duration_months")
    ap.add_argument("--days_per_month", type=float, default=DAYS_PER_MONTH)
    ap.add_argument("--drop_inconsistent", action="store_true")

    # MIL params
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--aggregator", default="attention", choices=["attention", "mean", "q90"])
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0)

    # CV
    ap.add_argument("--cv", default="kfold", choices=["loocv", "kfold"])
    ap.add_argument("--k", type=int, default=5)

    # === NEW: optional AE stage (like train.py) ===
    ap.add_argument("--use_ae", action="store_true", help="If set, train AE first and run Cox MIL on AE latent.")
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--ae_epochs", type=int, default=200)

    # === NEW: final full-data model for perturbation ===
    ap.add_argument("--train_full_model", action="store_true",
                    help="If set, train a final model on ALL patients and save mil/full/best_model.pth for perturbation.")

    args = ap.parse_args()

    # guard: mean/q90 ignores topk/tau
    if args.aggregator in ("mean", "q90"):
        if args.topk != 0 or args.tau != 0.0:
            print("[WARN] mean/q90 ignores topk/tau; forcing topk=0, tau=0.0")
        args.topk = 0
        args.tau = 0.0

    # Create output directories like train.py
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.outdir, f"run_{timestamp}")
    ae_dir = os.path.join(result_dir, "autoencoder")
    mil_dir = os.path.join(result_dir, "mil")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ae_dir, exist_ok=True)
    os.makedirs(mil_dir, exist_ok=True)

    # Load data
    adata = sc.read_h5ad(args.input_h5ad)
    input_dim_raw = adata.X.shape[1]
    print(f"[INFO] n_cells={adata.n_obs} | input_dim={input_dim_raw}")
    print(f"[INFO] obs cols: {list(adata.obs.columns)}")

    # === Optional AE stage ===
    if args.use_ae:
        print("\n" + "="*80)
        print("STEP 1: TRAIN AUTOENCODER (optional)")
        print("="*80)

        # preprocess_data should match your train.py behavior
        train_loader, val_loader, test_loader, input_dim = preprocess_data(adata)

        ae_model, train_losses, val_losses = train_autoencoder(
            train_loader, val_loader, input_dim, args.latent_dim, args.ae_epochs, save_path=ae_dir
        )

        # evaluate_autoencoder should write latent + maybe best_autoencoder already; we also save explicitly
        adata_latent, test_loss = evaluate_autoencoder(
            ae_model, test_loader, adata, adata.var_names.tolist(), save_path=ae_dir
        )

        # Save latent representations (same name as train.py)
        latent_file = os.path.join(ae_dir, "latent_representation.h5ad")
        adata_latent.write(latent_file)
        print(f"[INFO] saved: {latent_file}")

        # Save AE weights explicitly (stable name for your pipeline)
        ae_pth = os.path.join(ae_dir, "best_autoencoder.pth")
        torch.save(ae_model.state_dict(), ae_pth)
        print(f"[INFO] saved: {ae_pth}")

        # Use latent adata for downstream Cox MIL
        adata_use = adata_latent
        input_dim_mil = args.latent_dim
    else:
        # no AE: use raw features
        adata_use = adata
        input_dim_mil = input_dim_raw

    # Build full survival dataset (patient list + labels)
    full_ds = PatientBagSurvivalDataset(
        adata_use,
        patient_col=args.patient_col,
        relapse_col=args.relapse_col,
        ttr_col=args.ttr_col,
        fu_col=args.fu_col,
        days_per_month=args.days_per_month,
        drop_inconsistent=args.drop_inconsistent,
    )
    patients = np.array(full_ds.patient_list, dtype=object)
    time_days = np.array(full_ds.time_days, dtype=float)
    event = np.array(full_ds.event, dtype=int)
    print(f"[INFO] patients kept: {len(patients)} | events={event.sum()} | censored={(event==0).sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # ===== CV =====
    if args.cv == "loocv":
        risks = []
        fold_losses = []

        for i, test_pid in enumerate(patients, start=1):
            print(f"\n[LOOCV] {i}/{len(patients)} test_patient={test_pid}")

            train_pids = [p for p in patients if p != test_pid]

            train_adata = adata_use[adata_use.obs[args.patient_col].astype(str).isin(list(map(str, train_pids)))].copy()
            test_adata  = adata_use[adata_use.obs[args.patient_col].astype(str) == str(test_pid)].copy()

            train_ds = PatientBagSurvivalDataset(
                train_adata,
                patient_col=args.patient_col,
                relapse_col=args.relapse_col,
                ttr_col=args.ttr_col,
                fu_col=args.fu_col,
                days_per_month=args.days_per_month,
                drop_inconsistent=args.drop_inconsistent,
            )

            fold_save_path = os.path.join(mil_dir, f"fold_{i:02d}")
            model, best_loss = train_one_fold(
                train_ds=train_ds,
                input_dim=input_dim_mil,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                seed=args.seed + i,
                aggregator=args.aggregator,
                topk=args.topk,
                tau=args.tau,
                fold_save_path=fold_save_path,
            )
            fold_losses.append(best_loss)

            test_ds = PatientBagSurvivalDataset(
                test_adata,
                patient_col=args.patient_col,
                relapse_col=args.relapse_col,
                ttr_col=args.ttr_col,
                fu_col=args.fu_col,
                days_per_month=args.days_per_month,
                drop_inconsistent=args.drop_inconsistent,
            )
            bag, t, e, pid = test_ds[0]
            r = predict_risk(model, bag, device)
            risks.append(r)

        risks = np.array(risks, float)

    else:
        k = int(args.k)
        risks = np.full(len(patients), np.nan, dtype=float)
        fold_losses = []
        fold_cis = []
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(patients, event), start=1):
            train_pids = patients[tr_idx].tolist()
            val_pids   = patients[va_idx].tolist()

            print(f"\n[{k}-fold] fold={fold}/{k} train={len(train_pids)} val={len(val_pids)} val_events={event[va_idx].sum()}")

            train_adata = adata_use[adata_use.obs[args.patient_col].astype(str).isin(list(map(str, train_pids)))].copy()

            train_ds = PatientBagSurvivalDataset(
                train_adata,
                patient_col=args.patient_col,
                relapse_col=args.relapse_col,
                ttr_col=args.ttr_col,
                fu_col=args.fu_col,
                days_per_month=args.days_per_month,
                drop_inconsistent=args.drop_inconsistent,
            )

            fold_save_path = os.path.join(mil_dir, f"fold_{fold:02d}")
            model, best_loss = train_one_fold(
                train_ds=train_ds,
                input_dim=input_dim_mil,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                seed=args.seed + fold,
                aggregator=args.aggregator,
                topk=args.topk,
                tau=args.tau,
                fold_save_path=fold_save_path,
            )
            fold_losses.append(best_loss)

            # out-of-fold risks for val
            for pid in val_pids:
                test_adata = adata_use[adata_use.obs[args.patient_col].astype(str) == str(pid)].copy()

                test_ds = PatientBagSurvivalDataset(
                    test_adata,
                    patient_col=args.patient_col,
                    relapse_col=args.relapse_col,
                    ttr_col=args.ttr_col,
                    fu_col=args.fu_col,
                    days_per_month=args.days_per_month,
                    drop_inconsistent=args.drop_inconsistent,
                )
                bag, t, e, _ = test_ds[0]
                r = predict_risk(model, bag, device)

                idx = np.where(patients == pid)[0][0]
                risks[idx] = r

            # ===== NEW: per-fold C-index on THIS fold's validation set =====
            fold_risk = risks[va_idx]
            val_events = int(event[va_idx].sum())
            fold_ci = c_index(time_days[va_idx], event[va_idx], fold_risk)
            fold_cis.append(float(fold_ci))
            print(f"[{k}-fold] fold={fold} val_events={val_events} C-index={fold_ci:.4f}")

        if not np.isfinite(risks).all():
            bad = np.where(~np.isfinite(risks))[0]
            raise RuntimeError(f"Some patients missing out-of-fold predictions: idx={bad.tolist()}, pids={patients[bad].tolist()}")

    # ===== Evaluation summary =====
    ci = c_index(time_days, event, risks)

    print("\n===== Cox CV Results =====")
    print(f"CV={args.cv} k={args.k if args.cv=='kfold' else 'NA'}")
    print(f"C-index (simple): {ci:.4f}")
    if args.cv == "kfold":
        print("Per-fold C-index:", [round(x, 4) for x in fold_cis])
        print(f"Per-fold mean±std: {np.nanmean(fold_cis):.4f} ± {np.nanstd(fold_cis):.4f}")
    print("==========================\n")

    # ===== Optional: train final full-data model for perturbation =====
    full_model_loss = None
    if args.train_full_model:
        print("\n" + "="*80)
        print("STEP: TRAIN FINAL FULL-DATA MODEL (for perturbation)")
        print("="*80)

        full_save_path = os.path.join(mil_dir, "full")
        model_full, best_loss_full = train_one_fold(
            train_ds=full_ds,
            input_dim=input_dim_mil,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            seed=args.seed + 999,
            aggregator=args.aggregator,
            topk=args.topk,
            tau=args.tau,
            fold_save_path=full_save_path,
        )
        full_model_loss = float(best_loss_full)
        print(f"[INFO] saved final model: {os.path.join(full_save_path, 'best_model.pth')} (loss={best_loss_full:.6f})")

    print("\n===== Cox Overall OOF Results =====")
    ci = c_index(time_days, event, risks)
    print(f"C-index (simple): {ci:.4f}")
    print("=============================\n")

    # ===== Save outputs (match your current pattern) =====
    results = {
        "patients": patients.tolist(),
        "time_days": time_days.tolist(),
        "event": event.tolist(),
        "risk": risks.tolist(),                 # OOF risks
        "c_index": float(ci),
        "fold_c_index": [float(x) for x in fold_cis] if args.cv == "kfold" else None,
        "fold_best_loss": [float(x) for x in fold_losses],
        "full_model_best_loss": full_model_loss,
        "use_ae": bool(args.use_ae),
        "latent_dim": int(args.latent_dim) if args.use_ae else None,
        "args": vars(args),
        "result_dir": result_dir,
        "ae_dir": ae_dir if args.use_ae else None,
        "mil_dir": mil_dir,
    }

    tag = "loocv" if args.cv == "loocv" else f"kfold{args.k}"
    out_pkl = os.path.join(result_dir, f"cox_{tag}_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"[INFO] saved: {out_pkl}")

    out_csv = os.path.join(result_dir, f"cox_risk_{tag}.csv")
    with open(out_csv, "w") as f:
        f.write("patient_id,time_days,event,risk\n")
        for pid, t, e, r in zip(patients, time_days, event, risks):
            f.write(f"{pid},{t},{e},{r}\n")
    print(f"[INFO] saved: {out_csv}")

    print(f"\n[INFO] Done. Result dir: {result_dir}\n")


if __name__ == "__main__":
    main()