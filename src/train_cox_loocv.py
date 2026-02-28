#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold 
# from lifelines.utils import concordance_index
import scanpy as sc

from src.MIL import AttentionMIL
from src.Dataset import PatientBagSurvivalDataset, collate_survival, DAYS_PER_MONTH

class CoxPHLoss(nn.Module):
    """
    Negative partial log-likelihood (Breslow ties).
    """
    def __init__(self):
        super().__init__()


    def forward(self, risk, time, event):
        # risk: [B] (higher = higher hazard)
        # time: [B]
        # event: [B] float 0/1

        # if no events, return 0 loss (since no ordering constraints)
        if (event > 0).sum() == 0:
            # no events present, return 0 loss (since no ordering constraints)
            return risk.sum() * 0.0
        # ===================================
        device = risk.device

        # sort by time descending
        order = torch.argsort(time, descending=True)
        risk = risk[order]
        time = time[order]
        event = event[order]

        # log cumulative sum exp for risk set (since sorted desc, risk set = prefix)
        # but Breslow for ties needs grouping; implement explicitly
        uniq_times = torch.unique(time)
        nll = torch.tensor(0.0, device=device)

        for t in uniq_times:
            ix = (time == t)
            d = event[ix].sum()  # number of events at this time
            if d <= 0:
                continue
            # event risk sum
            sum_risk_events = risk[ix][event[ix] > 0].sum()

            # risk set: time >= t (since sorted desc)
            at_risk = (time >= t)
            log_denom = torch.logsumexp(risk[at_risk], dim=0)

            nll = nll - (sum_risk_events - d * log_denom)

        return nll


@torch.no_grad()
def predict_risk(model, bag, device, aggregator="attention"):
    model.eval()
    b = bag.to(device)
    out = model([b])

    # AttentionMIL already returns out["risk"] in your codebase
    r = out["risk"].squeeze(0)  # scalar
    return float(r.item())


def c_index(time, event, risk):
    """
    Basic concordance index (higher risk -> earlier event).
    Only comparable pairs where one had event and occurred earlier than the other time.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)

    n = 0
    conc = 0.0
    for i in range(len(time)):
        for j in range(len(time)):
            if i == j:
                continue
            # i is event, and i happened before j's time
            if event[i] == 1 and time[i] < time[j]:
                n += 1
                if risk[i] > risk[j]:
                    conc += 1
                elif risk[i] == risk[j]:
                    conc += 0.5
    return float(conc / n) if n > 0 else float("nan")


def train_one_fold(train_ds, input_dim, hidden_dim, dropout, device,
                   epochs=80, lr=5e-4, weight_decay=1e-2, batch_size=8,
                   patience=10, seed=1,
                   aggregator="attention", topk=0, tau=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=collate_survival)

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
            # forward (bags is list)
            bags = [b.to(device) for b in bags]
            out = model(bags)
            risk = out["risk"].view(-1)  # [B]

            times = times.to(device)
            events = events.to(device)

            loss = loss_fn(risk, times, events)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            steps += 1

        epoch_loss = running / max(steps, 1)

        # logging
        print(f"[train] ep={ep:03d} loss={epoch_loss:.4f} best={best:.4f} no_imp={no_imp}")

        if epoch_loss < best - 1e-8:
            best = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--outdir", default="results_cox")
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--relapse_col", default="relapse_y_n")
    ap.add_argument("--ttr_col", default="time_to_relapse_days")
    ap.add_argument("--fu_col", default="follow_up_duration_months")
    ap.add_argument("--days_per_month", type=float, default=DAYS_PER_MONTH)
    ap.add_argument("--drop_inconsistent", action="store_true")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--aggregator", default="attention", choices=["attention", "mean", "q90"])
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--cv", default="kfold", choices=["loocv", "kfold"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # Check and adjust arguments for specific aggregators
    if args.aggregator in ("mean", "q90"):
        if args.topk != 0 or args.tau != 0.0:
            print("[WARN] mean/q90 ignores topk/tau; forcing topk=0, tau=0.0")
        args.topk = 0
        args.tau = 0.0
    # ===================================
    os.makedirs(args.outdir, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    input_dim = adata.X.shape[1]
    print(f"[INFO] n_cells={adata.n_obs} | input_dim={input_dim}")
    print(f"[INFO] obs cols: {list(adata.obs.columns)}")

    full_ds = PatientBagSurvivalDataset(
        adata,
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
        # LOOCV: train on N-1, predict risk for held-out
        risks = []
        for i, test_pid in enumerate(patients, start=1):
            print(f"[LOOCV] {i}/{len(patients)} test_patient={test_pid}")

            train_pids = [p for p in patients if p != test_pid]
            train_adata = adata[adata.obs[args.patient_col].astype(str).isin(train_pids)].copy()
            test_adata  = adata[adata.obs[args.patient_col].astype(str) == str(test_pid)].copy()

            train_ds = PatientBagSurvivalDataset(
                train_adata,
                patient_col=args.patient_col,
                relapse_col=args.relapse_col,
                ttr_col=args.ttr_col,
                fu_col=args.fu_col,
                days_per_month=args.days_per_month,
                drop_inconsistent=args.drop_inconsistent,
            )
            model = train_one_fold(
                train_ds=train_ds,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed,
                aggregator=args.aggregator,
                topk=args.topk,
                tau=args.tau,
            )

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
        # k-fold (patient-level, stratified by event)
        k = int(args.k)
        risks = np.full(len(patients), np.nan, dtype=float)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(patients, event), start=1):
            train_pids = patients[tr_idx].tolist()
            val_pids   = patients[va_idx].tolist()

            print(f"[{k}-fold] fold={fold}/{k} train={len(train_pids)} val={len(val_pids)} val_events={event[va_idx].sum()}")

            train_adata = adata[adata.obs[args.patient_col].astype(str).isin(train_pids)].copy()

            train_ds = PatientBagSurvivalDataset(
                train_adata,
                patient_col=args.patient_col,
                relapse_col=args.relapse_col,
                ttr_col=args.ttr_col,
                fu_col=args.fu_col,
                days_per_month=args.days_per_month,
                drop_inconsistent=args.drop_inconsistent,
            )
            model = train_one_fold(
                train_ds=train_ds,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed + fold,  # different seed per fold
                aggregator=args.aggregator,
                topk=args.topk,
                tau=args.tau,
            )

            # predict risks for all val patients (out-of-fold)
            for pid in val_pids:
                test_adata = adata[adata.obs[args.patient_col].astype(str) == str(pid)].copy()

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

        if not np.isfinite(risks).all():
            bad = np.where(~np.isfinite(risks))[0]
            raise RuntimeError(f"Some patients missing out-of-fold predictions: idx={bad.tolist()}, pids={patients[bad].tolist()}")
    # =================

    ci = c_index(time_days, event, risks)
    ci_flip = c_index(time_days, event, -risks)

    # ci_ll = concordance_index(time_days, risks, event)
    # ci_ll_flip = concordance_index(time_days, -risks, event)

    print("\n===== Cox LOOCV Results =====")
    print(f"C-index (simple): {ci:.4f}")
    print(f"C-index (simple, flipped): {ci_flip:.4f}")
    # print(f"C-index (lifelines): {ci_ll:.4f}")
    # print(f"C-index (lifelines, flipped): {ci_ll_flip:.4f}")
    print("=============================\n")
    
    results = {
        "patients": patients.tolist(),
        "time_days": time_days.tolist(),
        "event": event.tolist(),
        "risk": risks.tolist(),
        "c_index": float(ci),
        "args": vars(args),
    }

    tag = "loocv" if args.cv == "loocv" else f"kfold{args.k}"
    out_pkl = os.path.join(args.outdir, f"cox_{tag}_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"[INFO] saved: {out_pkl}")

    out_csv = os.path.join(args.outdir, f"cox_risk_{tag}.csv")
    with open(out_csv, "w") as f:
        f.write("patient_id,time_days,event,risk\n")
        for pid, t, e, r in zip(patients, time_days, event, risks):
            f.write(f"{pid},{t},{e},{r}\n")
    print(f"[INFO] saved: {out_csv}")


if __name__ == "__main__":
    main()
