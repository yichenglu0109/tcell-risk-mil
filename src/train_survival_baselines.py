#!/usr/bin/env python3
import os
import argparse
import pickle
from datetime import datetime

import numpy as np
import scanpy as sc
from sklearn.model_selection import StratifiedKFold

from src.Dataset import build_patient_survival_table, DAYS_PER_MONTH


def _to_dense(x):
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            return x.toarray()
    except Exception:
        pass
    return np.asarray(x)


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


def pool_patient_features(X, mode="mean"):
    if mode == "mean":
        return np.mean(X, axis=0)
    if mode == "median":
        return np.median(X, axis=0)
    if mode == "q90":
        return np.quantile(X, q=0.9, axis=0)
    raise ValueError(f"Unknown --pool mode: {mode}")


def build_patient_matrix(adata, patient_col, patients, pool="mean"):
    pid_arr = adata.obs[patient_col].astype(str).to_numpy()
    X_rows = []
    for pid in patients:
        idx = np.where(pid_arr == str(pid))[0]
        if idx.size == 0:
            raise RuntimeError(f"Patient not found in adata.obs[{patient_col}]: {pid}")
        Xp = _to_dense(adata.X[idx]).astype(np.float32, copy=False)
        X_rows.append(pool_patient_features(Xp, mode=pool))
    return np.asarray(X_rows, dtype=np.float32)


def make_splits(n_patients, y_event, cv="kfold", k=5, seed=1):
    if cv == "loocv":
        return [(
            np.array([j for j in range(n_patients) if j != i], dtype=int),
            np.array([i], dtype=int)
        ) for i in range(n_patients)]
    if cv == "kfold":
        skf = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=seed)
        return list(skf.split(np.arange(n_patients), y_event))
    raise ValueError("--cv must be 'loocv' or 'kfold'")


def fit_predict_coxph(X_train, time_train, event_train, X_test):
    try:
        from lifelines import CoxPHFitter
    except Exception as e:
        raise ImportError(
            "lifelines is required for method=coxph. Install with: pip install lifelines"
        ) from e

    import pandas as pd

    feat_cols = [f"x{i}" for i in range(X_train.shape[1])]
    df_train = pd.DataFrame(X_train, columns=feat_cols)
    df_train["time"] = np.asarray(time_train, float)
    df_train["event"] = np.asarray(event_train, int)

    df_test = pd.DataFrame(X_test, columns=feat_cols)

    model = CoxPHFitter(penalizer=1e-3)
    model.fit(df_train, duration_col="time", event_col="event", show_progress=False)
    risk = model.predict_partial_hazard(df_test).to_numpy(dtype=float)
    return risk


def fit_predict_rsf(X_train, time_train, event_train, X_test, seed=1, n_estimators=300):
    try:
        from sksurv.ensemble import RandomSurvivalForest
    except Exception as e:
        raise ImportError(
            "scikit-survival is required for method=rsf. Install with: pip install scikit-survival"
        ) from e

    y_train = np.array(
        list(zip(np.asarray(event_train, dtype=bool), np.asarray(time_train, dtype=float))),
        dtype=[("event", "?"), ("time", "<f8")]
    )

    model = RandomSurvivalForest(
        n_estimators=int(n_estimators),
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=int(seed),
    )
    model.fit(X_train, y_train)
    risk = model.predict(X_test).astype(float)
    return risk


def run_one_method(
    method,
    X_pat,
    time_days,
    event,
    patients,
    cv="kfold",
    k=5,
    seed=1,
    rsf_n_estimators=300,
):
    n = len(patients)
    splits = make_splits(n, event, cv=cv, k=k, seed=seed)
    risks = np.full(n, np.nan, dtype=float)
    fold_cis = []

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        X_train = X_pat[tr_idx]
        X_test = X_pat[te_idx]
        t_train = time_days[tr_idx]
        e_train = event[tr_idx]

        if method == "coxph":
            pred = fit_predict_coxph(X_train, t_train, e_train, X_test)
        elif method == "rsf":
            pred = fit_predict_rsf(
                X_train, t_train, e_train, X_test,
                seed=seed + fold,
                n_estimators=rsf_n_estimators,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        risks[te_idx] = pred.reshape(-1)
        fold_ci = c_index(time_days[te_idx], event[te_idx], risks[te_idx])
        fold_cis.append(float(fold_ci))
        print(f"[{method}][fold {fold}/{len(splits)}] val_n={len(te_idx)} c-index={fold_ci:.4f}")

    if not np.isfinite(risks).all():
        bad = np.where(~np.isfinite(risks))[0]
        raise RuntimeError(f"[{method}] missing OOF prediction for patient idx={bad.tolist()}")

    overall_ci = c_index(time_days, event, risks)
    return {
        "risk": risks.tolist(),
        "c_index": float(overall_ci),
        "fold_c_index": [float(x) for x in fold_cis],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--outdir", default="results_survival_baselines")

    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--relapse_col", default="relapse_y_n")
    ap.add_argument("--ttr_col", default="time_to_relapse_days")
    ap.add_argument("--fu_col", default="follow_up_duration_months")
    ap.add_argument("--days_per_month", type=float, default=DAYS_PER_MONTH)
    ap.add_argument("--drop_inconsistent", action="store_true")

    ap.add_argument("--methods", default="coxph,rsf",
                    help="Comma-separated baseline methods: coxph,rsf")
    ap.add_argument("--pool", default="mean", choices=["mean", "median", "q90"])

    ap.add_argument("--cv", default="kfold", choices=["loocv", "kfold"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--rsf_n_estimators", type=int, default=300)
    args = ap.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    valid_methods = {"coxph", "rsf"}
    bad = [m for m in methods if m not in valid_methods]
    if bad:
        raise ValueError(f"Unsupported method(s): {bad}. Allowed: coxph,rsf")
    if len(methods) == 0:
        raise ValueError("Empty --methods")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(args.outdir, f"run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    patients, time_days, event = build_patient_survival_table(
        adata.obs,
        patient_col=args.patient_col,
        relapse_col=args.relapse_col,
        ttr_col=args.ttr_col,
        fu_col=args.fu_col,
        days_per_month=args.days_per_month,
        drop_inconsistent=args.drop_inconsistent,
    )
    patients = np.asarray(patients, dtype=object)
    time_days = np.asarray(time_days, dtype=float)
    event = np.asarray(event, dtype=int)

    print(f"[INFO] patients={len(patients)} events={int(event.sum())} censored={int((event==0).sum())}")
    print(f"[INFO] methods={methods} pool={args.pool} cv={args.cv}")

    X_pat = build_patient_matrix(
        adata=adata,
        patient_col=args.patient_col,
        patients=patients,
        pool=args.pool,
    )
    print(f"[INFO] patient feature matrix shape: {X_pat.shape}")

    all_results = {
        "patients": patients.tolist(),
        "time_days": time_days.tolist(),
        "event": event.tolist(),
        "args": vars(args),
        "result_dir": result_dir,
        "methods": {},
    }

    for method in methods:
        print("\n" + "=" * 80)
        print(f"METHOD: {method}")
        print("=" * 80)
        out = run_one_method(
            method=method,
            X_pat=X_pat,
            time_days=time_days,
            event=event,
            patients=patients,
            cv=args.cv,
            k=args.k,
            seed=args.seed,
            rsf_n_estimators=args.rsf_n_estimators,
        )
        all_results["methods"][method] = out

        print(f"[RESULT][{method}] C-index={out['c_index']:.4f}")
        if args.cv == "kfold":
            m = np.nanmean(out["fold_c_index"])
            s = np.nanstd(out["fold_c_index"])
            print(f"[RESULT][{method}] fold mean±std={m:.4f} ± {s:.4f}")

        out_csv = os.path.join(result_dir, f"{method}_risk_{args.cv if args.cv=='loocv' else f'kfold{args.k}'}.csv")
        with open(out_csv, "w") as f:
            f.write("patient_id,time_days,event,risk\n")
            for pid, t, e, r in zip(patients, time_days, event, out["risk"]):
                f.write(f"{pid},{t},{e},{r}\n")
        print(f"[INFO] saved: {out_csv}")

    out_pkl = os.path.join(result_dir, "survival_baselines_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(all_results, f)
    print(f"[INFO] saved: {out_pkl}")
    print(f"[INFO] done: {result_dir}")


if __name__ == "__main__":
    main()
