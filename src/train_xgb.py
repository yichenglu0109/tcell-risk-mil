#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)

from xgboost import XGBClassifier

from src.Dataset import PatientBagDataset


def mean_pool(bag: np.ndarray) -> np.ndarray:
    # bag: [n_cells, d]
    return bag.mean(axis=0)


def safe_metric(metric_fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return metric_fn(y_true, y_score)


def run_xgb_loocv(
    h5ad: str,
    label_col: str,
    patient_col: str = "patient_id",
    label_map: dict | None = None,
    normalize: bool = True,
    use_sample_source: bool = False,
    sample_source_col: str = "Sample_source",
    seed: int = 42,
    n_estimators: int = 500,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float | None = None,   # fixed value if provided
    auto_scale_pos_weight: bool = False,     # per-fold n_neg/n_pos
    n_jobs: int = -1,
    out_csv: str | None = "xgb_loocv_predictions.csv",
    out_json: str | None = None,
):
    # ---- load ----
    adata = sc.read_h5ad(h5ad)

    # ---- normalize ----
    if normalize:
        adata.X = (adata.X - 0.5) * 2

    # ---- dataset (patient bags) ----
    ds = PatientBagDataset(
        adata=adata,
        patient_col=patient_col,
        task_type="classification",
        label_col=label_col,
        label_map=label_map,
        drop_missing=True,
        use_sample_source=use_sample_source,
        sample_source_col=sample_source_col,
    )

    # ---- build patient-level X, y ----
    X_list, y_list, pid_list = [], [], []
    for i in range(len(ds)):
        out = ds[i]
        if len(out) == 4:
            bag_t, label_t, pid, one_hot_t = out
            one_hot = one_hot_t.numpy().astype(np.float32)
        else:
            bag_t, label_t, pid = out
            one_hot = None

        bag = bag_t.numpy().astype(np.float32)  # [n_cells, d]
        x = mean_pool(bag)                      # [d]
        if one_hot is not None:
            x = np.concatenate([x, one_hot], axis=0)

        X_list.append(x)
        y_list.append(int(label_t.item()))
        pid_list.append(str(pid))

    X = np.stack(X_list, axis=0)  # [N_patients, d(+cov)]
    y = np.array(y_list, dtype=int)
    pids = np.array(pid_list, dtype=object)

    print(f"[INFO] Patients used: {len(pids)}")
    print(f"[INFO] Feature dim: {X.shape[1]}")
    print(f"[INFO] Class counts: {pd.Series(y).value_counts().to_dict()}")
    print(f"[INFO] Normalization: {'(X-0.5)*2' if normalize else 'OFF'}")

    # ---- LOOCV ----
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y), dtype=float)
    y_pred = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        uniq = np.unique(y_tr)
        if len(uniq) < 2:
            only = int(uniq[0])
            p1 = 1.0 if only == 1 else 0.0
            y_prob[test_idx[0]] = float(p1)
            y_pred[test_idx[0]] = int(p1 >= 0.5)
            continue

        # ---- scale_pos_weight handling ----
        spw = None
        if scale_pos_weight is not None:
            spw = float(scale_pos_weight)
        elif auto_scale_pos_weight:
            n_pos = float((y_tr == 1).sum())
            n_neg = float((y_tr == 0).sum())
            spw = (n_neg / n_pos) if n_pos > 0 else 1.0

        # ---- build params (compat across xgboost versions) ----
        xgb_params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            min_child_weight=float(min_child_weight),
            gamma=float(gamma),
            reg_lambda=float(reg_lambda),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(seed),
            n_jobs=int(n_jobs),
            scale_pos_weight=float(spw) if spw is not None else 1.0,
        )

        # Some older xgboost versions support use_label_encoder; new ones may not.
        try:
            xgb = XGBClassifier(**xgb_params, use_label_encoder=False)
        except TypeError:
            xgb = XGBClassifier(**xgb_params)

        xgb.fit(X_tr, y_tr)

        prob = xgb.predict_proba(X_te)[0]
        cls_to_col = {c: j for j, c in enumerate(xgb.classes_)}
        p1 = prob[cls_to_col.get(1, int(np.argmax(prob)))]

        y_prob[test_idx[0]] = float(p1)
        y_pred[test_idx[0]] = int(p1 >= 0.5)

    # ---- metrics ----
    acc = accuracy_score(y, y_pred)
    n_correct = int((y_pred == y).sum())
    n_total = int(len(y))

    overall_auc = safe_metric(roc_auc_score, y, y_prob)
    overall_pr_auc = safe_metric(average_precision_score, y, y_prob)

    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", pos_label=1, zero_division=0
    )

    cm = confusion_matrix(y, y_pred)

    prec_by_class, rec_by_class, f1_by_class, support_by_class = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    # ---- print ----
    print("\n===== LOOCV Final Results (XGBoost) =====")
    print(f"Overall Accuracy: {acc:.4f} ({n_correct}/{n_total} patients correct)")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}" if np.isfinite(overall_auc) else "Overall AUC: NA")
    print(f"Overall PR-AUC: {overall_pr_auc:.4f}" if np.isfinite(overall_pr_auc) else "Overall PR-AUC: NA")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClass-specific Metrics:")
    print(f"class_0: Precision={prec_by_class[0]:.4f}, Recall={rec_by_class[0]:.4f}, Count={int(support_by_class[0])}")
    print(f"class_1: Precision={prec_by_class[1]:.4f}, Recall={rec_by_class[1]:.4f}, Count={int(support_by_class[1])}")

    # ---- predictions df ----
    out_pred = pd.DataFrame({
        "patient_id": pids,
        "y_true": y,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }).sort_values("patient_id")

    if out_csv is not None:
        out_pred.to_csv(out_csv, index=False)
        print(f"\n[INFO] Saved predictions to: {out_csv}")

    metrics = {
        "patients_used": int(len(pids)),
        "feature_dim": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
        "normalization": "(X-0.5)*2" if normalize else "OFF",
        "accuracy": float(acc),
        "n_correct": int(n_correct),
        "n_total": int(n_total),
        "precision_pos": float(overall_precision),
        "recall_pos": float(overall_recall),
        "f1_pos": float(overall_f1),
        "auroc": None if not np.isfinite(overall_auc) else float(overall_auc),
        "pr_auc": None if not np.isfinite(overall_pr_auc) else float(overall_pr_auc),
        "confusion_matrix": cm.tolist(),
        "xgb_params": {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "min_child_weight": float(min_child_weight),
            "gamma": float(gamma),
            "reg_lambda": float(reg_lambda),
            "scale_pos_weight": float(scale_pos_weight) if scale_pos_weight is not None else (
                "auto_per_fold(n_neg/n_pos)" if auto_scale_pos_weight else "default(1.0)"
            ),
        },
        "class_0": {
            "precision": float(prec_by_class[0]),
            "recall": float(rec_by_class[0]),
            "f1": float(f1_by_class[0]),
            "count": int(support_by_class[0]),
        },
        "class_1": {
            "precision": float(prec_by_class[1]),
            "recall": float(rec_by_class[1]),
            "f1": float(f1_by_class[1]),
            "count": int(support_by_class[1]),
        },
    }

    if out_json is not None:
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics to: {out_json}")

    return out_pred, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="Path to .h5ad (cell x feature) with obs metadata")
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--label_col", required=True, help="Binary label column in adata.obs")
    ap.add_argument("--label_map", default=None,
                    help='Optional JSON mapping, e.g. \'{"NO":0,"YES":1}\' (string)')

    ap.add_argument("--normalize", action="store_true", default=True,
                    help="Apply (X - 0.5) * 2 to adata.X (default: True)")
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable normalization (overrides --normalize)")

    ap.add_argument("--use_sample_source", action="store_true",
                    help="Append Sample_source one-hot to patient features (ONLY if Dataset.py supports it correctly)")
    ap.add_argument("--sample_source_col", default="Sample_source")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--reg_lambda", type=float, default=1.0)

    ap.add_argument("--scale_pos_weight", type=float, default=None,
                    help="If set, use XGBoost's scale_pos_weight. Typical: n_neg/n_pos from training fold.")
    ap.add_argument("--auto_scale_pos_weight", action="store_true",
                    help="Automatically set scale_pos_weight=n_neg/n_pos per fold (ignored if --scale_pos_weight is set).")

    ap.add_argument("--n_jobs", type=int, default=-1)

    ap.add_argument("--out_csv", default="xgb_loocv_predictions.csv")
    ap.add_argument("--out_json", default=None,
                    help="Optional: save summary metrics to JSON (e.g. results_xgb/xgb_metrics.json)")
    args = ap.parse_args()

    label_map = None
    if args.label_map is not None:
        label_map = json.loads(args.label_map)

    do_norm = args.normalize and (not args.no_normalize)

    run_xgb_loocv(
        h5ad=args.h5ad,
        label_col=args.label_col,
        patient_col=args.patient_col,
        label_map=label_map,
        normalize=do_norm,
        use_sample_source=args.use_sample_source,
        sample_source_col=args.sample_source_col,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=args.scale_pos_weight,
        auto_scale_pos_weight=args.auto_scale_pos_weight,
        n_jobs=args.n_jobs,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()