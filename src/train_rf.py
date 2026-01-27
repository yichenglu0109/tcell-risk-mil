#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)

from src.Dataset import PatientBagDataset


def mean_pool(bag: np.ndarray) -> np.ndarray:
    # bag: [n_cells, d]
    return bag.mean(axis=0)


def safe_metric(metric_fn, y_true, y_score):
    # If only one class appears, metrics like AUROC/AP are undefined
    if len(np.unique(y_true)) < 2:
        return np.nan
    return metric_fn(y_true, y_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="Path to .h5ad (cell x feature) with obs metadata")
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--label_col", required=True, help="Binary label column in adata.obs")
    ap.add_argument("--label_map", default=None,
                    help='Optional JSON mapping, e.g. \'{"NO":0,"YES":1}\' (string)')

    # Normalize to match your load_and_explore_data(): (X-0.5)*2
    ap.add_argument("--normalize", action="store_true", default=True,
                    help="Apply (X - 0.5) * 2 to adata.X (default: True)")
    ap.add_argument("--no_normalize", action="store_true",
                    help="Disable normalization (overrides --normalize)")

    # (Optional) Sample_source one-hot (NOTE: your Dataset.py currently has a bug if enabled; see note below)
    ap.add_argument("--use_sample_source", action="store_true",
                    help="Append Sample_source one-hot to patient features (ONLY if Dataset.py supports it correctly)")
    ap.add_argument("--sample_source_col", default="Sample_source")

    # RF hyperparams
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--n_jobs", type=int, default=-1)

    ap.add_argument("--out_csv", default="rf_loocv_predictions.csv")
    ap.add_argument("--out_json", default=None,
                    help="Optional: save summary metrics to JSON (e.g. results_rf/rf_metrics.json)")
    args = ap.parse_args()

    # ---- load ----
    adata = sc.read_h5ad(args.h5ad)

    # ---- normalize ----
    do_norm = args.normalize and (not args.no_normalize)
    if do_norm:
        adata.X = (adata.X - 0.5) * 2

    # ---- label_map ----
    label_map = None
    if args.label_map is not None:
        label_map = json.loads(args.label_map)

    # ---- dataset (patient bags) ----
    # NOTE: if you pass --use_sample_source, your current Dataset.py will likely crash because
    # self.patient_metadata is initialized as None but later treated as dict.
    ds = PatientBagDataset(
        adata=adata,
        patient_col=args.patient_col,
        task_type="classification",
        label_col=args.label_col,
        label_map=label_map,
        drop_missing=True,
        use_sample_source=args.use_sample_source,
        sample_source_col=args.sample_source_col,
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
    print(f"[INFO] Normalization: {'(X-0.5)*2' if do_norm else 'OFF'}")

    # ---- LOOCV ----
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y), dtype=float)
    y_pred = np.zeros(len(y), dtype=int)

    cw = None if args.class_weight == "none" else "balanced"

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=cw,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        rf.fit(X_tr, y_tr)

        prob = rf.predict_proba(X_te)[0]
        cls_to_col = {c: j for j, c in enumerate(rf.classes_)}
        p1 = prob[cls_to_col.get(1, np.argmax(prob))]  # probability of class 1

        y_prob[test_idx[0]] = float(p1)
        y_pred[test_idx[0]] = int(p1 >= 0.5)

    # ---- metrics ----
    acc = accuracy_score(y, y_pred)
    n_correct = int((y_pred == y).sum())
    n_total = int(len(y))

    overall_auc = safe_metric(roc_auc_score, y, y_prob)
    overall_pr_auc = safe_metric(average_precision_score, y, y_prob)

    # "Overall Precision/Recall/F1" as in your screenshot: for positive class (1)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", pos_label=1, zero_division=0
    )

    cm = confusion_matrix(y, y_pred)

    # class-specific precision/recall/support
    prec_by_class, rec_by_class, f1_by_class, support_by_class = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    # ---- print in your preferred format ----
    print("\n===== LOOCV Final Results =====")
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

    # ---- save predictions ----
    out = pd.DataFrame({
        "patient_id": pids,
        "y_true": y,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }).sort_values("patient_id")
    out.to_csv(args.out_csv, index=False)
    print(f"\n[INFO] Saved predictions to: {args.out_csv}")

    # ---- optional: save metrics json ----
    if args.out_json is not None:
        metrics = {
            "patients_used": int(len(pids)),
            "feature_dim": int(X.shape[1]),
            "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
            "normalization": "(X-0.5)*2" if do_norm else "OFF",
            "accuracy": float(acc),
            "n_correct": int(n_correct),
            "n_total": int(n_total),
            "precision_pos": float(overall_precision),
            "recall_pos": float(overall_recall),
            "f1_pos": float(overall_f1),
            "auroc": None if not np.isfinite(overall_auc) else float(overall_auc),
            "pr_auc": None if not np.isfinite(overall_pr_auc) else float(overall_pr_auc),
            "confusion_matrix": cm.tolist(),
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
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics to: {args.out_json}")


if __name__ == "__main__":
    main()
