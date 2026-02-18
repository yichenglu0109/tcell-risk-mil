#!/usr/bin/env python3
import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)

from src.Dataset import PatientBagDataset


def mean_pool(bag: np.ndarray) -> np.ndarray:
    # bag: [n_cells, d]
    return bag.mean(axis=0)


def safe_metric(metric_fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return metric_fn(y_true, y_score)


def run_lr_loocv(
    h5ad: str,
    label_col: str,
    patient_col: str = "patient_id",
    label_map: dict | None = None,
    normalize: bool = True,
    use_sample_source: bool = False,
    sample_source_col: str = "Sample_source",
    seed: int = 42,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "liblinear",
    max_iter: int = 2000,
    class_weight: str = "balanced",   # "balanced" or "none"
    standardize: bool = True,         # True = use StandardScaler
    out_csv: str | None = "lr_loocv_predictions.csv",
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
    print(f"[INFO] Standardize: {'ON' if standardize else 'OFF'}")

    # ---- LOOCV ----
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y), dtype=float)
    y_pred = np.zeros(len(y), dtype=int)

    cw = None if class_weight == "none" else "balanced"

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

        lr = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=cw,
            random_state=seed,
        )

        if standardize:
            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", lr),
            ])
        else:
            model = lr

        model.fit(X_tr, y_tr)

        prob = model.predict_proba(X_te)[0]

        # Get class ordering robustly
        if hasattr(model, "classes_"):
            classes = model.classes_
        else:
            classes = model.named_steps["lr"].classes_

        cls_to_col = {c: j for j, c in enumerate(classes)}
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
    print("\n===== LOOCV Final Results (Logistic Regression) =====")
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
        "standardize": bool(standardize),
        "accuracy": float(acc),
        "n_correct": int(n_correct),
        "n_total": int(n_total),
        "precision_pos": float(overall_precision),
        "recall_pos": float(overall_recall),
        "f1_pos": float(overall_f1),
        "auroc": None if not np.isfinite(overall_auc) else float(overall_auc),
        "pr_auc": None if not np.isfinite(overall_pr_auc) else float(overall_pr_auc),
        "confusion_matrix": cm.tolist(),
        "lr_params": {
            "C": float(C),
            "penalty": str(penalty),
            "solver": str(solver),
            "max_iter": int(max_iter),
            "class_weight": str(class_weight),
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
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    ap.add_argument("--penalty", default="l2", choices=["l2"])
    ap.add_argument("--solver", default="liblinear", choices=["liblinear", "lbfgs", "saga"])
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--no_standardize", action="store_true",
                    help="Disable StandardScaler on patient-level features (default: standardize ON)")

    ap.add_argument("--out_csv", default="lr_loocv_predictions.csv")
    ap.add_argument("--out_json", default=None,
                    help="Optional: save summary metrics to JSON (e.g. results_lr/lr_metrics.json)")
    args = ap.parse_args()

    label_map = None
    if args.label_map is not None:
        label_map = json.loads(args.label_map)

    do_norm = args.normalize and (not args.no_normalize)
    standardize = (not args.no_standardize)

    run_lr_loocv(
        h5ad=args.h5ad,
        label_col=args.label_col,
        patient_col=args.patient_col,
        label_map=label_map,
        normalize=do_norm,
        use_sample_source=args.use_sample_source,
        sample_source_col=args.sample_source_col,
        seed=args.seed,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        standardize=standardize,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()