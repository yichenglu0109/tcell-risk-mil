#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import LeaveOneOut

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
                    help='Optional JSON mapping, e.g. \'{"NR":0,"OR":1}\' (string)')

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
    args = ap.parse_args()

    # ---- load ----
    adata = sc.read_h5ad(args.h5ad)

    # ---- normalize ----
    do_norm = args.normalize and (not args.no_normalize)
    if do_norm:
        # matches your load_and_explore_data()
        adata.X = (adata.X - 0.5) * 2

    # ---- label_map ----
    label_map = None
    if args.label_map is not None:
        import json
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

    auroc = safe_metric(roc_auc_score, y, y_prob)
    ap_score = safe_metric(average_precision_score, y, y_prob)
    acc = accuracy_score(y, y_pred)

    print(f"[RESULT] LOOCV AUROC: {auroc:.4f}" if np.isfinite(auroc) else "[RESULT] LOOCV AUROC: NA (single-class?)")
    print(f"[RESULT] LOOCV PR-AUC: {ap_score:.4f}" if np.isfinite(ap_score) else "[RESULT] LOOCV PR-AUC: NA (single-class?)")
    print(f"[RESULT] LOOCV Accuracy: {acc:.4f}")

    out = pd.DataFrame({
        "patient_id": pids,
        "y_true": y,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }).sort_values("patient_id")
    out.to_csv(args.out_csv, index=False)
    print(f"[INFO] Saved predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
