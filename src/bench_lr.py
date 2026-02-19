#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)

from src.Dataset import PatientBagDataset

# ---------------- helpers ----------------
def mean_pool(bag: np.ndarray) -> np.ndarray:
    return bag.mean(axis=0)

def safe_metric(metric_fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return metric_fn(y_true, y_score)

def _to_jsonable(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    return x

def append_jsonl(path: str, record: dict):
    # rf.py 是 os.makedirs(os.path.dirname(path)), 但若 path 沒資料夾會炸
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_to_jsonable, ensure_ascii=False) + "\n")

def build_patient_matrix(
    adata,
    label_col: str,
    patient_col: str = "patient_id",
    label_map: Optional[Dict[str, int]] = None,
    drop_missing: bool = True,
    use_sample_source: bool = False,
    sample_source_col: str = "Sample_source",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # 兼容 Dataset.py 若沒有 cache_bags 參數
    try:
        ds = PatientBagDataset(
            adata=adata,
            patient_col=patient_col,
            task_type="classification",
            label_col=label_col,
            label_map=label_map,
            drop_missing=drop_missing,
            use_sample_source=use_sample_source,
            sample_source_col=sample_source_col,
            cache_bags=False,   # 防止 init 就 materialize 全部 bags
        )
    except TypeError:
        ds = PatientBagDataset(
            adata=adata,
            patient_col=patient_col,
            task_type="classification",
            label_col=label_col,
            label_map=label_map,
            drop_missing=drop_missing,
            use_sample_source=use_sample_source,
            sample_source_col=sample_source_col,
        )

    X_list, y_list, pid_list = [], [], []
    for i in range(len(ds)):
        out = ds[i]
        if len(out) == 4:
            bag_t, label_t, pid, one_hot_t = out
            one_hot = one_hot_t.numpy().astype(np.float32)
        else:
            bag_t, label_t, pid = out
            one_hot = None

        bag = bag_t.numpy().astype(np.float32)
        x = mean_pool(bag)
        if one_hot is not None:
            x = np.concatenate([x, one_hot], axis=0)

        X_list.append(x)
        y_list.append(int(label_t.item()))
        pid_list.append(str(pid))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    pids = np.array(pid_list, dtype=object)
    return X, y, pids

def summarize_metrics(y_true, y_prob, thr: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    auroc = safe_metric(roc_auc_score, y_true, y_prob)
    auprc = safe_metric(average_precision_score, y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": None if not np.isfinite(auroc) else float(auroc),      # AUROC
        "auprc": None if not np.isfinite(auprc) else float(auprc),
        "confusion_matrix": cm.tolist(),
    }

# ---------------- main runner ----------------
def run_lr_cv(
    h5ad: str,
    label_col: str,
    patient_col: str = "patient_id",
    label_map: Optional[Dict[str, int]] = None,
    normalize: bool = True,
    normalize_mode: str = "x05x2",
    use_sample_source: bool = False,
    sample_source_col: str = "Sample_source",
    seed: int = 42,
    cv: str = "kfold",
    k: int = 5,
    out_dir: str = "results_lr",
    results_jsonl: Optional[str] = None,
    append_log: bool = True,
    # LR params
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "liblinear",
    max_iter: int = 2000,
    class_weight: str = "balanced",   # "balanced" or "none"
    standardize: bool = True,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    adata = sc.read_h5ad(h5ad)
    if normalize and normalize_mode == "x05x2":
        adata.X = (adata.X - 0.5) * 2

    X, y, pids = build_patient_matrix(
        adata=adata,
        label_col=label_col,
        patient_col=patient_col,
        label_map=label_map,
        drop_missing=True,
        use_sample_source=use_sample_source,
        sample_source_col=sample_source_col,
    )

    print(f"[INFO] Patients used: {len(pids)}")
    print(f"[INFO] Feature dim: {X.shape[1]}")
    print(f"[INFO] Class counts: {pd.Series(y).value_counts().to_dict()}")
    print(f"[INFO] CV={cv} k={k if cv=='kfold' else 'NA'} seed={seed}")
    print(f"[INFO] LR params: C={C} penalty={penalty} solver={solver} "
          f"max_iter={max_iter} class_weight={class_weight} standardize={standardize}")

    if cv == "loocv":
        splits = list(LeaveOneOut().split(X))
    elif cv == "kfold":
        skf = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))
    else:
        raise ValueError("cv must be 'kfold' or 'loocv'")

    y_prob = np.zeros(len(y), dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te = X[te_idx]

        uniq = np.unique(y_tr)
        if len(uniq) < 2:
            only = int(uniq[0])
            y_prob[te_idx] = 1.0 if only == 1 else 0.0
            continue

        cw = None if class_weight == "none" else "balanced"

        lr = LogisticRegression(
            C=float(C),
            penalty=str(penalty),
            solver=str(solver),
            max_iter=int(max_iter),
            class_weight=cw,
            random_state=int(seed),
        )

        if standardize:
            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", lr),
            ])
        else:
            model = lr

        model.fit(X_tr, y_tr)

        prob = model.predict_proba(X_te)
        # robustly get column for class 1
        classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["lr"].classes_
        cls_to_col = {int(c): j for j, c in enumerate(classes)}
        j1 = cls_to_col.get(1, int(np.argmax(prob.mean(axis=0))))
        y_prob[te_idx] = prob[:, j1]

        if cv == "kfold":
            fold_metrics = summarize_metrics(y[te_idx], y_prob[te_idx], thr=0.5)
            print(f"[Fold {fold}/{len(splits)}] n_test={len(te_idx)} "
                  f"auc={fold_metrics['auc']} auprc={fold_metrics['auprc']} "
                  f"acc={fold_metrics['accuracy']:.3f} f1={fold_metrics['f1']:.3f}")

    overall = summarize_metrics(y, y_prob, thr=0.5)

    pred_df = pd.DataFrame({
        "patient_id": pids,
        "y_true": y,
        "y_prob": y_prob,
        "y_pred": (y_prob >= 0.5).astype(int),
    }).sort_values("patient_id")

    pred_csv = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"[INFO] Saved predictions: {pred_csv}")

    cv_results = {
        "model": "lr_meanpool",
        "cv": cv,
        "k": int(k) if cv == "kfold" else None,
        "seed": int(seed),
        "overall_metrics": overall,
        "patients_used": int(len(pids)),
        "feature_dim": int(X.shape[1]),
    }

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(cv_results, f, indent=2, default=_to_jsonable)
    print(f"[INFO] Saved summary: {summary_json}")

    # ---- append JSONL (prefer your logging_utils if available) ----
    if append_log and results_jsonl is not None:
        params = {
            "model": "lr_meanpool",
            "h5ad": h5ad,
            "label_col": label_col,
            "patient_col": patient_col,
            "cv": cv,
            "k": int(k) if cv == "kfold" else None,
            "seed": seed,
            "normalize": bool(normalize),
            "normalize_mode": normalize_mode,
            "use_sample_source": bool(use_sample_source),
            "sample_source_col": sample_source_col,
            # LR params
            "C": float(C),
            "penalty": str(penalty),
            "solver": str(solver),
            "max_iter": int(max_iter),
            "class_weight": str(class_weight),
            "standardize": bool(standardize),
        }
        try:
            from src.logging_utils import log_cv_run
            log_cv_run(results_jsonl, params=params, cv_results=cv_results)
            print(f"[INFO] Appended via logging_utils: {results_jsonl}")
        except Exception as e:
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "run_id": f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{int(time.time()*1000)%100000}',
                **{f"p_{k}": v for k, v in params.items()},
                **{f"m_{k}": v for k, v in overall.items()},
            }
            append_jsonl(results_jsonl, record)
            print(f"[WARN] logging_utils unavailable ({e}); appended fallback JSONL: {results_jsonl}")

    print("\n===== CV Final Results (LR mean pooling) =====")
    print(f"Accuracy: {overall['accuracy']:.4f}")
    print(f"Precision: {overall['precision']:.4f}")
    print(f"Recall: {overall['recall']:.4f}")
    print(f"F1: {overall['f1']:.4f}")
    print(f"AUROC: {overall['auc']}")
    print(f"AUPRC: {overall['auprc']}")
    print("Confusion Matrix:")
    print(np.array(overall["confusion_matrix"]))

    return cv_results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--patient_col", default="patient_id")
    ap.add_argument("--label_map", default=None, help='JSON string, e.g. \'{"NR":0,"R":1}\'')

    ap.add_argument("--cv", default="kfold", choices=["kfold", "loocv"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--normalize_mode", default="x05x2", choices=["x05x2", "off"])

    ap.add_argument("--use_sample_source", action="store_true")
    ap.add_argument("--sample_source_col", default="Sample_source")

    ap.add_argument("--out_dir", default="results_lr")
    ap.add_argument("--results_jsonl", default=None)
    ap.add_argument("--no_append_log", action="store_true")

    # LR params
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--penalty", default="l2", choices=["l2"])
    ap.add_argument("--solver", default="liblinear", choices=["liblinear", "lbfgs", "saga"])
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--no_standardize", action="store_true")

    args = ap.parse_args()

    label_map = None
    if args.label_map is not None:
        label_map = json.loads(args.label_map)

    do_norm = args.normalize and (not args.no_normalize) and (args.normalize_mode != "off")
    standardize = (not args.no_standardize)

    run_lr_cv(
        h5ad=args.h5ad,
        label_col=args.label_col,
        patient_col=args.patient_col,
        label_map=label_map,
        normalize=do_norm,
        normalize_mode=args.normalize_mode,
        use_sample_source=args.use_sample_source,
        sample_source_col=args.sample_source_col,
        seed=args.seed,
        cv=args.cv,
        k=args.k,
        out_dir=args.out_dir,
        results_jsonl=args.results_jsonl,
        append_log=(not args.no_append_log),
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        class_weight=args.class_weight,
        standardize=standardize,
    )

if __name__ == "__main__":
    main()