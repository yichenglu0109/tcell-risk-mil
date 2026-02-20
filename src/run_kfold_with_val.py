# src/run_kfold_with_val.py
import os
import scanpy as sc

from src.logging_utils import log_cv_run
from train_kfold_with_val import kfold_patient_cv_with_val  # 你把核心函數放這（下面我也給你）


def run_kfold_with_val(
    input_file,
    output_dir,
    input_dim=None,          # 可選：不給就用 adata.X.shape[1]
    hidden_dim=128,
    num_epochs=60,
    label_col="relapse_y_n",
    aggregator="attention",
    topk=0,
    tau=0.0,
    k=5,
    val_frac=0.2,
    seed=42,
    results_jsonl=None,
    append_log=True,
    select_metric="val_loss",   # or "val_balanced_accuracy"
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading:", input_file)
    adata = sc.read_h5ad(input_file)

    if input_dim is None:
        input_dim = adata.X.shape[1]
    print("[INFO] input_dim =", input_dim)

    results = kfold_patient_cv_with_val(
        adata=adata,
        input_dim=input_dim,
        label_col=label_col,
        save_path=output_dir,
        k=k,
        val_frac=val_frac,
        seed=seed,
        num_classes=2,
        hidden_dim=hidden_dim,
        aggregator=aggregator,
        topk=topk,
        tau=tau,
        num_epochs=num_epochs,
        select_metric=select_metric,
    )

    # ✅ append to jsonl (same style as run_kfold_only)
    if append_log and (results_jsonl is not None):
        params = dict(
            input_file=input_file,
            output_dir=output_dir,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            label_col=label_col,
            aggregator=aggregator,
            topk=topk,
            tau=tau,
            cv="kfold_with_val",
            k=k,
            val_frac=val_frac,
            seed=seed,
            select_metric=select_metric,
        )
        log_cv_run(results_jsonl, params=params, cv_results=results)

    return results