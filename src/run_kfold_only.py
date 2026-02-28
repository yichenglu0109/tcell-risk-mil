# src/run_kfold_only.py
import scanpy as sc
from src.train import cross_validation_mil
from src.logging_utils import log_cv_run   # for logging CV results to a JSONL file for easy tracking and analysis
import os

def run_kfold(
    input_file,
    output_dir,
    hidden_dim=128,
    num_epochs=60,
    label_col="relapse_y_n",
    aggregator="attention",
    topk=0,
    tau=0.0,
    cv="kfold",
    k=5,
    seed=42,
    results_jsonl=None,
    append_log=True,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading:", input_file)
    adata = sc.read_h5ad(input_file)
    input_dim = adata.X.shape[1]

    results = cross_validation_mil(
        adata=adata,
        input_dim=input_dim,
        num_classes=2,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        label_col=label_col,
        aggregator=aggregator,
        topk=topk,
        tau=tau,
        cv=cv,
        k=k,
        seed=seed,
        save_path=output_dir,
        store_attention=False,
    )

    # append to jsonl
    if append_log and results_jsonl is not None:
        params = dict(
            input_file=input_file,
            output_dir=output_dir,
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            label_col=label_col,
            aggregator=aggregator,
            topk=topk,
            tau=tau,
            cv=cv,
            k=k,
            seed=seed,
        )
        log_cv_run(results_jsonl, params=params, cv_results=results)

    return results