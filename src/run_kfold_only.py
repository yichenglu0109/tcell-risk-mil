# src/run_kfold_only.py

import scanpy as sc
from src.train import cross_validation_mil


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
):
    """
    Wrapper function for K-fold MIL training.
    Can be imported safely without side effects.
    """

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
    )

    return results

if __name__ == "__main__":

    input_file = "path/to/your_file.h5ad"
    output_dir = "path/to/output_dir"

    results = run_kfold(input_file, output_dir)
    print(results["overall_metrics"])