# run_kfold_only.py

import scanpy as sc
import os
from src.train import cross_validation_mil

input_file = "/path/to/your_latent_or_original.h5ad"
output_dir = "results/kfold_test"

os.makedirs(output_dir, exist_ok=True)

print("Loading:", input_file)
adata = sc.read_h5ad(input_file)

input_dim = adata.X.shape[1]
print("input_dim =", input_dim)

results = cross_validation_mil(
    adata=adata,
    input_dim=input_dim,
    num_classes=2,
    hidden_dim=128,
    num_epochs=60,
    label_col="relapse_y_n",
    aggregator="attention",
    cv="kfold",
    k=5,
    seed=42,
    save_path=output_dir,
)

print(results["overall_metrics"])