# create_latent.py
import os
from datetime import datetime
from src.Dataset import load_and_explore_data, preprocess_data
from src.Autoencoder import train_autoencoder, evaluate_autoencoder

def create_latent(
    input_file,
    output_dir="results_latent",
    latent_dim=32,
    num_epochs_ae=150,
    seed=42,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"run_{timestamp}")
    ae_dir = os.path.join(result_dir, "autoencoder")
    os.makedirs(ae_dir, exist_ok=True)

    print("="*80)
    print("STEP 1: LOAD")
    print("="*80)
    adata = load_and_explore_data(input_file)

    print("="*80)
    print("STEP 2: PREPROCESS (patient split)")
    print("="*80)
    train_loader, val_loader, test_loader, input_dim = preprocess_data(adata, random_state=seed)

    print("="*80)
    print("STEP 3: TRAIN AE")
    print("="*80)
    model, train_losses, val_losses = train_autoencoder(
        train_loader, val_loader, input_dim, latent_dim, num_epochs_ae, save_path=ae_dir
    )

    print("="*80)
    print("STEP 4: EVAL AE + EXPORT LATENT")
    print("="*80)
    adata_latent, test_loss = evaluate_autoencoder(
        model, test_loader, adata, adata.var_names.tolist(), save_path=ae_dir
    )

    latent_file = os.path.join(ae_dir, "latent_representation.h5ad")
    adata_latent.write(latent_file)
    print(f"[INFO] Saved latent: {latent_file}")
    print(f"[INFO] latent shape = {adata_latent.shape} (cells x latent_dim)")

    return latent_file  