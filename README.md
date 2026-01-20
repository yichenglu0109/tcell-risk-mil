# tcellMIL Framework

A computational framework for CAR T cell therapy response prediction using Multiple Instance Learning (MIL) on single-cell regulatory network data.

## Overview

The tcellMIL framework consists of two main components:
1. **Autoencoder**: Performs feature denoising of SCENIC AUC matrices
2. **Attention-based MIL**: Predicts patient-level responses using attention mechanisms

## Data Requirements

Your input data should be an `.h5ad` file containing:
- SCENIC AUC matrix (cells × transcription factors)
- Patient identifiers in `adata.obs['patient_id']`
- Response labels in `adata.obs['Response_3m']` (values: "NR", "OR")
- Optional: Sample source information in `adata.obs['Sample_source']`

## Quick Start

### 1. Basic Training

```python
from train_tcellMIL import run_pipeline_loocv

# Run complete pipeline with default parameters
results = run_pipeline_loocv(
    input_file='path/to/your/data.h5ad',
    output_dir='results'
)
```

### 2. Custom Parameters

```python
results = run_pipeline_loocv(
    input_file='path/to/your/data.h5ad',
    output_dir='results',
    latent_dim=64,           # Autoencoder latent dimension
    num_epochs_ae=200,       # Autoencoder training epochs
    num_epochs=50,           # MIL training epochs
    num_classes=2,           # Number of response classes
    hidden_dim=128,          # MIL hidden layer dimension
    sample_source_dim=4      # Sample source encoding dimension
)
```

## Framework Architecture

### Autoencoder Training
- Input: SCENIC AUC matrices (cells × TFs)
- Output: Latent representations (cells × latent_dim)
- Architecture: Encoder-Decoder with LayerNorm and Dropout

### MIL Training
- Input: Patient bags (collections of latent cell representations)
- Method: Leave-one-out cross-validation
- Architecture: Attention-based MIL with optional sample source integration
- Output: Patient-level response predictions


## Key Features

- **Leave-one-out cross-validation**: Robust evaluation for small patient cohorts
- **Class balancing**: Automatic handling of imbalanced response classes
- **Early stopping**: Prevents overfitting with patience-based stopping
- **Attention visualization**: Provides interpretable attention weights
- **GPU acceleration**: Automatic CUDA detection and usage

## Dependencies

- PyTorch
- Scanpy
- Scikit-learn
- NumPy
- Matplotlib

## Citation
Tsui K. C. Y.*, Rodrigues KB*, Zhan X*, Chen Y, Mo KC, Mackall CL, Miklos DB, Gevaert O§, Good Z§. (2025) Patient-level prediction from single-cell data using attention-based multiple instance learning with regulatory priors. The NeurIPS 2025 Workshop on AI Virtual Cells and Instruments: A New Era in Drug Discovery and Development (AI4D3 2025),
