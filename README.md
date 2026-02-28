# tcell-risk-mil

tcell-risk-mil is a patient-level learning framework for single-cell CAR-T data, focused on relapse risk and survival modeling.  
It supports:
- Attention-based multiple instance learning (MIL)
- Optional autoencoder (AE) feature compression
- Survival modeling (Cox objective)
- Binary classification pipelines

This project is derived from the original tcellMIL codebase and extends it for risk/survival-focused workflows.

## Repository Layout

- `src/train_cox_loocv_ae.py`: Cox MIL with optional AE stage (`--use_ae`)
- `src/train_cox_loocv.py`: Cox MIL without AE
- `src/train_survival_baselines.py`: pooled-feature survival baselines (CoxPH / RSF)
- `src/train.py`: function-based binary MIL pipeline (`run_pipeline_loocv`)
- `src/extract_attention_cells_cox.py`: extract top-attended cells from trained Cox models
- `src/cox_perturb.py`, `src/binary_perturb.py`: perturbation analyses

## Installation

This repository currently does not include a pinned `requirements.txt`.  
Create an environment and install the core dependencies:

```bash
pip install torch scanpy anndata numpy pandas scikit-learn scipy
```

Optional (for some scripts):

```bash
pip install xgboost
```

## Input Data Format (`.h5ad`)

`adata.X`:
- Cell-by-feature matrix (for example SCENIC AUC scores)

Common `adata.obs` columns:
- `patient_id`: patient identifier

For survival scripts:
- An event indicator column passed via `--relapse_col` (default: `relapse_y_n`; 0/1 or compatible Yes/No values)
- `time_to_relapse_days`: time-to-event in days
- `follow_up_duration_months`: fallback follow-up for censored patients

Legacy (original tcellMIL classification workflow):
- `Response_3m` (or another classification label column)

Custom endpoint naming is supported.
If your event column has a different name, pass it with `--relapse_col` in survival scripts.

## Quick Start

### 1) Cox MIL + Optional AE (recommended script)

```bash
python src/train_cox_loocv_ae.py \
  --input_h5ad /path/to/data.h5ad \
  --outdir results_cox \
  --cv kfold --k 5 \
  --aggregator attention \
  --use_ae --latent_dim 64 --ae_epochs 200 \
  --train_full_model
```

Key options:
- `--cv loocv|kfold`
- `--aggregator attention|mean|q90`
- `--topk`, `--tau` (attention aggregator only)
- `--use_ae` to train AE first and use latent features
- `--train_full_model` to save final full-data MIL checkpoint

### 2) Cox MIL (no AE)

```bash
python src/train_cox_loocv.py \
  --input_h5ad /path/to/data.h5ad \
  --outdir results_cox \
  --cv kfold --k 5 \
  --aggregator attention
```

### 3) Survival Baselines

```bash
python src/train_survival_baselines.py \
  --input_h5ad /path/to/data.h5ad \
  --outdir results_survival_baselines \
  --methods coxph,rsf \
  --pool mean \
  --cv kfold --k 5
```

## Output Structure

`train_cox_loocv_ae.py` creates:

- `results_cox/run_YYYYMMDD_HHMMSS/autoencoder/`
- `results_cox/run_YYYYMMDD_HHMMSS/mil/fold_XX/best_model.pth`
- `results_cox/run_YYYYMMDD_HHMMSS/mil/full/best_model.pth` (if `--train_full_model`)
- `results_cox/run_YYYYMMDD_HHMMSS/cox_<loocv|kfoldK>_results.pkl`
- `results_cox/run_YYYYMMDD_HHMMSS/cox_risk_<loocv|kfoldK>.csv`

## Classification Pipeline (Optional)

`src/train.py` exposes `run_pipeline_loocv(...)` for AE + MIL classification:

```python
from src.train import run_pipeline_loocv

results = run_pipeline_loocv(
    input_file="/path/to/data.h5ad",
    output_dir="results",
    latent_dim=64,
    num_epochs_ae=200,
    num_epochs=50,
    label_col="Response_3m",
    cv="loocv"
)
```

## Notes

- Most scripts auto-detect CUDA (`torch.cuda.is_available()`).
- If your patient ID column is not `patient_id`, pass `--patient_col`.
- For `mean`/`q90` aggregators, `topk` and `tau` are ignored by design.

## Citation

If you use this repository, please cite both:
- The original tcellMIL work (base framework)
- This repository (`tcell-risk-mil`) for the risk/survival extensions

Tsui K. C. Y.*, Rodrigues K. B.*, Zhan X.*, Chen Y, Mo K. C., Mackall C. L., Miklos D. B., Gevaert O., Good Z. (2025).  
Patient-level prediction from single-cell data using attention-based multiple instance learning with regulatory priors.  
NeurIPS 2025 Workshop on AI Virtual Cells and Instruments (AI4D3 2025).
