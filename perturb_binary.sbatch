#!/bin/bash
#SBATCH --job-name=perturb_binary
#SBATCH --partition=biochem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/perturb_binary_%j.out
#SBATCH --error=logs/perturb_binary_%j.err

set -euo pipefail

source ~/.bashrc
conda activate tcellmil_pip

# Repo root on cluster
cd /oak/stanford/groups/zinaida/peterl/tcellMIL_reg
mkdir -p logs results/perturb_binary

export PYTHONPATH=$PWD

# -------------------------------
# Fill these paths before submit
# -------------------------------
INPUT_H5AD="/oak/stanford/groups/zinaida/peterl/REPLACE_ME/input.h5ad"
AE_PATH="/oak/stanford/groups/zinaida/peterl/REPLACE_ME/autoencoder/best_autoencoder.pth"
OUT_CSV="results/perturb_binary/perturb_binary.csv"

# Option A: fold-ensemble (recommended if you trained kfold)
MIL_PATHS="\
/oak/stanford/groups/zinaida/peterl/REPLACE_ME/mil/fold_01/best_model.pth,\
/oak/stanford/groups/zinaida/peterl/REPLACE_ME/mil/fold_02/best_model.pth,\
/oak/stanford/groups/zinaida/peterl/REPLACE_ME/mil/fold_03/best_model.pth,\
/oak/stanford/groups/zinaida/peterl/REPLACE_ME/mil/fold_04/best_model.pth,\
/oak/stanford/groups/zinaida/peterl/REPLACE_ME/mil/fold_05/best_model.pth"

# Option B: single full model
# MIL_PATHS="/oak/stanford/groups/zinaida/peterl/REPLACE_ME/mil/full/best_model.pth"

python src/binary_perturb.py \
  --input_h5ad "${INPUT_H5AD}" \
  --out_csv "${OUT_CSV}" \
  --patient_col patient_id \
  --assume_scaled \
  --ae_path "${AE_PATH}" \
  --ae_latent_dim 64 \
  --mil_paths "${MIL_PATHS}" \
  --mil_hidden_dim 128 \
  --mil_dropout 0.25 \
  --aggregator attention \
  --topk 0 \
  --tau 0.0 \
  --pos_idx 1 \
  --mad_mult 3.0 \
  --tfs all \
  --directions up,down

