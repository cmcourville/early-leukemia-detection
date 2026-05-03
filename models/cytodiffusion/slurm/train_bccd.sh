#!/bin/bash
#SBATCH --job-name=cyto_train_bccd
#SBATCH --partition=short
#SBATCH --account=cngan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=train_bccd_%j.out
#SBATCH --error=train_bccd_%j.err

# models/cytodiffusion/slurm/train_bccd.sh
# Train CytoDiffusionModel on BCCD (3-class cross-domain benchmark).
# BCCD is small (~364 images → ~2K crops) so training is fast.
#
# Usage:
#   sbatch models/cytodiffusion/slurm/train_bccd.sh

set -e

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
CONDA_ENV_NAME="cyto534"
HF_CACHE="$PROJECT_ROOT/.hf_cache"

echo "[train_bccd] Job ID: $SLURM_JOB_ID"
echo "[train_bccd] GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

python "$PROJECT_ROOT/models/cytodiffusion/train.py" \
    --dataset     bccd              \
    --batch_size  32                \
    --num_workers 4                 \
    --epochs      30                \
    --lr          1e-3              \
    --hidden_dim  512               \
    --dropout     0.3               \
    --freeze_encoder                \
    --cache_dir   "$HF_CACHE"       \
    --device      cuda:0

echo "[train_bccd] Done."
