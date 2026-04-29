#!/bin/bash
#SBATCH --job-name=cyto_train_raabin
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_raabin_%j.out
#SBATCH --error=logs/train_raabin_%j.err

# models/cytodiffusion/slurm/train_raabin.sh
# Train CytoDiffusionModel on Raabin-WBC (primary benchmark).
#
# Usage:
#   sbatch models/cytodiffusion/slurm/train_raabin.sh
#
# Submit from the project root directory.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV_NAME="cyto534"
HF_CACHE="$PROJECT_ROOT/.hf_cache"

echo "[train_raabin] Job ID      : $SLURM_JOB_ID"
echo "[train_raabin] Node        : $SLURMD_NODENAME"
echo "[train_raabin] Project root: $PROJECT_ROOT"
echo "[train_raabin] GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

python "$PROJECT_ROOT/models/cytodiffusion/train.py" \
    --dataset     raabin            \
    --batch_size  32                \
    --num_workers 4                 \
    --epochs      30                \
    --lr          1e-3              \
    --hidden_dim  512               \
    --dropout     0.3               \
    --freeze_encoder                \
    --cache_dir   "$HF_CACHE"       \
    --device      cuda:0

echo "[train_raabin] Done."
