#!/bin/bash
#SBATCH --job-name=cyto_low_data
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/low_data_%j.out
#SBATCH --error=logs/low_data_%j.err

# models/cytodiffusion/slurm/low_data.sh
# Low-data (n-shot) experiments for CytoDiffusionModel.
# Runs 10-shot, 20-shot, 50-shot × 3 repeats on Raabin-WBC.
#
# Requires a trained checkpoint at:
#   models/cytodiffusion/checkpoints/cytodiffusion_raabin.pt
#
# Usage:
#   sbatch models/cytodiffusion/slurm/low_data.sh
#
#   For a different dataset:
#   DATASET=bccd sbatch models/cytodiffusion/slurm/low_data.sh

set -e

DATASET="${DATASET:-raabin}"
DATA_DIR="${DATA_DIR:-}"
HF_CACHE_DIR="${HF_CACHE:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV_NAME="cyto534"

echo "[low_data] Dataset: $DATASET"
echo "[low_data] GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

EXTRA_ARGS=""
[ -n "$DATA_DIR" ]      && EXTRA_ARGS="$EXTRA_ARGS --data_dir $DATA_DIR"
[ -n "$HF_CACHE_DIR" ]  && EXTRA_ARGS="$EXTRA_ARGS --cache_dir $HF_CACHE_DIR"

python "$PROJECT_ROOT/run_all.py" \
    --mode       low_data         \
    --models     cytodiffusion    \
    --dataset    "$DATASET"       \
    --batch_size 32               \
    --num_workers 4               \
    --shots      10 20 50         \
    --num_repeats 3               \
    --device     cuda:0           \
    $EXTRA_ARGS

echo "[low_data] Done. Results in shared/results/cytodiffusion/low_data/"
