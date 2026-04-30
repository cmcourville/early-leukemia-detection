#!/bin/bash
#SBATCH --job-name=cyto_evaluate
#SBATCH --partition=quick
#SBATCH --account=cngan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=evaluate_%j.out
#SBATCH --error=evaluate_%j.err

# models/cytodiffusion/slurm/evaluate.sh
# Evaluate CytoDiffusionModel on a given dataset.
#
# Usage (pass DATASET and optionally DATA_DIR as environment variables):
#
#   # Raabin
#   sbatch models/cytodiffusion/slurm/evaluate.sh
#
#   # BCCD
#   DATASET=bccd sbatch models/cytodiffusion/slurm/evaluate.sh
#
#   # CytoData
#   DATASET=cytodata DATA_DIR=$HOME/data/cytodata \
#       sbatch models/cytodiffusion/slurm/evaluate.sh

set -e

DATASET="${DATASET:-raabin}"
DATA_DIR="${DATA_DIR:-}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV_NAME="cyto534"

echo "[evaluate] Dataset: $DATASET"
echo "[evaluate] GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p "$PROJECT_ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

DATA_DIR_ARG=""
if [ -n "$DATA_DIR" ]; then
    DATA_DIR_ARG="--data_dir $DATA_DIR"
fi

# Test split
python "$PROJECT_ROOT/models/cytodiffusion/evaluate.py" \
    --dataset    "$DATASET"         \
    --split      test               \
    --batch_size 64                 \
    --num_workers 4                 \
    $DATA_DIR_ARG                   \
    --device     cuda:0

# Validation split
python "$PROJECT_ROOT/models/cytodiffusion/evaluate.py" \
    --dataset    "$DATASET"         \
    --split      val                \
    --batch_size 64                 \
    --num_workers 4                 \
    $DATA_DIR_ARG                   \
    --device     cuda:0

echo "[evaluate] Done. Results in shared/results/cytodiffusion/"
