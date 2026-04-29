#!/bin/bash
#SBATCH --job-name=cyto_train_cytodata
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_cytodata_%j.out
#SBATCH --error=logs/train_cytodata_%j.err

# models/cytodiffusion/slurm/train_cytodata.sh
# Train CytoDiffusionModel on CytoData (10-class, Addenbrooke's Hospital).
#
# BEFORE SUBMITTING:
#   1. Download CytoData (see models/cytodiffusion/data_prep/download_cytodata.sh)
#   2. Run prepare_cytodata.py to create the ImageFolder layout
#   3. Set CYTODATA_DIR below to your prepared dataset path
#
# Usage:
#   sbatch models/cytodiffusion/slurm/train_cytodata.sh

set -e

# ── SET THIS PATH ──────────────────────────────────────────────────────────────
CYTODATA_DIR="$HOME/data/cytodata"    # path to prepared ImageFolder directory
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CONDA_ENV_NAME="cyto534"

echo "[train_cytodata] Job ID      : $SLURM_JOB_ID"
echo "[train_cytodata] GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "[train_cytodata] Data dir    : $CYTODATA_DIR"

mkdir -p "$PROJECT_ROOT/logs"

if [ ! -d "$CYTODATA_DIR/train" ]; then
    echo "[train_cytodata] ERROR: $CYTODATA_DIR/train not found."
    echo "Run models/cytodiffusion/data_prep/prepare_cytodata.py first."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

python "$PROJECT_ROOT/models/cytodiffusion/train.py" \
    --dataset     cytodata          \
    --data_dir    "$CYTODATA_DIR"   \
    --batch_size  32                \
    --num_workers 4                 \
    --epochs      30                \
    --lr          1e-3              \
    --hidden_dim  512               \
    --dropout     0.3               \
    --freeze_encoder                \
    --device      cuda:0

echo "[train_cytodata] Done."
