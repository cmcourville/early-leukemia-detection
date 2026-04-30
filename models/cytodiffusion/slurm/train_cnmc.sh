#!/bin/bash
#SBATCH --job-name=cyto_train_cnmc
#SBATCH --partition=short
#SBATCH --account=cngan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=train_cnmc_%j.out
#SBATCH --error=train_cnmc_%j.err

# Train CytoDiffusionModel on C-NMC 2019 (binary: HEM vs ALL).
# Submit from project root: sbatch models/cytodiffusion/slurm/train_cnmc.sh

set -e

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
CONDA_ENV_NAME="cyto534"
HF_CACHE="$PROJECT_ROOT/.hf_cache"

echo "[train_cnmc] Job ID: $SLURM_JOB_ID"
echo "[train_cnmc] Node  : $SLURMD_NODENAME"
echo "[train_cnmc] GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

python "$PROJECT_ROOT/models/cytodiffusion/train.py" \
    --dataset     cnmc              \
    --batch_size  32                \
    --num_workers 4                 \
    --epochs      30                \
    --lr          1e-3              \
    --hidden_dim  512               \
    --dropout     0.3               \
    --freeze_encoder                \
    --cache_dir   "$HF_CACHE"       \
    --device      cuda:0

echo "[train_cnmc] Done."
