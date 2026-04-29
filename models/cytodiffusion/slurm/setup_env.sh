#!/bin/bash
# models/cytodiffusion/slurm/setup_env.sh
#
# One-time setup: creates the conda environment on the WPI cluster.
# Run this ONCE before submitting any SLURM jobs.
#
# Usage:
#   bash models/cytodiffusion/slurm/setup_env.sh
#
# NOTE: adjust CONDA_ENV_NAME if you prefer a different name.

set -e

CONDA_ENV_NAME="cyto534"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "[setup] Project root: $PROJECT_ROOT"
echo "[setup] Creating conda env: $CONDA_ENV_NAME (Python 3.10)"

# Activate conda in non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n "$CONDA_ENV_NAME" python=3.10

conda activate "$CONDA_ENV_NAME"

echo "[setup] Installing shared requirements..."
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/shared/requirements.txt"

echo "[setup] Installing CytoDiffusion requirements..."
pip install -r "$PROJECT_ROOT/models/cytodiffusion/requirements.txt"

echo "[setup] Done. Activate with: conda activate $CONDA_ENV_NAME"
