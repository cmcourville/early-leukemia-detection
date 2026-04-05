"""
shared/config.py — CS 534, Team 6 — WPI
Shared project-wide configuration constants.

All three model implementations should import dataset and class info
from here to stay in sync with each other.
"""

# Dataset

HF_DATASET_NAME = "polejowska/lcbsi-wbc-ap"
NUM_CLASSES      = 5
IMAGE_SIZE       = 224   # all models use 224×224

# Raabin-WBC class names (order matches HF dataset label integers 0-4)
CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
]

# Dataset-wise normalisation stats (computed from the training split)
RAABIN_MEAN = [0.7442, 0.6384, 0.7516]
RAABIN_STD  = [0.1580, 0.1914, 0.1225]

# Data split fractions (informational — the HF dataset ships pre-split)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20

# Shared experiment settings

# Low-data regime shot counts (used by all three models)
LOW_DATA_SHOTS   = [10, 20, 50]
LOW_DATA_REPEATS = 3          # number of independent random seeds per shot count
GLOBAL_SEED      = 42

# Model identifiers (used by run_all.py and results/ naming)

MODEL_NAMES = {
    "bcct":            "BccT (Corrin)",
    "cytodiffusion":   "CytoDiffusion / LDM (Darshan)",
    "vitcnn_ensemble": "ViT-CNN Ensemble (Sean)",
}

# Backbone shared by BccT and ViT-CNN Ensemble
VIT_BACKBONE_ID = "google/vit-base-patch16-224-in21k"

# Results directory layout (relative to project root)

RESULTS_ROOT    = "shared/results"
CHECKPOINT_DIRS = {
    "bcct":            "models/bcct/checkpoints",
    "cytodiffusion":   "models/cytodiffusion/checkpoints",
    "vitcnn_ensemble": "models/vitcnn_ensemble/checkpoints",
}
