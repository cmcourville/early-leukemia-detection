"""
shared/config.py — CS 534, Team 6 — WPI
Shared project-wide configuration constants.

All three model implementations should import dataset and class info
from here to stay in sync with each other.
"""

# Dataset (default — kept for backward compatibility)

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

# Multi-dataset configuration
# Each entry fully describes a supported dataset: where to load it from,
# how many classes it has, normalisation stats, and any special notes.

DATASET_CONFIGS = {

    # Raabin-WBC — 5-class WBC classification, pre-split on HuggingFace
    # Used as the primary benchmark in Zhu et al. (2026) / BccT paper.
    "raabin": {
        "display_name": "Raabin-WBC",
        "hf_name":      "polejowska/lcbsi-wbc-ap",
        "num_classes":  5,
        "class_names":  ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"],
        "mean":         [0.7442, 0.6384, 0.7516],
        "std":          [0.1580, 0.1914, 0.1225],
        "source":       "huggingface",         # loaded via datasets.load_dataset
        "task":         "classification",       # direct image→label pairs
        "hf_splits":    {"train": "train", "val": "validation", "test": "test"},
        "notes": (
            "16,633 images; 5 WBC classes. Pre-split 70/10/20. "
            "Test-A (in-domain) + Test-B (different microscope, domain shift). "
            "Primary benchmark for all three team models."
        ),
    },

    # BCCD — 3-class object detection dataset (WBC, RBC, Platelet).
    # Loaded from HuggingFace (keremberke/blood-cell-object-detection).
    # Bounding-box crops are extracted and used as classification samples.
    "bccd": {
        "display_name": "BCCD",
        "hf_name":      "keremberke/blood-cell-object-detection",
        "num_classes":  3,
        "class_names":  ["Platelet", "RBC", "WBC"],
        "mean":         [0.485, 0.456, 0.406],  # ImageNet defaults (stats not pre-computed)
        "std":          [0.229, 0.224, 0.225],
        "source":       "huggingface_detection",  # detection → crop each bbox for classification
        "task":         "classification",
        "hf_splits":    {"train": "train", "val": "validation", "test": "test"},
        "notes": (
            "364 images; WBC, RBC, Platelet. COCO/PASCAL VOC annotations. "
            "Each bounding box is cropped to form one classification sample. "
            "Small dataset — useful for object-level evaluation."
        ),
    },

    # CytoData — 10-class single-cell morphology dataset from Addenbrooke's Hospital.
    # NOT publicly available on HuggingFace. Must be requested from Cambridge
    # (see CytoDiffusion paper / CambridgeCIA GitHub for access instructions).
    # Expects a local directory with ImageFolder structure:
    #   <data_dir>/train/<ClassName>/image.jpg
    #   <data_dir>/val/<ClassName>/image.jpg
    #   <data_dir>/test/<ClassName>/image.jpg
    "cytodata": {
        "display_name": "CytoData",
        "hf_name":      None,    # not on HuggingFace — local path required
        "num_classes":  10,
        "class_names":  [
            "Basophil",       # BAS
            "Eosinophil",     # EOS
            "Erythroblast",   # EBO
            "Lymphocyte",     # LYT
            "Monocyte",       # MON
            "Myeloblast",     # MYB
            "Neutrophil",     # NEU (segmented)
            "Platelet",       # PLT
            "Promyelocyte",   # PMO
            "Artefact",       # NIF / artefact class
        ],
        "mean":         [0.485, 0.456, 0.406],  # ImageNet defaults (compute from your local copy)
        "std":          [0.229, 0.224, 0.225],
        "source":       "local",    # pass --data_dir /path/to/cytodata
        "task":         "classification",
        "hf_splits":    None,
        "notes": (
            "559,808 single-cell images from Addenbrooke's Hospital, Cambridge. "
            "Labeled subset: 4,996 images across 10 classes. "
            "Includes labeller confidence scores. Access via CambridgeCIA/CytoDiffusion GitHub. "
            "Class names above should match your local folder names exactly."
        ),
    },
}

# Convenience: valid dataset keys
DATASET_KEYS = list(DATASET_CONFIGS.keys())   # ["raabin", "bccd", "cytodata"]
