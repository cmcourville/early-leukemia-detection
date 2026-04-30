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

    # Raabin-WBC — 5-class WBC classification (primary benchmark).
    #
    # Source: Acevedo et al. (2020), "A dataset for microscopic peripheral blood
    # cell images for development of automatic recognition systems", Data in Brief.
    # Mendeley Data: https://data.mendeley.com/datasets/snkd93bnjr/1
    # (Used as the primary benchmark in place of polejowska/lcbsi-wbc-ap, which
    # has been removed from HuggingFace. The Acevedo dataset is the underlying
    # source of that HF dataset and uses the same 5 WBC classes.)
    #
    # Setup: download PBC_dataset_normal_DIB.zip from Mendeley, unzip to data/raabin/,
    # then run: python prepare_raabin.py
    # This creates data/raabin/{train,val,test}/<ClassName>/ with a 70/10/20 split.
    "raabin": {
        "display_name": "Raabin-WBC (Acevedo et al. 2020)",
        "hf_name":      None,   # not on HuggingFace — use local path
        "num_classes":  5,
        "class_names":  ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"],
        "mean":         [0.7442, 0.6384, 0.7516],
        "std":          [0.1580, 0.1914, 0.1225],
        "source":       "local",        # loaded via LocalImageFolderDataset
        "task":         "classification",
        "hf_splits":    None,
        "notes": (
            "10,299 images; 5 WBC classes (basophil, eosinophil, lymphocyte, monocyte, "
            "neutrophil). Split 70/10/20 via prepare_raabin.py. "
            "Source: Acevedo et al. 2020, Hospital Clinic of Barcelona. "
            "Local path: data/raabin/ (relative to early-leukemia-detection/). "
            "Run with: python run_all.py --dataset raabin --data_dir data/raabin"
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
        # Class names match the actual folder names in the downloaded dataset exactly
        # (all lowercase, as shipped by EBI BioStudies S-BSST2156)
        # Sorted alphabetically — this is the order LocalImageFolderDataset assigns
        # integer labels 0–9 when class_names is passed explicitly.
        #   0=artefact, 1=basophil, 2=blast, 3=eosinophil, 4=erythroblast,
        #   5=ig (immature granulocyte), 6=lymphocyte, 7=monocyte,
        #   8=neutrophil, 9=platelet
        "class_names":  [
            "artefact",       # imaging artefact / non-cellular
            "basophil",       # BAS
            "blast",          # leukemic blast cells (ALL/AML)
            "eosinophil",     # EOS
            "erythroblast",   # EBO — immature red blood cell
            "ig",             # immature granulocyte (promyelocyte/myelocyte/metamyelocyte)
            "lymphocyte",     # LYT
            "monocyte",       # MON
            "neutrophil",     # NEU
            "platelet",       # PLT / thrombocyte
        ],
        "mean":         [0.485, 0.456, 0.406],  # ImageNet defaults — recompute with compute_dataset_stats() if needed
        "std":          [0.229, 0.224, 0.225],
        "source":       "local",    # pass --data_dir data/cytodata (relative to project root)
        "task":         "classification",
        "hf_splits":    None,
        "notes": (
            "559,808 single-cell images from Addenbrooke's Hospital, Cambridge. "
            "Labeled subset: 4,996 images across 10 classes (train/val/test pre-split). "
            "Source: EBI BioStudies S-BSST2156. "
            "Local path: data/cytodata/ (relative to early-leukemia-detection/). "
            "Run with: python run_all.py --dataset cytodata --data_dir data/cytodata"
        ),
    },

    # C-NMC 2019 — binary leukemia detection dataset from the ISBI 2019 challenge.
    # Source: HuggingFace dwb2023/cnmc-leukemia-2019
    # Based on data collected at Tata Medical Center, Kolkata, India.
    # Two classes: ALL (Acute Lymphoblastic Leukemia blast cells) vs. HEM (healthy cells).
    # This is the only dataset in the pipeline with genuine leukemic blast cells,
    # making it the primary dataset for validating the clinical leukemia detection claim.
    #
    # Split strategy: HuggingFace ships train + test only (no validation split).
    # The loader automatically carves 20% of train as validation (stratified, seed=42).
    "cnmc": {
        "display_name": "C-NMC 2019",
        "hf_name":      "dwb2023/cnmc-leukemia-2019",
        "num_classes":  2,
        # Label 0 = ALL (leukemic blast cells), Label 1 = HEM (healthy/normal)
        # These match the integer label order in the HuggingFace dataset.
        "class_names":  ["all", "hem"],
        "mean":         [0.485, 0.456, 0.406],  # ImageNet defaults — recompute if needed
        "std":          [0.229, 0.224, 0.225],
        "source":       "huggingface",
        "task":         "binary_classification",  # leukemia (ALL) vs. healthy (HEM)
        # HuggingFace ships train + test only; val is carved from train at load time.
        "hf_splits":    {"train": "train", "val": None, "test": "test"},
        "val_fraction": 0.20,   # fraction of train used as validation
        "notes": (
            "10,661 single-cell images from 73 ALL patients + healthy donors. "
            "Source: C-NMC 2019 ISBI Challenge / Tata Medical Center, Kolkata. "
            "Classes: ALL (7,272 blast cells) vs. HEM (3,389 healthy cells). "
            "Class imbalance ~68/32 — monitor per-class sensitivity during evaluation. "
            "HuggingFace: dwb2023/cnmc-leukemia-2019 (fully public, no login required). "
            "Run with: python run_all.py --dataset cnmc"
        ),
    },

}

# Convenience: valid dataset keys
DATASET_KEYS = list(DATASET_CONFIGS.keys())   # ["raabin", "bccd", "cytodata", "cnmc"]
