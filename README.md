# early-leukemia-detection
Repository Structure & Nomenclature

This repository is organized to support reproducible and fair comparison of multiple deep learning approaches for early leukemia detection using blood smear images.

The structure separates shared infrastructure from model-specific implementations to ensure experimental consistency across methods.

⸻

Top-Level Directory Overview
.
├── data/
├── experiments/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── LICENSE

Directory Descriptions

data/

Contains dataset metadata and deterministic dataset splits.
Raw medical images are not committed to this repository.

data/
├── metadata/      # Label maps, dataset info, auxiliary JSON files
├── splits/        # Train/val/test CSV split definitions
└── README.md      # Dataset source and download instructions

Nomenclature
	•	splits/ contains fixed dataset partitions to ensure fair model comparison.
	•	metadata/ stores small structured files such as label mappings and dataset descriptors.
	•	Raw images should be stored locally (e.g., data/raw/) and excluded via .gitignore.

experiments/

Stores experiment configurations and outputs to ensure reproducibility.
experiments/
└── <experiment_name>/
    └── <method_name>/

Each experiment directory should contain:
	•	config.yaml
	•	metrics.json
	•	predictions.csv
	•	notes.md

Nomenclature
	•	<experiment_name> examples:
	•	baseline
	•	low_data
	•	domain_shift
	•	ablations
	•	<method_name>:
	•	bcct
	•	cyto_diffusion
	•	hybrid_cnn_vit

src/

Contains all project source code.
src/
├── common/
├── methods/
└── runners/
The src/ directory enforces separation between shared infrastructure and model implementations.

src/common/

Shared components used across all methods to guarantee consistency.
common/
├── config/     # YAML experiment configuration files
├── data/       # Dataset loaders and transforms
├── metrics/    # Evaluation metrics (AUROC, F1, etc.)
└── utils/      # Logging, seeding, device management

Purpose

All models must use:
	•	The same preprocessing pipeline
	•	The same train/validation/test splits
	•	The same evaluation metrics

This ensures academically valid and fair comparisons.

src/methods/

Contains independent implementations of each evaluated model.
methods/
├── bcct/
├── cyto_diffusion/
└── hybrid_cnn_vit/