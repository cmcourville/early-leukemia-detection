# Early Leukemia Detection — CS 534 Team 6, WPI

Comparative study of three deep learning approaches for early leukemia detection via white blood cell (WBC) classification. All three models are evaluated on the Raabin-WBC dataset under full-data and low-data (n-shot) regimes.

---

## Team & Models

| Teammate | Model | Status | Directory |
|----------|-------|--------|-----------|
| Corrin | **BccT** — Blood Cell Classification Transformer (Zhu et al. 2026) | ✅ Implemented | `models/bcct/` |
| Darshan | **CytoDiffusion** — Latent Diffusion-based classifier | 🔲 Stub | `models/cytodiffusion/` |
| Sean | **ViT-CNN Ensemble** — Hybrid Vision Transformer + CNN | 🔲 Stub | `models/vitcnn_ensemble/` |

---

## Repository Structure

```
early-leukemia-detection/
│
├── run_all.py                        # Unified pipeline — trains, evaluates, and compares all models
│
├── shared/                           # Single source of truth — used by ALL three models
│   ├── config.py                     # Global constants (dataset, class names, seeds, paths)
│   ├── metrics.py                    # Evaluation functions (accuracy, F1, AUROC, sensitivity/specificity)
│   ├── requirements.txt              # Shared Python dependencies
│   ├── results/                      # All evaluation outputs land here (gitignored)
│   └── data/
│       └── data_loader.py            # HuggingFace → PyTorch DataLoader bridge (shared)
│
└── models/
    ├── bcct/                         # BccT (Corrin) — fully implemented
    │   ├── bcct_model.py             # Top-level model: ViT backbone + Token Fusion + FRC
    │   ├── token_fusion.py           # Bipartite soft matching + size-weighted merge + log-z bias
    │   ├── fixed_random_classifier.py # ELM-style head, pseudo-inverse training
    │   ├── data_loader.py            # BccT-specific data loading and augmentation
    │   ├── train_bcct.py             # Training script (standalone)
    │   ├── evaluate_bcct.py          # Evaluation script (metrics + confusion matrix + AUROC plots)
    │   ├── low_data_test.py          # N-shot / low-data regime experiments
    │   ├── main.py                   # BccT CLI entry point (train / evaluate / low_data / full_pipeline)
    │   ├── requirements.txt          # BccT-specific dependencies
    │   └── checkpoints/              # Saved model weights — gitignored
    │
    ├── cytodiffusion/                # CytoDiffusion (Darshan) — stubs only
    │   ├── model.py                  # Interface defined, raise NotImplementedError
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── requirements.txt
    │   └── checkpoints/
    │
    └── vitcnn_ensemble/              # ViT-CNN Ensemble (Sean) — stubs only
        ├── model.py
        ├── train.py
        ├── evaluate.py
        ├── requirements.txt
        └── checkpoints/
```

---

## Dataset

**Raabin-WBC** (`polejowska/lcbsi-wbc-ap` on HuggingFace)

5 white blood cell classes: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil.
Pre-split: 70% train / 10% validation / 20% test. All images resized to 224×224.
The dataset streams automatically on first run — no manual download needed.

Normalisation stats (pre-computed from the training split, stored in `shared/config.py`):
- Mean: `[0.7442, 0.6384, 0.7516]`
- Std: `[0.1580, 0.1914, 0.1225]`

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd early-leukemia-detection

# Install shared dependencies
pip install -r shared/requirements.txt

# Install BccT-specific dependencies
pip install -r models/bcct/requirements.txt

# When teammates finish their models, install those too:
# pip install -r models/cytodiffusion/requirements.txt
# pip install -r models/vitcnn_ensemble/requirements.txt
```

**Core dependencies:** `torch>=2.1`, `torchvision>=0.16`, `transformers>=4.38`, `datasets>=2.18`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `Pillow>=9.5`, `tqdm>=4.65`

---

## How to Run

All commands are run from the project root (`early-leukemia-detection/`).

### Option 1 — Unified Pipeline (recommended)

`run_all.py` loads the dataset once, runs each active model through train → evaluate → low-data experiments, and prints a side-by-side comparison table at the end.

```bash
# Full pipeline: train → evaluate val + test → n-shot experiments
python run_all.py

# Train only
python run_all.py --mode train

# Evaluate only (requires saved checkpoints)
python run_all.py --mode evaluate

# Low-data / n-shot experiments only
python run_all.py --mode low_data

# Run a single model
python run_all.py --models bcct

# Common overrides
python run_all.py --batch_size 64 --num_workers 8 --device cuda --seed 0
```

**All flags for `run_all.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `full_pipeline` | `train`, `evaluate`, `low_data`, or `full_pipeline` |
| `--models` | all active | Space-separated model keys, e.g. `bcct` |
| `--batch_size` | `32` | DataLoader batch size |
| `--num_workers` | `4` | DataLoader worker processes |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--cache_dir` | `None` | HuggingFace model/dataset cache directory |
| `--shots` | `10 20 50` | Shot counts for low-data experiments |
| `--num_repeats` | `3` | Independent repeats per shot count |
| `--seed` | `42` | Global random seed |
| `--output_dir` | `shared/results` | Root directory for all output files |

### Option 2 — BccT Standalone

`models/bcct/main.py` is the dedicated BccT entry point with four sub-commands.

```bash
cd models/bcct

# Train on full dataset, save checkpoint
python main.py train

# Train with custom options
python main.py train --r 16 --ridge_lambda 1e-4 --batch_size 32 --output_dir ./results

# Evaluate a saved checkpoint on the test set
python main.py evaluate --checkpoint checkpoints/bcct_model.pt --split test

# Run low-data experiments (10/20/50-shot)
python main.py low_data --shots 10 20 50 --num_repeats 3

# Full pipeline in one command
python main.py full_pipeline --output_dir ./results
```

**BccT-specific flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--r` | `16` | Token Fusion merge budget per transformer block |
| `--ridge_lambda` | `1e-4` | Ridge regularisation λ for pseudo-inverse solve |
| `--d_hidden` | `3072` | FRC hidden layer width (default: 4 × 768) |
| `--pretrained_id` | `google/vit-base-patch16-224-in21k` | ViT backbone HuggingFace ID |
| `--frc_seed` | `42` | Seed for frozen random weight initialisation |
| `--checkpoint` | — | Path to `.pt` file for evaluate / low_data modes |
| `--split` | `test` | `val` or `test` for evaluate mode |

---

## How BccT Works

BccT is an efficient WBC classifier built on a frozen Vision Transformer backbone. It makes three core architectural choices that together eliminate gradient-based training entirely.

### 1. Frozen ViT-Base Backbone

BccT loads `google/vit-base-patch16-224-in21k` (12 transformer blocks, 768-dim hidden, 12 attention heads). All backbone parameters are immediately frozen — `requires_grad = False` throughout. A 224×224 input image is split into 196 non-overlapping 16×16 patches, giving a sequence of 197 tokens (196 patches + 1 [CLS] token).

### 2. Token Fusion (Section 3.2, Zhu et al. 2026)

Token Fusion is inserted **after the FFN** (feed-forward sub-layer) of each of the 12 ViT blocks. It progressively reduces the sequence length so the model is faster and the final CLS token encodes a richer summary. The algorithm per block:

**Step 1 — Partition** (Eq. 3): Patch tokens are split into alternating sets A (even indices) and B (odd indices). The [CLS] token is never touched.

**Step 2 — Head-averaged keys** (Eq. 4a): Inside each block's self-attention, key projections from all 12 heads are averaged: K̄ = (1/H) Σₕ Kₕ, giving a 64-dim per-token key regardless of head count.

**Step 3 — Bipartite soft matching** (Eq. 4b): For each A-token, its cosine similarity to every B-token is computed from K̄. The top r=16 (A, B) pairs with highest similarity are selected for merging.

**Step 4 — Size-weighted merge** (Eq. 11/12): Each selected pair is merged into the B-token using a weighted average: X_merged = (zᵢXᵢ + zⱼXⱼ) / (zᵢ + zⱼ). The A-token is then dropped. Each token carries a size counter z (starts at 1) that tracks how many original patches it represents.

**Step 5 — Log-z attention compensation** (Eq. 5): In subsequent blocks, before the softmax in self-attention, the log of each token's size is added column-wise to the attention logits: logits += log(z). This ensures merged (larger) tokens receive proportionally more attention weight.

Because HuggingFace's ViT does not expose intermediate attention scores via its hook API, the log-z injection requires monkey-patching `ViTSelfAttention.forward` directly — the same technique used by the ToMe (Token Merging) library (Bolya et al., ICLR 2023).

After 12 blocks of merging 16 pairs each, the sequence is reduced from 197 tokens to at most 197 − (12 × 16) = 5 patch tokens + 1 CLS = 6 tokens.

### 3. Fixed-Random Classifier (Section 3.3, Zhu et al. 2026)

The FRC head takes the final CLS token (768-dim) and classifies it into one of 5 WBC classes. It is an Extreme Learning Machine (ELM)-style architecture:

- **Hidden layer**: H = ReLU(x W^T + b), where W ∈ ℝ^{3072×768} and b ∈ ℝ^{3072} are drawn from N(0,1) at init and **permanently frozen**. They are stored as `register_buffer` (not parameters) so they move with the model to GPU but are never updated.
- **Skip/feedback link**: The raw 768-dim input is concatenated with H, giving a combined feature O = [H ; x] ∈ ℝ^{3840}.
- **Output weights**: F ∈ ℝ^{5×3840} starts as zeros and is solved analytically in one call: F = lstsq(O, Y) where O is the (N, 3840) feature matrix across all training samples and Y is the (N, 5) one-hot label matrix. Ridge augmentation (λ=1e-4) is applied for numerical stability.

**There are zero trainable parameters.** W, b, and F are all `register_buffer`. Training is a single forward pass through the dataset followed by a LAPACK least-squares solve.

### Training Procedure (end-to-end)

1. Load the frozen ViT backbone + install Token Fusion hooks.
2. Forward-pass the entire training set through the backbone. For each batch, collect the CLS token from the final encoder layer.
3. Stack all CLS tokens into an (N, 768) feature matrix.
4. Pass through the FRC hidden layer to get an (N, 3840) matrix O.
5. Solve F = lstsq(O, Y_one_hot) via `torch.linalg.lstsq` (LAPACK `gelsd` driver).
6. Save only the FRC buffers (W, b, F) as a `.pt` checkpoint. The backbone is always reloaded from HuggingFace.

---

## Shared Infrastructure

The `shared/` directory enforces experimental consistency across all three models.

**`shared/config.py`** — the single source of truth for all constants: dataset name, class names, normalisation stats, seed, shot counts, result/checkpoint paths, and backbone ID. All models import from here.

**`shared/data/data_loader.py`** — wraps the HuggingFace Raabin-WBC dataset into PyTorch DataLoaders. Provides:
- `get_dataloaders()` — full train/val/test loaders with standardised augmentation (RandomHorizontalFlip, ColorJitter for training; CenterCrop for eval).
- `get_few_shot_loaders(n_shot, seed)` — stratified n-shot sampling. Same seed → same samples, so low-data results are directly comparable across models.

**`shared/metrics.py`** — computes and formats all metrics required by the Phase 2 report:
- Accuracy, macro F1, per-class F1
- AUROC (One-vs-Rest, macro and per-class)
- Sensitivity (recall) and specificity per class and macro
- Confusion matrix (counts + normalised)
- `format_comparison_table()` — ASCII side-by-side table across all models

---

## Outputs

All outputs write to `shared/results/` (gitignored):

```
shared/results/
├── bcct/
│   ├── metrics_validation.json       # All metrics on the validation split
│   ├── metrics_test.json             # All metrics on the test split
│   ├── confusion_matrix_test.png     # Count + normalised confusion matrix
│   ├── auroc_test.png                # Per-class and macro AUROC curves
│   ├── low_data_summary.json         # Averaged metrics across repeats per shot count
│   └── low_data/
│       ├── 10shot/avg_metrics.json
│       ├── 20shot/avg_metrics.json
│       └── 50shot/avg_metrics.json
├── cytodiffusion/                    # Populated when Darshan's model is ready
├── vitcnn_ensemble/                  # Populated when Sean's model is ready
└── comparison_table.txt              # Side-by-side metric comparison (all models)
```

Checkpoints save to each model's `checkpoints/` folder:

```
models/bcct/checkpoints/bcct_model.pt
models/cytodiffusion/checkpoints/cytodiffusion_model.pt
models/vitcnn_ensemble/checkpoints/vitcnn_ensemble_model.pt
```

---

## Adding Your Model (Darshan / Sean)

1. Implement `model.py`, `train.py`, and `evaluate.py` in your model's directory. The stub files define the expected interface — each model needs `__init__`, `train_model(loader, device)`, `forward(x)`, `save(path)`, and `load(path, device)`.
2. Use `shared/data/data_loader.py` for data loading and `shared/metrics.py` for evaluation to keep results comparable.
3. In `run_all.py`, uncomment your model's import line and registry entry (both marked `# UNCOMMENT WHEN READY`).
4. Add your dependencies to your model's `requirements.txt`.
5. Test in isolation: `python run_all.py --models <your_key> --mode train`

---

## Citations

Zhu et al., *BccT: Blood Cell Classification Transformer for Early Leukemia Detection*, 2026.
Bolya et al., *Token Merging: Your ViT But Faster*, ICLR 2023.
Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
