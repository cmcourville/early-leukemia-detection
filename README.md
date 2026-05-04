# Early Leukemia Detection — CS 534 Team 6, WPI

Comparative study of three deep learning approaches for early leukemia detection via white blood cell (WBC) classification. All three models are evaluated on a shared benchmark under full-data and low-data (n-shot) regimes.

---

## Project Status (as of April 30, 2026)

> **Phase 3 is in progress.** Slides + video due **May 5, 2026**. Live presentation + demo **May 6, 2026 (6–9 PM)**.

### Implementation Status

| Component | Owner | Status | Notes |
|-----------|-------|--------|-------|
| BccT model code | Corrin | ✅ Complete | All scripts implemented and tested |
| Shared infrastructure (config, metrics, data_loader, run_all.py) | Corrin | ✅ Complete | Supports raabin, bccd, cnmc, cytodata |
| Raabin-WBC local dataset | Corrin | ✅ Available | `data/raabin/` — run `prepare_raabin.py` to split |
| CytoData local dataset | Corrin | ✅ Available | `data/cytodata/` — 3,500 / 494 / 1,000 train/val/test, 10 classes |
| CytoDiffusion model code | Darshan | ✅ Complete | `models/cytodiffusion/` — implemented and integrated into pipeline |
| ViT-CNN Ensemble model code | Sean | ❌ Stub only | `models/vitcnn_ensemble/` — raises NotImplementedError |

### Experiment Results — Current State

#### Raabin-WBC 5-class (Primary Benchmark) — ✅ COMPLETE

Results at `shared/results/raabin/bcct/`.

| Metric | BccT on Raabin-WBC (Test) |
|--------|---------------------------|
| Accuracy | **0.9583** |
| Macro F1 | **0.9449** |
| Macro AUROC | **0.9949** |
| Macro Sensitivity | **0.9461** |
| Macro Specificity | **0.9899** |

Low-data (n-shot) results — `shared/results/raabin/bcct/low_data_summary.json`:

| Setting | Acc | F1 | AUROC | Sensitivity |
|---------|-----|----|-------|-------------|
| 10-shot | 0.7518 | 0.7226 | 0.9421 | 0.7516 |
| 20-shot | 0.8537 | 0.8195 | 0.9671 | 0.8278 |
| 50-shot | 0.8845 | 0.8496 | 0.9756 | 0.8585 |

#### CytoData 10-class (Extended Benchmark) — ✅ COMPLETE

Full train/val/test evaluation done. All low-data experiments (10/20/50-shot) complete. Results at `shared/results/cytodata/bcct/`.

| Metric | BccT on CytoData (Test) |
|--------|-------------------------|
| Accuracy | **0.6680** |
| Macro F1 | **0.5721** |
| Macro AUROC | **0.9153** |
| Macro Sensitivity | **0.5514** |
| Macro Specificity | **0.9604** |

Low-data (n-shot) results — `shared/results/cytodata/bcct/low_data_summary.json`:

| Setting | Acc | F1 | AUROC | Sensitivity |
|---------|-----|----|-------|-------------|
| 10-shot | 0.5250 | 0.4492 | 0.8651 | 0.5667 |
| 20-shot | 0.5900 | 0.5273 | 0.8972 | 0.6350 |
| 50-shot | **0.6023** | **0.5452** | **0.9020** | **0.6477** |

> **Note:** Lower CytoData accuracy vs. Raabin-WBC reflects the much harder 10-class problem (vs. 5-class),
> the small dataset size (3,500 train images across 10 classes), and severe class imbalance
> (e.g., only 20 blast cells in the test set). AUROC remains high (0.9153) because BccT's ViT
> features are discriminative even when the ELM head struggles with imbalanced classes.

#### BCCD 3-class (Supplementary) — ✅ COMPLETE (Phase 2)

Results at `shared/results/bccd/bcct/`. These are from Phase 2 and are retained for reference.

| Metric | BccT on BCCD (Test) |
|--------|---------------------|
| Accuracy | **0.9568** |
| Macro F1 | **0.8865** |
| Macro AUROC | **0.9669** |
| Macro Sensitivity | **0.9073** |
| Macro Specificity | **0.9651** |

### Remaining Steps (Corrin)

Listed in priority order (🔴 = critical path for Phase 3):

**1. ✅ ~~Run BccT 50-shot on CytoData~~ — COMPLETE (April 17)**

50-shot results are in `shared/results/cytodata/bcct/low_data/50shot/avg_metrics.json` and the full `low_data_summary.json` (10/20/50-shot) has been rebuilt. **All of Corrin's experimental work is now complete.**

**2. ✅ ~~Build Phase 3 slides (Corrin's sections)~~ — COMPLETE (April 18)**

8-slide standalone deck generated at `corrin_phase3_slides.pptx`. Insert into team deck when Darshan and Sean complete their sections.

Slides included:
- Slide 1: Section divider (BccT section header)
- Slide 2: BccT method — Token Fusion + Fixed-Random Classifier
- Slide 3: Pipeline / Architecture diagram (uses `bcct_workflow_diagram.png`)
- Slide 4: BccT results — Raabin-WBC (with per-class table, confusion matrix, AUROC)
- Slide 5: BccT results — CytoData 10-class (with per-class table, key finding callout)
- Slide 6: Low-data (n-shot) performance — both benchmarks side-by-side
- Slide 7: Conclusions — 3-column blocks (Raabin performance, CytoData/blast cells, low-data)
- Slide 8: Future Work — 5 numbered items (ablations, class imbalance, domain-adapted backbones, clinical validation, whole-slide analysis)

Source text content adapted from `corrin_sections_draft.md` (Section 7, April 15 draft).

**3. ✅ ~~Write missing final paper sections~~ — COMPLETE (April 21)**

All content flagged for grade deductions in Phase 1 and Phase 2 feedback has been drafted and saved to `paper_additions_draft.md`. Ready to paste into the team Google Doc.

Content included:
- **Abstract** (≤ 200 words): purpose, approach, key BccT results, summary finding
- **Keywords** (10 terms): leukemia detection, WBC classification, Vision Transformer, few-shot learning, etc.
- **Target demographic paragraph**: pediatric/adolescent ALL primary; young adult secondary; 5-year survival context
- **Global epidemiology** (2 paragraphs): Globocan 2020 stats, region-by-region survival disparities, LMICs context, citations
- **3 real-world examples** with published citations:
  - Neurocognitive late effects (Krull et al., 2013 — 40%+ survivors with cognitive impairment)
  - Cardiotoxicity from anthracyclines (van der Pal et al., 2012 — 57% abnormal cardiac parameters at 23yr)
  - Secondary malignancy after HSCT (Bhatia et al., 2007 — 15× increased risk, >20% cumulative at 30yr)
- **Disadvantages for all 3 SOTA methods**: BccT (frozen backbone, class imbalance sensitivity, memory), CytoDiffusion (GPU overhead, indirect classification signal), ViT-CNN Ensemble (dual-model cost, calibration, overfitting)
- **11 new APA references** (all 2007–2021, published journals)
- **Formatting checklist** (margins, font, page count, consistency items)

**4. 🔴 Cross-model comparison once Sean's model is ready**

CytoDiffusion is now fully integrated. Once Sean uncomments his registry entry in `run_all.py`:

```bash
python run_all.py --dataset raabin --data_dir data/raabin --mode full_pipeline
python run_all.py --dataset cytodata --data_dir data/cytodata --mode full_pipeline
```

After cross-model results are available, update slides 4–6 with the comparison table rows and add a dedicated cross-model comparison slide.

**4. 🟠 Prepare working demo for May 6**

Show `run_all.py` running end-to-end, metrics printing, confusion matrix plots appearing.
Have screen recordings as backup if live demo fails.

**5. 🟠 Record presentation video (≥15 min, camera on)**

Corrin covers: BccT method + results, Pipeline/Architecture, Conclusions & Future Work.
Upload to YouTube (unlisted) or Google Drive (shareable link).

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| Apr 30, 2026 | Corrin | Integrated CytoDiffusion into `run_all.py` pipeline. Changes: `run_train()` now passes `val_loader`, dataset-correct `img_mean`/`img_std`, and `cache_dir` to each model constructor; uses `inspect.signature` to pass `val_loader` only to models that declare it (backward-compatible with BccT). CytoDiffusion registry constructor updated to forward those kwargs. `CytoDiffusionModel.__init__` now accepts `cache_dir` and passes it to `AutoencoderKL.from_pretrained()`. README updated: CytoDiffusion marked ✅ Complete, added How CytoDiffusion Works section, updated Adding Your Model to Sean-only. |
| Apr 21, 2026 | Corrin | Wrote all missing final paper sections flagged for professor grade deductions — saved to `paper_additions_draft.md`. Includes: Abstract (≤200 words), Keywords (10 terms), target demographic + global epidemiology with citations, 3 real-world outcome examples (Krull 2013 neurocognitive, van der Pal 2012 cardiotoxicity, Bhatia 2007 secondary malignancy), disadvantages for all 3 SOTA methods, 11 new APA references, and a formatting checklist. README updated: marked step 3 ✅ COMPLETE, renumbered remaining steps. |
| Apr 18, 2026 | Corrin | Built Phase 3 slides for Corrin's sections — 8-slide PPTX at `corrin_phase3_slides.pptx`. Covers BccT method (Token Fusion + FRC), pipeline diagram, Raabin-WBC results, CytoData results, low-data n-shot performance, conclusions, and future work. All metrics drawn from completed experiment results in `shared/results/`. Marked Build Phase 3 slides ✅ COMPLETE in Remaining Steps. |
| Apr 17, 2026 | Corrin | CytoData 50-shot experiment confirmed complete — results were already in `shared/results/cytodata/bcct/low_data/50shot/avg_metrics.json` (Acc=0.6023, F1=0.5452, AUROC=0.9020, Sens=0.6477, 3 repeats). `low_data_summary.json` already rebuilt with all three shot counts. Updated README: marked CytoData benchmark ✅ COMPLETE, filled in 50-shot row in results table, revised Remaining Steps to remove the completed experiment. All of Corrin's experimental work is now done. |
| Apr 15, 2026 | Corrin | Updated project status to reflect CytoData full-dataset and 10/20-shot results complete; 50-shot pending environment. Added Section 7 (Conclusions & Future Work) draft to `corrin_sections_draft.md` — ready to paste into Phase 3 report. Noted that `run_cytodata_50shot.py` is ready to execute and only requires a PyTorch environment. |
| Apr 14, 2026 | Corrin | Completed CytoData 10-shot and 20-shot low-data experiments. Added `run_cytodata_50shot.py` targeted script for the final missing experiment. Updated README with CytoData results and remaining steps. |
| Apr 5, 2026 | Corrin | Phase 2 submission. BccT fully implemented. Raabin-WBC primary benchmark complete (95.83% acc). All shared infrastructure finalized. |

---

## Team & Models

| Teammate | Model | Status | Directory |
|----------|-------|--------|-----------|
| Corrin | **BccT** — Blood Cell Classification Transformer (Zhu et al. 2026) | ✅ Implemented | `models/bcct/` |
| Darshan | **CytoDiffusion** — Latent Diffusion-based classifier | ✅ Implemented | `models/cytodiffusion/` |
| Sean | **ViT-CNN Ensemble** — Hybrid Vision Transformer + CNN | 🔲 Stub | `models/vitcnn_ensemble/` |

---

## Repository Structure

```
early-leukemia-detection/
│
├── run_all.py                        # Unified pipeline — trains, evaluates, and compares all models
├── prepare_raabin.py                 # One-time setup: splits data/raabin/ into train/val/test
├── prepare_cnmc.py                   # One-time setup: organizes downloaded CNMC BMP files into train/val/test
├── prepare_bccd.py                   # One-time setup: downloads BCCD from GitHub and crops bboxes
│
├── data/                             # Local datasets (gitignored)
│   ├── raabin/                       # Acevedo et al. 2020, 5-class WBC (run prepare_raabin.py first)
│   │   ├── basophil/                 # Flat class folders (pre-split source)
│   │   ├── eosinophil/
│   │   ├── lymphocyte/
│   │   ├── monocyte/
│   │   ├── neutrophil/
│   │   ├── train/                    # Created by prepare_raabin.py
│   │   ├── val/
│   │   └── test/
│   └── cytodata/                     # Addenbrooke's Hospital, 10-class (pre-split)
│       ├── train/
│       ├── val/
│       └── test/
│
├── shared/                           # Single source of truth — used by ALL three models
│   ├── config.py                     # Global constants (dataset configs, class names, seeds, paths)
│   ├── metrics.py                    # Evaluation functions (accuracy, F1, AUROC, sensitivity/specificity)
│   ├── requirements.txt              # Shared Python dependencies
│   ├── results/                      # All evaluation outputs land here (gitignored)
│   └── data/
│       └── data_loader.py            # Unified DataLoader for all four datasets
│
└── models/
    ├── bcct/                         # BccT (Corrin) — fully implemented
    │   ├── bcct_model.py             # Top-level model: ViT backbone + Token Fusion + FRC
    │   ├── token_fusion.py           # Bipartite soft matching + size-weighted merge + log-z bias
    │   ├── fixed_random_classifier.py # ELM-style head, pseudo-inverse training
    │   ├── data_loader.py            # BccT data loading (delegates to shared/data/data_loader.py)
    │   ├── train_bcct.py             # Training script (standalone)
    │   ├── evaluate_bcct.py          # Evaluation script (metrics + confusion matrix + AUROC plots)
    │   ├── low_data_test.py          # N-shot / low-data regime experiments
    │   ├── main.py                   # BccT CLI entry point (train / evaluate / low_data / full_pipeline)
    │   ├── requirements.txt          # BccT-specific dependencies
    │   └── checkpoints/              # Saved model weights (gitignored)
    │
    ├── cytodiffusion/                # CytoDiffusion (Darshan) — fully implemented
    │   ├── model.py                  # SD VAE encoder + MLP head; train_model / forward / save / load
    │   ├── train.py                  # Standalone training script
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

## Datasets

Four datasets are supported. Pass `--dataset <key>` to any entry point.

### Raabin-WBC (`--dataset raabin --data_dir data/raabin`) — Primary Benchmark

**Source:** Acevedo et al. (2020), *A dataset for microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief.
Download: [Mendeley Data — snkd93bnjr](https://data.mendeley.com/datasets/snkd93bnjr/1) (`PBC_dataset_normal_DIB.zip`)

**Setup (one-time):**
```bash
# After unzipping PBC_dataset_normal_DIB.zip to data/raabin/:
python prepare_raabin.py
```

**Details:** 10,299 images across 5 WBC classes after filtering to the relevant classes from the full 8-class zip. The split script applies a stratified 70/10/20 split (seed=42) producing approximately 7,209 / 1,030 / 2,060 train/val/test images.

- Classes: `basophil`, `eosinophil`, `lymphocyte`, `monocyte`, `neutrophil`
- Normalisation: mean `[0.7442, 0.6384, 0.7516]`, std `[0.1580, 0.1914, 0.1225]`
- Layout after `prepare_raabin.py`: `data/raabin/{train,val,test}/<classname>/`

> **Note:** The original HuggingFace dataset (`polejowska/lcbsi-wbc-ap`) that this project was initially configured for has been deleted. The Acevedo et al. dataset is its underlying source with identical class structure.

---

### BCCD (`--dataset bccd --data_dir data/bccd`) — Supplementary Benchmark

**Source:** [Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset) on GitHub — auto-downloaded by the prepare script, no login required.

**Setup (one-time):**
```bash
python prepare_bccd.py --data_dir data/bccd
```
The script downloads the repo (~30 MB), parses Pascal VOC XML annotations, crops each bounding box, and saves the crops in ImageFolder format. Takes under a minute.

**Details:** 364 microscopy images with Platelet/RBC/WBC bounding-box annotations. Each crop becomes one classification sample (~3,700 crops total after splitting).

- Classes: `Platelet`, `RBC`, `WBC`
- Normalisation: ImageNet defaults — mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Layout after prepare: `data/bccd/{train,val,test}/<classname>/`

---

### CytoData (`--dataset cytodata --data_dir data/cytodata`) — Extended Benchmark

**Source:** Addenbrooke's Hospital, Cambridge (EBI BioStudies S-BSST2156). Not publicly available on HuggingFace — request access via [CambridgeCIA/CytoDiffusion](https://github.com/CambridgeCIA/CytoDiffusion).

**Details:** 4,994-image labeled subset from a 559,808-image single-cell morphology study. Already split into train/val/test and present at `data/cytodata/`.

- Classes (10): `artefact`, `basophil`, `blast`, `eosinophil`, `erythroblast`, `ig`, `lymphocyte`, `monocyte`, `neutrophil`, `platelet`
- Normalisation: ImageNet defaults — mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Split: 3,500 train / 494 val / 1,000 test
- Layout: `data/cytodata/{train,val,test}/<classname>/` (already structured correctly)

> **Clinically relevant** — the `blast` class contains leukemic blast cells, making CytoData the most directly relevant dataset to the project's leukemia detection goal.

---

### C-NMC 2019 (`--dataset cnmc --data_dir data/cnmc`) — Binary Leukemia Detection

**Source:** Kaggle [`avk256/cnmc-leukemia`](https://www.kaggle.com/datasets/avk256/cnmc-leukemia) — requires a free Kaggle account.

**Setup (one-time):**
```bash
# 1. Install the Kaggle CLI and authenticate
pip install kaggle
# Place kaggle.json from kaggle.com/settings → API in ~/.kaggle/kaggle.json

# 2. Download and unzip (~600 MB)
kaggle datasets download -d avk256/cnmc-leukemia --path data/cnmc_raw --unzip

# 3. Organize into ImageFolder format (70/10/20 split, seed=42)
python prepare_cnmc.py --src_dir data/cnmc_raw --data_dir data/cnmc
```

**Details:** 10,661 single-cell BMP images from 73 ALL patients and healthy donors. The only dataset with genuine leukemic blast cells from a clinical challenge (ISBI 2019, Tata Medical Center, Kolkata). The HuggingFace mirror (`dwb2023/cnmc-leukemia-2019`) does not include full image data and cannot be used for streaming — local files are required.

- Classes: `all` (Acute Lymphoblastic Leukemia blasts), `hem` (healthy cells)
- Class imbalance: ~68% ALL / 32% HEM — monitor per-class sensitivity (recall on ALL is the key clinical metric)
- Normalisation: ImageNet defaults — mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Layout after prepare: `data/cnmc/{train,val,test}/{all,hem}/`

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd early-leukemia-detection

# 2. Install dependencies (includes CytoDiffusion SD VAE requirements)
pip install -r shared/requirements.txt
pip install -r models/bcct/requirements.txt

# 3. Raabin-WBC — download PBC_dataset_normal_DIB.zip from Mendeley:
#    https://data.mendeley.com/datasets/snkd93bnjr/1
#    Unzip so that data/raabin/basophil/, data/raabin/eosinophil/, etc. exist, then:
python prepare_raabin.py

# 4. BCCD — auto-downloads from GitHub (~30 MB):
python prepare_bccd.py --data_dir data/bccd

# 5. C-NMC 2019 — requires a free Kaggle account:
#    a. Download kaggle.json from kaggle.com/settings → API → Create Token
#    b. Place it at ~/.kaggle/kaggle.json  (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)
pip install kaggle
kaggle datasets download -d avk256/cnmc-leukemia --path data/cnmc_raw --unzip
python prepare_cnmc.py --src_dir data/cnmc_raw --data_dir data/cnmc
```

**Core dependencies:** `torch>=2.1`, `torchvision>=0.16`, `transformers>=4.38`, `datasets>=2.18,<3`, `diffusers>=0.27`, `accelerate>=0.27`, `safetensors>=0.4`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `Pillow>=9.5`, `tqdm>=4.65`

---

## How to Run

All commands are run from the project root (`early-leukemia-detection/`).

### Run all three datasets in one command

```bash
# Trains + evaluates both models on raabin, cnmc, and bccd sequentially (~4 hrs on GPU)
python run_all.py --models bcct cytodiffusion vitcnn_ensemble--dataset all --device cuda

# On Colab/cloud with data on Drive, use --data_root to point at the Drive folder:
python run_all.py --models bcct cytodiffusion vitcnn_ensemble --dataset all --data_root /path/to/data/root --device cuda
```

`--dataset all` expects the following folders inside the data root:
- `raabin/` — prepared by `prepare_raabin.py`
- `cnmc/` — prepared by `prepare_cnmc.py`
- `bccd/` — prepared by `prepare_bccd.py`

### Run a single dataset

```bash
python run_all.py --dataset raabin   --data_dir data/raabin --mode full_pipeline
python run_all.py --dataset cnmc     --data_dir data/cnmc   --mode full_pipeline
python run_all.py --dataset bccd     --data_dir data/bccd   --mode full_pipeline
python run_all.py --dataset cytodata --data_dir data/cytodata --mode full_pipeline
```

---

### Option 1 — Unified Pipeline (recommended)

`run_all.py` loads the dataset once, runs each active model through train → evaluate → low-data experiments, and prints a side-by-side comparison table at the end.

```bash
# Full pipeline on Raabin-WBC
python run_all.py --dataset raabin --data_dir data/raabin

# Train only
python run_all.py --dataset raabin --data_dir data/raabin --mode train

# Evaluate only (requires saved checkpoint)
python run_all.py --dataset raabin --data_dir data/raabin --mode evaluate

# Low-data / n-shot experiments only
python run_all.py --dataset raabin --data_dir data/raabin --mode low_data

# Run a single model
python run_all.py --dataset raabin --data_dir data/raabin --models bcct

# Common hardware overrides
python run_all.py --dataset raabin --data_dir data/raabin --batch_size 64 --num_workers 8 --device cuda
```

**All flags for `run_all.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `full_pipeline` | `train`, `evaluate`, `low_data`, or `full_pipeline` |
| `--models` | all active | Space-separated model keys, e.g. `bcct` |
| `--dataset` | `raabin` | `raabin`, `bccd`, `cytodata`, or `cnmc` |
| `--data_dir` | `None` | Local dataset root — **required for `raabin`, `cnmc`, `bccd`, and `cytodata`** |
| `--data_root` | `data/` | Base directory used by `--dataset all` (overrides default `<project>/data/`) |
| `--batch_size` | `32` | DataLoader batch size |
| `--num_workers` | `4` | DataLoader worker processes |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--cache_dir` | `None` | HuggingFace model/dataset cache directory |
| `--shots` | `10 20 50` | Shot counts for low-data experiments |
| `--num_repeats` | `3` | Independent repeats per shot count |
| `--seed` | `42` | Global random seed |
| `--output_dir` | `shared/results` | Root directory for all output files |

---

### Option 2 — BccT Standalone

`models/bcct/main.py` is the dedicated BccT entry point with four sub-commands.

```bash
# Train on Raabin-WBC (primary benchmark)
python models/bcct/main.py train --dataset raabin --data_dir data/raabin

# Train on BCCD (no data_dir needed — downloads from HuggingFace)
python models/bcct/main.py train --dataset bccd

# Train on CytoData
python models/bcct/main.py train --dataset cytodata --data_dir data/cytodata

# Train with custom BccT hyperparameters
python models/bcct/main.py train --dataset raabin --data_dir data/raabin \
    --r 16 --ridge_lambda 1e-4 --batch_size 32 --output_dir ./results

# Evaluate a saved checkpoint on the test set
python models/bcct/main.py evaluate --checkpoint checkpoints/bcct_model.pt \
    --dataset raabin --data_dir data/raabin --split test

# Run low-data experiments (10/20/50-shot)
python models/bcct/main.py low_data --dataset raabin --data_dir data/raabin \
    --shots 10 20 50 --num_repeats 3

# Full pipeline in one command
python models/bcct/main.py full_pipeline --dataset raabin --data_dir data/raabin
```

**BccT-specific flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `raabin` | `raabin`, `bccd`, `cytodata`, or `cnmc` |
| `--data_dir` | `None` | Local dataset root — required for `raabin` and `cytodata` |
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

Token Fusion is inserted **after the FFN** of each of the 12 ViT blocks. It progressively reduces sequence length so the final CLS token encodes a richer summary. The algorithm per block:

**Step 1 — Partition** (Eq. 3): Patch tokens are split into alternating sets A (even indices) and B (odd indices). The [CLS] token is never touched.

**Step 2 — Head-averaged keys** (Eq. 4a): Key projections from all 12 attention heads are averaged: K̄ = (1/H) Σₕ Kₕ, giving a 64-dim per-token key.

**Step 3 — Bipartite soft matching** (Eq. 4b): For each A-token, cosine similarity to every B-token is computed. The top r=16 (A, B) pairs with highest similarity are selected for merging.

**Step 4 — Size-weighted merge** (Eq. 11/12): Each selected pair merges into the B-token: X_merged = (zᵢXᵢ + zⱼXⱼ) / (zᵢ + zⱼ). The A-token is dropped. Each token carries a size counter z (starts at 1) tracking how many original patches it represents.

**Step 5 — Log-z attention compensation** (Eq. 5): In subsequent blocks, log(z) is added column-wise to the attention logits before softmax, ensuring merged (larger) tokens receive proportionally more attention weight.

After 12 blocks of merging 16 pairs each, the sequence reduces from 197 tokens to at most 6 tokens (5 patch + 1 CLS). Because HuggingFace's ViT doesn't expose intermediate attention scores, log-z injection requires monkey-patching `ViTSelfAttention.forward` directly.

### 3. Fixed-Random Classifier (Section 3.3, Zhu et al. 2026)

The FRC head takes the final CLS token (768-dim) and classifies it. It is an Extreme Learning Machine (ELM)-style architecture:

- **Hidden layer**: H = ReLU(x W^T + b), where W ∈ ℝ^{3072×768} and b ∈ ℝ^{3072} are drawn from N(0,1) at init and **permanently frozen** as `register_buffer` entries.
- **Skip link**: Raw 768-dim input is concatenated with H: O = [H ; x] ∈ ℝ^{3840}.
- **Output weights**: F ∈ ℝ^{num_classes×3840} is solved analytically in one call: F = lstsq(O, Y_one_hot) with ridge λ=1e-4.

**There are zero trainable parameters.** Training is a single forward pass through the dataset followed by a LAPACK least-squares solve. The number of classes is set automatically from the dataset (5 for Raabin-WBC, 3 for BCCD, 10 for CytoData, 2 for C-NMC).

### Training Procedure (end-to-end)

1. Load frozen ViT backbone and install Token Fusion hooks.
2. Forward-pass the entire training set. For each batch, collect the CLS token from the final encoder layer.
3. Stack all CLS tokens into an (N, 768) feature matrix.
4. Pass through the FRC hidden layer to get an (N, 3840) matrix O.
5. Solve F = lstsq(O, Y_one_hot) via `torch.linalg.lstsq`.
6. Save only the FRC buffers (W, b, F) as a `.pt` checkpoint. The backbone is always reloaded from HuggingFace at inference time.

---

## Understanding the Pipeline Output

When `run_all.py` finishes, it prints a summary and writes files to `shared/results/`. Here is what each output means:

### Terminal Output

```
[shared/data] Raabin — train:7209, val:1030, test:2060, classes:{0: 'basophil', ...}
```
Confirms the dataset loaded correctly with the expected split sizes and class mapping.

```
[BccT] Extracting features — train (7209 samples) ...
[BccT] Feature matrix: (7209, 3840)
[BccT] Solving FRC (ridge_lambda=0.0001) ...
[BccT] Training complete in 42.3s
```
BccT feature extraction and closed-form solve. No gradient steps — this is the entire "training".

```
[BccT] Test  — Acc: 0.9341  Macro-F1: 0.9289  AUROC: 0.9912
```
Final test-set metrics. Key numbers for the Phase 3 comparison table.

```
[BccT] 10-shot (seed=42): Acc=0.7821  F1=0.7654  AUROC=0.9234  Sens=0.7812
[BccT] 20-shot (seed=42): Acc=0.8243  F1=0.8102  ...
[BccT] 50-shot (seed=42): Acc=0.8891  F1=0.8834  ...
```
Low-data regime results. Each line is averaged over `--num_repeats` (default 3) independent seeds.

```
====== Model Comparison — raabin ======
Metric          BccT      CytoDiffusion  ViT-CNN
Accuracy        0.9341    —              —
Macro F1        0.9289    —              —
...
```
The cross-model comparison table. Dashes appear for models whose stubs raised `NotImplementedError`. This table is the core Phase 3 deliverable — it will fill in once teammates implement their models.

### Output Files

```
shared/results/
├── bcct/
│   ├── metrics_validation.json       # All metrics on the validation split
│   ├── metrics_test.json             # All metrics on the test split
│   ├── confusion_matrix_test.png     # Count + normalised confusion matrix heatmap
│   ├── auroc_test.png                # Per-class and macro ROC curves
│   ├── low_data_summary.json         # Averaged n-shot metrics (10/20/50-shot)
│   └── low_data/
│       ├── 10shot/avg_metrics.json   # Per-repeat breakdown for 10-shot
│       ├── 20shot/avg_metrics.json
│       └── 50shot/avg_metrics.json
├── cytodiffusion/                    # Populated by CytoDiffusion runs
├── vitcnn_ensemble/                  # Populated when Sean's model is ready
└── comparison_table.txt              # Side-by-side ASCII table (all models)
```

**metrics_test.json** contains:
- `accuracy` — fraction of test images classified correctly
- `macro_f1` — unweighted average F1 across all classes (primary summary metric)
- `per_class_f1` — F1 per class; low values flag which cell types are hardest
- `macro_auroc` — area under the ROC curve, macro-averaged (One-vs-Rest); insensitive to class imbalance
- `macro_sensitivity` — true positive rate averaged across classes (recall)
- `macro_specificity` — true negative rate averaged across classes
- `confusion_matrix` — raw counts; rows = true label, cols = predicted label
- `n_samples` — total test images evaluated

**low_data_summary.json** shows how much accuracy degrades when only n examples per class are available for training — BccT's ELM head re-fits in milliseconds, so this is fast to compute. Strong low-data performance (high accuracy at 10-shot) demonstrates the quality of the frozen ViT features.

**comparison_table.txt** is the primary deliverable for Phase 3 — a formatted table comparing all three models on the same dataset under the same conditions.

---

## Shared Infrastructure

The `shared/` directory enforces experimental consistency across all three models.

**`shared/config.py`** — single source of truth for all constants: dataset configs, class names, normalisation stats, seed, shot counts, result/checkpoint paths, and backbone ID. All models import from here. Four datasets are configured: `raabin`, `bccd`, `cytodata`, `cnmc`.

**`shared/data/data_loader.py`** — unified DataLoader for all four datasets. Provides:
- `get_dataloaders(dataset, data_dir, ...)` — full train/val/test loaders with standardised augmentation (RandomHorizontalFlip + ColorJitter for training; Resize 256 → CenterCrop 224 for eval).
- `get_few_shot_loaders(n_shot, dataset, seed, ...)` — stratified n-shot sampling. Same seed → same samples across models, ensuring fair low-data comparison.
- Dataset classes: `HFImageDataset` (HuggingFace), `BCCDCellDataset` (detection→crops), `LocalImageFolderDataset` (Raabin, CytoData).

**`shared/metrics.py`** — computes and saves all evaluation metrics:
- Accuracy, macro F1, per-class F1
- AUROC (One-vs-Rest, macro and per-class), sensitivity, specificity
- Confusion matrix (counts + normalised heatmap PNG)
- ROC curve plots (PNG)
- `format_comparison_table()` — ASCII side-by-side table across all models

---

## How CytoDiffusion Works

CytoDiffusion uses the VAE encoder from a pretrained Stable Diffusion model as a fixed feature extractor, then trains a lightweight MLP head on top via standard cross-entropy.

### Architecture

```
Input (B, 3, 224, 224) — dataset-normalised
  → denormalize to [0,1] → remap to [-1,1]
  → SD VAE encoder (runwayml/stable-diffusion-v1-5, frozen by default)
  → latent (B, 4, 28, 28)
  → AdaptiveAvgPool2d(8, 8) → flatten → (B, 256)
  → Linear(256→512) → GELU → Dropout(0.3)
  → Linear(512→256) → GELU → Dropout(0.15)
  → Linear(256→num_classes)
  → logits (B, num_classes)
```

The VAE's perceptual bottleneck separates cell morphology in latent space. The 256-dim pooled representation is compact enough that a 3-layer MLP reaches strong accuracy without requiring full diffusion model training.

### Training Procedure

1. Load the SD VAE encoder from HuggingFace. Freeze all encoder parameters (default).
2. For each epoch, forward-pass each batch: denormalize → remap to [-1,1] → encode → pool → MLP → cross-entropy loss with label smoothing (0.1).
3. Optimise with AdamW (`lr=1e-3`, `weight_decay=1e-4`) and cosine annealing (`T_max=epochs`, `eta_min=lr×0.01`).
4. Gradient clipping (`max_norm=1.0`) applied each step.
5. The MLP head is re-initialised from scratch at the start of each `train_model()` call — this ensures low-data / few-shot experiments start clean.
6. Checkpoint saves only the pool + classifier weights when the encoder is frozen (~2 MB). Encoder weights are always reloaded from HuggingFace (or local cache) at `load()` time.

### Key Differences from BccT

| Property | BccT | CytoDiffusion |
|----------|------|---------------|
| Backbone | ViT-Base (language-pretrained) | SD VAE (image-generation pretrained) |
| Training | Closed-form (no backprop) | Gradient-based (30 epochs) |
| Feature dim | 768 (CLS token) | 256 (pooled latent) |
| Checkpoint size | ~60 MB | ~2 MB (frozen encoder) |
| GPU required | No (CPU feasible) | Recommended (VAE encode on CPU is slow) |

### Running CytoDiffusion

```bash
# Train + evaluate + low-data on Raabin-WBC
python run_all.py --models cytodiffusion --dataset raabin --data_dir data/raabin

# Train only
python run_all.py --models cytodiffusion --dataset raabin --data_dir data/raabin --mode train

# Evaluate from checkpoint
python run_all.py --models cytodiffusion --dataset raabin --data_dir data/raabin --mode evaluate

# All three primary datasets sequentially
python run_all.py --models cytodiffusion --dataset all

# Specify a HuggingFace cache dir (avoids re-downloading the ~4GB VAE weights)
python run_all.py --models cytodiffusion --dataset raabin --data_dir data/raabin --cache_dir ~/.cache/huggingface
```

**First run** will download the SD VAE weights (~4 GB) from HuggingFace. Subsequent runs use the local cache. Set `--cache_dir` to control where they land.

---

## Adding Your Model (Sean)

CytoDiffusion is already integrated. For ViT-CNN Ensemble:

1. Implement `model.py`, `train.py`, and `evaluate.py` in `models/vitcnn_ensemble/`. The interface each model must satisfy: `__init__`, `train_model(loader, device)`, `forward(x)`, `save(path)`, `load(path, device)`.
2. Use `shared/data/data_loader.py` for data loading and `shared/metrics.py` for evaluation — this keeps all results directly comparable.
3. In `run_all.py`, uncomment the `vitcnn_ensemble` import line and registry entry (both marked `# UNCOMMENT WHEN READY`).
4. Add your dependencies to `models/vitcnn_ensemble/requirements.txt`.
5. Test in isolation first: `python run_all.py --models vitcnn_ensemble --dataset bccd --mode train`

---

## Citations

Zhu et al., *BccT: Blood Cell Classification Transformer for Early Leukemia Detection*, Multimedia Systems, 2026. DOI: 10.1007/s00530-025-02085-w

Acevedo et al., *A dataset for microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief, 2020. DOI: 10.17632/snkd93bnjr.1

Bolya et al., *Token Merging: Your ViT But Faster*, ICLR 2023.

Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.

Matek et al., *A Single-Cell Morphological Dataset of Leukocytes from AML Patients and Non-Malignant Controls* (CytoData / AML-Cytomorphology), The Cancer Imaging Archive, 2019.
