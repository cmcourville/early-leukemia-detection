"""
run_all.py — CS 534, Team 6 — WPI
Unified pipeline to train, evaluate, and compare all three SOTA models.

Team:
    Corrin  → BccT (Blood Cell Classification Transformer)
    Darshan → CytoDiffusion / LDM
    Sean    → ViT-CNN Ensemble

Usage:
    python run_all.py                          # full pipeline, all modes
    python run_all.py --mode train             # train only
    python run_all.py --mode evaluate          # evaluate only (needs checkpoints)
    python run_all.py --mode low_data          # n-shot experiments only
    python run_all.py --models bcct            # run one model only
    python run_all.py --batch_size 16 --device cuda:0

When a teammate finishes their implementation, uncomment the relevant block
below (marked UNCOMMENT WHEN READY) to include it in the pipeline.

Outputs (written to shared/results/):
    shared/results/
    ├── bcct/
    │   ├── metrics_test.json
    │   ├── confusion_matrix_test.png
    │   ├── auroc_test.png
    │   └── low_data/
    │       ├── 10shot/ ...
    │       ├── 20shot/ ...
    │       └── 50shot/ ...
    ├── cytodiffusion/          # populated once Darshan's model is ready
    ├── vitcnn_ensemble/        # populated once Sean's model is ready
    └── comparison_table.txt    # side-by-side metric comparison
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

# Path setup — make shared/ and models/* importable from any CWD

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "bcct"))
sys.path.insert(0, str(PROJECT_ROOT / "models"))          # cytodiffusion + vitcnn_ensemble
sys.path.insert(0, str(PROJECT_ROOT / "models" / "vitcnn_ensemble")) # Active

from shared.config import (
    LOW_DATA_SHOTS, LOW_DATA_REPEATS,
    GLOBAL_SEED, RESULTS_ROOT, DATASET_CONFIGS,
)
from shared.data.data_loader import get_dataloaders, get_few_shot_loaders
from shared.metrics import (
    compute_all_metrics, print_metrics, save_metrics,
    save_confusion_matrix_plot, save_auroc_plot, format_comparison_table,
)

# Model imports

# BccT (Corrin) — ACTIVE
from bcct_model import BccTModel
from token_fusion import reset_token_sizes

# CytoDiffusion (Darshan)
from cytodiffusion.model import CytoDiffusionModel

# ViT-CNN Ensemble (Sean) — NOT YET IMPLEMENTED
# UNCOMMENT WHEN READY:
from vitcnn_ensemble.model import ViTCNNEnsemble

# Model registry
# Each entry describes how to construct, train, and checkpoint a model.
# To activate a model, uncomment its entry and the import above.

MODEL_REGISTRY = {

    "bcct": {
        "name":        "BccT (Corrin)",
        "active":      True,     # ← Corrin's model is fully implemented
        "constructor": lambda num_classes, **kw: BccTModel(
            num_classes=num_classes,
            r=kw.get("r", 16),
            ridge_lambda=kw.get("ridge_lambda", 1e-4),
        ),
        "checkpoint_path": "models/bcct/checkpoints/bcct_model.pt",
        "save_fn":  lambda model, path: model.save(path),
        "load_fn":  lambda path, device: BccTModel.load(path, device),
    },

    "cytodiffusion": {
        "name":        "CytoDiffusion (Darshan)",
        "active":      True,
        "constructor": lambda num_classes, **kw: CytoDiffusionModel(
            num_classes=num_classes,
            cache_dir=kw.get("cache_dir"),
            img_mean=kw.get("img_mean"),
            img_std=kw.get("img_std"),
        ),
        "checkpoint_path": "models/cytodiffusion/checkpoints/cytodiffusion_model.pt",
        "save_fn":  lambda model, path: model.save(path),
        "load_fn":  lambda path, device: CytoDiffusionModel.load(path, device),
    },

     #ViT-CNN Ensemble (Sean) — UNCOMMENT WHEN READY
     "vitcnn_ensemble": {
         "name":        "ViT-CNN Ensemble (Sean)",
         "active":      True,
         "constructor": lambda num_classes, **kw: ViTCNNEnsemble(
             num_classes=num_classes,
             cnn_logits=True,
         ),
         "checkpoint_path": "models/vitcnn_ensemble/checkpoints/vitcnn_ensemble_model.pt",
         "save_fn":  lambda model, path: model.save(path),
         "load_fn":  lambda path, device: ViTCNNEnsemble.load(path, device),
     },
}

# Argument parsing

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CS 534 Team 6 — Unified model pipeline."
    )
    parser.add_argument(
        "--mode", type=str, default="full_pipeline",
        choices=["train", "evaluate", "low_data", "full_pipeline"],
        help="Pipeline mode (default: full_pipeline = train + evaluate + low_data).",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        metavar="MODEL_KEY",
        help=(
            "Which models to run (space-separated). "
            "Choices: bcct cytodiffusion vitcnn_ensemble. "
            "Default: all active models in the registry."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default="raabin",
        choices=["raabin", "bccd", "cytodata", "cnmc", "all"],
        help=(
            "Dataset to use for all models (default: raabin). "
            "Pass 'all' to run sequentially over raabin, cnmc, and cytodata."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=(
            "Local dataset root — required when --dataset cytodata. "
            "Default for 'all' mode: data/cytodata (relative to project root)."
        ),
    )
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--cache_dir",   type=str,   default=None,
                        help="HuggingFace cache directory.")
    parser.add_argument("--hf_token",    type=str,   default=None,
                        help=(
                            "HuggingFace API token for gated datasets (e.g. Raabin-WBC). "
                            "If omitted, the cached token from `huggingface-cli login` is used."
                        ))
    parser.add_argument("--device",      type=str,   default=None,
                        help="Compute device (auto-detected if omitted).")
    parser.add_argument("--shots",       type=int,   nargs="+",
                        default=LOW_DATA_SHOTS,
                        help="Shot counts for low-data experiments.")
    parser.add_argument("--num_repeats", type=int,   default=LOW_DATA_REPEATS)
    parser.add_argument("--seed",        type=int,   default=GLOBAL_SEED)
    parser.add_argument("--output_dir",  type=str,   default=RESULTS_ROOT,
                        help="Root directory for all result outputs.")
    parser.add_argument("--data_root",   type=str,   default=None,
                        help=(
                            "Override base data directory used by --dataset all. "
                            "Defaults to <project_root>/data/. "
                            "Useful in Colab when data lives on Drive: "
                            "--data_root /content/drive/MyDrive/.../CS534-Group6"
                        ))
    return parser.parse_args()

# Per-model pipeline steps

@torch.no_grad()
def _collect_predictions(model, loader, device) -> tuple:
    """Run inference on `loader` and return (y_true, y_pred, y_prob)."""
    import torch.nn.functional as F

    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        logits = model(imgs)                            # (B, C)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())

    return (
        np.array(all_true, dtype=int),
        np.array(all_pred, dtype=int),
        np.array(all_prob, dtype=float),
    )

def run_train(
    model_key: str,
    entry:     Dict,
    args:      argparse.Namespace,
    device:    torch.device,
    train_loader,
    val_loader,
    label_map: Dict,
) -> object:
    """Train a single model and return the trained model instance."""
    print(f"\n{'='*64}")
    print(f"  TRAINING: {entry['name']}")
    print(f"{'='*64}")

    num_classes = DATASET_CONFIGS[args.dataset]["num_classes"]
    model = entry["constructor"](num_classes=num_classes)
    model.to(device)

    t0 = time.time()
    sig = inspect.signature(model.train_model)
    if "val_loader" in sig.parameters:
        model.train_model(train_loader, device=device, val_loader=val_loader)
    else:
        model.train_model(train_loader, device=device)
    elapsed = time.time() - t0
    print(f"[pipeline] Training complete in {elapsed:.1f}s")

    # Save checkpoint
    ckpt_path = PROJECT_ROOT / entry["checkpoint_path"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    entry["save_fn"](model, str(ckpt_path))

    return model

def run_evaluate(
    model_key:  str,
    entry:      Dict,
    model,
    device:     torch.device,
    loader,
    split_name: str,
    class_names: List[str],
    out_dir:    Path,
) -> Dict:
    """Evaluate a model on `loader` and save results."""
    print(f"\n[pipeline] Evaluating {entry['name']} on {split_name} …")

    y_true, y_pred, y_prob = _collect_predictions(model, loader, device)

    metrics = compute_all_metrics(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        class_names=class_names,
        model_name=entry["name"],
        split_name=split_name,
    )
    print_metrics(metrics)

    # Persist
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(out_dir / f"metrics_{split_name.lower()}.json"))
    save_confusion_matrix_plot(
        cm=metrics["confusion_matrix"],
        class_names=class_names,
        model_name=entry["name"],
        split_name=split_name,
        out_path=str(out_dir / f"confusion_matrix_{split_name.lower()}.png"),
    )
    save_auroc_plot(
        y_true=y_true, y_prob=y_prob,
        class_names=class_names,
        model_name=entry["name"],
        split_name=split_name,
        out_path=str(out_dir / f"auroc_{split_name.lower()}.png"),
    )

    return metrics

def run_low_data(
    model_key:   str,
    entry:       Dict,
    model,
    device:      torch.device,
    test_loader,
    class_names: List[str],
    shots:       List[int],
    num_repeats: int,
    seed:        int,
    out_dir:     Path,
    cache_dir:   Optional[str],
    batch_size:  int,
    num_workers: int,
    dataset:     str = "raabin",
    data_dir:    Optional[str] = None,
    hf_token:    Optional[str] = None,
) -> Dict:
    """Run n-shot experiments for a single model."""
    summary = {}

    for n_shot in sorted(shots):
        repeat_metrics = []

        for rep in range(num_repeats):
            rep_seed = seed + rep
            print(f"\n[pipeline] {entry['name']} — {n_shot}-shot "
                  f"(repeat {rep+1}/{num_repeats}, seed={rep_seed})")

            # Build few-shot train loader
            few_train_dl, _, _, _ = get_few_shot_loaders(
                n_shot=n_shot,
                dataset=dataset,
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                cache_dir=cache_dir,
                seed=rep_seed,
                hf_token=hf_token,
            )

            # Re-collect CLS features and re-fit FRC (BccT-specific)
            # Generic interface: model.train_model(loader, device)
            model.train_model(few_train_dl, device=device)

            # Evaluate on full test set
            y_true, y_pred, y_prob = _collect_predictions(model, test_loader, device)
            m = compute_all_metrics(
                y_true=y_true, y_pred=y_pred, y_prob=y_prob,
                class_names=class_names,
                model_name=entry["name"],
                split_name=f"Test_{n_shot}shot_rep{rep+1}",
            )
            repeat_metrics.append(m)

        # Average across repeats
        avg = {
            "accuracy_mean":          float(np.mean([m["accuracy"]          for m in repeat_metrics])),
            "f1_macro_mean":          float(np.mean([m["f1_macro"]          for m in repeat_metrics])),
            "auroc_macro_mean":       float(np.mean([m["auroc_macro"]       for m in repeat_metrics])),
            "sensitivity_macro_mean": float(np.mean([m["sensitivity_macro"] for m in repeat_metrics])),
            "num_repeats":            num_repeats,
        }
        summary[f"{n_shot}-shot"] = avg

        shot_dir = out_dir / "low_data" / f"{n_shot}shot"
        shot_dir.mkdir(parents=True, exist_ok=True)
        with open(shot_dir / "avg_metrics.json", "w") as f:
            json.dump(avg, f, indent=2)
        print(f"[pipeline] {n_shot}-shot avg — Acc={avg['accuracy_mean']:.4f}  "
              f"F1={avg['f1_macro_mean']:.4f}  "
              f"AUROC={avg['auroc_macro_mean']:.4f}")

    # Summary JSON
    with open(out_dir / "low_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# Main pipeline

def main():
    args   = get_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 64)
    print("  CS 534 Team 6 — Unified Model Pipeline")
    print(f"  Mode   : {args.mode}")
    print(f"  Device : {device}")
    print("=" * 64)

    # --dataset all → run sequentially over the three primary datasets
    if args.dataset == "all":
        data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
        PRIMARY_DATASETS = [
            ("raabin", str(data_root / "raabin")),
            ("cnmc",   str(data_root / "cnmc")),
            ("bccd",   str(data_root / "bccd")),
        ]
        print("\n[pipeline] --dataset all: running raabin, cnmc, bccd sequentially.")
        print(f"[pipeline] data_root: {data_root}")
        for ds_name, ds_dir in PRIMARY_DATASETS:
            print(f"\n{'#'*64}")
            print(f"#  DATASET: {ds_name.upper()}")
            print(f"{'#'*64}")
            args.dataset  = ds_name
            args.data_dir = ds_dir
            # Recurse into the single-dataset path with updated args
            _run_single_dataset(args, device)
        print("\n[pipeline] All datasets complete.")
        return

    _run_single_dataset(args, device)


def _run_single_dataset(args: argparse.Namespace, device: torch.device) -> None:
    """Run the full pipeline for a single dataset (extracted so --dataset all can loop)."""
    # Determine which models to run
    requested = set(args.models) if args.models else None
    active_models = {
        k: v for k, v in MODEL_REGISTRY.items()
        if v["active"] and (requested is None or k in requested)
    }

    if not active_models:
        print("[pipeline] No active models to run.  "
              "Check MODEL_REGISTRY or --models argument.")
        return

    print(f"\n[pipeline] Models to run: {[v['name'] for v in active_models.values()]}")

    # Load dataset once — shared across all models
    print(f"\n[pipeline] Loading dataset: {args.dataset} …")
    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=(device.type == "cuda"),
        hf_token=getattr(args, "hf_token", None),
    )
    class_names = [label_map[i] for i in sorted(label_map.keys())]

    # Results go into a dataset-specific subdirectory so runs don't overwrite each other
    # e.g. shared/results/raabin/bcct/  or  shared/results/cnmc/bcct/
    out_root = PROJECT_ROOT / args.output_dir / args.dataset

    all_test_metrics: Dict[str, Dict] = {}
    model_cache: Dict[str, object]    = {}

    # Per-model execution
    for model_key, entry in active_models.items():
        model_out_dir = out_root / model_key
        model_out_dir.mkdir(parents=True, exist_ok=True)

        model = None

        try:
            # TRAIN
            if args.mode in ("train", "full_pipeline"):
                model = run_train(
                    model_key=model_key,
                    entry=entry,
                    args=args,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    label_map=label_map,
                )
                model_cache[model_key] = model

            # EVALUATE
            if args.mode in ("evaluate", "full_pipeline"):
                if model is None:
                    # Load from checkpoint for evaluate-only mode
                    ckpt_path = PROJECT_ROOT / entry["checkpoint_path"]
                    if not ckpt_path.exists():
                        print(f"[pipeline] WARNING: checkpoint not found at "
                              f"{ckpt_path} — skipping {entry['name']}.")
                        continue
                    model = entry["load_fn"](str(ckpt_path), device)

                # Validation set
                run_evaluate(
                    model_key=model_key, entry=entry, model=model,
                    device=device, loader=val_loader,
                    split_name="Validation", class_names=class_names,
                    out_dir=model_out_dir,
                )

                # Test set — stored for comparison table
                test_metrics = run_evaluate(
                    model_key=model_key, entry=entry, model=model,
                    device=device, loader=test_loader,
                    split_name="Test", class_names=class_names,
                    out_dir=model_out_dir,
                )
                all_test_metrics[model_key] = test_metrics

            # LOW-DATA
            if args.mode in ("low_data", "full_pipeline"):
                if model is None:
                    ckpt_path = PROJECT_ROOT / entry["checkpoint_path"]
                    if not ckpt_path.exists():
                        print(f"[pipeline] WARNING: checkpoint not found at "
                              f"{ckpt_path} — skipping {entry['name']} low-data.")
                        continue
                    model = entry["load_fn"](str(ckpt_path), device)

                run_low_data(
                    model_key=model_key, entry=entry, model=model,
                    device=device, test_loader=test_loader,
                    class_names=class_names,
                    shots=args.shots, num_repeats=args.num_repeats,
                    seed=args.seed, out_dir=model_out_dir,
                    cache_dir=args.cache_dir, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    dataset=args.dataset, data_dir=args.data_dir,
                    hf_token=getattr(args, "hf_token", None),
                )

        except Exception as exc:
            print(f"\n[pipeline] ERROR in {entry['name']}: {exc}")
            traceback.print_exc()
            print(f"[pipeline] Continuing with remaining models …\n")
            continue

    # Cross-model comparison table
    if all_test_metrics and len(all_test_metrics) > 0:
        table = format_comparison_table(all_test_metrics)
        print("\n" + "=" * 64)
        print(f"  Cross-Model Comparison — {args.dataset.upper()} (Test Set)")
        print("=" * 64)
        print(table)

        table_path = out_root / "comparison_table.txt"
        table_path.write_text(table)
        print(f"\n[pipeline] Comparison table saved → {table_path}")

    print(f"\n[pipeline] {args.dataset.upper()} complete.")
    print(f"[pipeline] Results written to: {out_root.resolve()}")

if __name__ == "__main__":
    main()
