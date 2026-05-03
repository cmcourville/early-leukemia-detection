"""
main.py — BccT Project (CS 534, Team 6)
Author: Corrin

Unified entry point for all BccT experiments.

Modes:
  train        — Train BccT on full Raabin-WBC training split.
  evaluate     — Evaluate a checkpoint on val / test split.
  low_data     — Run 10/20/50-shot low-data regime experiments.
  full_pipeline — Train + evaluate test set + run low-data tests in sequence.

Usage examples:
    python main.py train
    python main.py evaluate --checkpoint ./checkpoints/bcct_model.pt
    python main.py low_data --shots 10 20 50
    python main.py full_pipeline --output_dir ./results

All modes share the common flags below; run `python main.py <mode> --help`
for mode-specific options.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import torch

# Common arguments (shared by all modes)

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset", type=str, default="raabin",
        choices=["raabin", "bccd", "cytodata", "cnmc"],
        help="Dataset to use: 'raabin' (default), 'bccd', or 'cytodata'."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Local dataset root — required when --dataset cytodata."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', 'cuda:0' etc. Auto-detected if omitted."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for feature collection / evaluation."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker threads."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace cache directory for dataset and model weights."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Root output directory for checkpoints, metrics, and plots."
    )
    parser.add_argument(
        "--r", type=int, default=16,
        help="Token Fusion merge budget r (paper default: 16)."
    )
    parser.add_argument(
        "--ridge_lambda", type=float, default=1e-4,
        help="Ridge regularisation λ for FRC pseudo-inverse."
    )
    parser.add_argument(
        "--pretrained_id", type=str,
        default="google/vit-base-patch16-224-in21k",
        help="HuggingFace ViT backbone model ID."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed."
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help=(
            "HuggingFace API token for gated datasets (e.g. Raabin-WBC). "
            "If omitted, the cached token from `huggingface-cli login` is used."
        ),
    )

# Mode: train

def mode_train(args: argparse.Namespace) -> None:
    import train_bcct

    # Patch args to match train_bcct expectations
    args.d_hidden     = getattr(args, "d_hidden", None)
    args.frc_seed     = args.seed
    args.epochs       = 1
    args.run_val      = True

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(ckpt_dir)

    train_bcct.train(args)

# Mode: evaluate

def mode_evaluate(args: argparse.Namespace) -> None:
    from bcct_model import BccTModel
    from data_loader import get_dataloaders
    from evaluate_bcct import evaluate_model

    device = _resolve_device(args.device)

    if not hasattr(args, "checkpoint") or args.checkpoint is None:
        print("[main] ERROR: --checkpoint is required for evaluate mode.")
        sys.exit(1)

    model = BccTModel.load(args.checkpoint, device=device)

    train_dl, val_dl, test_dl, label_map = get_dataloaders(
        dataset=getattr(args, "dataset", "raabin"),
        data_dir=getattr(args, "data_dir", None),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        hf_token=getattr(args, "hf_token", None),
    )

    results_dir = Path(args.output_dir) / "eval"

    split = getattr(args, "split", "test")
    split_map = {
        "test":       (test_dl,  "Test"),
        "val":        (val_dl,   "Validation"),
        "validation": (val_dl,   "Validation"),
        "train":      (train_dl, "Train"),
    }
    loader, split_name = split_map.get(split, (test_dl, "Test"))

    evaluate_model(
        model=model,
        loader=loader,
        device=device,
        split_name=split_name,
        label_map=label_map,
        output_dir=str(results_dir),
        save_plots=True,
    )

# Mode: low_data

def mode_low_data(args: argparse.Namespace) -> None:
    import low_data_test

    args.shots        = getattr(args, "shots",        [10, 20, 50])
    args.num_repeats  = getattr(args, "num_repeats",  3)
    args.output_dir   = str(Path(args.output_dir) / "low_data")

    checkpoint = getattr(args, "checkpoint", None)
    args.checkpoint = checkpoint

    low_data_test.main.__wrapped__(args) if hasattr(low_data_test.main, "__wrapped__") \
        else _run_low_data(args)

def _run_low_data(args):
    """Direct call version of low_data_test.main using args namespace."""
    import low_data_test
    import json
    import time
    import numpy as np
    from pathlib import Path
    from bcct_model import BccTModel
    from data_loader import get_dataloaders

    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "checkpoint", None):
        model = BccTModel.load(args.checkpoint, device=device)
    else:
        model = BccTModel(
            r=args.r,
            ridge_lambda=args.ridge_lambda,
            pretrained_id=args.pretrained_id,
        )
        model.to(device)

    _, _, test_loader, label_map = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )

    shots = getattr(args, "shots", [10, 20, 50])
    num_repeats = getattr(args, "num_repeats", 3)
    summary = {}

    for n_shot in sorted(shots):
        repeat_metrics = []
        for rep in range(num_repeats):
            seed = args.seed + rep
            t0 = time.time()
            metrics = low_data_test.run_n_shot(
                n_shot=n_shot,
                model=model,
                test_loader=test_loader,
                device=device,
                label_map=label_map,
                output_dir=output_dir,
                seed=seed,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            elapsed = time.time() - t0
            print(f"[main] {n_shot}-shot rep {rep+1}/{num_repeats} in {elapsed:.1f}s")
            repeat_metrics.append(metrics)

        avg_acc   = float(np.mean([m["accuracy"]          for m in repeat_metrics]))
        avg_f1    = float(np.mean([m["f1_macro"]          for m in repeat_metrics]))
        avg_auroc = float(np.mean([m["auroc_macro"]        for m in repeat_metrics]))
        avg_sens  = float(np.mean([m["sensitivity_macro"]  for m in repeat_metrics]))

        summary[f"{n_shot}-shot"] = {
            "accuracy_mean": avg_acc,
            "f1_macro_mean": avg_f1,
            "auroc_macro_mean": avg_auroc,
            "sensitivity_macro_mean": avg_sens,
            "num_repeats": num_repeats,
        }

    print(f"\n{'='*60}")
    print("  Low-Data Regime Summary (BccT)")
    print(f"{'='*60}")
    for setting, m in summary.items():
        print(f"  {setting:<12}  Acc={m['accuracy_mean']:.4f}  "
              f"F1={m['f1_macro_mean']:.4f}  "
              f"AUROC={m['auroc_macro_mean']:.4f}  "
              f"Sens={m['sensitivity_macro_mean']:.4f}")

    with open(output_dir / "low_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[main] Low-data summary saved → {output_dir / 'low_data_summary.json'}")

# Mode: full_pipeline

def mode_full_pipeline(args: argparse.Namespace) -> None:
    """
    Full experimental pipeline:
      1. Train on full Raabin-WBC training split.
      2. Evaluate on val + test splits.
      3. Run 10/20/50-shot low-data experiments.
    """
    from pathlib import Path
    from bcct_model import BccTModel
    from data_loader import get_dataloaders
    from evaluate_bcct import evaluate_model
    import train_bcct

    device = _resolve_device(args.device)
    root   = Path(args.output_dir)

    print("\n" + "="*60)
    print("  BccT Full Pipeline — CS 534 Team 6")
    print("="*60)

    # Step 1: Train
    print("\n[Pipeline] Step 1: Training …")
    train_args = _copy_namespace(args)
    train_args.d_hidden    = None
    train_args.frc_seed    = args.seed
    train_args.epochs      = 1
    train_args.run_val     = False
    train_args.output_dir  = str(root / "checkpoints")

    model = train_bcct.train(train_args)
    ckpt_path = str(root / "checkpoints" / "bcct_model.pt")

    # Step 2: Evaluate val + test
    print("\n[Pipeline] Step 2: Evaluation …")
    train_dl, val_dl, test_dl, label_map = get_dataloaders(
        dataset=getattr(args, "dataset", "raabin"),
        data_dir=getattr(args, "data_dir", None),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        hf_token=getattr(args, "hf_token", None),
    )
    for loader, split_name in [(val_dl, "Validation"), (test_dl, "Test")]:
        evaluate_model(
            model=model,
            loader=loader,
            device=device,
            split_name=split_name,
            label_map=label_map,
            output_dir=str(root / "eval"),
            save_plots=True,
        )

    # Step 3: Low-data
    print("\n[Pipeline] Step 3: Low-data regime …")
    low_args = _copy_namespace(args)
    low_args.shots       = [10, 20, 50]
    low_args.num_repeats = 3
    low_args.checkpoint  = ckpt_path
    low_args.output_dir  = str(root / "low_data")
    _run_low_data(low_args)

    print("\n[Pipeline] All steps complete.")
    print(f"[Pipeline] Results directory: {root.resolve()}")

# Helpers

def _resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _copy_namespace(ns: argparse.Namespace) -> argparse.Namespace:
    import copy
    return copy.deepcopy(ns)

# Argument parsing

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bcct",
        description="BccT: Blood Cell Classification Transformer (Zhu et al. 2026)",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train BccT on Raabin-WBC.")
    add_common_args(p_train)
    p_train.add_argument("--d_hidden", type=int, default=None)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a BccT checkpoint.")
    add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "validation", "test"])

    # low_data
    p_low = subparsers.add_parser("low_data", help="Run n-shot low-data experiments.")
    add_common_args(p_low)
    p_low.add_argument("--checkpoint", type=str, default=None)
    p_low.add_argument("--shots", type=int, nargs="+", default=[10, 20, 50])
    p_low.add_argument("--num_repeats", type=int, default=3)

    # full_pipeline
    p_full = subparsers.add_parser("full_pipeline",
                                   help="Train + evaluate + low-data in one run.")
    add_common_args(p_full)

    return parser

# Entry point

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Set global seeds for reproducibility
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    mode_dispatch = {
        "train":         mode_train,
        "evaluate":      mode_evaluate,
        "low_data":      mode_low_data,
        "full_pipeline": mode_full_pipeline,
    }

    mode_fn = mode_dispatch.get(args.mode)
    if mode_fn is None:
        print(f"[main] Unknown mode: {args.mode}")
        sys.exit(1)

    mode_fn(args)

if __name__ == "__main__":
    main()
