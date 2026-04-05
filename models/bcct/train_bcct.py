"""
train_bcct.py — BccT Project (CS 534, Team 6)
Author: Corrin

Training script for the BccT model.

BccT's "training" is fundamentally different from gradient-based methods:
  1. Forward-pass all training images through the frozen ViT+Token Fusion
     backbone to accumulate CLS feature vectors.  (No backprop.)
  2. Call FixedRandomClassifier.fit() which solves a single closed-form
     pseudo-inverse system: F = O⁺Y  (or ridge variant).

This script handles:
  • CLI argument parsing (--epochs is accepted for interface compatibility
    but has no effect since there is no iterative optimisation).
  • Checkpoint saving (FRC weights only — backbone is always reloaded from HF).
  • Optional evaluation on the validation set immediately after training.
  • Timing and progress logging.

Usage:
    python train_bcct.py                        # default settings
    python train_bcct.py --r 16 --batch 32      # explicit args
    python train_bcct.py --cache_dir ./hf_cache # custom HF cache
    python train_bcct.py --output_dir ./ckpts   # save checkpoint
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch

from bcct_model import BccTModel
from data_loader import get_dataloaders
from evaluate_bcct import evaluate_model   # imported here to run val after train

# Argument parser

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BccT Blood Cell Classification Transformer."
    )

    # Data
    parser.add_argument(
        "--dataset", type=str, default="raabin",
        choices=["raabin", "bccd", "cytodata"],
        help="Dataset to train on (default: raabin)."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Local dataset root — required when --dataset cytodata."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Mini-batch size for feature collection (default: 32)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace dataset/model cache directory."
    )

    # Model
    parser.add_argument(
        "--r", type=int, default=16,
        help="Token Fusion merge budget per block (paper default: 16)."
    )
    parser.add_argument(
        "--d_hidden", type=int, default=None,
        help="FRC hidden-layer width (default: 4 × 768 = 3072)."
    )
    parser.add_argument(
        "--ridge_lambda", type=float, default=1e-4,
        help="Ridge regularisation λ for FRC pseudo-inverse (default: 1e-4)."
    )
    parser.add_argument(
        "--frc_seed", type=int, default=42,
        help="Random seed for FRC weight init (default: 42)."
    )
    parser.add_argument(
        "--pretrained_id", type=str,
        default="google/vit-base-patch16-224-in21k",
        help="HuggingFace ViT backbone model ID."
    )

    # Training control (kept for interface compatibility)
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="[Ignored] BccT trains in a single pseudo-inverse solve, not epochs."
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints",
        help="Directory to save model checkpoint (default: ./checkpoints)."
    )
    parser.add_argument(
        "--run_val", action="store_true", default=True,
        help="Evaluate on validation set after training (default: True)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', or 'cuda:N'. Auto-detected if None."
    )

    return parser.parse_args()

# Main training function

def train(args: Optional[argparse.Namespace] = None) -> BccTModel:
    """
    Execute BccT training.  Can be called programmatically or via CLI.

    Returns the trained BccTModel instance.
    """
    if args is None:
        args = get_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Data
    print(f"[train] Loading dataset: {args.dataset}")
    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        dataset=args.dataset,
        data_dir=getattr(args, "data_dir", None),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[train] Label map: {label_map}")

    # Model
    model = BccTModel(
        num_classes=len(label_map),
        r=args.r,
        d_hidden=args.d_hidden,
        ridge_lambda=args.ridge_lambda,
        pretrained_id=args.pretrained_id,
        frc_seed=args.frc_seed,
    )

    # Training (one-shot pseudo-inverse)
    t0 = time.time()
    model.train_model(train_loader, device=device)
    elapsed = time.time() - t0
    print(f"[train] Training wall-clock time: {elapsed:.1f}s")

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "bcct_model.pt"
    model.save(str(ckpt_path))

    # Optional validation eval
    if args.run_val:
        print("\n[train] Running validation evaluation …")
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            split_name="Validation",
            label_map=label_map,
        )
        print(f"[train] Validation Accuracy : {val_metrics['accuracy']:.4f}")
        print(f"[train] Validation Macro-F1 : {val_metrics['f1_macro']:.4f}")

    return model

# Entry point

if __name__ == "__main__":
    trained_model = train()
    print("\n[train] Done.")
