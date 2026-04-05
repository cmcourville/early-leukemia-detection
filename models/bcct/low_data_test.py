"""
low_data_test.py — BccT Project (CS 534, Team 6)
Author: Corrin

Low-data regime evaluation for BccT.

Tests the model's ability to classify WBC types with severely limited
labelled training data — a key advantage of BccT's ELM-style training.

Regime tested: n ∈ {10, 20, 50} shots per class.
  - 5 WBC classes × 10 = 50 training samples total (10-shot)
  - 5 WBC classes × 20 = 100 training samples total (20-shot)
  - 5 WBC classes × 50 = 250 training samples total (50-shot)

For each n-shot configuration:
  1. Sample n images per class from the full training split (stratified).
  2. Extract CLS features from the frozen ViT+Token Fusion backbone.
  3. Fit the FRC pseudo-inverse on these n×C samples.
  4. Evaluate on the FULL test split (same as the full-data experiment).
  5. Report all metrics (Accuracy, Macro-F1, AUROC, Sensitivity).

Results are saved to JSON files in --output_dir for inclusion in the
Phase 2 report tables.

Usage:
    python low_data_test.py                        # runs 10/20/50-shot
    python low_data_test.py --shots 10 20          # custom shot counts
    python low_data_test.py --checkpoint ./ckpts/bcct_model.pt
    python low_data_test.py --output_dir ./results/low_data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from bcct_model import BccTModel
from data_loader import get_dataloaders, get_few_shot_loaders
from evaluate_bcct import evaluate_model

# Argument parser

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BccT low-data regime (n-shot) evaluation."
    )
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[10, 20, 50],
        help="List of n-shot counts to evaluate (default: 10 20 50)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to an existing BccTModel checkpoint (.pt).  "
            "If None, a fresh model is constructed using the pretrained ViT backbone."
        )
    )
    parser.add_argument(
        "--r", type=int, default=16,
        help="Token Fusion merge budget (used only if --checkpoint is None)."
    )
    parser.add_argument(
        "--ridge_lambda", type=float, default=1e-4,
        help="FRC ridge regularisation λ."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--num_workers", type=int, default=4
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/low_data",
        help="Directory to save per-shot metric JSON files."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shot sampling."
    )
    parser.add_argument(
        "--num_repeats", type=int, default=1,
        help=(
            "Number of independent random sampling repeats per shot count.  "
            "Use ≥3 for stable estimates; results are averaged across repeats."
        )
    )
    parser.add_argument(
        "--device", type=str, default=None
    )
    parser.add_argument(
        "--pretrained_id", type=str,
        default="google/vit-base-patch16-224-in21k"
    )
    return parser.parse_args()

# Single n-shot run

def run_n_shot(
    n_shot:       int,
    model:        BccTModel,
    test_loader,
    device:       torch.device,
    label_map:    Dict,
    output_dir:   Path,
    seed:         int = 42,
    cache_dir:    Optional[str] = None,
    batch_size:   int = 32,
    num_workers:  int = 4,
    save_plots:   bool = False,   # skip plots for low-data to keep output tidy
) -> Dict:
    """
    Train on n-shot subset and evaluate on the full test set.

    The FRC output matrix F is *re-fitted* for each n-shot run, but the
    backbone weights and FRC hidden weights (W, b) are never changed.

    Returns:
        Dictionary of evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"  {n_shot}-shot experiment  (seed={seed})")
    print(f"{'='*60}")

    # Build few-shot train loader
    few_train_dl, _, _, _ = get_few_shot_loaders(
        n_shot=n_shot,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        seed=seed,
    )

    # Collect CLS features from few-shot training set
    model.vit.eval()
    model.to(device)

    all_features = []
    all_labels   = []

    from token_fusion import reset_token_sizes

    with torch.no_grad():
        for imgs, labels in few_train_dl:
            imgs = imgs.to(device)
            reset_token_sizes(model.block_states)
            cls = model.extract_cls(imgs)
            all_features.append(cls.cpu())
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels   = torch.cat(all_labels,   dim=0)

    print(f"[low_data] {n_shot}-shot: collected {features.shape[0]} training features.")

    # Re-fit FRC on the few-shot feature matrix
    model.frc.cpu()
    model.frc.fit(features, labels)
    model.frc.to(device)

    # Evaluate on full test set
    split_label = f"Test_{n_shot}shot"
    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        split_name=split_label,
        label_map=label_map,
        output_dir=str(output_dir / f"{n_shot}shot"),
        save_plots=save_plots,
    )

    return metrics

# Main

def main():
    args = get_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[low_data] Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or build model
    if args.checkpoint:
        print(f"[low_data] Loading checkpoint: {args.checkpoint}")
        model = BccTModel.load(args.checkpoint, device=device)
    else:
        print("[low_data] Building fresh BccTModel …")
        model = BccTModel(
            r=args.r,
            ridge_lambda=args.ridge_lambda,
            pretrained_id=args.pretrained_id,
        )
        model.to(device)

    # Full test loader (reused across all n-shot configs)
    _, _, test_loader, label_map = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )

    # Summary table
    summary: Dict[str, Dict] = {}

    for n_shot in sorted(args.shots):
        repeat_metrics: List[Dict] = []

        for rep in range(args.num_repeats):
            seed = args.seed + rep   # different seed each repeat
            t0 = time.time()
            metrics = run_n_shot(
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
            print(f"[low_data] {n_shot}-shot repeat {rep+1}/{args.num_repeats} "
                  f"done in {elapsed:.1f}s")
            repeat_metrics.append(metrics)

        # Average across repeats (if num_repeats > 1)
        import numpy as np
        avg_acc   = float(np.mean([m["accuracy"]   for m in repeat_metrics]))
        avg_f1    = float(np.mean([m["f1_macro"]   for m in repeat_metrics]))
        avg_auroc = float(np.mean([m["auroc_macro"] for m in repeat_metrics]))
        avg_sens  = float(np.mean([m["sensitivity_macro"] for m in repeat_metrics]))

        summary[f"{n_shot}-shot"] = {
            "accuracy_mean":         avg_acc,
            "f1_macro_mean":         avg_f1,
            "auroc_macro_mean":      avg_auroc,
            "sensitivity_macro_mean": avg_sens,
            "num_repeats":           args.num_repeats,
        }

    # Print summary table
    print(f"\n{'='*60}")
    print("  Low-Data Regime Summary (BccT)")
    print(f"{'='*60}")
    header = f"{'Setting':<12} {'Acc':>8} {'F1':>8} {'AUROC':>8} {'Sens':>8}"
    print(header)
    print("-" * len(header))
    for setting, m in summary.items():
        print(f"{setting:<12} "
              f"{m['accuracy_mean']:>8.4f} "
              f"{m['f1_macro_mean']:>8.4f} "
              f"{m['auroc_macro_mean']:>8.4f} "
              f"{m['sensitivity_macro_mean']:>8.4f}")
    print("=" * 60)

    # Save summary JSON
    summary_path = output_dir / "low_data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[low_data] Summary saved → {summary_path}")

if __name__ == "__main__":
    main()
