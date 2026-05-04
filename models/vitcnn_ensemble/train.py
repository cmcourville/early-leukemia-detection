"""
models/vitcnn_ensemble/train.py — CS 534, Team 6 — WPI
Author: Sean

Training entry point for the ViT-CNN Ensemble model.

=============================================================================
  TODO (Sean): Implement training logic here.
=============================================================================

Suggested CLI interface (mirrors models/bcct/train_bcct.py):

    python train.py [--batch_size N] [--epochs N] [--lr LR] [--output_dir PATH]

Shared data loading:
    from shared.data.data_loader import get_dataloaders
    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )

Suggested checkpoint save location: checkpoints/vitcnn_ensemble_model.pt
"""
from shared.data.data_loader import get_dataloaders
from shared.config import (
    CHECKPOINT_DIRS, DATASET_CONFIGS, GLOBAL_SEED,
)
import argparse
import torch
from typing import Optional
from pathlib import Path

from model import ViTCNNEnsemble

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training Module for the VitCNN Ensemble model."
    )
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
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Target number of epochs (default: 25)."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="Learning rate (default: 0.0001)."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay (default: 1e-5)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints",
        help="Directory to save model checkpoint (default: ./checkpoints)."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', or 'cuda:N'. Auto-detected if None."
    )
    return parser.parse_args()

def main() -> None:

    args = get_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} will be used for model training")

    print(f"Loading dataset from {args.dataset} ...")

    dataset_configs = DATASET_CONFIGS[args.dataset]
    num_classes = dataset_configs["num_classes"]

    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=(device.type == "cuda"),
    )

    model = ViTCNNEnsemble(
        num_classes= num_classes
    )
    model.to(device)
    model.train_model(train_loader, device)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else CHECKPOINT_DIRS["vitcnn_ensemble"]
    )

    checkpoint_path = out_dir / "vitcnn_ensemble.pt"
    model.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()

