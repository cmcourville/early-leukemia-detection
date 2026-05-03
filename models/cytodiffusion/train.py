"""
models/cytodiffusion/train.py — CS 534, Team 6 — WPI
Author: Darshan

Training entry point for CytoDiffusionModel.

Usage:
    python train.py                          # Raabin-WBC, defaults
    python train.py --dataset bccd           # BCCD 3-class
    python train.py --dataset cytodata --data_dir /path/to/cytodata
    python train.py --epochs 50 --no_freeze_encoder --lr 5e-4
    python train.py --device cuda:0 --batch_size 64
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "models"))

from shared.config import (
    CHECKPOINT_DIRS, DATASET_CONFIGS, GLOBAL_SEED,
)
from shared.data.data_loader import get_dataloaders
from cytodiffusion.model import CytoDiffusionModel


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CytoDiffusionModel.")
    p.add_argument("--dataset",     type=str,   default="raabin",
                   choices=["raabin", "bccd", "cytodata", "cnmc"])
    p.add_argument("--data_dir",    type=str,   default=None,
                   help="Local root for CytoData (ImageFolder layout).")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate for the classification head.")
    p.add_argument("--lr_encoder",  type=float, default=1e-5,
                   help="Learning rate for the VAE encoder (only with --no_freeze_encoder).")
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--hidden_dim",  type=int,   default=512)
    p.add_argument("--freeze_encoder",    action="store_true",  default=True,
                   help="Freeze VAE encoder weights (default: True).")
    p.add_argument("--no_freeze_encoder", dest="freeze_encoder",
                   action="store_false",
                   help="Fine-tune the VAE encoder (requires ≥16 GB VRAM).")
    p.add_argument("--cache_dir",   type=str,   default=None,
                   help="HuggingFace cache directory.")
    p.add_argument("--output_dir",  type=str,   default=None,
                   help="Directory for the saved checkpoint.")
    p.add_argument("--device",      type=str,   default=None)
    p.add_argument("--seed",        type=int,   default=GLOBAL_SEED)
    return p.parse_args()


def main() -> None:
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] Device : {device}")
    print(f"[train] Dataset: {args.dataset}")

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    num_classes = dataset_cfg["num_classes"]
    img_mean    = dataset_cfg["mean"]
    img_std     = dataset_cfg["std"]

    train_loader, val_loader, _, _ = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=(device.type == "cuda"),
    )

    model = CytoDiffusionModel(
        num_classes=num_classes,
        freeze_encoder=args.freeze_encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        img_mean=img_mean,
        img_std=img_std,
        epochs=args.epochs,
        lr_head=args.lr,
        lr_encoder=args.lr_encoder,
    )

    model.train_model(train_loader, device=device, val_loader=val_loader)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else _PROJECT_ROOT / CHECKPOINT_DIRS["cytodiffusion"]
    )
    ckpt_path = out_dir / f"cytodiffusion_{args.dataset}.pt"
    model.save(str(ckpt_path))
    print(f"[train] Checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
