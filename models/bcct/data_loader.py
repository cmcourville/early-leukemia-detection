"""
data_loader.py — BccT Project (CS 534, Team 6)
Author: Corrin

BccT data loading entry point.  Delegates to shared/data/data_loader.py so
that all three models use identical preprocessing and augmentation.

Supported datasets (pass dataset="..." to get_dataloaders):
    "raabin"   — Raabin-WBC, 5-class WBC classification (default, paper benchmark)
    "bccd"     — BCCD, 3-class (Platelet/RBC/WBC) detection → cell crops
    "cytodata" — CytoData, 10-class morphology; requires --data_dir <local path>

Paper augmentation (Section 4.1, Zhu et al. 2026) is applied when dataset="raabin".
Other datasets use the same augmentation pipeline with their own normalisation stats.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Make shared/ importable from models/bcct/
_project_root = Path(__file__).resolve().parent.parent.parent
_shared_dir   = _project_root / "shared"
for _p in [str(_project_root), str(_shared_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.data.data_loader import (
    get_dataloaders      as _get_dataloaders,
    get_few_shot_loaders as _get_few_shot_loaders,
    get_train_transform,
    get_eval_transform,
    HFImageDataset,
    BCCDCellDataset,
    LocalImageFolderDataset,
    compute_dataset_stats,
)
from shared.config import DATASET_CONFIGS, GLOBAL_SEED

# Re-export dataset classes so callers can import from here directly
__all__ = [
    "get_dataloaders",
    "get_few_shot_loaders",
    "get_train_transform",
    "get_eval_transform",
    "HFImageDataset",
    "BCCDCellDataset",
    "LocalImageFolderDataset",
    "compute_dataset_stats",
]


def get_dataloaders(
    dataset:     str = "raabin",
    data_dir:    Optional[str] = None,
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    pin_memory:  bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Return (train_loader, val_loader, test_loader, label_map) for BccT.
    Delegates to shared/data/data_loader.py for consistency across all models.

    Args:
        dataset     : "raabin" (default), "bccd", or "cytodata".
        data_dir    : Local path for CytoData (required when dataset="cytodata").
        batch_size  : Mini-batch size.
        num_workers : DataLoader worker count.
        cache_dir   : HuggingFace cache directory (raabin / bccd only).
        pin_memory  : Enable pinned memory for faster GPU transfers.
    """
    return _get_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        pin_memory=pin_memory,
    )


def get_few_shot_loaders(
    n_shot:      int,
    dataset:     str = "raabin",
    data_dir:    Optional[str] = None,
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    seed:        int = GLOBAL_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Build an n-shot training loader (stratified, n samples per class).
    Val and test loaders use the full splits.

    Args:
        n_shot   : Training examples per class (e.g. 10, 20, 50).
        dataset  : "raabin", "bccd", or "cytodata".
        seed     : Random seed — use the same seed across models for fair comparison.
    """
    return _get_few_shot_loaders(
        n_shot=n_shot,
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        seed=seed,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BccT data loader sanity check")
    parser.add_argument("--dataset",  default="raabin",
                        choices=["raabin", "bccd", "cytodata"])
    parser.add_argument("--data_dir", default=None,
                        help="Local path for CytoData")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"Testing data loader — dataset={args.dataset}")
    train_dl, val_dl, test_dl, lmap = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    imgs, labels = next(iter(train_dl))
    print(f"Batch shape : {imgs.shape}")
    print(f"Label map   : {lmap}")
    print(f"Pixel range : [{imgs.min():.3f}, {imgs.max():.3f}]")

    cfg = DATASET_CONFIGS[args.dataset]
    print(f"Num classes : {cfg['num_classes']}")
    print(f"Classes     : {cfg['class_names']}")
