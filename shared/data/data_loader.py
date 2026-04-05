"""
shared/data/data_loader.py — CS 534, Team 6 — WPI

Shared dataset utilities for all three model implementations.

All three teams (BccT, CytoDiffusion, ViT-CNN Ensemble) should use this
module so that data loading, augmentation, normalisation, and few-shot
sampling are *identical* across experiments.  This ensures results are
directly comparable in the Phase 2 report.

Exports:
    get_dataloaders(...)         — full-data train/val/test loaders
    get_few_shot_loaders(...)    — n-shot training loader + full val/test
    compute_dataset_stats(...)   — recompute Raabin-WBC mean/std
    HFImageDataset               — HuggingFace → PyTorch dataset bridge
    get_train_transform()        — training augmentation pipeline
    get_eval_transform()         — eval augmentation pipeline (none)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

# Make shared/config importable whether called from project root or subdirectory
_shared_dir = Path(__file__).resolve().parent.parent
if str(_shared_dir) not in sys.path:
    sys.path.insert(0, str(_shared_dir))

from config import (
    HF_DATASET_NAME,
    NUM_CLASSES,
    IMAGE_SIZE,
    RAABIN_MEAN,
    RAABIN_STD,
    GLOBAL_SEED,
)

# HuggingFace → PyTorch bridge

class HFImageDataset(torch.utils.data.Dataset):
    """
    Wraps a HuggingFace `datasets.Dataset` split so it behaves like a
    standard PyTorch Dataset.  Each item yields (image_tensor, label_int).
    """

    def __init__(self, hf_split, transform=None):
        self.data      = hf_split
        self.transform = transform
        sample = self.data[0]

        self.img_col = (
            "image" if "image" in sample
            else "img" if "img" in sample
            else (_ for _ in ()).throw(
                KeyError(f"No image column found. Keys: {list(sample.keys())}")
            )
        )
        self.lbl_col = (
            "label"  if "label"  in sample
            else "labels" if "labels" in sample
            else (_ for _ in ()).throw(
                KeyError(f"No label column found. Keys: {list(sample.keys())}")
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = item[self.img_col]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(item[self.lbl_col])

    @property
    def class_names(self) -> List[str]:
        try:
            return self.data.features[self.lbl_col].names
        except Exception:
            return [str(i) for i in range(NUM_CLASSES)]

# Transform factories  (Section 4.1, Zhu et al. 2026)

def get_train_transform() -> transforms.Compose:
    """
    Training augmentation (paper Section 4.1):
      RandomHorizontalFlip + mild ColorJitter → ToTensor → Normalise.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=RAABIN_MEAN, std=RAABIN_STD),
    ])

def get_eval_transform() -> transforms.Compose:
    """
    Val / test: augmentation-free (Resize → CenterCrop → Normalise).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=RAABIN_MEAN, std=RAABIN_STD),
    ])

# Full-data loaders

def get_dataloaders(
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    pin_memory:  bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Download (or load cached) Raabin-WBC and return three DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, label_map
        where label_map = {0: 'Basophil', 1: 'Eosinophil', ...}
    """
    print(f"[shared/data] Loading dataset: {HF_DATASET_NAME}")
    kw = {"cache_dir": cache_dir} if cache_dir else {}

    hf_train = load_dataset(HF_DATASET_NAME, split="train",      **kw)
    hf_val   = load_dataset(HF_DATASET_NAME, split="validation", **kw)
    hf_test  = load_dataset(HF_DATASET_NAME, split="test",       **kw)

    print(f"[shared/data] Sizes — train: {len(hf_train)}, "
          f"val: {len(hf_val)}, test: {len(hf_test)}")

    train_ds = HFImageDataset(hf_train, transform=get_train_transform())
    val_ds   = HFImageDataset(hf_val,   transform=get_eval_transform())
    test_ds  = HFImageDataset(hf_test,  transform=get_eval_transform())

    label_map = {i: name for i, name in enumerate(train_ds.class_names)}
    print(f"[shared/data] Classes: {label_map}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, label_map

# Few-shot subset loader

def get_few_shot_loaders(
    n_shot:      int,
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    seed:        int = GLOBAL_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Build an n-shot training loader: n samples per class (stratified).
    Val and test loaders use the full splits unchanged.

    Args:
        n_shot: Examples per class (5 classes → 5n training samples total).
        seed:   Random seed for reproducible sampling across models.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    kw = {"cache_dir": cache_dir} if cache_dir else {}
    hf_train = load_dataset(HF_DATASET_NAME, split="train",      **kw)
    hf_val   = load_dataset(HF_DATASET_NAME, split="validation", **kw)
    hf_test  = load_dataset(HF_DATASET_NAME, split="test",       **kw)

    sample   = hf_train[0]
    lbl_col  = "label" if "label" in sample else "labels"

    class_indices: Dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
    for idx in range(len(hf_train)):
        class_indices[int(hf_train[idx][lbl_col])].append(idx)

    selected: List[int] = []
    for cls, idxs in class_indices.items():
        if len(idxs) < n_shot:
            raise ValueError(
                f"Class {cls} has only {len(idxs)} samples, cannot sample {n_shot}."
            )
        selected.extend(random.sample(idxs, n_shot))

    print(f"[shared/data] {n_shot}-shot: {len(selected)} training samples "
          f"({n_shot}/class × {NUM_CLASSES} classes, seed={seed})")

    full_train_ds   = HFImageDataset(hf_train, transform=get_train_transform())
    few_shot_ds     = Subset(full_train_ds, selected)
    val_ds          = HFImageDataset(hf_val,  transform=get_eval_transform())
    test_ds         = HFImageDataset(hf_test, transform=get_eval_transform())

    label_map = {i: name for i, name in enumerate(
        HFImageDataset(hf_train).class_names
    )}

    few_shot_loader = DataLoader(
        few_shot_ds,
        batch_size=min(batch_size, len(few_shot_ds)),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return few_shot_loader, val_loader, test_loader, label_map

# Dataset statistics recomputation helper

def compute_dataset_stats(hf_split, num_workers: int = 4) -> Tuple[list, list]:
    """Recompute per-channel mean and std from the given HF split."""
    base_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    tmp_ds = HFImageDataset(hf_split, transform=base_tf)
    loader = DataLoader(tmp_ds, batch_size=128, num_workers=num_workers,
                        shuffle=False)

    ch_sum    = torch.zeros(3)
    ch_sq_sum = torch.zeros(3)
    n_pixels  = 0

    for imgs, _ in loader:
        B, C, H, W  = imgs.shape
        ch_sum     += imgs.sum(dim=[0, 2, 3])
        ch_sq_sum  += (imgs ** 2).sum(dim=[0, 2, 3])
        n_pixels   += B * H * W

    mean = (ch_sum / n_pixels).tolist()
    std  = ((ch_sq_sum / n_pixels - torch.tensor(mean) ** 2) ** 0.5).tolist()
    print(f"[shared/data] Recomputed stats — mean={mean}, std={std}")
    return mean, std

if __name__ == "__main__":
    train_dl, val_dl, test_dl, lmap = get_dataloaders(batch_size=8, num_workers=0)
    imgs, labels = next(iter(train_dl))
    print(f"Batch shape : {imgs.shape}")
    print(f"Label map   : {lmap}")
