"""
data_loader.py — BccT Project (CS 534, Team 6)
Author: Corrin

Loads the Raabin-WBC dataset from HuggingFace Hub (`polejowska/lcbsi-wbc-ap`),
applies paper-specified augmentation (Section 4.1 of Zhu et al. 2026), and
returns DataLoaders for train / val / test splits.

Paper Section 4.1 specs:
  - Images resized to 224×224
  - Normalised with dataset-wise mean/std (computed below if not cached)
  - Training augmentation: RandomHorizontalFlip + mild ColorJitter only
  - Validation / test: no augmentation (CenterCrop to 224 after Resize 256)
  - Split: 70 / 10 / 20 (train / val / test) — provided by the HF dataset
  - 5 WBC classes: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datasets import load_dataset                    # HuggingFace datasets
from PIL import Image

# Constants

HF_DATASET_NAME = "polejowska/lcbsi-wbc-ap"

# Dataset-wise mean/std pre-computed from the Raabin-WBC training split.
# These match the values commonly reported for Raabin-WBC in the literature.
# If you want to recompute, call `compute_dataset_stats()` below.
RAABIN_MEAN = [0.7442, 0.6384, 0.7516]
RAABIN_STD  = [0.1580, 0.1914, 0.1225]

NUM_CLASSES = 5
IMAGE_SIZE  = 224   # Section 4.1

# HuggingFace → PyTorch bridge

class HFImageDataset(torch.utils.data.Dataset):
    """
    Wraps a HuggingFace `datasets.Dataset` split so it behaves like a
    standard PyTorch Dataset.  Each item yields (image_tensor, label_int).
    """

    def __init__(self, hf_split, transform=None):
        self.data      = hf_split
        self.transform = transform

        # Determine the image column name (HF schema may call it 'image' or 'img')
        sample = self.data[0]
        if "image" in sample:
            self.img_col = "image"
        elif "img" in sample:
            self.img_col = "img"
        else:
            raise KeyError("Cannot find image column in HF dataset. "
                           f"Available keys: {list(sample.keys())}")

        # Label column
        if "label" in sample:
            self.lbl_col = "label"
        elif "labels" in sample:
            self.lbl_col = "labels"
        else:
            raise KeyError("Cannot find label column in HF dataset. "
                           f"Available keys: {list(sample.keys())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = item[self.img_col]

        # Ensure PIL Image in RGB mode
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(item[self.lbl_col])
        return image, label

    @property
    def class_names(self):
        """Return ordered list of class names if the HF split has ClassLabel info."""
        try:
            return self.data.features[self.lbl_col].names
        except Exception:
            return [str(i) for i in range(NUM_CLASSES)]

# Transform factories

def get_train_transform() -> transforms.Compose:
    """
    Section 4.1 — training augmentation:
      1. Resize to 224×224 (direct, not crop-based, to preserve cell morphology)
      2. RandomHorizontalFlip
      3. Mild ColorJitter (brightness/contrast ±0.2, saturation ±0.1, hue ±0.05)
      4. ToTensor + Normalise with Raabin-WBC mean/std
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=RAABIN_MEAN, std=RAABIN_STD),
    ])

def get_eval_transform() -> transforms.Compose:
    """
    Section 4.1 — val / test: augmentation-free.
      Resize to 256, then CenterCrop to 224 (standard ViT eval pipeline).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=RAABIN_MEAN, std=RAABIN_STD),
    ])

# Dataset statistics helper

def compute_dataset_stats(hf_split, num_workers: int = 4) -> Tuple[list, list]:
    """
    Recompute per-channel mean and std from scratch.
    Useful for verifying RAABIN_MEAN/RAABIN_STD above.

    Usage:
        ds_raw = load_dataset(HF_DATASET_NAME, split="train")
        mean, std = compute_dataset_stats(ds_raw)
    """
    base_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    tmp_ds = HFImageDataset(hf_split, transform=base_tf)
    loader = DataLoader(tmp_ds, batch_size=128, num_workers=num_workers,
                        shuffle=False, pin_memory=False)

    channel_sum   = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    n_pixels = 0

    for imgs, _ in loader:
        # imgs: (B, 3, H, W)
        B, C, H, W = imgs.shape
        channel_sum    += imgs.sum(dim=[0, 2, 3])
        channel_sq_sum += (imgs ** 2).sum(dim=[0, 2, 3])
        n_pixels       += B * H * W

    mean = (channel_sum / n_pixels).tolist()
    std  = ((channel_sq_sum / n_pixels - torch.tensor(mean) ** 2) ** 0.5).tolist()
    print(f"[compute_dataset_stats] mean={mean}, std={std}")
    return mean, std

# Main loader function

def get_dataloaders(
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    pin_memory:  bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Download (or load from cache) the Raabin-WBC HF dataset and return three
    DataLoaders: (train_loader, val_loader, test_loader, label_map).

    `label_map` maps integer class index → human-readable class name.

    Args:
        batch_size:  Mini-batch size for all loaders.
        num_workers: Dataloader worker processes.
        cache_dir:   HuggingFace cache directory (None = default ~/.cache/huggingface).
        pin_memory:  Pin memory for faster GPU transfer.

    Returns:
        train_loader, val_loader, test_loader, label_map
    """
    print(f"[data_loader] Loading dataset: {HF_DATASET_NAME}")

    load_kwargs = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    # The HF dataset ships with pre-defined train/validation/test splits.
    hf_train = load_dataset(HF_DATASET_NAME, split="train",      **load_kwargs)
    hf_val   = load_dataset(HF_DATASET_NAME, split="validation", **load_kwargs)
    hf_test  = load_dataset(HF_DATASET_NAME, split="test",       **load_kwargs)

    print(f"[data_loader] Split sizes — train: {len(hf_train)}, "
          f"val: {len(hf_val)}, test: {len(hf_test)}")

    train_ds = HFImageDataset(hf_train, transform=get_train_transform())
    val_ds   = HFImageDataset(hf_val,   transform=get_eval_transform())
    test_ds  = HFImageDataset(hf_test,  transform=get_eval_transform())

    label_map = {i: name for i, name in enumerate(train_ds.class_names)}
    print(f"[data_loader] Classes: {label_map}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, label_map

# Few-shot subset helper  (used by low_data_test.py)

def get_few_shot_loaders(
    n_shot:      int,
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    seed:        int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Build a *n*-shot training loader: for each of the 5 WBC classes, randomly
    select *n* samples.  Val and test loaders use the full splits (unchanged).

    Args:
        n_shot:  Number of labelled examples per class (e.g. 10, 20, 50).
        seed:    Random seed for reproducibility.

    Returns:
        few_shot_train_loader, val_loader, test_loader, label_map
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    load_kwargs = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    hf_train = load_dataset(HF_DATASET_NAME, split="train",      **load_kwargs)
    hf_val   = load_dataset(HF_DATASET_NAME, split="validation", **load_kwargs)
    hf_test  = load_dataset(HF_DATASET_NAME, split="test",       **load_kwargs)

    # Determine label column
    sample = hf_train[0]
    lbl_col = "label" if "label" in sample else "labels"

    # Gather indices per class
    class_indices: Dict[int, list] = {i: [] for i in range(NUM_CLASSES)}
    for idx in range(len(hf_train)):
        lbl = int(hf_train[idx][lbl_col])
        class_indices[lbl].append(idx)

    selected_indices = []
    for cls_idx, indices in class_indices.items():
        if len(indices) < n_shot:
            raise ValueError(
                f"Class {cls_idx} has only {len(indices)} samples, "
                f"cannot sample {n_shot}-shot."
            )
        selected_indices.extend(random.sample(indices, n_shot))

    print(f"[data_loader] {n_shot}-shot: using {len(selected_indices)} "
          f"training samples ({n_shot} per class × {NUM_CLASSES} classes)")

    full_train_ds = HFImageDataset(hf_train, transform=get_train_transform())
    few_shot_ds   = Subset(full_train_ds, selected_indices)

    val_ds  = HFImageDataset(hf_val,  transform=get_eval_transform())
    test_ds = HFImageDataset(hf_test, transform=get_eval_transform())

    label_map = {i: name for i, name in enumerate(
        HFImageDataset(hf_train).class_names
    )}

    few_shot_loader = DataLoader(
        few_shot_ds,
        batch_size=min(batch_size, len(few_shot_ds)),
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return few_shot_loader, val_loader, test_loader, label_map

# Quick sanity check

if __name__ == "__main__":
    train_dl, val_dl, test_dl, lmap = get_dataloaders(batch_size=8, num_workers=0)
    imgs, labels = next(iter(train_dl))
    print(f"Batch shape : {imgs.shape}")        # (8, 3, 224, 224)
    print(f"Labels      : {labels}")
    print(f"Label map   : {lmap}")
    print(f"Pixel range : [{imgs.min():.3f}, {imgs.max():.3f}]")
