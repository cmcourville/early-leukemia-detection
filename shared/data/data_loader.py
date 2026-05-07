"""
shared/data/data_loader.py — CS 534, Team 6 — WPI

Shared dataset utilities for all three model implementations.

All three teams (BccT, CytoDiffusion, ViT-CNN Ensemble) use this module so
that data loading, augmentation, normalisation, and few-shot sampling are
identical across experiments.  Results are therefore directly comparable.

Supported datasets (pass dataset="..." to get_dataloaders / get_few_shot_loaders):
    "raabin"    — Raabin-WBC, 5-class WBC classification (default)
                  Source: HuggingFace  polejowska/lcbsi-wbc-ap
    "bccd"      — BCCD, 3-class (WBC / RBC / Platelet) detection → classification
                  Source: HuggingFace  keremberke/blood-cell-object-detection
                  Each annotated bounding box is cropped to form one sample.
    "cnmc"      — C-NMC 2019, binary leukemia detection (ALL blast vs. HEM healthy)
                  Source: HuggingFace  dwb2023/cnmc-leukemia-2019  (fully public)
                  10,661 images from 73 patients; val split carved from train (20%).
    "cytodata"  — CytoData, 10-class single-cell morphology (Addenbrooke's Hospital)
                  Source: local directory (pass data_dir=<path>)
                  Expected layout: <data_dir>/{train,val,test}/<ClassName>/image.jpg
                  Local path: data/cytodata/ (relative to project root)

Exports:
    get_dataloaders(dataset, data_dir, ...)   — full train/val/test loaders
    get_few_shot_loaders(n_shot, dataset, ...)— n-shot training loader
    get_train_transform(mean, std)            — training augmentation pipeline
    get_eval_transform(mean, std)             — eval transform (no augmentation)
    HFImageDataset                            — HuggingFace → PyTorch bridge
    BCCDCellDataset                           — BCCD detection → cell crops
    LocalImageFolderDataset                   — CytoData / local ImageFolder
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image

# Make shared/config importable from any working directory
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
    DATASET_CONFIGS,
)

# Transform factories

def get_train_transform(mean: list, std: list) -> transforms.Compose:
    """
    Training augmentation:
      Resize → RandomHorizontalFlip → ColorJitter → ToTensor → Normalise.
    Mean/std are dataset-specific (see DATASET_CONFIGS in config.py).
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_eval_transform(mean: list, std: list) -> transforms.Compose:
    """
    Val / test: augmentation-free (Resize 256 → CenterCrop 224 → Normalise).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# Dataset A — Raabin-WBC / generic HuggingFace classification dataset

class HFImageDataset(Dataset):
    """
    Wraps a HuggingFace classification Dataset split as a PyTorch Dataset.
    Each item yields (image_tensor, label_int).
    Auto-detects image and label column names.
    """

    def __init__(self, hf_split, transform=None):
        self.data      = hf_split
        self.transform = transform
        sample = self.data[0]

        if "image" in sample:
            self.img_col = "image"
        elif "img" in sample:
            self.img_col = "img"
        else:
            raise KeyError(f"No image column found. Keys: {list(sample.keys())}")

        if "label" in sample:
            self.lbl_col = "label"
        elif "labels" in sample:
            self.lbl_col = "labels"
        else:
            raise KeyError(f"No label column found. Keys: {list(sample.keys())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = item[self.img_col]
        if isinstance(image, dict):
            import io
            b = image.get("bytes")
            p = image.get("path")
            if b:
                image = Image.open(io.BytesIO(b))
            elif p:
                image = Image.open(p)
            else:
                raise ValueError(f"Image dict has no bytes or path: {image}")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = item[self.lbl_col]
        if isinstance(label, str):
            if not hasattr(self, "_str_label_map"):
                unique = sorted(set(self.data[self.lbl_col]))
                self._str_label_map = {v: i for i, v in enumerate(unique)}
            label = self._str_label_map[label]
        return image, int(label)

    @property
    def class_names(self) -> List[str]:
        try:
            return self.data.features[self.lbl_col].names
        except Exception:
            return [str(i) for i in range(NUM_CLASSES)]

# Dataset B — BCCD (detection → cell crops)

class BCCDCellDataset(Dataset):
    """
    Converts the BCCD object-detection dataset (keremberke/blood-cell-object-detection)
    into a classification dataset by cropping each annotated bounding box.

    Each sample is a single cropped cell with its category label:
        0 = Platelet
        1 = RBC (Red Blood Cell)
        2 = WBC (White Blood Cell)

    The HuggingFace dataset uses COCO-style annotations.  Each example has:
        image     : PIL Image
        objects   : dict with keys "bbox" (list of [x, y, w, h]) and "category" (list of int)

    A minimum crop size (min_crop_px) is enforced to skip degenerate boxes.
    """

    # Category ID → class index mapping for keremberke BCCD
    CATEGORY_TO_IDX = {0: 0, 1: 1, 2: 2}  # Platelet=0, RBC=1, WBC=2
    CLASS_NAMES = ["Platelets", "RBC", "WBC"]

    def __init__(self, hf_split, transform=None, min_crop_px: int = 8):
        self.transform    = transform
        self.min_crop_px  = min_crop_px
        self.samples: List[Tuple[Image.Image, int]] = []

        print(f"[BCCDCellDataset] Building cell-crop dataset from "
              f"{len(hf_split)} images …")
        for item in hf_split:
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")
            W, H  = image.size

            objects = item.get("objects", {})

            # Support both COCO [x, y, w, h] and [x1, y1, x2, y2] bbox formats
            bboxes     = objects.get("bbox",     objects.get("bboxes",     []))
            categories = objects.get("category", objects.get("categories", []))

            for bbox, cat in zip(bboxes, categories):
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                # If w,h look like coordinates (> x,y significantly), treat as x2,y2
                if w > x and h > y and (w - x) < W and (h - y) < H:
                    x1, y1, x2, y2 = x, y, w, h
                else:
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                # Clamp to image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(W,  int(x2))
                y2 = min(H,  int(y2))

                if (x2 - x1) < self.min_crop_px or (y2 - y1) < self.min_crop_px:
                    continue  # skip tiny / degenerate boxes

                crop  = image.crop((x1, y1, x2, y2))
                label = self.CATEGORY_TO_IDX.get(int(cat), int(cat))
                self.samples.append((crop, label))

        print(f"[BCCDCellDataset] Total cell crops: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Dataset C — CytoData (local ImageFolder)

class LocalImageFolderDataset(Dataset):
    """
    Loads a locally stored dataset organised as an ImageFolder:

        <root>/<split>/<ClassName>/image.jpg

    where <split> is one of "train", "val", "test".

    Used for CytoData (Addenbrooke's Hospital), which is not publicly on
    HuggingFace.  Request access via https://github.com/CambridgeCIA/CytoDiffusion

    Args:
        root_dir   : Path to dataset root (contains train/, val/, test/).
        split      : "train", "val", or "test".
        class_names: Ordered list of class names matching folder names.
                     If None, folder names are sorted alphabetically.
        transform  : torchvision transform pipeline.
    """

    def __init__(
        self,
        root_dir:    str,
        split:       str,
        class_names: Optional[List[str]] = None,
        transform=None,
    ):
        self.root_dir  = Path(root_dir) / split
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"[LocalImageFolderDataset] Directory not found: {self.root_dir}\n"
                f"Expected layout: <data_dir>/{{train,val,test}}/<ClassName>/image.jpg\n"
                f"For CytoData: see https://github.com/CambridgeCIA/CytoDiffusion"
            )

        # Build class list
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = sorted(
                d.name for d in self.root_dir.iterdir() if d.is_dir()
            )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Collect all image paths
        EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for fp in cls_dir.iterdir():
                if fp.suffix.lower() in EXTENSIONS and not fp.name.startswith('.'):
                    self.samples.append((fp, self.class_to_idx[cls_name]))

        print(f"[LocalImageFolderDataset] {split}: {len(self.samples)} images, "
              f"{len(self.class_names)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Internal helpers

def _load_raabin(
    batch_size:  int,
    num_workers: int,
    cache_dir:   Optional[str],
    pin_memory:  bool,
    data_dir:    Optional[str] = None,
    hf_token:    Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Load Raabin-WBC (Acevedo et al. 2020) from a local ImageFolder directory.

    The dataset must first be split into train/val/test by running:
        python prepare_raabin.py
    which reads from data/raabin/<ClassName>/ and creates:
        data/raabin/{train,val,test}/<ClassName>/

    Pass --data_dir data/raabin (relative to project root) when calling run_all.py.
    """
    cfg = DATASET_CONFIGS["raabin"]

    if not data_dir:
        raise ValueError(
            "\n\n" + "="*64 + "\n"
            "  ERROR: Raabin dataset requires --data_dir\n"
            "="*64 + "\n\n"
            "  The Raabin-WBC dataset is stored locally.\n"
            "  Provide the path to the dataset root:\n\n"
            "    python run_all.py --dataset raabin --data_dir data/raabin\n\n"
            "  If you haven't split the dataset yet, run first:\n"
            "    python prepare_raabin.py\n"
            + "="*64 + "\n"
        )

    # Verify train/val/test splits exist (i.e. prepare_raabin.py has been run)
    data_path = Path(data_dir)
    missing = [s for s in ["train", "val", "test"] if not (data_path / s).is_dir()]
    if missing:
        raise FileNotFoundError(
            "\n\n" + "="*64 + "\n"
            f"  ERROR: Missing split folders in {data_dir}: {missing}\n"
            "="*64 + "\n\n"
            "  Run the split script first:\n"
            "    python prepare_raabin.py\n"
            + "="*64 + "\n"
        )

    print(f"[shared/data] Loading Raabin-WBC from local path: {data_dir}")
    train_tf = get_train_transform(cfg["mean"], cfg["std"])
    eval_tf  = get_eval_transform(cfg["mean"],  cfg["std"])

    train_ds = LocalImageFolderDataset(data_dir, "train", cfg["class_names"], train_tf)
    val_ds   = LocalImageFolderDataset(data_dir, "val",   cfg["class_names"], eval_tf)
    test_ds  = LocalImageFolderDataset(data_dir, "test",  cfg["class_names"], eval_tf)

    label_map = {i: n for i, n in enumerate(cfg["class_names"])}
    print(f"[shared/data] Raabin — train:{len(train_ds)}, val:{len(val_ds)}, "
          f"test:{len(test_ds)}, classes:{label_map}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory, drop_last=False),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        label_map,
    )


def _load_bccd(
    batch_size:  int,
    num_workers: int,
    cache_dir:   Optional[str],
    pin_memory:  bool,
    data_dir:    Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Load BCCD.

    Preferred path: local ImageFolder of pre-cropped cells (from prepare_bccd.py).
    Fallback: HuggingFace keremberke/blood-cell-object-detection (may be unavailable).
    """
    cfg = DATASET_CONFIGS["bccd"]

    if data_dir:
        data_path = Path(data_dir)
        missing = [s for s in ["train", "val", "test"] if not (data_path / s).is_dir()]
        if missing:
            raise FileNotFoundError(
                "\n\n" + "="*64 + "\n"
                f"  ERROR: Missing split folders in {data_dir}: {missing}\n"
                "="*64 + "\n\n"
                "  Run the prepare script first:\n"
                "    python prepare_bccd.py\n"
                + "="*64 + "\n"
            )
        print(f"[shared/data] Loading BCCD from local path: {data_dir}")
        train_tf = get_train_transform(cfg["mean"], cfg["std"])
        eval_tf  = get_eval_transform(cfg["mean"],  cfg["std"])
        train_ds = LocalImageFolderDataset(data_dir, "train", cfg["class_names"], train_tf)
        val_ds   = LocalImageFolderDataset(data_dir, "val",   cfg["class_names"], eval_tf)
        test_ds  = LocalImageFolderDataset(data_dir, "test",  cfg["class_names"], eval_tf)
        label_map = {i: n for i, n in enumerate(cfg["class_names"])}
        print(f"[shared/data] BCCD — train:{len(train_ds)}, val:{len(val_ds)}, "
              f"test:{len(test_ds)}, classes:{label_map}")
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory, drop_last=False),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
            label_map,
        )

    # HuggingFace fallback
    from datasets import load_dataset as _load

    kw  = {"cache_dir": cache_dir} if cache_dir else {}

    print(f"[shared/data] Loading BCCD ({cfg['hf_name']}) …")
    try:
        hf_train = _load(cfg["hf_name"], name="full", split=cfg["hf_splits"]["train"], **kw)
        hf_val   = _load(cfg["hf_name"], name="full", split=cfg["hf_splits"]["val"],   **kw)
        hf_test  = _load(cfg["hf_name"], name="full", split=cfg["hf_splits"]["test"],  **kw)
    except RuntimeError as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            raise RuntimeError(
                f"\n\n{'='*64}\n"
                f"  ERROR: BCCD loader is incompatible with your `datasets` version\n"
                f"  Current runtime rejected dataset script for: {cfg['hf_name']}\n"
                f"{'='*64}\n\n"
                f"  Fix: python prepare_bccd.py  then pass --data_dir data/bccd\n"
                f"{'='*64}\n"
            ) from exc
        raise

    train_tf = get_train_transform(cfg["mean"], cfg["std"])
    eval_tf  = get_eval_transform(cfg["mean"],  cfg["std"])

    train_ds = BCCDCellDataset(hf_train, transform=train_tf)
    val_ds   = BCCDCellDataset(hf_val,   transform=eval_tf)
    test_ds  = BCCDCellDataset(hf_test,  transform=eval_tf)

    label_map = {i: n for i, n in enumerate(cfg["class_names"])}
    print(f"[shared/data] BCCD — train:{len(train_ds)}, val:{len(val_ds)}, "
          f"test:{len(test_ds)}, classes:{label_map}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory, drop_last=False),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        label_map,
    )


def _load_cnmc(
    batch_size:  int,
    num_workers: int,
    cache_dir:   Optional[str],
    pin_memory:  bool,
    hf_token:    Optional[str] = None,
    data_dir:    Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Load C-NMC 2019 — binary leukemia detection (ALL blast vs. HEM healthy).

    Preferred path: local ImageFolder prepared by prepare_cnmc.py.
    The HuggingFace dataset (dwb2023/cnmc-leukemia-2019) stores only BMP file
    headers (~20 bytes) in the bytes field, so streaming from HF does not work.
    """
    cfg = DATASET_CONFIGS["cnmc"]

    if data_dir:
        data_path = Path(data_dir)
        missing = [s for s in ["train", "val", "test"] if not (data_path / s).is_dir()]
        if missing:
            raise FileNotFoundError(
                "\n\n" + "="*64 + "\n"
                f"  ERROR: Missing split folders in {data_dir}: {missing}\n"
                "="*64 + "\n\n"
                "  Run the prepare script first:\n"
                "    python prepare_cnmc.py --src_dir <path-to-cnmc-download>\n"
                + "="*64 + "\n"
            )
        print(f"[shared/data] Loading C-NMC 2019 from local path: {data_dir}")
        train_tf = get_train_transform(cfg["mean"], cfg["std"])
        eval_tf  = get_eval_transform(cfg["mean"],  cfg["std"])
        train_ds = LocalImageFolderDataset(data_dir, "train", cfg["class_names"], train_tf)
        val_ds   = LocalImageFolderDataset(data_dir, "val",   cfg["class_names"], eval_tf)
        test_ds  = LocalImageFolderDataset(data_dir, "test",  cfg["class_names"], eval_tf)
        label_map = {i: n for i, n in enumerate(cfg["class_names"])}
        print(f"[shared/data] C-NMC 2019 — train:{len(train_ds)}, val:{len(val_ds)}, "
              f"test:{len(test_ds)}, classes:{label_map}")
        print(f"[shared/data] NOTE: C-NMC is class-imbalanced (~68% ALL, ~32% HEM). "
              f"Recall on ALL is the key clinical metric.")
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory, drop_last=False),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory),
            label_map,
        )

    raise RuntimeError(
        "\n\n" + "="*64 + "\n"
        "  ERROR: C-NMC 2019 requires local files — HuggingFace streaming is broken.\n"
        "  The HF dataset (dwb2023/cnmc-leukemia-2019) stores only BMP file headers,\n"
        "  not full images, so it cannot be loaded via the Hub.\n"
        "="*64 + "\n\n"
        "  Steps to fix:\n"
        "  1. Download the C-NMC dataset from Kaggle:\n"
        "       kaggle datasets download -d SBI-LAB/c-nmc-2019 -p data/cnmc_raw --unzip\n"
        "  2. Run the prepare script:\n"
        "       python prepare_cnmc.py --src_dir data/cnmc_raw\n"
        "  3. Rerun with --data_dir:\n"
        "       python run_all.py --dataset cnmc --data_dir data/cnmc\n"
        + "="*64 + "\n"
    )


def _load_cytodata(
    data_dir:    str,
    batch_size:  int,
    num_workers: int,
    pin_memory:  bool,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Load CytoData from a local ImageFolder directory."""
    cfg = DATASET_CONFIGS["cytodata"]

    if not data_dir:
        raise ValueError(
            "[shared/data] CytoData requires --data_dir pointing to the local dataset.\n"
            "Expected layout: <data_dir>/{train,val,test}/<ClassName>/image.jpg\n"
            "Request access: https://github.com/CambridgeCIA/CytoDiffusion"
        )

    print(f"[shared/data] Loading CytoData from local path: {data_dir}")
    train_tf = get_train_transform(cfg["mean"], cfg["std"])
    eval_tf  = get_eval_transform(cfg["mean"],  cfg["std"])

    train_ds = LocalImageFolderDataset(data_dir, "train", cfg["class_names"], train_tf)
    val_ds   = LocalImageFolderDataset(data_dir, "val",   cfg["class_names"], eval_tf)
    test_ds  = LocalImageFolderDataset(data_dir, "test",  cfg["class_names"], eval_tf)

    label_map = {i: n for i, n in enumerate(cfg["class_names"])}
    print(f"[shared/data] CytoData — train:{len(train_ds)}, val:{len(val_ds)}, "
          f"test:{len(test_ds)}, classes:{label_map}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory, drop_last=False),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        label_map,
    )

# Public API

def get_dataloaders(
    dataset:     str = "raabin",
    data_dir:    Optional[str] = None,
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    pin_memory:  bool = True,
    hf_token:    Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Return (train_loader, val_loader, test_loader, label_map) for the
    chosen dataset.

    Args:
        dataset     : One of "raabin", "bccd", "cnmc", "cytodata".
        data_dir    : Required for dataset="cytodata" — path to local dataset root.
        batch_size  : Mini-batch size for all three loaders.
        num_workers : DataLoader worker processes.
        cache_dir   : HuggingFace cache directory (raabin / bccd only).
        pin_memory  : Pin memory for faster GPU transfer.
        hf_token    : HuggingFace API token for gated datasets (raabin).
                      If None, uses the cached token from `huggingface-cli login`.

    Returns:
        train_loader, val_loader, test_loader, label_map
        where label_map = {0: 'ClassName', 1: 'ClassName', ...}
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(DATASET_CONFIGS.keys())}"
        )

    if dataset == "raabin":
        return _load_raabin(batch_size, num_workers, cache_dir, pin_memory, data_dir, hf_token)
    elif dataset == "bccd":
        return _load_bccd(batch_size, num_workers, cache_dir, pin_memory, data_dir)
    elif dataset == "cnmc":
        return _load_cnmc(batch_size, num_workers, cache_dir, pin_memory, hf_token, data_dir)
    elif dataset == "cytodata":
        return _load_cytodata(data_dir, batch_size, num_workers, pin_memory)



def get_few_shot_loaders(
    n_shot:      int,
    dataset:     str = "raabin",
    data_dir:    Optional[str] = None,
    batch_size:  int = 32,
    num_workers: int = 4,
    cache_dir:   Optional[str] = None,
    seed:        int = GLOBAL_SEED,
    hf_token:    Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    Build an n-shot training loader (n samples per class, stratified).
    Val and test loaders use the full splits.

    Works for all three supported datasets.

    Args:
        n_shot   : Examples per class.
        dataset  : One of "raabin", "bccd", "cnmc", "cytodata".
        data_dir : Required for dataset="cytodata".
        seed     : Random seed — same seed → same samples across models.
        hf_token : HuggingFace API token for gated datasets (raabin).

    Returns:
        few_shot_train_loader, val_loader, test_loader, label_map
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Load the full dataloaders first, then subsample the training set
    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        pin_memory=False,
        hf_token=hf_token,
    )

    full_train_ds = train_loader.dataset
    num_classes   = DATASET_CONFIGS[dataset]["num_classes"]

    # Gather indices per class
    class_indices: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
    for idx in range(len(full_train_ds)):
        _, lbl = full_train_ds[idx]
        class_indices[int(lbl)].append(idx)

    selected: List[int] = []
    for cls, idxs in sorted(class_indices.items()):
        if len(idxs) < n_shot:
            raise ValueError(
                f"[shared/data] Class {cls} ('{label_map.get(cls, cls)}') has only "
                f"{len(idxs)} samples — cannot sample {n_shot}-shot."
            )
        selected.extend(random.sample(idxs, n_shot))

    few_shot_ds = Subset(full_train_ds, selected)
    print(f"[shared/data] {n_shot}-shot ({dataset}): "
          f"{len(selected)} samples ({n_shot}/class × {num_classes} classes, seed={seed})")

    few_shot_loader = DataLoader(
        few_shot_ds,
        batch_size=min(batch_size, len(few_shot_ds)),
        shuffle=True,
        num_workers=num_workers,
    )

    return few_shot_loader, val_loader, test_loader, label_map


def compute_dataset_stats(dataset_obj: Dataset, num_workers: int = 4) -> Tuple[list, list]:
    """
    Recompute per-channel mean and std from a PyTorch Dataset.
    Useful for computing normalisation stats for BCCD or CytoData.

    Usage:
        ds = LocalImageFolderDataset(data_dir, "train", ...)
        mean, std = compute_dataset_stats(ds)
    """
    base_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    # Temporarily replace transform
    orig_tf = dataset_obj.transform if hasattr(dataset_obj, "transform") else None
    if hasattr(dataset_obj, "transform"):
        dataset_obj.transform = base_tf

    loader = DataLoader(dataset_obj, batch_size=128, num_workers=num_workers, shuffle=False)
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

    if hasattr(dataset_obj, "transform"):
        dataset_obj.transform = orig_tf

    print(f"[shared/data] Stats — mean={[round(m,4) for m in mean]}, "
          f"std={[round(s,4) for s in std]}")
    return mean, std


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="raabin",
                        choices=["raabin", "bccd", "cnmc", "cytodata"])
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    train_dl, val_dl, test_dl, lmap = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=8,
        num_workers=0,
    )
    imgs, labels = next(iter(train_dl))
    print(f"Dataset     : {args.dataset}")
    print(f"Batch shape : {imgs.shape}")
    print(f"Labels      : {labels}")
    print(f"Label map   : {lmap}")
    print(f"Pixel range : [{imgs.min():.3f}, {imgs.max():.3f}]")
