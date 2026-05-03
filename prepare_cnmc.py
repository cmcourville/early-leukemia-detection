"""
prepare_cnmc.py — CS 534, Team 6
Organizes a downloaded C-NMC 2019 dataset into ImageFolder format
so the pipeline can load it via LocalImageFolderDataset.

Class detection: reads the `_all` or `_hem` suffix from each BMP filename.
    UID_1_1_1_all.bmp  → class "all"  (leukemic blast cells)
    UID_2_3_4_hem.bmp  → class "hem"  (healthy cells)

Output layout:
    <data_dir>/train/all/
    <data_dir>/train/hem/
    <data_dir>/val/all/
    <data_dir>/val/hem/
    <data_dir>/test/all/
    <data_dir>/test/hem/

Download the source data first (any of these work):

    Option A — Kaggle (recommended):
        kaggle datasets download -d SBI-LAB/c-nmc-2019 -p data/cnmc_raw --unzip

    Option B — Manual download from TCIA:
        https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223

    Option C — Colab / Drive:
        # Mount Drive first, then:
        !cp -r "/content/drive/MyDrive/grad-school/Spring2026/CS534/CS534-Group6/cnmc_raw" data/

Usage:
    python prepare_cnmc.py --src_dir data/cnmc_raw
    python prepare_cnmc.py --src_dir data/cnmc_raw --data_dir data/cnmc --train 0.70 --val 0.10 --test 0.20
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASSES  = ["all", "hem"]
IMG_EXT  = {".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
SUFFIXES = {"_all": "all", "_hem": "hem"}


def _class_from_filename(name: str) -> str | None:
    stem = Path(name).stem.lower()
    for suffix, cls in SUFFIXES.items():
        if stem.endswith(suffix):
            return cls
    return None


def find_images(src_dir: Path) -> dict[str, list[Path]]:
    """Recursively find all BMP/image files and group by class."""
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(src_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        cls = _class_from_filename(p.name)
        if cls is None:
            # Fall back: check parent folder name
            parent = p.parent.name.lower()
            if parent in CLASSES:
                cls = parent
        if cls is not None:
            by_class[cls].append(p)
    return by_class


def split_and_copy(
    by_class:   dict[str, list[Path]],
    data_dir:   Path,
    train_frac: float,
    val_frac:   float,
    test_frac:  float,
    seed:       int,
) -> None:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    random.seed(seed)
    total = 0

    for cls in CLASSES:
        files = by_class.get(cls, [])
        if not files:
            print(f"  [SKIP] {cls} — no images found")
            continue

        random.shuffle(files)
        n       = len(files)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        splits = {
            "train": files[:n_train],
            "val":   files[n_train : n_train + n_val],
            "test":  files[n_train + n_val :],
        }

        print(f"  {cls}: {n} total → "
              f"train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])}")

        for split_name, img_list in splits.items():
            dest_dir = data_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in img_list:
                dest = dest_dir / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)
            total += len(img_list)

    print(f"\nDone — {total} images copied to {data_dir}/")
    print(f"\nRun with: python run_all.py --dataset cnmc --data_dir {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare C-NMC 2019 dataset split")
    parser.add_argument("--src_dir",  required=True,
                        help="Path to the downloaded C-NMC dataset (contains BMP files)")
    parser.add_argument("--data_dir", default="data/cnmc",
                        help="Output directory (default: data/cnmc)")
    parser.add_argument("--train",    type=float, default=0.70)
    parser.add_argument("--val",      type=float, default=0.10)
    parser.add_argument("--test",     type=float, default=0.20)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    src_dir  = Path(args.src_dir)
    data_dir = Path(args.data_dir)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    print(f"Scanning {src_dir} for C-NMC images …")
    by_class = find_images(src_dir)

    total_found = sum(len(v) for v in by_class.values())
    if total_found == 0:
        raise RuntimeError(
            f"No images found under {src_dir}.\n"
            "Check that the download extracted correctly and contains .bmp files."
        )

    print(f"Found {total_found} images: "
          + ", ".join(f"{cls}={len(by_class.get(cls,[]))}" for cls in CLASSES))
    print(f"\nSplitting {args.train:.0%} / {args.val:.0%} / {args.test:.0%}, seed={args.seed}\n")

    split_and_copy(by_class, data_dir, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
