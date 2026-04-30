"""
prepare_raabin.py — CS 534, Team 6
Splits the flat PBC/Acevedo dataset in data/raabin/ into
train / val / test subfolders using a stratified 70/10/20 split.

Usage (from early-leukemia-detection/):
    python prepare_raabin.py
    python prepare_raabin.py --data_dir data/raabin --train 0.70 --val 0.10 --test 0.20 --seed 42

What it does:
  1. Reads images from data/raabin/<ClassName>/*.jpg  (flat, pre-split)
  2. Shuffles each class with a fixed seed (reproducible)
  3. Copies (not moves) images into:
       data/raabin/train/<ClassName>/
       data/raabin/val/<ClassName>/
       data/raabin/test/<ClassName>/
  4. Leaves the original flat class folders untouched

Classes used: Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil
  (erythroblast / ig / platelet are ignored — delete them separately if desired)
"""

import argparse
import random
import shutil
from pathlib import Path


CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
SPLITS  = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def split_dataset(data_dir: Path, train_frac: float, val_frac: float,
                  test_frac: float, seed: int) -> None:

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "train + val + test fractions must sum to 1.0"

    random.seed(seed)
    total_copied = 0

    for cls in CLASSES:
        src_dir = data_dir / cls
        if not src_dir.is_dir():
            print(f"  [SKIP] {cls} — folder not found at {src_dir}")
            continue

        images = sorted(p for p in src_dir.iterdir()
                        if p.suffix.lower() in IMG_EXT and not p.name.startswith('.'))
        if not images:
            print(f"  [SKIP] {cls} — no images found")
            continue

        random.shuffle(images)
        n      = len(images)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        # test gets the remainder so totals are exact
        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        print(f"  {cls}: {n} total → "
              f"train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])}")

        for split_name, files in splits.items():
            dest_dir = data_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in files:
                dest = dest_dir / src.name
                if not dest.exists():
                    shutil.copy2(src, dest)
            total_copied += len(files)

    print(f"\nDone — {total_copied} images copied into train/val/test.")
    print(f"Original flat class folders are untouched.")
    print(f"\nRun with: python run_all.py --dataset raabin --data_dir data/raabin")


def main():
    parser = argparse.ArgumentParser(description="Prepare Raabin/Acevedo dataset split")
    parser.add_argument("--data_dir", default="data/raabin",
                        help="Path to flat dataset root (default: data/raabin)")
    parser.add_argument("--train", type=float, default=0.70, help="Train fraction")
    parser.add_argument("--val",   type=float, default=0.10, help="Val fraction")
    parser.add_argument("--test",  type=float, default=0.20, help="Test fraction")
    parser.add_argument("--seed",  type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # Check that flat class folders exist (not already split)
    found = [cls for cls in CLASSES if (data_dir / cls).is_dir()]
    if not found:
        print("No flat class folders found — dataset may already be split.")
        return

    print(f"Splitting {data_dir} → train/val/test "
          f"({args.train:.0%} / {args.val:.0%} / {args.test:.0%}), seed={args.seed}\n")
    split_dataset(data_dir, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
