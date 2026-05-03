"""
models/cytodiffusion/data_prep/prepare_cytodata.py
Author: Darshan

Convert the raw EBI BioStudies S-BSST2156 download into the ImageFolder
layout expected by shared/data/data_loader.py:

    <out_dir>/
    ├── train/
    │   ├── Basophil/
    │   ├── Eosinophil/
    │   └── ...  (10 classes)
    ├── val/
    └── test/

Input (--raw_dir) can be either:
  A) Pre-split CSVs  : raw_dir/train_labels.csv, val_labels.csv, test_labels.csv
  B) Single CSV      : raw_dir/labels.csv  (script creates 70/10/20 split)
  C) Folder-per-abbrev: raw_dir/images/BAS/, raw_dir/images/EOS/, ... (no CSV)

Cell-type abbreviation → folder-name mapping (matches shared/config.py):
    BAS → Basophil     EOS → Eosinophil   EBO → Erythroblast
    LYT → Lymphocyte   MON → Monocyte     MYB → Myeloblast
    NEU → Neutrophil   PLT → Platelet     PMO → Promyelocyte
    NIF → Artefact

Usage:
    python models/cytodiffusion/data_prep/prepare_cytodata.py \\
        --raw_dir  ~/data/cytodata_raw \\
        --out_dir  ~/data/cytodata     \\
        --min_conf 0.5                 \\   # skip low-confidence labels
        --seed     42
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path

# Abbreviation → full class name (order matches shared/config.py CLASS_NAMES)
ABBREV_TO_FULL: dict[str, str] = {
    "BAS": "Basophil",
    "EOS": "Eosinophil",
    "EBO": "Erythroblast",
    "LYT": "Lymphocyte",
    "MON": "Monocyte",
    "MYB": "Myeloblast",
    "NEU": "Neutrophil",
    "PLT": "Platelet",
    "PMO": "Promyelocyte",
    "NIF": "Artefact",
}

SPLIT_FRACS = {"train": 0.70, "val": 0.10, "test": 0.20}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare CytoData into ImageFolder layout."
    )
    p.add_argument("--raw_dir",  required=True,
                   help="Path to raw EBI BioStudies download.")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for the prepared ImageFolder dataset.")
    p.add_argument("--min_conf", type=float, default=0.5,
                   help="Minimum labeller confidence to include a sample (default 0.5).")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--copy",     action="store_true", default=False,
                   help="Copy files instead of creating symlinks (slower, more disk).")
    return p.parse_args()


# ── CSV-based loaders ──────────────────────────────────────────────────────────

def _load_csv(csv_path: Path, raw_dir: Path, min_conf: float) -> list[tuple[Path, str]]:
    """
    Read a labels CSV and return [(image_path, class_name), ...].

    Supported CSV column names:
        image / image_path / filename  → path to image (relative to raw_dir or absolute)
        label / cell_type / class      → abbreviation (BAS) or full name (Basophil)
        confidence / conf / score      → float in [0, 1] (optional)
    """
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = {k.lower().strip(): k for k in (reader.fieldnames or [])}

        img_col  = next((fields[k] for k in ("image", "image_path", "filename", "path") if k in fields), None)
        lbl_col  = next((fields[k] for k in ("label", "cell_type",  "class",    "type") if k in fields), None)
        conf_col = next((fields[k] for k in ("confidence", "conf", "score") if k in fields), None)

        if img_col is None or lbl_col is None:
            raise ValueError(
                f"[prepare] Cannot find image/label columns in {csv_path}.\n"
                f"  Available columns: {list(reader.fieldnames)}"
            )

        for row in reader:
            img_str  = row[img_col].strip()
            lbl_str  = row[lbl_col].strip().upper()
            conf_val = float(row[conf_col]) if conf_col else 1.0

            if conf_val < min_conf:
                continue

            # Resolve to full class name
            full_name = ABBREV_TO_FULL.get(lbl_str) or ABBREV_TO_FULL.get(lbl_str[:3])
            if full_name is None:
                # Maybe it's already a full name
                for abbr, full in ABBREV_TO_FULL.items():
                    if lbl_str == full.upper():
                        full_name = full
                        break
            if full_name is None:
                continue  # unknown label — skip

            # Resolve image path
            img_path = Path(img_str)
            if not img_path.is_absolute():
                for candidate in [
                    raw_dir / img_str,
                    raw_dir / "images" / img_str,
                    raw_dir / "data" / img_str,
                ]:
                    if candidate.exists():
                        img_path = candidate
                        break
                else:
                    continue  # image file not found — skip

            if img_path.exists():
                samples.append((img_path, full_name))

    return samples


def _load_folder(raw_dir: Path) -> list[tuple[Path, str]]:
    """
    Fallback: if no CSV found, treat raw_dir/images/<ABBREV>/ as ImageFolder.
    Also handles raw_dir/<ABBREV>/ directly.
    """
    samples = []
    for search_root in [raw_dir / "images", raw_dir]:
        if not search_root.exists():
            continue
        for subdir in sorted(search_root.iterdir()):
            if not subdir.is_dir():
                continue
            key = subdir.name.upper()
            full_name = ABBREV_TO_FULL.get(key) or ABBREV_TO_FULL.get(key[:3])
            if full_name is None:
                # Try matching full name directly
                for full in ABBREV_TO_FULL.values():
                    if key == full.upper():
                        full_name = full
                        break
            if full_name is None:
                continue
            for fp in sorted(subdir.iterdir()):
                if fp.suffix.lower() in IMAGE_EXTS:
                    samples.append((fp, full_name))
        if samples:
            break
    return samples


# ── Writer ─────────────────────────────────────────────────────────────────────

def _write_split(
    samples:  list[tuple[Path, str]],
    out_root: Path,
    split:    str,
    do_copy:  bool,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src_path, class_name in samples:
        dst_dir = out_root / split / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / src_path.name

        # Avoid collisions when image names aren't unique
        stem, ext = src_path.stem, src_path.suffix
        counter = 0
        while dst_path.exists():
            counter += 1
            dst_path = dst_dir / f"{stem}_{counter}{ext}"

        if do_copy:
            shutil.copy2(src_path, dst_path)
        else:
            if dst_path.exists():
                dst_path.unlink()
            os.symlink(src_path.resolve(), dst_path)

        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = get_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    random.seed(args.seed)

    print(f"[prepare] Raw dir : {raw_dir}")
    print(f"[prepare] Out dir : {out_dir}")
    print(f"[prepare] Min conf: {args.min_conf}")

    # ── Detect input format ────────────────────────────────────────────────────

    train_csv = raw_dir / "train_labels.csv"
    val_csv   = raw_dir / "val_labels.csv"
    test_csv  = raw_dir / "test_labels.csv"
    combined  = raw_dir / "labels.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        print("[prepare] Mode: pre-split CSV files")
        split_samples = {
            "train": _load_csv(train_csv, raw_dir, args.min_conf),
            "val":   _load_csv(val_csv,   raw_dir, args.min_conf),
            "test":  _load_csv(test_csv,  raw_dir, args.min_conf),
        }

    elif combined.exists():
        print("[prepare] Mode: single labels.csv — splitting 70/10/20")
        all_samples = _load_csv(combined, raw_dir, args.min_conf)
        print(f"[prepare] Total samples after confidence filter: {len(all_samples)}")

        # Stratified split per class
        by_class: dict[str, list] = {}
        for item in all_samples:
            by_class.setdefault(item[1], []).append(item)
        for cls in by_class:
            random.shuffle(by_class[cls])

        split_samples: dict[str, list] = {"train": [], "val": [], "test": []}
        for cls, items in sorted(by_class.items()):
            n = len(items)
            n_train = int(n * SPLIT_FRACS["train"])
            n_val   = int(n * SPLIT_FRACS["val"])
            split_samples["train"].extend(items[:n_train])
            split_samples["val"].extend(items[n_train:n_train + n_val])
            split_samples["test"].extend(items[n_train + n_val:])

    else:
        print("[prepare] Mode: folder-per-abbreviation (no CSV) — splitting 70/10/20")
        all_samples = _load_folder(raw_dir)
        if not all_samples:
            print(
                "[prepare] ERROR: Could not locate any labeled images.\n"
                "  Expected one of:\n"
                "    raw_dir/train_labels.csv + val_labels.csv + test_labels.csv\n"
                "    raw_dir/labels.csv\n"
                "    raw_dir/images/<BAS|EOS|...>/*.jpg"
            )
            sys.exit(1)

        by_class: dict[str, list] = {}
        for item in all_samples:
            by_class.setdefault(item[1], []).append(item)
        for cls in by_class:
            random.shuffle(by_class[cls])

        split_samples = {"train": [], "val": [], "test": []}
        for cls, items in sorted(by_class.items()):
            n = len(items)
            n_train = int(n * SPLIT_FRACS["train"])
            n_val   = int(n * SPLIT_FRACS["val"])
            split_samples["train"].extend(items[:n_train])
            split_samples["val"].extend(items[n_train:n_train + n_val])
            split_samples["test"].extend(items[n_train + n_val:])

    # ── Write output ───────────────────────────────────────────────────────────

    total = 0
    for split, samples in split_samples.items():
        counts = _write_split(samples, out_dir, split, args.copy)
        n = sum(counts.values())
        total += n
        print(f"[prepare] {split:5s}: {n:5d} images  |  " +
              "  ".join(f"{c}={v}" for c, v in sorted(counts.items())))

    print(f"\n[prepare] Total images written: {total}")
    print(f"[prepare] Dataset ready at: {out_dir}")
    print()
    print("Verify layout:")
    print(f"  ls {out_dir}/train/")
    print()
    print("Run training:")
    print(f"  python models/cytodiffusion/train.py \\")
    print(f"    --dataset cytodata --data_dir {out_dir}")


if __name__ == "__main__":
    main()
