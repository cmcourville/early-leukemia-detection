"""
prepare_bccd.py — CS 534, Team 6
Downloads the BCCD dataset from GitHub, parses Pascal VOC XML bounding-box
annotations, crops each annotated cell, and saves the crops in ImageFolder
format so the pipeline can load it via LocalImageFolderDataset.

Output layout:
    <data_dir>/train/Platelet/
    <data_dir>/train/RBC/
    <data_dir>/train/WBC/
    <data_dir>/val/Platelet/
    <data_dir>/val/RBC/
    <data_dir>/val/WBC/
    <data_dir>/test/Platelet/
    <data_dir>/test/RBC/
    <data_dir>/test/WBC/

Source: https://github.com/Shenggan/BCCD_Dataset (public, no login required)
    Contains 364 microscopy images + Pascal VOC XML annotations.
    The repo ships train/val/test splits in ImageSets/Main/.

Usage:
    python prepare_bccd.py                              # auto-downloads from GitHub
    python prepare_bccd.py --src_dir BCCD_Dataset       # use existing local clone
    python prepare_bccd.py --src_dir BCCD_Dataset --data_dir data/bccd
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

GITHUB_ZIP = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"
CLASSES    = ["Platelet", "RBC", "WBC"]
MIN_CROP   = 8      # skip bounding boxes smaller than this in either dimension


def download_bccd(dest_dir: Path) -> Path:
    """Download the BCCD GitHub repo as a ZIP and extract it."""
    zip_path = dest_dir / "bccd_master.zip"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BCCD from GitHub …")
    urllib.request.urlretrieve(GITHUB_ZIP, zip_path)
    print(f"Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()

    extracted = dest_dir / "BCCD_Dataset-master"
    print(f"Extracted to {extracted}")
    return extracted


def _load_split_names(imageset_dir: Path, split: str) -> set[str]:
    """Read ImageSets/Main/<split>.txt — returns a set of bare image names."""
    txt = imageset_dir / f"{split}.txt"
    if not txt.exists():
        return set()
    return {line.strip() for line in txt.read_text().splitlines() if line.strip()}


def _parse_annotation(xml_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Parse a Pascal VOC XML file → list of (class_name, (x1, y1, x2, y2))."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASSES:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = int(float(bb.findtext("xmin", "0")))
        y1 = int(float(bb.findtext("ymin", "0")))
        x2 = int(float(bb.findtext("xmax", "0")))
        y2 = int(float(bb.findtext("ymax", "0")))
        results.append((name, (x1, y1, x2, y2)))
    return results


def process(src_dir: Path, data_dir: Path) -> None:
    """
    Crop all annotated bounding boxes and save to ImageFolder splits.

    Expects:
        src_dir/BCCD/JPEGImages/*.jpg
        src_dir/BCCD/Annotations/*.xml
        src_dir/BCCD/ImageSets/Main/{train,val,test}.txt
    """
    bccd_dir    = src_dir / "BCCD"
    images_dir  = bccd_dir / "JPEGImages"
    annot_dir   = bccd_dir / "Annotations"
    imageset_dir = bccd_dir / "ImageSets" / "Main"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"JPEGImages not found at {images_dir}.\n"
            f"Ensure src_dir points to the root of the BCCD_Dataset repo."
        )

    # Map image name → split
    name_to_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for name in _load_split_names(imageset_dir, split):
            name_to_split[name] = split

    # If no split files found, fall back to 70/15/15 split
    if not name_to_split:
        print("[prepare_bccd] No ImageSets split files found — using 70/15/15 auto-split.")
        all_names = sorted(p.stem for p in images_dir.glob("*.jpg"))
        import random
        random.seed(42)
        random.shuffle(all_names)
        n = len(all_names)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        for nm in all_names[:n_train]:
            name_to_split[nm] = "train"
        for nm in all_names[n_train : n_train + n_val]:
            name_to_split[nm] = "val"
        for nm in all_names[n_train + n_val :]:
            name_to_split[nm] = "test"

    counters: dict[str, dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in ("train", "val", "test")}
    skipped = 0

    for xml_path in sorted(annot_dir.glob("*.xml")):
        img_name = xml_path.stem
        split    = name_to_split.get(img_name, "train")   # default to train if not listed

        img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            skipped += 1
            continue

        annotations = _parse_annotation(xml_path)
        if not annotations:
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        for cls_name, (x1, y1, x2, y2) in annotations:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if (x2 - x1) < MIN_CROP or (y2 - y1) < MIN_CROP:
                continue

            crop = img.crop((x1, y1, x2, y2))

            dest_dir = data_dir / split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            crop_name = f"{img_name}_{x1}_{y1}_{x2}_{y2}.jpg"
            crop.save(dest_dir / crop_name, "JPEG", quality=95)
            counters[split][cls_name] += 1

    print(f"\nDone. Crops per split:")
    for split in ("train", "val", "test"):
        totals = " | ".join(f"{c}={counters[split][c]}" for c in CLASSES)
        print(f"  {split}: {totals}")
    if skipped:
        print(f"  ({skipped} XML files skipped — image not found)")

    print(f"\nRun with: python run_all.py --dataset bccd --data_dir {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BCCD dataset (crop bboxes → ImageFolder)")
    parser.add_argument("--src_dir",  default=None,
                        help="Path to existing BCCD_Dataset repo clone (skips download)")
    parser.add_argument("--data_dir", default="data/bccd",
                        help="Output directory (default: data/bccd)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.src_dir:
        src_dir = Path(args.src_dir)
        if not src_dir.is_dir():
            raise FileNotFoundError(f"src_dir not found: {src_dir}")
    else:
        # Auto-download to a temp location beside data_dir
        dl_dir  = data_dir.parent / "bccd_download"
        src_dir = download_bccd(dl_dir)

    process(src_dir, data_dir)


if __name__ == "__main__":
    main()
