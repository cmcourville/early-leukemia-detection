"""
models/cytodiffusion/evaluate.py — CS 534, Team 6 — WPI
Author: Darshan

Evaluation entry point for CytoDiffusionModel.

Usage:
    python evaluate.py                              # test split, raabin
    python evaluate.py --split val                  # validation split
    python evaluate.py --dataset bccd               # BCCD 3-class
    python evaluate.py --checkpoint path/to/ckpt.pt
    python evaluate.py --dataset cytodata --data_dir /path/to/cytodata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "models"))

from shared.config import (
    CHECKPOINT_DIRS, DATASET_CONFIGS, GLOBAL_SEED, RESULTS_ROOT,
)
from shared.data import get_dataloaders
from shared.metrics import (
    compute_all_metrics, print_metrics, save_metrics,
    save_confusion_matrix_plot, save_auroc_plot,
)
from cytodiffusion.model import CytoDiffusionModel


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CytoDiffusionModel.")
    p.add_argument("--checkpoint",  type=str,   default=None,
                   help="Path to .pt checkpoint (auto-detected if omitted).")
    p.add_argument("--dataset",     type=str,   default="raabin",
                   choices=["raabin", "bccd", "cytodata", "cnmc"])
    p.add_argument("--data_dir",    type=str,   default=None)
    p.add_argument("--split",       type=str,   default="test",
                   choices=["val", "test"])
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--cache_dir",   type=str,   default=None)
    p.add_argument("--output_dir",  type=str,   default=None)
    p.add_argument("--device",      type=str,   default=None)
    p.add_argument("--seed",        type=int,   default=GLOBAL_SEED)
    return p.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())
    return (
        np.array(all_true, dtype=int),
        np.array(all_pred, dtype=int),
        np.array(all_prob, dtype=float),
    )


def main() -> None:
    args = get_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[evaluate] Device : {device}")
    print(f"[evaluate] Dataset: {args.dataset}  Split: {args.split}")

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    class_names = dataset_cfg["class_names"]

    _, val_loader, test_loader, _ = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=(device.type == "cuda"),
    )
    loader      = test_loader if args.split == "test" else val_loader
    split_label = "Test" if args.split == "test" else "Validation"

    ckpt = args.checkpoint or str(
        _PROJECT_ROOT / CHECKPOINT_DIRS["cytodiffusion"]
        / f"cytodiffusion_{args.dataset}.pt"
    )
    print(f"[evaluate] Checkpoint: {ckpt}")
    model = CytoDiffusionModel.load(ckpt, device)
    model.to(device)

    print(f"[evaluate] Running inference …")
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    metrics = compute_all_metrics(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        class_names=class_names,
        model_name="CytoDiffusion",
        split_name=split_label,
    )
    print_metrics(metrics)

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else _PROJECT_ROOT / RESULTS_ROOT / "cytodiffusion"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_{args.dataset}"

    save_metrics(metrics, str(out_dir / f"metrics_{tag}.json"))
    save_confusion_matrix_plot(
        cm=metrics["confusion_matrix"],
        class_names=class_names,
        model_name="CytoDiffusion",
        split_name=split_label,
        out_path=str(out_dir / f"confusion_matrix_{tag}.png"),
    )
    save_auroc_plot(
        y_true=y_true, y_prob=y_prob,
        class_names=class_names,
        model_name="CytoDiffusion",
        split_name=split_label,
        out_path=str(out_dir / f"auroc_{tag}.png"),
    )
    print(f"[evaluate] Results saved → {out_dir}")


if __name__ == "__main__":
    main()
