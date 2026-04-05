"""
evaluate_bcct.py — BccT Project (CS 534, Team 6)
Author: Corrin

Comprehensive evaluation script for BccT.

Computes all metrics required by the Phase 2 Progress Report:
  • Accuracy (overall)
  • Macro-averaged F1 score
  • Per-class F1 score
  • AUROC (macro One-vs-Rest)
  • Sensitivity (= Recall) per class and macro-average
  • Specificity per class and macro-average
  • Confusion matrix (raw counts + normalised)
  • Classification report (precision / recall / F1 per class)

Results are both printed to stdout and returned as a dictionary for
programmatic use (e.g. by main.py or low_data_test.py).

Optionally saves:
  • Confusion matrix PNG
  • AUROC curve PNG (one plot, all classes)
  • JSON metrics dump
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Core evaluation function

@torch.no_grad()
def evaluate_model(
    model,
    loader:      DataLoader,
    device:      torch.device,
    split_name:  str = "Test",
    label_map:   Optional[Dict[int, str]] = None,
    output_dir:  Optional[str] = None,
    save_plots:  bool = True,
) -> Dict:
    """
    Run evaluation on `loader` and compute all metrics.

    Args:
        model      : BccTModel (or any callable that returns logits (B, C)).
        loader     : DataLoader yielding (pixel_values, labels).
        device     : Compute device.
        split_name : Label for printing ("Train" / "Validation" / "Test").
        label_map  : {int: str} class-index → class-name mapping.
        output_dir : If provided, save confusion matrix + AUROC plots here.
        save_plots : Whether to save PNG plots (requires matplotlib).

    Returns:
        Dictionary with keys:
            accuracy, f1_macro, f1_per_class, auroc_macro, auroc_per_class,
            sensitivity_macro, sensitivity_per_class,
            specificity_macro, specificity_per_class,
            confusion_matrix (np.ndarray), report_str
    """
    model.eval()
    model.to(device)

    all_preds:  List[int]       = []
    all_labels: List[int]       = []
    all_probs:  List[np.ndarray] = []   # softmax probabilities for AUROC

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)                    # (B, C)
        probs  = F.softmax(logits, dim=1)       # (B, C)
        preds  = logits.argmax(dim=1)           # (B,)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy())

    y_true  = np.array(all_labels, dtype=int)
    y_pred  = np.array(all_preds,  dtype=int)
    y_prob  = np.array(all_probs)               # (N, C)

    num_classes = y_prob.shape[1]
    class_names = (
        [label_map[i] for i in range(num_classes)]
        if label_map else [str(i) for i in range(num_classes)]
    )

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # F1
    f1_macro     = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None,     zero_division=0).tolist()

    # AUROC (One-vs-Rest, macro)
    try:
        auroc_macro = roc_auc_score(
            y_true, y_prob, average="macro", multi_class="ovr"
        )
        auroc_per_class = roc_auc_score(
            y_true, y_prob, average=None, multi_class="ovr"
        ).tolist()
    except ValueError as e:
        # Can fail if a class is absent from y_true in few-shot regime
        print(f"[evaluate] AUROC warning: {e}")
        auroc_macro     = float("nan")
        auroc_per_class = [float("nan")] * num_classes

    # Sensitivity (Recall) and Specificity per class
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    sensitivity_per_class = []
    specificity_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp          # false negatives (row sum − TP)
        fp = cm[:, c].sum() - tp          # false positives (col sum − TP)
        tn = cm.sum() - tp - fp - fn      # true negatives

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sensitivity_per_class.append(float(sens))
        specificity_per_class.append(float(spec))

    sensitivity_macro = float(np.mean(sensitivity_per_class))
    specificity_macro = float(np.mean(specificity_per_class))

    # Classification report
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    # Print summary
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  BccT Evaluation — {split_name} Set")
    print(sep)
    print(f"  Accuracy              : {accuracy:.4f}")
    print(f"  Macro F1              : {f1_macro:.4f}")
    print(f"  Macro AUROC           : {auroc_macro:.4f}")
    print(f"  Macro Sensitivity     : {sensitivity_macro:.4f}")
    print(f"  Macro Specificity     : {specificity_macro:.4f}")
    print(sep)
    print("  Per-class metrics:")
    for i, cn in enumerate(class_names):
        print(f"    {cn:<20s}  "
              f"F1={f1_per_class[i]:.4f}  "
              f"Sens={sensitivity_per_class[i]:.4f}  "
              f"Spec={specificity_per_class[i]:.4f}  "
              f"AUROC={auroc_per_class[i]:.4f}")
    print(sep)
    print("\nClassification Report:")
    print(report_str)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(sep)

    # Plots and JSON
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_dict = {
            "split":                 split_name,
            "accuracy":              accuracy,
            "f1_macro":              f1_macro,
            "f1_per_class":          {cn: v for cn, v in zip(class_names, f1_per_class)},
            "auroc_macro":           auroc_macro,
            "auroc_per_class":       {cn: v for cn, v in zip(class_names, auroc_per_class)},
            "sensitivity_macro":     sensitivity_macro,
            "sensitivity_per_class": {cn: v for cn, v in zip(class_names, sensitivity_per_class)},
            "specificity_macro":     specificity_macro,
            "specificity_per_class": {cn: v for cn, v in zip(class_names, specificity_per_class)},
        }
        json_path = out_path / f"metrics_{split_name.lower()}.json"
        with open(json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"[evaluate] Metrics saved → {json_path}")

        if save_plots and HAS_MPL:
            _save_confusion_matrix_plot(
                cm, class_names, split_name, out_path
            )
            _save_auroc_plot(y_true, y_prob, class_names, split_name, out_path)

    # Build return dict
    result = {
        "accuracy":              accuracy,
        "f1_macro":              f1_macro,
        "f1_per_class":          f1_per_class,
        "auroc_macro":           auroc_macro,
        "auroc_per_class":       auroc_per_class,
        "sensitivity_macro":     sensitivity_macro,
        "sensitivity_per_class": sensitivity_per_class,
        "specificity_macro":     specificity_macro,
        "specificity_per_class": specificity_per_class,
        "confusion_matrix":      cm,
        "report_str":            report_str,
    }
    return result

# Plot helpers

def _save_confusion_matrix_plot(
    cm:          np.ndarray,
    class_names: List[str],
    split_name:  str,
    out_path:    Path,
) -> None:
    """Save a labelled confusion matrix heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (data, title) in zip(axes, [
        (cm,                                  "Counts"),
        (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1), "Normalised"),
    ]):
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if title == "Counts" else "Oranges")
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix ({title}) — {split_name}",
        )
        ax.xaxis.set_tick_params(rotation=45)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = f"{data[i,j]:.2f}" if title == "Normalised" else f"{data[i,j]}"
                ax.text(j, i, val,
                        ha="center", va="center",
                        color="white" if data[i,j] > thresh else "black",
                        fontsize=8)

    plt.tight_layout()
    save_file = out_path / f"confusion_matrix_{split_name.lower()}.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {save_file}")

def _save_auroc_plot(
    y_true:      np.ndarray,
    y_prob:      np.ndarray,
    class_names: List[str],
    split_name:  str,
    out_path:    Path,
) -> None:
    """Save per-class ROC curves on a single figure."""
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, (cn, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        try:
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
            label = f"{cn} (AUC={auc:.3f})"
        except ValueError:
            label = f"{cn} (AUC=N/A)"
        ax.plot(fpr, tpr, color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title=f"ROC Curves — {split_name} Set")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    save_file = out_path / f"auroc_{split_name.lower()}.png"
    plt.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] AUROC plot saved → {save_file}")

# CLI entry point

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a trained BccT model checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to saved BccT checkpoint (.pt)."
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "validation", "test"],
        help="Dataset split to evaluate on (default: test)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--num_workers", type=int, default=4
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory to save metrics JSON and plots."
    )
    parser.add_argument(
        "--device", type=str, default=None
    )
    args = parser.parse_args()

    from bcct_model import BccTModel
    from data_loader import get_dataloaders

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[evaluate] Loading checkpoint: {args.checkpoint}")
    model = BccTModel.load(args.checkpoint, device=device)

    train_dl, val_dl, test_dl, label_map = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=(device.type == "cuda"),
    )

    split_map = {
        "train": (train_dl, "Train"),
        "val":   (val_dl,   "Validation"),
        "validation": (val_dl, "Validation"),
        "test":  (test_dl,  "Test"),
    }
    loader, split_name = split_map[args.split]

    evaluate_model(
        model=model,
        loader=loader,
        device=device,
        split_name=split_name,
        label_map=label_map,
        output_dir=args.output_dir,
        save_plots=True,
    )

if __name__ == "__main__":
    main()
