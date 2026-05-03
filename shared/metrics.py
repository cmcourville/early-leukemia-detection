"""
shared/metrics.py — CS 534, Team 6 — WPI

Shared evaluation metrics for all three model implementations.

All models should use this module to compute their performance metrics
so that results are computed identically and are directly comparable
in the Phase 2 progress report.

Primary function:
    compute_all_metrics(y_true, y_pred, y_prob, class_names)
        → Dict with accuracy, F1, AUROC, sensitivity, specificity,
          confusion matrix, and classification report.

Also exports:
    format_metrics_table(metrics_dict)   — ASCII summary for printing
    save_metrics(metrics_dict, path)     — JSON dump
    save_confusion_matrix_plot(...)      — PNG heatmap
    save_auroc_plot(...)                 — PNG ROC curves
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Core metric computation

def compute_all_metrics(
    y_true:      np.ndarray,          # (N,)  integer class labels
    y_pred:      np.ndarray,          # (N,)  integer predicted labels
    y_prob:      np.ndarray,          # (N, C) softmax probabilities
    class_names: Optional[List[str]] = None,
    model_name:  str = "Model",
    split_name:  str = "Test",
) -> Dict:
    """
    Compute all metrics required for the Phase 2 report.

    Returns a dictionary with keys:
        model_name, split_name, class_names,
        accuracy, f1_macro, f1_per_class,
        auroc_macro, auroc_per_class,
        sensitivity_macro, sensitivity_per_class,
        specificity_macro, specificity_per_class,
        confusion_matrix (np.ndarray),
        report_str (str)
    """
    C = y_prob.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    # Accuracy
    accuracy = float(accuracy_score(y_true, y_pred))

    # F1
    f1_macro     = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # AUROC
    # AUROC — binary case needs special handling (sklearn quirk)
    try:
        if C == 2:
            auroc_macro     = float(roc_auc_score(y_true, y_prob[:, 1]))
            auroc_per_class = [auroc_macro, auroc_macro]
        else:
            auroc_macro     = float(roc_auc_score(y_true, y_prob, average="macro",
                                                   multi_class="ovr"))
            auroc_per_class = roc_auc_score(y_true, y_prob, average=None,
                                             multi_class="ovr").tolist()
    except ValueError as e:
        print(f"[metrics] AUROC warning: {e}")
        auroc_macro     = float("nan")
        auroc_per_class = [float("nan")] * C

    # Confusion matrix → sensitivity & specificity
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    sensitivity_per_class, specificity_per_class = [], []

    for c in range(C):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum()) - tp
        fp = int(cm[:, c].sum()) - tp
        tn = int(cm.sum()) - tp - fp - fn

        sensitivity_per_class.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    sensitivity_macro = float(np.mean(sensitivity_per_class))
    specificity_macro = float(np.mean(specificity_per_class))

    # Classification report
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return {
        "model_name":            model_name,
        "split_name":            split_name,
        "class_names":           class_names,
        "accuracy":              accuracy,
        "f1_macro":              f1_macro,
        "f1_per_class":          dict(zip(class_names, f1_per_class)),
        "auroc_macro":           auroc_macro,
        "auroc_per_class":       dict(zip(class_names, auroc_per_class)),
        "sensitivity_macro":     sensitivity_macro,
        "sensitivity_per_class": dict(zip(class_names, sensitivity_per_class)),
        "specificity_macro":     specificity_macro,
        "specificity_per_class": dict(zip(class_names, specificity_per_class)),
        "confusion_matrix":      cm,
        "report_str":            report_str,
    }

# Printing helpers

def print_metrics(metrics: Dict) -> None:
    """Print a formatted metrics summary to stdout."""
    sep = "─" * 64
    model = metrics.get("model_name", "Model")
    split = metrics.get("split_name", "Test")
    print(f"\n{sep}")
    print(f"  {model} — {split} Set")
    print(sep)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Macro F1          : {metrics['f1_macro']:.4f}")
    print(f"  Macro AUROC       : {metrics['auroc_macro']:.4f}")
    print(f"  Macro Sensitivity : {metrics['sensitivity_macro']:.4f}")
    print(f"  Macro Specificity : {metrics['specificity_macro']:.4f}")
    print(sep)
    print("  Per-class breakdown:")
    for cn in metrics["class_names"]:
        print(f"    {cn:<20s}  "
              f"F1={metrics['f1_per_class'][cn]:.4f}  "
              f"Sens={metrics['sensitivity_per_class'][cn]:.4f}  "
              f"Spec={metrics['specificity_per_class'][cn]:.4f}  "
              f"AUROC={metrics['auroc_per_class'][cn]:.4f}")
    print(sep)
    print("\nClassification Report:")
    print(metrics["report_str"])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(metrics["confusion_matrix"])
    print(sep)

def format_comparison_table(all_metrics: Dict[str, Dict]) -> str:
    """
    Format a side-by-side comparison table for all three models.

    Args:
        all_metrics: {model_key: metrics_dict}  e.g. {'bcct': {...}, ...}

    Returns:
        Formatted string table suitable for printing / pasting into report.
    """
    models = list(all_metrics.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:>20}" for m in models)
    sep    = "─" * (30 + 20 * len(models))
    lines  = [sep, header, sep]

    metric_keys = [
        ("Accuracy",          "accuracy"),
        ("Macro F1",          "f1_macro"),
        ("Macro AUROC",       "auroc_macro"),
        ("Macro Sensitivity", "sensitivity_macro"),
        ("Macro Specificity", "specificity_macro"),
    ]
    for label, key in metric_keys:
        row = f"{label:<30}"
        for m in models:
            val = all_metrics[m].get(key, float("nan"))
            row += f"{val:>20.4f}" if isinstance(val, float) else f"{'N/A':>20}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)

# Persistence helpers

def save_metrics(metrics: Dict, path: str) -> None:
    """Save metrics to a JSON file (confusion_matrix converted to list)."""
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in metrics.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[metrics] Saved → {path}")

def load_metrics(path: str) -> Dict:
    """Load a metrics JSON and restore confusion_matrix as np.ndarray."""
    with open(path) as f:
        m = json.load(f)
    if "confusion_matrix" in m:
        m["confusion_matrix"] = np.array(m["confusion_matrix"])
    return m

# Plot helpers

def save_confusion_matrix_plot(
    cm:          np.ndarray,
    class_names: List[str],
    model_name:  str,
    split_name:  str,
    out_path:    str,
) -> None:
    """Save a labelled confusion matrix heatmap (counts + normalised)."""
    if not HAS_MPL:
        print("[metrics] matplotlib not installed — skipping confusion matrix plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (data, title) in zip(axes, [
        (cm, "Counts"),
        (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1), "Normalised"),
    ]):
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if title == "Counts" else "Oranges")
        plt.colorbar(im, ax=ax)
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names,
               xlabel="Predicted", ylabel="True",
               title=f"{model_name} — {split_name} ({title})")
        ax.xaxis.set_tick_params(rotation=45)
        thresh = data.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = f"{data[i,j]:.2f}" if title == "Normalised" else f"{int(data[i,j])}"
                ax.text(j, i, val, ha="center", va="center", fontsize=8,
                        color="white" if data[i,j] > thresh else "black")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[metrics] Confusion matrix saved → {out_path}")

def save_auroc_plot(
    y_true:      np.ndarray,
    y_prob:      np.ndarray,
    class_names: List[str],
    model_name:  str,
    split_name:  str,
    out_path:    str,
) -> None:
    """Save per-class ROC curves on a single figure."""
    if not HAS_MPL:
        print("[metrics] matplotlib not installed — skipping AUROC plot.")
        return

    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    C = len(class_names)
    # label_binarize returns (N,1) for binary — expand to (N,2) manually
    if C == 2:
        y_bin = np.column_stack([1 - y_true.astype(float), y_true.astype(float)])
    else:
        y_bin = label_binarize(y_true, classes=list(range(C)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, C))

    for i, (cn, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        try:
            auc_val = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
            lbl = f"{cn} (AUC={auc_val:.3f})"
        except ValueError:
            lbl = f"{cn} (AUC=N/A)"
        ax.plot(fpr, tpr, color=color, lw=2, label=lbl)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC Curves — {model_name} ({split_name})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[metrics] AUROC plot saved → {out_path}")
