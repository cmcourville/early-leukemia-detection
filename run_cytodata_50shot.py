"""
run_cytodata_50shot.py — CS 534, Team 6 — WPI
Author: Corrin

Standalone script to complete the missing 50-shot low-data experiment
for BccT on the CytoData 10-class dataset, then aggregate the full
low_data_summary.json from the existing 10-shot, 20-shot, and new 50-shot
results.

Prerequisite: CytoData full-pipeline results must already exist at
    shared/results/cytodata/bcct/metrics_test.json
    shared/results/cytodata/bcct/low_data/10shot/avg_metrics.json
    shared/results/cytodata/bcct/low_data/20shot/avg_metrics.json

Run from the project root (early-leukemia-detection/):
    python run_cytodata_50shot.py

Optional flags:
    --data_dir     data/cytodata   (default)
    --num_repeats  3               (default)
    --seed         42              (default)
    --device       cpu             (auto-detected if omitted)
    --batch_size   32              (default)
    --num_workers  4               (default)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — make shared/ and models/bcct/ importable from any CWD
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "shared"))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "bcct"))

from shared.data import get_dataloaders, get_few_shot_loaders
from shared.metrics import compute_all_metrics
from bcct_model import BccTModel
from token_fusion import reset_token_sizes


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model: BccTModel, loader, device: torch.device):
    """Return (y_true, y_pred, y_prob) arrays from a single pass over loader."""
    import torch.nn.functional as F

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run BccT 50-shot experiment on CytoData and rebuild summary."
    )
    p.add_argument("--data_dir",    type=str, default="data/cytodata",
                   help="Path to the CytoData root (default: data/cytodata).")
    p.add_argument("--num_repeats", type=int, default=3,
                   help="Number of independent random seeds per shot count.")
    p.add_argument("--seed",        type=int, default=42,
                   help="Base random seed (incremented per repeat).")
    p.add_argument("--device",      type=str, default=None,
                   help="Compute device: 'cpu', 'cuda', etc. Auto-detected if omitted.")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--r",           type=int, default=16,
                   help="BccT Token Fusion merge budget (default: 16).")
    p.add_argument("--ridge_lambda",type=float, default=1e-4,
                   help="FRC ridge regularisation lambda (default: 1e-4).")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 64)
    print("  BccT — CytoData 50-shot Experiment")
    print(f"  Device   : {device}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Repeats  : {args.num_repeats}  (seeds {args.seed} … {args.seed + args.num_repeats - 1})")
    print("=" * 64)

    # Reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -----------------------------------------------------------------------
    # 1. Load full CytoData test set (used to evaluate after each re-fit)
    # -----------------------------------------------------------------------
    print("\n[50-shot] Loading CytoData test set …")
    _, _, test_loader, label_map = get_dataloaders(
        dataset     = "cytodata",
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
    )
    class_names = [label_map[i] for i in sorted(label_map.keys())]
    num_classes = len(class_names)
    print(f"[50-shot] Classes ({num_classes}): {class_names}")

    # -----------------------------------------------------------------------
    # 2. Instantiate a fresh BccTModel with 10 classes
    #    (BccT has zero trainable parameters — the ViT backbone is always frozen
    #     and the FRC is re-solved analytically for each n-shot subset)
    # -----------------------------------------------------------------------
    print("\n[50-shot] Initialising BccTModel …")
    model = BccTModel(
        num_classes  = num_classes,
        r            = args.r,
        ridge_lambda = args.ridge_lambda,
        frc_seed     = args.seed,
    )
    model.to(device)

    # -----------------------------------------------------------------------
    # 3. Run 50-shot experiment (3 independent seeds)
    # -----------------------------------------------------------------------
    N_SHOT       = 50
    repeat_metrics = []

    for rep in range(args.num_repeats):
        rep_seed = args.seed + rep
        print(f"\n[50-shot] Repeat {rep + 1}/{args.num_repeats}  (seed={rep_seed})")

        # Build 50-shot training loader (stratified sampling)
        few_train_dl, _, _, _ = get_few_shot_loaders(
            n_shot      = N_SHOT,
            dataset     = "cytodata",
            data_dir    = args.data_dir,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            seed        = rep_seed,
        )

        # Re-fit FRC on the 50-shot subset (ViT backbone stays frozen throughout)
        t0 = time.time()
        model.train_model(few_train_dl, device=device)
        elapsed = time.time() - t0
        print(f"[50-shot]   Training complete in {elapsed:.1f}s")

        # Evaluate on the full CytoData test set
        y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)
        m = compute_all_metrics(
            y_true      = y_true,
            y_pred      = y_pred,
            y_prob      = y_prob,
            class_names = class_names,
            model_name  = "BccT (Corrin)",
            split_name  = f"Test_50shot_rep{rep + 1}",
        )
        repeat_metrics.append(m)
        print(f"[50-shot]   Rep {rep + 1}: Acc={m['accuracy']:.4f}  "
              f"F1={m['f1_macro']:.4f}  "
              f"AUROC={m['auroc_macro']:.4f}  "
              f"Sens={m['sensitivity_macro']:.4f}")

    # -----------------------------------------------------------------------
    # 4. Average across repeats and save 50-shot result
    # -----------------------------------------------------------------------
    avg_50shot = {
        "accuracy_mean":          float(np.mean([m["accuracy"]          for m in repeat_metrics])),
        "f1_macro_mean":          float(np.mean([m["f1_macro"]          for m in repeat_metrics])),
        "auroc_macro_mean":       float(np.mean([m["auroc_macro"]       for m in repeat_metrics])),
        "sensitivity_macro_mean": float(np.mean([m["sensitivity_macro"] for m in repeat_metrics])),
        "num_repeats":            args.num_repeats,
    }

    shot_dir = PROJECT_ROOT / "shared" / "results" / "cytodata" / "bcct" / "low_data" / "50shot"
    shot_dir.mkdir(parents=True, exist_ok=True)
    shot_path = shot_dir / "avg_metrics.json"
    with open(shot_path, "w") as f:
        json.dump(avg_50shot, f, indent=2)
    print(f"\n[50-shot] 50-shot results saved → {shot_path}")
    print(f"          Acc={avg_50shot['accuracy_mean']:.4f}  "
          f"F1={avg_50shot['f1_macro_mean']:.4f}  "
          f"AUROC={avg_50shot['auroc_macro_mean']:.4f}  "
          f"Sens={avg_50shot['sensitivity_macro_mean']:.4f}")

    # -----------------------------------------------------------------------
    # 5. Rebuild low_data_summary.json from all three shot counts
    # -----------------------------------------------------------------------
    print("\n[50-shot] Rebuilding low_data_summary.json …")
    low_data_dir = PROJECT_ROOT / "shared" / "results" / "cytodata" / "bcct" / "low_data"

    summary: dict = {}
    for n_shot in [10, 20, 50]:
        shot_file = low_data_dir / f"{n_shot}shot" / "avg_metrics.json"
        if not shot_file.exists():
            print(f"[50-shot] WARNING: {shot_file} not found — skipping {n_shot}-shot entry.")
            continue
        with open(shot_file) as f:
            summary[f"{n_shot}-shot"] = json.load(f)

    summary_path = PROJECT_ROOT / "shared" / "results" / "cytodata" / "bcct" / "low_data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[50-shot] low_data_summary.json saved → {summary_path}")

    # -----------------------------------------------------------------------
    # 6. Print final summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  CytoData Low-Data Regime Summary — BccT")
    print("=" * 64)
    print(f"  {'Setting':<12}  {'Acc':>8}  {'F1':>8}  {'AUROC':>8}  {'Sens':>8}")
    print("  " + "-" * 50)
    for setting, m in summary.items():
        print(f"  {setting:<12}  "
              f"{m['accuracy_mean']:>8.4f}  "
              f"{m['f1_macro_mean']:>8.4f}  "
              f"{m['auroc_macro_mean']:>8.4f}  "
              f"{m['sensitivity_macro_mean']:>8.4f}")
    print("=" * 64)
    print("\n[50-shot] Done. All CytoData low-data results are complete.")


if __name__ == "__main__":
    main()
