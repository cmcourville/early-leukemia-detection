"""
models/cytodiffusion/evaluate.py — CS 534, Team 6 — WPI
Author: Darshan

Evaluation entry point for the CytoDiffusion / LDM model.

=============================================================================
  TODO (Darshan): Implement evaluation logic here.
=============================================================================

Use the shared metrics module so results are computed identically to the
other two models:

    from shared.metrics import compute_all_metrics, print_metrics, save_metrics

    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        model_name="CytoDiffusion",
        split_name="Test",
    )
    print_metrics(metrics)
    save_metrics(metrics, "shared/results/cytodiffusion_test.json")
"""

raise NotImplementedError(
    "CytoDiffusion evaluation script not yet implemented. "
    "Darshan: please implement this module."
)
