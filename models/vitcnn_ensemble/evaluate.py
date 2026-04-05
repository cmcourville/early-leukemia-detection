"""
models/vitcnn_ensemble/evaluate.py — CS 534, Team 6 — WPI
Author: Sean

Evaluation entry point for the ViT-CNN Ensemble model.

=============================================================================
  TODO (Sean): Implement evaluation logic here.
=============================================================================

Use the shared metrics module so results are computed identically to the
other two models:

    from shared.metrics import compute_all_metrics, print_metrics, save_metrics

    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        model_name="ViT-CNN Ensemble",
        split_name="Test",
    )
    print_metrics(metrics)
    save_metrics(metrics, "shared/results/vitcnn_ensemble_test.json")
"""

raise NotImplementedError(
    "ViT-CNN Ensemble evaluation script not yet implemented. "
    "Sean: please implement this module."
)
