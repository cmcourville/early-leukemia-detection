"""
models/vitcnn_ensemble/model.py — CS 534, Team 6 — WPI
Author: Sean

ViT-CNN Ensemble model for WBC classification.

=============================================================================
  TODO (Sean): Implement the ViT-CNN Ensemble model here.
=============================================================================

Suggested class interface (mirrors BccTModel so run_all.py can call all
three models the same way):

    class ViTCNNEnsemble(nn.Module):

        def __init__(self, num_classes=5, ...):
            ...

        def train_model(self, train_loader, device):
            # Full training loop — gradient-based
            ...

        def forward(self, pixel_values) -> Tensor:
            # Returns logits (B, num_classes)
            ...

        def save(self, path: str) -> None: ...

        @classmethod
        def load(cls, path: str, device) -> "ViTCNNEnsemble": ...

=============================================================================

The ViT backbone is shared with BccT — import from shared config:
    from shared.config import VIT_BACKBONE_ID   # "google/vit-base-patch16-224-in21k"

The pipeline in run_all.py will call:
    model = ViTCNNEnsemble(num_classes=5)
    model.train_model(train_loader, device)
    logits = model(pixel_values)          # (B, 5)

Shared utilities to import:
    from shared.data.data_loader import get_dataloaders, get_few_shot_loaders
    from shared.metrics import compute_all_metrics, print_metrics
    from shared.config import NUM_CLASSES, VIT_BACKBONE_ID
"""

raise NotImplementedError(
    "ViT-CNN Ensemble model not yet implemented. "
    "Sean: please implement this module."
)
