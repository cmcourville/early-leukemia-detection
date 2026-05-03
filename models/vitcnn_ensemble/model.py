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
The pipeline in run_all.py will call:
    model = ViTCNNEnsemble(num_classes=5)
    model.train_model(train_loader, device)
    logits = model(pixel_values)          # (B, 5)

Shared utilities to import:
    from shared.data.data_loader import get_dataloaders, get_few_shot_loaders
    from shared.metrics import compute_all_metrics, print_metrics
    from shared.config import NUM_CLASSES, VIT_BACKBONE_ID
"""
from torch._dynamo.variables import optimizer

"""
Author: Sean St.Pierre
Class: CS534
This CNN model is based on the model from:Alshehri, O.M., Shaf, A., Irfan, M., Jalal, M.M., Altayar, M.A. et al. (2025).
A Hybrid CNN-Transformer Framework for Normal Blood Cell Classification: Towards Automated Hematological Analysis. Computer Modeling in Engineering & Sciences,
144(1), 1165–1196. https://doi.org/10.32604/cmes.2025.067150
"""

import torch
import torch.nn as nn

from cnn_backbone import InceptionNetV3Backbone
from vit_model import ViT


class ViTCNNEnsemble(nn.Module):
    def __init__(self,
                 num_classes: int = 8,
                 embedding_dimension: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.2,
                 feedforward_dimension: int = 256,
                 freeze_layer_index: str = "Mixed_6e"):
        super().__init__()
        """
        Common Datastructure used for storing the model state helpful for load and save
        """
        self.config = {
            "num_classes": num_classes,
            "embedding_dimension": embedding_dimension,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "feedforward_dimension": feedforward_dimension,
            "freeze_layer_index": freeze_layer_index
        }

        self.backbone = InceptionNetV3Backbone(frozen_layer_index=freeze_layer_index)
        self.vit = ViT(input_dimension=2048,  # fixed value as this is the output size of inceptionNetV3
                       num_heads=num_heads,
                       embedding_dimension=embedding_dimension,
                       dropout_rate=dropout_rate,
                       feedforward_dimension=feedforward_dimension)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(x)
        cls_rep = self.vit(tokens)
        logits = self.classifier(cls_rep)
        return logits

    def train_model(self, train_loader, device):
        self.train()
        training_loss = 0
        training_accuracy = 0
        training_steps = 0

        optimizer = torch.optim.Adam(self.vit.parameters(), lr=5e-5) #TODO: Check this param
        criterion = nn.CrossEntropyLoss() #TODO: Confirm Loss function with paper

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * images.size(0) #TODO Check this
            training_accuracy += (logits.argmax(dim=1) == labels).sum().item()
            training_steps += images.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"    batch {batch_idx + 1}/{len(train_loader)} " f"loss: {loss.item():.4f}  "
                      f"acc: {training_accuracy / training_steps:.4f}")

    def save(self, path: str) -> None:
        current_model_state = {
            "config": self.config,
            "model_state_dict": self.state_dict(),
        }
        torch.save(current_model_state, path)

    @classmethod
    def load(cls, path: str, device) -> "ViTCNNEnsemble":
        current_model_state = torch.load(path, map_location=device, weights_only=False)
        config = current_model_state["config"]
        model = cls(**config)
        model.load_state_dict(current_model_state["model_state_dict"])
        model.to(device)
        print(f"Loaded model from {path}")
        print(f"Current model state: {current_model_state}")

        return model

