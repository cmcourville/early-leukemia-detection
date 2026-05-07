"""
models/vitcnn_ensemble/model.py — CS 534, Team 6 — WPI
Author: Sean

ViT-CNN Ensemble model for WBC classification.

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
   VIT_BACKBONE_ID

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
                 freeze_layer_index: str = "Mixed_5d",
                 cnn_logits: bool = True,
                 training_epochs: int = 15,):
        super().__init__()
        """
        Common Data structure used for storing the model state helpful for load and save
        """
        self.config = {
            "num_classes": num_classes,
            "embedding_dimension": embedding_dimension,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "feedforward_dimension": feedforward_dimension,
            "freeze_layer_index": freeze_layer_index,
            "training_epochs": training_epochs
        }
        self.backbone = InceptionNetV3Backbone(logits_setting=cnn_logits, frozen_layer_index=freeze_layer_index)
        """
        ViT Configuration were pulled from paper matching the custom transformer developed
        """
        self.vit = ViT(input_dimension=2048,  # fixed value as this is the output size of inceptionNetV3
                       num_heads=num_heads,
                       embedding_dimension=embedding_dimension,
                       dropout_rate=dropout_rate,
                       feedforward_dimension=feedforward_dimension)
        """
        Classifier match those described in the paper
        """
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

        """
        AdamW optimizer used with custom Learning Rates for CNN Backbone, ViT, and Classifier
        parameters below match that of the paper
        """
        optimizer = torch.optim.AdamW([
            {"params": self.backbone.parameters(), "lr":1e-5},
            {"params": self.vit.parameters(), "lr":1e-4},
            {"params": self.classifier.parameters(), "lr":1e-4}], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, int(self.config["training_epochs"])+1):
            epoch_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = self(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()
                training_loss += loss.item() * images.size(0) #TODO Check this
                training_accuracy += (logits.argmax(dim=1) == labels).sum().item()
                training_steps += images.size(0)

            print("=" * 64)
            print(f"Epoch {epoch}/{int(self.config["training_epochs"])} ")
            print(f"Training loss: {epoch_loss:.4f} ")
            print(f"Training accuracy: {training_accuracy / training_steps:.4f}")
            print("=" * 64)
        self.backbone.aux_logits = False

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

        return model

