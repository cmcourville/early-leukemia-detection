"""
bcct_model.py — BccT Project (CS 534, Team 6)
Author: Corrin

Full BccT model assembly: frozen ViT-Base backbone + Token Fusion hooks
+ Fixed-Random Classifier head.

Architecture summary (Zhu et al., 2026):
  • Backbone  : google/vit-base-patch16-224-in21k
                - 12 transformer blocks, 768-dim hidden, 12 attention heads
                - Input: 224×224 → 196 patch tokens + 1 CLS = 197 tokens
                - Entirely frozen (requires_grad = False throughout)
  • Token Fusion: inserted after FFN in each of the 12 blocks (Section 3.2)
                - Budget r=16 pairs merged per block
                - Max tokens removed: 12 × 16 = 192 → ≥ 5 tokens survive
                - After 12 blocks the sequence is ≤ 197 - 192 = 5+CLS = 6 tokens
  • Classifier: FixedRandomClassifier head on the final CLS token (768-dim)
                - ELM hidden layer (4×768 = 3072 units), frozen random weights
                - Output F trained once via pseudo-inverse on the training set

The entire pipeline uses **no gradient descent**.  Training = one forward
pass through the frozen backbone to collect CLS features, then one call to
`FixedRandomClassifier.fit()`.

Usage:
    model = BccTModel(num_classes=5, r=16)
    model.to(device)

    # Training
    model.train_model(train_loader, device)

    # Inference
    logits = model(pixel_values)   # (B, num_classes)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import ViTModel, ViTConfig

from token_fusion import patch_token_fusion_hooks, reset_token_sizes, BlockState
from fixed_random_classifier import FixedRandomClassifier

class BccTModel(nn.Module):
    """
    BccT: Blood Cell Classification Transformer (Zhu et al., 2026).

    Args:
        num_classes   : Number of WBC classes (5 for Raabin-WBC).
        r             : Token Fusion merge budget per block (paper default: 16).
        d_hidden      : Hidden-layer width for FRC (default: 4 × 768 = 3072).
        ridge_lambda  : Ridge regularisation for FRC pseudo-inverse (default: 1e-4).
        pretrained_id : HuggingFace model ID for the ViT backbone.
        frc_seed      : Random seed for FRC weight initialisation.
    """

    def __init__(
        self,
        num_classes:   int  = 5,
        r:             int  = 16,
        d_hidden:      Optional[int] = None,
        ridge_lambda:  float = 1e-4,
        pretrained_id: str  = "google/vit-base-patch16-224-in21k",
        frc_seed:      int  = 42,
    ):
        super().__init__()

        self.num_classes  = num_classes
        self.r            = r
        self.pretrained_id = pretrained_id

        # 1. ViT backbone (frozen)
        print(f"[BccT] Loading backbone: {pretrained_id}")
        self.vit = ViTModel.from_pretrained(pretrained_id)

        # Freeze ALL backbone parameters — no gradient descent on ViT
        for param in self.vit.parameters():
            param.requires_grad_(False)

        d_model = self.vit.config.hidden_size   # 768 for ViT-Base

        # 2. Token Fusion hooks (after FFN in each block)
        self.block_states: List[BlockState] = patch_token_fusion_hooks(
            self.vit, r=r
        )
        print(f"[BccT] Token Fusion hooks registered on "
              f"{len(self.block_states)} blocks, r={r}")

        # 3. Fixed-Random Classifier head
        self.frc = FixedRandomClassifier(
            d_in=d_model,
            num_classes=num_classes,
            d_hidden=d_hidden,
            ridge_lambda=ridge_lambda,
            seed=frc_seed,
        )
        # FRC weights (W, b, F) are all buffers — not parameters.
        # There are literally zero trainable parameters in this model.
        print(f"[BccT] FRC head: d_in={d_model}, d_hidden={self.frc.d_hidden}, "
              f"C={num_classes}")
        print(f"[BccT] Trainable parameters: "
              f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    # Feature extraction  (ViT forward with Token Fusion)                 #

    @torch.no_grad()
    def extract_cls(self, pixel_values: Tensor) -> Tensor:
        """
        Run the frozen ViT backbone (with Token Fusion active) and return
        the CLS token from the final encoder block.

        Args:
            pixel_values : (B, 3, 224, 224)

        Returns:
            cls_features : (B, 768)
        """
        # Reset block states so previous batch's token sizes don't carry over
        reset_token_sizes(self.block_states)

        outputs = self.vit(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=False,
        )
        # last_hidden_state: (B, N_remaining, 768)
        # CLS token is always at index 0 (never merged)
        cls = outputs.last_hidden_state[:, 0, :]   # (B, 768)
        return cls

    # Training: one-shot pseudo-inverse fit                               #

    def train_model(
        self,
        train_loader,
        device: torch.device,
    ) -> None:
        """
        Full training procedure for BccT:
          1. Forward-pass all training images through the frozen ViT+TokenFusion
             to collect CLS feature vectors.
          2. Fit the FRC output weights F via one-shot pseudo-inverse.

        No backpropagation is performed at any point.

        Args:
            train_loader : DataLoader yielding (pixel_values, labels).
            device       : Target device.
        """
        self.vit.eval()           # Disable dropout in ViT during collection
        self.to(device)

        all_features: List[Tensor] = []
        all_labels:   List[Tensor] = []

        print("[BccT] Collecting CLS features from training set …")
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            cls = self.extract_cls(imgs)          # (B, 768)
            all_features.append(cls.cpu())
            all_labels.append(labels.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * imgs.shape[0]} samples …")

        features = torch.cat(all_features, dim=0)   # (N, 768)
        labels   = torch.cat(all_labels,   dim=0)   # (N,)

        print(f"[BccT] Feature matrix shape: {features.shape}")

        # Move FRC to CPU for the lstsq solve (LAPACK is CPU-only)
        self.frc.cpu()
        self.frc.fit(features, labels)
        # Move FRC back to target device
        self.frc.to(device)

        print("[BccT] Training complete (pseudo-inverse solve).")

    # Forward pass  (inference)                                           #

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        End-to-end forward pass: ViT + Token Fusion → CLS → FRC → logits.

        Args:
            pixel_values : (B, 3, 224, 224)

        Returns:
            logits : (B, num_classes)
        """
        cls    = self.extract_cls(pixel_values)   # (B, 768)
        logits = self.frc(cls)                    # (B, C)
        return logits

    # Convenience: save / load model state                                #

    def save(self, path: str) -> None:
        """
        Save the FRC weights (F, W, b) and config.  The backbone is not saved
        (it's a frozen pretrained model that can always be reloaded from HF).
        """
        state = {
            "frc_state":   self.frc.state_dict(),
            "num_classes": self.num_classes,
            "r":           self.r,
            "d_hidden":    self.frc.d_hidden,
            "ridge_lambda": self.frc.ridge_lambda,
            "pretrained_id": self.pretrained_id,
            "fitted":      self.frc._fitted,
        }
        torch.save(state, path)
        print(f"[BccT] Model saved → {path}")

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "BccTModel":
        """
        Reload a saved BccTModel.  Backbone is re-downloaded from HuggingFace.
        """
        state = torch.load(path, map_location="cpu")
        model = cls(
            num_classes   = state["num_classes"],
            r             = state["r"],
            d_hidden      = state["d_hidden"],
            ridge_lambda  = state["ridge_lambda"],
            pretrained_id = state["pretrained_id"],
        )
        model.frc.load_state_dict(state["frc_state"])
        model.frc._fitted = state["fitted"]
        model.to(device)
        print(f"[BccT] Model loaded ← {path}")
        return model

    def extra_repr(self) -> str:
        return (f"num_classes={self.num_classes}, r={self.r}, "
                f"d_hidden={self.frc.d_hidden}, "
                f"fitted={self.frc._fitted}")

# Quick sanity check

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BccTModel(num_classes=5, r=16)
    model.to(device)

    # Dummy batch
    x = torch.randn(4, 3, 224, 224, device=device)

    print("\n--- Forward pass (before training / F=zeros) ---")
    with torch.no_grad():
        logits = model(x)
    print(f"Logits shape: {logits.shape}")   # (4, 5)

    print("\n--- Simulated one-shot training ---")
    # Create a tiny fake dataloader
    from torch.utils.data import TensorDataset, DataLoader
    fake_imgs   = torch.randn(50, 3, 224, 224)
    fake_labels = torch.randint(0, 5, (50,))
    fake_dl     = DataLoader(TensorDataset(fake_imgs, fake_labels), batch_size=16)

    model.train_model(fake_dl, device=device)

    print("\n--- Forward pass (after training) ---")
    with torch.no_grad():
        logits = model(x)
    print(f"Logits shape : {logits.shape}")
    print(f"Predictions  : {logits.argmax(dim=1)}")
