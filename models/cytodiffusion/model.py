"""
models/cytodiffusion/model.py — CS 534, Team 6 — WPI
Author: Darshan

CytoDiffusion classifier: pretrained Stable-Diffusion VAE encoder used
as a fixed feature extractor, with a lightweight MLP classification head
trained end-to-end via cross-entropy.

Architecture (inspired by CytoDiffusion / CambridgeCIA):
    SD VAE encoder (runwayml/stable-diffusion-v1-5, frozen by default)
    → latent (B, 4, 28, 28) for 224×224 input
    → AdaptiveAvgPool2d(8,8) → flatten → (B, 256)
    → Linear(256→hidden) → GELU → Dropout
    → Linear(hidden→hidden//2) → GELU → Dropout
    → Linear(hidden//2→num_classes)

The VAE's perceptual bottleneck separates cell morphology in latent
space; a linear probe on this representation achieves strong few-shot
accuracy without requiring full diffusion model training.

Interface (identical to BccTModel so run_all.py drives all models uniformly):
    model = CytoDiffusionModel(num_classes)
    model.train_model(train_loader, device)
    logits = model(pixel_values)          # (B, num_classes)
    model.save(path)
    model = CytoDiffusionModel.load(path, device)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from shared.config import RAABIN_MEAN, RAABIN_STD, IMAGE_SIZE

try:
    from diffusers import AutoencoderKL
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

_SD_MODEL_ID  = "runwayml/stable-diffusion-v1-5"
_SD_SUBFOLDER = "vae"

_LATENT_CH      = 4
_LATENT_SPATIAL = IMAGE_SIZE // 8   # 28 for 224-px input
_POOL_SIZE      = 8                 # adaptive-pool target: (8, 8)
_FEAT_DIM       = _LATENT_CH * _POOL_SIZE * _POOL_SIZE   # 256


class CytoDiffusionModel(nn.Module):

    def __init__(
        self,
        num_classes:    int   = 5,
        sd_model_id:    str   = _SD_MODEL_ID,
        freeze_encoder: bool  = True,
        hidden_dim:     int   = 512,
        dropout:        float = 0.3,
        img_mean: Optional[List[float]] = None,
        img_std:  Optional[List[float]] = None,
        epochs:       int   = 30,
        lr_head:      float = 1e-3,
        lr_encoder:   float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        if not HAS_DIFFUSERS:
            raise ImportError(
                "diffusers is required for CytoDiffusionModel.\n"
                "Install: pip install diffusers>=0.27.0 accelerate safetensors"
            )

        self.config = dict(
            num_classes=num_classes,
            sd_model_id=sd_model_id,
            freeze_encoder=freeze_encoder,
            hidden_dim=hidden_dim,
            dropout=dropout,
            img_mean=img_mean or list(RAABIN_MEAN),
            img_std=img_std  or list(RAABIN_STD),
            epochs=epochs,
            lr_head=lr_head,
            lr_encoder=lr_encoder,
            weight_decay=weight_decay,
        )

        # Per-dataset normalization buffers; used to convert the
        # data_loader's normalized tensors to [-1, 1] for the VAE.
        mean = img_mean if img_mean is not None else list(RAABIN_MEAN)
        std  = img_std  if img_std  is not None else list(RAABIN_STD)
        self.register_buffer(
            "img_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std",  torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
        )

        # VAE encoder (pretrained SD weights)
        print(f"[CytoDiffusion] Loading VAE encoder from {sd_model_id} …")
        self.encoder: AutoencoderKL = AutoencoderKL.from_pretrained(
            sd_model_id, subfolder=_SD_SUBFOLDER
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        # Spatial pooling + classification head
        self.pool = nn.AdaptiveAvgPool2d((_POOL_SIZE, _POOL_SIZE))
        self.classifier = nn.Sequential(
            nn.Linear(_FEAT_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_head_weights()

    # ------------------------------------------------------------------
    # Weight initialisation

    def _init_head_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass

    def _encode(self, pixel_values: Tensor) -> Tensor:
        """Encode images to VAE latents (B, 4, H/8, W/8)."""
        # pixel_values are dataset-normalised; convert to [-1, 1] for VAE
        x = pixel_values * self.img_std + self.img_mean   # ≈ [0, 1]
        x = (x * 2.0 - 1.0).clamp(-1.0, 1.0)            # [-1, 1]
        with torch.set_grad_enabled(not self.config["freeze_encoder"]):
            return self.encoder.encode(x).latent_dist.mean

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W), normalized to dataset mean/std
        Returns:
            logits: (B, num_classes)
        """
        latent = self._encode(pixel_values)   # (B, 4, 28, 28)
        pooled = self.pool(latent)            # (B, 4,  8,  8)
        flat   = pooled.flatten(1)            # (B, 256)
        return self.classifier(flat)          # (B, C)

    # ------------------------------------------------------------------
    # Training

    def train_model(
        self,
        train_loader,
        device:     torch.device,
        val_loader=None,
    ) -> None:
        """Full training loop. Re-initialises the head each call so that
        low-data / few-shot experiments start from a clean slate."""
        self._init_head_weights()
        self.to(device)

        freeze_enc   = self.config["freeze_encoder"]
        epochs       = self.config["epochs"]
        lr_head      = self.config["lr_head"]
        lr_encoder   = self.config["lr_encoder"]
        weight_decay = self.config["weight_decay"]

        if freeze_enc:
            self.encoder.eval()
            trainable = list(self.pool.parameters()) + list(self.classifier.parameters())
            optimizer = torch.optim.AdamW(
                trainable, lr=lr_head, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.AdamW([
                {"params": list(self.pool.parameters()) +
                           list(self.classifier.parameters()), "lr": lr_head},
                {"params": self.encoder.parameters(), "lr": lr_encoder},
            ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_head * 0.01
        )

        # Compute class weights from training data to handle imbalance
        label_counts = torch.zeros(self.config["num_classes"])
        for _, lbls in train_loader:
            for l in lbls:
                label_counts[int(l)] += 1
        total = label_counts.sum()
        class_weights = (total / (self.config["num_classes"] * label_counts.clamp(min=1))).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        for epoch in range(1, epochs + 1):
            self.train()
            if freeze_enc:
                self.encoder.eval()

            total_loss, correct, total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = self(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                correct    += (logits.argmax(1) == labels).sum().item()
                total      += imgs.size(0)

            scheduler.step()
            acc = correct / total if total > 0 else 0.0
            avg = total_loss / total if total > 0 else 0.0
            print(f"[CytoDiffusion] Epoch {epoch:3d}/{epochs}  "
                  f"loss={avg:.4f}  acc={acc:.4f}")

        print("[CytoDiffusion] Training complete.")

    # ------------------------------------------------------------------
    # Checkpoint helpers

    def save(self, path: str) -> None:
        """Save config + head weights to a compact checkpoint.
        Encoder weights are omitted when frozen (reload from HuggingFace on load)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt: dict = {
            "config": self.config,
            "head_state": {
                "pool":       self.pool.state_dict(),
                "classifier": self.classifier.state_dict(),
            },
        }
        if not self.config["freeze_encoder"]:
            ckpt["encoder_state"] = self.encoder.state_dict()
        torch.save(ckpt, path)
        print(f"[CytoDiffusion] Checkpoint saved → {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "CytoDiffusionModel":
        """Reload model from checkpoint."""
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt["config"])
        model.pool.load_state_dict(ckpt["head_state"]["pool"])
        model.classifier.load_state_dict(ckpt["head_state"]["classifier"])
        if "encoder_state" in ckpt:
            model.encoder.load_state_dict(ckpt["encoder_state"])
        return model.to(device)
