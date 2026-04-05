"""
fixed_random_classifier.py — BccT Project (CS 534, Team 6)
Author: Corrin

Implements the Fixed-Random Classifier (FRC) head from Zhu et al. (2026),
"BccT: An Efficient Transformer Model for Blood Cell Classification."

Architecture (paper Section 3.3, Fig. 3):

    Input x  ∈ ℝ^{d_in}   (CLS token, 768-dim for ViT-Base)
        │
        ├─────────────────────────────────────────────┐
        │ Hidden layer (ELM-style):                   │ Skip/feedback link
        │   H = ReLU(x W^T + b)                      │ (direct input → output)
        │   W ∈ ℝ^{d_h × d_in}, b ∈ ℝ^{d_h}         │
        │   Initialised ~ N(0,1), frozen forever      │
        │                                             │
        └────────── concat ──────────────────────────┘
                       │
                   [H ; x]  ∈ ℝ^{d_h + d_in}
                       │
        Output layer F ∈ ℝ^{C × (d_h + d_in)}
           Ŷ = [H ; x] F^T
                       │
                  logits ∈ ℝ^C

Training (Eq. 12 of paper / ELM closed-form):
   Given feature matrix O = [H ; x] ∈ ℝ^{N × (d_h + d_in)}
   and one-hot label matrix Y ∈ ℝ^{N × C}:

       F = (O^T O + λ I)^{-1} O^T Y   — ridge version (λ=0 → pure pseudo-inverse)
     or equivalently via least-squares:
       F = lstsq(O, Y)   — solved via torch.linalg.lstsq (LAPACK driver)

   Only F is ever updated; W and b are frozen from the moment of
   initialisation.  No gradient descent anywhere in the model.

Hidden-layer width:
  The paper does not specify d_h explicitly; common ELM practice is
  d_h = 4 * d_in (= 3072 for ViT-Base).  We use this as default but expose
  it as a constructor argument.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FixedRandomClassifier(nn.Module):
    """
    ELM-inspired Fixed-Random Classifier as described in Section 3.3 of
    Zhu et al. (2026).

    Key properties:
      • W and b are initialised from N(0,1) and **immediately frozen**.
      • F (output weights) starts as zeros and is set once via
        `fit(features, labels)` using a closed-form pseudo-inverse solve.
      • No gradient passes through this module at any point during training.
      • The skip/feedback link concatenates the raw input to the hidden
        representation before applying F.

    Args:
        d_in     : Dimensionality of input features (768 for ViT-Base CLS).
        num_classes : Number of output classes (5 for Raabin-WBC).
        d_hidden : Width of the fixed random hidden layer.
                   Default: 4 * d_in (= 3072).
        ridge_lambda : Regularisation strength λ for the ridge pseudo-inverse.
                   0.0 = pure Moore-Penrose.  Small positive value (e.g. 1e-4)
                   improves conditioning when N ≈ d.
        seed     : Random seed for reproducible W, b initialisation.
    """

    def __init__(
        self,
        d_in:         int,
        num_classes:  int,
        d_hidden:     Optional[int] = None,
        ridge_lambda: float = 0.0,
        seed:         int   = 42,
    ):
        super().__init__()

        self.d_in        = d_in
        self.num_classes = num_classes
        self.d_hidden    = d_hidden if d_hidden is not None else 4 * d_in
        self.ridge_lambda = ridge_lambda

        # Fixed random weights (W, b) — frozen at init
        # Initialise from N(0,1) as specified in the paper.
        gen = torch.Generator()
        gen.manual_seed(seed)

        W = torch.randn(self.d_hidden, d_in, generator=gen)   # (d_h, d_in)
        b = torch.randn(self.d_hidden,       generator=gen)   # (d_h,)

        # Register as buffers so they move with .to(device) but are NOT
        # parameters (will not appear in model.parameters()).
        self.register_buffer("W", W)   # frozen hidden-layer weights
        self.register_buffer("b", b)   # frozen hidden-layer biases

        # Output weights F — learnable only via pseudo-inverse
        # d_out_in = d_hidden + d_in  (hidden output concatenated with skip)
        self.d_combined = self.d_hidden + d_in
        F_init = torch.zeros(num_classes, self.d_combined)    # (C, d_h+d_in)
        self.register_buffer("F", F_init)

        self._fitted = False   # guard: warn if forward called before fit()

    # Feature extraction helper                                            #

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Compute the concatenated feature vector [H ; x] used as input to F.

        H = ReLU(x W^T + b)   — hidden representation  (B, d_h)
        skip = x               — direct skip link        (B, d_in)
        output = [H ; x]                                 (B, d_h + d_in)

        Args:
            x : (B, d_in) — CLS token features

        Returns:
            combined : (B, d_h + d_in)
        """
        # Hidden layer: ReLU(x W^T + b)
        # W: (d_h, d_in)  →  x W^T: (B, d_h)
        H = F.relu(x @ self.W.t() + self.b)    # (B, d_h)

        # Skip/feedback: concatenate raw input (Fig. 3)
        combined = torch.cat([H, x], dim=1)     # (B, d_h + d_in)
        return combined

    # Closed-form output-weight fit  (Eq. 12 / ELM training)              #

    @torch.no_grad()
    def fit(self, features: Tensor, labels: Tensor) -> None:
        """
        Solve for the output matrix F using the Moore-Penrose pseudo-inverse
        (or ridge regression if ridge_lambda > 0).

        Eq. 12:   F = O⁺ Y
        where:
            O = extract_features(features)   shape (N, d_combined)
            Y = one_hot(labels)              shape (N, C)

        For numerical stability we use torch.linalg.lstsq (LAPACK 'gelsd'
        driver), which is more stable than the explicit pseudo-inverse for
        tall or ill-conditioned systems.

        If ridge_lambda > 0, we augment the system:
            O_aug = [O ; √λ I]     shape (N + d_combined, d_combined)
            Y_aug = [Y ; 0]        shape (N + d_combined, C)
        and solve the augmented least-squares problem.

        Args:
            features : (N, d_in)  — CLS tokens from the training set
            labels   : (N,)       — integer class labels [0, C)
        """
        device = self.W.device
        features = features.to(device)
        labels   = labels.to(device)

        N = features.shape[0]

        # Feature matrix O: (N, d_combined)
        O = self.extract_features(features)     # (N, d_combined)

        # One-hot target matrix Y: (N, C)
        Y = F.one_hot(labels.long(), num_classes=self.num_classes).float()  # (N, C)

        if self.ridge_lambda > 0.0:
            # Ridge augmentation for regularised pseudo-inverse
            sqrt_lam = math.sqrt(self.ridge_lambda)
            reg_block = sqrt_lam * torch.eye(self.d_combined, device=device)  # (d, d)
            zeros     = torch.zeros(self.d_combined, self.num_classes, device=device)
            O_aug = torch.cat([O, reg_block], dim=0)    # (N + d, d)
            Y_aug = torch.cat([Y, zeros],     dim=0)    # (N + d, C)
        else:
            O_aug, Y_aug = O, Y

        # Solve O_aug F^T = Y_aug  →  F^T = lstsq(O_aug, Y_aug)
        # torch.linalg.lstsq returns a named-tuple; .solution is F^T: (d, C)
        solution = torch.linalg.lstsq(O_aug, Y_aug, driver="gelsd").solution  # (d, C)

        # F: (C, d_combined)
        self.F.copy_(solution.t())
        self._fitted = True
        print(f"[FRC] fit() complete — F shape: {self.F.shape}, "
              f"N_train={N}, ridge_lambda={self.ridge_lambda}")

    # Forward pass                                                         #

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute class logits.

        Args:
            x : (B, d_in) — CLS token from the final ViT block

        Returns:
            logits : (B, num_classes)
        """
        if not self._fitted:
            # Warn but don't crash — allows forward pass before fit for
            # architecture testing.  F starts as all-zeros so output is zero.
            import warnings
            warnings.warn(
                "[FixedRandomClassifier] forward() called before fit(). "
                "Output weights F are still all-zeros."
            )

        combined = self.extract_features(x)    # (B, d_combined)
        # Ŷ = combined F^T
        logits = combined @ self.F.t()         # (B, C)
        return logits

    # Utilities                                                            #

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted class indices. (B,) int64."""
        return self.forward(x).argmax(dim=1)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities via softmax. (B, C)."""
        return torch.softmax(self.forward(x), dim=1)

    def extra_repr(self) -> str:
        return (f"d_in={self.d_in}, d_hidden={self.d_hidden}, "
                f"num_classes={self.num_classes}, "
                f"d_combined={self.d_combined}, "
                f"ridge_lambda={self.ridge_lambda}, "
                f"fitted={self._fitted}")

# Quick sanity check

if __name__ == "__main__":
    torch.manual_seed(0)

    d_in = 768
    C    = 5
    N    = 200   # simulated training samples

    frc = FixedRandomClassifier(d_in=d_in, num_classes=C, ridge_lambda=1e-4)
    print(frc)

    # Confirm W and b are not parameters
    param_names = [n for n, _ in frc.named_parameters()]
    print(f"Learnable parameters: {param_names}")   # should be empty
    buf_names   = [n for n, _ in frc.named_buffers()]
    print(f"Buffers (W, b, F): {buf_names}")

    # Simulate a training set
    x_train  = torch.randn(N, d_in)
    y_train  = torch.randint(0, C, (N,))

    # One-shot fit
    frc.fit(x_train, y_train)

    # Inference
    x_test  = torch.randn(8, d_in)
    logits  = frc(x_test)
    proba   = frc.predict_proba(x_test)
    preds   = frc.predict(x_test)

    print(f"Logits  shape : {logits.shape}")    # (8, 5)
    print(f"Probas  shape : {proba.shape}")     # (8, 5)
    print(f"Preds         : {preds}")
    print(f"Proba row-sum : {proba.sum(dim=1)}")  # should be ~1.0
