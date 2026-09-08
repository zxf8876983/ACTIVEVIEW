"""Trainable, low-complexity action-belief estimator."""

from __future__ import annotations

import torch
from torch import nn


class ActionBeliefEstimator(nn.Module):
    """Predict a 12-way action belief from two visited observations.

    Each observation contains a 256-D frozen ST-GCN feature, a 12-D
    recognizer log-probability vector and a 768-D mean DINO spatial feature.
    The concatenated ``h0, h1, h1-h0`` vector is intentionally processed by a
    two-layer MLP only; no future candidate observation is an input.
    """

    def __init__(self, input_dim: int = 3 * (256 + 12 + 768), hidden_dim: int = 256, num_classes: int = 12) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(f"expected [B,{self.input_dim}] inputs, got {tuple(inputs.shape)}")
        return self.network(inputs)


__all__ = ["ActionBeliefEstimator"]
