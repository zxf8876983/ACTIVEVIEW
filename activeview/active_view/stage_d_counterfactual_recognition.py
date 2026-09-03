"""Small history/candidate model that predicts future ST-GCN log probabilities."""

from __future__ import annotations

import torch
from torch import nn


class CounterfactualRecognitionModel(nn.Module):
    """Predict a candidate recognition distribution from observed history."""

    def __init__(self, geometry_dim: int = 9) -> None:
        super().__init__()
        self.recognition = nn.Sequential(nn.Linear(16, 64), nn.GELU())
        self.rgb = nn.Sequential(nn.Linear(768, 128), nn.GELU())
        self.skeleton = nn.Sequential(nn.Linear(102, 128), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, 32), nn.GELU())
        self.observation = nn.Sequential(nn.Linear(64 + 128 + 128 + 32, 128), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(geometry_dim, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.history = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(128, 16)

    def forward(
        self,
        history_recognition: torch.Tensor,
        history_rgb: torch.Tensor,
        history_skeleton: torch.Tensor,
        history_geometry: torch.Tensor,
        candidate_geometry: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits with shape ``[batch, classes]``."""
        if history_recognition.ndim != 3:
            raise ValueError("history_recognition must be [B,H,16]")
        obs = torch.cat(
            [
                self.recognition(history_recognition),
                self.rgb(history_rgb),
                self.skeleton(history_skeleton),
                self.geometry(history_geometry),
            ],
            dim=-1,
        )
        tokens = self.observation(obs)
        encoded = self.history(tokens, src_key_padding_mask=None if history_mask is None else ~history_mask.bool())
        if history_mask is None:
            state = encoded.mean(dim=1)
        else:
            weights = history_mask.to(encoded.dtype).unsqueeze(-1)
            state = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(state + self.candidate(candidate_geometry))

