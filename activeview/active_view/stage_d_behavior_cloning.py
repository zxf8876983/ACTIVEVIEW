"""Spatial-RGB behavior-cloning policy for the EXP027 Stage-D audit."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from activeview.active_view.stage_d_policy import CURRENT_DIM, DELTA_SEMANTIC_DIM, GEOMETRY_DIM
from activeview.active_view.stage_d_rgb_context import DINO_EMBED_DIM
from activeview.active_view.stage_d_rgb_spatial import SPATIAL_TOKEN_COUNT


class SpatialRGBBehaviorCloner(nn.Module):
    """Small three-action policy using legal current-state observations only."""

    input_state_dim = 128 + 128 + 32 + 128 + 128 + 128

    def __init__(self) -> None:
        super().__init__()
        self.s0_encoder = nn.Sequential(nn.Linear(CURRENT_DIM, 128), nn.LayerNorm(128), nn.GELU())
        self.s1_encoder = nn.Sequential(nn.Linear(CURRENT_DIM, 128), nn.LayerNorm(128), nn.GELU())
        self.delta_encoder = nn.Sequential(nn.Linear(DELTA_SEMANTIC_DIM, 32), nn.LayerNorm(32), nn.GELU())
        self.rgb_projector = nn.Sequential(nn.Linear(DINO_EMBED_DIM, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256, dropout=0.1,
            batch_first=True, activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.state_projection = nn.Sequential(
            nn.Linear(self.input_state_dim, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.geometry_encoder = nn.Sequential(nn.Linear(GEOMETRY_DIM, 64), nn.LayerNorm(64), nn.GELU())
        self.candidate_projection = nn.Sequential(nn.Linear(128 + 64, 128), nn.LayerNorm(128), nn.GELU())
        self.stay_head = nn.Linear(128, 1)
        self.candidate_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def _encode_rgb(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM):
            raise ValueError("RGB spatial inputs must have shape (batch, 16, 768)")
        return self.spatial_encoder(self.rgb_projector(values)).mean(dim=1)

    def forward(
        self,
        s0_feature: torch.Tensor,
        s1_feature: torch.Tensor,
        delta_semantic: torch.Tensor,
        candidate_geometry: torch.Tensor,
        candidate_mask: torch.Tensor,
        rgb_s0: torch.Tensor,
        rgb_s1: torch.Tensor,
    ) -> torch.Tensor:
        if s0_feature.ndim != 2 or s0_feature.size(-1) != CURRENT_DIM:
            raise ValueError("s0_feature must have shape (batch, 275)")
        if s1_feature.shape != s0_feature.shape:
            raise ValueError("s1_feature must align with s0_feature")
        if delta_semantic.ndim != 2 or delta_semantic.size(-1) != DELTA_SEMANTIC_DIM:
            raise ValueError("delta_semantic must have shape (batch, 19)")
        if candidate_geometry.ndim != 3 or candidate_geometry.size(-1) != GEOMETRY_DIM:
            raise ValueError("candidate_geometry must have shape (batch, candidates, 11)")
        if candidate_mask.shape != candidate_geometry.shape[:2]:
            raise ValueError("candidate_mask must align with candidate_geometry")
        if candidate_geometry.size(1) > 2:
            raise ValueError("EXP027 supports at most p2 and p3 candidates")

        s0 = self.s0_encoder(s0_feature)
        s1 = self.s1_encoder(s1_feature)
        delta = self.delta_encoder(delta_semantic)
        rgb0 = self._encode_rgb(rgb_s0)
        rgb1 = self._encode_rgb(rgb_s1)
        state = self.state_projection(torch.cat([s0, s1, delta, rgb0, rgb1, rgb1 - rgb0], dim=1))
        count = candidate_geometry.size(1)
        context = torch.cat([state.unsqueeze(1).expand(-1, count, -1), self.geometry_encoder(candidate_geometry)], dim=-1)
        candidate_tokens = self.candidate_projection(context)
        candidate_logits = self.candidate_head(candidate_tokens).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask.bool(), torch.finfo(candidate_logits.dtype).min)
        return torch.cat([self.stay_head(state), candidate_logits], dim=1)


def oracle_action_index(true_utilities: Sequence[float]) -> int:
    """Return the frozen Fixed-first action index (Stay=0, cache-order candidates)."""
    values = np.asarray([0.0, *[float(value) for value in true_utilities]], dtype=np.float64)
    if values.ndim != 1 or values.size not in (2, 3) or not np.isfinite(values).all():
        raise ValueError("true_utilities must contain one or two finite candidates")
    return int(np.argmax(values))


def select_behavior_action(logits: Sequence[float], candidate_ids: Sequence[int]) -> tuple[int, int | None]:
    """Select Stay/p2/p3 with deterministic Stay-first tie behavior."""
    scores = np.asarray(logits, dtype=np.float64)
    ids = [int(value) for value in candidate_ids]
    if scores.shape != (3,) or len(ids) not in (1, 2) or not np.isfinite(scores[: len(ids) + 1]).all():
        raise ValueError("behavior logits must be [Stay,p2,p3] and candidates must be one or two")
    best = int(np.argmax(scores[: len(ids) + 1]))
    return best, None if best == 0 else ids[best - 1]
