"""Contextual-bandit policy used by the EXP021 second-step experiment."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from activeview.active_view.stage_d_policy import (
    CURRENT_DIM,
    DELTA_SEMANTIC_DIM,
    GEOMETRY_DIM,
)


class ContextualBanditRanker(nn.Module):
    """Small contextual scorer for Stay (fixed zero), p2 and p3."""

    def __init__(self) -> None:
        super().__init__()
        self.s0_encoder = nn.Sequential(
            nn.Linear(CURRENT_DIM, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.s1_encoder = nn.Sequential(
            nn.Linear(CURRENT_DIM, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(DELTA_SEMANTIC_DIM, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(GEOMETRY_DIM, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.token_projection = nn.Sequential(
            nn.Linear(128 + 128 + 32 + 64, 128), nn.LayerNorm(128), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(layer, num_layers=2)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(
        self,
        s0_feature: torch.Tensor,
        s1_feature: torch.Tensor,
        delta_semantic: torch.Tensor,
        candidate_geometry: torch.Tensor,
        candidate_mask: torch.Tensor,
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
        count = candidate_geometry.size(1)
        context = torch.cat(
            [
                self.s0_encoder(s0_feature).unsqueeze(1).expand(-1, count, -1),
                self.s1_encoder(s1_feature).unsqueeze(1).expand(-1, count, -1),
                self.delta_encoder(delta_semantic).unsqueeze(1).expand(-1, count, -1),
                self.geometry_encoder(candidate_geometry),
            ],
            dim=-1,
        )
        tokens = self.token_projection(context)
        tokens = self.interaction(tokens, src_key_padding_mask=~candidate_mask.bool())
        return self.action_head(tokens).squeeze(-1)


def expected_reward_loss(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``-E_pi[r]`` and the corresponding expected reward."""
    if scores.shape != rewards.shape or scores.shape != candidate_mask.shape:
        raise ValueError("scores, rewards and candidate_mask must have the same shape")
    probabilities = action_probabilities(scores, candidate_mask)
    all_rewards = torch.cat(
        [torch.zeros((rewards.size(0), 1), dtype=rewards.dtype, device=rewards.device), rewards],
        dim=1,
    )
    expected = (probabilities * all_rewards).sum(dim=1)
    return -expected.mean(), expected


def action_probabilities(scores: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """Return softmax probabilities for ``[Stay, p2, p3]`` with Stay fixed at zero."""
    if scores.ndim != 2 or candidate_mask.shape != scores.shape:
        raise ValueError("scores and candidate_mask must be aligned 2-D tensors")
    stay_score = torch.zeros((scores.size(0), 1), dtype=scores.dtype, device=scores.device)
    all_scores = torch.cat([stay_score, scores], dim=1)
    all_mask = torch.cat(
        [
            torch.ones((candidate_mask.size(0), 1), dtype=torch.bool, device=candidate_mask.device),
            candidate_mask.bool(),
        ],
        dim=1,
    )
    masked_scores = all_scores.masked_fill(~all_mask, torch.finfo(scores.dtype).min)
    return torch.softmax(masked_scores, dim=1)


def expected_reward_loss_with_entropy(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    candidate_mask: torch.Tensor,
    beta: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return entropy-regularized loss, expected reward and policy entropy."""
    if beta < 0.0 or not np.isfinite(float(beta)):
        raise ValueError("entropy coefficient must be finite and non-negative")
    if scores.shape != rewards.shape or scores.shape != candidate_mask.shape:
        raise ValueError("scores, rewards and candidate_mask must have the same shape")
    probabilities = action_probabilities(scores, candidate_mask)
    all_rewards = torch.cat(
        [torch.zeros((rewards.size(0), 1), dtype=rewards.dtype, device=rewards.device), rewards],
        dim=1,
    )
    expected = (probabilities * all_rewards).sum(dim=1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)
    loss = -expected.mean() - float(beta) * entropy.mean()
    return loss, expected, entropy


def supervised_candidate_utility_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked default SmoothL1 loss for Phase-A utility warm-start."""
    if scores.shape != targets.shape or scores.shape != candidate_mask.shape:
        raise ValueError("scores, targets and candidate_mask must have the same shape")
    if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
        raise ValueError("scores and targets must be finite")
    valid = candidate_mask.bool()
    if not bool(valid.any()):
        raise ValueError("candidate_mask must contain at least one valid candidate")
    losses = nn.functional.smooth_l1_loss(scores, targets, reduction="none")
    return losses.masked_select(valid).mean()


def select_bandit_actions(
    scores: Sequence[float], candidate_ids: Sequence[int]
) -> tuple[bool, int | None, float]:
    """Select Stay/p2/p3 by deterministic argmax with Stay first on ties."""
    values = np.asarray(scores, dtype=np.float64)
    ids = [int(value) for value in candidate_ids]
    if values.ndim != 1 or values.size != len(ids) or values.size == 0:
        raise ValueError("candidate scores and IDs must be non-empty and aligned")
    if not np.isfinite(values).all() or len(set(ids)) != len(ids):
        raise ValueError("candidate scores must be finite and IDs unique")
    best_index = int(np.argmax(np.asarray([0.0, *values.tolist()], dtype=np.float64)))
    if best_index == 0:
        return True, None, 0.0
    return False, ids[best_index - 1], float(values[best_index - 1])


def action_name(selected_stays: bool, selected_id: int | None, candidate_ids: Sequence[int]) -> str:
    """Return the canonical human-readable action label."""
    if selected_stays:
        return "Stay"
    index = [int(value) for value in candidate_ids].index(int(selected_id))
    return "p2" if index == 0 else "p3"
