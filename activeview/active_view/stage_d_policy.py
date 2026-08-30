"""Sequential Stage D policy primitives and the small trainable ranker."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


TOP_K = 3
CURRENT_DIM = 275
DELTA_SEMANTIC_DIM = 19
GEOMETRY_DIM = 11


def order_candidates(
    predicted_utilities: Sequence[float],
    candidate_ids: Sequence[int],
    geodesics: Sequence[float],
    *,
    top_k: int | None = None,
) -> list[int]:
    """Order candidates by utility, geodesic, then viewpoint id."""
    if len(predicted_utilities) != len(candidate_ids) or len(candidate_ids) != len(geodesics):
        raise ValueError("candidate arrays must have equal lengths")
    if not candidate_ids:
        return []
    values = [
        (int(candidate_ids[index]), float(predicted_utilities[index]), float(geodesics[index]))
        for index in range(len(candidate_ids))
    ]
    if not np.isfinite(np.asarray(predicted_utilities, dtype=np.float64)).all():
        raise ValueError("predicted utilities must be finite")
    if not np.isfinite(np.asarray(geodesics, dtype=np.float64)).all():
        raise ValueError("geodesics must be finite")
    values.sort(key=lambda item: (-item[1], item[2], item[0]))
    ids = [item[0] for item in values]
    return ids if top_k is None else ids[: int(top_k)]


def first_step_decision(
    predicted_utilities: Sequence[float],
    candidate_ids: Sequence[int],
    geodesics: Sequence[float],
) -> tuple[bool, int | None, float]:
    """Apply the frozen Stage C-v0 Move/Stay rule."""
    ordered = order_candidates(predicted_utilities, candidate_ids, geodesics)
    if not ordered:
        raise ValueError("first-step candidate set must not be empty")
    best_id = ordered[0]
    index = list(candidate_ids).index(best_id)
    best_value = float(predicted_utilities[index])
    return best_value <= 0.0, (None if best_value <= 0.0 else best_id), best_value


def second_step_decision(
    predicted_utilities: Sequence[float],
    candidate_ids: Sequence[int],
    geodesics: Sequence[float],
) -> tuple[bool, int | None, float]:
    """Apply the Stage D Stay-inclusive second-step rule."""
    return first_step_decision(predicted_utilities, candidate_ids, geodesics)


def second_step_utility(candidate_logp_true: float, s1_logp_true: float) -> float:
    """Return U2(candidate | s1) for supervision/evaluation only."""
    value = float(candidate_logp_true) - float(s1_logp_true)
    if not np.isfinite(value):
        raise ValueError("second-step utility must be finite")
    return value


def semantic_delta(s0_feature: Sequence[float], s1_feature: Sequence[float]) -> np.ndarray:
    """Return the observable 19-D semantic transition from s0 to s1."""
    first = np.asarray(s0_feature, dtype=np.float32)
    second = np.asarray(s1_feature, dtype=np.float32)
    if first.shape != (CURRENT_DIM,) or second.shape != (CURRENT_DIM,):
        raise ValueError("s0 and s1 features must be 275-D")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("s0 and s1 features must be finite")
    value = second[256:] - first[256:]
    if value.shape != (DELTA_SEMANTIC_DIM,) or not np.isfinite(value).all():
        raise ValueError("semantic delta must be finite 19-D")
    return value.astype(np.float32)


def trajectory_cost(first_step_geodesic: float, second_step_geodesic: float | None = None) -> float:
    """Sum the geodesic cost of the visited transitions."""
    first = float(first_step_geodesic)
    second = 0.0 if second_step_geodesic is None else float(second_step_geodesic)
    if first < 0.0 or second < 0.0 or not np.isfinite([first, second]).all():
        raise ValueError("trajectory geodesics must be finite and non-negative")
    return first + second


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class SequentialObservationRanker(nn.Module):
    """Small second-step ranker conditioned on the observed s0→s1 transition."""

    model_type = "sequential_observation_ranker"

    def __init__(
        self,
        *,
        current_dim: int = CURRENT_DIM,
        delta_dim: int = DELTA_SEMANTIC_DIM,
        geometry_dim: int = GEOMETRY_DIM,
    ) -> None:
        super().__init__()
        self.s0_encoder = _Encoder(current_dim, 128)
        self.s1_encoder = _Encoder(current_dim, 128)
        self.delta_encoder = _Encoder(delta_dim, 32)
        self.geometry_encoder = _Encoder(geometry_dim, 64)
        self.token_projection = nn.Sequential(
            nn.Linear(128 + 128 + 32 + 64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
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
        self.utility_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(
        self,
        s0_feature: torch.Tensor,
        s1_feature: torch.Tensor,
        delta_semantic: torch.Tensor,
        candidate_geometry: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if s0_feature.ndim != 2 or s0_feature.size(-1) != CURRENT_DIM:
            raise ValueError("s0_feature must have shape (batch, 275)")
        if s1_feature.shape != s0_feature.shape:
            raise ValueError("s1_feature must align with s0_feature")
        if delta_semantic.ndim != 2 or delta_semantic.size(-1) != DELTA_SEMANTIC_DIM:
            raise ValueError("delta_semantic must have shape (batch, 19)")
        if candidate_geometry.ndim != 3 or candidate_geometry.size(-1) != GEOMETRY_DIM:
            raise ValueError("candidate_geometry must have shape (batch, candidates, 11)")
        if candidate_geometry.size(0) != s0_feature.size(0):
            raise ValueError("candidate batch and current batch differ")
        s0 = self.s0_encoder(s0_feature).unsqueeze(1)
        s1 = self.s1_encoder(s1_feature).unsqueeze(1)
        delta = self.delta_encoder(delta_semantic).unsqueeze(1)
        geometry = self.geometry_encoder(candidate_geometry)
        count = geometry.size(1)
        context = torch.cat(
            [s0.expand(-1, count, -1), s1.expand(-1, count, -1), delta.expand(-1, count, -1), geometry],
            dim=-1,
        )
        tokens = self.token_projection(context)
        padding_mask = None if candidate_mask is None else ~candidate_mask.bool()
        tokens = self.interaction(tokens, src_key_padding_mask=padding_mask)
        return self.utility_head(tokens).squeeze(-1)


def schema_metadata() -> Mapping[str, Any]:
    return {
        "model_type": SequentialObservationRanker.model_type,
        "s0_feature_dim": CURRENT_DIM,
        "s1_feature_dim": CURRENT_DIM,
        "delta_semantic_dim": DELTA_SEMANTIC_DIM,
        "candidate_geometry_dim": GEOMETRY_DIM,
        "proposal_top_k": TOP_K,
        "visited_s1_perception_used_as_input": True,
        "future_unvisited_candidate_perception_used_as_input": False,
        "gt_label_used_as_input": False,
        "logp_true_used_as_input": False,
        "safe_oracle_used_as_input": False,
    }
