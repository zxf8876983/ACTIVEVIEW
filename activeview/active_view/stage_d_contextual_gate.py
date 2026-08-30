"""Frozen EXP014 contextual-latent features for EXP020.

This module deliberately keeps the EXP014 ranker unchanged.  It exposes the
candidate token immediately before the frozen utility head and builds the
executed-candidate gate examples used by the small EXP020 binary head.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from activeview.active_view.stage_d_error_decomposition import _index
from activeview.active_view.stage_d_policy import (
    CURRENT_DIM,
    DELTA_SEMANTIC_DIM,
    GEOMETRY_DIM,
    SequentialObservationRanker,
    order_candidates,
)


CONTEXTUAL_TOKEN_DIM = 128
CONTEXTUAL_GATE_INPUT_DIM = CONTEXTUAL_TOKEN_DIM + 1


def freeze_exp014_ranker(model: SequentialObservationRanker) -> SequentialObservationRanker:
    """Put a loaded EXP014 ranker in inference-only feature-extractor mode."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def contextual_candidate_tokens(
    model: SequentialObservationRanker,
    s0_feature: torch.Tensor,
    s1_feature: torch.Tensor,
    delta_semantic: torch.Tensor,
    candidate_geometry: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return frozen EXP014 contextual tokens immediately before utility_head.

    The operations intentionally mirror ``SequentialObservationRanker.forward``
    up to (but excluding) ``model.utility_head``.  No true utility, label or
    future perception enters this computation.
    """
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
    if candidate_mask.shape != candidate_geometry.shape[:2]:
        raise ValueError("candidate_mask must align with candidate_geometry")

    s0 = model.s0_encoder(s0_feature).unsqueeze(1)
    s1 = model.s1_encoder(s1_feature).unsqueeze(1)
    delta = model.delta_encoder(delta_semantic).unsqueeze(1)
    geometry = model.geometry_encoder(candidate_geometry)
    count = geometry.size(1)
    context = torch.cat(
        [
            s0.expand(-1, count, -1),
            s1.expand(-1, count, -1),
            delta.expand(-1, count, -1),
            geometry,
        ],
        dim=-1,
    )
    tokens = model.token_projection(context)
    return model.interaction(tokens, src_key_padding_mask=~candidate_mask.bool())


class ContextualExecutedGateMLP(nn.Module):
    """EXP020's only trainable component: a fixed 129-D binary gate."""

    def __init__(self, input_dim: int = CONTEXTUAL_GATE_INPUT_DIM) -> None:
        super().__init__()
        if input_dim != CONTEXTUAL_GATE_INPUT_DIM:
            raise ValueError(f"EXP020 input dimension must be {CONTEXTUAL_GATE_INPUT_DIM}")
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != CONTEXTUAL_GATE_INPUT_DIM:
            raise ValueError(
                f"features must have shape (batch, {CONTEXTUAL_GATE_INPUT_DIM})"
            )
        return self.network(features).squeeze(-1)


def _normalization_vector(
    values: Sequence[float], expected_dim: int, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected_dim,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {expected_dim}-D vector")
    return array


def build_contextual_gate_rows(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    current_mean: Sequence[float],
    current_std: Sequence[float],
    delta_mean: Sequence[float],
    delta_std: Sequence[float],
    geometry_mean: Sequence[float],
    geometry_std: Sequence[float],
    split: str,
) -> list[dict[str, Any]]:
    """Build full-context rows with targets from true U2 at frozen c_hat.

    ``c_hat`` is selected only from frozen EXP014 predicted utilities.  The
    true utility is retained as a target/diagnostic and is never included in
    ``s0_feature``, ``s1_feature``, ``delta_semantic`` or geometry inputs.
    """
    expected_split = str(split).lower()
    if expected_split not in {"train", "val"}:
        raise ValueError("EXP020 accepts Train and Val only; Test is locked")
    feature_index = _index(feature_rows, f"Stage D {expected_split} features")
    prediction_index = _index(prediction_rows, f"EXP014 {expected_split} predictions")
    if set(feature_index) != set(prediction_index):
        missing = sorted(set(feature_index) - set(prediction_index))
        extra = sorted(set(prediction_index) - set(feature_index))
        raise ValueError(
            "Feature/prediction episode IDs mismatch; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    current_mu = _normalization_vector(current_mean, CURRENT_DIM, "current_mean")
    current_sigma = _normalization_vector(current_std, CURRENT_DIM, "current_std")
    delta_mu = _normalization_vector(delta_mean, DELTA_SEMANTIC_DIM, "delta_mean")
    delta_sigma = _normalization_vector(delta_std, DELTA_SEMANTIC_DIM, "delta_std")
    geometry_mu = _normalization_vector(geometry_mean, GEOMETRY_DIM, "geometry_mean")
    geometry_sigma = _normalization_vector(geometry_std, GEOMETRY_DIM, "geometry_std")
    if np.any(current_sigma <= 0.0) or np.any(delta_sigma <= 0.0) or np.any(geometry_sigma <= 0.0):
        raise ValueError("normalization standard deviations must be positive")

    rows: list[dict[str, Any]] = []
    for episode_id, feature in feature_index.items():
        prediction = prediction_index[episode_id]
        if (
            str(feature.get("policy_split", "")).lower() != expected_split
            or str(prediction.get("policy_split", "")).lower() != expected_split
        ):
            raise ValueError(
                f"{expected_split} rows must explicitly carry policy_split={expected_split}: "
                f"{episode_id}"
            )
        ids = [int(value) for value in feature["remaining_candidate_ids"]]
        prediction_ids = [int(value) for value in prediction["remaining_candidate_ids"]]
        if ids != prediction_ids or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Feature/prediction candidate IDs disagree for {episode_id}")
        predicted = np.asarray(prediction["predicted_utilities"], dtype=np.float64)
        true = np.asarray(feature["second_step_utility_targets"], dtype=np.float64)
        geodesics = np.asarray(feature["second_step_candidate_geodesic"], dtype=np.float64)
        geometry = np.asarray(feature["second_step_candidate_geometry"], dtype=np.float64)
        s0 = np.asarray(feature["s0_feature"], dtype=np.float64)
        s1 = np.asarray(feature["s1_feature"], dtype=np.float64)
        delta = np.asarray(feature["delta_semantic"], dtype=np.float64)
        if (
            predicted.shape != (len(ids),)
            or true.shape != (len(ids),)
            or geodesics.shape != (len(ids),)
            or geometry.shape != (len(ids), GEOMETRY_DIM)
            or s0.shape != (CURRENT_DIM,)
            or s1.shape != (CURRENT_DIM,)
            or delta.shape != (DELTA_SEMANTIC_DIM,)
        ):
            raise ValueError(f"Invalid Stage D feature shape for {episode_id}")
        all_values = np.concatenate([s0, s1, delta, geometry.ravel(), predicted, true, geodesics])
        if not np.isfinite(all_values).all():
            raise ValueError(f"Stage D features and utilities must be finite for {episode_id}")

        ordered = order_candidates(predicted.tolist(), ids, geodesics.tolist())
        candidate_id = int(ordered[0])
        selected_index = ids.index(candidate_id)
        rows.append(
            {
                "episode_id": episode_id,
                "policy_split": expected_split,
                "candidate_id": candidate_id,
                "selected_index": selected_index,
                "s0_feature": ((s0 - current_mu) / current_sigma).astype(np.float32),
                "s1_feature": ((s1 - current_mu) / current_sigma).astype(np.float32),
                "delta_semantic": ((delta - delta_mu) / delta_sigma).astype(np.float32),
                "candidate_geometry": ((geometry - geometry_mu) / geometry_sigma).astype(np.float32),
                "candidate_mask": np.ones((len(ids),), dtype=bool),
                "predicted_utility": float(predicted[selected_index]),
                "target": int(true[selected_index] > 0.0),
                "true_utility": float(true[selected_index]),
            }
        )
    return rows


def apply_contextual_gate_decision(candidate_id: int, logit: float) -> dict[str, Any]:
    """Apply fixed EXP020 threshold while preserving frozen candidate identity."""
    move = float(logit) > 0.0
    return {
        "predicted_stays": not move,
        "predicted_candidate_viewpoint_id": int(candidate_id) if move else None,
        "candidate_id": int(candidate_id),
        "gate_logit": float(logit),
        "gate_probability": float(1.0 / (1.0 + np.exp(-float(logit)))),
    }
