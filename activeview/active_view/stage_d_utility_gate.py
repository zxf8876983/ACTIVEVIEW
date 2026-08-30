"""Raw executed-utility regression gate used by Stage D EXP022."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from activeview.active_view.stage_d_contextual_gate import (
    CONTEXTUAL_GATE_INPUT_DIM,
    build_contextual_gate_rows,
)


class UtilityExecutedGateMLP(nn.Module):
    """Small 129-D regressor for the utility of frozen ``c_hat``."""

    def __init__(self, input_dim: int = CONTEXTUAL_GATE_INPUT_DIM) -> None:
        super().__init__()
        if input_dim != CONTEXTUAL_GATE_INPUT_DIM:
            raise ValueError(f"EXP022 input dimension must be {CONTEXTUAL_GATE_INPUT_DIM}")
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


def build_utility_gate_rows(
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
    """Build EXP022 rows with raw true U2 as regression target only."""
    rows = build_contextual_gate_rows(
        feature_rows=feature_rows,
        prediction_rows=prediction_rows,
        current_mean=current_mean,
        current_std=current_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        split=split,
    )
    for row in rows:
        target = float(row["true_utility"])
        if not np.isfinite(target):
            raise ValueError(f"EXP022 target must be finite: {row['episode_id']}")
        row["target_regression"] = target
    return rows


def apply_utility_gate_decision(candidate_id: int, predicted_utility: float) -> dict[str, Any]:
    """Apply strict raw-utility sign semantics while preserving ``c_hat``."""
    value = float(predicted_utility)
    if not np.isfinite(value):
        raise ValueError("predicted executed utility must be finite")
    move = value > 0.0
    return {
        "predicted_stays": not move,
        "predicted_candidate_viewpoint_id": int(candidate_id) if move else None,
        "candidate_id": int(candidate_id),
        "predicted_utility": value,
    }


def utility_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean default SmoothL1 loss over valid candidate entries."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("predictions and targets must be finite")
    losses = nn.functional.smooth_l1_loss(predictions, targets, reduction="none")
    if mask is None:
        return losses.mean()
    if mask.shape != predictions.shape:
        raise ValueError("mask must align with predictions")
    valid = mask.bool()
    if not bool(valid.any()):
        raise ValueError("mask must contain at least one valid value")
    return losses.masked_select(valid).mean()
