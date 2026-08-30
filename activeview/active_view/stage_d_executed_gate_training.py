"""Training data and model for EXP019 executed-candidate gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from activeview.active_view.stage_d_error_decomposition import _index
from activeview.active_view.stage_d_policy import order_candidates


EXECUTED_GATE_FEATURE_DIM = 12


class ExecutedCandidateGateMLP(nn.Module):
    """Small fixed binary gate: normalized candidate geometry plus score."""

    def __init__(self, input_dim: int = EXECUTED_GATE_FEATURE_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(-1) != EXECUTED_GATE_FEATURE_DIM:
            raise ValueError(
                f"features must have shape (batch, {EXECUTED_GATE_FEATURE_DIM})"
            )
        return self.network(features).squeeze(-1)


def build_executed_gate_examples(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    geometry_mean: Sequence[float],
    geometry_std: Sequence[float],
    split: str,
) -> list[dict[str, Any]]:
    """Build legal gate inputs and executed-candidate labels.

    The frozen learned ranking selects ``c_hat``.  True U2 is retained only
    as an offline target/diagnostic and is never included in ``features``.
    """
    expected_split = str(split).lower()
    if expected_split not in {"train", "val"}:
        raise ValueError("EXP019 accepts Train and Val only; Test is locked")
    feature_index = _index(feature_rows, f"Stage D {expected_split} features")
    prediction_index = _index(prediction_rows, f"EXP014 {expected_split} predictions")
    if set(feature_index) != set(prediction_index):
        missing = sorted(set(feature_index) - set(prediction_index))
        extra = sorted(set(prediction_index) - set(feature_index))
        raise ValueError(
            "Feature/prediction episode IDs mismatch; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    mean = np.asarray(geometry_mean, dtype=np.float64)
    std = np.asarray(geometry_std, dtype=np.float64)
    if (
        mean.shape != (11,)
        or std.shape != (11,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError("geometry normalization statistics must be finite 11-D vectors")

    examples: list[dict[str, Any]] = []
    for episode_id, feature in feature_index.items():
        prediction = prediction_index[episode_id]
        if (
            str(feature.get("policy_split", "")).lower() != expected_split
            or str(prediction.get("policy_split", "")).lower() != expected_split
        ):
            raise ValueError(
                f"{expected_split} rows must explicitly carry "
                f"policy_split={expected_split}: {episode_id}"
            )
        ids = [int(value) for value in feature["remaining_candidate_ids"]]
        prediction_ids = [int(value) for value in prediction["remaining_candidate_ids"]]
        if ids != prediction_ids:
            raise ValueError(f"Feature/prediction candidate IDs disagree for {episode_id}")
        geometry = np.asarray(feature["second_step_candidate_geometry"], dtype=np.float64)
        predicted = np.asarray(prediction["predicted_utilities"], dtype=np.float64)
        true = np.asarray(feature["second_step_utility_targets"], dtype=np.float64)
        geodesics = np.asarray(feature["second_step_candidate_geodesic"], dtype=np.float64)
        if (
            geometry.shape != (len(ids), 11)
            or predicted.shape != (len(ids),)
            or true.shape != (len(ids),)
            or geodesics.shape != (len(ids),)
        ):
            raise ValueError(f"Candidate arrays have inconsistent shape for {episode_id}")
        if (
            not np.isfinite(geometry).all()
            or not np.isfinite(predicted).all()
            or not np.isfinite(true).all()
            or not np.isfinite(geodesics).all()
        ):
            raise ValueError(f"Candidate arrays must be finite for {episode_id}")
        selected_id = int(order_candidates(
            predicted.tolist(), ids, geodesics.tolist()
        )[0])
        selected_index = ids.index(selected_id)
        normalized_geometry = (geometry[selected_index] - mean) / std
        model_features = np.concatenate(
            [normalized_geometry, np.asarray([predicted[selected_index]], dtype=np.float64)]
        )
        if (
            model_features.shape != (EXECUTED_GATE_FEATURE_DIM,)
            or not np.isfinite(model_features).all()
        ):
            raise ValueError(f"Invalid executed-gate features for {episode_id}")
        examples.append(
            {
                "episode_id": episode_id,
                "candidate_id": selected_id,
                "features": model_features.astype(np.float32),
                "target": int(true[selected_index] > 0.0),
                "true_utility": float(true[selected_index]),
                "predicted_utility": float(predicted[selected_index]),
            }
        )
    return examples
