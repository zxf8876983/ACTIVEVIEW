"""Whitelisted current-observation and candidate-geometry features."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


FEATURE_SCHEMA_VERSION = "stage-c-features-v2-egocentric-geometry"
RELATIVE_FEATURE_SCHEMA_VERSION = "stage-c-features-v3-relative-geometry"
CURRENT_ACTION_FEATURE_DIM = 256
CURRENT_LOG_PROB_DIM = 16
CURRENT_FEATURE_DIM = CURRENT_ACTION_FEATURE_DIM + CURRENT_LOG_PROB_DIM + 3
BASE_CANDIDATE_GEOMETRY_DIM = 11
CURRENT_FEATURE_NAMES = (
    [f"stgcn_feature_{index}" for index in range(CURRENT_ACTION_FEATURE_DIM)]
    + [f"current_log_prob_{index}" for index in range(CURRENT_LOG_PROB_DIM)]
    + ["current_entropy", "current_top1_top2_margin", "current_pose_confidence"]
)
CANDIDATE_GEOMETRY_NAMES = (
    "ego_relative_position_x", "ego_relative_position_y", "ego_relative_position_z",
    "euclidean_distance_m", "geodesic_distance_m", "sin_relative_azimuth",
    "cos_relative_azimuth", "path_ratio", "current_radius_m",
    "candidate_radius_m", "delta_radius_m",
)
RELATIVE_CANDIDATE_GEOMETRY_NAMES = (
    "radius_zscore", "radius_rank", "geodesic_zscore", "geodesic_rank",
    "delta_radius_normalized",
)
CANDIDATE_GEOMETRY_DIM = len(CANDIDATE_GEOMETRY_NAMES)
RELATIVE_CANDIDATE_GEOMETRY_DIM = len(RELATIVE_CANDIDATE_GEOMETRY_NAMES)


def _finite_array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} != {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def current_state_features(
    stgcn_feature: Sequence[float],
    log_probs: Sequence[float],
    pose_confidence: float,
) -> np.ndarray:
    """Build the 275-D current-only observable feature vector."""
    feature = _finite_array(stgcn_feature, name="stgcn_feature", shape=(CURRENT_ACTION_FEATURE_DIM,))
    logp = _finite_array(log_probs, name="current_log_probs", shape=(CURRENT_LOG_PROB_DIM,)).astype(np.float64)
    probabilities = np.exp(logp)
    order = np.sort(probabilities)
    entropy = float(-(probabilities * logp).sum())
    margin = float(order[-1] - order[-2])
    confidence = float(pose_confidence)
    if not np.isfinite(confidence):
        raise ValueError("pose confidence is non-finite")
    return np.concatenate(
        [feature.astype(np.float32), logp.astype(np.float32), np.asarray([entropy, margin, confidence], dtype=np.float32)]
    ).astype(np.float32)


def candidate_geometry_features(
    candidate: Mapping[str, Any],
    *,
    current_position: Sequence[float],
    current_rotation_wxyz: Sequence[float],
    placement_position: Sequence[float],
) -> np.ndarray:
    """Build geometry in the current agent frame (no world-coordinate shortcut)."""
    current = _finite_array(current_position, name="current_position", shape=(3,))
    rotation = _finite_array(current_rotation_wxyz, name="current_rotation_wxyz", shape=(4,))
    norm = float(np.linalg.norm(rotation))
    if norm <= 1e-8:
        raise ValueError("current rotation quaternion is degenerate")
    w, x, y, z = (rotation / norm).tolist()
    yaw = float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))
    candidate_position = _finite_array(
        candidate["snapped_position"], name="candidate_snapped_position", shape=(3,)
    )
    # Stage A's relative_position is the audited agent-to-agent displacement;
    # use it as the world delta so the y convention matches navigation exactly.
    world_delta = _finite_array(candidate["relative_position"], name="relative_position", shape=(3,))
    cos_yaw, sin_yaw = float(np.cos(yaw)), float(np.sin(yaw))
    relative = np.asarray(
        [cos_yaw * world_delta[0] - sin_yaw * world_delta[2], world_delta[1],
         sin_yaw * world_delta[0] + cos_yaw * world_delta[2]],
        dtype=np.float32,
    )
    euclidean = float(candidate["euclidean_distance_m"])
    geodesic = float(candidate["geodesic_distance_m"])
    azimuth = np.deg2rad(float(candidate["relative_azimuth_deg"]))
    placement = _finite_array(placement_position, name="placement_position", shape=(3,))
    current_radius = float(np.linalg.norm((current - placement)[[0, 2]]))
    candidate_radius = float(np.linalg.norm((candidate_position - placement)[[0, 2]]))
    values = np.asarray(
        [
            *relative.tolist(), euclidean, geodesic, np.sin(azimuth), np.cos(azimuth),
            geodesic / (euclidean + 1e-6), current_radius, candidate_radius,
            candidate_radius - current_radius,
        ],
        dtype=np.float32,
    )
    if values.shape != (CANDIDATE_GEOMETRY_DIM,) or not np.isfinite(values).all():
        raise ValueError("candidate geometry feature is invalid")
    return values


def candidate_geometry_matrix(
    candidates: Sequence[Mapping[str, Any]],
    *,
    current_position: Sequence[float],
    current_rotation_wxyz: Sequence[float],
    placement_position: Sequence[float],
    include_relative_features: bool = False,
) -> np.ndarray:
    if not candidates:
        raise ValueError("candidate set must not be empty")
    base = np.stack(
        [candidate_geometry_features(item, current_position=current_position, current_rotation_wxyz=current_rotation_wxyz, placement_position=placement_position) for item in candidates],
        axis=0,
    )
    if not include_relative_features:
        return base
    return np.concatenate([base, candidate_set_relative_features(base)], axis=1)


def _set_zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _set_rank(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    average_positions = (starts + ends - 1) / 2.0
    return (average_positions[inverse] / float(values.size - 1)).astype(np.float32)


def candidate_set_relative_features(base_geometry: np.ndarray) -> np.ndarray:
    """Compute candidate-set-relative geometry from the existing 11D geometry."""
    base = np.asarray(base_geometry, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != BASE_CANDIDATE_GEOMETRY_DIM:
        raise ValueError(
            f"base_geometry shape {base.shape} must be (N, {BASE_CANDIDATE_GEOMETRY_DIM})"
        )
    if base.shape[0] == 0 or not np.isfinite(base).all():
        raise ValueError("base_geometry must be non-empty and finite")
    radius = base[:, 9].astype(np.float64)
    geodesic = base[:, 4].astype(np.float64)
    delta_radius = base[:, 10].astype(np.float64)
    mean_radius = float(np.mean(radius))
    denominator = max(abs(mean_radius), 1e-6)
    relative = np.column_stack(
        [
            _set_zscore(radius),
            _set_rank(radius),
            _set_zscore(geodesic),
            _set_rank(geodesic),
            (delta_radius / denominator).astype(np.float32),
        ]
    ).astype(np.float32)
    if relative.shape != (base.shape[0], RELATIVE_CANDIDATE_GEOMETRY_DIM) or not np.isfinite(relative).all():
        raise ValueError("candidate-set-relative geometry is invalid")
    return relative


def frozen_current_features(
    model: torch.nn.Module,
    skeleton: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen ST-GCN once and return its 256-D feature and log-probabilities."""
    array = _finite_array(skeleton, name="current_skeleton", shape=(3, 30, 17))
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.inference_mode():
        feature_tensor = model.forward_features(tensor)
        logits = model.fc(feature_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)
    feature = feature_tensor[0].detach().cpu().numpy().astype(np.float32)
    logp = log_probs[0].detach().cpu().numpy().astype(np.float32)
    if feature.shape != (CURRENT_ACTION_FEATURE_DIM,) or logp.shape != (CURRENT_LOG_PROB_DIM,):
        raise ValueError(f"Unexpected frozen ST-GCN outputs: {feature.shape}, {logp.shape}")
    return feature, logp


def schema_metadata(
    *,
    include_relative_features: bool = False,
) -> dict[str, Any]:
    geometry_names = list(CANDIDATE_GEOMETRY_NAMES)
    version = FEATURE_SCHEMA_VERSION
    if include_relative_features:
        geometry_names.extend(RELATIVE_CANDIDATE_GEOMETRY_NAMES)
        version = RELATIVE_FEATURE_SCHEMA_VERSION
    metadata = {
        "version": version,
        "current_feature_dim": CURRENT_FEATURE_DIM,
        "current_feature_names": list(CURRENT_FEATURE_NAMES),
        "candidate_geometry_dim": len(geometry_names),
        "candidate_geometry_names": geometry_names,
        "body_yaw_used": False,
        "movement_cost_penalty_used": False,
        "future_candidate_perception_used_as_input": False,
        "input_whitelist": ["current_stgcn_feature", "current_log_probabilities", "current_entropy", "current_margin", "current_pose_confidence", "candidate_geometry"],
        "forbidden_input_fields": ["label_id", "action_label", "candidate_skeleton", "candidate_confidence", "candidate_log_probs", "candidate_entropy", "candidate_utility", "candidate_prediction", "gt_correctness", "viewpoint_id"],
    }
    if include_relative_features:
        metadata["relative_geometry_features"] = True
    return metadata
