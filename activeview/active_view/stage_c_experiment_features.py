"""Small, leakage-safe geometry variants for Stage C-v1 diagnostic experiments."""

from __future__ import annotations

import math
from typing import Callable, Dict, Mapping

import numpy as np

from activeview.active_view.stage_c_features import (
    BASE_CANDIDATE_GEOMETRY_DIM,
    CANDIDATE_GEOMETRY_NAMES,
)


VARIANT_NAMES = (
    "radius_ablation",
    "direction_geometry",
    "candidate_relations",
)


def _validate_base(base_geometry: np.ndarray) -> np.ndarray:
    base = np.asarray(base_geometry, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != BASE_CANDIDATE_GEOMETRY_DIM:
        raise ValueError(
            f"base_geometry shape {base.shape} must be (N, {BASE_CANDIDATE_GEOMETRY_DIM})"
        )
    if base.shape[0] == 0 or not np.isfinite(base).all():
        raise ValueError("base_geometry must be non-empty and finite")
    return base


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(np.mean(values))) / std).astype(np.float32)


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _direction_features(base_geometry: np.ndarray) -> np.ndarray:
    """Return four circular, candidate-set-relative azimuth features."""
    angles = np.arctan2(base_geometry[:, 5], base_geometry[:, 6]).astype(np.float64)
    center = float(np.arctan2(np.sin(angles).sum(), np.cos(angles).sum()))
    offsets = np.asarray(_wrap(angles - center), dtype=np.float64)
    # Rank by positive angular offset from the circular mean.  The modulo
    # operation avoids an artificial -pi/+pi cut in the input representation.
    positive_offsets = np.mod(offsets, 2.0 * np.pi)
    ranks = _set_rank(positive_offsets)
    deviations = np.abs(offsets) / np.pi
    nearest = np.zeros(len(angles), dtype=np.float64)
    density = np.zeros(len(angles), dtype=np.float64)
    threshold = np.deg2rad(45.0)
    for index, angle in enumerate(angles):
        if len(angles) > 1:
            distances = np.abs(np.asarray(_wrap(angle - angles), dtype=np.float64))
            distances[index] = np.inf
            nearest[index] = float(np.min(distances)) / np.pi
            density[index] = float(np.sum(distances <= threshold)) / float(len(angles) - 1)
    return np.column_stack(
        [ranks, deviations.astype(np.float32), nearest.astype(np.float32), density.astype(np.float32)]
    ).astype(np.float32)


def _relation_features(base_geometry: np.ndarray) -> np.ndarray:
    """Return five simple candidate-to-candidate geometry summaries."""
    radius = base_geometry[:, 9].astype(np.float64)
    geodesic = base_geometry[:, 4].astype(np.float64)
    angles = np.arctan2(base_geometry[:, 5], base_geometry[:, 6]).astype(np.float64)
    radius_z = _zscore(radius).astype(np.float32)
    geodesic_z = _zscore(geodesic).astype(np.float32)
    n = len(base_geometry)
    nearest = np.zeros(n, dtype=np.float32)
    mean_distance = np.zeros(n, dtype=np.float32)
    density = np.zeros(n, dtype=np.float32)
    radius_scale = max(float(np.std(radius)), 1e-6)
    geodesic_scale = max(float(np.std(geodesic)), 1e-6)
    for index in range(n):
        if n <= 1:
            continue
        distances = []
        for other in range(n):
            if other == index:
                continue
            angular = float(abs(_wrap(angles[index] - angles[other]))) / np.pi
            distances.append(
                math.sqrt(
                    ((radius[index] - radius[other]) / radius_scale) ** 2
                    + ((geodesic[index] - geodesic[other]) / geodesic_scale) ** 2
                    + angular**2
                )
            )
        nearest[index] = float(min(distances))
        mean_distance[index] = float(np.mean(distances))
        density[index] = float(np.mean(np.asarray(distances) <= 1.0))
    return np.column_stack(
        [radius_z, geodesic_z, nearest, mean_distance, density]
    ).astype(np.float32)


def transform_geometry(base_geometry: np.ndarray, variant: str) -> np.ndarray:
    """Transform frozen 11-D geometry for one independent EXP004/5/7 variant."""
    base = _validate_base(base_geometry)
    if variant == "radius_ablation":
        # Keep egocentric xyz, Euclidean/geodesic distance, azimuth and path
        # ratio; retain geodesic set statistics but remove every radius cue.
        result = np.column_stack([base[:, :8], _zscore(base[:, 4]), _set_rank(base[:, 4])])
    elif variant == "direction_geometry":
        result = np.concatenate([base, _direction_features(base)], axis=1)
    elif variant == "candidate_relations":
        result = np.concatenate([base, _relation_features(base)], axis=1)
    else:
        raise ValueError(f"Unknown geometry variant: {variant}")
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{variant} geometry contains non-finite values")
    return result


def _set_rank(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    average_positions = (starts + ends - 1) / 2.0
    return (average_positions[inverse] / float(values.size - 1)).astype(np.float32)


def variant_geometry_names(variant: str) -> list[str]:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unknown geometry variant: {variant}")
    if variant == "radius_ablation":
        return [
            *CANDIDATE_GEOMETRY_NAMES[:8],
            "geodesic_zscore",
            "geodesic_rank",
        ]
    if variant == "direction_geometry":
        return [
            *CANDIDATE_GEOMETRY_NAMES,
            "azimuth_rank",
            "azimuth_deviation_from_set_center",
            "nearest_angular_neighbor_distance",
            "angular_local_density",
        ]
    return [
        *CANDIDATE_GEOMETRY_NAMES,
        "delta_radius_to_set_mean",
        "delta_geodesic_to_set_mean",
        "nearest_candidate_geometry_distance",
        "mean_candidate_geometry_distance",
        "candidate_geometry_density",
    ]


def variant_schema(variant: str) -> Dict[str, object]:
    names = variant_geometry_names(variant)
    return {
        "version": f"stage-c-features-v4-{variant}",
        "current_feature_dim": 275,
        "candidate_geometry_dim": len(names),
        "candidate_geometry_names": names,
        "base_candidate_geometry_names": list(CANDIDATE_GEOMETRY_NAMES),
        "variant": variant,
        "body_yaw_used": False,
        "movement_cost_penalty_used": False,
        "future_candidate_perception_used_as_input": False,
        "input_whitelist": [
            "current_stgcn_feature",
            "current_log_probabilities",
            "current_entropy",
            "current_margin",
            "current_pose_confidence",
            "candidate_geometry",
        ],
        "forbidden_input_fields": [
            "label_id",
            "action_label",
            "candidate_skeleton",
            "candidate_confidence",
            "candidate_log_probs",
            "candidate_entropy",
            "candidate_utility",
            "candidate_prediction",
            "gt_correctness",
            "viewpoint_id",
        ],
    }


VARIANT_BUILDERS: Mapping[str, Callable[[np.ndarray], np.ndarray]] = {
    name: (lambda base, name=name: transform_geometry(base, name))
    for name in VARIANT_NAMES
}
