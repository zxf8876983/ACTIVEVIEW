"""文件用途：
    实现主动视角策略运行时逻辑。

主要输入：
    - 策略特征、候选视点和访问历史。
主要输出：
    - 动作、rollout 或 utility 预测。
项目角色：
    - 属于 methods.active_view 方法模块。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.data.preprocessing.policy_features import candidate_geometry_features

VIEW_COUNT = 32
TOP_K = 3
CURRENT_DIM = 275
DELTA_SEMANTIC_DIM = 19
GEOMETRY_DIM = 11
ContextKey = tuple[str, str, str]


def context_key(row: Mapping[str, Any]) -> ContextKey:
    return (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))


def viewpoint_azimuth(viewpoint_id: int) -> float:
    return float(int(viewpoint_id) % 8 * 45.0)


def viewpoint_radius(viewpoint_id: int) -> float:
    return (1.5, 2.0, 2.5, 3.0)[int(viewpoint_id) // 8]


def relative_view_descriptor(
    positions: np.ndarray, current_position: np.ndarray, viewpoint_id: int,
) -> np.ndarray:
    """Return the legal 9-D navigation descriptor used by WM-E."""
    delta = np.asarray(positions[int(viewpoint_id)] - current_position, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    azimuth = np.deg2rad(viewpoint_azimuth(viewpoint_id))
    bearing = np.arctan2(float(delta[0]), float(delta[2]))
    return np.asarray(
        [viewpoint_radius(viewpoint_id) / 3.0, np.sin(azimuth), np.cos(azimuth),
         float(delta[0]), float(delta[1]), float(delta[2]), distance,
         np.sin(bearing), np.cos(bearing)], dtype=np.float32,
    )


def load_pairwise_and_azimuths(
    data_root: Path, rows: Sequence[Mapping[str, Any]],
    sources: Mapping[ContextKey, str],
) -> tuple[dict[tuple[str, str], dict[int, dict[int, float]]], dict[tuple[str, str], dict[int, float]]]:
    """Load frozen navigation metadata without changing candidate semantics."""
    pair_root = data_root / "datasets/policy_v11_5/pairwise_viewpoint_geodesic"
    pairwise: dict[tuple[str, str], dict[int, dict[int, float]]] = {}
    azimuths: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        scene_region = (str(row["scene_id"]), str(row["region"]))
        if scene_region in pairwise:
            continue
        # Imported lazily to avoid a data/geometry import cycle.
        from activeview.data.preprocessing.cache import load_pairwise_geodesic
        pairwise[scene_region] = load_pairwise_geodesic(pair_root / scene_region[0] / f"{scene_region[1]}.json")
        manifest = Path(sources[context_key(row)]).parents[1] / "candidate_metadata" / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        placements = [item for item in payload["placements_data"] if str(item["region"]) == scene_region[1]]
        if len(placements) != 1:
            raise ValueError(f"candidate metadata placement mismatch for {scene_region}")
        values = {int(view["viewpoint_id"]): float(view["azimuth_deg"]) for view in placements[0]["viewpoints"]}
        if len(values) != VIEW_COUNT:
            raise ValueError(f"expected 32 azimuths for {scene_region}")
        azimuths[scene_region] = values
    return pairwise, azimuths


def candidate_order(
    row: Mapping[str, Any], current: int, visited: set[int],
    pairwise: Mapping[int, Mapping[int, float]], azimuths: Mapping[int, float],
) -> list[int]:
    """Return legal viewpoints in the frozen distance/azimuth/id order."""
    candidates: list[tuple[float, float, int]] = []
    current_azimuth = float(azimuths[current])
    for viewpoint_id in range(VIEW_COUNT):
        if viewpoint_id in visited or viewpoint_id not in pairwise.get(current, {}):
            continue
        distance = float(pairwise[current][viewpoint_id])
        if not np.isfinite(distance):
            continue
        delta = (float(azimuths[viewpoint_id]) - current_azimuth + 180.0) % 360.0 - 180.0
        candidates.append((distance, abs(delta), viewpoint_id))
    candidates.sort()
    return [viewpoint_id for _, _, viewpoint_id in candidates]


def order_candidates(predicted_utilities: Sequence[float], candidate_ids: Sequence[int], geodesics: Sequence[float], *, top_k: int | None = None) -> list[int]:
    if len(predicted_utilities) != len(candidate_ids) or len(candidate_ids) != len(geodesics):
        raise ValueError("candidate arrays must have equal lengths")
    if not candidate_ids:
        return []
    utilities = np.asarray(predicted_utilities, dtype=np.float64)
    distances = np.asarray(geodesics, dtype=np.float64)
    if not np.isfinite(utilities).all() or not np.isfinite(distances).all():
        raise ValueError("candidate utilities and geodesics must be finite")
    values = sorted(zip(candidate_ids, utilities, distances), key=lambda item: (-float(item[1]), float(item[2]), int(item[0])))
    ids = [int(item[0]) for item in values]
    return ids if top_k is None else ids[: int(top_k)]


def first_step_decision(predicted_utilities: Sequence[float], candidate_ids: Sequence[int], geodesics: Sequence[float]) -> tuple[bool, int | None, float]:
    ordered = order_candidates(predicted_utilities, candidate_ids, geodesics)
    if not ordered:
        raise ValueError("candidate set must not be empty")
    best_id = ordered[0]
    value = float(predicted_utilities[list(candidate_ids).index(best_id)])
    return value <= 0.0, (None if value <= 0.0 else best_id), value


def second_step_decision(predicted_utilities: Sequence[float], candidate_ids: Sequence[int], geodesics: Sequence[float]) -> tuple[bool, int | None, float]:
    return first_step_decision(predicted_utilities, candidate_ids, geodesics)


def second_step_utility(candidate_logp_true: float, s1_logp_true: float) -> float:
    value = float(candidate_logp_true) - float(s1_logp_true)
    if not np.isfinite(value):
        raise ValueError("second-step utility must be finite")
    return value


def semantic_delta(s0_feature: Sequence[float], s1_feature: Sequence[float]) -> np.ndarray:
    first, second = np.asarray(s0_feature, dtype=np.float32), np.asarray(s1_feature, dtype=np.float32)
    if first.shape != (CURRENT_DIM,) or second.shape != (CURRENT_DIM,) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("s0 and s1 features must be finite 275-D vectors")
    return (second[256:] - first[256:]).astype(np.float32)


def trajectory_cost(first_step_geodesic: float, second_step_geodesic: float | None = None) -> float:
    first = float(first_step_geodesic); second = 0.0 if second_step_geodesic is None else float(second_step_geodesic)
    if first < 0.0 or second < 0.0 or not np.isfinite([first, second]).all():
        raise ValueError("trajectory geodesics must be finite and non-negative")
    return first + second


def wrap_relative_azimuth(candidate_azimuth_deg: float, current_azimuth_deg: float) -> float:
    """Match Stage-A radial azimuth difference in ``[-180, 180)``."""
    return float((float(candidate_azimuth_deg) - float(current_azimuth_deg) + 180.0) % 360.0 - 180.0)


def second_step_geometry(
    *, s1_position: Sequence[float], s1_rotation_wxyz: Sequence[float],
    target_position: Sequence[float], target_snapped_position: Sequence[float],
    target_geodesic: float, placement_position: Sequence[float],
    relative_azimuth_deg: float,
) -> np.ndarray:
    """Build the frozen 11-D Stage-C geometry schema anchored at s1."""
    delta = np.asarray(target_position, dtype=np.float32) - np.asarray(s1_position, dtype=np.float32)
    candidate = {
        "relative_position": delta.tolist(),
        "snapped_position": np.asarray(target_snapped_position, dtype=np.float32).tolist(),
        "euclidean_distance_m": float(np.linalg.norm(delta)),
        "geodesic_distance_m": float(target_geodesic),
        "relative_azimuth_deg": float(relative_azimuth_deg),
    }
    return candidate_geometry_features(
        candidate,
        current_position=np.asarray(s1_position, dtype=np.float32),
        current_rotation_wxyz=np.asarray(s1_rotation_wxyz, dtype=np.float32),
        placement_position=np.asarray(placement_position, dtype=np.float32),
    )


__all__ = ["ContextKey", "CURRENT_DIM", "DELTA_SEMANTIC_DIM", "GEOMETRY_DIM", "TOP_K", "candidate_order", "context_key", "first_step_decision", "load_pairwise_and_azimuths", "order_candidates", "relative_view_descriptor", "second_step_decision", "second_step_geometry", "second_step_utility", "semantic_delta", "trajectory_cost", "viewpoint_azimuth", "viewpoint_radius", "wrap_relative_azimuth"]
