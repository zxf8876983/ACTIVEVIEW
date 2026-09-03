"""Canonical geometry and candidate-ordering primitives for the final method."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.active_view.stage_d_dataset import load_pairwise_geodesic
from activeview.active_view.stage_d_policy import order_candidates

VIEW_COUNT = 32
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


__all__ = ["ContextKey", "candidate_order", "context_key", "load_pairwise_and_azimuths", "order_candidates", "relative_view_descriptor", "viewpoint_azimuth", "viewpoint_radius"]
