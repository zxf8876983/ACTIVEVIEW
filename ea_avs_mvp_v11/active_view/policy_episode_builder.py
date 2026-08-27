"""Stage A episode construction from cached v11.5 offline observations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REGIONS: Tuple[str, ...] = ("bedroom", "living_room", "kitchen", "dining_area")
OFFLINE_VERSION = "semantic-region-offline-v2"
CANDIDATE_VERSION = "semantic-region-v2"
EXPECTED_SKELETON_SHAPE = (32, 3, 30, 17)
PathCostFn = Callable[[Any, np.ndarray, np.ndarray], Optional[float]]


@dataclass(frozen=True)
class SceneIndex:
    scene_id: str
    scene_dir: Path
    manifest: Mapping[str, Any]
    placements: Mapping[str, Mapping[str, Any]]
    items: Mapping[Tuple[str, str], Mapping[str, Any]]


def stable_episode_seed(global_seed: int, record_id: str, scene_id: str, region: str) -> int:
    """Create a reproducible seed independent of Python hash randomization."""
    payload = f"{global_seed}|{record_id}|{scene_id}|{region}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def choose_current_view(valid_viewpoint_ids: Sequence[int], seed: int) -> int:
    """Choose one current view deterministically from sorted valid IDs."""
    if not valid_viewpoint_ids:
        raise ValueError("Cannot choose a current view from an empty set")
    rng = np.random.default_rng(seed)
    values = np.asarray(sorted(int(item) for item in valid_viewpoint_ids), dtype=np.int64)
    return int(values[int(rng.integers(0, len(values)))])


def build_navigation_geometry_pool(
    *, current_viewpoint_id: int, views: Mapping[int, Mapping[str, Any]],
    pathfinder: Any, path_cost_fn: PathCostFn,
) -> List[Dict[str, Any]]:
    """Build action-independent candidates using geometry and dynamic paths."""
    current = views[int(current_viewpoint_id)]
    current_position = np.asarray(current["agent_position"], dtype=np.float32)
    candidates: List[Dict[str, Any]] = []
    for viewpoint_id in sorted(views):
        if int(viewpoint_id) == int(current_viewpoint_id):
            continue
        item = views[viewpoint_id]
        candidate_position = np.asarray(item["agent_position"], dtype=np.float32)
        if not np.isfinite(candidate_position).all():
            continue
        geodesic = path_cost_fn(pathfinder, current_position, candidate_position)
        if geodesic is None or not np.isfinite(geodesic):
            continue
        current_azimuth = float(current.get("azimuth_deg", 0.0))
        candidate_azimuth = float(item.get("azimuth_deg", 0.0))
        relative_azimuth = (candidate_azimuth - current_azimuth + 180.0) % 360.0 - 180.0
        candidates.append({
            "viewpoint_id": int(viewpoint_id),
            "position": np.asarray(item["position"], dtype=np.float32).tolist(),
            "snapped_position": np.asarray(item["snapped_position"], dtype=np.float32).tolist(),
            "relative_position": (candidate_position - current_position).tolist(),
            "euclidean_distance_m": float(np.linalg.norm(candidate_position - current_position)),
            "geodesic_distance_m": float(geodesic),
            "relative_azimuth_deg": float(relative_azimuth),
            "rotation_wxyz": list(item["rotation_wxyz"]),
        })
    return candidates


def materialize_candidate_pool(
    geometry_pool: Sequence[Mapping[str, Any]], *, valid_skeleton_ids: Sequence[int],
    archive_path: Path, pose_confidence_available: bool,
) -> List[Dict[str, Any]]:
    """Bind record-specific skeleton metadata after geometry filtering."""
    valid_ids = {int(item) for item in valid_skeleton_ids}
    source_path = str(archive_path.resolve())
    materialized: List[Dict[str, Any]] = []
    for candidate in geometry_pool:
        if int(candidate["viewpoint_id"]) not in valid_ids:
            continue
        materialized.append({
            **dict(candidate),
            "skeleton_source_path": source_path,
            "pose_confidence_available": bool(pose_confidence_available),
        })
    return materialized


def build_dynamic_candidate_pool(
    *, current_viewpoint_id: int, views: Mapping[int, Mapping[str, Any]],
    valid_skeleton_ids: Sequence[int], pathfinder: Any, path_cost_fn: PathCostFn,
) -> List[Dict[str, Any]]:
    """Compatibility wrapper for non-cached, record-local candidate materialization.

    Episode construction uses :func:`build_navigation_geometry_pool` for caching
    and :func:`materialize_candidate_pool` per record. This wrapper intentionally
    never participates in the geometry cache.
    """
    geometry_pool = build_navigation_geometry_pool(
        current_viewpoint_id=current_viewpoint_id,
        views=views,
        pathfinder=pathfinder,
        path_cost_fn=path_cost_fn,
    )
    valid_ids = {int(item) for item in valid_skeleton_ids}
    return [
        {
            **candidate,
            "skeleton_source_path": str(views[int(candidate["viewpoint_id"])]
                                      ["skeleton_source_path"]),
            "pose_confidence_available": bool(
                views[int(candidate["viewpoint_id"])]
                .get("pose_confidence_available", False)
            ),
        }
        for candidate in geometry_pool
        if int(candidate["viewpoint_id"]) in valid_ids
    ]


def load_scene_index(scene_dir: Path) -> SceneIndex:
    """Validate the v2 scene and candidate manifests and index action packs."""
    manifest_path = scene_dir / "manifest.json"
    candidate_path = scene_dir / "candidate_metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_path.read_text(encoding="utf-8"))
    if manifest.get("version") != OFFLINE_VERSION:
        raise ValueError(f"Unsupported offline manifest version in {scene_dir}")
    if candidate_manifest.get("version") != CANDIDATE_VERSION:
        raise ValueError(f"Unsupported candidate manifest version in {scene_dir}")
    if int(manifest.get("records", 0)) != 980 or int(manifest.get("regions", 0)) != 4:
        raise ValueError(f"Unexpected scene record/region count in {scene_dir}")
    if int(manifest.get("views_per_record", 0)) != 32:
        raise ValueError(f"Unexpected view count in {scene_dir}")
    items: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for item in manifest.get("items", []):
        key = (str(item["record_id"]), str(item["region"]))
        if key in items:
            raise ValueError(f"Duplicate offline item {key} in {scene_dir}")
        items[key] = item
    placements = {str(item["region"]): item for item in candidate_manifest.get("placements_data", [])}
    if set(placements) != set(REGIONS):
        raise ValueError(f"Missing candidate regions in {scene_dir}")
    return SceneIndex(scene_dir.name, scene_dir, manifest, placements, items)


def _archive_view_data(archive_path: Path, item: Mapping[str, Any]) -> Tuple[Dict[int, Mapping[str, Any]], List[int]]:
    """Read only cached skeleton shape/finiteness and geometry arrays."""
    if not archive_path.exists():
        raise FileNotFoundError(str(archive_path))
    with np.load(archive_path, allow_pickle=False) as archive:
        if "skeleton" not in archive or tuple(archive["skeleton"].shape) != EXPECTED_SKELETON_SHAPE:
            raise ValueError(f"Unexpected skeleton shape in {archive_path}")
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        if viewpoint_ids.shape != (32,):
            raise ValueError(f"Unexpected viewpoint_ids shape in {archive_path}")
        required = ("viewpoint_positions", "viewpoint_snapped_positions", "viewpoint_agent_positions", "viewpoint_rotations_wxyz")
        if any(key not in archive for key in required):
            raise ValueError(f"Missing navigation fields in {archive_path}")
        geometry = {
            "position": np.asarray(archive["viewpoint_positions"], dtype=np.float32),
            "snapped_position": np.asarray(archive["viewpoint_snapped_positions"], dtype=np.float32),
            "agent_position": np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32),
            "rotation_wxyz": np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32),
        }
        pose_confidence_available = "confidence" in archive
    if any(value.shape != (32, 3) for key, value in geometry.items() if key != "rotation_wxyz") or geometry["rotation_wxyz"].shape != (32, 4):
        raise ValueError(f"Invalid navigation array shapes in {archive_path}")
    finite = np.isfinite(skeleton).all(axis=(1, 2, 3))
    views: Dict[int, Mapping[str, Any]] = {}
    for index, viewpoint_id in enumerate(viewpoint_ids.tolist()):
        views[int(viewpoint_id)] = {
            "position": geometry["position"][index],
            "snapped_position": geometry["snapped_position"][index],
            "agent_position": geometry["agent_position"][index],
            "rotation_wxyz": geometry["rotation_wxyz"][index].tolist(),
            "pose_confidence_available": bool(pose_confidence_available),
            "skeleton_source_path": str(archive_path.resolve()),
        }
    return views, [int(viewpoint_id) for viewpoint_id, ok in zip(viewpoint_ids.tolist(), finite.tolist()) if ok]


def _scene_region_geometry(
    index: SceneIndex, region: str, *, pathfinder: Any, path_cost_fn: PathCostFn,
    clearance_m: float,
) -> Dict[int, Mapping[str, Any]]:
    placement = index.placements[region]
    placement_position = np.asarray(placement["position"], dtype=np.float32)
    views: Dict[int, Mapping[str, Any]] = {}
    for raw_view in placement["viewpoints"]:
        viewpoint_id = int(raw_view["viewpoint_id"])
        position = np.asarray(raw_view["position"], dtype=np.float32)
        snapped = np.asarray(raw_view.get("snapped_position", position), dtype=np.float32)
        if not np.isfinite(position).all() or not np.isfinite(snapped).all():
            continue
        if not pathfinder.is_navigable(snapped):
            continue
        try:
            if float(pathfinder.distance_to_closest_obstacle(snapped)) < clearance_m:
                continue
        except RuntimeError:
            continue
        if path_cost_fn(pathfinder, placement_position, snapped) is None:
            continue
        views[viewpoint_id] = {
            "position": position,
            "snapped_position": snapped,
            "agent_position": snapped,
            "rotation_wxyz": list(raw_view["camera_rotation_wxyz"]),
            "azimuth_deg": float(raw_view.get("azimuth_deg", 0.0)),
            "radius_m": float(raw_view.get("radius_m", 0.0)),
        }
    return views


def iter_scene_region_episodes(
    index: SceneIndex, *, policy_records: Sequence[Mapping[str, Any]], region: str,
    pathfinder: Any, path_cost_fn: PathCostFn, global_seed: int, clearance_m: float,
) -> Iterable[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """Yield ``(episode, exclusion)`` for one scene/region."""
    geometry = _scene_region_geometry(index, region, pathfinder=pathfinder, path_cost_fn=path_cost_fn, clearance_m=clearance_m)
    # Cache only scene/region navigation geometry. Record-specific skeleton
    # validity and source paths are intersected/bound below for every record.
    dynamic_cache: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    placement = index.placements[region]
    for record in policy_records:
        record_id = str(record["record_id"])
        item = index.items.get((record_id, region))
        base_exclusion = {"scene_id": index.scene_id, "region": region, "record_id": record_id, "policy_split": record["policy_split"]}
        if item is None:
            yield None, {**base_exclusion, "excluded_reason": "missing_cached_skeleton"}
            continue
        archive_path = index.scene_dir / str(item["path"])
        try:
            cached_views, valid_skeleton_ids = _archive_view_data(archive_path, item)
        except (FileNotFoundError, OSError, ValueError):
            yield None, {**base_exclusion, "excluded_reason": "missing_cached_skeleton", "skeleton_source_path": str(archive_path)}
            continue
        merged_views: Dict[int, Mapping[str, Any]] = {}
        for viewpoint_id, view in geometry.items():
            if viewpoint_id not in cached_views:
                continue
            merged = dict(view)
            merged["skeleton_source_path"] = str(archive_path.resolve())
            merged["pose_confidence_available"] = bool(cached_views[viewpoint_id]["pose_confidence_available"])
            merged_views[viewpoint_id] = merged
        valid_starts = sorted(set(merged_views) & set(valid_skeleton_ids))
        if not valid_starts:
            yield None, {**base_exclusion, "excluded_reason": "no_valid_grid_start"}
            continue
        current_id = choose_current_view(valid_starts, stable_episode_seed(global_seed, record_id, index.scene_id, region))
        cache_key = (index.scene_id, region, current_id)
        if cache_key not in dynamic_cache:
            dynamic_cache[cache_key] = build_navigation_geometry_pool(
                current_viewpoint_id=current_id, views=geometry,
                pathfinder=pathfinder, path_cost_fn=path_cost_fn,
            )
        candidate_pool = materialize_candidate_pool(
            dynamic_cache[cache_key],
            valid_skeleton_ids=valid_skeleton_ids,
            archive_path=archive_path,
            pose_confidence_available=bool(cached_views[current_id]["pose_confidence_available"]),
        )
        if not candidate_pool:
            yield None, {**base_exclusion, "excluded_reason": "no_reachable_next_candidate", "current_viewpoint_id": current_id}
            continue
        current = merged_views[current_id]
        episode_id = f"{record_id}__{index.scene_id}__{region}__v{current_id:02d}"
        current_view = {
            "viewpoint_id": current_id,
            "position": np.asarray(current["position"], dtype=np.float32).tolist(),
            "snapped_position": np.asarray(current["snapped_position"], dtype=np.float32).tolist(),
            "agent_position": np.asarray(cached_views[current_id]["agent_position"], dtype=np.float32).tolist(),
            "rotation_wxyz": list(current["rotation_wxyz"]),
            "skeleton_source_path": str(archive_path.resolve()),
            "pose_confidence_available": bool(current["pose_confidence_available"]),
        }
        yield {
            "episode_id": episode_id,
            "policy_split": str(record["policy_split"]),
            "record_id": record_id,
            "action_label": str(record["action_label"]),
            "label_id": int(record["label_id"]),
            "scene_id": index.scene_id,
            "region": region,
            "placement_id": str(placement["placement_id"]),
            "current_view": current_view,
            "candidate_pool": candidate_pool,
            "candidate_count": len(candidate_pool),
            "protocol_version": "policy-episode-v11.5-stage-a",
            "candidate_selection_basis": "dynamic_navmesh_reachability_and_cached_data_integrity_only",
        }, None
