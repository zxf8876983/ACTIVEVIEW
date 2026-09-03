"""文件用途：
    生成 ActiveView 离线策略数据。

主要输入：
    - 离线观测、动作记录和识别结果。
主要输出：
    - episode、utility 或数据审计结果。
项目角色：
    - 属于 data.generation 数据生成模块。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


REGIONS: Tuple[str, ...] = ("bedroom", "living_room", "kitchen", "dining_area")
OFFLINE_VERSION = "semantic-region-offline-v2"
CANDIDATE_VERSION = "semantic-region-v2"
EXPECTED_SKELETON_SHAPE = (32, 3, 30, 17)
PathCostFn = Callable[[Any, np.ndarray, np.ndarray], Optional[float]]
_FINITE_CANDIDATE_FIELDS = ("euclidean_distance_m", "geodesic_distance_m", "relative_azimuth_deg")
_FORBIDDEN_EPISODE_FIELDS = {
    "prediction", "predicted_class", "entropy", "correctness", "q_pred",
    "utility_label", "utility_target", "future_rgb", "future_depth",
    "future_rgb_path", "future_depth_path", "rgb", "depth", "rgb_path",
    "depth_path", "pose_2d", "pose_3d", "gt_skeleton", "gt_joints",
    "smpl_joints", "valid_skeleton_ids", "candidate_entropy",
    "candidate_prediction", "candidate_stgcn_probabilities",
    "candidate_pose_confidence", "candidate_skeleton", "candidate_correct",
    "future_skeleton", "stgcn_probabilities", "pose_confidence",
}


@dataclass(frozen=True)
class SceneIndex:
    scene_id: str
    scene_dir: Path
    manifest: Mapping[str, Any]
    placements: Mapping[str, Mapping[str, Any]]
    items: Mapping[Tuple[str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class CachedSkeletonInfo:
    """Archive schema and per-view validity/geometry information."""

    valid_skeleton_ids: Set[int]
    nonfinite_skeleton_ids: Set[int]
    error: Optional[str]
    geometry: Mapping[int, Mapping[str, Any]]


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


def _finite_sequence(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def _finite_shape(value: Any, shape: Tuple[int, ...]) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
        return array.shape == shape and bool(np.isfinite(array).all())
    except (TypeError, ValueError):
        return False


def _same_geometry(left: Any, right: Any, shape: Tuple[int, ...], atol: float = 1e-5) -> bool:
    """Compare serialized Episode geometry with its cached NPZ counterpart."""
    if not _finite_shape(left, shape):
        return False
    try:
        return bool(np.allclose(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            rtol=0.0,
            atol=atol,
        ))
    except (TypeError, ValueError):
        return False


def _forbidden_keys(value: Any) -> Set[str]:
    """Return forbidden schema keys found recursively in an Episode."""
    found: Set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_EPISODE_FIELDS:
                found.add(key_text)
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def _inspect_cached_skeleton(path: Path) -> CachedSkeletonInfo:
    """Validate archive schema while allowing individual failed viewpoints."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "skeleton" not in archive or "viewpoint_ids" not in archive:
                return CachedSkeletonInfo(set(), set(), "missing_skeleton_fields", {})
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            if skeleton.shape != EXPECTED_SKELETON_SHAPE or viewpoint_ids.shape != (32,):
                return CachedSkeletonInfo(set(), set(), "invalid_skeleton_shape", {})
            required = (
                "viewpoint_positions", "viewpoint_snapped_positions",
                "viewpoint_agent_positions", "viewpoint_rotations_wxyz",
            )
            if any(key not in archive for key in required):
                return CachedSkeletonInfo(set(), set(), "missing_navigation_fields", {})
            if len(set(viewpoint_ids.tolist())) != 32:
                return CachedSkeletonInfo(set(), set(), "duplicate_viewpoint_ids", {})
            expected_shapes = {
                "viewpoint_positions": (32, 3),
                "viewpoint_snapped_positions": (32, 3),
                "viewpoint_agent_positions": (32, 3),
                "viewpoint_rotations_wxyz": (32, 4),
            }
            for key, shape in expected_shapes.items():
                values = np.asarray(archive[key], dtype=np.float32)
                if values.shape != shape:
                    return CachedSkeletonInfo(set(), set(), "invalid_navigation_shapes", {})
                if not np.isfinite(values).all():
                    return CachedSkeletonInfo(set(), set(), "invalid_navigation_finiteness", {})
            if not np.isin(viewpoint_ids, np.arange(32, dtype=np.int64)).all():
                return CachedSkeletonInfo(set(), set(), "invalid_viewpoint_ids", {})
            finite = np.isfinite(skeleton).all(axis=(1, 2, 3))
            if finite.shape != (32,):
                return CachedSkeletonInfo(set(), set(), "invalid_skeleton_finiteness", {})
            finite_ids = {
                int(viewpoint_id)
                for viewpoint_id, is_finite in zip(viewpoint_ids.tolist(), finite.tolist())
                if is_finite
            }
            nonfinite_ids = {
                int(viewpoint_id)
                for viewpoint_id, is_finite in zip(viewpoint_ids.tolist(), finite.tolist())
                if not is_finite
            }
            geometry = {
                int(viewpoint_id): {
                    "position": np.asarray(archive["viewpoint_positions"][index], dtype=np.float32),
                    "snapped_position": np.asarray(archive["viewpoint_snapped_positions"][index], dtype=np.float32),
                    "agent_position": np.asarray(archive["viewpoint_agent_positions"][index], dtype=np.float32),
                    "rotation_wxyz": np.asarray(archive["viewpoint_rotations_wxyz"][index], dtype=np.float32),
                }
                for index, viewpoint_id in enumerate(viewpoint_ids.tolist())
            }
            return CachedSkeletonInfo(finite_ids, nonfinite_ids, None, geometry)
    except (OSError, ValueError, TypeError):
        return CachedSkeletonInfo(set(), set(), "unreadable_skeleton_archive", {})


def audit_episode_files(
    episode_files: Mapping[str, Path], *,
    expected_record_splits: Optional[Mapping[str, str]] = None,
    valid_viewpoint_ids: Optional[Set[int]] = None,
    validate_cached_skeletons: bool = False,
) -> Dict[str, Any]:
    """Audit serialized Episode manifests and derive integrity booleans.

    ``validate_cached_skeletons`` performs an archive-level shape, navigation
    field, and finite-view check. It is disabled by default for fast schema
    checks and can be enabled by the Stage A acceptance command.
    """
    valid_ids = set(range(32)) if valid_viewpoint_ids is None else {int(item) for item in valid_viewpoint_ids}
    counters = {
        "episodes": 0,
        "malformed_records": 0,
        "current_in_pool_violations": 0,
        "duplicate_candidate_ids": 0,
        "invalid_candidate_ids": 0,
        "candidate_count_mismatch": 0,
        "nonfinite_candidate_costs": 0,
        "nonfinite_candidate_geometry": 0,
        "dynamic_reachability_failures": 0,
        "split_mismatch": 0,
        "same_record_split_violations": 0,
        "candidate_path_mismatch": 0,
        "missing_skeleton_paths": 0,
        "cached_skeleton_file_errors": 0,
        "cached_skeleton_shape_violations": 0,
        # Informational only: individual failed viewpoint archives are allowed
        # as long as the Episode never references them.
        "nonfinite_cached_skeleton_viewpoints": 0,
        "current_view_data_violations": 0,
        "candidate_skeleton_data_violations": 0,
        "npz_geometry_mismatches": 0,
        "duplicate_episode_keys": 0,
        "duplicate_episode_ids": 0,
        "empty_candidate_pool": 0,
        "forbidden_information_violations": 0,
    }
    record_splits: Dict[str, str] = {}
    records_by_split: Dict[str, Set[str]] = {str(split): set() for split in episode_files}
    archive_cache: Dict[str, CachedSkeletonInfo] = {}
    seen_episode_keys: Set[Tuple[str, str, str]] = set()
    seen_episode_ids: Set[str] = set()

    for split, path in episode_files.items():
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError:
                    counters["malformed_records"] += 1
                    continue
                counters["episodes"] += 1
                if not isinstance(episode, Mapping):
                    counters["malformed_records"] += 1
                    continue
                record_id = str(episode.get("record_id", ""))
                declared_split = str(episode.get("policy_split", ""))
                episode_id = str(episode.get("episode_id", ""))
                episode_key = (
                    record_id,
                    str(episode.get("scene_id", "")),
                    str(episode.get("region", "")),
                )
                if episode_key in seen_episode_keys:
                    counters["duplicate_episode_keys"] += 1
                seen_episode_keys.add(episode_key)
                if episode_id in seen_episode_ids:
                    counters["duplicate_episode_ids"] += 1
                seen_episode_ids.add(episode_id)
                records_by_split.setdefault(str(split), set()).add(record_id)
                if declared_split != str(split):
                    counters["split_mismatch"] += 1
                if expected_record_splits is not None and declared_split != expected_record_splits.get(record_id):
                    counters["split_mismatch"] += 1
                previous_split = record_splits.setdefault(record_id, declared_split)
                if previous_split != declared_split:
                    counters["same_record_split_violations"] += 1

                if _forbidden_keys(episode):
                    counters["forbidden_information_violations"] += 1
                current = episode.get("current_view", {})
                if not isinstance(current, Mapping):
                    counters["current_view_data_violations"] += 1
                    current = {}
                current_id = current.get("viewpoint_id")
                current_path = current.get("skeleton_source_path")
                for field, shape in (
                    ("position", (3,)), ("snapped_position", (3,)),
                    ("agent_position", (3,)), ("rotation_wxyz", (4,)),
                ):
                    if not _finite_shape(current.get(field), shape):
                        counters["current_view_data_violations"] += 1
                if not isinstance(current_id, int) or current_id not in valid_ids:
                    counters["invalid_candidate_ids"] += 1
                if not isinstance(current_path, str) or not Path(current_path).exists():
                    counters["missing_skeleton_paths"] += 1
                cached_valid_ids: Set[int] = set()
                cache_error: Optional[str] = "not_checked"
                if validate_cached_skeletons and isinstance(current_path, str):
                    if current_path not in archive_cache:
                        archive_cache[current_path] = _inspect_cached_skeleton(Path(current_path))
                    cached = archive_cache[current_path]
                    cached_valid_ids, cache_error = cached.valid_skeleton_ids, cached.error
                    if cache_error is not None:
                        counters["cached_skeleton_file_errors"] += 1
                        if cache_error in {
                            "invalid_skeleton_shape", "missing_skeleton_fields",
                            "missing_navigation_fields", "duplicate_viewpoint_ids",
                            "invalid_navigation_shapes", "invalid_viewpoint_ids",
                        }:
                            counters["cached_skeleton_shape_violations"] += 1
                    counters["nonfinite_cached_skeleton_viewpoints"] += len(cached.nonfinite_skeleton_ids)
                    if not isinstance(current_id, int) or current_id not in cached_valid_ids:
                        counters["current_view_data_violations"] += 1
                    elif cache_error is None:
                        cached_geometry = cached.geometry.get(current_id)
                        if cached_geometry is None or any(
                            not _same_geometry(current.get(field), cached_geometry[field], shape)
                            for field, shape in (
                                ("position", (3,)), ("snapped_position", (3,)),
                                ("agent_position", (3,)), ("rotation_wxyz", (4,)),
                            )
                        ):
                            counters["npz_geometry_mismatches"] += 1

                candidates = episode.get("candidate_pool", [])
                if not isinstance(candidates, list) or not candidates:
                    counters["empty_candidate_pool"] += 1
                    continue
                if episode.get("candidate_count") != len(candidates):
                    counters["candidate_count_mismatch"] += 1
                candidate_ids = []
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        counters["malformed_records"] += 1
                        continue
                    viewpoint_id = candidate.get("viewpoint_id")
                    candidate_ids.append(viewpoint_id)
                    if not isinstance(viewpoint_id, int) or viewpoint_id not in valid_ids:
                        counters["invalid_candidate_ids"] += 1
                    if validate_cached_skeletons and viewpoint_id not in cached_valid_ids:
                        counters["candidate_skeleton_data_violations"] += 1
                    if validate_cached_skeletons and cache_error is None:
                        cached_geometry = cached.geometry.get(viewpoint_id)
                        if cached_geometry is None or any(
                            not _same_geometry(candidate.get(field), cached_geometry[field], shape)
                            for field, shape in (
                                ("position", (3,)), ("snapped_position", (3,)),
                                ("rotation_wxyz", (4,)),
                            )
                        ):
                            counters["npz_geometry_mismatches"] += 1
                    if viewpoint_id == current_id:
                        counters["current_in_pool_violations"] += 1
                    if not all(_finite_sequence(candidate.get(field)) for field in _FINITE_CANDIDATE_FIELDS):
                        counters["nonfinite_candidate_costs"] += 1
                        counters["dynamic_reachability_failures"] += 1
                    if not all(_finite_shape(candidate.get(field), shape) for field, shape in (
                        ("position", (3,)), ("snapped_position", (3,)),
                        ("relative_position", (3,)), ("rotation_wxyz", (4,)),
                    )):
                        counters["nonfinite_candidate_geometry"] += 1
                    if candidate.get("skeleton_source_path") != current_path:
                        counters["candidate_path_mismatch"] += 1
                    if not isinstance(candidate.get("skeleton_source_path"), str) or not Path(candidate["skeleton_source_path"]).exists():
                        counters["missing_skeleton_paths"] += 1
                if len(candidate_ids) != len(set(candidate_ids)):
                    counters["duplicate_candidate_ids"] += 1

    split_sets = list(records_by_split.values())
    overlap = any(split_sets[i].intersection(split_sets[j]) for i in range(len(split_sets)) for j in range(i + 1, len(split_sets)))
    counters["split_overlap"] = bool(overlap)
    return {
        "counts": counters,
        "integrity_checks": {
            # Preserve the historical meaning: ``false`` means no overlap.
            "split_overlap": bool(counters["split_overlap"]),
            "same_record_same_split_across_scenes": counters["same_record_split_violations"] == 0,
            "current_not_in_candidate_pool": counters["current_in_pool_violations"] == 0,
            "all_candidate_ids_valid": counters["invalid_candidate_ids"] == 0,
            "all_candidate_costs_finite": counters["nonfinite_candidate_costs"] == 0,
            "all_candidates_dynamic_reachable": counters["dynamic_reachability_failures"] == 0,
            "all_candidate_paths_match_record": counters["candidate_path_mismatch"] == 0,
            "all_current_views_cached": counters["missing_skeleton_paths"] == 0,
            "all_cached_skeletons_complete": (
                counters["cached_skeleton_file_errors"] == 0
                and counters["cached_skeleton_shape_violations"] == 0
            ),
            "all_current_view_data_valid": counters["current_view_data_violations"] == 0,
            "all_candidate_skeleton_data_valid": counters["candidate_skeleton_data_violations"] == 0,
            "unique_record_scene_region": counters["duplicate_episode_keys"] == 0,
            "unique_episode_ids": counters["duplicate_episode_ids"] == 0,
            "episode_geometry_matches_npz": counters["npz_geometry_mismatches"] == 0,
            "no_forbidden_future_perception_fields": counters["forbidden_information_violations"] == 0,
        },
    }


def audit_episode_coverage(
    episode_files: Mapping[str, Path], exclusions_file: Path,
    *, policy_records: Sequence[Mapping[str, Any]],
    complete_scene_ids: Sequence[str], regions: Sequence[str],
) -> Dict[str, Any]:
    """Verify that every expected record/scene/region tuple is accounted for.

    A tuple is accounted for exactly once by either a serialized Episode or a
    record-level exclusion. Scene-level exclusions (for example an incomplete
    scene that is not in ``complete_scene_ids``) are intentionally ignored.
    """
    expected = {
        (str(record["record_id"]), str(scene_id), str(region))
        for record in policy_records
        for scene_id in complete_scene_ids
        for region in regions
    }
    episode_tuples: List[Tuple[str, str, str]] = []
    malformed_episode_tuples = 0
    for path in episode_files.values():
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError:
                    malformed_episode_tuples += 1
                    continue
                if not isinstance(episode, Mapping):
                    malformed_episode_tuples += 1
                    continue
                key = (
                    str(episode.get("record_id", "")),
                    str(episode.get("scene_id", "")),
                    str(episode.get("region", "")),
                )
                if not all(key):
                    malformed_episode_tuples += 1
                    continue
                episode_tuples.append(key)

    exclusion_tuples: List[Tuple[str, str, str]] = []
    malformed_exclusion_tuples = 0
    if Path(exclusions_file).exists():
        with Path(exclusions_file).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    exclusion = json.loads(line)
                except json.JSONDecodeError:
                    malformed_exclusion_tuples += 1
                    continue
                if not isinstance(exclusion, Mapping):
                    malformed_exclusion_tuples += 1
                    continue
                key = (
                    str(exclusion.get("record_id", "")),
                    str(exclusion.get("scene_id", "")),
                    str(exclusion.get("region", "")),
                )
                # Scene-level failures do not represent an expected
                # record×scene×region tuple and are excluded from coverage.
                if not all(key):
                    continue
                exclusion_tuples.append(key)
    else:
        malformed_exclusion_tuples += 1

    episodes_set = set(episode_tuples)
    exclusions_set = set(exclusion_tuples)
    accounted = episodes_set | exclusions_set
    duplicate_accounted = len(episode_tuples) + len(exclusion_tuples) - len(
        set(episode_tuples + exclusion_tuples)
    )
    overlap = episodes_set & exclusions_set
    missing = expected - accounted
    unexpected = accounted - expected
    counts = {
        "expected_tuple_count": len(expected),
        "episode_tuple_count": len(episode_tuples),
        "exclusion_tuple_count": len(exclusion_tuples),
        "observed_tuple_count": len(accounted),
        "missing_tuple_count": len(missing),
        "unexpected_tuple_count": len(unexpected),
        "duplicate_accounted_tuple_count": duplicate_accounted,
        "episode_and_exclusion_overlap": len(overlap),
        "malformed_episode_tuple_count": malformed_episode_tuples,
        "malformed_exclusion_tuple_count": malformed_exclusion_tuples,
    }
    return {
        "counts": counts,
        "integrity_checks": {
            "complete_tuple_coverage": (
                counts["missing_tuple_count"] == 0
                and counts["unexpected_tuple_count"] == 0
                and counts["duplicate_accounted_tuple_count"] == 0
                and counts["episode_and_exclusion_overlap"] == 0
                and malformed_episode_tuples == 0
                and malformed_exclusion_tuples == 0
            ),
        },
    }


def audit_scene_coverage(
    target_scene_ids: Sequence[str], used_scene_ids: Sequence[str],
) -> Dict[str, Any]:
    """Ensure every requested scene entered Episode generation exactly once."""
    target = [str(scene_id) for scene_id in target_scene_ids]
    used = [str(scene_id) for scene_id in used_scene_ids]
    target_set = set(target)
    used_set = set(used)
    missing = sorted(target_set - used_set)
    unexpected = sorted(used_set - target_set)
    counts = {
        "target_scene_count": len(target_set),
        "used_scene_count": len(used_set),
        "failed_scene_count": len(missing),
        "missing_scene_ids": missing,
        "unexpected_scene_ids": unexpected,
        "duplicate_target_scene_count": len(target) - len(target_set),
        "duplicate_used_scene_count": len(used) - len(used_set),
    }
    return {
        "counts": counts,
        "integrity_checks": {
            "all_target_scenes_used": (
                not missing
                and not unexpected
                and counts["duplicate_target_scene_count"] == 0
                and counts["duplicate_used_scene_count"] == 0
            ),
        },
    }


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
    for name, payload in (("offline", manifest), ("candidate", candidate_manifest)):
        if payload.get("rotation_reference") != "exact_offline_render_state":
            raise ValueError(f"{name} rotation metadata is not tied to the offline render state in {scene_dir}")
        if not math.isclose(float(payload.get("sensor_height_m", -1.0)), 1.1, abs_tol=1e-6):
            raise ValueError(f"Unexpected sensor height in {name} manifest: {scene_dir}")
        if not math.isclose(float(payload.get("target_height_m", -1.0)), 0.85, abs_tol=1e-6):
            raise ValueError(f"Unexpected target height in {name} manifest: {scene_dir}")
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
