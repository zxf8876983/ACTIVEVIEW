#!/usr/bin/env python3
"""Known-map plus current-depth movement-aware NBV audit.

This is a Train/Moving-Val diagnostic for the reduced12 eight-placement
protocol.  RGBGlobal-Visibility is the deployable observation score; Habitat
navmesh paths and transient current-frame depth are used only to score
movement distance/risk.  Raw depth, point clouds and candidate depth are
never persisted.  Policy Test is intentionally never loaded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path  # noqa: E402
from activeview.data.motion.babel_clean_dataset_generator import (  # noqa: E402
    MotionConverter,
    _load_resampled_motion,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.data.motion.habitat_h36m17_fk import H36M17_LINKS  # noqa: E402
from activeview.scripts.data.generate_hm3d_train_rgb_observations import _load_skeleton_metadata, _set_agent_state  # noqa: E402
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation  # noqa: E402
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (  # noqa: E402
    VisibilityPredictor,
    _option_geometry,
    _predict,
)
from activeview.scripts.experiments.run_reduced12_known_map_geometric_nbv_upgrade import _load_map_cache  # noqa: E402
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (  # noqa: E402
    _load_rows,
    _read_npz,
    _signature,
    _train_prior,
)
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import (  # noqa: E402
    _candidate_mask,
    _load_yaw8_options,
)
from activeview.scripts.experiments.run_reduced12_yaw8_shared_head_fairness_audit import SharedHead  # noqa: E402
from activeview.scripts.experiments.rgbd_human_state_helpers import (  # noqa: E402
    SENSOR_HEIGHT,
    camera_backproject,
    placement_yaw,
    rotation_wxyz_to_matrix,
)

SEED = 42
MAX_OPTIONS = 22
IMAGE_SIZE = 256
HFOV_DEG = 75.0
DEPTH_RUNTIME_REL = Path("diagnostics/movement_aware_depth_nbv")
OUTPUT_DEFAULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/movement_aware_depth_nbv"
VIS_CKPT_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "frame0_visibility_predictor_v1/RGBGlobal+Geometry.pth"
)
HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
)
RAW_VAL_REL = Path("datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json")


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _scene_file(scene_root: Path, scene_id: str, suffix: str) -> Path:
    matches = sorted((scene_root / scene_id).glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {suffix} asset for {scene_id}, found {len(matches)}")
    return matches[0]


def _scene_sim(scene_root: Path, scene_id: str) -> Any:
    import habitat_sim

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(_scene_file(scene_root, scene_id, ".basis.glb"))
    config.enable_physics = True
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "depth_0"
    sensor.sensor_type = habitat_sim.SensorType.DEPTH
    sensor.resolution = [IMAGE_SIZE, IMAGE_SIZE]
    sensor.position = _mn().Vector3(0.0, SENSOR_HEIGHT, 0.0)
    sensor.hfov = HFOV_DEG
    sensor.near = 0.01
    sensor.far = 100.0
    agent = habitat_sim.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    navmesh = _scene_file(scene_root, scene_id, ".basis.navmesh")
    if not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError(f"failed to load navmesh: {scene_id}")
    return sim


def _mn() -> Any:
    import magnum as mn

    return mn


def _motion_map(data_root: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads((data_root / RAW_VAL_REL).read_text(encoding="utf-8"))
    return {str(item["record_id"]): item for item in payload}


def _sample_polyline(points: Sequence[Any], interval: float = 0.10) -> np.ndarray:
    values = [np.asarray(point, dtype=np.float32) for point in points]
    if len(values) < 2:
        return np.asarray(values, dtype=np.float32)
    output: list[np.ndarray] = [values[0]]
    for left, right in zip(values[:-1], values[1:]):
        delta = right - left
        distance = float(np.linalg.norm(delta))
        count = max(1, int(math.ceil(distance / interval)))
        for step in range(1, count + 1):
            output.append(left + delta * (step / count))
    return np.asarray(output, dtype=np.float32)


def _path(sim: Any, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, float]:
    import habitat_sim

    request = habitat_sim.ShortestPath()
    snapped_start = np.asarray(sim.pathfinder.snap_point(start), dtype=np.float32)
    snapped_end = np.asarray(sim.pathfinder.snap_point(end), dtype=np.float32)
    request.requested_start = _mn().Vector3(*snapped_start.tolist())
    request.requested_end = _mn().Vector3(*snapped_end.tolist())
    if not sim.pathfinder.find_path(request) or not np.isfinite(float(request.geodesic_distance)):
        return np.empty((0, 3), dtype=np.float32), float("inf")
    return _sample_polyline(request.points), float(request.geodesic_distance)


def _frame0_human_pose(
    human: Any,
    converted: Mapping[str, Any],
    placement: np.ndarray,
    yaw_deg: float,
) -> None:
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=yaw_deg)
    apply_humanoid_pose(
        human,
        joints[0],
        roots[0],
        base_position=placement,
        scene_yaw_deg=yaw_deg,
        floor_y=float(placement[1]),
        grounding_offset=float(offsets[0]),
    )


def _occupancy_worker(payload: Mapping[str, Any]) -> str:
    """Render current depth for one scene and persist only risk summaries."""
    data_root = Path(str(payload["data_root"]))
    scene_root = Path(str(payload["scene_root"]))
    output = Path(str(payload["output"]))
    rows = list(payload["rows"])
    scene_id = str(payload["scene_id"])
    yolo_path = Path(str(payload["yolo_path"]))
    with np.load(yolo_path, allow_pickle=False) as yolo_file:
        yolo = {key: np.asarray(yolo_file[key]) for key in yolo_file.files}
    motions = _motion_map(data_root)
    sim = _scene_sim(scene_root, scene_id)
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    metadata_cache: dict[str, dict[str, np.ndarray]] = {}
    converted_cache: dict[str, Mapping[str, Any]] = {}
    path_cost = np.full((len(rows), MAX_OPTIONS), np.inf, dtype=np.float32)
    human_risk = np.zeros((len(rows), MAX_OPTIONS), dtype=np.float32)
    depth_risk = np.zeros((len(rows), MAX_OPTIONS), dtype=np.float32)
    human_clearance = np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32)
    depth_clearance = np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32)
    rendered = np.zeros(len(rows), dtype=bool)
    obstacle_counts = np.zeros(len(rows), dtype=np.int32)
    human_point_counts = np.zeros(len(rows), dtype=np.int32)
    nonhuman_navigable_counts = np.zeros(len(rows), dtype=np.int32)
    occupancy_sample_counts = np.zeros(len(rows), dtype=np.int32)
    nonhuman_counts = np.zeros(len(rows), dtype=np.int32)
    focal = 0.5 * IMAGE_SIZE / math.tan(math.radians(HFOV_DEG) * 0.5)
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    for local, item in enumerate(rows):
        index = int(item["index"])
        source = str(item["archive_path"])
        metadata = metadata_cache.get(source)
        if metadata is None:
            with np.load(Path(source), allow_pickle=False) as archive:
                metadata = {
                    "placement_position": np.asarray(archive["placement_position"], dtype=np.float32),
                    "viewpoint_agent_positions": np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32),
                    "viewpoint_rotations_wxyz": np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32),
                }
            metadata_cache[source] = metadata
        record_id = str(item["record_id"])
        if record_id not in converted_cache:
            converted_cache[record_id] = converter.convert(_load_resampled_motion(motions[record_id], 30))
        _frame0_human_pose(
            human,
            converted_cache[record_id],
            metadata["placement_position"],
            placement_yaw(Path(source), str(item["region"])),
        )
        current = int(item["current_viewpoint_id"])
        positions = metadata["viewpoint_agent_positions"]
        rotations = metadata["viewpoint_rotations_wxyz"]
        _set_agent_state(sim.get_agent(0), positions[current], rotations[current])
        depth = np.asarray(sim.get_sensor_observations([0])[0]["depth_0"], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.01) & (depth < 100.0)
        cam = np.stack(((xx - 0.5 * IMAGE_SIZE) * depth / focal, -(yy - 0.5 * IMAGE_SIZE) * depth / focal, -depth), axis=-1)
        rotation = rotation_wxyz_to_matrix(rotations[current])
        sensor = positions[current].astype(np.float64) + np.asarray([0.0, SENSOR_HEIGHT, 0.0])
        world = np.einsum("ij,hwj->hwi", rotation, cam) + sensor
        bbox = np.asarray(yolo["bbox_xyxy"][index], dtype=np.float32)
        detected = bool(np.asarray(yolo.get("detected", np.zeros(len(yolo["bbox_xyxy"]), dtype=bool)))[index])
        bbox_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        human_center = None
        if detected and np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0, x1 = max(0, x0), min(IMAGE_SIZE - 1, x1)
            y0, y1 = max(0, y0), min(IMAGE_SIZE - 1, y1)
            bbox_mask[y0 : y1 + 1, x0 : x1 + 1] = True
            tx0, tx1 = int(x0 + 0.30 * (x1 - x0)), int(x0 + 0.70 * (x1 - x0))
            ty0, ty1 = int(y0 + 0.20 * (y1 - y0)), int(y0 + 0.75 * (y1 - y0))
            torso = valid[ty0 : ty1 + 1, tx0 : tx1 + 1]
            torso_depths = depth[ty0 : ty1 + 1, tx0 : tx1 + 1][torso]
            if torso_depths.size:
                median_depth = float(np.median(torso_depths))
                pixel = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
                _, center = camera_backproject(pixel, median_depth, positions[current], rotations[current])
                human_center = np.asarray(center, dtype=np.float32)[[0, 2]]
        # Archives store world coordinates with a scene-specific floor
        # elevation.  The protocol's 0.15--1.8 m band is relative to that
        # placement floor, not an absolute Habitat Y coordinate.
        floor_y = float(metadata["placement_position"][1])
        heights = world[..., 1] - floor_y
        obstacle_mask = valid & (heights >= 0.15) & (heights <= 1.8) & ~bbox_mask
        human_mask = valid & (heights >= 0.15) & (heights <= 1.8) & bbox_mask
        obstacle_points = world[obstacle_mask][..., [0, 2]].astype(np.float32)
        human_points = world[human_mask][..., [0, 2]].astype(np.float32)
        if obstacle_points.shape[0] > 2048:
            obstacle_points = obstacle_points[np.linspace(0, obstacle_points.shape[0] - 1, 2048).astype(np.int64)]
        occupancy_sample_counts[local] = int(obstacle_points.shape[0])
        obstacle_counts[local] = int(np.sum(obstacle_mask))
        human_point_counts[local] = int(np.sum(human_mask))
        nonhuman_counts[local] = int(np.sum(obstacle_mask))
        if obstacle_points.size:
            nonhuman_navigable_counts[local] = int(
                sum(bool(sim.pathfinder.is_navigable(_mn().Vector3(float(p[0]), 0.0, float(p[1])))) for p in obstacle_points)
            )
        path_cost[local, 0] = 0.0
        rendered[local] = True
        for slot, candidate in enumerate(item["candidate_ids"], 1):
            viewpoint = int(candidate)
            polyline, distance = _path(sim, positions[current], positions[viewpoint])
            path_cost[local, slot] = distance
            if polyline.size == 0:
                continue
            xz_path = polyline[:, [0, 2]]
            if human_center is not None:
                hdist = np.linalg.norm(xz_path[:, None, :] - human_center[None, None, :], axis=2).min()
                human_clearance[local, slot] = float(hdist)
                human_risk[local, slot] = float(max(0.0, 0.60 - hdist) / 0.60)
            if obstacle_points.size:
                ddist = np.linalg.norm(xz_path[:, None, :] - obstacle_points[None, :, :], axis=2).min()
                depth_clearance[local, slot] = float(ddist)
                depth_risk[local, slot] = float(max(0.0, 0.35 - ddist) / 0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=np.asarray([int(item["index"]) for item in rows], dtype=np.int64),
        path_cost=path_cost,
        human_risk=human_risk,
        depth_risk=depth_risk,
        human_clearance=human_clearance,
        depth_clearance=depth_clearance,
        rendered=rendered,
        obstacle_counts=obstacle_counts,
        human_point_counts=human_point_counts,
        nonhuman_counts=nonhuman_counts,
        nonhuman_navigable_counts=nonhuman_navigable_counts,
        occupancy_sample_counts=occupancy_sample_counts,
    )
    sim.close()
    return str(output)


def _build_depth_cache(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    scene_root: Path,
    workers: int,
    yolo_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    runtime = data_root / DEPTH_RUNTIME_REL
    path = runtime / f"{split}.npz"
    meta_path = runtime / f"{split}.json"
    signature = _signature(rows)
    if path.is_file() and meta_path.is_file():
        metadata = _read_json(meta_path)
        if metadata.get("signature") == signature and int(metadata.get("rows", -1)) == len(rows) and int(metadata.get("schema_version", 0)) >= 2:
            with np.load(path, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}, metadata
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["scene_id"]), []).append({"index": index, **{key: row[key] for key in ("archive_path", "record_id", "region", "scene_id", "current_viewpoint_id", "candidate_ids")}})
    shard_root = runtime / f"{split}_shards"
    jobs = [{"data_root": str(data_root), "scene_root": str(scene_root), "scene_id": scene, "rows": scene_rows, "yolo_path": str(yolo_path), "output": str(shard_root / f"{scene}.npz")} for scene, scene_rows in sorted(grouped.items())]
    shard_root.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not Path(str(job["output"])).is_file()]
    if workers > 1 and pending:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = [executor.submit(_occupancy_worker, job) for job in pending]
            for done, future in enumerate(futures, 1):
                future.result()
                print(f"depth occupancy {split}: {done}/{len(pending)} scenes", flush=True)
    else:
        for done, job in enumerate(pending, 1):
            _occupancy_worker(job)
            print(f"depth occupancy {split}: {done}/{len(pending)} scenes", flush=True)
    arrays = {
        "path_cost": np.full((len(rows), MAX_OPTIONS), np.inf, dtype=np.float32),
        "human_risk": np.zeros((len(rows), MAX_OPTIONS), dtype=np.float32),
        "depth_risk": np.zeros((len(rows), MAX_OPTIONS), dtype=np.float32),
        "human_clearance": np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32),
        "depth_clearance": np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32),
        "rendered": np.zeros(len(rows), dtype=bool),
        "obstacle_counts": np.zeros(len(rows), dtype=np.int32),
        "human_point_counts": np.zeros(len(rows), dtype=np.int32),
        "nonhuman_counts": np.zeros(len(rows), dtype=np.int32),
        "nonhuman_navigable_counts": np.zeros(len(rows), dtype=np.int32),
        "occupancy_sample_counts": np.zeros(len(rows), dtype=np.int32),
    }
    for scene in grouped:
        with np.load(shard_root / f"{scene}.npz", allow_pickle=False) as shard:
            for local, index in enumerate(np.asarray(shard["indices"], dtype=np.int64)):
                for key in arrays:
                    if key in shard.files:
                        arrays[key][int(index)] = shard[key][local]
                    elif key == "occupancy_sample_counts":
                        arrays[key][int(index)] = min(int(shard["obstacle_counts"][local]), 2048)
    runtime.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata = {
        "split": split,
        "schema_version": 2,
        "rows": len(rows),
        "signature": signature,
        "current_frame_only": True,
        "candidate_depth_used": False,
        "raw_depth_persisted": False,
        "point_cloud_persisted": False,
        "path_sampling_m": 0.10,
        "human_radius_m": 0.35,
        "human_safety_m": 0.60,
        "depth_safety_m": 0.35,
        "test_used": False,
    }
    _write_json(meta_path, metadata)
    return arrays, metadata


def _metadata_path_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Use Stage-A geodesic metadata for Train-only lambda holdout selection."""
    values = np.full((len(rows), MAX_OPTIONS), np.inf, dtype=np.float32)
    for index, row in enumerate(rows):
        values[index, 0] = 0.0
        geodesic = np.asarray(row.get("candidate_geodesic", []), dtype=np.float32)
        values[index, 1 : 1 + geodesic.size] = geodesic
    return {"path_cost": values}


def _candidate_norm(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, -np.inf, dtype=np.float64)
    for index in range(values.shape[0]):
        active = np.flatnonzero(mask[index])
        finite = active[np.isfinite(values[index, active])]
        if finite.size == 0:
            continue
        low, high = float(np.min(values[index, finite])), float(np.max(values[index, finite]))
        output[index, finite] = 0.0 if high - low <= 1e-12 else (values[index, finite] - low) / (high - low)
    return output


def _select(scores: np.ndarray, rows: Sequence[Mapping[str, Any]], mask: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        if active.size == 0:
            actions.append(int(row["current_viewpoint_id"]))
            continue
        slot = min(active.tolist(), key=lambda value: (-float(scores[index, value]), int(row["candidate_ids"][value - 1]), int(value)))
        actions.append(int(row["candidate_ids"][slot - 1]))
    return actions


def _terminal(options: Mapping[str, np.ndarray], actions: Sequence[int]) -> np.ndarray:
    predictions = np.zeros(len(actions), dtype=np.int64)
    for index, action in enumerate(actions):
        slots = np.flatnonzero((options["ids"][index] == int(action)) & options["mask"][index])
        if slots.size != 1:
            raise ValueError(f"action {action} absent at row {index}")
        predictions[index] = int(np.argmax(options["logp"][index, int(slots[0])] ))
    return predictions


def _metric(name: str, rows: Sequence[Mapping[str, Any]], labels: np.ndarray, pred: np.ndarray, actions: Sequence[int], depth: Mapping[str, np.ndarray], options: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result = classification(labels, pred)
    path = np.full(len(rows), np.nan, dtype=np.float64)
    human = np.full(len(rows), np.nan, dtype=np.float64)
    risk = np.zeros(len(rows), dtype=np.float64)
    current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
    for index, (row, action) in enumerate(zip(rows, actions)):
        slots = np.flatnonzero(options["ids"][index] == int(action))
        if slots.size:
            slot = int(slots[0])
            path[index] = float(depth["path_cost"][index, slot])
            human[index] = float(depth["human_clearance"][index, slot])
            risk[index] = float(0.5 * depth["human_risk"][index, slot] + 0.5 * depth["depth_risk"][index, slot])
    finite_human = np.isfinite(human)
    result.update({
        "method": name,
        "contexts": int(labels.size),
        "move_rate": float(np.mean(np.asarray(actions) != current)),
        "stay_rate": float(np.mean(np.asarray(actions) == current)),
        "movement": {
            "mean_geodesic_m": float(np.mean(path[np.isfinite(path)])) if np.isfinite(path).any() else None,
            "median_geodesic_m": float(np.median(path[np.isfinite(path)])) if np.isfinite(path).any() else None,
            "p90_geodesic_m": float(np.percentile(path[np.isfinite(path)], 90)) if np.isfinite(path).any() else None,
            "mean_human_clearance_m": float(np.nanmean(human[finite_human])) if finite_human.any() else None,
            "minimum_human_clearance_m": float(np.nanmin(human[finite_human])) if finite_human.any() else None,
            "depth_risk_path_rate": float(np.mean(risk > 0.0)),
        },
        "test_used": False,
    })
    return result


def _switch(old: Sequence[int], new: Sequence[int], labels: np.ndarray, old_pred: np.ndarray, new_pred: np.ndarray) -> dict[str, Any]:
    old_array, new_array = np.asarray(old), np.asarray(new)
    changed = old_array != new_array
    return {
        "switch_count": int(changed.sum()),
        "switch_rate": float(changed.mean()),
        "old_correct_new_wrong": int(np.sum((old_pred == labels) & (new_pred != labels) & changed)),
        "old_wrong_new_correct": int(np.sum((old_pred != labels) & (new_pred == labels) & changed)),
    }


def _holdout_lambda(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], rgb: np.ndarray, path: Mapping[str, np.ndarray]) -> tuple[float, dict[str, Any]]:
    records = sorted({str(row["record_id"]) for row in rows})
    holdout_records = set(records[-max(1, int(round(0.1 * len(records)))) :])
    holdout = np.asarray([str(row["record_id"]) in holdout_records for row in rows], dtype=bool)
    candidates = _candidate_mask(options["mask"])
    q = _candidate_norm(rgb, candidates)
    d = _candidate_norm(path["path_cost"], candidates)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    best: tuple[float, float] | None = None
    scores: dict[str, Any] = {}
    for value in (0.10, 0.25, 0.50):
        selected = _select(_combine(q, d, None, value), rows, candidates)
        pred = _terminal(options, selected)
        accuracy = float(np.mean(pred[holdout] == labels[holdout])) if holdout.any() else 0.0
        holdout_path = path["path_cost"][holdout]
        finite_path = holdout_path[np.isfinite(holdout_path)]
        distance = float(np.mean(finite_path)) if finite_path.size else 0.0
        scores[str(value)] = {"holdout_accuracy": accuracy, "selected_mean_path_m": distance}
        key = (accuracy, -distance)
        if best is None or key > (best[1], -scores[str(best[0])]["selected_mean_path_m"]):
            best = (value, accuracy)
    if best is None:
        raise RuntimeError("unable to select lambda")
    return float(best[0]), {"selection_source": "Policy Train record holdout", "holdout_records": len(holdout_records), "holdout_contexts": int(holdout.sum()), "criterion": "highest holdout accuracy, tie lower path", "candidates": scores, "selected_lambda": float(best[0]), "test_used": False}


def _combine(q: np.ndarray, distance: np.ndarray, risk: np.ndarray | None, lam: float, mu: float = 0.0) -> np.ndarray:
    """Combine normalized candidate scores without invalid inf-inf arithmetic."""
    output = np.full(q.shape, -np.inf, dtype=np.float64)
    valid = np.isfinite(q)
    if lam > 0.0:
        valid &= np.isfinite(distance)
    if risk is not None and mu > 0.0:
        valid &= np.isfinite(risk)
    output[valid] = q[valid]
    if lam > 0.0:
        output[valid] -= lam * distance[valid]
    if risk is not None and mu > 0.0:
        output[valid] -= mu * risk[valid]
    return output


def _high_occlusion(rows: Sequence[Mapping[str, Any]], visibility: np.ndarray, methods: Mapping[str, dict[str, Any]], predictions: Mapping[str, np.ndarray], actions: Mapping[str, Sequence[int]], options: Mapping[str, np.ndarray], depth: Mapping[str, np.ndarray]) -> dict[str, Any]:
    threshold = float(np.quantile(visibility[:, 0], 1.0 / 3.0))
    selected = visibility[:, 0] < threshold
    output: dict[str, Any] = {"definition": "strict lower tertile of current-slot 17-joint Frame0 visibility", "count": int(selected.sum()), "threshold": threshold, "methods": {}}
    for name in methods:
        if name not in predictions:
            continue
        idx = np.flatnonzero(selected)
        subset_rows = [rows[int(i)] for i in idx]
        output["methods"][name] = _metric(name, subset_rows, np.asarray([int(r["label_id"]) for r in subset_rows]), predictions[name][selected], [actions[name][int(i)] for i in idx], {key: value[selected] for key, value in depth.items()}, {key: value[selected] for key, value in options.items()})
    return output


def _analysis(result: Mapping[str, Any]) -> str:
    m = result["main_metrics"]
    rgb = m["RGBGlobal-Visibility"]
    map_m = m["RGBGlobal-Visibility + MapPath"]
    depth_m = m["RGBGlobal-Visibility + MapPath + CurrentDepth"]
    rgb_acc, map_acc = float(rgb["accuracy"]), float(map_m["accuracy"])
    path_reduction = 100.0 * (float(rgb["movement"]["mean_geodesic_m"]) - float(map_m["movement"]["mean_geodesic_m"])) / max(float(rgb["movement"]["mean_geodesic_m"]), 1e-9)
    acc_drop = 100.0 * (map_acc - rgb_acc)
    switch = result["selection_disagreement"]["map_vs_depth"]
    clearance_rgb = depth_m["movement"].get("mean_human_clearance_m")
    clearance_map = map_m["movement"].get("mean_human_clearance_m")
    clearance_delta = None if clearance_rgb is None or clearance_map is None else 100.0 * (float(clearance_rgb) - float(clearance_map)) / max(abs(float(clearance_map)), 1e-9)
    risk_map = float(map_m["movement"]["depth_risk_path_rate"])
    risk_depth = float(depth_m["movement"]["depth_risk_path_rate"])
    depth_switch = float(switch["switch_rate"])
    path_decision = "KEEP MOVEMENT-AWARE NBV" if path_reduction >= 20.0 and acc_drop >= -0.5 else ("PATH-COST NBV NOT USEFUL" if path_reduction < 10.0 or acc_drop < -1.5 else "MIXED MOVEMENT-COST EVIDENCE")
    depth_decision = "KEEP CURRENT DEPTH FOR LOCAL MOVEMENT" if ((risk_map - risk_depth) >= 0.20 and acc_drop >= -1.0) or (clearance_delta is not None and clearance_delta >= 15.0) else ("CURRENT DEPTH IS REDUNDANT WITH KNOWN MAP" if depth_switch < 0.02 and (clearance_delta is None or clearance_delta < 5.0) and (risk_map - risk_depth) < 0.05 else "MIXED DEPTH EVIDENCE")
    lines = [
        "# Known-Map + Current-Depth Movement-Aware NBV",
        "",
        "Train record-holdout lambda selection and Moving Val evaluation only (10,080 contexts). Policy Test was not read.",
        "Current depth is rendered only at Frame 0/current viewpoint. Raw depth and point clouds are transient; compact path/risk summaries are cached outside Git.",
        "",
        "| Method | Acc | Macro-F1 | Mean path (m) | Median path (m) | Human clearance (m) | Depth-risk rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("StaticPrior", "RGBGlobal-Visibility", "RGBGlobal-Visibility + MapPath", "RGBGlobal-Visibility + MapPath + CurrentDepth", "GT SceneVisibility", "GT SceneVisibility + MapPath", "GT SceneVisibility + MapPath + Depth", "GT-TrueLogP Oracle"):
        if name not in m:
            continue
        item = m[name]
        movement = item.get("movement", {})
        lines.append(f"| {name} | {float(item['accuracy']):.6f} | {float(item['macro_f1']):.6f} | {movement.get('mean_geodesic_m', float('nan')):.4f} | {movement.get('median_geodesic_m', float('nan')):.4f} | {movement.get('mean_human_clearance_m', float('nan')):.4f} | {movement.get('depth_risk_path_rate', float('nan')):.4f} |")
    lines += [
        "",
        "## Decision summary",
        f"- Stage-A RGB→MapPath selected λ={result['lambda_selection']['rgb']['selected_lambda']:.2f}; mean path change={path_reduction:+.2f}% and Accuracy change={acc_drop:+.3f}pp. **{path_decision}**.",
        f"- MapPath→MapPath+Depth switched {100*depth_switch:.2f}% of selections; human-clearance relative change={clearance_delta if clearance_delta is not None else float('nan'):+.2f}%; depth-risk rate changed from {risk_map:.4f} to {risk_depth:.4f}. **{depth_decision}**.",
        f"- Depth occupancy: {result['depth_occupancy']['nonhuman_navigable_fraction']:.4f} of non-human obstacle points were in navmesh/free-space proxy cells; {result['depth_occupancy']['human_attributable_fraction']:.4f} of obstacle points were attributable to the YOLO-bbox human footprint.",
        f"- High-occlusion subset contains {result['high_occlusion']['count']} contexts; details are in high_occlusion_metrics.json.",
        "",
        "## Required scientific answers",
        "1. Known-map path cost is useful only if it meets the preregistered distance/accuracy gate; the exact deltas are reported above.",
        "2. Current depth changes decisions only through the compact transient occupancy and dynamic human footprint; disagreement and risk deltas are reported above.",
        "3. Depth provides no candidate RGB/depth or future perception. A high human-attributable fraction means the static HM3D benchmark contains little non-human transient geometry.",
        "4. High-occlusion movement results are kept separate and are not used to retune thresholds.",
        "5. This remains an oracle/diagnostic audit for known Habitat geometry; no new model, map, skeleton, RGB or DINO was trained/generated.",
        "",
        "```text",
        "policy_test_used=false",
        "training_used=false (Train is used only for prior and lambda holdout selection)",
        "habitat_gt_scene_geometry_used=true",
        "current_frame_depth_used_for_local_occupancy=true",
        "future_candidate_rgb_used=false",
        "future_candidate_skeleton_used_only_for_terminal_evaluation=true",
        "deployable_rgb_selector=true; depth_context_is_current_only=true",
        "```",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    scene_root = args.habitat_root.resolve()
    started = time.monotonic()
    train_rows, val_rows = _load_rows(data_root)
    labels_train = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    labels_val = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    head_path = data_root / HEAD_REL
    vis_path = data_root / VIS_CKPT_REL
    if not head_path.is_file() or not vis_path.is_file():
        raise FileNotFoundError(f"missing frozen recognizer/predictor: {head_path} {vis_path}")
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    train_options, train_meta = _load_yaw8_options(data_root, train_rows, "train", head, device)
    val_options, val_meta = _load_yaw8_options(data_root, val_rows, "moving_val", head, device)
    train_map, train_map_meta = _load_map_cache(data_root, "train", train_rows)
    val_map, val_map_meta = _load_map_cache(data_root, "val", val_rows)
    if not np.array_equal(train_map["ids"], train_options["ids"]) or not np.array_equal(val_map["ids"], val_options["ids"]):
        raise ValueError("map and recognizer viewpoint ids differ")
    candidate_train, candidate_val = _candidate_mask(train_options["mask"]), _candidate_mask(val_options["mask"])
    stats = _read_json(data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/stage_c_feature_stats.json")
    train_geometry, train_ids, train_geom_mask = _option_geometry(train_rows, stats)
    val_geometry, val_ids, val_geom_mask = _option_geometry(val_rows, stats)
    if not np.array_equal(train_ids, train_options["ids"]) or not np.array_equal(val_ids, val_options["ids"]):
        raise ValueError("geometry and recognizer viewpoint ids differ")
    dino_root = data_root / "features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/frame0_current"
    train_dino = np.load(dino_root / "train.npy", mmap_mode="r")
    val_dino = np.load(dino_root / "val.npy", mmap_mode="r")
    vis_model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    vis_model.load_state_dict(torch.load(vis_path, map_location=device, weights_only=False)["state_dict"])
    vis_model.eval()
    train_rgb = np.asarray(_predict(vis_model, train_geometry, train_dino, device), dtype=np.float64)
    val_rgb = np.asarray(_predict(vis_model, val_geometry, val_dino, device), dtype=np.float64)
    depth_runtime = data_root / "diagnostics/rgbd_human_state_recovery_v1/yolo"
    yolo_val = depth_runtime / "moving_val.npz"
    if not yolo_val.is_file():
        raise FileNotFoundError("current-frame YOLO cache is required; no new YOLO run is allowed")
    # Lambda is selected on a deterministic Policy-Train record holdout.  The
    # holdout only needs the existing Stage-A geodesic metadata; current depth
    # is deliberately rendered for Moving Val only to keep this audit a
    # compact current-observation diagnostic.
    train_depth = _metadata_path_cost(train_rows)
    train_depth_meta = {"split": "train", "rows": len(train_rows), "source": "Stage-A candidate_geodesic metadata", "current_frame_rendered": False, "test_used": False}
    val_depth, val_depth_meta = _build_depth_cache(data_root, val_rows, "moving_val", scene_root, args.workers, yolo_val)
    prior, prior_meta = _train_prior(train_rows, train_options)
    train_prior = np.full(train_options["ids"].shape, -np.inf, dtype=np.float64)
    val_prior = np.full(val_options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(train_rows):
        for slot in np.flatnonzero(candidate_train[index]):
            train_prior[index, slot] = float(prior.get(int(train_options["ids"][index, slot]), prior.get(-1, 0.0)))
    for index, row in enumerate(val_rows):
        for slot in np.flatnonzero(candidate_val[index]):
            val_prior[index, slot] = float(prior.get(int(val_options["ids"][index, slot]), prior.get(-1, 0.0)))
    train_gt_vis = np.mean(train_map["features"][:, :, 0:17], axis=2)
    val_gt_vis = np.mean(val_map["features"][:, :, 0:17], axis=2)
    rgb_lambda, rgb_lambda_meta = _holdout_lambda(train_rows, train_options, train_rgb, train_depth)
    gt_lambda, gt_lambda_meta = _holdout_lambda(train_rows, train_options, train_gt_vis, train_depth)
    actions: dict[str, list[int]] = {}
    predictions: dict[str, np.ndarray] = {}
    methods: dict[str, dict[str, Any]] = {}
    candidate_mask = candidate_val
    def add(name: str, scores: np.ndarray) -> None:
        action = _select(scores, val_rows, candidate_mask)
        actions[name] = action
        predictions[name] = _terminal(val_options, action)
        methods[name] = _metric(name, val_rows, labels_val, predictions[name], action, val_depth, val_options)
    add("StaticPrior", val_prior)
    add("RGBGlobal-Visibility", val_rgb)
    q_rgb, d_val = _candidate_norm(val_rgb, candidate_mask), _candidate_norm(val_depth["path_cost"], candidate_mask)
    risk_val = 0.5 * val_depth["human_risk"] + 0.5 * val_depth["depth_risk"]
    add("RGBGlobal-Visibility + MapPath", _combine(q_rgb, d_val, None, rgb_lambda))
    add("RGBGlobal-Visibility + MapPath + CurrentDepth", _combine(q_rgb, d_val, risk_val, rgb_lambda, 0.25))
    q_gt = _candidate_norm(val_gt_vis, candidate_mask)
    add("GT SceneVisibility", val_gt_vis)
    add("GT SceneVisibility + MapPath", _combine(q_gt, d_val, None, gt_lambda))
    add("GT SceneVisibility + MapPath + Depth", _combine(q_gt, d_val, risk_val, gt_lambda, 0.25))
    true_values = np.full(val_options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(val_rows):
        for slot in np.flatnonzero(candidate_mask[index]):
            true_values[index, slot] = float(val_options["logp"][index, slot, int(row["label_id"])])
    add("GT-TrueLogP Oracle", true_values)
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice([int(v) for v in row["candidate_ids"]])) if row["candidate_ids"] else int(row["current_viewpoint_id"]) for row in val_rows]
    actions["Random"] = random_actions
    predictions["Random"] = _terminal(val_options, random_actions)
    methods["Random"] = _metric("Random", val_rows, labels_val, predictions["Random"], random_actions, val_depth, val_options)
    selection_disagreement = {
        "rgb_vs_map": _switch(actions["RGBGlobal-Visibility"], actions["RGBGlobal-Visibility + MapPath"], labels_val, predictions["RGBGlobal-Visibility"], predictions["RGBGlobal-Visibility + MapPath"]),
        "map_vs_depth": _switch(actions["RGBGlobal-Visibility + MapPath"], actions["RGBGlobal-Visibility + MapPath + CurrentDepth"], labels_val, predictions["RGBGlobal-Visibility + MapPath"], predictions["RGBGlobal-Visibility + MapPath + CurrentDepth"]),
    }
    depth_occupancy = {
        "rendered_rate": float(np.mean(val_depth["rendered"])),
        "obstacle_point_count": int(np.sum(val_depth["obstacle_counts"])),
        "human_bbox_point_count": int(np.sum(val_depth["human_point_counts"])),
        "nonhuman_point_count": int(np.sum(val_depth["nonhuman_counts"])),
        "human_attributable_fraction": float(np.sum(val_depth["human_point_counts"]) / max(np.sum(val_depth["human_point_counts"]) + np.sum(val_depth["nonhuman_counts"]), 1)),
        "nonhuman_navigable_fraction": float(np.sum(val_depth["nonhuman_navigable_counts"]) / max(np.sum(val_depth["occupancy_sample_counts"]), 1)),
        "definition": "navmesh is_navigable(point) proxy on a deterministic 2,048-point sample per context",
        "test_used": False,
    }
    high = _high_occlusion(val_rows, val_map["features"][:, :, 0:17].mean(axis=2), methods, predictions, actions, val_options, val_depth)
    train_holdout = {"rgb": rgb_lambda_meta, "gt": gt_lambda_meta}
    tradeoff: dict[str, Any] = {}
    for lam in (0.0, 0.10, 0.25, 0.50):
        score = _combine(_candidate_norm(val_rgb, candidate_mask), d_val, None, lam)
        action = _select(score, val_rows, candidate_mask)
        pred = _terminal(val_options, action)
        metric = _metric(f"RGB lambda={lam:.2f}", val_rows, labels_val, pred, action, val_depth, val_options)
        tradeoff[str(lam)] = {"accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], **metric["movement"]}
    candidate_target = np.full(val_options["ids"].shape, np.nan, dtype=np.float64)
    for index, row in enumerate(val_rows):
        for slot in np.flatnonzero(candidate_mask[index]):
            candidate_target[index, slot] = float(val_options["logp"][index, slot, int(row["label_id"])])
    correlations = {}
    for name, value in {"RGBGlobal-Visibility": val_rgb, "MapPathCost": -val_depth["path_cost"], "DepthRisk": -risk_val, "GTSceneVisibility": val_gt_vis}.items():
        valid = candidate_mask & np.isfinite(value) & np.isfinite(candidate_target)
        correlations[name] = {"candidate_count": int(valid.sum()), "candidate_level_spearman": correlation(value[valid], candidate_target[valid], spearman=True) if valid.sum() > 1 else 0.0}
    protocol = {
        "policy_train_contexts": len(train_rows),
        "moving_val_contexts": len(val_rows),
        "moving_val_candidate_samples": int(candidate_val.sum()),
        "candidate_count_mean": float(candidate_val.sum(axis=1).mean()),
        "candidate_count_min": int(candidate_val.sum(axis=1).min()),
        "candidate_count_max": int(candidate_val.sum(axis=1).max()),
        "action_set": "Stage-A legal candidate-only; current/stay excluded for selectors",
        "recognizer_checkpoint": str((data_root / HEAD_REL).resolve()),
        "yaw8_cache_train": train_meta,
        "yaw8_cache_moving_val": val_meta,
        "map_cache_train": train_map_meta,
        "map_cache_moving_val": val_map_meta,
        "depth_cache_train": train_depth_meta,
        "depth_cache_moving_val": val_depth_meta,
        "lambda_selection": "Policy Train record holdout only; Moving Val not used",
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_MOVEMENT_AWARE_DEPTH_NBV",
        "status": "COMPLETED",
        "labels": list(LABELS),
        "population": {"policy_train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "moving_val_candidate_samples": int(candidate_val.sum())},
        "main_metrics": methods,
        "lambda_selection": {"rgb": rgb_lambda_meta, "gt": gt_lambda_meta},
        "selection_disagreement": selection_disagreement,
        "depth_occupancy": depth_occupancy,
        "correlations": correlations,
        "tradeoff": tradeoff,
        "high_occlusion": high,
        "protocol": protocol,
        "flags": {"policy_test_used": False, "training_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "candidate_depth_used": False, "current_frame_depth_rendered": True, "habitat_gt_scene_geometry_used": True, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "deployable_rgb_selector": True, "depth_context_deployable_current_only": True},
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "workers": args.workers, "seed": SEED, "elapsed_seconds": time.monotonic() - started},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "config.json", {"experiment_id": result["experiment_id"], "workers": args.workers, "device": str(device), "mu": 0.25, "lambdas": [0.0, 0.10, 0.25, 0.50], "path_sampling_m": 0.10, "test_used": False})
    _write_json(output_root / "path_cost_metrics.json", {"lambda_selection": result["lambda_selection"], "selection_disagreement": selection_disagreement, "methods": {name: methods[name].get("movement", {}) for name in methods}})
    _write_json(output_root / "depth_occupancy_metrics.json", depth_occupancy)
    _write_json(output_root / "main_metrics.json", methods)
    _write_json(output_root / "tradeoff_metrics.json", tradeoff)
    _write_json(output_root / "high_occlusion_metrics.json", high)
    _write_json(output_root / "result.json", result)
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--habitat-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
