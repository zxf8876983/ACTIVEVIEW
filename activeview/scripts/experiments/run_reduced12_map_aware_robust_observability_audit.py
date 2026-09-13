#!/usr/bin/env python3
"""Train/Val-only Known-3D-Map Robust Observability NBV audit.

The audit tests deterministic map-derived visibility, motion-envelope
robustness, projection and navigation features.  Exact frame-0 human pose and
static Habitat geometry are privileged diagnostics; future action/pose,
candidate observations and Policy Test are never used as selector inputs.
Large map-feature caches and the small utility checkpoint are written below
``ACTIVEVIEW_DATA_ROOT`` and are intentionally kept outside Git.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (
    ARCHIVE_REL,
    H36M17_LINKS,
    JOINT_COUNT,
    LABELS,
    _archive_path,
    _load_npz,
    _load_raw_records,
    _load_resampled_motion,
    _placement_map,
    _ray_visible,
    _scene_sim,
    _world_h36m17,
)
from activeview.scripts.eval.analyze_reduced12_human_view_observability_oracle import _project_joints
from activeview.scripts.eval.reduced12_nbv_utils import classification, correlation
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import _load_options
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import _option_geometry
from activeview.scripts.experiments.map_aware_historical import historical_methods


NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
MAX_OPTIONS = 22
SEED = 42
EPOCHS = 12
OBS_PER_RECORD = 16
TRAIN_BATCH = 512
EVAL_BATCH = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
LISTWISE_WEIGHT = 0.5
LISTWISE_TAU = 0.5
HFOV_DEG = 75.0
IMAGE_SIZE = 256.0
SENSOR_HEIGHT_M = 1.10
DENSE_BONE_SAMPLES = 12
JOINT_OFFSET_RADIUS = 0.04
CLEARANCE_JOINTS = (0, 8, 10, 12, 13, 14, 15, 16)
CLEARANCE_OFFSET = 0.03
ROBUST_EPSILONS = (0.05, 0.10, 0.20)
ROBUST_DIRECTIONS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
DATASET_NAME = "policy_reduced12_eight_placement_v1"
STRUCTURED_REL = Path("diagnostics/structured_observability_final_audit")
SCALAR_REL = Path("diagnostics/frame0_visibility_predictor_v1")
RUNTIME_REL = Path("diagnostics/map_aware_robust_observability_audit")
CHECKPOINT_REL = Path("checkpoints/policy_reduced12_eight_placement_v1/map_aware_robust_observability_audit/mapfeature_utility.pth")
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/map_aware_robust_observability_audit"
HISTORICAL_STRUCTURED = REPO_ROOT / "experiments/reduced12_eight_placement_v1/structured_observability_final_audit/result.json"
HISTORICAL_SHARED = REPO_ROOT / "experiments/reduced12_eight_placement_v1/historical_route1_shared_head_synergy/result.json"

EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)
EDGE_PARTS = (4, 4, 5, 4, 4, 5, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3)
PART_NAMES = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")
JOINT_PART = (1, 4, 5, 5, 4, 4, 5, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3)

# map vector layout; the first 17 entries intentionally match frame-0 joint visibility.
JOINT_SLICE = slice(0, 17)
DENSE_INDEX = 17
PART_SLICE = slice(18, 24)
ROBUST_INDEX = {0.05: 24, 0.10: 25, 0.20: 26}
ROBUST_MEAN_INDEX = 27
CLEARANCE_SLICE = slice(28, 31)
PROJECTION_SLICE = slice(31, 36)  # area, height, width, in-FOV fraction, boundary margin
DISTANCE_SLICE = slice(36, 40)  # euclidean, horizontal, height difference, elevation
SCENE_CLEARANCE_INDEX = 40
CANDIDATE_FREE_SPACE_INDEX = 41
NAV_SLICE = slice(42, 45)  # geodesic, geodesic/euclidean, camera yaw delta
POSE_SLICE = slice(45, 96)  # root-relative current frame-0 H36M17 pose
MAP_DIM = 96


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _yaw_from_quaternion(q: Sequence[float]) -> float:
    values = np.asarray(q, dtype=np.float64)
    values = values / max(float(np.linalg.norm(values)), 1e-12)
    w, x, y, z = values
    return float(math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _dense_points(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    parts: list[int] = []
    for (left, right), part in zip(EDGES, EDGE_PARTS):
        for value in np.linspace(joints[left], joints[right], DENSE_BONE_SAMPLES):
            points.append(np.asarray(value, dtype=np.float32))
            parts.append(part)
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    for joint, part in enumerate(JOINT_PART):
        points.append(joints[joint])
        parts.append(part)
        for direction in offsets:
            points.append(joints[joint] + JOINT_OFFSET_RADIUS * np.asarray(direction, dtype=np.float32))
            parts.append(part)
    return np.asarray(points, dtype=np.float32), np.asarray(parts, dtype=np.int64)


def _projection_metrics(world_joints: np.ndarray, camera_position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    points, front, in_fov, _ = _project_joints(world_joints[None], camera_position, rotation)
    xy = points[0]
    valid = in_fov[0] & np.isfinite(xy).all(axis=-1)
    front_fraction = float(np.mean(front[0]))
    fov_fraction = float(np.mean(in_fov[0]))
    if int(valid.sum()) < 2:
        return np.zeros(5, dtype=np.float32)
    selected = xy[valid]
    width = float(np.max(selected[:, 0]) - np.min(selected[:, 0])) / IMAGE_SIZE
    height = float(np.max(selected[:, 1]) - np.min(selected[:, 1])) / IMAGE_SIZE
    area = float(np.clip(width * height, 0.0, 1.0)) * fov_fraction * front_fraction
    margin = float(np.min(np.stack((selected[:, 0], IMAGE_SIZE - selected[:, 0], selected[:, 1], IMAGE_SIZE - selected[:, 1]))) / IMAGE_SIZE)
    return np.asarray([area, height, width, fov_fraction, max(0.0, margin)], dtype=np.float32)


def _cone_clearance(sim: Any, camera: np.ndarray, target: np.ndarray) -> np.ndarray:
    vector = target - camera
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros(3, dtype=np.float32)
    direction = vector / norm
    basis = np.cross(direction, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    if float(np.linalg.norm(basis)) <= 1e-6:
        basis = np.cross(direction, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    basis = basis / max(float(np.linalg.norm(basis)), 1e-6)
    second = np.cross(direction, basis)
    neighbours = (target + CLEARANCE_OFFSET * basis, target - CLEARANCE_OFFSET * basis, target + CLEARANCE_OFFSET * second, target - CLEARANCE_OFFSET * second)
    values = [float(_ray_visible(sim, camera, endpoint)) for endpoint in neighbours]
    return np.asarray([np.mean(values), np.percentile(values, 10), np.min(values)], dtype=np.float32)

def _candidate_feature(sim: Any, world_joints: np.ndarray, dense: np.ndarray, dense_parts: np.ndarray, camera_position: np.ndarray, rotation: np.ndarray, geodesic: float, current_yaw: float, pathfinder: Any, root_clearance: float) -> np.ndarray:
    camera = np.asarray(camera_position, dtype=np.float32) + np.asarray([0.0, SENSOR_HEIGHT_M, 0.0], dtype=np.float32)
    joints = np.asarray(world_joints[0], dtype=np.float32)
    joint_visibility = np.asarray([float(_ray_visible(sim, camera, point)) for point in joints], dtype=np.float32)
    dense_visibility = np.asarray([float(_ray_visible(sim, camera, point)) for point in dense], dtype=np.float32)
    part_visibility = np.asarray([np.mean(dense_visibility[dense_parts == part]) for part in range(6)], dtype=np.float32)
    robust_values: list[float] = []
    for epsilon in ROBUST_EPSILONS:
        probes = [joints + epsilon * np.asarray(direction, dtype=np.float32) for direction in ROBUST_DIRECTIONS]
        robust_values.append(float(np.mean([_ray_visible(sim, camera, point) for point in np.concatenate(probes)])))
    robust = np.asarray(robust_values + [float(np.mean(robust_values))], dtype=np.float32)
    clearance = np.mean([_cone_clearance(sim, camera, joints[index]) for index in CLEARANCE_JOINTS], axis=0).astype(np.float32)
    projection = _projection_metrics(joints, camera_position, rotation)
    delta = joints[0] - camera_position
    euclidean = float(np.linalg.norm(delta))
    horizontal = float(np.linalg.norm(delta[[0, 2]]))
    elevation = float(math.atan2(abs(float(delta[1])), max(horizontal, 1e-6)))
    candidate_clearance = float(pathfinder.distance_to_closest_obstacle(np.asarray(camera_position, dtype=np.float32)))
    camera_yaw = _yaw_from_quaternion(rotation)
    yaw_delta = abs((camera_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi)
    pose = (joints - joints[0]).reshape(-1).astype(np.float32)
    output = np.zeros(MAP_DIM, dtype=np.float32)
    output[JOINT_SLICE] = joint_visibility
    output[DENSE_INDEX] = float(np.mean(dense_visibility))
    output[PART_SLICE] = part_visibility
    output[24:28] = robust
    output[CLEARANCE_SLICE] = clearance
    output[PROJECTION_SLICE] = projection
    output[DISTANCE_SLICE] = np.asarray([euclidean, horizontal, abs(float(delta[1])), elevation], dtype=np.float32)
    output[SCENE_CLEARANCE_INDEX] = root_clearance
    output[CANDIDATE_FREE_SPACE_INDEX] = candidate_clearance
    output[NAV_SLICE] = np.asarray([float(geodesic), float(geodesic) / max(euclidean, 1e-6), yaw_delta], dtype=np.float32)
    output[POSE_SLICE] = pose
    return output

def _extract_scene(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract one scene in an isolated process; outputs are resumable NPZ files."""
    scene_id = str(payload["scene_id"])
    rows = payload["rows"]
    indices = np.asarray(payload["indices"], dtype=np.int64)
    data_root = Path(str(payload["data_root"]))
    scene_root = Path(str(payload["scene_root"]))
    output_path = Path(str(payload["output_path"]))
    raw_records = _load_raw_records(data_root)
    # Habitat's cast_ray requires the Bullet physics backend even though only
    # static scene geometry is queried; no humanoid is added to this simulator.
    sim = _scene_sim(scene_root, scene_id, physics=True)
    human_sim = _scene_sim(scene_root, scene_id, physics=True)
    human = human_sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    from activeview.data.motion.motion_converter import MotionConverter
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    placements = _placement_map(data_root, scene_id)
    converted: dict[str, Mapping[str, Any]] = {}
    dense_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    max_candidates = max(len(row["candidate_ids"]) + 1 for row in rows)
    features = np.full((len(rows), max_candidates, MAP_DIM), np.nan, dtype=np.float32)
    ids = np.full((len(rows), max_candidates), -1, dtype=np.int64)
    mask = np.zeros((len(rows), max_candidates), dtype=bool)
    started = time.perf_counter()
    try:
        for local, row in enumerate(rows):
            record_id, region = str(row["record_id"]), str(row["region"])
            if record_id not in raw_records or region not in placements:
                raise ValueError(f"missing raw record/placement: {row['episode_id']}")
            if record_id not in converted:
                converted[record_id] = converter.convert(_load_resampled_motion(raw_records[record_id], 30))
            key = (record_id, region)
            if key not in dense_cache:
                world = _world_h36m17(human, converted[record_id], placements[region], float(placements[region]["yaw_deg"]))
                dense_cache[key] = (world, *_dense_points(world[0]))
            world, dense, dense_parts = dense_cache[key]
            archive = _load_npz(_archive_path(data_root, row))
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
            rotations = np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32)
            index_by_id = {int(value): offset for offset, value in enumerate(viewpoint_ids.tolist())}
            actions = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
            geodesics = [0.0] + [float(value) for value in row["candidate_geodesic"]]
            current_yaw = _yaw_from_quaternion(rotations[index_by_id[actions[0]]])
            root_clearance = float(sim.pathfinder.distance_to_closest_obstacle(np.asarray(placements[region]["position"], dtype=np.float32)))
            for slot, (action, geodesic) in enumerate(zip(actions, geodesics)):
                view = index_by_id[action]
                features[local, slot] = _candidate_feature(sim, world, dense, dense_parts, positions[view], rotations[view], geodesic, current_yaw, sim.pathfinder, root_clearance)
                ids[local, slot] = action
                mask[local, slot] = True
            if (local + 1) % 50 == 0:
                print(f"[map-{payload['split']}-{scene_id}] {local + 1}/{len(rows)} ({time.perf_counter() - started:.1f}s)", flush=True)
    finally:
        human_sim.close()
        sim.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, indices=indices, features=features, ids=ids, mask=mask)
    return {"scene_id": scene_id, "rows": len(rows), "path": str(output_path), "elapsed_seconds": time.perf_counter() - started}


def _load_runtime_cache(path: Path, meta_path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray] | None:
    if not path.is_file() or not meta_path.is_file():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != row_signature(rows) or int(metadata.get("rows", -1)) != len(rows) or bool(metadata.get("test_used", True)):
        return None
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _extract_split(data_root: Path, scene_root: Path, rows: Sequence[Mapping[str, Any]], split: str, workers: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    runtime = data_root / RUNTIME_REL
    runtime.mkdir(parents=True, exist_ok=True)
    cache_path, meta_path = runtime / f"{split}.npz", runtime / f"{split}.json"
    cached = _load_runtime_cache(cache_path, meta_path, rows)
    if cached is not None:
        return cached, json.loads(meta_path.read_text(encoding="utf-8"))
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["scene_id"])].append((index, row))
    scene_dir = runtime / "scenes" / split
    payloads: list[dict[str, Any]] = []
    max_candidates = max(len(row["candidate_ids"]) + 1 for row in rows)
    for scene_id, entries in sorted(groups.items()):
        scene_path = scene_dir / f"{scene_id}.npz"
        if scene_path.is_file():
            try:
                with np.load(scene_path, allow_pickle=False) as archive:
                    cached_indices = np.asarray(archive["indices"], dtype=np.int64)
                    cached_features = np.asarray(archive["features"])
                expected_indices = np.asarray([index for index, _ in entries], dtype=np.int64)
                if cached_features.shape[0] == len(entries) and np.array_equal(cached_indices, expected_indices):
                    continue
            except (OSError, ValueError, KeyError):
                pass
        payloads.append({"scene_id": scene_id, "rows": [dict(row) for _, row in entries], "indices": [index for index, _ in entries], "data_root": str(data_root), "scene_root": str(scene_root), "split": split, "output_path": str(scene_path)})
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    if payloads:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = [pool.submit(_extract_scene, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())
                print(json.dumps(completed[-1]), flush=True)
    features = np.full((len(rows), max_candidates, MAP_DIM), np.nan, dtype=np.float32)
    ids = np.full((len(rows), max_candidates), -1, dtype=np.int64)
    mask = np.zeros((len(rows), max_candidates), dtype=bool)
    for scene_id, entries in sorted(groups.items()):
        path = scene_dir / f"{scene_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            local_features, local_ids, local_mask, local_indices = (np.asarray(archive[key]) for key in ("features", "ids", "mask", "indices"))
        if local_features.shape[0] != len(entries) or local_features.shape[-1] != MAP_DIM:
            raise ValueError(f"invalid scene cache shape: {path}")
        for local, (global_index, _) in enumerate(entries):
            features[global_index, : local_features.shape[1]] = local_features[local]
            ids[global_index, : local_ids.shape[1]] = local_ids[local]
            mask[global_index, : local_mask.shape[1]] = local_mask[local]
            if int(local_indices[local]) != global_index:
                raise ValueError(f"scene cache index mismatch: {path}")
    if not np.isfinite(features[mask]).all():
        raise ValueError(f"non-finite map features for {split}")
    metadata = {"split": split, "rows": len(rows), "signature": row_signature(rows), "map_dim": MAP_DIM, "candidate_count": int(mask[:, 1:].sum()), "scene_count": len(groups), "workers": workers, "dense_sample_count": int(16 * DENSE_BONE_SAMPLES + 17 * 7), "dense_points_per_context": int(16 * DENSE_BONE_SAMPLES + 17 * 7), "robust_definition": "17 current H36M17 joints x +/-x,+/-y,+/-z at eps 0.05/0.10/0.20m", "clearance_definition": "four endpoint cone rays around eight representative current joints", "test_used": False, "elapsed_seconds": time.perf_counter() - started, "scene_jobs_completed": len(completed)}
    np.savez_compressed(cache_path, features=features, ids=ids, mask=mask)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"features": features, "ids": ids, "mask": mask}, metadata


def _pad_map_options(cache: Mapping[str, np.ndarray], width: int) -> dict[str, np.ndarray]:
    """Pad a subset map cache to the recognizer's fixed option width."""
    current = int(cache["features"].shape[1])
    if current > width:
        raise ValueError(f"map option width {current} exceeds recognizer width {width}")
    if current == width:
        return {key: np.asarray(value) for key, value in cache.items()}
    rows = int(cache["features"].shape[0])
    features = np.full((rows, width, MAP_DIM), np.nan, dtype=np.float32)
    ids = np.full((rows, width), -1, dtype=np.int64)
    mask = np.zeros((rows, width), dtype=bool)
    features[:, :current] = cache["features"]
    ids[:, :current] = cache["ids"]
    mask[:, :current] = cache["mask"]
    return {"features": features, "ids": ids, "mask": mask}


class UtilityMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape[:-1]
        return self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(shape)


def _utility_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(pred[mask], target[mask])
    terms: list[torch.Tensor] = []
    for index in range(pred.shape[0]):
        active = mask[index]
        if int(active.sum()) >= 2:
            target_prob = torch.softmax(target[index, active] / LISTWISE_TAU, dim=0)
            terms.append(-(target_prob * torch.log_softmax(pred[index, active] / LISTWISE_TAU, dim=0)).sum())
    ranking = torch.stack(terms).mean() if terms else regression.new_zeros(())
    return regression + LISTWISE_WEIGHT * ranking, regression, ranking


def _sample_rows(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["record_id"])].append(index)
    selected: list[int] = []
    for record in sorted(groups):
        values = np.asarray(groups[record], dtype=np.int64)
        selected.extend(rng.choice(values, OBS_PER_RECORD, replace=len(values) < OBS_PER_RECORD).tolist())
    return np.asarray(selected, dtype=np.int64)


def _train_utility(train_x: np.ndarray, val_x: np.ndarray, train_target: np.ndarray, val_target: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, train_rows: Sequence[Mapping[str, Any]], data_root: Path, output_root: Path, device: torch.device) -> tuple[UtilityMLP, dict[str, Any]]:
    checkpoint = data_root / CHECKPOINT_REL
    summary_path = output_root / "mapfeature_utility_training.json"
    model = UtilityMLP(train_x.shape[-1]).to(device)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if int(payload.get("input_dim", -1)) != train_x.shape[-1] or bool(payload.get("test_used", True)):
            raise ValueError("MapFeature-Utility checkpoint metadata mismatch")
        model.load_state_dict(payload["state_dict"])
        return model.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    history: list[dict[str, Any]] = []
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        sampled = _sample_rows(train_rows, rng)
        order = rng.permutation(len(sampled))
        model.train()
        train_losses: list[float] = []
        for start in range(0, len(order), TRAIN_BATCH):
            index = sampled[order[start : start + TRAIN_BATCH]]
            x = torch.from_numpy(train_x[index]).to(device, non_blocking=True)
            y = torch.from_numpy(train_target[index]).to(device, non_blocking=True)
            mask = torch.from_numpy(train_mask[index]).to(device, non_blocking=True)
            total, _, _ = _utility_loss(model(x), y, mask)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(total.detach().cpu()))
        model.eval()
        values: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_x), EVAL_BATCH):
                sl = slice(start, start + EVAL_BATCH)
                pred = model(torch.from_numpy(val_x[sl]).to(device, non_blocking=True))
                loss, _, _ = _utility_loss(pred, torch.from_numpy(val_target[sl]).to(device, non_blocking=True), torch.from_numpy(val_mask[sl]).to(device, non_blocking=True))
                values.append(float(loss.cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_total_loss": float(np.mean(values)), "sampled_contexts": int(len(sampled)), "unique_records": int(len({str(item['record_id']) for item in train_rows}))}
        history.append(row)
        print(f"[MapFeature-Utility] epoch={epoch:02d} train={row['train_loss']:.6f} val={row['val_total_loss']:.6f}", flush=True)
        if row["val_total_loss"] < best_loss:
            best_loss, best_epoch = row["val_total_loss"], epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "input_dim": train_x.shape[-1], "epoch": epoch, "seed": SEED, "test_used": False}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {"branch": "MapFeature-Utility", "input_dim": int(train_x.shape[-1]), "architecture": "Linear(input_dim,256)-GELU-Linear(256,256)-GELU-Linear(256,1)", "epochs": EPOCHS, "seed": SEED, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "observations_per_record": OBS_PER_RECORD, "checkpoint_selection": "minimum Val utility loss", "best_epoch": best_epoch, "best_val_total_loss": best_loss, "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model.eval(), summary

def _predict(model: UtilityMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), EVAL_BATCH):
            result.append(model(torch.from_numpy(x[start : start + EVAL_BATCH]).to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(result, axis=0)

def _select(scores: np.ndarray, ids: np.ndarray, mask: np.ndarray, geodesic: np.ndarray | None = None) -> list[int]:
    actions: list[int] = []
    for index in range(len(scores)):
        active = np.flatnonzero(mask[index])
        def key(slot: int) -> tuple[float, float, int]:
            cost = 0.0 if geodesic is None else float(geodesic[index, slot])
            return (-float(scores[index, slot]), cost, int(ids[index, slot]))
        actions.append(int(ids[index, min(active.tolist(), key=key)]))
    return actions

def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    output: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((ids[index] == int(action)) & mask[index])
        if slots.size != 1:
            raise ValueError(f"selected action is not legal/unique at row {index}: {action}")
        output.append(int(np.argmax(logp[index, int(slots[0])])))
    return np.asarray(output, dtype=np.int64)


def _metric(name: str, labels: np.ndarray, predictions: np.ndarray, rows: Sequence[Mapping[str, Any]], actions: Sequence[int]) -> dict[str, Any]:
    value = classification(labels, predictions)
    moves = np.asarray([int(action) != int(row["current_viewpoint_id"]) for action, row in zip(actions, rows)], dtype=bool)
    value.update({"selector": name, "move_rate": float(np.mean(moves)), "stay_rate": float(1.0 - np.mean(moves)), "test_used": False})
    return value


def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(std, 1e-6)


def _ranking(name: str, scores: np.ndarray, target: np.ndarray, mask: np.ndarray, ids: np.ndarray, oracle_actions: Sequence[int]) -> dict[str, Any]:
    valid = mask
    candidate = mask.copy(); candidate[:, 0] = False
    within: list[float] = []
    top1: list[int] = []
    for index in range(scores.shape[0]):
        active = np.flatnonzero(valid[index])
        if active.size < 2:
            continue
        within.append(correlation(scores[index, active], target[index, active], spearman=True))
        order = active[np.argsort(-scores[index, active], kind="mergesort")]
        oracle = int(np.flatnonzero(ids[index] == int(oracle_actions[index]))[0])
        top1.append(int(order[0] == oracle))
    return {"method": name, "candidate_level_spearman": correlation(scores[valid], target[valid], spearman=True), "candidate_only_spearman": correlation(scores[candidate], target[candidate], spearman=True), "within_context_spearman_mean": float(np.mean(within)) if within else 0.0, "within_context_spearman_median": float(np.median(within)) if within else 0.0, "gt_true_logp_top1_overlap": float(np.mean(top1)) if top1 else 0.0}


def _navigation_metrics(actions: Sequence[int], rows: Sequence[Mapping[str, Any]], map_features: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    costs: list[float] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((ids[index] == int(action)) & mask[index])
        costs.append(float(map_features[index, int(slots[0]), NAV_SLICE][0]))
    values = np.asarray(costs, dtype=np.float64)
    return {"mean_geodesic_m": float(np.mean(values)), "median_geodesic_m": float(np.median(values)), "p90_geodesic_m": float(np.percentile(values, 90)), "move_rate": float(np.mean(values > 1e-8))}


def _visibility_consistency(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    map_cache: Mapping[str, np.ndarray],
    scalar: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare new frame-0 rays with the accepted structured visibility cache."""
    root = data_root / STRUCTURED_REL
    metadata = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        raise ValueError(f"structured visibility signature mismatch for {split}")
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        old = np.asarray(archive["visibility"], dtype=np.float32)
    active = np.asarray(map_cache["mask"], dtype=bool)
    new = np.asarray(map_cache["features"], dtype=np.float32)[..., JOINT_SLICE]
    if old.shape != new.shape:
        raise ValueError(f"structured/new visibility shape mismatch for {split}: {old.shape} vs {new.shape}")
    valid = active & np.isfinite(old).all(axis=-1) & np.isfinite(new).all(axis=-1)
    differences = np.abs(old[valid] - new[valid])
    result: dict[str, Any] = {
        "available": True,
        "valid_slots": int(valid.sum()),
        "max_abs_difference": float(np.max(differences)) if differences.size else 0.0,
        "mean_abs_difference": float(np.mean(differences)) if differences.size else 0.0,
        "tolerance": 1e-6,
        "pass": bool(differences.size == 0 or float(np.max(differences)) <= 1e-6),
    }
    if scalar is not None:
        old_mean = np.mean(old, axis=-1)
        scalar_valid = active & np.isfinite(scalar)
        aligned = scalar_valid & valid
        scalar_diff = np.abs(old_mean[aligned] - scalar[aligned])
        result["scalar_mean_max_abs_difference"] = float(np.max(scalar_diff)) if scalar_diff.size else 0.0
        result["scalar_mean_pass"] = bool(scalar_diff.size == 0 or float(np.max(scalar_diff)) <= 1e-6)
    return result

def _load_scalar(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> np.ndarray:
    root = data_root / SCALAR_REL
    metadata = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    source_rows: Sequence[Mapping[str, Any]] = rows
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        # Smoke runs intentionally use a prefix. Reuse the full validated
        # cache only for that exact prefix; arbitrary subsets fail closed.
        full_train, full_val = load_rows(data_root)
        full_rows = full_train if split == "train" else full_val
        if len(rows) > len(full_rows) or any(dict(a) != dict(b) for a, b in zip(rows, full_rows[: len(rows)])):
            raise ValueError(f"scalar visibility signature mismatch for {split}")
        if int(metadata.get("rows", -1)) != len(full_rows) or metadata.get("signature") != row_signature(full_rows):
            raise ValueError(f"scalar visibility cache metadata mismatch for {split}")
        source_rows = full_rows
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        scores, ids, mask = np.asarray(archive["scores"]), np.asarray(archive["ids"]), np.asarray(archive["mask"], dtype=bool)
    output = np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32)
    for index, row in enumerate(source_rows[: len(rows)]):
        old = {int(value): slot for slot, value in enumerate(ids[index, mask[index]].tolist())}
        for slot, action in enumerate([int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]):
            output[index, slot] = float(scores[index, old[action]])
    return output

def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]
    scalar = methods.get("ScalarVisibility+Geometry", {})
    selector_names = [
        name for name in result["method_order"]
        if name != "GT-TrueLogP Oracle" and name in methods
    ]
    best = max((methods[name]["accuracy"] for name in selector_names), default=0.0)
    decision = "STRONG KEEP" if best >= 0.60 else "KEEP" if best >= 0.58 else "WEAK KEEP" if best >= 0.55 else "KILL MAP-AWARE ROUTE"
    lines = [
        "# Known-3D-Map Robust Observability NBV Audit", "",
        "Experiment: Map-aware Robust Observability NBV Audit", "Problem: pre-action single-step active viewpoint selection for HAR", "Environment: pre-built static/quasi-static HM3D map; unknown-space exploration=false; SLAM=outside scope", "Current human information: exact frame-0 H36M17 pose (privileged diagnostic)", "Future human motion/action/candidate RGB/skeleton: not used for selection", "Action set: Stay + Stage-A legal candidates", "Recognizer: frozen reduced12 ST-GCN + frozen shared head", "Policy Test: false", "",
        "## Moving-Val results", "", "| Method | Accuracy | Macro-F1 | Mean nav distance (m) | Move rate |", "|---|---:|---:|---:|---:|",
    ]
    for name in result["method_order"]:
        metric = methods[name]
        nav = result["navigation_cost_metrics"].get(name, {})
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {nav.get('mean_geodesic_m', 0.0):.4f} | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend(["", "## Gains versus old scalar visibility", "", f"- DenseGain: {100.0 * (methods['DenseVisibility']['accuracy'] - scalar.get('accuracy', 0.0)):+.3f} pp", f"- RobustGain: {100.0 * (methods['RobustComposite']['accuracy'] - scalar.get('accuracy', 0.0)):+.3f} pp", f"- MapFeatureGain: {100.0 * (methods['MapFeature-Utility']['accuracy'] - scalar.get('accuracy', 0.0)):+.3f} pp", "", "## Ranking diagnostics", "", "| Method | Candidate Spearman | Within-context Spearman | Oracle top-1 overlap |", "|---|---:|---:|---:|"])
    for name, value in result["ranking_metrics"].items():
        lines.append(f"| {name} | {value['candidate_only_spearman']:.6f} | {value['within_context_spearman_mean']:.6f} | {value['gt_true_logp_top1_overlap']:.6f} |")
    consistency = result.get("visibility_consistency", {})
    lines.extend(["", "## Accepted frame-0 visibility consistency", "", "| Split | Max abs diff vs structured cache | Scalar-mean max abs diff | Pass |", "|---|---:|---:|---|"])
    for split in ("train", "val"):
        value = consistency.get(split, {})
        lines.append(f"| {split} | {value.get('max_abs_difference', float('nan')):.8f} | {value.get('scalar_mean_max_abs_difference', float('nan')):.8f} | {value.get('pass', False)} |")
    high = result["occlusion_stratified_metrics"]
    lines.extend(["", f"## High-occlusion subset ({high['count']} contexts)", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name, value in high["methods"].items():
        lines.append(f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} |")
    lines.extend(["", "## Scene-density stratification", "", "| Density | ScalarVisibility | RobustComposite | MapFeature-Utility | Oracle |", "|---|---:|---:|---:|---:|"])
    for name, value in result["scene_density_metrics"].items():
        lines.append(f"| {name} | {value['ScalarVisibility+Geometry']['accuracy']:.6f} | {value['RobustComposite']['accuracy']:.6f} | {value['MapFeature-Utility']['accuracy']:.6f} | {value['GT-TrueLogP Oracle']['accuracy']:.6f} |")
    lines.extend(["", "## Scientific judgment", "", f"Decision: **{decision}** (best non-oracle Moving-Val Accuracy={best:.6f}; preregistered thresholds are 0.55/0.58/0.60).", "The map-aware values are privileged geometry diagnostics, not deployable policies. If the decision is KILL, do not continue depth, point-cloud, top-down-map neural selectors or larger map encoders; prioritize future recognizer evidence or sequential information acquisition.", "", "```text", "policy_test_used=false", "training_split=Policy Train", "evaluation_split=Moving Val", "current_frame0_pose_used=true", "future_motion_used=false", "future_candidate_observation_used=false", "gt_action_used=false", "deployable=false", "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything()
    device = require_cuda(args.device)
    data_root, scene_root, output_root = args.data_root.resolve(), args.scene_root.resolve(), args.output_root.resolve()
    all_train_rows, all_val_rows = load_rows(data_root)
    shared_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth"
    shared_head = SharedHead().to(device)
    shared_head.load_state_dict(torch.load(shared_path, map_location=device, weights_only=False)["state_dict"])
    shared_head.eval()
    all_train_logp, all_train_ids, all_train_mask = _load_options(data_root, all_train_rows, "train", shared_head, device)
    all_val_logp, all_val_ids, all_val_mask = _load_options(data_root, all_val_rows, "val", shared_head, device)
    if args.max_contexts is None:
        train_rows, val_rows = all_train_rows, all_val_rows
        train_logp, train_ids, train_mask = all_train_logp, all_train_ids, all_train_mask
        val_logp, val_ids, val_mask = all_val_logp, all_val_ids, all_val_mask
    else:
        limit = int(args.max_contexts)
        train_rows, val_rows = all_train_rows[:limit], all_val_rows[:limit]
        train_logp, train_ids, train_mask = all_train_logp[:limit], all_train_ids[:limit], all_train_mask[:limit]
        val_logp, val_ids, val_mask = all_val_logp[:limit], all_val_ids[:limit], all_val_mask[:limit]
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    train_map, train_map_meta = _extract_split(data_root, scene_root, train_rows, "train", args.workers)
    val_map, val_map_meta = _extract_split(data_root, scene_root, val_rows, "val", args.workers)
    train_map = _pad_map_options(train_map, train_ids.shape[1])
    val_map = _pad_map_options(val_map, val_ids.shape[1])
    scalar = _load_scalar(data_root, val_rows, "val")
    visibility_consistency = {
        "train": _visibility_consistency(data_root, train_rows, "train", train_map),
        "val": _visibility_consistency(data_root, val_rows, "val", val_map, scalar),
    }
    # Map extraction allocates the smallest option dimension for the selected
    # subset, whereas the recognizer cache keeps the fixed MAX_OPTIONS width.
    # Compare only legal slots; padded ``-1`` entries are representation
    # details, not action mismatches.
    for split_name, recognizer_ids, recognizer_mask, map_cache in (
        ("train", train_ids, train_mask, train_map),
        ("val", val_ids, val_mask, val_map),
    ):
        for index in range(len(recognizer_ids)):
            expected = recognizer_ids[index, recognizer_mask[index]].tolist()
            observed = map_cache["ids"][index, map_cache["mask"][index]].tolist()
            if expected != observed:
                raise ValueError(f"recognizer/map action IDs differ for {split_name} row {index}")
    stats = json.loads((data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8"))
    train_geometry, _, _ = _option_geometry(train_rows, stats)
    val_geometry, _, _ = _option_geometry(val_rows, stats)
    # Invalid padded candidate slots are masked during loss/selection, but the
    # MLP still evaluates every slot in a dense tensor.  Replace their NaNs by
    # zeros so masked padding cannot poison otherwise valid candidate scores.
    train_map_features = np.nan_to_num(train_map["features"], nan=0.0, posinf=0.0, neginf=0.0)
    val_map_features = np.nan_to_num(val_map["features"], nan=0.0, posinf=0.0, neginf=0.0)
    train_geometry = np.nan_to_num(train_geometry, nan=0.0, posinf=0.0, neginf=0.0)
    val_geometry = np.nan_to_num(val_geometry, nan=0.0, posinf=0.0, neginf=0.0)
    train_x = np.concatenate((train_map_features, train_geometry), axis=-1).astype(np.float32)
    val_x = np.concatenate((val_map_features, val_geometry), axis=-1).astype(np.float32)
    train_target = np.full(train_mask.shape, np.nan, dtype=np.float32)
    val_target = np.full(val_mask.shape, np.nan, dtype=np.float32)
    for index, label in enumerate(train_labels):
        train_target[index, train_mask[index]] = train_logp[index, train_mask[index], int(label)]
    for index, label in enumerate(val_labels):
        val_target[index, val_mask[index]] = val_logp[index, val_mask[index], int(label)]
    output_root.mkdir(parents=True, exist_ok=True)
    model, training_summary = _train_utility(train_x, val_x, train_target, val_target, train_mask, val_mask, train_rows, data_root, output_root, device)
    map_scores = _predict(model, val_x, device)
    train_values = train_map["features"][train_map["mask"]]
    feature_mean = np.mean(train_values, axis=0)
    feature_std = np.std(train_values, axis=0)
    geo_values = train_map["features"][..., NAV_SLICE][train_map["mask"]]
    geo_mean, geo_std = float(np.mean(geo_values[:, 0])), float(np.std(geo_values[:, 0]))
    robust_raw = val_map["features"][..., ROBUST_MEAN_INDEX]
    robust = _zscore(robust_raw, float(feature_mean[ROBUST_MEAN_INDEX]), float(feature_std[ROBUST_MEAN_INDEX]))
    dense = val_map["features"][..., DENSE_INDEX]
    dense_z = _zscore(dense, float(feature_mean[DENSE_INDEX]), float(feature_std[DENSE_INDEX]))
    fov_z = _zscore(val_map["features"][..., 35], float(feature_mean[35]), float(feature_std[35]))
    area_z = _zscore(val_map["features"][..., 31], float(feature_mean[31]), float(feature_std[31]))
    clear_z = _zscore(val_map["features"][..., 28], float(feature_mean[28]), float(feature_std[28]))
    robust_composite = dense_z + robust + 0.5 * fov_z + 0.5 * area_z + 0.5 * clear_z
    nav_z = _zscore(val_map["features"][..., 42], geo_mean, geo_std)
    nav_scores = {lam: robust_composite - lam * nav_z for lam in (0.1, 0.25, 0.5)}
    score_arrays: dict[str, np.ndarray] = {
        "JointVisibility": np.mean(val_map["features"][..., JOINT_SLICE], axis=-1),
        "DenseVisibility": dense,
        "RobustVis005": val_map["features"][..., 24],
        "RobustVis010": val_map["features"][..., 25],
        "RobustVis020": val_map["features"][..., 26],
        "RobustVisMean": val_map["features"][..., ROBUST_MEAN_INDEX],
        "ProjectionArea": val_map["features"][..., 31],
        "FOVMargin": val_map["features"][..., 35],
        "RobustComposite": robust_composite,
        "RobustComposite-nav λ=.1": nav_scores[0.1],
        "RobustComposite-nav λ=.25": nav_scores[0.25],
        "RobustComposite-nav λ=.5": nav_scores[0.5],
        "MapFeature-Utility": map_scores,
    }
    stay_actions = [int(row["current_viewpoint_id"]) for row in val_rows]
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice(val_ids[index, val_mask[index]])) for index in range(len(val_rows))]
    oracle_actions = _select(val_target, val_ids, val_mask, val_map["features"][..., 42])
    methods: dict[str, dict[str, Any]] = {}
    actions: dict[str, list[int]] = {"Stay": stay_actions, "Random": random_actions, "GT-TrueLogP Oracle": oracle_actions}
    for name, selected in actions.items():
        methods[name] = _metric(name, val_labels, _terminal(val_logp, val_ids, val_mask, selected), val_rows, selected)
    for name, scores in score_arrays.items():
        selected = _select(scores, val_ids, val_mask, val_map["features"][..., 42])
        actions[name] = selected
        methods[name] = _metric(name, val_labels, _terminal(val_logp, val_ids, val_mask, selected), val_rows, selected)
    methods.update(historical_methods(HISTORICAL_STRUCTURED, HISTORICAL_SHARED))
    order = ["Stay", "Random", "JointVisibility", "DenseVisibility", "RobustVis005", "RobustVis010", "RobustVis020", "RobustVisMean", "ProjectionArea", "FOVMargin", "RobustComposite", "RobustComposite-nav λ=.1", "RobustComposite-nav λ=.25", "RobustComposite-nav λ=.5", "MapFeature-Utility", "ScalarVisibility+Geometry", "Frame0 deployable best", "GT-TrueLogP Oracle"]
    order = [name for name in order if name in methods]
    ranking = {name: _ranking(name, val_target if name == "GT-TrueLogP Oracle" else (score_arrays[name] if name in score_arrays else val_target), val_target, val_mask, val_ids, oracle_actions) for name in [*score_arrays, "GT-TrueLogP Oracle"]}
    navigation = {name: _navigation_metrics(actions[name], val_rows, val_map["features"], val_ids, val_mask) for name in actions}
    high_mask = scalar[:, 0] < np.quantile(scalar[:, 0], 1.0 / 3.0)
    high_methods = {
        name: classification(
            val_labels[high_mask],
            _terminal(
                val_logp[high_mask],
                val_ids[high_mask],
                val_mask[high_mask],
                [actions[name][i] for i in np.flatnonzero(high_mask)],
            ),
        )
        for name in order
        if name in actions
    }
    density_values = val_map["features"][:, 0, SCENE_CLEARANCE_INDEX]
    edges = np.quantile(density_values, [1.0 / 3.0, 2.0 / 3.0])
    density_metrics: dict[str, Any] = {}
    for density_name, selector in (("cluttered", density_values <= edges[0]), ("medium", (density_values > edges[0]) & (density_values <= edges[1])), ("open", density_values > edges[1])):
        density_metrics[density_name] = {}
        for name in ("ScalarVisibility+Geometry", "RobustComposite", "MapFeature-Utility", "GT-TrueLogP Oracle"):
            if name not in actions or not np.any(selector):
                density_metrics[density_name][name] = classification([], [])
                continue
            pred = _terminal(val_logp[selector], val_ids[selector], val_mask[selector], [actions[name][i] for i in np.flatnonzero(selector)])
            density_metrics[density_name][name] = classification(val_labels[selector], pred)
    definition = {"map_dim": MAP_DIM, "layout": {"joint_visibility": list(range(0, 17)), "dense_visibility": DENSE_INDEX, "body_part_visibility": {name: 18 + index for index, name in enumerate(PART_NAMES)}, "robust_vis_005": 24, "robust_vis_010": 25, "robust_vis_020": 26, "robust_vis_mean": ROBUST_MEAN_INDEX, "clearance_mean_p10_min": list(range(28, 31)), "projection_area_height_width_fov_fraction_boundary_margin": list(range(31, 36)), "camera_distances": list(range(36, 40)), "scene_clearance": SCENE_CLEARANCE_INDEX, "candidate_free_space": CANDIDATE_FREE_SPACE_INDEX, "navigation_geodesic_ratio_yaw": list(range(42, 45)), "current_pose_root_relative": list(range(45, 96))}, "body_parts": list(PART_NAMES), "dense_definition": "16 samples per H36M17 bone (including endpoints) plus center and six offsets per joint", "robust_definition": "17 current joints x six axis perturbations at eps={0.05,0.10,0.20}m", "clearance_definition": "four endpoint cone rays around eight representative current joints", "projection_definition": "existing 256x256 HFOV75 WXYZ camera convention", "navigation_definition": "Stage-A geodesic; Stay=0", "test_used": False}
    result: dict[str, Any] = {"experiment_id": "REDUCED12_KNOWN_3D_MAP_ROBUST_OBSERVABILITY_AUDIT", "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows), "train_records": len({str(row['record_id']) for row in train_rows}), "val_records": len({str(row['record_id']) for row in val_rows}), "candidate_count": int(val_mask[:, 1:].sum())}, "labels": list(LABELS), "method_order": order, "methods": methods, "ranking_metrics": ranking, "navigation_cost_metrics": navigation, "occlusion_stratified_metrics": {"count": int(high_mask.sum()), "threshold": float(np.quantile(scalar[:, 0], 1.0 / 3.0)), "methods": high_methods}, "scene_density_metrics": density_metrics, "map_feature_definition": definition, "training_summary": training_summary, "protocol": {"action_set": "Stay/current + Stage-A legal candidate_pool", "terminal": "selected real archived full 30-frame skeleton through frozen ST-GCN/shared head", "current_human": "exact frame-0 H36M17 world pose", "map": "static Habitat HM3D scene/navmesh", "unknown_space_exploration": False, "slam": "outside scope"}, "leakage_flags": {"policy_test_used": False, "training_new_model_used": True, "new_rgb_generated": False, "new_skeleton_generated": False, "new_dino_generated": False, "frozen_stgcn_modified": False, "future_motion_used": False, "future_candidate_rgb_used": False, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "gt_action_used": False, "candidate_recognizer_output_used": False, "deployable": False}, "artifacts": {"train_map_cache": str((data_root / RUNTIME_REL / "train.npz").resolve()), "val_map_cache": str((data_root / RUNTIME_REL / "val.npz").resolve()), "train_map_meta": train_map_meta, "val_map_meta": val_map_meta, "shared_head": str(shared_path.resolve())}, "runtime": {"device": str(device), "workers": args.workers, "seed": SEED, "epochs": EPOCHS, "map_feature_dim": MAP_DIM, "output_root": str(output_root)}}
    result["visibility_consistency"] = visibility_consistency
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output_root / "analysis.md")
    (output_root / "config.json").write_text(json.dumps({"map_feature_dim": MAP_DIM, "dense_bone_samples": DENSE_BONE_SAMPLES, "robust_epsilons_m": list(ROBUST_EPSILONS), "robust_directions": ROBUST_DIRECTIONS, "clearance_joints": list(CLEARANCE_JOINTS), "workers": args.workers, "seed": SEED, "epochs": EPOCHS, "test_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "map_feature_definition.json").write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    (output_root / "per_method_metrics.json").write_text(json.dumps({name: methods[name] for name in order}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "ranking_metrics.json").write_text(json.dumps(ranking, indent=2) + "\n", encoding="utf-8")
    (output_root / "occlusion_stratified_metrics.json").write_text(json.dumps(result["occlusion_stratified_metrics"], indent=2) + "\n", encoding="utf-8")
    (output_root / "scene_density_metrics.json").write_text(json.dumps(density_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "navigation_cost_metrics.json").write_text(json.dumps(navigation, indent=2) + "\n", encoding="utf-8")
    (output_root / "dense_visibility_audit.json").write_text(json.dumps({"train": train_map_meta, "val": val_map_meta, "dense_points": definition["dense_definition"]}, indent=2) + "\n", encoding="utf-8")
    (output_root / "robust_visibility_audit.json").write_text(json.dumps({"epsilons_m": list(ROBUST_EPSILONS), "directions": ROBUST_DIRECTIONS, "definition": definition["robust_definition"]}, indent=2) + "\n", encoding="utf-8")
    (output_root / "projection_quality_audit.json").write_text(json.dumps({"definition": definition["projection_definition"]}, indent=2) + "\n", encoding="utf-8")
    (output_root / "coverage_audit.json").write_text(json.dumps({"train_candidate_count": int(train_mask[:, 1:].sum()), "val_candidate_count": int(val_mask[:, 1:].sum()), "all_legal_finite": bool(np.isfinite(val_map["features"][val_map["mask"]]).all()), "joint_visibility_consistency": visibility_consistency}, indent=2) + "\n", encoding="utf-8")
    (output_root / "leakage_audit.json").write_text(json.dumps(result["leakage_flags"], indent=2) + "\n", encoding="utf-8")
    (output_root / "training_summary.json").write_text(json.dumps(training_summary, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-contexts", type=int, default=None, help="small smoke subset; omit for full Train/Moving-Val")
    args = parser.parse_args()
    if args.workers <= 0 or (args.max_contexts is not None and args.max_contexts <= 0):
        raise ValueError("workers/max-contexts must be positive")
    result = run(args)
    print(json.dumps({"status": result["status"], "population": result["population"], "methods": {name: {"accuracy": result["methods"][name]["accuracy"], "macro_f1": result["methods"][name]["macro_f1"]} for name in result["method_order"]}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
