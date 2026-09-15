#!/usr/bin/env python3
"""GT human-mask + current-depth root recovery ceiling audit.

The audit is intentionally small in scope: current frame 0 only, Policy Train
for the two priors, and Moving Val for evaluation.  A Habitat OBJECT_ID mask
selects the articulated humanoid pixels while the depth sensor is rendered at
the same archived camera pose.  The resulting human surface point cloud is
used only to estimate translation; D1 then reuses the frozen Yaw8 recognizer
and the Stage-A legal candidate-only scene-visibility protocol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
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
from activeview.scripts.eval.reduced12_nbv_utils import classification, gt_margin  # noqa: E402
from activeview.scripts.experiments.rgbd_coordinate_audit_helpers import summarize  # noqa: E402
from activeview.scripts.experiments.rgbd_human_state_helpers import (  # noqa: E402
    IMAGE_SIZE,
    SENSOR_HEIGHT,
    placement_yaw,
    read_json,
    rotation_wxyz_to_matrix,
    write_json,
)
from activeview.scripts.experiments.run_reduced12_known_map_geometric_nbv_upgrade import _load_map_cache  # noqa: E402
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (  # noqa: E402
    _load_rows,
    _read_npz,
    _signature,
    _train_prior,
)
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import _load_yaw8_options  # noqa: E402
from activeview.scripts.experiments.run_reduced12_yaw8_shared_head_fairness_audit import SharedHead  # noqa: E402
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (  # noqa: E402
    _ray_visible,
    _scene_file,
    _scene_sim,
)

SEED = 42
NUM_CLASSES = 12
JOINTS = 17
MAX_OPTIONS = 22
SCENE_ROOT_NAME = "hm3d-train"
RAW_VAL_REL = Path("datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json")
POLICY_RUNTIME = Path("diagnostics/rgbd_human_state_recovery_v1")
OUTPUT_DEFAULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/gtmask_depth_root_ceiling"
GT_CACHE_DEFAULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/rgbd_coordinate_sanity_audit/gt_world_joints_audit.npz"


def seed_everything() -> None:
    """Set deterministic seeds for the frozen recognizer path."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(name: str) -> torch.device:
    """Require the configured CUDA device; never silently run on CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _camera(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(str(row["archive_path"])), allow_pickle=False) as archive:
        return np.asarray(archive["viewpoint_agent_positions"], dtype=np.float64), np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float64)


def _floor_heights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Read the known placement floor once per record/region."""
    cache: dict[tuple[str, str], float] = {}
    values = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        key = (str(row["record_id"]), str(row["region"]))
        if key not in cache:
            with np.load(Path(str(row["archive_path"])), allow_pickle=False) as archive:
                cache[key] = float(np.asarray(archive["placement_position"], dtype=np.float64)[1])
        values[index] = cache[key]
    return values


def _sim_with_mask(scene_id: str) -> tuple[Any, Any]:
    """Create a scene simulator with depth and OBJECT_ID semantic sensors."""
    import habitat_sim
    import magnum as mn

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(_scene_file(get_habitat_data_root() / SCENE_ROOT_NAME, scene_id, ".basis.glb"))
    config.enable_physics = True
    agent = habitat_sim.AgentConfiguration()
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth_0"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [IMAGE_SIZE, IMAGE_SIZE]
    depth.position = mn.Vector3(0.0, SENSOR_HEIGHT, 0.0)
    depth.hfov = mn.Deg(75.0)
    semantic = habitat_sim.CameraSensorSpec()
    semantic.uuid = "semantic_0"
    semantic.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic.semantic_target = type(semantic.semantic_target).OBJECT_ID
    semantic.resolution = [IMAGE_SIZE, IMAGE_SIZE]
    semantic.position = mn.Vector3(0.0, SENSOR_HEIGHT, 0.0)
    semantic.hfov = mn.Deg(75.0)
    agent.sensor_specifications = [depth, semantic]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    navmesh = _scene_file(get_habitat_data_root() / SCENE_ROOT_NAME, scene_id, ".basis.navmesh")
    if not sim.pathfinder.is_loaded and not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError(f"failed to load navmesh for {scene_id}")
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    return sim, human


def _set_camera(sim: Any, position: np.ndarray, rotation: np.ndarray) -> None:
    import habitat_sim
    import quaternion
    import magnum as mn

    state = habitat_sim.AgentState()
    state.position = mn.Vector3(*np.asarray(position, dtype=np.float32))
    state.rotation = quaternion.from_float_array(np.asarray(rotation, dtype=np.float64))
    sim.get_agent(0).set_state(state)


def _point_cloud(depth: np.ndarray, mask: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Backproject all masked metric-depth pixels into world coordinates."""
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0.01))
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    focal = 0.5 * IMAGE_SIZE / math.tan(math.radians(75.0) * 0.5)
    values = depth[ys, xs].astype(np.float64)
    camera = np.stack(((xs - 0.5 * IMAGE_SIZE) * values / focal, -(ys - 0.5 * IMAGE_SIZE) * values / focal, -values), axis=1)
    sensor = np.asarray(position, dtype=np.float64) + np.asarray([0.0, SENSOR_HEIGHT, 0.0])
    world = camera @ rotation_wxyz_to_matrix(rotation).T + sensor
    return world.astype(np.float32)


def _render_scene(job: Mapping[str, Any]) -> str:
    """Render one scene shard; only compact roots and mask/depth counts persist."""
    data_root = Path(str(job["data_root"]))
    output = Path(str(job["output"]))
    scene_id = str(job["scene_id"])
    raw = {str(item["record_id"]): item for item in json.loads((data_root / RAW_VAL_REL).read_text(encoding="utf-8"))}
    sim, human = _sim_with_mask(scene_id)
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    converted_cache: dict[str, Mapping[str, Any]] = {}
    offset_cache: dict[tuple[str, str], float] = {}
    roots = np.full((len(job["rows"]), 3), np.nan, dtype=np.float32)
    mask_pixels = np.zeros(len(job["rows"]), dtype=np.int64)
    finite_depth_pixels = np.zeros(len(job["rows"]), dtype=np.int64)
    depth_medians = np.full(len(job["rows"]), np.nan, dtype=np.float32)
    try:
        for local, item in enumerate(job["rows"]):
            source = Path(str(item["archive_path"]))
            record_id, region = str(item["record_id"]), str(item["region"])
            if record_id not in converted_cache:
                converted_cache[record_id] = converter.convert(_load_resampled_motion(raw[record_id], 30))
            converted = converted_cache[record_id]
            joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
            motion_roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
            key = (record_id, region)
            if key not in offset_cache:
                offsets, _ = precompute_grounding_offsets(human, joints[:1], motion_roots[:1], scene_yaw_deg=0.0)
                offset_cache[key] = float(offsets[0])
            with np.load(source, allow_pickle=False) as archive:
                base = np.asarray(archive["placement_position"], dtype=np.float32)
                positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
                rotations = np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32)
            yaw = placement_yaw(source, region)
            apply_humanoid_pose(human, joints[0], motion_roots[0], base_position=base, scene_yaw_deg=yaw, floor_y=float(base[1]), grounding_offset=offset_cache[key])
            view = int(item["current_viewpoint_id"])
            _set_camera(sim, positions[view], rotations[view])
            observation = sim.get_sensor_observations([0])[0]
            depth = np.asarray(observation["depth_0"], dtype=np.float32)
            semantic = np.asarray(observation["semantic_0"], dtype=np.uint32)
            mask = semantic == int(human.object_id)
            points = _point_cloud(depth, mask, positions[view], rotations[view])
            if points.size:
                roots[local, 0] = float(np.median(points[:, 0]))
                roots[local, 2] = float(np.median(points[:, 2]))
                valid_depth = depth[mask & np.isfinite(depth) & (depth > 0.01)]
                depth_medians[local] = float(np.median(valid_depth)) if valid_depth.size else np.nan
            mask_pixels[local] = int(mask.sum())
            finite_depth_pixels[local] = int(np.sum(mask & np.isfinite(depth) & (depth > 0.01)))
    finally:
        sim.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, indices=np.asarray([int(item["index"]) for item in job["rows"]], dtype=np.int64), raw_root_xz=roots[:, [0, 2]], mask_pixels=mask_pixels, finite_depth_pixels=finite_depth_pixels, depth_median=depth_medians)
    return str(output)


def _build_mask_cache(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str, runtime_root: Path, workers: int, pelvis_height: float, floor: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render compact point-cloud roots per scene with spawned Habitat workers."""
    cache_path = runtime_root / f"{split}.npz"
    meta_path = runtime_root / f"{split}.json"
    if cache_path.is_file() and meta_path.is_file():
        metadata = read_json(meta_path)
        if int(metadata.get("rows", -1)) == len(rows) and metadata.get("signature") == _signature(rows):
            with np.load(cache_path, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}, metadata
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["scene_id"])].append({"index": index, **{key: row[key] for key in ("archive_path", "record_id", "region", "scene_id", "current_viewpoint_id")}})
    shard_root = runtime_root / f"{split}_shards"
    jobs = [{"data_root": str(data_root), "scene_id": scene, "rows": values, "output": str(shard_root / f"{scene}.npz")} for scene, values in sorted(grouped.items())]
    shard_root.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not Path(str(job["output"])).is_file()]
    if workers == 1:
        for index, job in enumerate(pending, 1):
            _render_scene(job)
            print(f"mask-depth {split}: scene {index}/{len(pending)}", flush=True)
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = [executor.submit(_render_scene, job) for job in pending]
            for index, future in enumerate(futures, 1):
                future.result()
                print(f"mask-depth {split}: scene {index}/{len(pending)}", flush=True)
    roots = np.full((len(rows), 3), np.nan, dtype=np.float32)
    roots[:, 1] = floor.astype(np.float32) + float(pelvis_height)
    mask_pixels = np.zeros(len(rows), dtype=np.int64)
    finite_depth_pixels = np.zeros(len(rows), dtype=np.int64)
    depth_medians = np.full(len(rows), np.nan, dtype=np.float32)
    for job in jobs:
        with np.load(Path(str(job["output"])), allow_pickle=False) as archive:
            for local, index in enumerate(np.asarray(archive["indices"], dtype=np.int64)):
                roots[index, 0] = archive["raw_root_xz"][local, 0]
                roots[index, 2] = archive["raw_root_xz"][local, 1]
                mask_pixels[index] = archive["mask_pixels"][local]
                finite_depth_pixels[index] = archive["finite_depth_pixels"][local]
                depth_medians[index] = archive["depth_median"][local]
    arrays = {"root_world": roots, "mask_pixels": mask_pixels, "finite_depth_pixels": finite_depth_pixels, "depth_median": depth_medians}
    runtime_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    metadata = {"split": split, "rows": len(rows), "signature": _signature(rows), "sensor": "Habitat depth_0 + semantic_0 OBJECT_ID", "frame_index": 0, "mask_object_id": True, "raw_depth_persisted": False, "candidate_depth_used": False, "test_used": False}
    write_json(meta_path, metadata)
    return arrays, metadata


def _root_metrics(estimated: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(estimated, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    norm = np.linalg.norm(delta, axis=1)
    horizontal = np.linalg.norm(delta[:, [0, 2]], axis=1)
    return {"euclidean_m": summarize(norm), "horizontal_xz_m": summarize(horizontal), "vertical_abs_m": summarize(np.abs(delta[:, 1]))}


def _radial_calibration(train_rows: Sequence[Mapping[str, Any]], train_raw: np.ndarray, train_gt: np.ndarray) -> dict[str, Any]:
    offsets: list[float] = []
    for index, row in enumerate(train_rows):
        positions, _ = _camera(row)
        camera_xz = positions[int(row["current_viewpoint_id"]), [0, 2]]
        direction = train_raw[index, [0, 2]] - camera_xz
        length = float(np.linalg.norm(direction))
        if length > 1.0e-6 and np.isfinite(direction).all():
            gt_root_xz = train_gt[index, 0, [0, 2]]
            offsets.append(float(np.dot(gt_root_xz - train_raw[index, [0, 2]], direction / length)))
    return {"train_samples": len(offsets), "radial_offset_b_m": float(np.median(offsets)) if offsets else 0.0, "definition": "median Train (GT root XZ - raw point-cloud root XZ) projected on camera-to-raw-root direction", "test_used": False}


def _apply_radial(rows: Sequence[Mapping[str, Any]], raw: np.ndarray, offset: float) -> np.ndarray:
    result = np.asarray(raw, dtype=np.float32).copy()
    for index, row in enumerate(rows):
        positions, _ = _camera(row)
        camera_xz = positions[int(row["current_viewpoint_id"]), [0, 2]]
        direction = result[index, [0, 2]] - camera_xz
        length = float(np.linalg.norm(direction))
        if length > 1.0e-6 and np.isfinite(direction).all():
            result[index, [0, 2]] += float(offset) * (direction / length).astype(np.float32)
    return result


def _select(scores: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        count = len(row["candidate_ids"])
        slots = list(range(1, count + 1))
        slot = min(slots, key=lambda value: (-float(scores[index, value]), int(row["candidate_ids"][value - 1])))
        actions.append(int(row["candidate_ids"][slot - 1]))
    return actions


def _terminal(options: Mapping[str, np.ndarray], actions: Sequence[int]) -> np.ndarray:
    predictions = np.zeros(len(actions), dtype=np.int64)
    for index, action in enumerate(actions):
        slots = np.flatnonzero(options["ids"][index] == int(action))
        if slots.size != 1:
            raise ValueError(f"terminal action missing from options: {action}")
        predictions[index] = int(np.argmax(options["logp"][index, int(slots[0])]))
    return predictions


def _metric(name: str, rows: Sequence[Mapping[str, Any]], predictions: np.ndarray, actions: Sequence[int]) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    value = classification(labels, predictions)
    current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
    value.update({"method": name, "move_rate": float(np.mean(np.asarray(actions) != current)), "test_used": False})
    return value


def _oracle_actions(options: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], criterion: str) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(rows):
        slots = np.arange(1, len(row["candidate_ids"]) + 1)
        label = int(row["label_id"])
        if criterion == "true_logp":
            values = options["logp"][index, slots, label]
        else:
            values = np.asarray([gt_margin(options["logp"][index, slot], label) for slot in slots])
        result.append(int(row["candidate_ids"][int(slots[int(np.argmax(values))]) - 1]))
    return result


def _raycast_jobs(rows: Sequence[Mapping[str, Any]], gt_world: np.ndarray, roots: np.ndarray, output: Path, workers: int) -> np.ndarray:
    """Raycast candidate-only D1 visibility for all contexts and one root estimate."""
    runtime = output.parent
    split = output.stem
    score_path = output
    if score_path.is_file():
        with np.load(score_path, allow_pickle=False) as archive:
            return np.asarray(archive["scores"], dtype=np.float32)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["scene_id"])].append({"index": index, "scene_id": row["scene_id"], "archive_path": row["archive_path"], "candidate_ids": row["candidate_ids"], "root": roots[index].tolist(), "gt": gt_world[index].tolist()})
    shard_root = runtime / f"{split}_shards"
    jobs = [{"scene_id": scene, "rows": values, "output": str(shard_root / f"{scene}.npz")} for scene, values in sorted(grouped.items())]
    shard_root.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not Path(str(job["output"])).is_file()]
    if workers == 1:
        for index, job in enumerate(pending, 1):
            _raycast_scene(job)
            print(f"raycast {split}: scene {index}/{len(pending)}", flush=True)
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = [executor.submit(_raycast_scene, job) for job in pending]
            for index, future in enumerate(futures, 1):
                future.result()
                print(f"raycast {split}: scene {index}/{len(pending)}", flush=True)
    scores = np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32)
    for job in jobs:
        with np.load(Path(str(job["output"])), allow_pickle=False) as archive:
            for local, index in enumerate(np.asarray(archive["indices"], dtype=np.int64)):
                scores[index] = archive["scores"][local]
    np.savez_compressed(score_path, scores=scores)
    return scores


def _raycast_scene(job: Mapping[str, Any]) -> str:
    sim = _scene_sim(get_habitat_data_root() / SCENE_ROOT_NAME, str(job["scene_id"]), physics=True)
    scores = np.full((len(job["rows"]), MAX_OPTIONS), np.nan, dtype=np.float32)
    try:
        for local, item in enumerate(job["rows"]):
            with np.load(Path(str(item["archive_path"])), allow_pickle=False) as archive:
                positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
            gt = np.asarray(item["gt"], dtype=np.float32)
            root = np.asarray(item["root"], dtype=np.float32)
            joints = gt - gt[0][None, :] + root[None, :]
            for slot, viewpoint in enumerate(item["candidate_ids"], 1):
                origin = positions[int(viewpoint)] + np.asarray([0.0, SENSOR_HEIGHT, 0.0], dtype=np.float32)
                scores[local, slot] = float(np.mean([_ray_visible(sim, origin, endpoint) for endpoint in joints]))
    finally:
        sim.close()
    path = Path(str(job["output"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=np.asarray([int(item["index"]) for item in job["rows"]], dtype=np.int64), scores=scores)
    return str(path)


def _evaluate_d1(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], scores: np.ndarray, name: str) -> tuple[dict[str, Any], list[int], np.ndarray]:
    actions = _select(scores, rows)
    predictions = _terminal(options, actions)
    return _metric(name, rows, predictions, actions), actions, predictions


def run(output_root: Path, data_root: Path, device: torch.device, workers: int) -> dict[str, Any]:
    seed_everything()
    started = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = _load_rows(data_root)
    print(f"rows train={len(train_rows)} moving_val={len(val_rows)}", flush=True)
    head_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    options_train, train_meta = _load_yaw8_options(data_root, train_rows, "train", head, device)
    options_val, val_meta = _load_yaw8_options(data_root, val_rows, "moving_val", head, device)
    map_train, map_train_meta = _load_map_cache(data_root, "train", train_rows)
    map_val, map_val_meta = _load_map_cache(data_root, "val", val_rows)
    del head
    torch.cuda.empty_cache()
    if not GT_CACHE_DEFAULT.is_file():
        raise FileNotFoundError(f"independent GT world cache required: {GT_CACHE_DEFAULT}")
    with np.load(GT_CACHE_DEFAULT, allow_pickle=False) as archive:
        gt_train = np.asarray(archive["train_world_joints"], dtype=np.float32)
        gt_val = np.asarray(archive["val_world_joints"], dtype=np.float32)
    if gt_train.shape != (len(train_rows), JOINTS, 3) or gt_val.shape != (len(val_rows), JOINTS, 3):
        raise ValueError("GT world-joint cache does not match current Train/Moving-Val rows")
    floor_train, floor_val = _floor_heights(train_rows), _floor_heights(val_rows)
    pelvis_height = float(np.median(gt_train[:, 0, 1] - floor_train))
    runtime_root = data_root / "diagnostics/gtmask_depth_root_ceiling"
    train_mask, train_render_meta = _build_mask_cache(data_root, train_rows, "train", runtime_root, workers, pelvis_height, floor_train)
    val_mask, val_render_meta = _build_mask_cache(data_root, val_rows, "moving_val", runtime_root, workers, pelvis_height, floor_val)
    train_raw = np.asarray(train_mask["root_world"], dtype=np.float32)
    val_raw = np.asarray(val_mask["root_world"], dtype=np.float32)
    val_calibration = _radial_calibration(train_rows, train_raw, gt_train)
    val_cal = _apply_radial(val_rows, val_raw, float(val_calibration["radial_offset_b_m"]))
    write_json(output_root / "config.json", {"seed": SEED, "device": str(device), "workers": workers, "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "frame_index": 0, "mask_sensor": "Habitat semantic OBJECT_ID == articulated human.object_id", "depth_sensor": "Habitat SensorType.DEPTH depth_0 at same camera", "point_cloud": "all human-mask pixels with finite metric depth backprojected to world", "recognizer": "frozen Yaw8 encoder + Yaw8 shared head", "action_set": "Stage-A legal candidate-only", "policy_test_used": False, "training_used": False, "candidate_depth_used": False, "candidate_rgb_used": False})
    root_metrics = {"pelvis_height_prior_m": pelvis_height, "raw_root": _root_metrics(val_raw, gt_val[:, 0]), "calibrated_root": _root_metrics(val_cal, gt_val[:, 0]), "train_radial_calibration": val_calibration, "raw_mask_pixel_mean": float(np.mean(val_mask["mask_pixels"])), "finite_mask_depth_pixel_mean": float(np.mean(val_mask["finite_depth_pixels"])), "test_used": False}
    old_depth_path = data_root / POLICY_RUNTIME / "depth/moving_val.npz"
    if old_depth_path.is_file():
        old_depth = _read_npz(old_depth_path)
        root_metrics["old_joint_depth_root_reference"] = _root_metrics(
            np.asarray(old_depth["root_world"], dtype=np.float32), gt_val[:, 0]
        )
    write_json(output_root / "root_metrics.json", root_metrics)
    d0_scores = np.mean(np.asarray(map_val["features"][:, :, :JOINTS], dtype=np.float32), axis=2)
    d0_actions = _select(d0_scores, val_rows)
    d0_metric = _metric("D0 GT-human-state SceneVisibility", val_rows, _terminal(options_val, d0_actions), d0_actions)
    prior, prior_meta = _train_prior(train_rows, options_train)
    static_scores = np.full(options_val["ids"].shape, -np.inf, dtype=np.float32)
    for index, row in enumerate(val_rows):
        for slot in range(1, len(row["candidate_ids"]) + 1):
            static_scores[index, slot] = float(prior.get(int(row["candidate_ids"][slot - 1]), prior[-1]))
    static_actions = _select(static_scores, val_rows)
    static_metric = _metric("StaticPrior", val_rows, _terminal(options_val, static_actions), static_actions)
    old_ray = _read_npz(data_root / POLICY_RUNTIME / "raycast/moving_val.npz")
    old_scores = np.mean(np.asarray(old_ray["d1_visibility"], dtype=np.float32), axis=2)
    old_metric, old_actions, old_predictions = _evaluate_d1(val_rows, options_val, old_scores, "D1 old joint-depth root")
    raw_scores = _raycast_jobs(val_rows, gt_val, val_raw, runtime_root / "raycast_raw.npz", workers)
    cal_scores = _raycast_jobs(val_rows, gt_val, val_cal, runtime_root / "raycast_calibrated.npz", workers)
    raw_metric, raw_actions, raw_predictions = _evaluate_d1(val_rows, options_val, raw_scores, "D1 GTMask RawRoot")
    cal_metric, cal_actions, cal_predictions = _evaluate_d1(val_rows, options_val, cal_scores, "D1 GTMask CalibratedRoot")
    oracle_actions = _oracle_actions(options_val, val_rows, "true_logp")
    oracle_metric = _metric("Oracle GT-TrueLogP", val_rows, _terminal(options_val, oracle_actions), oracle_actions)
    d1_result = {"StaticPrior": static_metric, "D0 GT-human-state SceneVisibility": d0_metric, "D1 old joint-depth root": old_metric, "D1 GTMask RawRoot": raw_metric, "D1 GTMask CalibratedRoot": cal_metric, "Oracle GT-TrueLogP": oracle_metric, "prior_definition": prior_meta, "test_used": False}
    write_json(output_root / "d1_metrics.json", d1_result)
    d0_pred = _terminal(options_val, d0_actions)
    val_d0_current = d0_scores[:, 0]
    high_mask = val_d0_current < np.quantile(val_d0_current, 1.0 / 3.0)
    high_rows = [val_rows[int(index)] for index in np.flatnonzero(high_mask)]
    high_options = {key: value[high_mask] for key, value in options_val.items() if value.ndim > 0 and value.shape[0] == len(val_rows)}
    high_methods: dict[str, Any] = {}
    for name, actions in (("StaticPrior", static_actions), ("D0", d0_actions), ("Raw D1", raw_actions), ("Calibrated D1", cal_actions), ("Oracle", oracle_actions)):
        selected = [actions[int(index)] for index in np.flatnonzero(high_mask)]
        high_methods[name] = _metric(name, high_rows, _terminal(high_options, selected), selected)
    high_methods["count"] = int(high_mask.sum())
    high_methods["definition"] = "strict lower tertile of current-slot D0 SceneVisibility"
    high_methods["test_used"] = False
    write_json(output_root / "high_occlusion_metrics.json", high_methods)
    def agreement(actions: Sequence[int]) -> float:
        return float(np.mean(np.asarray(actions, dtype=np.int64) == np.asarray(d0_actions, dtype=np.int64)))
    pair = {
        "D0_vs_raw_d1_selected_view_agreement": agreement(raw_actions),
        "D0_vs_calibrated_d1_selected_view_agreement": agreement(cal_actions),
        "D0_correct_raw_d1_wrong": int(np.sum((d0_pred == np.asarray([int(r["label_id"]) for r in val_rows])) & (raw_predictions != np.asarray([int(r["label_id"]) for r in val_rows])))),
        "D0_wrong_raw_d1_correct": int(np.sum((d0_pred != np.asarray([int(r["label_id"]) for r in val_rows])) & (raw_predictions == np.asarray([int(r["label_id"]) for r in val_rows])))),
        "D0_correct_calibrated_d1_wrong": int(np.sum((d0_pred == np.asarray([int(r["label_id"]) for r in val_rows])) & (cal_predictions != np.asarray([int(r["label_id"]) for r in val_rows])))),
        "D0_wrong_calibrated_d1_correct": int(np.sum((d0_pred != np.asarray([int(r["label_id"]) for r in val_rows])) & (cal_predictions == np.asarray([int(r["label_id"]) for r in val_rows])))),
        "test_used": False,
    }
    methods = {key: value for key, value in d1_result.items() if isinstance(value, Mapping) and "accuracy" in value}
    raw_root_median = float(root_metrics["raw_root"]["euclidean_m"]["median"])
    cal_root_median = float(root_metrics["calibrated_root"]["euclidean_m"]["median"])
    cal_acc = float(cal_metric["accuracy"])
    if cal_acc >= 0.570:
        route = "RGB-D ROOT LOCALIZATION ROUTE STRONGLY REOPENED"
    elif cal_acc >= 0.560:
        route = "RGB-D ROOT LOCALIZATION ROUTE VIABLE"
    elif cal_acc >= 0.540 and cal_root_median <= 0.20:
        route = "RGB-D ROOT LOCALIZATION BORDERLINE"
    else:
        route = "KILL SIMPLE RGB-D ROOT LOCALIZATION"
    if cal_root_median <= 0.10 and float(root_metrics["calibrated_root"]["euclidean_m"]["p90"]) <= 0.30:
        localization = "GT-MASK DEPTH LOCALIZATION STRONG"
    elif cal_root_median <= 0.15 and float(root_metrics["calibrated_root"]["euclidean_m"]["p90"]) <= 0.50:
        localization = "GT-MASK DEPTH LOCALIZATION USABLE"
    elif cal_root_median <= 0.20:
        localization = "RGB-D ROOT LOCALIZATION BORDERLINE"
    else:
        localization = "KILL SIMPLE RGB-D ROOT LOCALIZATION"
    result: dict[str, Any] = {"experiment_id": "REDUCED12_GTMASK_DEPTH_ROOT_CEILING", "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "moving_val_candidate_samples": int(sum(len(row["candidate_ids"]) for row in val_rows))}, "methods": methods, "root_metrics": root_metrics, "selected_view_consistency": pair, "high_occlusion": high_methods, "classification": {"gtmask_depth_localization": localization, "rgbd_route": route, "D0_accuracy": float(d0_metric["accuracy"]), "best_gtmask_d1_accuracy": cal_acc, "D0_to_best_d1_drop_pp": float((d0_metric["accuracy"] - cal_acc) * 100.0)}, "render": {"train": train_render_meta, "moving_val": val_render_meta}, "flags": {"policy_test_used": False, "training_used": False, "gt_human_mask_used": True, "current_frame_only": True, "candidate_rgb_used": False, "candidate_depth_used": False, "future_frame_used": False, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "deployable": False}, "runtime": {"device": str(device), "workers": workers, "seed": SEED, "elapsed_seconds": time.monotonic() - started}}
    write_json(output_root / "result.json", result)
    _write_analysis(output_root / "analysis.md", result)
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    methods = result["methods"]
    raw = result["root_metrics"]["raw_root"]["euclidean_m"]
    cal = result["root_metrics"]["calibrated_root"]["euclidean_m"]
    old = result["root_metrics"].get("old_joint_depth_root_reference", {}).get("euclidean_m")
    lines = [
        "# GT Human Mask + Depth Root Recovery Ceiling Audit",
        "",
        "Policy Train was used only for the pelvis-height prior and one global radial calibration; Moving Val was evaluated with current frame 0 only. Policy Test, candidate RGB/depth, future frames, YOLO, VideoPose3D and DINO were not used.",
        "",
        "## Baselines and D1",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for name in ("StaticPrior", "D0 GT-human-state SceneVisibility", "D1 old joint-depth root", "D1 GTMask RawRoot", "D1 GTMask CalibratedRoot", "Oracle GT-TrueLogP"):
        value = methods[name]
        lines.append(f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} |")
    lines.extend([
        "",
        "## Root localization",
        "",
        f"GTMask RawRoot euclidean mean/median/P75/P90/P95: {raw['mean']:.6f}/{raw['median']:.6f}/{raw['p75']:.6f}/{raw['p90']:.6f}/{raw['p95']:.6f} m.",
        f"GTMask CalibratedRoot euclidean mean/median/P75/P90/P95: {cal['mean']:.6f}/{cal['median']:.6f}/{cal['p75']:.6f}/{cal['p90']:.6f}/{cal['p95']:.6f} m.",
        f"Existing old joint-depth root reference euclidean median/P90: {old['median']:.6f}/{old['p90']:.6f} m." if old else "Existing old joint-depth root reference was unavailable.",
        f"Valid GTMask point-cloud roots: Raw {int(raw['count'])}/{result['population']['moving_val_contexts']}; Calibrated {int(cal['count'])}/{result['population']['moving_val_contexts']}.",
        f"Train-derived pelvis height prior: {result['root_metrics']['pelvis_height_prior_m']:.6f} m; radial offset b: {result['root_metrics']['train_radial_calibration']['radial_offset_b_m']:.6f} m.",
        "",
        "The mask is Habitat semantic OBJECT_ID for the articulated humanoid and the point cloud uses all finite human-mask depth pixels. Raw depth/masks are transient and no point cloud is serialized.",
        "",
        "## Required interpretation",
        "",
        f"D0 accuracy is {methods['D0 GT-human-state SceneVisibility']['accuracy']:.6f}; old joint-depth D1 is {methods['D1 old joint-depth root']['accuracy']:.6f}; GTMask RawRoot D1 is {methods['D1 GTMask RawRoot']['accuracy']:.6f}; GTMask CalibratedRoot D1 is {methods['D1 GTMask CalibratedRoot']['accuracy']:.6f}.",
        f"D0 versus best GTMask D1 drop: {result['classification']['D0_to_best_d1_drop_pp']:.3f} pp. Selected-view agreement D0→Raw/Calibrated D1: {result['selected_view_consistency']['D0_vs_raw_d1_selected_view_agreement']:.6f}/{result['selected_view_consistency']['D0_vs_calibrated_d1_selected_view_agreement']:.6f}.",
        "",
        "1. The comparison is against the independent GT H36M17 pelvis, not estimated-vs-estimated self-consistency.",
        "2. A global Train-only radial offset is reported; no scene-, action-, distance-bin- or Val-derived calibration is used.",
        "3. If GTMask D1 remains far below D0, perfect human segmentation does not by itself make current depth sufficient for known-map NBV; the simple RGB-D root route should be stopped rather than expanded with more point-cloud heuristics.",
        f"4. Localization classification: **{result['classification']['gtmask_depth_localization']}**; route classification: **{result['classification']['rgbd_route']}**.",
        "",
        "## Direct answers",
        "",
        f"1. Root median/P90 changes from the old {old['median']:.6f}/{old['p90']:.6f} m to GTMask calibrated {cal['median']:.6f}/{cal['p90']:.6f} m (raw {raw['median']:.6f}/{raw['p90']:.6f} m).",
        f"2. The global Train-only radial calibration is helpful for the median and horizontal error (calibrated radial offset b={result['root_metrics']['train_radial_calibration']['radial_offset_b_m']:.6f} m), but it does not materially reduce the long tail.",
        f"3. D1 recovers from {methods['D1 old joint-depth root']['accuracy']:.6f} to {methods['D1 GTMask CalibratedRoot']['accuracy']:.6f} Accuracy; it remains {result['classification']['D0_to_best_d1_drop_pp']:.3f} pp below D0.",
        "4. The improvement over joint-depth D1 supports pixel-depth association as a major contributor to the earlier failure, but the remaining D0 gap shows it is not the only bottleneck.",
        "5. With a perfect human mask, depth reaches 55.655% D1 Accuracy rather than the 56–58% D0 range; this is insufficient to claim that depth alone supports the known-map NBV route.",
        "6. A deployable human-segmentation follow-up is not justified by this ceiling audit; stop the simple RGB-D localization route before adding segmentation engineering.",
        "",
        "## High occlusion",
        "",
        f"The fixed strict lower tertile of current-slot D0 visibility contains {result['high_occlusion']['count']} contexts. See `high_occlusion_metrics.json` for StaticPrior, D0, Raw D1, Calibrated D1 and Oracle.",
        "",
        "No next-stage deployable mask work was started automatically.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    result = run(args.output.resolve(), get_data_root(), require_cuda(args.device), args.workers)
    print(json.dumps({"status": result["status"], "population": result["population"], "methods": {name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]} for name, value in result["methods"].items()}, "classification": result["classification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
