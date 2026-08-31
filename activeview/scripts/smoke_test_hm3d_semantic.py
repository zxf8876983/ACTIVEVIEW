#!/usr/bin/env python3
"""Run a bounded frame-15 Habitat semantic/depth smoke test.

The test reads one immutable skeleton record and never writes dataset files.
It is intentionally a gate for EXP029; a failed check exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.dataset.babel_clean_dataset_generator import (
    URDF_PATH,
    MotionConverter,
    _load_resampled_motion,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.scripts.generate_hm3d_train_rgb_observations import (
    FRAME_INDEX,
    TARGET_FRAMES,
    _load_skeleton_metadata,
    _load_source_records,
    _set_agent_state,
)
from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.scripts.repair_hm3d_semantic_mapping import discover_assets


IMAGE_SIZE = 256
SENSOR_HEIGHT_M = 1.1
HFOV_DEG = 75.0


def _simulator(habitat_root: Path, scene_id: str, config_path: Path) -> tuple[Any, Any]:
    import habitat_sim
    import magnum as mn

    scene_dir = habitat_root / "hm3d-train" / scene_id
    basis = next(scene_dir.glob("*.basis.glb"))
    navmesh = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(basis)
    backend.scene_dataset_config_file = str(config_path)
    backend.enable_physics = True
    backend.load_semantic_mesh = True
    backend.force_separate_semantic_scene_graph = True
    # HM3D annotated semantic meshes carry the labels; texture IDs are not
    # populated for these assets and would produce an all-background image.
    backend.use_semantic_textures = False
    sensors = []
    for uuid, sensor_type in (("color", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH), ("semantic", habitat_sim.SensorType.SEMANTIC)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [IMAGE_SIZE, IMAGE_SIZE]
        spec.position = mn.Vector3(0.0, SENSOR_HEIGHT_M, 0.0)
        spec.hfov = HFOV_DEG
        sensors.append(spec)
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    sim.pathfinder.load_nav_mesh(str(navmesh))
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    return sim, human


def _category_for_id(objects: list[Any], semantic_id: int) -> tuple[str, str]:
    # Habitat semantic object IDs are strings such as ``chair_17`` while the
    # rendered integer is the trailing object index.
    for obj in objects:
        object_id = str(getattr(obj, "id", ""))
        if object_id == str(semantic_id) or object_id.rsplit("_", 1)[-1] == str(semantic_id):
            category = getattr(obj, "category", None)
            name = category.name() if category is not None else ""
            return object_id, str(name)
    raise RuntimeError(f"No semantic descriptor/category mapping for ID {semantic_id}")


_SEMANTIC_CHANNELS = {
    "wall": 4,
    "table": 5,
    "chair": 6,
    "bed": 7,
    "couch": 8,
    "cabinet": 9,
    "other_object": 10,
}


def _normalize_category_name(name: str) -> str:
    """Map Habitat category labels to the fixed EXP029 channels."""
    value = " ".join(str(name).strip().lower().replace("_", " ").split())
    if value == "sofa":
        return "couch"
    if value == "kitchen cabinet":
        return "cabinet"
    return value if value in _SEMANTIC_CHANNELS else "other_object"


def _semantic_channel_map(objects: list[Any]) -> dict[int, int]:
    """Build rendered semantic-id -> EXP029 BEV channel mapping."""
    mapping: dict[int, int] = {}
    for obj in objects:
        object_id = str(getattr(obj, "id", ""))
        suffix = object_id.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            continue
        category = getattr(obj, "category", None)
        name = category.name() if category is not None else ""
        mapping[int(suffix)] = _SEMANTIC_CHANNELS[_normalize_category_name(str(name))]
    return mapping


def _rotation_matrix(rotation_wxyz: np.ndarray) -> np.ndarray:
    import quaternion

    return np.asarray(quaternion.as_rotation_matrix(quaternion.from_float_array(rotation_wxyz)), dtype=np.float64)


def _world_to_s1(world: np.ndarray, s1_position: np.ndarray, s1_rotation_wxyz: np.ndarray) -> np.ndarray:
    return _rotation_matrix(s1_rotation_wxyz).T @ (np.asarray(world, dtype=np.float64) - np.asarray(s1_position, dtype=np.float64))


def _bev_cell(egocentric_xyz: np.ndarray) -> tuple[int, int] | None:
    x, z = float(egocentric_xyz[0]), float(egocentric_xyz[2])
    col, row = int(math.floor((x + 4.0) / 0.1)), int(math.floor((z + 4.0) / 0.1))
    return (row, col) if 0 <= row < 80 and 0 <= col < 80 else None


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    points: list[tuple[int, int]] = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err = dc - dr
    while True:
        points.append((r0, c0))
        if (r0, c0) == (r1, c1):
            break
        twice = 2 * err
        if twice > -dr:
            err -= dr
            c0 += sc
        if twice < dc:
            err += dc
            r0 += sr
    return points


def _project_observation_to_bev(
    depth: np.ndarray,
    semantic: np.ndarray,
    *,
    render_camera: Any,
    s1_position: np.ndarray,
    s1_rotation_wxyz: np.ndarray,
    semantic_channels: dict[int, int] | None = None,
) -> np.ndarray:
    """Project Habitat-unprojected depth rays through world/s1 frames into BEV."""
    import magnum as mn

    bev = np.zeros((15, 80, 80), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0.0)
    center_ray = render_camera.unproject(mn.Vector2i(IMAGE_SIZE // 2, IMAGE_SIZE // 2), normalized=False)
    camera_world = np.asarray(center_ray.origin, dtype=np.float64)
    camera_s1 = _world_to_s1(camera_world, s1_position, s1_rotation_wxyz)
    camera_cell = _bev_cell(camera_s1)
    if camera_cell is None:
        raise RuntimeError("Camera origin is outside the 8m s1-centered BEV")
    rows, cols = np.where(valid[::8, ::8])
    occupied_cells: set[tuple[int, int]] = set()
    semantic_cells: set[tuple[int, int]] = set()
    semantic_endpoint_channels: dict[tuple[int, int], int] = {}
    free_cells: set[tuple[int, int]] = set()
    for y, x in zip(rows * 8, cols * 8):
        distance = float(depth[y, x])
        ray = render_camera.unproject(mn.Vector2i(int(x), int(y)), normalized=False)
        endpoint_world = np.asarray(ray.origin, dtype=np.float64) + np.asarray(ray.direction, dtype=np.float64) * distance
        endpoint_s1 = _world_to_s1(endpoint_world, s1_position, s1_rotation_wxyz)
        endpoint_cell = _bev_cell(endpoint_s1)
        if endpoint_cell is None:
            continue
        ray = _bresenham(camera_cell, endpoint_cell)
        free_cells.update(ray[:-1])
        semantic_id = int(semantic[y, x])
        if semantic_id > 0:
            occupied_cells.add(endpoint_cell)
            semantic_cells.add(endpoint_cell)
            semantic_endpoint_channels[endpoint_cell] = int(
                (semantic_channels or {}).get(semantic_id, 10)
            )
        else:
            free_cells.add(endpoint_cell)
    free_cells.difference_update(occupied_cells)
    for row, col in free_cells | occupied_cells:
        bev[0, row, col] = 1  # observed ray cells and endpoint
    for row, col in free_cells:
        bev[2, row, col] = 1  # free/traversed cells
    for row, col in occupied_cells:
        bev[1, row, col] = 1  # endpoint surface
        if (row, col) in semantic_cells:
            channel = semantic_endpoint_channels.get((row, col), 10)
            bev[channel, row, col] = 1
        bev[2, row, col] = 0
    return bev


def _add_markers(
    bev: np.ndarray,
    positions: dict[str, np.ndarray],
    *,
    s1_position: np.ndarray,
    s1_rotation_wxyz: np.ndarray,
) -> dict[str, tuple[int, int]]:
    channels = {"s0": 11, "s1": 12, "p2": 13, "p3": 14}
    cells: dict[str, tuple[int, int]] = {}
    for name, position in positions.items():
        cell = _bev_cell(_world_to_s1(position, s1_position, s1_rotation_wxyz))
        if cell is None:
            raise RuntimeError(f"{name} marker is outside the 8m s1-centered BEV")
        cells[name] = cell
        row, col = cell
        bev[channels[name], row, col] = 1
    return cells


def _save_debug_bev(bev: np.ndarray, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for the bounded BEV debug image") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 80, 3), dtype=np.float32)
    image[:, :, 0] = bev[1]  # occupied: red
    image[:, :, 1] = bev[2]  # free: green
    image[:, :, 2] = bev[0] * 0.25  # observed: blue tint
    plt.imsave(path, image, vmin=0.0, vmax=1.0)


def _check_unprojection(
    depth: np.ndarray,
    *,
    render_camera: Any,
    agent_position: np.ndarray,
    agent_rotation_wxyz: np.ndarray,
) -> float:
    """Check Habitat's ray against the -Z pinhole convention at center pixel."""
    import magnum as mn

    cx = cy = IMAGE_SIZE // 2
    distance = float(depth[cy, cx])
    if not np.isfinite(distance) or distance <= 0.0:
        raise RuntimeError("Center depth is not finite and positive")
    ray = render_camera.unproject(mn.Vector2i(cx, cy), normalized=False)
    origin = np.asarray(ray.origin, dtype=np.float64)
    direction = np.asarray(ray.direction, dtype=np.float64)
    rotation = _rotation_matrix(agent_rotation_wxyz)
    forward_world = rotation @ np.array([0.0, 0.0, -1.0])
    endpoint = origin + direction * distance
    if float(np.dot(endpoint - origin, forward_world)) <= 0.0 or float(direction[2]) >= 0.0:
        raise RuntimeError(f"Center depth endpoint is not on camera -Z forward side: direction={direction}")
    fx = fy = IMAGE_SIZE / (2.0 * np.tan(np.deg2rad(HFOV_DEG) / 2.0))
    manual_camera = np.array([(cx - IMAGE_SIZE / 2.0) * distance / fx, (IMAGE_SIZE / 2.0 - cy) * distance / fy, -distance])
    manual_endpoint = np.asarray(agent_position, dtype=np.float64) + rotation @ (np.array([0.0, SENSOR_HEIGHT_M, 0.0]) + manual_camera)
    error = float(np.linalg.norm(endpoint - manual_endpoint))
    # Pixel-center conventions differ by at most half a pixel; at this
    # distance the expected numerical discrepancy is below two centimetres.
    if error > 2e-2:
        raise RuntimeError(f"Habitat unproject/manual -Z mismatch: error={error}")
    return error


def run_smoke(scene_id: str | None = None) -> dict[str, Any]:
    habitat_root, skeleton_root, _, config_path, assets = discover_assets()
    selected_scene = scene_id or assets[0].scene_id
    if selected_scene not in {row.scene_id for row in assets}:
        raise ValueError(f"Scene is not canonical: {selected_scene}")
    data_root = get_data_root() / "datasets" / "policy_v11_5"
    cache_summary = data_root / "stage_d" / "EXP014_two_step_sequential" / "stage_d_feature_summary.json"
    if not cache_summary.is_file():
        raise FileNotFoundError(f"Missing frozen Stage-D cache summary: {cache_summary}")
    cache = json.loads(cache_summary.read_text(encoding="utf-8"))
    train_features = Path(cache["feature_files"]["train"])
    feature_rows = [row for row in load_jsonl(train_features) if str(row.get("scene_id")) == selected_scene and len(row.get("remaining_candidate_ids", [])) >= 2]
    if not feature_rows:
        raise RuntimeError(f"No Train Stage-D eligible episode with p2/p3 in {selected_scene}")
    feature = feature_rows[0]
    stage_a_path = Path(json.loads((data_root / "stage_a_summary.json").read_text(encoding="utf-8"))["episode_files"]["train"])
    stage_a = next((row for row in load_jsonl(stage_a_path) if str(row["episode_id"]) == str(feature["episode_id"])), None)
    if stage_a is None or str(stage_a.get("policy_split")) != "train":
        raise RuntimeError(f"Stage-A Train episode not aligned: {feature['episode_id']}")
    motion_root = get_data_root() / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"
    record = None
    for motion_split in ("train", "val"):
        motion_path = motion_root / f"{motion_split}.json"
        motion_ids = {str(item["record_id"]) for item in json.loads(motion_path.read_text(encoding="utf-8"))}
        if str(feature["record_id"]) in motion_ids:
            candidates, _ = _load_source_records(skeleton_root, motion_path, [selected_scene])
            record = next((item for item in candidates if item.record_id == str(feature["record_id"]) and item.region == str(feature["region"])), None)
            break
    if record is None:
        raise RuntimeError(f"Unable to resolve motion record for {feature['record_id']}")
    metadata = _load_skeleton_metadata(record.source_path)
    pool = {int(item["viewpoint_id"]): item for item in stage_a["candidate_pool"]}
    expected_ids = {"s0": int(feature["s0_viewpoint_id"]), "s1": int(feature["s1_viewpoint_id"]), "p2": int(feature["remaining_candidate_ids"][0]), "p3": int(feature["remaining_candidate_ids"][1])}
    expected_episode_positions = {"s0": stage_a["current_view"]["agent_position"], **{name: pool[viewpoint_id]["position"] for name, viewpoint_id in expected_ids.items() if name != "s0"}}
    for name, viewpoint_id in expected_ids.items():
        if not np.allclose(metadata["viewpoint_agent_positions"][viewpoint_id], np.asarray(expected_episode_positions[name], dtype=np.float32), rtol=0.0, atol=1e-5):
            raise RuntimeError(f"{name} episode/NPZ camera position mismatch for viewpoint {viewpoint_id}")
    sim, human = _simulator(habitat_root, selected_scene, config_path)
    try:
        motion = _load_resampled_motion(record.motion, TARGET_FRAMES)
        converted = MotionConverter(URDF_PATH).convert(motion)
        joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
        roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
        offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=0.0)
        base = np.asarray(metadata["placement_position"], dtype=np.float32)
        apply_humanoid_pose(human, joints[FRAME_INDEX], roots[FRAME_INDEX], base_position=base, scene_yaw_deg=0.0, floor_y=float(base[1]), grounding_offset=float(offsets[FRAME_INDEX]))
        viewpoint_ids = {"s0": int(feature["s0_viewpoint_id"]), "s1": int(feature["s1_viewpoint_id"]), "p2": int(feature["remaining_candidate_ids"][0]), "p3": int(feature["remaining_candidate_ids"][1])}
        observations_by_view: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        unprojection_error = None
        for name in ("s0", "s1"):
            viewpoint_id = viewpoint_ids[name]
            _set_agent_state(sim.get_agent(0), metadata["viewpoint_agent_positions"][viewpoint_id], metadata["viewpoint_rotations_wxyz"][viewpoint_id])
            observation = sim.get_sensor_observations([0])[0]
            observations_by_view[name] = (np.asarray(observation["depth"]), np.asarray(observation["semantic"]))
            if name == "s1":
                unprojection_error = _check_unprojection(
                    observations_by_view[name][0],
                    render_camera=sim._sensors["depth"]._sensor_object.render_camera,
                    agent_position=metadata["viewpoint_agent_positions"][viewpoint_id],
                    agent_rotation_wxyz=metadata["viewpoint_rotations_wxyz"][viewpoint_id],
                )
        depth, semantic = observations_by_view["s1"]
        if sim.semantic_scene is None or len(sim.semantic_scene.objects) <= 0:
            raise RuntimeError("semantic_scene is empty")
        if depth.shape != (IMAGE_SIZE, IMAGE_SIZE) or semantic.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Unexpected sensor shapes: depth={depth.shape}, semantic={semantic.shape}")
        if not np.issubdtype(semantic.dtype, np.integer):
            raise RuntimeError(f"Semantic sensor dtype is not integer: {semantic.dtype}")
        unique = np.unique(semantic)
        nonzero = unique[unique > 0]
        if unique.size <= 1 or nonzero.size == 0 or int(np.count_nonzero(semantic)) == 0:
            raise RuntimeError(f"Semantic image is empty: unique={unique.tolist()}")
        mappings = [_category_for_id(sim.semantic_scene.objects, int(value)) for value in nonzero[:10]]
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        if float(valid_depth.mean()) <= 0.0:
            raise RuntimeError("Depth has no finite positive pixels")
        s1_id = viewpoint_ids["s1"]
        semantic_channels = _semantic_channel_map(sim.semantic_scene.objects)
        bev = _project_observation_to_bev(depth, semantic, render_camera=sim._sensors["depth"]._sensor_object.render_camera, s1_position=metadata["viewpoint_agent_positions"][s1_id], s1_rotation_wxyz=metadata["viewpoint_rotations_wxyz"][s1_id], semantic_channels=semantic_channels)
        human_world = np.asarray(human.translation, dtype=np.float64)
        marker_world = {
            "s0": metadata["viewpoint_agent_positions"][viewpoint_ids["s0"]],
            "s1": metadata["viewpoint_agent_positions"][s1_id],
            "p2": metadata["viewpoint_agent_positions"][viewpoint_ids["p2"]],
            "p3": metadata["viewpoint_agent_positions"][viewpoint_ids["p3"]],
        }
        marker_cells = _add_markers(bev, {key: np.asarray(value, dtype=np.float64) for key, value in marker_world.items()}, s1_position=metadata["viewpoint_agent_positions"][s1_id], s1_rotation_wxyz=metadata["viewpoint_rotations_wxyz"][s1_id])
        expected_s1 = _bev_cell(np.zeros(3, dtype=np.float64))
        if marker_cells["s1"] != expected_s1:
            raise RuntimeError(f"s1 marker transform mismatch: {marker_cells['s1']} != {expected_s1}")
        human_cell = _bev_cell(_world_to_s1(human_world, metadata["viewpoint_agent_positions"][s1_id], metadata["viewpoint_rotations_wxyz"][s1_id]))
        if human_cell is None:
            raise RuntimeError("Human marker is outside the 8m s1-centered BEV")
        bev[3, human_cell[0], human_cell[1]] = 1
        if bev.shape != (15, 80, 80) or int(bev[0].sum()) <= 0 or int(bev[1].sum()) <= 0 or int(bev[2].sum()) <= 0:
            raise RuntimeError(f"BEV smoke failed: shape={bev.shape}, observed={int(bev[0].sum())}, occupied={int(bev[1].sum())}, free={int(bev[2].sum())}")
        if np.any((bev[1] > 0) & (bev[2] > 0)):
            raise RuntimeError("BEV occupied/free cell conflict")
        debug_path = data_root / "experiments" / "stage_d" / "EXP029_observed_local_semantic_bev" / "runtime" / "smoke_bev_debug.png"
        _save_debug_bev(bev, debug_path)
        result = {
            "scene_id": selected_scene,
            "record_id": record.record_id,
            "episode_id": feature["episode_id"],
            "policy_split": feature["policy_split"],
            "viewpoint_ids": viewpoint_ids,
            "viewpoint_id": 0,
            "frame_index": FRAME_INDEX,
            "semantic_scene_objects": len(sim.semantic_scene.objects),
            "semantic_shape": list(semantic.shape),
            "semantic_dtype": str(semantic.dtype),
            "semantic_unique_count": int(unique.size),
            "semantic_nonzero_ratio": float(np.count_nonzero(semantic) / semantic.size),
            "semantic_min_id": int(unique.min()),
            "semantic_max_id": int(unique.max()),
            "semantic_mappings": [{"semantic_id": int(value), "object_id": obj, "category": cat} for value, (obj, cat) in zip(nonzero[:10], mappings)],
            "depth_shape": list(depth.shape),
            "depth_valid_ratio": float(valid_depth.mean()),
            "unprojection_manual_error_m": unprojection_error,
            "center_ray_forward_convention": "Habitat camera -Z",
            "center_ray_negative_z_sanity": True,
            "bev_shape": list(bev.shape),
            "bev_observed_cells": int(bev[0].sum()),
            "bev_occupied_cells": int(bev[1].sum()),
            "bev_free_cells": int(bev[2].sum()),
            "bev_semantic_cells": int(bev[4:11].sum()),
            "marker_cells_s1_centered": {key: list(value) for key, value in marker_cells.items()},
            "human_world_position": human_world.tolist(),
            "human_bev_cell": list(human_cell) if human_cell is not None else None,
            "debug_bev_path": str(debug_path.resolve()),
            "scene_dataset_config": str(config_path),
            "semantic_smoke_test": "PASS",
        }
        print(json.dumps(result, indent=2))
        print("SEMANTIC_SMOKE_TEST=PASS")
        return result
    finally:
        sim.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id")
    args = parser.parse_args()
    try:
        run_smoke(args.scene_id)
    except Exception as exc:  # smoke gate must emit an unambiguous failure marker
        print(json.dumps({"semantic_smoke_test": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
        print("SEMANTIC_SMOKE_TEST=FAIL")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
