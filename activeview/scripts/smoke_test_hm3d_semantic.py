#!/usr/bin/env python3
"""Run a bounded frame-15 Habitat semantic/depth smoke test.

The test reads one immutable skeleton record and never writes dataset files.
It is intentionally a gate for EXP029; a failed check exits non-zero.
"""

from __future__ import annotations

import argparse
import json
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


def _bev_smoke(depth: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Project sparse aligned pixels into the fixed 8 m / 0.1 m BEV."""
    bev = np.zeros((15, 80, 80), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.where(valid[::8, ::8])
    if ys.size:
        sample_y, sample_x = ys * 8, xs * 8
        d = depth[sample_y, sample_x]
        # Camera-forward z and lateral x, centered in an 8m square.
        fx = fy = IMAGE_SIZE / (2.0 * np.tan(np.deg2rad(HFOV_DEG) / 2.0))
        x = (sample_x - IMAGE_SIZE / 2.0) * d / fx
        z = d
        gx = np.floor((x + 4.0) / 0.1).astype(int)
        gz = np.floor((z + 4.0) / 0.1).astype(int)
        keep = (gx >= 0) & (gx < 80) & (gz >= 0) & (gz < 80)
        bev[0, gz[keep], gx[keep]] = 1
        semantic_sample = semantic[sample_y, sample_x]
        object_keep = keep & (semantic_sample > 0)
        bev[1, gz[object_keep], gx[object_keep]] = 1
        bev[2, gz[object_keep], gx[object_keep]] = 1
    return bev


def run_smoke(scene_id: str | None = None) -> dict[str, Any]:
    habitat_root, skeleton_root, _, config_path, assets = discover_assets()
    selected_scene = scene_id or assets[0].scene_id
    if selected_scene not in {row.scene_id for row in assets}:
        raise ValueError(f"Scene is not canonical: {selected_scene}")
    motion_manifest = get_data_root() / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed" / "val.json"
    records, _ = _load_source_records(skeleton_root, motion_manifest, [selected_scene])
    if not records:
        raise RuntimeError(f"No canonical skeleton record for {selected_scene}")
    record = records[0]
    metadata = _load_skeleton_metadata(record.source_path)
    sim, human = _simulator(habitat_root, selected_scene, config_path)
    try:
        motion = _load_resampled_motion(record.motion, TARGET_FRAMES)
        converted = MotionConverter(URDF_PATH).convert(motion)
        joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
        roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
        offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=0.0)
        base = np.asarray(metadata["placement_position"], dtype=np.float32)
        apply_humanoid_pose(human, joints[FRAME_INDEX], roots[FRAME_INDEX], base_position=base, scene_yaw_deg=0.0, floor_y=float(base[1]), grounding_offset=float(offsets[FRAME_INDEX]))
        _set_agent_state(sim.get_agent(0), metadata["viewpoint_agent_positions"][0], metadata["viewpoint_rotations_wxyz"][0])
        observation = sim.get_sensor_observations([0])[0]
        depth = np.asarray(observation["depth"])
        semantic = np.asarray(observation["semantic"])
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
        bev = _bev_smoke(depth, semantic)
        if bev.shape != (15, 80, 80) or int(bev[0].sum()) <= 0 or int(bev[2].sum()) <= 0:
            raise RuntimeError(f"BEV smoke failed: shape={bev.shape}, observed={int(bev[0].sum())}, semantic={int(bev[2].sum())}")
        result = {
            "scene_id": selected_scene,
            "record_id": record.record_id,
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
            "bev_shape": list(bev.shape),
            "bev_observed_cells": int(bev[0].sum()),
            "bev_semantic_cells": int(bev[2].sum()),
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
