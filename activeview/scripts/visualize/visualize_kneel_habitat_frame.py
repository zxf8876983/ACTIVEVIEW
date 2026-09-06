#!/usr/bin/env python3
"""Render one canonical frame of one kneel motion in a real Habitat scene.

This is a read-only visualization helper.  It deliberately does not invoke
the RGB perception stack (YOLO, VideoPose3D or ST-GCN); it restores the
frame-15 Habitat humanoid pose and saves the raw color observation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Tuple

import habitat_sim
import numpy as np
import quaternion

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.data.motion.babel_clean_dataset_generator import (
    _load_resampled_motion,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.data.motion.humanoid_grounding import humanoid_geometry_y_bounds, select_floor_height
from activeview.data.motion.motion_converter import MotionConverter

LOGGER = logging.getLogger(__name__)
URDF_PATH = get_humanoid_urdf_path("male_0")


def _load_record(manifest: Path, label: str) -> Mapping[str, Any]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {manifest}")
    for row in rows:
        if str(row.get("action_label", "")).strip().lower() == label.strip().lower():
            return row
    raise ValueError(f"No {label!r} record found in {manifest}")


def _scene_sim(scene_root: Path, scene_id: str, image_size: int) -> Tuple[Any, Any]:
    scene_dir = scene_root / scene_id
    glbs = sorted(scene_dir.glob("*.basis.glb"))
    navmeshes = sorted(scene_dir.glob("*.basis.navmesh"))
    if not glbs or not navmeshes:
        raise FileNotFoundError(f"Habitat scene/navmesh not found under {scene_dir}")
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glbs[0])
    backend.enable_physics = True
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color_sensor"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [image_size, image_size]
    sensor.position = [0.0, 1.10, 0.0]
    sensor.hfov = 75.0
    agent = habitat_sim.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    if not sim.pathfinder.load_nav_mesh(str(navmeshes[0])):
        sim.close()
        raise RuntimeError(f"Unable to load navmesh for {scene_id}")
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(URDF_PATH))
    return sim, human


def _sample_placement(sim: Any, seed: int) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    for _ in range(300):
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
        if np.isfinite(point).all() and sim.pathfinder.distance_to_closest_obstacle(point) >= 0.8:
            ray = habitat_sim.geo.Ray(
                point + np.array([0.0, 3.0, 0.0], dtype=np.float32),
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
            )
            floor = select_floor_height(sim.cast_ray(ray).hits, reference_y=float(point[1]))
            return np.array([point[0], floor, point[2]], dtype=np.float32), float(rng.uniform(0.0, 360.0))
    raise RuntimeError("Could not find a clear navigable placement")


def _camera_state(
    base: np.ndarray,
    floor_y: float,
    yaw_deg: float,
    target_y: float,
    distance: float = 2.8,
) -> habitat_sim.AgentState:
    yaw = math.radians(yaw_deg)
    position = np.array(
        [base[0] + distance * math.sin(yaw), floor_y, base[2] + distance * math.cos(yaw)],
        dtype=np.float32,
    )
    sensor_position = position + np.array([0.0, 1.10, 0.0], dtype=np.float32)
    target = np.array([base[0], target_y, base[2]], dtype=np.float32)
    direction = target - sensor_position
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    cam_yaw = math.atan2(-float(direction[0]), -float(direction[2]))
    cam_pitch = math.asin(float(direction[1]))
    rotation = quaternion.from_rotation_vector([0.0, cam_yaw, 0.0]) * quaternion.from_rotation_vector(
        [cam_pitch, 0.0, 0.0]
    )
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = rotation
    return state


def main() -> None:
    data_root = get_data_root()
    default_manifest = (
        data_root
        / "datasets/reduced15_replacement_babel_diversity_v1/activeview_official_val/val.json"
    )
    default_output = Path("/home/zxf/.codex/visualizations/activeview_kneel_habitat")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-minival")
    parser.add_argument("--scene-id", default="00800-TEEsavR23oF")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--frame-index", type=int, default=15)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="kneel")
    args = parser.parse_args()
    if not 0 <= args.frame_index < args.target_frames:
        raise ValueError("frame-index must be within target-frames")
    record = _load_record(args.manifest, args.label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converter = MotionConverter(URDF_PATH)
    motion = _load_resampled_motion(record, args.target_frames)
    converted = converter.convert(motion)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)

    sim, human = _scene_sim(args.scene_root, args.scene_id, args.image_size)
    placement, scene_yaw = _sample_placement(sim, args.seed)
    try:
        offsets, centers = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=scene_yaw)
        idx = args.frame_index
        apply_humanoid_pose(
            human,
            joints[idx],
            roots[idx],
            base_position=placement,
            scene_yaw_deg=scene_yaw,
            floor_y=float(placement[1]),
            grounding_offset=float(offsets[idx]),
        )
        geometry_min_y, geometry_max_y = humanoid_geometry_y_bounds(human, URDF_PATH)
        floor_alignment_error = float(geometry_min_y - float(placement[1]))
        if abs(floor_alignment_error) > 1e-4:
            raise RuntimeError(
                "Grounding check failed: "
                f"human_min_y={geometry_min_y:.8f}, floor_y={float(placement[1]):.8f}"
            )
        state = _camera_state(
            placement,
            float(placement[1]),
            scene_yaw + 180.0,
            float(placement[1] + centers[idx] + offsets[idx]),
        )
        sim.get_agent(0).set_state(state)
        observation = np.asarray(sim.get_sensor_observations()["color_sensor"][:, :, :3], dtype=np.uint8)
        if observation.shape != (args.image_size, args.image_size, 3):
            raise ValueError(f"Unexpected RGB shape: {observation.shape}")
        if not np.any(observation):
            raise ValueError("Rendered RGB frame is all zero")
        safe_label = args.label.strip().lower().replace(" ", "_").replace("/", "_")
        output_path = args.output_dir / f"{safe_label}_frame{idx}.png"
        from PIL import Image

        Image.fromarray(observation, mode="RGB").save(output_path)
        metadata = {
            "scene_id": args.scene_id,
            "scene_root": str(args.scene_root.resolve()),
            "record_id": str(record["record_id"]),
            "action_label": str(record["action_label"]),
            "source_path": str(record["source_path"]),
            "frame_index": int(idx),
            "target_frames": int(args.target_frames),
            "placement": placement.tolist(),
            "scene_yaw_deg": float(scene_yaw),
            "floor_y": float(placement[1]),
            "human_geometry_min_y": float(geometry_min_y),
            "human_geometry_max_y": float(geometry_max_y),
            "floor_alignment_error_m": floor_alignment_error,
            "image_shape": list(observation.shape),
            "dtype": str(observation.dtype),
            "perception_used": False,
            "test_used": False,
            "output_path": str(output_path),
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOGGER.info("saved %s", output_path)
        LOGGER.info("scene=%s record=%s placement=%s", args.scene_id, record["record_id"], placement.tolist())
        LOGGER.info("grounding check: min_y-floor_y=%.3e m", floor_alignment_error)
    finally:
        sim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
