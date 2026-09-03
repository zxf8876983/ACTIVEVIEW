#!/usr/bin/env python3
"""文件用途：
    生成场景或观测可视化。

主要输入：
    - 离线场景记录与渲染结果。
主要输出：
    - 调试图像或视频。
项目角色：
    - 属于只读可视化入口。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Dict, Mapping

import habitat_sim
import numpy as np
import quaternion

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import (
    get_data_root,
    get_habitat_data_root,
    get_humanoid_urdf_path,
)
from activeview.data.motion.babel_clean_dataset_generator import (
    BabelCleanDatasetGenerator,
    _load_resampled_motion,
    apply_humanoid_pose,
    compose_root_rotation,
    precompute_grounding_offsets,
    transform_camera_sequence_to_gravity,
)
from activeview.data.motion.humanoid_grounding import select_floor_height
from activeview.perception.skeleton import get_skeleton_definition
from activeview.scripts.visualize.visualize_household_rgb_to_3d_pipeline import (
    _extract_2d_keypoints,
    _render_side_by_side,
    _load_one_record_per_label,
)
LOGGER = logging.getLogger(__name__)
DEFAULT_SCENE_ROOT = get_habitat_data_root() / "hm3d-minival"
URDF_PATH = get_humanoid_urdf_path("male_0")


def _scene_sim(scene_root: Path, scene_id: str, image_size: int):
    scene_dir = scene_root / scene_id
    glb = next(scene_dir.glob("*.basis.glb"))
    navmesh = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
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
    if not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError(f"Unable to load navmesh for {scene_id}")
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(URDF_PATH))
    return sim, human


def _sample_placement(sim, seed: int = 11):
    rng = np.random.default_rng(seed)
    for _ in range(300):
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
        if np.isfinite(point).all() and sim.pathfinder.distance_to_closest_obstacle(point) >= 0.8:
            ray = habitat_sim.geo.Ray(point + np.array([0, 3, 0], dtype=np.float32), np.array([0, -1, 0], dtype=np.float32))
            floor = select_floor_height(sim.cast_ray(ray).hits, reference_y=float(point[1]))
            return np.array([point[0], floor, point[2]], dtype=np.float32), float(rng.uniform(0, 360))
    raise RuntimeError("Could not find a clear navigable placement")


def _camera_state(base: np.ndarray, floor_y: float, yaw_deg: float, target_y: float, distance: float = 2.8):
    yaw = math.radians(yaw_deg)
    position = np.array([base[0] + distance * math.sin(yaw), floor_y, base[2] + distance * math.cos(yaw)], dtype=np.float32)
    sensor_position = position + np.array([0.0, 1.10, 0.0], dtype=np.float32)
    target = np.array([base[0], target_y, base[2]], dtype=np.float32)
    direction = target - sensor_position
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    cam_yaw = math.atan2(-float(direction[0]), -float(direction[2]))
    cam_pitch = math.asin(float(direction[1]))
    rotation = quaternion.from_rotation_vector([0.0, cam_yaw, 0.0]) * quaternion.from_rotation_vector([cam_pitch, 0.0, 0.0])
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = rotation
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = quaternion.as_rotation_matrix(rotation).astype(np.float32)
    c2w[:3, 3] = sensor_position
    return state, c2w


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/val.json")
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene-id", default="00800-TEEsavR23oF")
    parser.add_argument("--output-dir", type=Path, default=data_root / "visualizations/selected16_yolo26_real_hm3d_00800_rgb_3d")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected: Dict[str, Mapping[str, object]] = _load_one_record_per_label(args.manifest)
    if args.labels:
        selected = {label: selected[label] for label in args.labels}
    generator = BabelCleanDatasetGenerator(output_root=args.output_dir / "cache", image_size=args.image_size, target_frames=args.target_frames, camera_height=1.2, device=args.device, pose_backend="ultralytics_yolo26n", yolo_weights=data_root / "checkpoints/ultralytics/yolo26n-pose.pt")
    skeleton_def = get_skeleton_definition(backend="h36m_17")
    sim, human = _scene_sim(args.scene_root, args.scene_id, args.image_size)
    placement, yaw = _sample_placement(sim)
    LOGGER.info("scene=%s placement=%s yaw=%.2f", args.scene_id, placement.tolist(), yaw)
    manifest = {"scene_id": args.scene_id, "placement": placement.tolist(), "scene_yaw_deg": yaw, "videos": {}}
    try:
        for label, record in sorted(selected.items()):
            motion = _load_resampled_motion(record, args.target_frames)
            converted = generator.converter.convert(motion)
            joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
            roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
            offsets, centers = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=yaw)
            rgb, c2w = [], []
            for i, (pose, root) in enumerate(zip(joints, roots)):
                apply_humanoid_pose(human, pose, root, base_position=placement, scene_yaw_deg=yaw, floor_y=float(placement[1]), grounding_offset=float(offsets[i]))
                state, transform = _camera_state(placement, float(placement[1]), yaw + 180.0, float(placement[1] + centers[i] + offsets[i]))
                sim.get_agent(0).set_state(state)
                rgb.append(np.asarray(sim.get_sensor_observations()["color_sensor"][:, :, :3], dtype=np.uint8).copy())
                c2w.append(transform)
            c2w_arr = np.asarray(c2w, dtype=np.float32)
            pose3d, conf3d = generator.estimator.estimate_sequence(rgb)
            pose3d = transform_camera_sequence_to_gravity(pose3d, c2w_arr)
            kp2d, conf2d = _extract_2d_keypoints(generator.estimator, rgb)
            output = args.output_dir / f"{label.replace('/', '_').replace(' ', '_')}_{record['record_id']}.mp4"
            _render_side_by_side(rgb, kp2d, conf2d, pose3d, label, str(record["record_id"]), output, skeleton_def.edges, args.fps)
            manifest["videos"][label] = {"record_id": str(record["record_id"]), "video_path": str(output), "mean_2d_confidence": float(conf2d.mean()), "mean_3d_confidence": float(conf3d.mean())}
            LOGGER.info("saved %s", output)
    finally:
        sim.close()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
