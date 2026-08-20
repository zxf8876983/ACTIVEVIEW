"""
多动作多视角 Episode 数据集生成脚本 —— generate_dataset.py
======================================================

功能：
    1. 遍历典型老人动作清单；
    2. 计算围绕人体的多个观察视点；
    3. 批量生成时序 Episode 数据集与 Dataset Catalog 清单。

运行方式：
    python -m ea_avs_mvp_v7.scripts.generate_dataset [--actions standing sitting fall_related] [--num-views 4]
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root, from_relative_data_path
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent, resolve_humanoid_urdf_path
from ea_avs_mvp_v7.motion.amass_loader import load_amass_motion
from ea_avs_mvp_v7.motion.motion_converter import MotionConverter
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor
from ea_avs_mvp_v7.observation.recorder import ObservationRecorder
from ea_avs_mvp_v7.dataset.episode_generator import EpisodeGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_dataset")


def compute_surrounding_viewpoints(
    human_pos: np.ndarray,
    radius: float = 2.0,
    num_views: int = 4,
    camera_height: float = 1.2,
) -> List[dict]:
    """计算围绕人体的多角度观察位姿。"""
    viewpoints = []
    for i in range(num_views):
        angle_deg = i * (360.0 / num_views)
        angle_rad = math.radians(angle_deg)
        vx = float(human_pos[0] + radius * math.sin(angle_rad))
        vz = float(human_pos[2] + radius * math.cos(angle_rad))
        vy = float(human_pos[1] + camera_height)

        dx = human_pos[0] - vx
        dz = human_pos[2] - vz
        yaw_deg = math.degrees(math.atan2(dx, dz))

        viewpoints.append({
            "view_id": f"view_{i:02d}_{int(angle_deg)}deg",
            "position": [vx, float(human_pos[1]), vz],
            "yaw_deg": yaw_deg,
            "angle_deg": angle_deg,
            "radius": radius,
        })
    return viewpoints


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Action Multi-View Active Perception Dataset")
    parser.add_argument("--actions", nargs="+", default=["fall_related", "standing", "sitting"], help="Action classes")
    parser.add_argument("--num-views", type=int, default=4, help="Number of viewpoints per action")
    parser.add_argument("--frames-per-episode", type=int, default=10, help="Max frames per episode")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dataset directory")
    args = parser.parse_args()

    cfg = load_v7_config()
    manifest_p = get_data_root() / cfg.motion.get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    selected_items = [m for m in manifest if m.get("target_class") in args.actions]
    if not selected_items:
        selected_items = manifest[:3]

    out_base = Path(args.output_dir) if args.output_dir else get_data_root() / "datasets" / "v70_episodes"
    converted_dir = get_data_root() / cfg.motion.get("converted_dir", "assets/motions/converted")
    urdf_path, _ = resolve_humanoid_urdf_path(cfg.humanoid)
    converter = MotionConverter(urdf_path)

    # 初始化环境
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()

    robot = RobotAgent(sim)
    sensor = RGBDSensor(sim, cfg.sensor)
    recorder = ObservationRecorder(out_base)
    ep_gen = EpisodeGenerator(env=env, humanoid=humanoid, robot=robot, sensor=sensor, recorder=recorder)

    human_pos = np.array([0.0, 0.1, 0.0], dtype=np.float32)
    viewpoints = compute_surrounding_viewpoints(human_pos, radius=2.0, num_views=args.num_views)

    episodes_catalog = []

    for item in selected_items:
        target_class = item.get("target_class", "action")
        sid = item.get("babel_sid", "0")
        pkl_path = converted_dir / f"{target_class}_{sid}.pkl"

        if not pkl_path.exists():
            npz_p = from_relative_data_path(item["local_motion_path"])
            norm_motion = load_amass_motion(npz_p, item.get("start_frame"), item.get("end_frame"), item)
            pkl_path = converter.convert_and_save(norm_motion, pkl_path)

        player = MotionPlayer(pkl_path, playback_fps=float(cfg.motion.get("playback_fps", 30.0)))

        for vp in viewpoints:
            v_id = vp["view_id"]
            ep_id = f"ep_{target_class}_{sid}_{v_id}"
            ep_out = out_base / ep_id
            episode = ep_gen.generate_single_episode(
                episode_id=ep_id,
                motion_player=player,
                camera_position=vp["position"],
                camera_yaw_deg=vp["yaw_deg"],
                human_position=human_pos.tolist(),
                human_yaw_rad=0.0,
                output_dir=out_base,
                max_frames=args.frames_per_episode,
            )
            episodes_catalog.append({
                "episode_id": ep_id,
                "action_class": target_class,
                "babel_sid": sid,
                "viewpoint": vp,
                "num_frames": episode.num_frames,
            })

    env.close()

    catalog_path = out_base / "dataset_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_episodes": len(episodes_catalog),
            "episodes": episodes_catalog,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"[Dataset Generation Summary] Generated {len(episodes_catalog)} episodes in {out_base}")
    print(f"  Catalog: {catalog_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
