"""
v7.0 多动作多视角 RGB-D 数据集生成脚本 —— generate_observation_dataset.py
======================================================================

功能：
    1. 遍历 motion_asset_manifest.json 中定义的典型老人动作；
    2. 在 Habitat 室内场景中将 Humanoid 置于指定安全位姿；
    3. 支持在围绕人体的多个视角 (多半径、多角度) 下捕获时序 RGB-D 与 3D 骨架真值；
    4. 导出结构化数据集清单与索引。

运行方式：
    python -m ea_avs_mvp_v7.scripts.generate_observation_dataset [--actions standing sitting fall_related] [--num-views 4] [--frames-per-action 10]
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml

from tools.motion_assets.data_paths import get_data_root, from_relative_data_path
from ea_avs_mvp_v7.humanoid.humanoid_loader import resolve_humanoid_assets
from ea_avs_mvp_v7.humanoid.humanoid_agent import HumanoidAgent
from ea_avs_mvp_v7.motion.motion_converter import convert_single_amass_motion
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v7.simulation.scene_loader import HabitatSceneLoader
from ea_avs_mvp_v7.simulation.robot_sensor import RobotSensorRig
from ea_avs_mvp_v7.simulation.observation_generator import ObservationGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_observation_dataset")


def compute_viewpoints_around_human(
    human_pos: np.ndarray,
    radius: float = 2.0,
    num_views: int = 4,
    camera_height: float = 1.2,
) -> List[dict]:
    """计算围绕人体的一组观察视点 (位置与面对人体的朝向角)。"""
    viewpoints = []
    for i in range(num_views):
        angle_deg = i * (360.0 / num_views)
        angle_rad = math.radians(angle_deg)
        # 视点坐标 (在 XZ 平面围绕人体)
        vx = float(human_pos[0] + radius * math.sin(angle_rad))
        vz = float(human_pos[2] + radius * math.cos(angle_rad))
        vy = float(human_pos[1] + camera_height)

        # 相机面对人体方向 (yaw)
        dx = human_pos[0] - vx
        dz = human_pos[2] - vz
        yaw_rad = math.atan2(dx, dz)
        yaw_deg = math.degrees(yaw_rad)

        viewpoints.append({
            "view_id": f"view_{i:02d}_{int(angle_deg)}deg",
            "position": [vx, float(human_pos[1]), vz],
            "yaw_deg": yaw_deg,
            "angle_deg": angle_deg,
            "radius": radius,
        })
    return viewpoints


def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Action Multi-View RGB-D Dataset")
    parser.add_argument("--config", type=str, default="ea_avs_mvp_v7/configs/v70_humanoid_sim.yaml")
    parser.add_argument("--actions", nargs="+", default=["fall_related", "standing", "sitting"], help="Action classes to generate")
    parser.add_argument("--num-views", type=int, default=4, help="Number of surrounding viewpoints per action")
    parser.add_argument("--frames-per-action", type=int, default=10, help="Max frames to capture per action")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dataset directory")
    args = parser.parse_args()

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    data_root = get_data_root()
    manifest_path = data_root / cfg.get("motion", {}).get("manifest_path", "assets/motions/raw/motion_asset_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 过滤动作
    selected_items = [m for m in manifest if m.get("target_class") in args.actions]
    if not selected_items:
        selected_items = manifest[:3]

    humanoid_assets = resolve_humanoid_assets(cfg)
    converted_dir = data_root / cfg.get("motion", {}).get("converted_dir", "assets/motions/converted")
    out_base = Path(args.output_dir) if args.output_dir else data_root / "datasets" / "v70_observations"

    # 初始化场景与 Agent
    scene_loader = HabitatSceneLoader(cfg)
    sim = scene_loader.create_simulator()
    humanoid = HumanoidAgent(sim, config=cfg, assets=humanoid_assets)
    humanoid.load()

    humanoid_pos = np.array([0.0, 0.1, 0.0], dtype=np.float32)
    humanoid.set_base_pose(humanoid_pos, yaw_rad=0.0)

    robot = RobotSensorRig(sim, config=cfg)
    obs_generator = ObservationGenerator(
        scene_loader=scene_loader,
        humanoid_agent=humanoid,
        robot_sensor=robot,
        config=cfg,
    )

    viewpoints = compute_viewpoints_around_human(
        human_pos=humanoid_pos,
        radius=2.0,
        num_views=args.num_views,
        camera_height=float(cfg.get("camera", {}).get("camera_height", 1.2)),
    )

    total_records = 0
    dataset_catalog = []

    for item in selected_items:
        target_class = item.get("target_class", "action")
        sid = item.get("babel_sid", "0")
        pkl_path = converted_dir / f"{target_class}_{sid}.pkl"

        if not pkl_path.exists():
            pkl_path = convert_single_amass_motion(
                manifest_item=item,
                urdf_path=humanoid_assets.urdf_path,
                output_dir=converted_dir,
            )

        player = MotionPlayer(pkl_path, playback_fps=float(cfg.get("motion", {}).get("playback_fps", 30.0)))

        for vp in viewpoints:
            v_id = vp["view_id"]
            seq_out = out_base / f"{target_class}_{sid}" / v_id
            records = obs_generator.run_sequence(
                motion_player=player,
                camera_pos=vp["position"],
                camera_yaw_deg=vp["yaw_deg"],
                output_dir=seq_out,
                frame_step=max(1, player.total_frames // args.frames_per_action),
                max_frames=args.frames_per_action,
            )
            total_records += len(records)
            dataset_catalog.append({
                "action_class": target_class,
                "babel_sid": sid,
                "viewpoint": vp,
                "num_frames": len(records),
                "output_dir": str(seq_out),
            })

    sim.close()

    # 导出总数据集 Catalog
    catalog_path = out_base / "dataset_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_actions": len(selected_items),
            "total_viewpoints": len(viewpoints),
            "total_captured_frames": total_records,
            "catalog": dataset_catalog,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print(f"[Dataset Generation Summary] Total frames captured: {total_records}")
    print(f"  Dataset Catalog: {catalog_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
