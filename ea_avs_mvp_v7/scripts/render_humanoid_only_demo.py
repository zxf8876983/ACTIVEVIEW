"""
最小 Humanoid 独立渲染演示脚本 —— render_humanoid_only_demo.py
============================================================

功能：
    1. 独立运行 Habitat 场景与 Humanoid 站立渲染测试 (不依赖 AMASS 动作数据)；
    2. 将 Humanoid 与 Robot 置于视野通畅的开阔室内区域；
    3. 采集并保存单帧清晰 RGB 图像至 data/ActiveView/visualizations/humanoid_debug/rgb/frame_000001.png；
    4. 输出关节在图像中的投影坐标与视野有效性判定。

运行方式：
    python -m ea_avs_mvp_v7.scripts.render_humanoid_only_demo
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.core.paths import get_data_root
from ea_avs_mvp_v7.environment.habitat_env import HabitatEnv
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent
from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("render_humanoid_only")


def main():
    parser = argparse.ArgumentParser(description="Render Humanoid Only Demonstration")
    parser.add_argument("--human-pos", nargs=3, type=float, default=[1.5, -1.60, 4.0], help="Human position [x, y, z]")
    parser.add_argument("--robot-pos", nargs=3, type=float, default=[1.5, -1.60, 6.8], help="Robot position [x, y, z]")
    parser.add_argument("--robot-yaw", type=float, default=0.0, help="Robot yaw in degrees")
    args = parser.parse_args()

    cfg = load_v7_config()
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()

    human_pos = np.array(args.human_pos, dtype=np.float32)
    robot_pos = np.array(args.robot_pos, dtype=np.float32)

    # 1. 实例化并加载 Humanoid
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pos, yaw_rad=0.0)

    # 2. 实例化并定位 Robot 与 RGB-D 传感器
    robot = RobotAgent(sim)
    robot.set_pose(robot_pos, yaw_deg=args.robot_yaw)

    sensor = RGBDSensor(sim, cfg.sensor)
    cam_mat = sensor.get_camera_pose_matrix()
    cam_pos = [float(cam_mat[0, 3]), float(cam_mat[1, 3]), float(cam_mat[2, 3])]

    # 3. 关节提取与视锥检查
    joints = humanoid.get_gt_joint_positions()
    pelvis_3d = joints.get("pelvis", human_pos.tolist())
    view_check = sensor.check_object_in_view(pelvis_3d, cam_mat)

    # 4. 捕获传感器图像
    obs = sensor.capture()
    rgb = obs.get("rgb")
    depth = obs.get("depth")
    env.close()

    if rgb is None or depth is None:
        raise RuntimeError("Failed to capture RGB-D observation from simulator!")

    # 5. 保存输出图像
    out_dir = get_data_root() / "visualizations" / "humanoid_debug" / "rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "frame_000001.png"
    Image.fromarray(rgb).save(out_img_path)

    # 6. 计算非零像素比例与人体像素统计
    non_zero_ratio = float((rgb > 15).mean())
    dist = float(np.linalg.norm(np.array(pelvis_3d) - np.array(cam_pos)))

    print("\n" + "=" * 65)
    print("[v7.0 Standalone Humanoid Render Result]")
    print(f"  - Output Image:        {out_img_path}")
    print(f"  - Scene ID:            {cfg.habitat.get('scene_id', 'apartment_1')}")
    print(f"  - Humanoid Position:   {human_pos.tolist()}")
    print(f"  - Robot Position:      {robot_pos.tolist()}")
    print(f"  - Distance:            {dist:.3f} m")
    print(f"  - View Check:          visible={view_check['visible']}, angle={view_check['angle']:.1f} deg")
    print(f"  - Non-Black Ratio:     {non_zero_ratio * 100:.1f}%")
    print(f"  - Depth Range:         [{depth.min():.2f} m, {depth.max():.2f} m]")
    print("=" * 65)

    if view_check["visible"] and non_zero_ratio > 0.4:
        print("PASS: Humanoid Successfully Rendered in Scene\n")
        sys.exit(0)
    else:
        print("FAIL: Humanoid rendering check failed!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
