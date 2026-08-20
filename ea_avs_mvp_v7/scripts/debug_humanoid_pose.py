"""
Humanoid 姿态与空间坐标对齐调试脚本 —— debug_humanoid_pose.py
============================================================

功能：
    1. 独立验证 Habitat 场景与 Humanoid 正确地面站立与正面朝向；
    2. 执行坐标系规范：
       - Human:  [0.0, 0.0, 0.0], yaw=180.0 deg (面向 +Z 轴)
       - Robot:  [0.0, 0.0, 2.5], yaw=0.0 deg   (面向 -Z 轴，相机正对 Humanoid)
    3. 输出 [V7 Spatial Debug] 标准空间对齐日志；
    4. 生成 visualizations/humanoid_debug/rgb.png。

运行方式：
    python -m ea_avs_mvp_v7.scripts.debug_humanoid_pose
"""

import argparse
import logging
import math
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
logger = logging.getLogger("debug_pose")


def main():
    parser = argparse.ArgumentParser(description="Debug Humanoid Pose and Spatial Alignment")
    parser.add_argument("--human-pos", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Humanoid position [x, y, z]")
    parser.add_argument("--robot-pos", nargs=3, type=float, default=[0.0, 0.0, 2.5], help="Robot position [x, y, z]")
    parser.add_argument("--human-yaw", type=float, default=180.0, help="Humanoid yaw in degrees")
    parser.add_argument("--robot-yaw", type=float, default=0.0, help="Robot yaw in degrees")
    args = parser.parse_args()

    cfg = load_v7_config()
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()

    human_pos = [float(x) for x in args.human_pos]
    robot_pos = [float(x) for x in args.robot_pos]

    # 1. 实例化并定位 Humanoid
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pos, yaw_rad=math.radians(args.human_yaw))

    # 2. 实例化并定位 Robot 与传感器
    robot = RobotAgent(sim)
    robot.set_pose(robot_pos, yaw_deg=args.robot_yaw)

    sensor = RGBDSensor(sim, cfg.sensor)
    cam_mat = sensor.get_camera_pose_matrix()
    cam_pos = [float(cam_mat[0, 3]), float(cam_mat[1, 3]), float(cam_mat[2, 3])]

    # 3. 打印标准化空间调试日志
    print("\n" + "=" * 65)
    print("[V7 Spatial Debug]")
    print(f"Human:  {human_pos}")
    print(f"Robot:  {robot_pos}")
    print(f"Camera: {[round(x, 3) for x in cam_pos]}")
    print("=" * 65)

    # 4. 关节与视锥检查
    joints = humanoid.get_gt_joint_positions()
    pelvis_3d = joints.get("pelvis", human_pos)
    view_check = sensor.check_object_in_view(pelvis_3d, cam_mat)

    # 5. 捕获并保存图像
    obs = sensor.capture()
    rgb = obs.get("rgb")
    depth = obs.get("depth")
    env.close()

    if rgb is None or depth is None:
        raise RuntimeError("Failed to capture RGB-D observation from simulator!")

    out_dir = get_data_root() / "visualizations" / "humanoid_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "rgb.png"
    Image.fromarray(rgb).save(out_png)
    logger.info("Saved Humanoid Pose Debug RGB to: %s", out_png)

    dist = float(np.linalg.norm(np.array(pelvis_3d) - np.array(cam_pos)))
    non_black_ratio = float((rgb > 15).mean())

    print(f"  - Output Image:        {out_png}")
    print(f"  - Distance:            {dist:.3f} m")
    print(f"  - View Check:          visible={view_check['visible']}, angle={view_check['angle']:.1f} deg")
    print(f"  - Non-Black Ratio:     {non_black_ratio * 100:.1f}%")
    print(f"  - Depth Range:         [{depth.min():.2f} m, {depth.max():.2f} m]")
    print("=" * 65)

    if view_check["visible"] and non_black_ratio > 0.3:
        print("PASS: Humanoid Pose & Coordinate Alignment Verified\n")
        sys.exit(0)
    else:
        print("FAIL: Humanoid not clearly visible in debug view!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
