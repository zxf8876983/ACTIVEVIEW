"""
Humanoid 可见性与相机视锥调试脚本 —— debug_humanoid_visibility.py
================================================================

功能：
    1. 独立验证 Habitat 场景加载、Humanoid 实体生成与机器人相机对准；
    2. 执行坐标系规范 (Human: [0, 0, 0], yaw=180; Robot: [0, 0, 2.5], yaw=0)；
    3. 输出标准化 [V7 Humanoid Visibility Debug] 调试报告并保存调试视图。

运行方式：
    python -m ea_avs_mvp_v7.scripts.debug_humanoid_visibility
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
logger = logging.getLogger("debug_visibility")


def main():
    parser = argparse.ArgumentParser(description="Debug Humanoid Visibility in Habitat Scene")
    parser.add_argument("--human-pos", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Humanoid base position [x, y, z]")
    parser.add_argument("--robot-pos", nargs=3, type=float, default=[0.0, 0.0, 2.5], help="Robot base position [x, y, z]")
    parser.add_argument("--human-yaw", type=float, default=180.0, help="Humanoid yaw in degrees")
    parser.add_argument("--robot-yaw", type=float, default=0.0, help="Robot yaw in degrees (0 deg faces -Z)")
    parser.add_argument("--output-image", type=str, default=None, help="Path to save rendered debug RGB image")
    args = parser.parse_args()

    cfg = load_v7_config()
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()

    human_pos = np.array(args.human_pos, dtype=np.float32)
    robot_pos = np.array(args.robot_pos, dtype=np.float32)

    # 1. 实例化 Humanoid
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    if not humanoid.is_loaded:
        raise RuntimeError("Humanoid failed to load into Habitat scene!")

    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pos, yaw_rad=math.radians(args.human_yaw))

    # 2. 实例化 Robot 与传感器
    robot = RobotAgent(sim)
    robot.set_pose(robot_pos, yaw_deg=args.robot_yaw)

    sensor = RGBDSensor(sim, cfg.sensor)
    cam_mat = sensor.get_camera_pose_matrix()
    cam_pos = [float(cam_mat[0, 3]), float(cam_mat[1, 3]), float(cam_mat[2, 3])]

    # 3. 关节提取与投影
    joints = humanoid.get_gt_joint_positions()
    if not joints:
        raise RuntimeError("Failed to extract Humanoid 3D joints!")

    pelvis_3d = joints.get("pelvis", human_pos.tolist())
    view_check = sensor.check_object_in_view(pelvis_3d, cam_mat)

    # 计算 2D Bounding Box
    inv_cam = np.linalg.inv(cam_mat)
    intr = sensor.intrinsics
    u_coords, v_coords = [], []
    for j_name, j_pos in joints.items():
        pt_w = np.array(j_pos + [1.0], dtype=np.float32)
        pt_c = inv_cam @ pt_w
        z_c = -pt_c[2]
        if z_c > 0.01:
            u = intr["cx"] + intr["fx"] * (pt_c[0] / z_c)
            v = intr["cy"] - intr["fy"] * (pt_c[1] / z_c)
            u_coords.append(u)
            v_coords.append(v)

    if u_coords and v_coords:
        bbox_2d = [
            round(float(min(u_coords)), 1),
            round(float(min(v_coords)), 1),
            round(float(max(u_coords)), 1),
            round(float(max(v_coords)), 1),
        ]
    else:
        bbox_2d = [-1.0, -1.0, -1.0, -1.0]

    dist = float(np.linalg.norm(np.array(pelvis_3d) - np.array(cam_pos)))

    # 4. 捕获渲染结果
    obs = sensor.capture()
    rgb = obs.get("rgb")
    depth = obs.get("depth")
    env.close()

    if rgb is not None:
        out_p = Path(args.output_image) if args.output_image else get_data_root() / "visualizations" / "humanoid_debug" / "visibility_check.png"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out_p)
        logger.info("Saved debug RGB snapshot to: %s", out_p)

    scene_id = cfg.habitat.get("scene_id", "apartment_1")

    print("\n" + "=" * 65)
    print("[V7 Humanoid Visibility Debug]")
    print(f"  - Habitat scene:         {scene_id}")
    print(f"  - Humanoid loaded:       {humanoid.is_loaded}")
    print(f"  - Humanoid visible:      {humanoid.is_visible and view_check['visible']}")
    print(f"  - Humanoid position:     {human_pos.tolist()}")
    print(f"  - Robot position:        {robot_pos.tolist()}")
    print(f"  - Camera position:       {[round(x, 3) for x in cam_pos]}")
    print(f"  - Human-camera distance: {dist:.3f} m")
    print(f"  - Human bounding box:    {bbox_2d}")
    print(f"  - View frustum check:    visible={view_check['visible']}, angle={view_check['angle']:.1f} deg")
    print("=" * 65)

    if humanoid.is_loaded and view_check["visible"]:
        print("PASS: Humanoid Visibility Verified in Scene\n")
        sys.exit(0)
    else:
        print("FAIL: Humanoid not visible from camera!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
