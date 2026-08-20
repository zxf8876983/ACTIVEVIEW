"""
Humanoid 地面接触与坐标校准调试脚本 —— debug_humanoid_grounding.py
================================================================

功能：
    1. 独立验证 Habitat 场景中 neutral_0 Humanoid 的地面着地 (Grounding) 与基座变换；
    2. 不加载 AMASS 动作，仅测试 rest_position 下的标准站立姿态；
    3. 输出 [V7 Humanoid Grounding] 标准着地状态日志；
    4. 渲染并保存验证图像至 visualizations/humanoid_grounding_debug.png。

运行方式：
    python -m ea_avs_mvp_v7.scripts.debug_humanoid_grounding
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
logger = logging.getLogger("debug_grounding")


def main():
    parser = argparse.ArgumentParser(description="Debug Humanoid Grounding Alignment")
    parser.add_argument("--human-pos", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Human position [x, y, z]")
    parser.add_argument("--robot-pos", nargs=3, type=float, default=[0.0, 0.0, 2.5], help="Robot position [x, y, z]")
    parser.add_argument("--human-yaw", type=float, default=180.0, help="Humanoid yaw in degrees")
    parser.add_argument("--robot-yaw", type=float, default=0.0, help="Robot yaw in degrees")
    args = parser.parse_args()

    cfg = load_v7_config()
    env = HabitatEnv(cfg.habitat, cfg.sensor)
    sim = env.start()

    human_pos = [float(x) for x in args.human_pos]
    robot_pos = [float(x) for x in args.robot_pos]

    # 1. 实例化 Humanoid 并置为 rest 站立姿态
    humanoid = HumanoidAgent(sim, cfg.humanoid)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.agent.set_rest_position()
    humanoid.set_base_pose(human_pos, yaw_rad=math.radians(args.human_yaw))

    # 2. 实例化 Robot 与 RGB-D 相机
    robot = RobotAgent(sim)
    robot.set_pose(robot_pos, yaw_deg=args.robot_yaw)

    sensor = RGBDSensor(sim, cfg.sensor)
    cam_mat = sensor.get_camera_pose_matrix()
    cam_pos = [float(cam_mat[0, 3]), float(cam_mat[1, 3]), float(cam_mat[2, 3])]

    # 3. 提取着地摘要信息
    grounding = humanoid.get_grounding_summary()
    pelvis = grounding["pelvis_position"]
    l_foot = grounding["left_foot_position"]
    r_foot = grounding["right_foot_position"]
    base_t = grounding["base_transformation"]

    # 4. 打印标准调试日志
    print("\n" + "=" * 65)
    print("[V7 Humanoid Grounding]")
    print(f"base_position:       {humanoid.base_position.tolist()}")
    print(f"root_transform:      yaw={args.human_yaw} deg, ground_offset={humanoid.ground_offset:.3f}")
    print(f"base_transformation: {[round(x, 4) for x in base_t]}")
    print(f"pelvis position:     {[round(x, 4) for x in pelvis]}")
    print(f"left foot position:  {[round(x, 4) for x in l_foot]}")
    print(f"right foot position: {[round(x, 4) for x in r_foot]}")
    print(f"foot Y close to 0:   {grounding['foot_grounded']}")
    print("=" * 65)

    # 5. 捕获并保存图像
    obs = sensor.capture()
    rgb = obs.get("rgb")
    depth = obs.get("depth")
    env.close()

    if rgb is None or depth is None:
        raise RuntimeError("Failed to capture RGB-D observation!")

    out_p = get_data_root() / "visualizations" / "humanoid_grounding_debug.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_p)
    logger.info("Saved Humanoid Grounding Debug RGB to: %s", out_p)

    dist = float(np.linalg.norm(np.array(pelvis) - np.array(cam_pos)))
    non_black_ratio = float((rgb > 15).mean())

    print(f"  - Output Image:        {out_p}")
    print(f"  - Distance:            {dist:.3f} m")
    print(f"  - Non-Black Ratio:     {non_black_ratio * 100:.1f}%")
    print(f"  - Depth Range:         [{depth.min():.2f} m, {depth.max():.2f} m]")
    print("=" * 65)

    if grounding["foot_grounded"] and non_black_ratio > 0.3:
        print("PASS: Humanoid Grounding Calibration Verified\n")
        sys.exit(0)
    else:
        print("FAIL: Humanoid grounding check failed!\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
