"""
单视点最小渲染验证脚本 —— debug_view_render.py
=============================================

功能：
    1. 加载 Habitat 场景 (apartment_1.glb)；
    2. 加载并放置 Humanoid (neutral_0) 至物理地面；
    3. 固定一个机器人测试视点；
    4. 同步更新相机并渲染 RGB 图像；
    5. 打印 [V8 Camera Debug] 并保存渲染结果至 visualizations/v8_debug_render.png。

运行方式：
    python -m ea_avs_mvp_v8.scripts.debug_view_render
"""

import logging
import sys
from pathlib import Path
from PIL import Image

from ea_avs_mvp_v8.core.config import load_v8_config
from ea_avs_mvp_v8.core.paths import get_data_root
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from ea_avs_mvp_v8.human.human_placement import HumanPlacement
from ea_avs_mvp_v8.robot.robot_adapter import V8RobotAdapter

# 复用 v7 Humanoid
from ea_avs_mvp_v7.human.humanoid_agent import HumanoidAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_view_render")


def main():
    cfg = load_v8_config()
    scene_id = cfg.scene.get("scene_id", "apartment_1")

    # 1. 启动 Habitat 场景
    logger.info("Initializing Habitat Environment...")
    env_adapter = V8EnvironmentAdapter(cfg.scene, cfg.camera)
    sim = env_adapter.start()

    # 2. 放置 Humanoid
    logger.info("Loading Humanoid...")
    human_placer = HumanPlacement(cfg.human)
    human_pose = human_placer.sample_position(scene_id=scene_id)

    humanoid = HumanoidAgent(sim, cfg.human)
    humanoid.load()
    humanoid.set_visibility(True)
    humanoid.set_base_pose(human_pose.position, yaw_rad=human_pose.yaw_deg)

    # 3. 创建测试视点 (固定在人体前方 2.2 米处正对人体)
    test_view = CandidateViewpoint(
        viewpoint_id="debug_view_frontal",
        position=[1.5, -1.60, 6.2],  # 人体位于 [1.5, -1.60, 4.0]
        yaw_deg=0.0,
        radius=2.2,
        angle_deg=0.0,
        camera_height=cfg.camera.get("camera_height", 1.2),
        ground_height=-1.60,
    )

    # 4. 设置机器人与相机
    robot_adapter = V8RobotAdapter(sim, cfg.camera)
    cam_info = robot_adapter.set_viewpoint(test_view, verbose=True)

    # 5. 渲染并保存 RGB 图像
    obs = robot_adapter.capture_observation()
    env_adapter.close()

    rgb = obs.get("rgb")
    if rgb is None or rgb.size == 0:
        print("ERROR: Failed to capture RGB observation!")
        sys.exit(1)

    out_dir = get_data_root() / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / "v8_debug_render.png"

    img = Image.fromarray(rgb)
    img.save(out_img_path)

    print("\n" + "=" * 65)
    print("[V8 Single View Render Verification]")
    print(f"Scene ID:         {scene_id}")
    print(f"Human Position:   {human_pose.position}")
    print(f"Robot Position:   {test_view.position}")
    print(f"Camera Position:  {cam_info['camera_position']}")
    print(f"Image Resolution: {img.size[0]} x {img.size[1]}")
    print(f"Saved Image:      {out_img_path}")
    print("Status:           SUCCESS")
    print("=" * 65 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
