#!/usr/bin/env python
"""
RGB-D Humanoid 可见性调试 —— debug_humanoid_rgbd.py
=====================================================

功能：
    1. 加载 scene
    2. 创建 Humanoid（standing）
    3. 放在可导航位置
    4. 在人体周围 4 个方向放置相机（前/左/后/右）
    5. 分别 render RGB + depth
    6. 保存结果到 outputs/humanoid_rgbd_debug/

验收：
    - front / side / back 图像中人体外观明显不同
    - 人体确实存在于 RGB
    - 人体对应位置在 depth 中有合理深度
    - 家具可以真实遮挡人体（环境遮挡）
    - 人体自身表面产生 self-occlusion

运行命令：
    python scripts/debug_humanoid_rgbd.py --config configs/mvp50_humanoid.yaml
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v5.config import load_config
from ea_avs_v5.habitat_runner import HabitatRunner
from ea_avs_v5.humanoid_manager import HumanoidManager
from ea_avs_v5.visualization import save_rgb_image
from PIL import Image


def save_depth_npy(depth, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(depth, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description="v5.0 Humanoid RGB-D 调试")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/humanoid_rgbd_debug", help="输出目录")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    runner = HabitatRunner(config)
    manager = HumanoidManager(runner, config)

    try:
        manager.load()
        # 放置到导航点，standing
        pt = runner.sample_navigable_point()
        hpos = np.array(pt, dtype=np.float32)
        manager.set_base_pose(hpos, 0.0)
        manager.set_pose("standing")
        hy = hpos[1]
        cam_h = config["camera"]["camera_height"]

        # 4 个方向：前(+Z)、左(+X 但观察从负侧)、后(-Z)、右
        # humanoid forward 朝 +Z（base_yaw=0）
        offsets = {
            "front": (np.array([0, 0, 2.2], dtype=np.float32), 0.0),
            "back": (np.array([0, 0, -2.2], dtype=np.float32), np.pi),
            "left": (np.array([-2.2, 0, 0], dtype=np.float32), -np.pi / 2),
            "right": (np.array([2.2, 0, 0], dtype=np.float32), np.pi / 2),
        }

        print(f"Humanoid 位于 {np.round(hpos, 3)}，在周围 4 方向渲染")
        for side, (rel, yaw) in offsets.items():
            cam_base = hpos + rel
            cam_base[1] = hy  # 相机在人体脚底同一高度上抬 camera_height
            obs = runner.render_at(cam_base, yaw)
            rgb = obs["rgb"]
            depth = obs["depth"]

            rgb_path = os.path.join(out_dir, f"{side}_rgb.png")
            depth_path = os.path.join(out_dir, f"{side}_depth.npy")
            save_rgb_image(rgb, rgb_path)
            save_depth_npy(depth, depth_path)

            # 深度统计（非 0 区域）
            dz = np.asarray(depth, dtype=np.float32)
            if dz.ndim == 3:
                dz = dz[..., 0]
            nz = dz[dz > 0]
            depth_range = f"{nz.min():.3f}~{nz.max():.3f}" if len(nz) else "empty"
            print(f"  {side:6s}: rgb={rgb_path}  depth={depth_path} "
                  f"depth_range={depth_range}")

        print(f"\n✅ RGB-D 已保存到 {out_dir}/")
        print("检查：人物应在 front/back/left/right 四个方向外观明显不同；")
        print("      depth 在人体位置有合理深度值。")

    finally:
        manager.close()
        runner.close()


if __name__ == "__main__":
    main()