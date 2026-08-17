#!/usr/bin/env python
"""
Humanoid forward-axis 校准 —— debug_humanoid_forward_axis.py
==============================================================

目的：
    Humanoid base_yaw = 0 时，分别在 +X / -X / +Z / -Z 四个观察方向渲染，
    通过 GT 锚定 depth 验证 + 保存图像，人工一次性确认人体正面朝向。

由于没有视觉分类器，本脚本生成四张图（front_candidate_+X 等）供人工判断
"哪个方向看起来是人体正面"。确认后将对应轴向写入配置：
    humanoid.forward_axis: "+Z"   # 或 -Z / +X / -X

打印：
    base_rot
    sim_obj transformation
    candidate forward vectors

运行命令：
    python scripts/debug_humanoid_forward_axis.py --config configs/mvp50_humanoid.yaml
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v5.config import load_config
from ea_avs_v5.habitat_runner import HabitatRunner
from ea_avs_v5.humanoid_manager import HumanoidManager
from ea_avs_v5.humanoid_skeleton_adapter import get_humanoid_gt_skeleton
from ea_avs_v5.geometry import compute_look_at_yaw
from ea_avs_v5.visualization import save_rgb_image


def main():
    parser = argparse.ArgumentParser(description="v5.0 Humanoid forward-axis 校准")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/humanoid_forward_axis", help="输出目录")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    mcfg = config["camera"]

    runner = HabitatRunner(config)
    manager = HumanoidManager(runner, config)

    try:
        manager.load()
        pt = runner.sample_navigable_point()
        hpos = np.array(pt, dtype=np.float32)
        manager.set_base_pose(hpos, 0.0)   # base_yaw = 0
        manager.set_pose("standing")
        gt_result = get_humanoid_gt_skeleton(manager, strict=False)
        skeleton = gt_result["skeleton"]

        st = manager.get_state()
        print("base_rot:", round(st.actual_base_yaw, 4), "rad")
        print("configured forward_axis:", config["humanoid"].get("forward_axis", "+Z"))
        print("manager forward:", np.round(manager.get_humanoid_forward_vector(), 3))

        # 四个候选观察方向（相机放在人体周围，看向人体）
        hy = hpos[1]
        directions = {
            "front_candidate_+X": np.array([2.4, 0.0, 0.0], dtype=np.float32),
            "front_candidate_-X": np.array([-2.4, 0.0, 0.0], dtype=np.float32),
            "front_candidate_+Z": np.array([0.0, 0.0, 2.4], dtype=np.float32),
            "front_candidate_-Z": np.array([0.0, 0.0, -2.4], dtype=np.float32),
        }

        print("=" * 70)
        print("在 base_yaw=0 下从 4 个方向观察 Humanoid：")
        print("（人工查看以下 4 张图，确认哪个方向是人体正面，")
        print("  并把对应轴向写入 humanoid.forward_axis）")
        print("=" * 70)

        for name, rel in directions.items():
            cam_base = hpos + rel
            cam_base[1] = hy
            cam_yaw = compute_look_at_yaw(cam_base, hpos)
            obs = runner.render_at(cam_base, cam_yaw)
            path = os.path.join(out_dir, f"{name}.png")
            save_rgb_image(obs["rgb"], path)
            print(f"  {name}: saved {path}  (camera yaw={np.degrees(cam_yaw):.0f}°)")

        # 打印候选 forward vectors（配置 forward_axis 对应的世界前向）
        print("\ncandidate forward vectors (given base_yaw=0):")
        for ax in ["+X", "-X", "+Z", "-Z"]:
            tmp = {"humanoid": {**config["humanoid"], "forward_axis": ax}}
            save_ax = ax
            base_yaw = 0.0
            axis_map = {"+Z": np.array([0, 0, 1]), "-Z": np.array([0, 0, -1]),
                        "+X": np.array([1, 0, 0]), "-X": np.array([-1, 0, 0])}
            local = axis_map[save_ax]
            # RotY(0) = identity
            print(f"  forward_axis={save_ax} -> world forward {local.tolist()}")

        print("\n完成：请查看上述 4 张图，确认人体正面所在的轴向，")
        print("然后在 configs/mvp50_humanoid.yaml 设置 humanoid.forward_axis。")

    finally:
        manager.close()
        runner.close()


if __name__ == "__main__":
    main()