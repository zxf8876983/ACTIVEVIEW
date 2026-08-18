#!/usr/bin/env python
"""
RGB-D Humanoid 可见性调试 —— debug_humanoid_rgbd.py
=====================================================

功能：
    1. 加载 scene
    2. 创建 Humanoid（standing）
    3. 放在可导航位置
    4. 在人体周围 4 个方向放置相机（前/左/后/右）
    5. 相机始终通过 compute_look_at_yaw 看向人体
    6. 分别 render RGB + depth
    7. 保存结果到 outputs/humanoid_rgbd_debug/

验收：
    - 四个方向相机都 look-at 人体（不用手工指定 yaw）
    - humanoid_render_success 为真实测量（GT 锚定 depth 验证）
    - front / side / back 图像中人体外观明显不同
    - 人体对应位置在 depth 中有合理深度

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
from ea_avs_v5.humanoid_skeleton_adapter import get_humanoid_gt_skeleton
from ea_avs_v5.visualization import save_rgb_image
from ea_avs_v5.geometry import compute_look_at_yaw


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
        # 显式设置 semantic id（semantic sensor 验证用）
        if config["humanoid"].get("semantic_enabled", True):
            manager.assign_semantic_id_to_links(
                config["humanoid"].get("semantic_id", 100))
        # 放置到导航点，standing
        pt = runner.sample_navigable_point()
        hpos = np.array(pt, dtype=np.float32)
        manager.set_base_pose(hpos, 0.0)
        manager.set_pose("standing")

        # 从 actual Humanoid 提取 GT skeleton（15/15），用于 GT 锚定验证
        gt_result = get_humanoid_gt_skeleton(manager, strict=False)
        skeleton = gt_result["skeleton"]

        hy = hpos[1]
        # 四个方向只定义相机 position；yaw 全部通过 compute_look_at_yaw 自动计算
        offsets = {
            "front": np.array([0, 0, 2.2], dtype=np.float32),
            "back": np.array([0, 0, -2.2], dtype=np.float32),
            "left": np.array([-2.2, 0, 0], dtype=np.float32),
            "right": np.array([2.2, 0, 0], dtype=np.float32),
        }

        print(f"Humanoid 位于 {np.round(hpos, 3)}，在周围 4 方向渲染（look-at human）")
        for side, rel in offsets.items():
            cam_base = hpos + rel
            cam_base[1] = hy  # 相机在人体脚底同一高度上抬 camera_height
            # ⚠ 禁止硬编码 yaw；统一看向人体
            cam_yaw = compute_look_at_yaw(cam_base, hpos)
            obs = runner.render_at(cam_base, cam_yaw)
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

            # 真实测量 Humanoid 渲染（优先级：semantic -> GT-depth proxy）
            from ea_avs_v5.humanoid_validation import compute_humanoid_render_stats
            semantic_ids = [config["humanoid"].get("semantic_id", 100)] \
                if config["humanoid"].get("semantic_enabled", True) else []
            rs = compute_humanoid_render_stats(
                obs, config, cam_base, cam_yaw, skeleton, semantic_ids)
            ok = "✅" if rs["humanoid_render_success"] else "⚠️"
            print(f"  {ok} {side:6s}: rgb={rgb_path} depth={depth_path} "
                  f"depth_range={depth_range} "
                  f"validation={rs['humanoid_validation_source']} "
                  f"pixel={rs.get('humanoid_semantic_pixel_count') or rs.get('humanoid_proxy_pixel_count', 0)} "
                  f"render_ok={rs['humanoid_render_success']}")

        print(f"\n✅ RGB-D 已保存到 {out_dir}/")
        print("检查：人物应在 front/back/left/right 四个方向外观明显不同；")
        print("      depth 在人体位置有合理深度值（GT 锚定验证）。")

    finally:
        manager.close()
        runner.close()


if __name__ == "__main__":
    main()