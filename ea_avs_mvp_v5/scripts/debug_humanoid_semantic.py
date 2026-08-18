#!/usr/bin/env python
"""
Humanoid semantic debug —— debug_humanoid_semantic.py
=======================================================

目的（v5.0 第三轮）：
    显式给 Humanoid link/visual scene node 设置 semantic_id 后，验证 semantic
    sensor 是否能真正输出 Humanoid 像素（前两轮因未设置 semantic id 而全 0）。

输出：
    semantic unique IDs
    配置的 Humanoid semantic ID
    Humanoid semantic pixel count / ratio / bbox
    保存 semantic.npy 与 semantic_mask.png

运行命令：
    python scripts/debug_humanoid_semantic.py --config configs/mvp50_humanoid.yaml

结论：
    若 pixel count > 0 → humanoid_validation_source 可用 "semantic"
    否则需说明尝试过的 API 与结果，并回退 gt_depth_proxy。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v5.config import load_config
from ea_avs_v5.habitat_runner import HabitatRunner
from ea_avs_v5.humanoid_manager import HumanoidManager
from ea_avs_v5.geometry import compute_look_at_yaw
from ea_avs_v5.visualization import save_rgb_image
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="v5.0 Humanoid semantic 调试")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/humanoid_semantic_debug", help="输出目录")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    mcfg = config["camera"]

    semantic_id = config["humanoid"].get("semantic_id", 100)

    runner = HabitatRunner(config)
    manager = HumanoidManager(runner, config)

    try:
        manager.load()
        pt = runner.sample_navigable_point()
        hpos = np.array(pt, dtype=np.float32)
        manager.set_base_pose(hpos, 0.0)
        manager.set_pose("standing")

        # 尝试 1：显式给所有 link scene node + visual node 设 semantic_id
        n_set = manager.assign_semantic_id_to_links(semantic_id)
        print(f"[semantic] 显式设置 semantic_id={semantic_id} 到 "
              f"{n_set} 个 link scene node（+ visual node）")

        # 相机 look-at human
        cam_base = np.array([hpos[0] + 2.2, hpos[1], hpos[2]], dtype=np.float32)
        cam_yaw = compute_look_at_yaw(cam_base, hpos)
        obs = runner.render_at(cam_base, cam_yaw)

        sem = obs.get("semantic")
        rgb = obs.get("rgb")
        if sem is None:
            print("[semantic] ⚠ 未启用 semantic sensor（config semantic_enabled 或 "
                  "sensor 未添加）")
            print("  尝试过的 API：CameraSensorSpec(SensorType.SEMANTIC)")
            print("  结果：无 semantic 观测 → humanoid_validation_source = "
                  "gt_depth_proxy")
            sys.exit(1)

        sem = np.asarray(sem)
        if sem.ndim == 3:
            sem = sem[..., 0]
        vals, counts = np.unique(sem, return_counts=True)
        print("[semantic] unique IDs =", vals.tolist())
        print(f"[semantic] configured humanoid semantic id = {semantic_id}")
        mask = sem == semantic_id
        cnt = int(mask.sum())
        h, w = sem.shape
        ratio = cnt / float(h * w) if h * w > 0 else 0.0
        bbox = None
        if cnt > 0:
            ys, xs = np.where(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        print(f"[semantic] Humanoid semantic pixel count = {cnt}")
        print(f"[semantic] Humanoid semantic pixel ratio = {ratio:.4f}")
        print(f"[semantic] Humanoid semantic bbox = {bbox}")

        # 保存
        np.save(os.path.join(out_dir, "semantic.npy"), sem)
        mask_im = Image.fromarray((mask.astype(np.uint8)) * 255)
        mask_im.save(os.path.join(out_dir, "semantic_mask.png"))
        if rgb is not None:
            save_rgb_image(rgb, os.path.join(out_dir, "rgb.png"))

        if cnt > 0:
            print("[semantic] ✅ 成功：humanoid_validation_source 可使用 'semantic'")
        else:
            print("[semantic] ⚠ semantic 像素仍为 0 → 回退 gt_depth_proxy")
            print("  尝试过的 API：link scene node.semantic_id、"
                  "link visual node.semantic_id、root_scene_node.semantic_id")

    finally:
        manager.close()
        runner.close()


if __name__ == "__main__":
    main()