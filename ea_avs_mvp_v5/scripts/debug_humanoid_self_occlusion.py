#!/usr/bin/env python
"""
Humanoid self-occlusion ray test —— debug_humanoid_self_occlusion.py
====================================================================

目标：
    判断当前 Habitat ray casting 是否会命中 Humanoid articulated body / links，
    从而能否区分 environment 遮挡与 self-occlusion。

测试内容：
    从 front / side / back 三种观察位置，分别向 head / wrist / knee 发射射线，
    输出：
        target_keypoint
        hit_object_id
        hit_distance
        target_distance
        hit_is_humanoid
        occlusion_source（environment / humanoid_self / none / unknown）

结论：
    - humanoid_self_occlusion_supported_pred 的取值依据（是否命中 Humanoid link）
      输出 supported / unsupported。

运行命令：
    python scripts/debug_humanoid_self_occlusion.py \
        --config configs/mvp50_humanoid.yaml
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v5.config import load_config
from ea_avs_v5.habitat_runner import HabitatRunner
from ea_avs_v5.humanoid_manager import HumanoidManager
from ea_avs_v5.raycast_utils import cast_ray_to_point

# 参与测试的关键点及其对应 URDF link
TEST_KEYPOINTS = ["head", "left_wrist", "left_knee"]

# 观察位置（相对人体根节点）
OBS_POSITIONS = {
    "front": np.array([2.2, 0.0, 0.0]),
    "side": np.array([0.0, 0.0, 2.2]),
    "back": np.array([-2.2, 0.0, 0.0]),
}


def main():
    parser = argparse.ArgumentParser(description="v5.0 Humanoid self-occlusion 测试")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置")
    args = parser.parse_args()

    config = load_config(args.config)
    mcfg = config["camera"]
    runner = HabitatRunner(config)

    # 记录是否命中过 Humanoid link（决定 self-occlusion 支持结论）
    any_humanoid_hit = False

    try:
        manager = HumanoidManager(runner, config)
        manager.load()
        pt = runner.sample_navigable_point()
        hpos = np.array(pt, dtype=np.float32)
        manager.set_base_pose(hpos, 0.0)
        manager.set_pose("standing")

        humanoid_ids = manager.get_humanoid_object_ids()
        print("=" * 70)
        print(f"Humanoid object_id: {sorted(humanoid_ids)[:10]} ... "
              f"(共 {len(humanoid_ids)} 个 object id)")
        print(
            "从 front / side / back 观察位置向关键点发射射线，"
            "判断是否命中 Humanoid 本体")
        print("=" * 70)

        obj = manager.sim_obj
        cam_h = mcfg["camera_height"]

        for obs_name, rel in OBS_POSITIONS.items():
            cam_base = hpos + np.array(rel, dtype=np.float32)
            cam_base[1] = hpos[1]
            cam_pos = cam_base + np.array([0.0, cam_h, 0.0])
            print(f"\n--- 观察位置: {obs_name} (camera={np.round(cam_base,2)}) ---")
            for kp in TEST_KEYPOINTS:
                try:
                    link_id = obj.get_link_id_from_name(kp)
                    kpos = np.array(
                        obj.get_link_scene_node(link_id).transformation.translation)
                except Exception as e:
                    print(f"  {kp:12s}: 取 link 世界坐标失败: {e}")
                    continue

                ray = cast_ray_to_point(
                    runner, cam_pos, kpos,
                    target_tolerance=config["occlusion"].get(
                        "target_tolerance", 0.08),
                    humanoid_object_ids=humanoid_ids,
                )
                hit_self = bool(ray.get("hit_is_humanoid", False))
                any_humanoid_hit = any_humanoid_hit or hit_self
                print(
                    f"  {kp:12s}: hit={ray['hit']} "
                    f"hit_obj_id={ray.get('hit_object_id')} "
                    f"hit_dist={ray['hit_distance']:.3f} "
                    f"target_dist={ray['target_distance']:.3f} "
                    f"is_humanoid={hit_self} "
                    f"source={ray.get('occlusion_source')}",
                )

        print("\n" + "=" * 70)
        if any_humanoid_hit:
            print("✅ ray casting 能命中 Humanoid articulated body links")
            print("   → self-occlusion 可区分 ⇒ "
                  "humanoid_self_occlusion_supported_pred = True/1")
            print("   → 可将配置 humanoid.raycast_self_occlusion_supported 设为 true")
        else:
            print("⚠️ ray casting 未命中 Humanoid 本体（collision mesh 不含皮肤）")
            print("   → self-occlusion predictor 不支持 ⇒ pred=environment, "
                  "true=environment+rendered self-occlusion（文档 Case B）")
        print("=" * 70)

    finally:
        runner.close()


if __name__ == "__main__":
    main()