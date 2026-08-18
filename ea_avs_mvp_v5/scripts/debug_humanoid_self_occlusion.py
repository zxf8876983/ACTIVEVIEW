#!/usr/bin/env python
"""
Humanoid self-occlusion ray test —— debug_humanoid_self_occlusion.py
=====================================================================

目标（v5.0 第三轮）：
    判断当前 Habitat ray casting 能否区分 5 态遮挡：
        none / target_surface / environment / humanoid_self / unknown
    只有同时观察到 target_surface 与 humanoid_self 两类样例，分类机制才算
    得到正向验证（validation = "validated"），否则 "inconclusive"。

测试内容：
    从 front / side / back 三种观察位置，向 head/left_wrist/right_wrist/
    left_knee/right_knee 发射射线，对每次命中输出：
        target_keypoint / target_link_name / target_link_id / target_link_object_id
        hit_object_id / hit_link_name（反查）
        hit_distance / target_distance
        hit_is_humanoid / hit_is_target_surface
        occlusion_source

运行命令：
    python scripts/debug_humanoid_self_occlusion.py \
        --config configs/mvp50_humanoid.yaml
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v5.config import load_config
from ea_avs_v5.habitat_runner import HabitatRunner
from ea_avs_v5.humanoid_manager import HumanoidManager
from ea_avs_v5.raycast_utils import cast_ray_to_point

TEST_KEYPOINTS = ["head", "left_wrist", "right_wrist", "left_knee", "right_knee"]

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

    source_counter = Counter()

    try:
        manager = HumanoidManager(runner, config)
        manager.load()
        pt = runner.sample_navigable_point()
        hpos = np.array(pt, dtype=np.float32)
        manager.set_base_pose(hpos, 0.0)
        manager.set_pose("standing")

        humanoid_ids = manager.get_humanoid_object_ids()
        link_meta = manager.get_humanoid_link_metadata()

        # object_id -> link_name 反查表
        oid_to_name = {
            v["object_id"]: name for name, v in link_meta.items()
        }

        print("=" * 78)
        print(f"Humanoid 共 {len(humanoid_ids)} 个 Habitat object id "
              f"(root+links)，raycast hit.object_id 可与之直接匹配")
        print("从 front / side / back 观察位置向关键点发射射线（5 态分类）")
        print("=" * 78)

        obj = manager.sim_obj
        cam_h = mcfg["camera_height"]

        for obs_name, rel in OBS_POSITIONS.items():
            cam_base = hpos + np.array(rel, dtype=np.float32)
            cam_base[1] = hpos[1]
            cam_pos = cam_base + np.array([0.0, cam_h, 0.0])
            print(f"\n--- 观察位置: {obs_name} ---")
            for kp in TEST_KEYPOINTS:
                m = link_meta.get(kp)
                if m is None:
                    print(f"  {kp:12s}: 无 link 元数据，跳过")
                    continue
                try:
                    kpos = np.array(
                        obj.get_link_scene_node(m["link_id"]).transformation.translation)
                except Exception as e:
                    print(f"  {kp:12s}: 取 link 世界坐标失败: {e}")
                    continue

                ray = cast_ray_to_point(
                    runner, cam_pos, kpos,
                    target_tolerance=config["occlusion"].get(
                        "target_tolerance", 0.08),
                    humanoid_object_ids=humanoid_ids,
                    target_link_object_ids={m["object_id"]} if m.get("object_id") is not None else None,
                )
                src = ray.get("occlusion_source", "unknown")
                source_counter[src] += 1

                hoid = ray.get("hit_object_id")
                hit_name = oid_to_name.get(hoid) if hoid is not None else None
                print(
                    f"  {kp:12s} (link_id={m['link_id']}, obj_id={m['object_id']}): "
                    f"source={src:<14} "
                    f"hit_obj={hoid} hit_link='{hit_name}' "
                    f"is_humanoid={ray['hit_is_humanoid']} "
                    f"is_target_surface={ray['hit_is_target_surface']} "
                    f"d_hit={ray['hit_distance']:.3f} d_tgt={ray['target_distance']:.3f}",
                )

        # ---- 汇总与判定 ----
        print("\n" + "=" * 78)
        print("5 态分类计数:", dict(source_counter))
        ts = source_counter.get("target_surface", 0)
        hs = source_counter.get("humanoid_self", 0)
        env = source_counter.get("environment", 0)
        unk = source_counter.get("unknown", 0)
        if ts > 0 and hs > 0:
            print("✅ 同时观察到 target_surface 与 humanoid_self 样例 ⇒ "
                  "分类机制正向验证")
            status = "validated"
        elif ts > 0 or hs > 0:
            print("⚠️ 仅观察到一种样例（target_surface 或 humanoid_self 单独存在）"
                  "⇒ 机制部分可见")
            status = "inconclusive"
        else:
            print("⚠️ standing 姿态下未自然出现 target_surface / humanoid_self 样例 "
                  "⇒ 无法下结论")
            status = "inconclusive"
        print(f"self-occlusion validation = {status}")
        print(f"  target_surface_case_count = {ts}")
        print(f"  humanoid_self_case_count = {hs}")
        print(f"  environment_case_count = {env}")
        print(f"  unknown_case_count = {unk}")
        print("建议将 config humanoid.raycast_self_occlusion_status 更新为 "
              f"'{status}'")
        print("=" * 78)

    finally:
        runner.close()


if __name__ == "__main__":
    main()