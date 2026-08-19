"""
静态场景射线检测验证脚本 —— debug_static_scene_raycast.py
=========================================================

功能：
    验证 HabitatRunner.cast_ray_static_scene 与 cast_ray 的行为差异。
    关键验收：
        当射线经过真实 Humanoid 身体时：
        - full_collision_ray (GT 路径) 会击中 Humanoid 铰接碰撞体 (object_id > 0)；
        - static_scene_ray (Estimated 路径) 必须自动忽略 Humanoid 身体，
          仅允许命中 stage_id 或返回 clear，绝不返回 Humanoid object ID。
"""

import argparse
import sys
import numpy as np
import habitat_sim

from ea_avs_v6.config import load_config
from ea_avs_v6.habitat_runner import HabitatRunner
from ea_avs_v6.humanoid_manager import HumanoidManager
from ea_avs_v6.humanoid_skeleton_adapter import get_humanoid_gt_skeleton
from ea_avs_v6.raycast_utils import cast_ray_to_point, cast_ray_to_estimated_point


def main():
    parser = argparse.ArgumentParser(description="Debug Static Scene Raycast vs Full Collision")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp60_estimated_state.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("\n=======================================================")
    print(" Debug Static Scene Raycast vs Full Collision Raycast")
    print("=======================================================\n")

    runner = HabitatRunner(config)
    print(f"[HabitatRunner] Resolved stage_id: {runner.stage_id}")

    humanoid = HumanoidManager(runner, config)
    humanoid.load()
    runner.attach_humanoid_manager(humanoid)

    human_pos = runner.sample_navigable_point()
    humanoid.set_base_pose(human_pos, 0.0)
    humanoid.set_pose("standing")

    gt_data = get_humanoid_gt_skeleton(humanoid, strict=True)
    gt_skeleton = gt_data["skeleton"]
    humanoid_oids = humanoid.get_humanoid_object_ids()

    torso_pos = 0.5 * (gt_skeleton["left_shoulder"] + gt_skeleton["right_shoulder"])
    cam_pos = torso_pos + np.array([0.0, 0.0, 2.0])  # 正前方 2 米处

    print(f"Humanoid Root Pos: {human_pos.tolist()}")
    print(f"Torso Pos:         {torso_pos.tolist()}")
    print(f"Humanoid Object IDs: {humanoid_oids}\n")

    passed = True

    # -------------------------------------------------------------
    # Case 1: 射向 Humanoid Torso (核心验证)
    # -------------------------------------------------------------
    dir_to_torso = (torso_pos - cam_pos) / np.linalg.norm(torso_pos - cam_pos)
    full_ray_torso = runner.cast_ray(cam_pos, dir_to_torso, max_distance=6.0)
    static_ray_torso = runner.cast_ray_static_scene(cam_pos, dir_to_torso, max_distance=6.0)

    print("--- [Case 1: Ray Cast Directly at Humanoid Torso (max_dist=6.0m)] ---")
    print(f"Full Collision Ray:   has_hits={full_ray_torso['has_hits']}, dist={full_ray_torso['hit_distance']:.3f}m, obj_id={full_ray_torso['hit_object_id']} (is_humanoid={full_ray_torso['hit_object_id'] in humanoid_oids})")
    print(f"Static Scene Ray:     has_hits={static_ray_torso['has_hits']}, dist={static_ray_torso['hit_distance']}, obj_id={static_ray_torso['hit_object_id']}, source={static_ray_torso['hit_source']}")

    est_res = cast_ray_to_estimated_point(runner, cam_pos, torso_pos)
    gt_res = cast_ray_to_point(runner, cam_pos, torso_pos, humanoid_object_ids=humanoid_oids)
    print(f"Estimated Evaluator:  occluded={est_res['occluded']}, valid={est_res['valid']}, cause={est_res['occlusion_source']}")
    print(f"GT Evaluator:         occluded={gt_res['occluded']}, valid={gt_res['valid']}, cause={gt_res['occlusion_source']}")

    if static_ray_torso["has_hits"]:
        if static_ray_torso["hit_object_id"] != runner.stage_id:
            print(f"[FAIL] static_ray_torso hit_object_id {static_ray_torso['hit_object_id']} != stage_id {runner.stage_id}")
            passed = False
        if static_ray_torso["hit_object_id"] in humanoid_oids:
            print("[FAIL] static_ray_torso hit Humanoid articulated object ID!")
            passed = False

    # -------------------------------------------------------------
    # Case 2: 射向地面 / 墙体
    # -------------------------------------------------------------
    down_dir = np.array([0.0, -1.0, 0.0])
    full_ray_floor = runner.cast_ray(cam_pos, down_dir, max_distance=5.0)
    static_ray_floor = runner.cast_ray_static_scene(cam_pos, down_dir, max_distance=5.0)

    print("\n--- [Case 2: Ray Cast Straight Down to Floor (max_dist=5.0m)] ---")
    print(f"Full Collision Ray:   has_hits={full_ray_floor['has_hits']}, dist={full_ray_floor['hit_distance']:.3f}m, obj_id={full_ray_floor['hit_object_id']}")
    print(f"Static Scene Ray:     has_hits={static_ray_floor['has_hits']}, dist={static_ray_floor['hit_distance']:.3f}m, obj_id={static_ray_floor['hit_object_id']}, source={static_ray_floor['hit_source']}")

    if static_ray_floor["has_hits"]:
        if static_ray_floor["hit_object_id"] != runner.stage_id:
            print(f"[FAIL] static_ray_floor hit_object_id {static_ray_floor['hit_object_id']} != stage_id {runner.stage_id}")
            passed = False

    # -------------------------------------------------------------
    # Case 3: 射向空旷天花板 / 空旷无障碍区域
    # -------------------------------------------------------------
    up_dir = np.array([0.0, 1.0, 0.0])
    static_ray_up = runner.cast_ray_static_scene(cam_pos, up_dir, max_distance=0.5)

    print("\n--- [Case 3: Short Ray Cast Upwards (max_dist=0.5m)] ---")
    print(f"Static Scene Ray:     has_hits={static_ray_up['has_hits']}, dist={static_ray_up['hit_distance']}, source={static_ray_up['hit_source']}")

    # -------------------------------------------------------------
    # 核心断言与退出码
    # -------------------------------------------------------------
    if full_ray_torso["has_hits"] and full_ray_torso["hit_object_id"] in humanoid_oids:
        if static_ray_torso["has_hits"] and static_ray_torso["hit_object_id"] != runner.stage_id:
            passed = False

    runner.close()

    if passed:
        print("\n[VERIFICATION SUCCESS] Static scene raycast cleanly ignores Humanoid articulated collision body! (PASS)\n")
        sys.exit(0)
    else:
        print("\n[VERIFICATION FAILURE] Static scene raycast failed verification! (FAIL)\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
