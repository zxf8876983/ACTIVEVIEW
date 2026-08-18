#!/usr/bin/env python
"""
Humanoid 冒烟测试 —— smoke_test_humanoid.py
=============================================

v5.0 三阶段验收：
    Stage A：官方 neutral_0 成功加载；standing 成功；
            Humanoid 渲染由 GT 锚定 depth 验证（非图像方差）
    Stage B：walking motion 驱动 mesh 且关节变化；base 位移连续、
            不瞬移回原点（teleport 检查）；位移不超过阈值

运行命令：
    python scripts/smoke_test_humanoid.py --config configs/mvp50_humanoid.yaml

每个阶段失败时明确提示失败点；不静默跳过 Humanoid 相关错误。
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


def main():
    parser = argparse.ArgumentParser(description="EA-AVS-MVP v5.0 Humanoid smoke test")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置")
    parser.add_argument("--num-walk-frames", type=int, default=10,
                        help="walking 动画推进帧数")
    args = parser.parse_args()

    config = load_config(args.config)
    mcfg = config["camera"]
    vcfg = config.get("humanoid_validation", {})
    max_walk_disp = vcfg.get("max_short_walk_displacement", 2.0)

    print("=" * 60)
    print("v5.0 Humanoid Smoke Test")
    print(f"  avatar: {config['humanoid']['avatar_name']}")
    print(f"  assets_root: {config['humanoid']['assets_root']}")
    print("=" * 60)

    runner = HabitatRunner(config)
    manager = HumanoidManager(runner, config)
    results = []

    def report(stage, ok, detail=""):
        results.append((stage, ok))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {stage} {detail}")

    try:
        # ---- Stage A: 加载 + standing + GT-锚定渲染验证 ----
        try:
            manager.load()
            report("Humanoid asset loaded", True,
                   f": {config['humanoid']['avatar_name']}")
        except Exception as e:
            report("Humanoid asset loaded", False, f": {e}")
            print("\n❌ 阶段失败（加载）——按文档要求，加载失败则停止，不继续 NBV 集成。")
            manager.close()
            runner.close()
            sys.exit(1)

        report("Humanoid object created", True)

        pt = runner.sample_navigable_point()
        hpos = np.array(pt, dtype=np.float32)
        manager.set_base_pose(hpos, 0.0)
        print(f"  Humanoid 置于: {np.round(pt, 3)}")

        try:
            manager.set_pose("standing")
            report("Standing pose applied", True)
        except Exception as e:
            report("Standing pose applied", False, f": {e}")

        # 显式设置 semantic id（semantic 验证用）
        if config["humanoid"].get("semantic_enabled", True):
            manager.assign_semantic_id_to_links(
                config["humanoid"].get("semantic_id", 100))

        try:
            gt_result = get_humanoid_gt_skeleton(manager, strict=True)
            skeleton = gt_result["skeleton"]
            report("GT skeleton 15/15 links", True,
                   f": source={gt_result['source']}")
        except Exception as e:
            report("GT skeleton 15/15 links", False, f": {e}")
            skeleton = None

        # 相机 look-at human（不硬编码 yaw）
        hy = hpos[1]
        camera_base = np.array([hpos[0] + 2.0, hy, hpos[2]], dtype=np.float32)
        cam_yaw = compute_look_at_yaw(camera_base, hpos)
        obs = runner.render_at(camera_base, cam_yaw)
        print(f"  RGB shape: {None if obs['rgb'] is None else obs['rgb'].shape} "
              f"Depth shape: {None if obs['depth'] is None else obs['depth'].shape}")
        print(f"  camera look-at yaw: {np.degrees(cam_yaw):.1f}°")

        rgb_ok = obs["rgb"] is not None
        depth_ok = obs["depth"] is not None
        report("Depth available", depth_ok)

        # Humanoid 渲染验证：semantic -> GT-depth proxy（不用图像方差）
        if skeleton is not None and depth_ok:
            from ea_avs_v5.humanoid_validation import compute_humanoid_render_stats
            semantic_ids = [config["humanoid"].get("semantic_id", 100)] \
                if config["humanoid"].get("semantic_enabled", True) else []
            rs = compute_humanoid_render_stats(
                obs, config, camera_base, cam_yaw, skeleton, semantic_ids)
            report(
                "RGB contains humanoid (semantic/GT-depth)",
                rs["humanoid_render_success"],
                f": validation={rs['humanoid_validation_source']} "
                f"pixel={rs.get('humanoid_semantic_pixel_count') or rs.get('humanoid_proxy_pixel_count', 0)} "
                f"match={rs.get('humanoid_proxy_match_ratio', rs.get('humanoid_depth_valid_ratio', 0)):.2f}",
            )
        else:
            report("RGB contains humanoid (semantic/GT-depth)", False,
                   ": 无法获取 GT skeleton 或 depth")

        # ---- Stage B: walking motion（验证动画链路；joint 变化 + 无 teleport）----
        walk_fail = False
        base_before = None
        joints0 = None
        try:
            manager.set_pose("walking")
            base_before = np.asarray(manager.get_state().base_position).copy()
            joints0 = np.asarray(manager.sim_obj.joint_positions).copy()
            max_disp = 0.0
            changed = False
            for i in range(args.num_walk_frames):
                manager.step_motion(1.0 / 30.0)
                joints = np.asarray(manager.sim_obj.joint_positions).copy()
                if np.abs(joints - joints0).sum() > 1e-6:
                    changed = True
                base_now = np.asarray(manager.get_state().base_position).copy()
                disp = float(np.linalg.norm(base_now - base_before))
                max_disp = max(max_disp, disp)
            base_after = np.asarray(manager.get_state().base_position).copy()
            jump = float(np.linalg.norm(base_after))
            origin_jump = bool(
                jump < 0.2 and np.linalg.norm(base_before) > 1.0)
            disp_ok = bool(max_disp <= max_walk_disp)
            report(
                "Walking motion update",
                changed and disp_ok and not origin_jump,
                f": {args.num_walk_frames} 帧 joints_changed={changed} "
                f"base_disp_max={max_disp:.3f}m "
                f"base_after_norm={jump:.3f} origin_jump={origin_jump}",
            )
        except Exception as e:
            walk_fail = True
            report("Walking motion update", False, f": {e}")

        # 用 standing 收尾
        try:
            manager.set_pose("standing")
        except Exception:
            pass

    except Exception as e:
        print(f"\n❌ 冒烟测试异常: {e}")
        report("Smoke test overall", False)
    finally:
        manager.close()
        runner.close()

    print("\n" + "=" * 60)
    ok_count = sum(1 for _, ok in results if ok)
    fail = [s for s, ok in results if not ok]
    print(f"通过 {ok_count}/{len(results)}")
    if fail:
        print(f"失败阶段: {fail}")
        sys.exit(1)
    print("✅ Humanoid smoke test 全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()