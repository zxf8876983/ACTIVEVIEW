#!/usr/bin/env python
"""
Humanoid 冒烟测试 —— smoke_test_humanoid.py
=============================================

v5.0 第一阶段验收：
    Stage A：官方 neutral_0 成功加载；standing 成功；RGB 中存在 Humanoid；Depth 可输出
    Stage B：walking motion 能驱动 mesh；连续若干帧人体姿态发生变化；不崩溃

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


def check_rgb_contains_humanoid(rgb):
    """粗略检查 RGB 中心附近是否非纯背景（有物体/人体）。

    返回 'PASS' 或 'UNKNOWN'（不做严格检测，仅供冒烟提示）。
    """
    if rgb is None:
        return False
    # 中心区域像素方差
    patch = rgb[160:320, 240:400]
    std = float(np.asarray(patch, dtype=np.float32).std())
    return std > 10.0


def main():
    parser = argparse.ArgumentParser(description="EA-AVS-MVP v5.0 Humanoid smoke test")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置")
    parser.add_argument("--num-walk-frames", type=int, default=10,
                        help="walking 动画推进帧数")
    args = parser.parse_args()

    config = load_config(args.config)
    sim_config = config  # noqa

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
        # ---- Stage A: 加载 + standing + render ----
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

        # 放置到导航点
        pt = runner.sample_navigable_point()
        manager.set_base_pose(np.array(pt, dtype=np.float32), 0.0)
        print(f"  Humanoid 置于: {np.round(pt, 3)}")

        # standing
        try:
            manager.set_pose("standing")
            report("Standing pose applied", True)
        except Exception as e:
            report("Standing pose applied", False, f": {e}")

        # 将相机放到 Humanoid 前方，渲染 RGB+depth
        hy = manager.get_state().base_position[1]
        camera_base = np.array(
            [pt[0] + 2.0, hy, pt[2]], dtype=np.float32)
        obs = runner.render_at(camera_base, 0.0)
        print(f"  RGB shape: {None if obs['rgb'] is None else obs['rgb'].shape} "
              f"Depth shape: {None if obs['depth'] is None else obs['depth'].shape}")

        rgb_ok = obs["rgb"] is not None
        depth_ok = obs["depth"] is not None
        report("RGB contains humanoid",
               True if (rgb_ok and check_rgb_contains_humanoid(obs["rgb"])) else False,
               "UNKNOWN" if (rgb_ok and not check_rgb_contains_humanoid(obs["rgb"])) else "")
        report("Depth available", depth_ok)

        # ---- Stage B: walking motion ----
        walk_fail = False
        try:
            manager.set_pose("walking")
            # 记录初始关节状态
            joints0 = np.asarray(manager.sim_obj.joint_positions).copy()
            changed = False
            for i in range(args.num_walk_frames):
                manager.step_motion(1.0 / 30.0)
                joints = np.asarray(manager.sim_obj.joint_positions).copy()
                if np.abs(joints - joints0).sum() > 1e-6:
                    changed = True
            report("Walking motion update",
                   True if changed else False,
                   f": {args.num_walk_frames} 帧，姿态已变化={changed}")
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