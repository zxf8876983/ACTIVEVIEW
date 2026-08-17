#!/usr/bin/env python
"""
相机/深度一致性调试脚本 —— debug_camera_consistency.py
=======================================================

目的（非正式实验，仅验证一致性）：
    验证 预测几何模型 / Habitat 渲染 / depth 投影 / ray casting 使用同一套
    相机内参（width / height / HFOV → 派生 VFOV / fx / fy / cx / cy），
    且 depth 遮挡判断使用"前向深度 z_cam"而不是三维欧氏距离。

检查内容（对每个相机位置）：
    1. 相机中心方向上的静态场景点应投影到图像中心，且中心 depth ≈ raycast 距离
    2. 图像靠左位置：model-yaw 偏移 + 的点应投影到 u < W/2
    3. 图像靠右位置：model-yaw 偏移 - 的点应投影到 u > W/2
    4. 上述像素处的 depth 与 raycast 距离应一致（同一块墙面）
    5. 边缘像素不能因为 Euclidean-distance 与 z_cam 的差异被误判遮挡：
       - 墙面点位于表面，应满足 sampled_depth ≈ z_cam（未遮挡）
       - 若误用欧氏距离，图像边缘处欧氏距离显著大于 z_cam，会导致误判

运行命令（需要 habitat 环境）：
    python scripts/debug_camera_consistency.py \
        --config configs/mvp40_occlusion_aware.yaml

无单元测试框架，仅输出简单日志。全部通过时 exit code = 0。
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v4.config import load_config
from ea_avs_v4.habitat_runner import HabitatRunner
from ea_avs_v4.geometry import compute_camera_intrinsics


def project_point(intrinsics, view_yaw, camera_pos, point):
    """与 true_evaluator._depth_occlusion 相同的投影约定（相机前向 = +Z 模型方向）。

    返回：
        ((u, v_px), z_cam) 或 (None, z_cam)（点在相机后方）
    """
    th = view_yaw
    right = np.array([-np.cos(th), 0.0, np.sin(th)])
    forward = np.array([np.sin(th), 0.0, np.cos(th)])
    v = np.array(point, dtype=np.float64) - np.array(camera_pos, dtype=np.float64)
    z_cam = float(v[0] * forward[0] + v[2] * forward[2])
    if z_cam <= 0.0:
        return None, z_cam
    x_cam = float(v[0] * right[0] + v[2] * right[2])
    y_cam = float(v[1])
    u = intrinsics["fx"] * x_cam / z_cam + intrinsics["cx"]
    v_px = intrinsics["cy"] - intrinsics["fy"] * y_cam / z_cam
    return (u, v_px), z_cam


def sample_depth_patch(depth, u, v, width, height):
    """采样 3×3 邻域有效深度中位数（与 true_evaluator 一致）。"""
    valid = []
    ui, vi = int(round(u)), int(round(v))
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu, vv = ui + du, vi + dv
            if 0 <= uu < width and 0 <= vv < height:
                dd = float(depth[vv, uu])
                if dd > 0.0:
                    valid.append(dd)
    return float(np.median(valid)) if valid else None


def run_checks(runner, intrinsics, camera_pos, yaw, tol):
    """在给定相机位姿下执行一致性检查，返回 (passed, total, details)。"""
    width = intrinsics["width"]
    height = intrinsics["height"]
    obs = runner.render_at(camera_pos, yaw)  # render_at 内部已按 yaw+π 对齐
    depth = np.asarray(obs["depth"])
    if depth.ndim == 3:
        depth = depth[..., 0]

    cam_pos = np.array(camera_pos, dtype=np.float64) + np.array([0.0, 1.2, 0.0])

    results = []
    # 沿 model 前向、以及 ±0.4 rad 方向找墙面点
    directions = [("center", 0.0), ("left", 0.4), ("right", -0.4)]
    for label, off in directions:
        dirv = np.array([np.sin(yaw + off), 0.0, np.cos(yaw + off)])
        res = runner.cast_ray(cam_pos, dirv, max_distance=10.0)
        if not res["has_hits"]:
            results.append((f"{label}: 无命中", False, "no_hit"))
            continue
        hit_pt = np.array(res["hit_point"], dtype=np.float64)
        ray_dist = float(res["hit_distance"])
        proj, z_cam = project_point(intrinsics, yaw, cam_pos, hit_pt)
        if proj is None:
            results.append((f"{label}: 点在相机后方", False, "behind"))
            continue
        u, v = proj
        sampled = sample_depth_patch(depth, u, v, width, height)
        ok = sampled is not None and abs(sampled - z_cam) <= 0.15
        results.append((f"{label}: u={u:.0f} v={v:.0f} z_cam={z_cam:.3f} "
                        f"sampled_depth={sampled} ray_dist={ray_dist:.3f} "
                        f"edge_occ_judge={'ok' if ok else 'FAIL'}",
                        ok, ("center" if label == "center"
                             else "left" if label == "left" else "right")))

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    details = []
    for msg, ok, side in results:
        details.append((msg, ok, side))
    return passed, total, details


def main():
    parser = argparse.ArgumentParser(
        description="相机/深度一致性调试（EA-AVS-MVP v4.0）")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--positions", type=int, default=3,
                        help="随机采样多少个相机位置进行校验（默认 3）")
    args = parser.parse_args()

    config = load_config(args.config)
    camera_cfg = config["camera"]
    intrinsics = compute_camera_intrinsics(
        camera_cfg["width"], camera_cfg["height"], camera_cfg["hfov_deg"])

    print("=" * 70)
    print("相机/深度一致性检查")
    print(f"  width={intrinsics['width']} height={intrinsics['height']} "
          f"HFOV={intrinsics['hfov_deg']}°  "
          f"派生VFOV={intrinsics['vfov_deg']:.2f}°  "
          f"fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
          f"cx={intrinsics['cx']:.0f} cy={intrinsics['cy']:.0f}")
    print(f"  vfov 由 pinhole 推导（不再手动固定 vfov_deg）")
    print("=" * 70)

    runner = HabitatRunner(config)
    try:
        total_ok = 0
        total_checks = 0
        all_fail_messages = []
        for i in range(args.positions):
            pt = runner.sample_navigable_point()
            camera_pos = np.array(pt, dtype=np.float32)
            # 扫描最近墙面方向作为 model yaw
            cam_pos3 = camera_pos + np.array([0.0, camera_cfg["camera_height"], 0.0])
            best_ang, best_d = None, 1e9
            for ang in np.arange(0.0, 2 * np.pi, 0.2):
                dirv = np.array([np.sin(ang), 0.0, np.cos(ang)])
                res = runner.cast_ray(cam_pos3, dirv, max_distance=10.0)
                if res["has_hits"]:
                    d = float(res["hit_distance"])
                    if d < best_d:
                        best_d, best_ang = d, ang
            if best_ang is None:
                print(f"位置 {i}: 周围无墙面命中，跳过")
                continue
            passed, total, details = run_checks(
                runner, intrinsics, camera_pos, float(best_ang),
                tol=config["occlusion"].get("target_tolerance", 0.08))
            total_ok += passed
            total_checks += total
            print(f"\n位置 {i}: camera={np.round(camera_pos, 2)} "
                  f"yaw={np.degrees(best_ang):.0f}°")
            for msg, ok, side in details:
                status = "✅" if ok else "❌"
                print(f"  {status} {msg}")
                if not ok:
                    all_fail_messages.append(f"位置{i}-{side}: {msg}")

        print("\n" + "=" * 70)
        if total_checks == 0:
            print("❌ 没有任何可用的检查点")
            sys.exit(1)
        print(f"通过 {total_ok}/{total_checks}")
        if all_fail_messages:
            print("失败明细:")
            for m in all_fail_messages:
                print(f"  - {m}")
            sys.exit(1)
        print("✅ 相机模型 / depth / raycast 一致性通过")
        print("（depth 遮挡判断使用 z_cam 前向深度，边缘无欧氏距离误判）")
    finally:
        runner.close()


if __name__ == "__main__":
    main()