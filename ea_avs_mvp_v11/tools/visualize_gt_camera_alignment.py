#!/usr/bin/env python3
"""
ACTIVEVIEW v10.0 GT 空间转换与相机坐标系对齐全流程验证可视化器 —— visualize_gt_camera_alignment.py
========================================================================================

职责：
    1. 严格检验 Habitat GT World Skeleton -> GT Camera Skeleton -> Estimated Skeleton 坐标流；
    2. 验证坐标系定义：
       - +X: Right (右)
       - +Y: Up (上)
       - +Z: Forward / Depth (前 / 深度)
       - 验证 Habitat OpenGL (-Z forward) 到标准相机系 (+Z forward) 转换无误；
    3. 验证运动学几何关系：head_y > pelvis_y > ankle_y；
    4. 生成 4 面板高分辨率全流程对齐诊断图，直观确认空间变换正确性。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.paths import get_repo_root, get_v10_dataset_root
from ea_avs_mvp_v10.core.types import V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.evaluation.skeleton_alignment import (
    extract_aligned_joint_pairs,
    transform_gt_to_camera_frame,
)
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v10.visualization.skeleton_visualizer import SkeletonVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualize_gt_camera_alignment")


def verify_and_plot_gt_camera_alignment(
    sample_id: str = "v10_sample_000000",
    output_path: Optional[Path] = None,
) -> Path:
    dataset_root = get_v10_dataset_root()
    skel_def = get_skeleton_definition()
    pipeline = V10PerceptionPipeline(dataset_root=dataset_root)

    with open(dataset_root / "metadata" / "samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    sample_dict = next((s for s in samples if s["sample_id"] == sample_id), None)
    if not sample_dict:
        raise ValueError(f"Sample {sample_id} not found!")

    sample_obj = V10Sample.from_dict(sample_dict)
    rgb_img = np.array(Image.open(dataset_root / sample_obj.rgb_path))
    depth_map = np.load(dataset_root / sample_obj.depth_path)

    # 1. 估计骨架 (RGB-D 驱动)
    skel, _ = pipeline.process_sample(sample_obj, rgb_image=rgb_img, depth_map=depth_map, save_outputs=False)

    # 2. GT 骨架 (世界系)
    gt_file = dataset_root / "ground_truth" / "skeleton" / f"{sample_id}.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_world_dict = json.load(f)["joints_3d"]

    c2w = np.array(sample_obj.camera_pose.matrix_4x4, dtype=np.float32)
    cam_pos = np.array(sample_obj.camera_pose.position, dtype=np.float32)

    # 3. 转换 GT 骨架至相机系
    gt_cam_dict = transform_gt_to_camera_frame(gt_world_dict, c2w)

    # 4. 提取匹配关节对
    P_est, P_gt_cam, joint_names = extract_aligned_joint_pairs(
        estimated_skeleton=skel,
        gt_joints_dict=gt_world_dict,
        camera_matrix_4x4=c2w,
        skel_def=skel_def,
    )

    # 5. 打印校验数值
    print(f"\n=======================================================")
    print(f"GT CAMERA TRANSFORM VERIFICATION: {sample_id}")
    print(f"Action: {sample_obj.action_label} | Camera Pos: {cam_pos}")
    print(f"=======================================================")
    print(f"{'Joint Name':16s} | {'GT World (m)':24s} | {'GT Cam (m)':24s} | {'Est Cam (m)':24s}")
    print("-" * 95)
    for name in joint_names:
        w_p = gt_world_dict.get(name, [0, 0, 0])
        c_p = gt_cam_dict.get(name, np.zeros(3))
        # 查找对应 estimated joint
        gt_idx = joint_names.index(name)
        e_p = P_est[gt_idx]
        print(f"{name:16s} | ({w_p[0]:6.3f}, {w_p[1]:6.3f}, {w_p[2]:6.3f}) | ({c_p[0]:6.3f}, {c_p[1]:6.3f}, {c_p[2]:6.3f}) | ({e_p[0]:6.3f}, {e_p[1]:6.3f}, {e_p[2]:6.3f})")

    # 运动学与方向断言校验
    head_c = gt_cam_dict["head"]
    pelvis_c = gt_cam_dict["pelvis"]
    l_ankle_c = gt_cam_dict["left_ankle"]

    print("\n--- Diagnostic Assertions ---")
    print(f"1. [Y-Axis / Height]: Head Y ({head_c[1]:.3f}m) > Pelvis Y ({pelvis_c[1]:.3f}m) > Ankle Y ({l_ankle_c[1]:.3f}m)")
    assert head_c[1] > pelvis_c[1] > l_ankle_c[1], "GT Camera Y-axis hierarchy failure!"
    print("   -> PASS: +Y is strictly UP in camera coordinate system.")

    print(f"2. [Z-Axis / Depth]:  Depth ranges from {min(p[2] for p in gt_cam_dict.values()):.3f}m to {max(p[2] for p in gt_cam_dict.values()):.3f}m")
    assert all(p[2] > 0.5 for p in gt_cam_dict.values()), "GT Camera Z-axis depth must be positive!"
    print("   -> PASS: +Z is strictly FORWARD/DEPTH in camera coordinate system.")

    # 6. 绘图 4 Panel
    fig = plt.figure(figsize=(22, 5.5))

    # Panel 1: RGB Image
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f"1. RGB Observation\n({sample_id} | {sample_obj.action_label.upper()})", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # Panel 2: GT World Skeleton + Camera Frustum in World Space
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    gt_w_pts = np.array(list(gt_world_dict.values()))
    ax2.scatter(gt_w_pts[:, 0], gt_w_pts[:, 2], gt_w_pts[:, 1], color="#E74C3C", s=25, label="GT World Joints")
    # 绘制相机位置
    ax2.scatter([cam_pos[0]], [cam_pos[2]], [cam_pos[1]], color="#2980B9", s=60, marker="^", label="Camera Pos")
    ax2.plot([cam_pos[0], gt_world_dict["pelvis"][0]],
             [cam_pos[2], gt_world_dict["pelvis"][2]],
             [cam_pos[1], gt_world_dict["pelvis"][1]], color="#2980B9", linestyle=":", label="Camera Optical Ray")
    ax2.set_title("2. GT Skeleton in World Frame\n(Habitat Scene Coordinates)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("World X (m)", fontsize=7)
    ax2.set_ylabel("World Z (m)", fontsize=7)
    ax2.set_zlabel("World Y (m)", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # Panel 3: GT Skeleton in Camera Frame (+X: Right, +Y: Up, +Z: Depth)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    gt_c_pts = np.array(list(gt_cam_dict.values()))
    ax3.scatter(gt_c_pts[:, 0], gt_c_pts[:, 2], gt_c_pts[:, 1], color="#E67E22", s=30, label="GT Camera Joints")
    gt_edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ("neck", "head"), ("pelvis", "spine3"), ("spine3", "neck"),
    ]
    for j1_n, j2_n in gt_edges:
        if j1_n in gt_cam_dict and j2_n in gt_cam_dict:
            p1, p2 = gt_cam_dict[j1_n], gt_cam_dict[j2_n]
            ax3.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], color="#E67E22", linewidth=2.0)

    gt_c_span = max(np.ptp(gt_c_pts[:, 0]), np.ptp(gt_c_pts[:, 1]), np.ptp(gt_c_pts[:, 2]), 0.8)
    gt_c_mid = np.mean(gt_c_pts, axis=0)
    ax3.set_xlim(gt_c_mid[0] - gt_c_span/2, gt_c_mid[0] + gt_c_span/2)
    ax3.set_ylim(gt_c_mid[2] - gt_c_span/2, gt_c_mid[2] + gt_c_span/2)
    ax3.set_zlim(gt_c_mid[1] - gt_c_span/2, gt_c_mid[1] + gt_c_span/2)
    ax3.set_title("3. GT Skeleton in Camera Frame\n(+X: Right, +Y: Up, +Z: Depth)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Cam X (m)", fontsize=7)
    ax3.set_ylabel("Cam Z (m)", fontsize=7)
    ax3.set_zlabel("Cam Y (m)", fontsize=7)
    ax3.view_init(elev=10, azim=-70)
    ax3.legend(loc="upper right", fontsize=7)

    # Panel 4: GT Camera vs Estimated 3D Pose Overlay
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(P_gt_cam[:, 0], P_gt_cam[:, 2], P_gt_cam[:, 1], color="#E67E22", s=35, label="GT in Cam Frame")
    ax4.scatter(P_est[:, 0], P_est[:, 2], P_est[:, 1], color="#3498DB", s=35, label="Estimated (RGB-D)")
    for i in range(len(joint_names)):
        ax4.plot([P_gt_cam[i, 0], P_est[i, 0]],
                 [P_gt_cam[i, 2], P_est[i, 2]],
                 [P_gt_cam[i, 1], P_est[i, 1]],
                 color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    all_c_pts = np.vstack([P_gt_cam, P_est])
    all_c_span = max(np.ptp(all_c_pts[:, 0]), np.ptp(all_c_pts[:, 1]), np.ptp(all_c_pts[:, 2]), 0.8)
    all_c_mid = np.mean(all_c_pts, axis=0)
    ax4.set_xlim(all_c_mid[0] - all_c_span/2, all_c_mid[0] + all_c_span/2)
    ax4.set_ylim(all_c_mid[2] - all_c_span/2, all_c_mid[2] + all_c_span/2)
    ax4.set_zlim(all_c_mid[1] - all_c_span/2, all_c_mid[1] + all_c_span/2)
    ax4.set_title("4. Spatial Alignment: GT vs Estimated\n(Confirmed Coordinate Parity)", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Cam X (m)", fontsize=7)
    ax4.set_ylabel("Cam Z (m)", fontsize=7)
    ax4.set_zlabel("Cam Y (m)", fontsize=7)
    ax4.view_init(elev=10, azim=-70)
    ax4.legend(loc="upper right", fontsize=7)

    plt.suptitle(
        f"ACTIVEVIEW v10.0 GT Camera Transformation & Coordinate Frame Verification | {sample_id}",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout()

    out_p = output_path or (get_repo_root() / "ea_avs_mvp_v10" / "examples" / "v10_phase2_demo" / "gt_camera_alignment_verification.png")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved GT camera alignment verification visualization to: %s", out_p)
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Verify GT Camera Transform and Alignment")
    parser.add_argument("--sample_id", type=str, default="v10_sample_000000", help="Sample ID")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    verify_and_plot_gt_camera_alignment(args.sample_id, out_p)


if __name__ == "__main__":
    main()
