#!/usr/bin/env python3
"""
ACTIVEVIEW v10.0 估计 3D 骨架与真值 (GT) 空间对比与误差可视化器 —— skeleton_compare_visualizer.py
========================================================================================

职责：
    1. 接收 Estimated 3D Skeleton 与 Habitat GT 3D Skeleton；
    2. 基于 `configs/skeleton_definition.json` 自动提取匹配关节点；
    3. 绘制估计骨架 (蓝色/青色) 与真值骨架 (橙色/洋红) 的 3D 重叠图；
    4. 绘制各关节误差向量 (Error Vector Lines)，直观反映估计偏离方向与距离；
    5. 绘制逐关节误差与分部位误差柱状图；
    6. 保存高分辨率诊断对比图 `overlay_comparison.png`。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.paths import get_v10_dataset_root
from ea_avs_mvp_v10.core.types import V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.evaluation.skeleton_alignment import (
    extract_aligned_joint_pairs,
    transform_gt_to_camera_frame,
)
from ea_avs_mvp_v10.evaluation.skeleton_evaluator import EvaluationMetrics, SkeletonEvaluator
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v10.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition
from ea_avs_mvp_v10.visualization.skeleton_visualizer import SkeletonVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("skeleton_compare_visualizer")


def plot_skeleton_comparison(
    sample_id: str,
    rgb_img: np.ndarray,
    estimated_skeleton: EstimatedSkeleton3D,
    gt_joints_dict: Dict[str, List[float]],
    camera_matrix_4x4: np.ndarray,
    metrics: EvaluationMetrics,
    output_path: Path,
    skel_def: Optional[SkeletonDefinition] = None,
) -> Path:
    """生成估计 3D 骨架与 GT 骨架对比全景图 (RGB 2D, 估计 3D, GT 3D, 空间对齐误差对比, 逐关节误差直方图)。"""
    skel_def = skel_def or get_skeleton_definition()
    visualizer = SkeletonVisualizer(skel_def=skel_def)

    fig = plt.figure(figsize=(24, 5.5))

    # Panel 1: RGB + 2D 骨架
    ax1 = fig.add_subplot(1, 5, 1)
    overlay_2d = visualizer.draw_2d_skeleton_on_rgb(rgb_img, estimated_skeleton)
    ax1.imshow(overlay_2d)
    ax1.set_title(f"1. RGB Observation\nAction: {metrics.action_label.upper()}", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # 提取对齐后的点集与关节名
    P_est, P_gt, joint_names = extract_aligned_joint_pairs(
        estimated_skeleton=estimated_skeleton,
        gt_joints_dict=gt_joints_dict,
        camera_matrix_4x4=camera_matrix_4x4,
        skel_def=skel_def,
    )

    # 根节点中心化 (以 pelvis 为原点对比姿态)
    if "pelvis" in joint_names:
        root_idx = joint_names.index("pelvis")
        p_est_root = P_est[root_idx]
        p_gt_root = P_gt[root_idx]
    else:
        p_est_root = np.mean(P_est, axis=0)
        p_gt_root = np.mean(P_gt, axis=0)

    P_est_rel = P_est - p_est_root
    P_gt_rel = P_gt - p_gt_root

    # 设定标准 3D 坐标范围
    all_pts = np.vstack([P_est_rel, P_gt_rel])
    max_span = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]), np.ptp(all_pts[:, 2]), 0.8)
    mid_x, mid_y, mid_z = np.mean(all_pts, axis=0)

    # Panel 2: Estimated 3D Skeleton (Front View)
    ax2 = fig.add_subplot(1, 5, 2, projection="3d")
    visualizer.draw_3d_limbs(
        ax2,
        estimated_skeleton.joints_3d_camera,
        estimated_skeleton.perception_confidence,
        f"2. Estimated 3D Skeleton\n(Reconstructed from RGB-D)",
        is_normalized=False,
        view_mode="front",
    )

    # Panel 3: GT 3D Skeleton in Camera Frame (Front View)
    ax3 = fig.add_subplot(1, 5, 3, projection="3d")
    gt_cam_dict = transform_gt_to_camera_frame(gt_joints_dict, camera_matrix_4x4)
    # 绘制 GT 关键点
    for name, p in gt_cam_dict.items():
        ax3.scatter(p[0], p[2], p[1], color="#E74C3C", s=25)
    # 绘制 GT 主要连线
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
            p1 = gt_cam_dict[j1_n]
            p2 = gt_cam_dict[j2_n]
            ax3.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], color="#E74C3C", linewidth=2.2)

    gt_pts = np.array(list(gt_cam_dict.values()))
    gt_span = max(np.ptp(gt_pts[:, 0]), np.ptp(gt_pts[:, 1]), np.ptp(gt_pts[:, 2]), 0.8)
    gt_mid = np.mean(gt_pts, axis=0)
    ax3.set_xlim(gt_mid[0] - gt_span/2, gt_mid[0] + gt_span/2)
    ax3.set_ylim(gt_mid[2] - gt_span/2, gt_mid[2] + gt_span/2)
    ax3.set_zlim(gt_mid[1] - gt_span/2, gt_mid[1] + gt_span/2)
    ax3.set_title("3. Habitat Ground Truth 3D\n(Simulation Upper Bound)", fontsize=10, fontweight="bold")
    ax3.set_xlabel("X (m)", fontsize=7)
    ax3.set_ylabel("Z (m)", fontsize=7)
    ax3.set_zlabel("Y (m)", fontsize=7)
    ax3.view_init(elev=0, azim=-90)

    # Panel 4: 空间重叠与误差向量 (Spatial Overlay with Error Vectors)
    ax4 = fig.add_subplot(1, 5, 4, projection="3d")
    ax4.scatter(P_est_rel[:, 0], P_est_rel[:, 2], P_est_rel[:, 1], color="#3498DB", s=30, label="Estimated")
    ax4.scatter(P_gt_rel[:, 0], P_gt_rel[:, 2], P_gt_rel[:, 1], color="#E74C3C", s=30, label="Ground Truth")

    # 绘制误差虚线向量
    for i in range(len(joint_names)):
        ax4.plot(
            [P_est_rel[i, 0], P_gt_rel[i, 0]],
            [P_est_rel[i, 2], P_gt_rel[i, 2]],
            [P_est_rel[i, 1], P_gt_rel[i, 1]],
            color="black", linestyle="--", linewidth=1.2, alpha=0.7,
        )

    ax4.set_xlim(mid_x - max_span/2, mid_x + max_span/2)
    ax4.set_ylim(mid_z - max_span/2, mid_z + max_span/2)
    ax4.set_zlim(mid_y - max_span/2, mid_y + max_span/2)
    ax4.set_title(f"4. 3D Overlay & Error Vectors\n(MPJPE={metrics.mpjpe_mm:.1f}mm, PA={metrics.pa_mpjpe_mm:.1f}mm)", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Norm X (m)", fontsize=7)
    ax4.set_ylabel("Norm Z (m)", fontsize=7)
    ax4.set_zlabel("Norm Y (m)", fontsize=7)
    ax4.legend(loc="upper right", fontsize=7.5)
    ax4.view_init(elev=15, azim=-60)

    # Panel 5: 逐关节误差分布条形图
    ax5 = fig.add_subplot(1, 5, 5)
    names = list(metrics.per_joint_errors_m.keys())
    err_vals_cm = [metrics.per_joint_errors_m[k] * 100.0 for k in names]
    bar_colors = ["#2ECC71" if err < 10.0 else ("#F39C12" if err < 20.0 else "#E74C3C") for err in err_vals_cm]

    ax5.barh(range(len(names)), err_vals_cm, color=bar_colors, height=0.65)
    ax5.axvline(10.0, color="gray", linestyle="--", label="10cm Thresh")
    ax5.set_yticks(range(len(names)))
    ax5.set_yticklabels([n.replace("_", " ")[:12] for n in names], fontsize=7)
    ax5.set_xlabel("Error (cm)", fontsize=8)
    ax5.set_title(f"5. Per-Joint Error (cm)\nPCK@10cm: {metrics.pck_10cm*100:.1f}%", fontsize=10, fontweight="bold")
    ax5.grid(axis="x", alpha=0.3)
    ax5.legend(loc="lower right", fontsize=7.5)

    plt.suptitle(
        f"ACTIVEVIEW v10.0 Phase 2.1 GT Verification | Sample: {sample_id} | MPJPE: {metrics.mpjpe_mm:.1f}mm | PA-MPJPE: {metrics.pa_mpjpe_mm:.1f}mm",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compare_sample(sample_id: str, output_path: Optional[Path] = None) -> Path:
    """对单个样本执行 GT 对比并输出诊断图。"""
    dataset_root = get_v10_dataset_root()
    skel_def = get_skeleton_definition()
    pipeline = V10PerceptionPipeline(dataset_root=dataset_root)
    evaluator = SkeletonEvaluator(skel_def=skel_def)

    with open(dataset_root / "metadata" / "samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    sample_dict = next((s for s in samples if s["sample_id"] == sample_id), None)
    if not sample_dict:
        raise ValueError(f"Sample {sample_id} not found in manifest!")

    sample_obj = V10Sample.from_dict(sample_dict)
    rgb_img = np.array(Image.open(dataset_root / sample_obj.rgb_path))
    depth_map = np.load(dataset_root / sample_obj.depth_path)

    skel, _ = pipeline.process_sample(sample_obj, rgb_image=rgb_img, depth_map=depth_map, save_outputs=False)

    gt_file = dataset_root / "ground_truth" / "skeleton" / f"{sample_id}.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)["joints_3d"]

    c2w = np.array(sample_obj.camera_pose.matrix_4x4, dtype=np.float32)
    metrics = evaluator.evaluate_sample(
        estimated_skeleton=skel,
        gt_joints_dict=gt_data,
        camera_matrix_4x4=c2w,
        sample_id=sample_id,
        action_label=sample_obj.action_label,
    )

    out_p = output_path or (dataset_root / "perception" / "visualization" / f"{sample_id}_gt_comparison.png")
    plot_skeleton_comparison(
        sample_id=sample_id,
        rgb_img=rgb_img,
        estimated_skeleton=skel,
        gt_joints_dict=gt_data,
        camera_matrix_4x4=c2w,
        metrics=metrics,
        output_path=out_p,
        skel_def=skel_def,
    )
    logger.info("Saved comparison visualization to: %s", out_p)
    return out_p


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v10.0 Skeleton GT Comparison Visualizer")
    parser.add_argument("--sample_id", type=str, default="v10_sample_000000", help="Sample ID")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    compare_sample(args.sample_id, out_p)


if __name__ == "__main__":
    main()
