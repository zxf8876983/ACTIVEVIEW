#!/usr/bin/env python3
"""
ACTIVEVIEW v10.0 单样本感知与骨架自动可视化检查工具 —— check_sample.py
===================================================================

用法：
    # 指定 sample_id 检查
    python tools/v10/check_sample.py --sample_id v10_sample_000000

    # 随机挑选一个样本检查
    python tools/v10/check_sample.py --random

    # 指定动作类别随机挑选
    python tools/v10/check_sample.py --action walking

    # 批量检查指定数量样本
    python tools/v10/check_sample.py --batch 5

输出：
    生成多模态诊断图 (RGB + 2D 骨架、Camera 3D 骨架、Normalized 3D 骨架、置信度与坐标校验卡)。
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将仓库根目录加入 sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.paths import get_data_root, get_v10_dataset_root
from ea_avs_mvp_v10.core.types import V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.perception.coordinate_validator import CoordinateValidator
from ea_avs_mvp_v10.perception.rgbd_skeleton_extractor import RGBDSkeletonExtractor
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer
from ea_avs_mvp_v10.visualization.skeleton_visualizer import SkeletonVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_sample")


def plot_single_sample_inspection(
    sample_id: str,
    rgb_img: np.ndarray,
    depth_map: np.ndarray,
    skeleton: EstimatedSkeleton3D,
    sample_meta: Dict[str, Any],
    val_status: str,
    val_reasons: List[str],
    output_path: Path,
    visualizer: SkeletonVisualizer,
) -> Path:
    """绘制 4 面板标准单样本检查图。"""
    fig = plt.figure(figsize=(19, 5.0))
    pairs = visualizer.get_skeleton_pairs()
    confs = skeleton.perception_confidence

    # 1. Panel 1: RGB + 2D Pose Overlay
    ax1 = fig.add_subplot(1, 4, 1)
    overlay = visualizer.draw_2d_skeleton_on_rgb(rgb_img, skeleton)
    ax1.imshow(overlay)
    ax1.set_title(f"1. RGB + 2D Skeleton ({skeleton.joint_format})\nAction: {sample_meta.get('action_label', '').upper()}", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # 2. Panel 2: Camera Coordinate 3D Skeleton (Front View)
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    j_cam = skeleton.joints_3d_camera
    for j1, j2 in pairs:
        if j1 < len(confs) and j2 < len(confs):
            if confs[j1] >= 0.35 and confs[j2] >= 0.35:
                ax2.plot(
                    [j_cam[j1, 0], j_cam[j2, 0]],
                    [j_cam[j1, 2], j_cam[j2, 2]],
                    [j_cam[j1, 1], j_cam[j2, 1]],
                    color="deepskyblue", linewidth=2.2,
                )
    valid_idx = np.where(confs >= 0.35)[0]
    ax2.scatter(j_cam[valid_idx, 0], j_cam[valid_idx, 2], j_cam[valid_idx, 1], color="blue", s=30)
    depth_mean = np.mean(j_cam[valid_idx, 2]) if len(valid_idx) > 0 else 0.0
    ax2.set_title(f"2. Camera 3D Skeleton (Front View)\n(Depth Z={depth_mean:.2f}m)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (m)", fontsize=8)
    ax2.set_ylabel("Z (m)", fontsize=8)
    ax2.set_zlabel("Y (m)", fontsize=8)
    ax2.view_init(elev=0, azim=-90)

    # 3. Panel 3: Normalized 3D Skeleton (ST-GCN Input)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    j_norm = skeleton.joints_3d_normalized if skeleton.joints_3d_normalized is not None else j_cam
    for j1, j2 in pairs:
        if j1 < len(confs) and j2 < len(confs):
            if confs[j1] >= 0.35 and confs[j2] >= 0.35:
                ax3.plot(
                    [j_norm[j1, 0], j_norm[j2, 0]],
                    [j_norm[j1, 2], j_norm[j2, 2]],
                    [j_norm[j1, 1], j_norm[j2, 1]],
                    color="#8E44AD", linewidth=2.2,
                )
    ax3.scatter(j_norm[valid_idx, 0], j_norm[valid_idx, 2], j_norm[valid_idx, 1], color="#E67E22", s=30)
    ax3.scatter(0, 0, 0, color="red", marker="^", s=80, label="Root (0,0,0)")
    ax3.set_title("3. Normalized 3D Skeleton\n(Root-Centered & Scale-Normalized)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Norm X", fontsize=8)
    ax3.set_ylabel("Norm Z", fontsize=8)
    ax3.set_zlabel("Norm Y", fontsize=8)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.view_init(elev=15, azim=-60)

    # 4. Panel 4: Perception Confidence & Sanity Info Card
    ax4 = fig.add_subplot(1, 4, 4)
    names = [name[:6] for name in skeleton.joint_names]
    colors = ["#2ECC71" if c >= 0.35 else "#E74C3C" for c in confs]
    ax4.barh(range(len(names)), confs, color=colors, height=0.65)
    ax4.axvline(0.35, color="red", linestyle="--", linewidth=1.2, label="Uncertainty (0.35)")
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names, fontsize=6.5)
    ax4.set_xlim(0.0, 1.05)
    ax4.set_xlabel("Confidence", fontsize=9)
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(axis="x", alpha=0.3)

    status_color = "green" if val_status == "VALID" else ("orange" if val_status == "WARNING" else "red")
    info_text = f"Status: {val_status} | Mean Conf: {np.mean(confs):.2f}\nValid Joints: {len(valid_idx)}/{len(confs)}"
    if val_reasons:
        info_text += f"\nNote: {', '.join(val_reasons[:2])}"
    ax4.set_title(f"4. Confidence & Sanity\n[{val_status}]", fontsize=11, fontweight="bold", color=status_color)

    plt.suptitle(
        f"ACTIVEVIEW v10.0 Perception Sample Inspection | Sample: {sample_id} | Action: {sample_meta.get('action_label', '').upper()} | View: {sample_meta.get('view_id', '')}",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def check_sample(
    sample_id: Optional[str] = None,
    action_filter: Optional[str] = None,
    pick_random: bool = False,
    batch_count: int = 1,
    custom_output: Optional[Path] = None,
) -> List[Path]:
    """检查一个或多个样本并生成多模态诊断图。"""
    dataset_root = get_v10_dataset_root()
    manifest_p = dataset_root / "metadata" / "samples.json"

    if not manifest_p.exists():
        logger.error("Phase 1 samples manifest not found at: %s", manifest_p)
        sys.exit(1)

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    samples = manifest_data.get("samples", [])

    if len(samples) == 0:
        logger.error("No samples found in manifest!")
        sys.exit(1)

    # 筛选候选样本
    candidate_samples = samples
    if action_filter:
        candidate_samples = [s for s in samples if s.get("action_label") == action_filter.lower()]
        if not candidate_samples:
            logger.error("No samples found matching action: %s", action_filter)
            sys.exit(1)

    selected_dicts = []
    if sample_id:
        matches = [s for s in samples if s.get("sample_id") == sample_id]
        if not matches:
            logger.error("Sample ID '%s' not found!", sample_id)
            sys.exit(1)
        selected_dicts = matches
    elif pick_random or action_filter:
        k = min(batch_count, len(candidate_samples))
        selected_dicts = random.sample(candidate_samples, k)
    else:
        k = min(batch_count, len(candidate_samples))
        selected_dicts = candidate_samples[:k]

    # 初始化流水线与渲染器
    pipeline = V10PerceptionPipeline(dataset_root=dataset_root)
    visualizer = SkeletonVisualizer(output_dpi=150)
    validator = CoordinateValidator()

    generated_paths = []
    logger.info("Inspecting %d sample(s)...", len(selected_dicts))

    for s_dict in selected_dicts:
        sid = s_dict["sample_id"]
        sample_obj = V10Sample.from_dict(s_dict)

        # 读取 RGB 与 Depth
        rgb_p = dataset_root / sample_obj.rgb_path
        depth_p = dataset_root / sample_obj.depth_path
        rgb_img = np.array(Image.open(rgb_p))
        depth_map = np.load(depth_p)

        # 执行感知估计
        skel, record = pipeline.process_sample(sample_obj, rgb_image=rgb_img, depth_map=depth_map, save_outputs=False)
        val_res = validator.validate(skel)

        # 确定输出路径
        if custom_output:
            out_p = custom_output if len(selected_dicts) == 1 else custom_output.parent / f"{sid}_check.png"
        else:
            out_p = dataset_root / "perception" / "visualization" / f"{sid}_check.png"

        saved_file = plot_single_sample_inspection(
            sample_id=sid,
            rgb_img=rgb_img,
            depth_map=depth_map,
            skeleton=skel,
            sample_meta=s_dict,
            val_status=val_res.status,
            val_reasons=val_res.reasons,
            output_path=out_p,
            visualizer=visualizer,
        )

        logger.info("[✓] Sample %s | Action: %s | Status: %s | Saved -> %s",
                    sid, s_dict.get("action_label"), val_res.status, saved_file)
        generated_paths.append(saved_file)

    return generated_paths


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v10.0 Perception Sample Inspection Tool")
    parser.add_argument("--sample_id", type=str, default=None, help="Target sample ID to inspect (e.g., v10_sample_000000)")
    parser.add_argument("--action", type=str, default=None, help="Filter by action category (e.g., walking, sitting)")
    parser.add_argument("--random", action="store_true", help="Randomly select a sample to inspect")
    parser.add_argument("--batch", type=int, default=1, help="Number of samples to inspect in batch mode")
    parser.add_argument("--output", type=str, default=None, help="Custom output image path")

    args = parser.parse_args()

    custom_out = Path(args.output) if args.output else None
    check_sample(
        sample_id=args.sample_id,
        action_filter=args.action,
        pick_random=args.random,
        batch_count=args.batch,
        custom_output=custom_out,
    )


if __name__ == "__main__":
    main()
