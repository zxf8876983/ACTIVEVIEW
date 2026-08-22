#!/usr/bin/env python3
"""
ACTIVEVIEW v10.0 骨架关节 ID 与 2D/3D 一致性诊断可视化器 —— skeleton_debug_visualizer.py
=============================================================================

职责：
    1. 在 RGB 图像与 2D 骨架上打印每个关键点的名称与 ID (如 nose(0), left_shoulder(11))；
    2. 在 3D Camera 系骨架与 Normalized 骨架上逐点标注 3D 关节名称与 ID；
    3. 支持正交正视图 (Front View) 与 3D 空间透视视角 (Perspective View)；
    4. 供研究员肉眼逐点核查 2D 与 3D 的 100% 几何与拓扑一致性；
    5. 严格读取 `configs/skeleton_definition.json`，无任何硬编码索引。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ea_avs_mvp_v10.core.paths import get_v10_dataset_root
from ea_avs_mvp_v10.core.types import V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.perception.coordinate_validator import CoordinateValidator
from ea_avs_mvp_v10.perception.rgbd_skeleton_extractor import RGBDSkeletonExtractor
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v10.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("skeleton_debug_visualizer")


def draw_labeled_2d_skeleton(
    rgb_image: np.ndarray,
    skeleton: EstimatedSkeleton3D,
    skel_def: SkeletonDefinition,
    conf_thresh: float = 0.35,
) -> np.ndarray:
    """在 RGB 图像上绘制 2D 骨骼并附带每个关节的名称与 ID 标注。"""
    pil_img = Image.fromarray(rgb_image.astype(np.uint8)).copy()
    draw = ImageDraw.Draw(pil_img)
    kpts_2d = skeleton.joints_2d
    confs = skeleton.perception_confidence

    # 1. 绘制骨骼连线
    for j1, j2 in skel_def.edges:
        if confs[j1] >= conf_thresh and confs[j2] >= conf_thresh:
            p1 = (float(kpts_2d[j1, 0]), float(kpts_2d[j1, 1]))
            p2 = (float(kpts_2d[j2, 0]), float(kpts_2d[j2, 1]))
            draw.line([p1, p2], fill=(0, 255, 128), width=2)

    # 2. 绘制关节点与文字 ID
    for i, jdef in enumerate(skel_def.joints):
        u, v = float(kpts_2d[i, 0]), float(kpts_2d[i, 1])
        c = float(confs[i])
        color = (0, 230, 255) if c >= conf_thresh else (255, 60, 60)
        draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=color, outline=(0, 0, 0))

        # 标注文字 (主要身体关节标注完整名称，四肢末端标注 ID)
        if c >= conf_thresh:
            label = f"{jdef.name[:6]}({i})"
            draw.text((u + 4, v - 5), label, fill=(255, 255, 0))

    return np.array(pil_img)


def plot_debug_inspection(
    sample_id: str,
    rgb_img: np.ndarray,
    depth_map: np.ndarray,
    skeleton: EstimatedSkeleton3D,
    sample_meta: Dict[str, Any],
    output_path: Path,
    skel_def: SkeletonDefinition,
) -> Path:
    """生成 4 面板高分辨率关节 ID 对齐诊断图。"""
    fig = plt.figure(figsize=(24, 6.0))
    confs = skeleton.perception_confidence
    valid_idx = np.where(confs >= 0.35)[0]

    # Panel 1: RGB + 2D Keypoints with Joint IDs
    ax1 = fig.add_subplot(1, 4, 1)
    overlay = draw_labeled_2d_skeleton(rgb_img, skeleton, skel_def)
    ax1.imshow(overlay)
    ax1.set_title(f"1. RGB + 2D Joints with IDs\nAction: {sample_meta.get('action_label', '').upper()}", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Camera 3D Skeleton (Front View, matching camera projection)
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    j_cam = skeleton.joints_3d_camera
    for j1, j2 in skel_def.edges:
        if confs[j1] >= 0.35 and confs[j2] >= 0.35:
            ax2.plot(
                [j_cam[j1, 0], j_cam[j2, 0]],
                [j_cam[j1, 2], j_cam[j2, 2]],
                [j_cam[j1, 1], j_cam[j2, 1]],
                color="deepskyblue", linewidth=2.0,
            )
    ax2.scatter(j_cam[valid_idx, 0], j_cam[valid_idx, 2], j_cam[valid_idx, 1], color="blue", s=25)
    for idx in valid_idx:
        ax2.text(j_cam[idx, 0] + 0.02, j_cam[idx, 2], j_cam[idx, 1], f"{skel_def.joints[idx].name[:5]}({idx})", fontsize=6.5, color="darkblue")

    mean_d = float(np.mean(j_cam[valid_idx, 2])) if len(valid_idx) > 0 else 0.0
    ax2.set_title(f"2. Camera 3D Pose (Front View)\n(Z_depth={mean_d:.2f}m)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (m)", fontsize=8)
    ax2.set_ylabel("Z (m)", fontsize=8)
    ax2.set_zlabel("Y (m)", fontsize=8)
    ax2.view_init(elev=0, azim=-90)  # 正视图对齐相机光轴

    # Panel 3: Camera 3D Skeleton (Perspective 3D Orbit View)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    for j1, j2 in skel_def.edges:
        if confs[j1] >= 0.35 and confs[j2] >= 0.35:
            ax3.plot(
                [j_cam[j1, 0], j_cam[j2, 0]],
                [j_cam[j1, 2], j_cam[j2, 2]],
                [j_cam[j1, 1], j_cam[j2, 1]],
                color="deepskyblue", linewidth=2.0,
            )
    ax3.scatter(j_cam[valid_idx, 0], j_cam[valid_idx, 2], j_cam[valid_idx, 1], color="blue", s=25)
    for idx in valid_idx:
        ax3.text(j_cam[idx, 0] + 0.02, j_cam[idx, 2], j_cam[idx, 1], f"{skel_def.joints[idx].name[:5]}({idx})", fontsize=6.5, color="darkblue")

    ax3.set_title("3. Camera 3D Pose (3D Orbit View)\n(+X:Right, +Y:Up, +Z:Forward)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (m)", fontsize=8)
    ax3.set_ylabel("Z (m)", fontsize=8)
    ax3.set_zlabel("Y (m)", fontsize=8)
    ax3.view_init(elev=15, azim=-60)

    # Panel 4: Normalized 3D Skeleton with Root Alignment
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    j_norm = skeleton.joints_3d_normalized if skeleton.joints_3d_normalized is not None else j_cam
    for j1, j2 in skel_def.edges:
        if confs[j1] >= 0.35 and confs[j2] >= 0.35:
            ax4.plot(
                [j_norm[j1, 0], j_norm[j2, 0]],
                [j_norm[j1, 2], j_norm[j2, 2]],
                [j_norm[j1, 1], j_norm[j2, 1]],
                color="#8E44AD", linewidth=2.0,
            )
    ax4.scatter(j_norm[valid_idx, 0], j_norm[valid_idx, 2], j_norm[valid_idx, 1], color="#E67E22", s=25)
    ax4.scatter(0, 0, 0, color="red", marker="^", s=80, label="Root / Hip Center (0,0,0)")
    for idx in valid_idx:
        ax4.text(j_norm[idx, 0] + 0.05, j_norm[idx, 2], j_norm[idx, 1], f"{idx}", fontsize=7, color="purple")

    ax4.set_title("4. Normalized 3D Pose\n(ST-GCN Input Ready)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Norm X", fontsize=8)
    ax4.set_ylabel("Norm Z", fontsize=8)
    ax4.set_zlabel("Norm Y", fontsize=8)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.view_init(elev=15, azim=-60)

    plt.suptitle(
        f"ACTIVEVIEW v10.0 Skeleton Debug Audit | Sample: {sample_id} | Action: {sample_meta.get('action_label', '').upper()} | Backend: {skel_def.backend}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def debug_sample(sample_id: str, output_dir: Optional[Path] = None) -> Path:
    """对单个样本执行详细 2D/3D ID 对齐诊断并保存。"""
    dataset_root = get_v10_dataset_root()
    manifest_p = dataset_root / "metadata" / "samples.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    target_sample = None
    for s_dict in manifest_data.get("samples", []):
        if s_dict["sample_id"] == sample_id:
            target_sample = s_dict
            break

    if not target_sample:
        raise ValueError(f"Sample ID {sample_id} not found in manifest!")

    sample_obj = V10Sample.from_dict(target_sample)
    skel_def = get_skeleton_definition()
    pipeline = V10PerceptionPipeline(dataset_root=dataset_root)

    rgb_p = dataset_root / sample_obj.rgb_path
    depth_p = dataset_root / sample_obj.depth_path
    rgb_img = np.array(Image.open(rgb_p))
    depth_map = np.load(depth_p)

    skel, _ = pipeline.process_sample(sample_obj, rgb_image=rgb_img, depth_map=depth_map, save_outputs=False)

    out_d = output_dir or (dataset_root / "perception" / "visualization")
    out_file = out_d / f"{sample_id}_debug.png"

    plot_debug_inspection(
        sample_id=sample_id,
        rgb_img=rgb_img,
        depth_map=depth_map,
        skeleton=skel,
        sample_meta=target_sample,
        output_path=out_file,
        skel_def=skel_def,
    )
    logger.info("Saved debug inspection to: %s", out_file)
    return out_file


def main():
    parser = argparse.ArgumentParser(description="ACTIVEVIEW v10.0 Skeleton Debug Visualizer")
    parser.add_argument("--sample_id", type=str, default="v10_sample_000000", help="Target sample ID")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_d = Path(args.output_dir) if args.output_dir else None
    debug_sample(args.sample_id, out_d)


if __name__ == "__main__":
    main()
