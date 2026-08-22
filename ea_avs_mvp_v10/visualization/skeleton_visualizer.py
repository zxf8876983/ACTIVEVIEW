"""
RGB-D 3D 骨架与多模态感知可视化渲染器 —— skeleton_visualizer.py
============================================================

职责：
    1. 严格基于 Extractor 官方关节拓扑 (MediaPipe 33 / COCO 17) 渲染骨骼连线与关节点；
    2. 禁止硬编码错误骨骼连线；
    3. 绘制 RGB + 2D 投影、Camera 坐标系 3D 骨架与 Normalized 3D 骨架；
    4. 绘制感知置信度直方图与健康度校验卡。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image, ImageDraw

from ea_avs_mvp_v10.perception.pose_estimator import (
    COCO_KEYPOINTS,
    COCO_SKELETON_PAIRS,
)
from ea_avs_mvp_v10.perception.rgbd_skeleton_extractor import (
    MEDIAPIPE_33_KEYPOINTS,
    MEDIAPIPE_33_SKELETON_PAIRS,
)
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D

logger = logging.getLogger(__name__)


class SkeletonVisualizer:
    """标准 3D 骨架与多模态感知可视化渲染器。"""

    def __init__(self, output_dpi: int = 150):
        self.output_dpi = output_dpi

    def get_skeleton_pairs(self, joint_format: str) -> List[Tuple[int, int]]:
        """获取对应拓扑的官方骨骼连线定义。"""
        if joint_format == "MediaPipe33":
            return MEDIAPIPE_33_SKELETON_PAIRS
        elif joint_format == "COCO17":
            return COCO_SKELETON_PAIRS
        else:
            return []

    def draw_2d_skeleton_on_rgb(
        self,
        rgb_image: Union[np.ndarray, Image.Image],
        skeleton: EstimatedSkeleton3D,
        conf_thresh: float = 0.35,
    ) -> np.ndarray:
        """在 RGB 图像上叠加绘制官方拓扑 2D 骨架。"""
        if isinstance(rgb_image, np.ndarray):
            pil_img = Image.fromarray(rgb_image.astype(np.uint8)).copy()
        else:
            pil_img = rgb_image.copy()

        draw = ImageDraw.Draw(pil_img)
        kpts_2d = skeleton.joints_2d
        confs = skeleton.perception_confidence
        pairs = self.get_skeleton_pairs(skeleton.joint_format)

        # 绘制官方骨骼连线
        for j1, j2 in pairs:
            if j1 < len(confs) and j2 < len(confs):
                if confs[j1] >= conf_thresh and confs[j2] >= conf_thresh:
                    p1 = (float(kpts_2d[j1, 0]), float(kpts_2d[j1, 1]))
                    p2 = (float(kpts_2d[j2, 0]), float(kpts_2d[j2, 1]))
                    avg_c = (confs[j1] + confs[j2]) / 2.0
                    line_color = (0, 255, 128) if avg_c > 0.6 else (255, 200, 0)
                    draw.line([p1, p2], fill=line_color, width=3)

        # 绘制关键点圆圈
        r = 3 if skeleton.joint_format == "MediaPipe33" else 4
        for i in range(len(confs)):
            u, v = float(kpts_2d[i, 0]), float(kpts_2d[i, 1])
            c = float(confs[i])
            if c >= conf_thresh:
                node_color = (0, 230, 255) if c > 0.6 else (255, 150, 0)
            else:
                node_color = (255, 50, 50)  # 红色表示低置信 / 不确定

            draw.ellipse([u - r, v - r, u + r, v + r], fill=node_color, outline=(0, 0, 0))

        return np.array(pil_img)

    def plot_sample_multimodal(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        skeleton: EstimatedSkeleton3D,
        sample_meta: Optional[Dict[str, Any]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        生成 5 面板多模态感知全景诊断图：
        [1. RGB + 2D 骨架] [2. 深度图 + 投影] [3. 3D 骨架 (相机系)] [4. Normalized 3D 骨架] [5. 逐关节置信度]
        """
        fig = plt.figure(figsize=(22, 4.5))
        pairs = self.get_skeleton_pairs(skeleton.joint_format)
        confs = skeleton.perception_confidence

        # Panel 1: RGB + 2D 骨架
        ax1 = fig.add_subplot(1, 5, 1)
        rgb_overlay = self.draw_2d_skeleton_on_rgb(rgb_image, skeleton)
        ax1.imshow(rgb_overlay)
        ax1.set_title(f"1. RGB + 2D Skeleton ({skeleton.joint_format})", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Depth Map + 关键点投影
        ax2 = fig.add_subplot(1, 5, 2)
        d_plot = ax2.imshow(depth_map, cmap="plasma", vmin=0.5, vmax=5.0)
        kpts_2d = skeleton.joints_2d
        valid_idx = np.where(confs >= 0.35)[0]
        ax2.scatter(kpts_2d[valid_idx, 0], kpts_2d[valid_idx, 1], c="cyan", s=20, edgecolors="white")
        ax2.set_title(f"2. Depth Map + Projected ({len(valid_idx)}/{len(confs)})", fontsize=10, fontweight="bold")
        ax2.axis("off")
        fig.colorbar(d_plot, ax=ax2, fraction=0.046, pad=0.04)

        # Panel 3: 3D Camera Coordinate Skeleton
        ax3 = fig.add_subplot(1, 5, 3, projection="3d")
        j3d = skeleton.joints_3d_camera
        for j1, j2 in pairs:
            if j1 < len(confs) and j2 < len(confs):
                if confs[j1] >= 0.35 and confs[j2] >= 0.35:
                    ax3.plot(
                        [j3d[j1, 0], j3d[j2, 0]],
                        [j3d[j1, 2], j3d[j2, 2]],
                        [j3d[j1, 1], j3d[j2, 1]],
                        color="deepskyblue", linewidth=2.2,
                    )
        ax3.scatter(j3d[valid_idx, 0], j3d[valid_idx, 2], j3d[valid_idx, 1], color="blue", s=25)
        ax3.set_title("3. Extracted 3D Pose (Cam Frame)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("X (m)", fontsize=7)
        ax3.set_ylabel("Z/Depth (m)", fontsize=7)
        ax3.set_zlabel("Y/Up (m)", fontsize=7)
        ax3.view_init(elev=15, azim=-60)

        # Panel 4: Normalized 3D Skeleton (Root centered at origin & scale normalized)
        ax4 = fig.add_subplot(1, 5, 4, projection="3d")
        norm_j3d = skeleton.joints_3d_normalized if skeleton.joints_3d_normalized is not None else skeleton.joints_3d_camera
        for j1, j2 in pairs:
            if j1 < len(confs) and j2 < len(confs):
                if confs[j1] >= 0.35 and confs[j2] >= 0.35:
                    ax4.plot(
                        [norm_j3d[j1, 0], norm_j3d[j2, 0]],
                        [norm_j3d[j1, 2], norm_j3d[j2, 2]],
                        [norm_j3d[j1, 1], norm_j3d[j2, 1]],
                        color="#9B59B6", linewidth=2.2,
                    )
        ax4.scatter(norm_j3d[valid_idx, 0], norm_j3d[valid_idx, 2], norm_j3d[valid_idx, 1], color="#E67E22", s=25)
        ax4.scatter(0, 0, 0, color="red", marker="^", s=60, label="Root (Origin)")
        ax4.set_title("4. Normalized 3D Pose (ST-GCN Ready)", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Norm X", fontsize=7)
        ax4.set_ylabel("Norm Z", fontsize=7)
        ax4.set_zlabel("Norm Y", fontsize=7)
        ax4.legend(loc="upper right", fontsize=7)
        ax4.view_init(elev=15, azim=-60)

        # Panel 5: 逐关节置信度直方图
        ax5 = fig.add_subplot(1, 5, 5)
        names = [name[:6] for name in skeleton.joint_names]
        colors = ["#2ECC71" if c >= 0.35 else "#E74C3C" for c in confs]
        ax5.barh(range(len(names)), confs, color=colors, height=0.65)
        ax5.axvline(0.35, color="red", linestyle="--", linewidth=1.2, label="Uncertainty (0.35)")
        ax5.set_yticks(range(len(names)))
        ax5.set_yticklabels(names, fontsize=6.5)
        ax5.set_xlim(0.0, 1.05)
        ax5.set_xlabel("Perception Confidence", fontsize=8)
        ax5.set_title("5. Perception Confidence", fontsize=10, fontweight="bold")
        ax5.legend(loc="lower right", fontsize=7.5)
        ax5.grid(axis="x", alpha=0.3)

        title_str = f"ACTIVEVIEW v10.0 Phase 2: RGB-D Skeleton Extractor ({skeleton.joint_format})"
        if sample_meta:
            title_str += f" | Action: {sample_meta.get('action_label', '').upper()} | Sample: {sample_meta.get('sample_id', '')}"
        plt.suptitle(title_str, fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=self.output_dpi, bbox_inches="tight")
            logger.info("Saved perception multimodal visualization to: %s", p)

        return fig
