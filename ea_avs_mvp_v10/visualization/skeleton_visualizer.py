"""
3D 骨架与感知流水线多模态可视化 —— skeleton_visualizer.py
======================================================

职责：
    1. 绘制 RGB 图像叠加 2D 估计骨架连线与关键点 (带置信度色阶)；
    2. 绘制 Depth 深度图与 2D 关键点空间投影点；
    3. 绘制 3D 空间骨架连通图 (3D Matplotlib 散点与运动学拓扑折线)；
    4. 绘制 16 关节融合置信度直方图与遮挡阈值警戒线；
    5. 支持输出单样本诊断图与多视角/多动作对比大图。
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

from ea_avs_mvp_v10.perception.pose_estimator import COCO_SKELETON_PAIRS, Pose2DResult
from ea_avs_mvp_v10.perception.skeleton_converter import (
    UNIFIED_JOINT_NAMES,
    UNIFIED_SKELETON_PAIRS,
    EstimatedSkeleton3D,
)

logger = logging.getLogger(__name__)


class SkeletonVisualizer:
    """感知流水线 2D/3D 骨架多模态可视化渲染器。"""

    def __init__(self, output_dpi: int = 150):
        self.output_dpi = output_dpi

    def draw_2d_skeleton_on_rgb(
        self,
        rgb_image: Union[np.ndarray, Image.Image],
        skeleton: EstimatedSkeleton3D,
        conf_thresh: float = 0.35,
    ) -> np.ndarray:
        """在 RGB 图像上叠加绘制 2D 估计骨架与关节节点。"""
        if isinstance(rgb_image, np.ndarray):
            pil_img = Image.fromarray(rgb_image.astype(np.uint8)).copy()
        else:
            pil_img = rgb_image.copy()

        draw = ImageDraw.Draw(pil_img)
        kpts_2d = skeleton.joints_2d
        confs = skeleton.confidence

        # 绘制骨骼连线
        for j1, j2 in UNIFIED_SKELETON_PAIRS:
            if confs[j1] >= conf_thresh and confs[j2] >= conf_thresh:
                p1 = (float(kpts_2d[j1, 0]), float(kpts_2d[j1, 1]))
                p2 = (float(kpts_2d[j2, 0]), float(kpts_2d[j2, 1]))
                # 根据两端平均置信度着色 (高置信绿色，中置信黄色)
                avg_c = (confs[j1] + confs[j2]) / 2.0
                line_color = (0, 255, 128) if avg_c > 0.6 else (255, 200, 0)
                draw.line([p1, p2], fill=line_color, width=3)

        # 绘制关键点圆圈
        r = 4
        for i in range(len(UNIFIED_JOINT_NAMES)):
            u, v = float(kpts_2d[i, 0]), float(kpts_2d[i, 1])
            c = float(confs[i])
            if c >= conf_thresh:
                node_color = (0, 230, 255) if c > 0.6 else (255, 150, 0)
            else:
                node_color = (255, 50, 50)  # 红色表示遮挡/低置信

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
        生成 4 面板多模态感知全景诊断图：
        [1. RGB + 2D 估计骨架] [2. 深度图 + 投影] [3. 3D 空间骨架几何] [4. 逐关节置信度分布]
        """
        fig = plt.figure(figsize=(18, 4.5))

        # Panel 1: RGB + 2D 骨架
        ax1 = fig.add_subplot(1, 4, 1)
        rgb_overlay = self.draw_2d_skeleton_on_rgb(rgb_image, skeleton)
        ax1.imshow(rgb_overlay)
        ax1.set_title("1. RGB + Estimated 2D Skeleton", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Depth Map + 关键点
        ax2 = fig.add_subplot(1, 4, 2)
        d_plot = ax2.imshow(depth_map, cmap="plasma", vmin=0.5, vmax=5.0)
        kpts_2d = skeleton.joints_2d
        confs = skeleton.confidence
        valid_idx = np.where(confs >= 0.35)[0]
        ax2.scatter(kpts_2d[valid_idx, 0], kpts_2d[valid_idx, 1], c="cyan", s=25, edgecolors="white")
        ax2.set_title("2. Depth Map + Projected Joints", fontsize=11, fontweight="bold")
        ax2.axis("off")
        fig.colorbar(d_plot, ax=ax2, fraction=0.046, pad=0.04)

        # Panel 3: 3D Spatial Skeleton
        ax3 = fig.add_subplot(1, 4, 3, projection="3d")
        j3d = skeleton.joints_3d_cam
        # 绘制 3D 骨骼连线
        for j1, j2 in UNIFIED_SKELETON_PAIRS:
            if confs[j1] >= 0.35 and confs[j2] >= 0.35:
                ax3.plot(
                    [j3d[j1, 0], j3d[j2, 0]],
                    [j3d[j1, 2], j3d[j2, 2]],  # 将深度 Z 映射到 Y 轴视觉便于观察
                    [-j3d[j1, 1], -j3d[j2, 1]], # 将 Y 反向使头部向上
                    color="lime" if (confs[j1] + confs[j2]) / 2.0 > 0.6 else "orange",
                    linewidth=2.5,
                )

        # 绘制 3D 关节点散点
        for i in range(len(UNIFIED_JOINT_NAMES)):
            if confs[i] >= 0.35:
                ax3.scatter(j3d[i, 0], j3d[i, 2], -j3d[i, 1], color="deepskyblue", s=35, edgecolors="black")
            else:
                ax3.scatter(j3d[i, 0], j3d[i, 2], -j3d[i, 1], color="red", s=25, alpha=0.5)

        ax3.set_title("3. Estimated 3D Skeleton (Cam Frame)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("X (m)", fontsize=8)
        ax3.set_ylabel("Z / Depth (m)", fontsize=8)
        ax3.set_zlabel("-Y / Height (m)", fontsize=8)
        ax3.view_init(elev=15, azim=-60)

        # Panel 4: 关节置信度直方图
        ax4 = fig.add_subplot(1, 4, 4)
        names = [name[:6] for name in UNIFIED_JOINT_NAMES]
        colors = ["#2ECC71" if c >= 0.35 else "#E74C3C" for c in confs]
        bars = ax4.barh(range(len(names)), confs, color=colors, height=0.7)
        ax4.axvline(0.35, color="red", linestyle="--", linewidth=1.5, label="Occlusion Thresh (0.35)")
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names, fontsize=8)
        ax4.set_xlim(0.0, 1.05)
        ax4.set_xlabel("Confidence", fontsize=9)
        ax4.set_title("4. Joint Confidence & Occlusion", fontsize=11, fontweight="bold")
        ax4.legend(loc="lower right", fontsize=8)
        ax4.grid(axis="x", alpha=0.3)

        title_str = "ACTIVEVIEW v10.0 Phase 2: RGB-D Perception Pipeline (Estimated 3D Skeleton)"
        if sample_meta:
            title_str += f" | Action: {sample_meta.get('action_label', '').upper()} | Sample: {sample_meta.get('sample_id', '')}"
        plt.suptitle(title_str, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=self.output_dpi, bbox_inches="tight")
            logger.info("Saved perception multimodal visualization to: %s", p)

        return fig
