"""
RGB-D 3D 骨架与多模态感知可视化渲染器 —— skeleton_visualizer.py
============================================================

职责：
    1. 严格从 `configs/skeleton_definition.json` 读取官方骨骼连线与关节定义；
    2. 采用真实物理长宽比 (True Metric Aspect Ratio)，消除 3D Matplotlib 坐标轴被压缩挤压变形；
    3. 肢体区分颜色渲染 (左肢: 蓝色, 右肢: 绿色, 躯干: 紫色, 头部/颈部: 橙色)；
    4. 自动连接颈部骨骼 (双肩中心至头部鼻尖)，确保头部与躯干自然衔接；
    5. 提供标准正投影 (Front View) 与 3D 空间透视投影 (Perspective View)。
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

from ea_avs_mvp_v11.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v11.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


class SkeletonVisualizer:
    """标准 3D 骨架与多模态感知可视化渲染器。"""

    def __init__(self, output_dpi: int = 150, skel_def: Optional[SkeletonDefinition] = None):
        self.output_dpi = output_dpi
        self.skel_def = skel_def or get_skeleton_definition()

    def get_skeleton_pairs(self) -> List[Tuple[int, int]]:
        """从中央骨架定义获取官方骨骼连线列表。"""
        return self.skel_def.edges

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
        pairs = self.get_skeleton_pairs()

        # 1. 颈部连接线 (双肩中心至鼻尖)
        if len(confs) > 12 and confs[0] >= conf_thresh and confs[11] >= conf_thresh and confs[12] >= conf_thresh:
            neck_2d = ((kpts_2d[11, 0] + kpts_2d[12, 0]) / 2.0, (kpts_2d[11, 1] + kpts_2d[12, 1]) / 2.0)
            draw.line([neck_2d, (kpts_2d[0, 0], kpts_2d[0, 1])], fill=(255, 140, 0), width=3)

        # 2. 绘制官方骨骼连线 (肢体区分色彩)
        for j1, j2 in pairs:
            if j1 < len(confs) and j2 < len(confs):
                if confs[j1] >= conf_thresh and confs[j2] >= conf_thresh:
                    p1 = (float(kpts_2d[j1, 0]), float(kpts_2d[j1, 1]))
                    p2 = (float(kpts_2d[j2, 0]), float(kpts_2d[j2, 1]))

                    part1, part2 = self.skel_def.joints[j1].part, self.skel_def.joints[j2].part
                    if "left" in part1 or "left" in part2:
                        line_color = (52, 152, 219)   # 左肢蓝色
                    elif "right" in part1 or "right" in part2:
                        line_color = (46, 204, 113)  # 右肢绿色
                    elif "head" in part1 or "head" in part2:
                        line_color = (230, 126, 34)  # 头部橙色
                    else:
                        line_color = (155, 89, 182)  # 躯干紫色

                    draw.line([p1, p2], fill=line_color, width=3)

        # 3. 绘制关键点圆圈
        r = 3
        for i in range(len(confs)):
            u, v = float(kpts_2d[i, 0]), float(kpts_2d[i, 1])
            c = float(confs[i])
            if c >= conf_thresh:
                node_color = (0, 230, 255) if c > 0.6 else (255, 200, 0)
            else:
                node_color = (255, 50, 50)  # 红色表示低置信 / 不确定

            draw.ellipse([u - r, v - r, u + r, v + r], fill=node_color, outline=(0, 0, 0))

        return np.array(pil_img)

    def draw_3d_limbs(
        self,
        ax: Axes3D,
        joints_3d: np.ndarray,
        confs: np.ndarray,
        title: str,
        is_normalized: bool = False,
        view_mode: str = "front",
    ):
        """以真实物理尺度与肢体着色绘制 3D 骨架。"""
        valid_idx = np.where(confs >= 0.35)[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(len(confs))

        # 设定等比例长宽比，消除 Matplotlib 3D 盒体挤压变形
        x_span = np.ptp(joints_3d[valid_idx, 0])
        y_span = np.ptp(joints_3d[valid_idx, 1])
        z_span = np.ptp(joints_3d[valid_idx, 2])
        max_span = max(x_span, y_span, z_span, 0.6 if is_normalized else 0.8)

        mid_x = float(np.mean(joints_3d[valid_idx, 0]))
        mid_y = float(np.mean(joints_3d[valid_idx, 1]))
        mid_z = float(np.mean(joints_3d[valid_idx, 2]))

        ax.set_xlim(mid_x - max_span / 2.0, mid_x + max_span / 2.0)
        ax.set_ylim(mid_z - max_span / 2.0, mid_z + max_span / 2.0)
        ax.set_zlim(mid_y - max_span / 2.0, mid_y + max_span / 2.0)

        # 颈部骨骼 (双肩中心至鼻尖)
        if len(joints_3d) > 12 and confs[0] >= 0.35 and confs[11] >= 0.35 and confs[12] >= 0.35:
            neck = (joints_3d[11] + joints_3d[12]) / 2.0
            ax.plot([neck[0], joints_3d[0, 0]], [neck[2], joints_3d[0, 2]], [neck[1], joints_3d[0, 1]], color="#E67E22", linewidth=2.5)

        # 各肢体骨骼着色连线
        for j1, j2 in self.skel_def.edges:
            if j1 < len(confs) and j2 < len(confs):
                if confs[j1] >= 0.35 and confs[j2] >= 0.35:
                    p1, p2 = self.skel_def.joints[j1].part, self.skel_def.joints[j2].part
                    if "left" in p1 or "left" in p2:
                        c = "#3498DB"  # 蓝色 (左肢)
                    elif "right" in p1 or "right" in p2:
                        c = "#2ECC71"  # 绿色 (右肢)
                    elif "head" in p1 or "head" in p2:
                        c = "#E67E22"  # 橙色 (头部)
                    else:
                        c = "#9B59B6"  # 紫色 (躯干)

                    ax.plot(
                        [joints_3d[j1, 0], joints_3d[j2, 0]],
                        [joints_3d[j1, 2], joints_3d[j2, 2]],
                        [joints_3d[j1, 1], joints_3d[j2, 1]],
                        color=c, linewidth=2.2,
                    )

        ax.scatter(joints_3d[valid_idx, 0], joints_3d[valid_idx, 2], joints_3d[valid_idx, 1], color="#1B4F72", s=25)
        if is_normalized:
            ax.scatter(0, 0, 0, color="red", marker="^", s=70, label="Root (0,0,0)")
            ax.legend(loc="upper right", fontsize=7)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Z (m)", fontsize=7)
        ax.set_zlabel("Y (m)", fontsize=7)

        if view_mode == "front":
            ax.view_init(elev=0, azim=-90)
        else:
            ax.view_init(elev=15, azim=-60)

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
        [1. RGB + 2D 骨架] [2. 深度图 + 投影] [3. 3D 骨架 (Front View)] [4. Normalized 3D 骨架] [5. 逐关节置信度]
        """
        fig = plt.figure(figsize=(22, 4.5))
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

        # Panel 3: 3D Camera Coordinate Skeleton (Front View, True Aspect Ratio)
        ax3 = fig.add_subplot(1, 5, 3, projection="3d")
        j3d = skeleton.joints_3d_camera
        depth_mean = float(np.mean(j3d[valid_idx, 2])) if len(valid_idx) > 0 else 0.0
        self.draw_3d_limbs(ax3, j3d, confs, f"3. Extracted 3D Pose\n(Z_depth={depth_mean:.2f}m)", is_normalized=False, view_mode="front")

        # Panel 4: Normalized 3D Skeleton (Root centered & scale normalized, Perspective Orbit View)
        ax4 = fig.add_subplot(1, 5, 4, projection="3d")
        norm_j3d = skeleton.joints_3d_normalized if skeleton.joints_3d_normalized is not None else j3d
        self.draw_3d_limbs(ax4, norm_j3d, confs, "4. Normalized 3D Pose\n(ST-GCN Input Ready)", is_normalized=True, view_mode="orbit")

        # Panel 5: 逐关节置信度直方图
        ax5 = fig.add_subplot(1, 5, 5)
        names = [self.skel_def.id_to_name.get(i, f"j_{i}")[:6] for i in range(len(confs))]
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
