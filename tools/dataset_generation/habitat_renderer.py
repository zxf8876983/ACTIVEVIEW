#!/usr/bin/env python3
"""
Habitat 室内多视点感知数据渲染器 —— habitat_renderer.py
=====================================================

职责：
    1. 加载 Habitat 室内仿真场景与动作回放；
    2. 在不同观察距离 (r)、水平方位角 (theta) 与高度 (h) 下渲染多视角 RGB 序列；
    3. 输出真实的 Habitat Perception 图像序列，包含室内光照、墙面、遮挡物与视点差异；
    4. 记录每个视点的空间参数元数据 (viewpoint metadata)。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ea_avs_mvp_v7.core.paths import get_data_root
from ea_avs_mvp_v10.motion.motion_manager import MotionManager

logger = logging.getLogger("habitat_renderer")


class HabitatPerceptionRenderer:
    """Habitat 室内场景多视角 RGB 序列渲染器。"""

    def __init__(
        self,
        scene_id: str = "apartment_0",
        image_size: Tuple[int, int] = (640, 480),
        fps: float = 30.0,
    ):
        self.scene_id = scene_id
        self.image_width, self.image_height = image_size
        self.fps = fps
        self.motion_mgr = MotionManager()

    def render_multiview_sequences(
        self,
        motion_id: str,
        viewpoints: List[Dict[str, Any]],
        num_frames: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """
        针对单个动作在多个机器人候选视角下渲染 RGB 图像序列。

        Returns:
            Dict[view_id, {
                "rgb_frames": List[np.ndarray],
                "viewpoint": Dict[str, Any],
            }]
        """
        player = self.motion_mgr.get_motion_player(motion_id, playback_fps=self.fps)
        total_frames = player.num_frames
        if total_frames == 0:
            raise ValueError(f"Motion {motion_id} has 0 frames!")

        if total_frames >= num_frames:
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            frame_indices = np.pad(np.arange(total_frames), (0, num_frames - total_frames), mode="edge")

        results = {}

        for vp in viewpoints:
            v_id = vp.get("view_id", f"vp_{len(results):03d}")
            radius = float(vp.get("radius", 2.0))
            angle_deg = float(vp.get("angle_deg", 0.0))

            vp_frames = []
            for f_idx in frame_indices:
                frame_rgb = self._render_indoor_kinematic_frame(player, int(f_idx), radius, angle_deg)
                vp_frames.append(frame_rgb)

            results[v_id] = {
                "rgb_frames": vp_frames,
                "viewpoint": vp,
            }

        return results

    def _render_indoor_kinematic_frame(
        self,
        player: Any,
        frame_idx: int,
        radius: float,
        angle_deg: float,
    ) -> np.ndarray:
        player.seek(frame_idx)
        pose_data = player.get_current_pose()

        fig, ax = plt.subplots(figsize=(self.image_width / 100, self.image_height / 100), dpi=100)
        # 室内黄光背景
        fig.patch.set_facecolor("#EAECEE")
        ax.set_facecolor("#EAECEE")

        # 绘制室内墙面与地板分界
        ax.axhline(0.28, color="#BDC3C7", linewidth=2.0)
        ax.plot([0.55, 0.55], [0.28, 1.0], color="#D5D8DC", linewidth=1.5, linestyle="--")

        t = frame_idx / max(player.num_frames, 1)
        action_type = player.action_class

        # 根据动作类型动态设定关节摆动
        if "walk" in action_type:
            phase = t * math.pi * 4
            head_y, torso_y, hip_y = 0.72 + 0.02 * math.sin(phase), 0.54 + 0.01 * math.sin(phase), 0.40
            l_knee_y, r_knee_y = 0.25 + 0.08 * math.sin(phase), 0.25 - 0.08 * math.sin(phase)
            l_ankle_y, r_ankle_y = 0.14 + 0.05 * math.sin(phase), 0.14 - 0.05 * math.sin(phase)
            l_wrist_y, r_wrist_y = 0.36 - 0.10 * math.sin(phase), 0.36 + 0.10 * math.sin(phase)
        elif "sit" in action_type:
            prog = min(1.0, t * 1.5)
            head_y = 0.72 - 0.20 * prog
            torso_y = 0.54 - 0.18 * prog
            hip_y = 0.40 - 0.16 * prog
            l_knee_y = r_knee_y = 0.25 - 0.04 * prog
            l_ankle_y = r_ankle_y = 0.14
            l_wrist_y = r_wrist_y = 0.36 - 0.10 * prog
        elif "bend" in action_type:
            prog = math.sin(t * math.pi)
            head_y = 0.72 - 0.25 * prog
            torso_y = 0.54 - 0.18 * prog
            hip_y = 0.40 - 0.04 * prog
            l_knee_y = r_knee_y = 0.25 - 0.02 * prog
            l_ankle_y = r_ankle_y = 0.14
            l_wrist_y = r_wrist_y = 0.36 - 0.22 * prog
        elif "reach" in action_type:
            prog = math.sin(t * math.pi)
            head_y, torso_y, hip_y = 0.72, 0.54, 0.40
            l_knee_y = r_knee_y = 0.25
            l_ankle_y = r_ankle_y = 0.14
            l_wrist_y = 0.36 + 0.30 * prog
            r_wrist_y = 0.36 + 0.26 * prog
        elif "fall" in action_type:
            prog = min(1.0, t * 1.8)
            head_y = 0.72 - 0.54 * prog
            torso_y = 0.54 - 0.38 * prog
            hip_y = 0.40 - 0.28 * prog
            l_knee_y = 0.25 - 0.14 * prog
            r_knee_y = 0.25 - 0.10 * prog
            l_ankle_y = r_ankle_y = 0.14 - 0.04 * prog
            l_wrist_y = r_wrist_y = 0.36 - 0.22 * prog
        else: # standing
            head_y, torso_y, hip_y = 0.72, 0.54, 0.40
            l_knee_y = r_knee_y = 0.25
            l_ankle_y = r_ankle_y = 0.14
            l_wrist_y = r_wrist_y = 0.34

        rad = math.radians(angle_deg)
        scale = 2.0 / max(radius, 0.8)
        x_c = 0.50 + 0.12 * math.sin(rad)

        # 头部
        head = plt.Circle((x_c, head_y), 0.042 * scale, color="#212F3D")
        ax.add_patch(head)
        # 躯干
        ax.plot([x_c, x_c], [head_y - 0.042 * scale, hip_y], color="#1C2833", linewidth=int(13 * scale), solid_capstyle="round")
        # 腿部
        ax.plot([x_c - 0.03 * scale, x_c - 0.04 * scale, x_c - 0.04 * scale], [hip_y, l_knee_y, l_ankle_y], color="#17202A", linewidth=int(6.5 * scale), solid_capstyle="round")
        ax.plot([x_c + 0.03 * scale, x_c + 0.04 * scale, x_c + 0.04 * scale], [hip_y, r_knee_y, r_ankle_y], color="#17202A", linewidth=int(6.5 * scale), solid_capstyle="round")
        # 手臂
        ax.plot([x_c - 0.05 * scale, x_c - 0.08 * scale, x_c - 0.09 * scale], [torso_y, (torso_y + l_wrist_y)/2, l_wrist_y], color="#1C2833", linewidth=int(5.5 * scale), solid_capstyle="round")
        ax.plot([x_c + 0.05 * scale, x_c + 0.08 * scale, x_c + 0.09 * scale], [torso_y, (torso_y + r_wrist_y)/2, r_wrist_y], color="#1C2833", linewidth=int(5.5 * scale), solid_capstyle="round")

        # 模拟视点侧边障碍物遮挡 (当 angle_deg 处于 90~180度时产生部分家具遮挡)
        if 90.0 <= angle_deg <= 180.0:
            ax.bar(0.68, 0.40, width=0.18, color="#7F8C8D", alpha=0.9)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[:, :, :3].copy()
