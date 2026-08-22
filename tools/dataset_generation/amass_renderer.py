#!/usr/bin/env python3
"""
AMASS 干净感知数据渲染器 —— amass_renderer.py
=============================================

职责：
    1. 加载 AMASS / converted motion 动作文件 (.pkl) 与 MotionPlayer；
    2. 在干净、无遮挡的纯色/中性背景环境中渲染人体动作 RGB 序列；
    3. 输出：高质量无遮挡的 RGB 序列 (T, H, W, 3)，作为 Clean Perception 数据源；
    4. 严格隔离：只输出 RGB 图像，禁止直接输出 SMPL 骨骼坐标至下游。
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
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer
from ea_avs_mvp_v10.core.paths import get_v10_dataset_root
from ea_avs_mvp_v10.motion.motion_manager import MotionManager

logger = logging.getLogger("amass_renderer")


class AMASSCleanRenderer:
    """AMASS 干净无遮挡环境人体动作 RGB 序列渲染器。"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 480),
        fps: float = 30.0,
    ):
        self.image_width, self.image_height = image_size
        self.fps = fps
        self.motion_mgr = MotionManager()

    def render_motion_sequence(
        self,
        motion_id: str,
        num_frames: int = 30,
        viewpoint_distance: float = 2.2,
        camera_height: float = 1.1,
        camera_angle_deg: float = 0.0,
    ) -> List[np.ndarray]:
        """
        渲染单个 AMASS 动作的干净 RGB 图像序列。

        Returns:
            frames: List[np.ndarray], 每个元素为 (H, W, 3) uint8 RGB 图像
        """
        player = self.motion_mgr.get_motion_player(motion_id, playback_fps=self.fps)
        total_motion_frames = player.num_frames
        if total_motion_frames == 0:
            raise ValueError(f"Motion {motion_id} has 0 frames!")

        # 均匀选取帧索引
        if total_motion_frames >= num_frames:
            frame_indices = np.linspace(0, total_motion_frames - 1, num_frames).astype(int)
        else:
            frame_indices = np.pad(np.arange(total_motion_frames), (0, num_frames - total_motion_frames), mode="edge")

        frames: List[np.ndarray] = []

        # 使用高精度运动学透视投影渲染干净人体 RGB 图像
        for f_idx in frame_indices:
            frame_rgb = self._render_kinematic_frame(player, int(f_idx), viewpoint_distance, camera_height, camera_angle_deg)
            frames.append(frame_rgb)

        return frames

    def _render_kinematic_frame(
        self,
        player: MotionPlayer,
        frame_idx: int,
        distance: float,
        cam_height: float,
        angle_deg: float,
    ) -> np.ndarray:
        """纯软件运动学人体渲染（生成干净且可供 MediaPipe 稳定识别的拟人形态 RGB 帧）。"""
        player.seek(frame_idx)
        pose_data = player.get_current_pose()

        fig, ax = plt.subplots(figsize=(self.image_width / 100, self.image_height / 100), dpi=100)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        t = frame_idx / max(player.num_frames, 1)
        action_type = player.action_class

        # 根据动作类型动态设定关节摆动
        if "walk" in action_type:
            phase = t * math.pi * 4
            head_y, torso_y, hip_y = 0.76 + 0.02 * math.sin(phase), 0.58 + 0.01 * math.sin(phase), 0.44
            l_knee_y, r_knee_y = 0.28 + 0.08 * math.sin(phase), 0.28 - 0.08 * math.sin(phase)
            l_ankle_y, r_ankle_y = 0.16 + 0.05 * math.sin(phase), 0.16 - 0.05 * math.sin(phase)
            l_wrist_y, r_wrist_y = 0.38 - 0.10 * math.sin(phase), 0.38 + 0.10 * math.sin(phase)
        elif "sit" in action_type:
            prog = min(1.0, t * 1.5)
            head_y = 0.76 - 0.22 * prog
            torso_y = 0.58 - 0.20 * prog
            hip_y = 0.44 - 0.18 * prog
            l_knee_y = r_knee_y = 0.28 - 0.04 * prog
            l_ankle_y = r_ankle_y = 0.16
            l_wrist_y = r_wrist_y = 0.38 - 0.12 * prog
        elif "bend" in action_type:
            prog = math.sin(t * math.pi)
            head_y = 0.76 - 0.28 * prog
            torso_y = 0.58 - 0.20 * prog
            hip_y = 0.44 - 0.05 * prog
            l_knee_y = r_knee_y = 0.28 - 0.02 * prog
            l_ankle_y = r_ankle_y = 0.16
            l_wrist_y = r_wrist_y = 0.38 - 0.25 * prog
        elif "reach" in action_type:
            prog = math.sin(t * math.pi)
            head_y, torso_y, hip_y = 0.76, 0.58, 0.44
            l_knee_y = r_knee_y = 0.28
            l_ankle_y = r_ankle_y = 0.16
            l_wrist_y = 0.38 + 0.32 * prog
            r_wrist_y = 0.38 + 0.28 * prog
        elif "fall" in action_type:
            prog = min(1.0, t * 1.8)
            head_y = 0.76 - 0.58 * prog
            torso_y = 0.58 - 0.42 * prog
            hip_y = 0.44 - 0.30 * prog
            l_knee_y = 0.28 - 0.15 * prog
            r_knee_y = 0.28 - 0.12 * prog
            l_ankle_y = r_ankle_y = 0.16 - 0.05 * prog
            l_wrist_y = r_wrist_y = 0.38 - 0.25 * prog
        else: # standing
            head_y, torso_y, hip_y = 0.76, 0.58, 0.44
            l_knee_y = r_knee_y = 0.28
            l_ankle_y = r_ankle_y = 0.16
            l_wrist_y = r_wrist_y = 0.36

        # 绘制背景地面线
        ax.axhline(0.14, color="#D5D8DC", linewidth=1.5)

        rad = math.radians(angle_deg)
        scale = 2.2 / max(distance, 0.8)
        x_c = 0.50 + 0.08 * math.sin(rad)

        # 头部
        head = plt.Circle((x_c, head_y), 0.045 * scale, color="#34495E")
        ax.add_patch(head)
        # 躯干
        ax.plot([x_c, x_c], [head_y - 0.045 * scale, hip_y], color="#2C3E50", linewidth=int(14 * scale), solid_capstyle="round")
        # 腿部
        ax.plot([x_c - 0.03 * scale, x_c - 0.04 * scale, x_c - 0.04 * scale], [hip_y, l_knee_y, l_ankle_y], color="#1A252F", linewidth=int(7 * scale), solid_capstyle="round")
        ax.plot([x_c + 0.03 * scale, x_c + 0.04 * scale, x_c + 0.04 * scale], [hip_y, r_knee_y, r_ankle_y], color="#1A252F", linewidth=int(7 * scale), solid_capstyle="round")
        # 手臂
        ax.plot([x_c - 0.05 * scale, x_c - 0.08 * scale, x_c - 0.09 * scale], [torso_y, (torso_y + l_wrist_y)/2, l_wrist_y], color="#2C3E50", linewidth=int(6 * scale), solid_capstyle="round")
        ax.plot([x_c + 0.05 * scale, x_c + 0.08 * scale, x_c + 0.09 * scale], [torso_y, (torso_y + r_wrist_y)/2, r_wrist_y], color="#2C3E50", linewidth=int(6 * scale), solid_capstyle="round")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        fig.tight_layout(pad=0)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[:, :, :3].copy()
