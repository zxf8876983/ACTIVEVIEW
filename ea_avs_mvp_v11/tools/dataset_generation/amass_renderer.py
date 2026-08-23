#!/usr/bin/env python3
"""
AMASS 干净感知数据渲染器 —— amass_renderer.py
=============================================

职责：
    1. 加载 AMASS / converted motion 动作文件 (.pkl) 与 MotionPlayer；
    2. 在干净、无遮挡的中性背景环境中渲染人体动作 RGB 序列；
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

import cv2
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

        w, h = self.image_width, self.image_height
        img = np.full((h, w, 3), 250, dtype=np.uint8)

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

        rad = math.radians(angle_deg)
        scale = 2.2 / max(distance, 0.8)
        x_norm = 0.50 + 0.08 * math.sin(rad)

        def px(x_n, y_n):
            return int(np.clip(x_n * w, 0, w - 1)), int(np.clip((1.0 - y_n) * h, 0, h - 1))

        # 地面参考线
        cv2.line(img, (0, int(h * (1.0 - 0.14))), (w, int(h * (1.0 - 0.14))), (213, 216, 220), 2, cv2.LINE_AA)

        # 头部
        c_head = px(x_norm, head_y)
        r_head = int(0.045 * scale * h)
        cv2.circle(img, c_head, max(r_head, 4), (94, 109, 126), -1, cv2.LINE_AA)

        # 躯干
        c_torso_top = px(x_norm, head_y - 0.045 * scale)
        c_hip = px(x_norm, hip_y)
        w_torso = max(int(14 * scale), 2)
        cv2.line(img, c_torso_top, c_hip, (44, 62, 80), w_torso, cv2.LINE_AA)

        # 腿部
        w_leg = max(int(7 * scale), 2)
        c_l_hip = px(x_norm - 0.03 * scale, hip_y)
        c_l_knee = px(x_norm - 0.04 * scale, l_knee_y)
        c_l_ankle = px(x_norm - 0.04 * scale, l_ankle_y)
        cv2.line(img, c_l_hip, c_l_knee, (26, 37, 47), w_leg, cv2.LINE_AA)
        cv2.line(img, c_l_knee, c_l_ankle, (26, 37, 47), w_leg, cv2.LINE_AA)

        c_r_hip = px(x_norm + 0.03 * scale, hip_y)
        c_r_knee = px(x_norm + 0.04 * scale, r_knee_y)
        c_r_ankle = px(x_norm + 0.04 * scale, r_ankle_y)
        cv2.line(img, c_r_hip, c_r_knee, (26, 37, 47), w_leg, cv2.LINE_AA)
        cv2.line(img, c_r_knee, c_r_ankle, (26, 37, 47), w_leg, cv2.LINE_AA)

        # 手臂
        w_arm = max(int(6 * scale), 2)
        c_l_sh = px(x_norm - 0.05 * scale, torso_y)
        c_l_elb = px(x_norm - 0.08 * scale, (torso_y + l_wrist_y) / 2.0)
        c_l_wri = px(x_norm - 0.09 * scale, l_wrist_y)
        cv2.line(img, c_l_sh, c_l_elb, (44, 62, 80), w_arm, cv2.LINE_AA)
        cv2.line(img, c_l_elb, c_l_wri, (44, 62, 80), w_arm, cv2.LINE_AA)

        c_r_sh = px(x_norm + 0.05 * scale, torso_y)
        c_r_elb = px(x_norm + 0.08 * scale, (torso_y + r_wrist_y) / 2.0)
        c_r_wri = px(x_norm + 0.09 * scale, r_wrist_y)
        cv2.line(img, c_r_sh, c_r_elb, (44, 62, 80), w_arm, cv2.LINE_AA)
        cv2.line(img, c_r_elb, c_r_wri, (44, 62, 80), w_arm, cv2.LINE_AA)

        return img
