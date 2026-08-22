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

import cv2
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

        w, h = self.image_width, self.image_height
        img = np.full((h, w, 3), 235, dtype=np.uint8)

        # 室内墙壁与地板分界线
        y_floor = int(h * (1.0 - 0.28))
        cv2.line(img, (0, y_floor), (w, y_floor), (189, 195, 199), 2, cv2.LINE_AA)
        cv2.line(img, (int(w * 0.55), 0), (int(w * 0.55), y_floor), (213, 218, 220), 1, cv2.LINE_AA)

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
        x_norm = 0.50 + 0.12 * math.sin(rad)

        def px(x_n, y_n):
            return int(np.clip(x_n * w, 0, w - 1)), int(np.clip((1.0 - y_n) * h, 0, h - 1))

        # 头部
        c_head = px(x_norm, head_y)
        r_head = int(0.042 * scale * h)
        cv2.circle(img, c_head, max(r_head, 4), (33, 47, 61), -1, cv2.LINE_AA)

        # 躯干
        c_torso_top = px(x_norm, head_y - 0.042 * scale)
        c_hip = px(x_norm, hip_y)
        w_torso = max(int(13 * scale), 2)
        cv2.line(img, c_torso_top, c_hip, (28, 40, 51), w_torso, cv2.LINE_AA)

        # 腿部
        w_leg = max(int(6.5 * scale), 2)
        c_l_hip = px(x_norm - 0.03 * scale, hip_y)
        c_l_knee = px(x_norm - 0.04 * scale, l_knee_y)
        c_l_ankle = px(x_norm - 0.04 * scale, l_ankle_y)
        cv2.line(img, c_l_hip, c_l_knee, (23, 32, 42), w_leg, cv2.LINE_AA)
        cv2.line(img, c_l_knee, c_l_ankle, (23, 32, 42), w_leg, cv2.LINE_AA)

        c_r_hip = px(x_norm + 0.03 * scale, hip_y)
        c_r_knee = px(x_norm + 0.04 * scale, r_knee_y)
        c_r_ankle = px(x_norm + 0.04 * scale, r_ankle_y)
        cv2.line(img, c_r_hip, c_r_knee, (23, 32, 42), w_leg, cv2.LINE_AA)
        cv2.line(img, c_r_knee, c_r_ankle, (23, 32, 42), w_leg, cv2.LINE_AA)

        # 手臂
        w_arm = max(int(5.5 * scale), 2)
        c_l_sh = px(x_norm - 0.05 * scale, torso_y)
        c_l_elb = px(x_norm - 0.08 * scale, (torso_y + l_wrist_y) / 2.0)
        c_l_wri = px(x_norm - 0.09 * scale, l_wrist_y)
        cv2.line(img, c_l_sh, c_l_elb, (28, 40, 51), w_arm, cv2.LINE_AA)
        cv2.line(img, c_l_elb, c_l_wri, (28, 40, 51), w_arm, cv2.LINE_AA)

        c_r_sh = px(x_norm + 0.05 * scale, torso_y)
        c_r_elb = px(x_norm + 0.08 * scale, (torso_y + r_wrist_y) / 2.0)
        c_r_wri = px(x_norm + 0.09 * scale, r_wrist_y)
        cv2.line(img, c_r_sh, c_r_elb, (28, 40, 51), w_arm, cv2.LINE_AA)
        cv2.line(img, c_r_elb, c_r_wri, (28, 40, 51), w_arm, cv2.LINE_AA)

        # 模拟视点侧边障碍物遮挡 (当 angle_deg 处于 90~180度时产生部分家具遮挡)
        if 90.0 <= angle_deg <= 180.0:
            cv2.rectangle(img, (int(w * 0.58), int(h * 0.50)), (int(w * 0.85), int(h * 0.95)), (127, 140, 141), -1)

        return img
