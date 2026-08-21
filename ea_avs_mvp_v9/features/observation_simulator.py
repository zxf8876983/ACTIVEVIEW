"""
外部视觉姿态估计与感知质量模拟器 —— observation_simulator.py
============================================================

职责：
    1. 模拟现实机器人从单视角 RGB 观测中提取的人体估计姿态与置信度；
    2. 基于视点空间几何与遮挡关系，模拟关节点置信度衰减、高斯定位噪声与关节点缺失；
    3. 输出结构化 ObservationState，作为 Perception-Aware 决策模型的唯一输入。
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ea_avs_mvp_v9.core.types import ObservationState

# 7 大身体关键解剖部位关联关节
BODY_PART_JOINTS = {
    "head": ["head", "neck"],
    "torso": ["spine", "chest"],
    "pelvis": ["pelvis", "left_hip", "right_hip"],
    "left_hand": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_hand": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_leg": ["left_knee", "left_ankle"],
    "right_leg": ["right_knee", "right_ankle"],
}

ALL_STANDARD_JOINTS = [
    "pelvis", "head", "neck", "spine", "chest",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class ObservationSimulator:
    """视觉姿态估计与感知退化模拟器。"""

    def __init__(
        self,
        noise_std_visible: float = 0.01,
        noise_std_occluded: float = 0.06,
        missing_threshold: float = 0.20,
    ):
        self.noise_std_vis = noise_std_visible
        self.noise_std_occ = noise_std_occluded
        self.missing_thresh = missing_threshold

    def simulate_observation(
        self,
        gt_joints: Dict[str, List[float]],
        camera_pos: List[float],
        human_pos: List[float],
        human_yaw_deg: float = 0.0,
        degradation_mode: str = "auto",
        rng: Optional[np.random.RandomState] = None,
    ) -> ObservationState:
        """
        根据当前相机视点位置与人体相对空间几何，模拟生成视觉估计结果。
        
        Args:
            gt_joints: 真实关节点 3D 坐标
            camera_pos: 相机 [x, y, z] 坐标
            human_pos: 人体根部 [x, y, z] 坐标
            human_yaw_deg: 人体朝向角 (度)
            degradation_mode: "auto", "none", "self_occlusion", "furniture_occlusion", "low_confidence"
            rng: 随机数生成器
        """
        r = rng if rng is not None else np.random.RandomState(42)

        # 计算相机相对人体的水平偏角
        dx = camera_pos[0] - human_pos[0]
        dz = camera_pos[2] - human_pos[2]
        dist = math.sqrt(dx * dx + dz * dz)
        cam_angle_rad = math.atan2(dx, dz)
        human_yaw_rad = math.radians(human_yaw_deg)
        rel_angle_deg = math.degrees(cam_angle_rad - human_yaw_rad) % 360.0

        estimated_joints = {}
        joint_confidences = {}

        pelvis_gt = gt_joints.get("pelvis", human_pos)

        for j_name in ALL_STANDARD_JOINTS:
            if j_name in gt_joints:
                gx, gy, gz = gt_joints[j_name]
            else:
                gx, gy, gz = pelvis_gt

            # 判断该关节相对相机的自遮挡情况 (例如背向视角遮挡前胸与双手)
            is_front = (rel_angle_deg < 90.0 or rel_angle_deg > 270.0)
            is_right_side = (rel_angle_deg >= 0.0 and rel_angle_deg < 180.0)

            # 默认基础置信度
            base_conf = 0.95

            # 距离衰减
            dist_factor = max(0.5, 1.0 - (dist - 1.5) * 0.15)
            base_conf *= dist_factor

            # 根据姿态与相对视角模拟遮挡退化
            if degradation_mode == "none":
                conf = float(np.clip(base_conf - r.uniform(0.0, 0.05), 0.85, 1.0))
            elif degradation_mode == "self_occlusion":
                # 背对相机时前胸及手腕严重遮挡
                if not is_front and any(k in j_name for k in ["chest", "wrist", "elbow"]):
                    conf = float(r.uniform(0.10, 0.35))
                elif not is_right_side and "right" in j_name:
                    conf = float(r.uniform(0.15, 0.40))
                else:
                    conf = float(np.clip(base_conf - r.uniform(0.0, 0.15), 0.60, 0.95))
            elif degradation_mode == "furniture_occlusion":
                # 下肢被桌椅/家具严重遮挡
                if any(k in j_name for k in ["knee", "ankle", "hip"]):
                    conf = float(r.uniform(0.05, 0.25))
                else:
                    conf = float(np.clip(base_conf - r.uniform(0.0, 0.10), 0.70, 0.95))
            elif degradation_mode == "low_confidence":
                conf = float(r.uniform(0.20, 0.50))
            else:  # "auto" 几何计算
                if not is_front and any(k in j_name for k in ["wrist", "elbow"]):
                    conf = float(r.uniform(0.20, 0.50))
                else:
                    conf = float(np.clip(base_conf - r.uniform(0.0, 0.10), 0.65, 0.98))

            joint_confidences[j_name] = conf

            # 根据置信度加入定位噪声或缺失处理
            if conf < self.missing_thresh:
                # 缺失关节：位置退化至骨盆估计加上较大偏差
                ex = pelvis_gt[0] + float(r.normal(0, 0.20))
                ey = pelvis_gt[1] + float(r.normal(0, 0.20))
                ez = pelvis_gt[2] + float(r.normal(0, 0.20))
            else:
                noise_scale = self.noise_std_occ if conf < 0.60 else self.noise_std_vis
                ex = gx + float(r.normal(0, noise_scale))
                ey = gy + float(r.normal(0, noise_scale))
                ez = gz + float(r.normal(0, noise_scale))

            estimated_joints[j_name] = [round(ex, 3), round(ey, 3), round(ez, 3)]

        # 计算 7 大部位置信度
        body_part_confs = {}
        for part_name, p_joints in BODY_PART_JOINTS.items():
            vals = [joint_confidences[j] for j in p_joints if j in joint_confidences]
            body_part_confs[part_name] = float(np.mean(vals)) if vals else 0.5

        mean_c = float(np.mean(list(joint_confidences.values())))
        missing_c = sum(1 for c in joint_confidences.values() if c < self.missing_thresh)

        return ObservationState(
            estimated_joints_3d=estimated_joints,
            joint_confidences=joint_confidences,
            body_part_confidences=body_part_confs,
            mean_confidence=mean_c,
            missing_joint_count=missing_c,
        )
