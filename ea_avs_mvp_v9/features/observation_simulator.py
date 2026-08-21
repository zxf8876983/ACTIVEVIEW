"""
外部视觉姿态估计与感知质量模拟器 —— observation_simulator.py
============================================================

科研定位与信息边界：
    1. GT is only used for supervision/evaluation. It must never enter model forward pass.
    2. 本模块作为感知抽象层 (Perception Abstraction Layer)，模拟真实视觉姿态估计器的误差、遮挡与缺失；
    3. BaseObservationProvider: 统一感知提供者抽象基类 (为未来 v10.0 RGB Pose Estimator 预留接口)；
    4. ObservationSimulator: 模拟真实视觉感知误差与不完整观测：
       - Joint localization noise: p_est = p_gt + epsilon
       - Joint confidence: c_i ∈ [0.0, 1.0]
       - Missing joint simulation: 随机/几何缺失 (wrists, ankles, elbows, knees)
       - Body part confidence: head, torso, pelvis, hands, legs (7 大解剖部位)
    5. 支持参数化控制：
       - 随机种子控制 (seed=0, 1, 2, 3, 4) 保证实验完全可复现
       - 噪声等级控制 (low_noise, medium_noise, high_noise)
       - 遮挡等级控制 (self_occlusion_weak, self_occlusion_medium, self_occlusion_strong)
       - 缺失关节点比例控制 (missing_keypoints_ratio)
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional, Tuple, Union
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


class BaseObservationProvider(ABC):
    """感知数据提供者抽象基类 (为 v10+ 真实视觉感知模型预留统一接口)。"""

    @abstractmethod
    def get_observation(
        self,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ObservationState:
        """获取当前视角下的人体估计感知状态。"""
        pass


def compute_observation_quality(obs: ObservationState, dist: float = 2.0) -> float:
    """
    计算单个观测状态的综合感知质量得分 (Quality(O)):
      Quality = 0.40 * c_joints + 0.35 * c_parts + 0.25 * completeness - 0.10 * dist_pen
    """
    c_joints = obs.mean_confidence
    c_parts = float(np.mean(list(obs.body_part_confidences.values()))) if obs.body_part_confidences else 0.5
    completeness = obs.completeness_score
    dist_pen = min(1.0, abs(dist - 2.0) / 2.0)

    q = 0.40 * c_joints + 0.35 * c_parts + 0.25 * completeness - 0.10 * dist_pen
    return float(np.clip(q, 0.0, 1.0))


class ObservationSimulator(BaseObservationProvider):
    """视觉姿态估计与感知退化模拟器。"""

    def __init__(
        self,
        noise_std_visible: float = 0.015,
        noise_std_occluded: float = 0.060,
        missing_threshold: float = 0.25,
    ):
        self.noise_std_vis = noise_std_visible
        self.noise_std_occ = noise_std_occluded
        self.missing_thresh = missing_threshold

    def get_observation(
        self,
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ObservationState:
        """统一感知接口实现。"""
        gt_joints = kwargs.get("gt_joints", {})
        camera_pos = kwargs.get("camera_pos", [0.0, 1.2, 2.0])
        human_pos = kwargs.get("human_pos", [0.0, 0.0, 0.0])
        human_yaw_deg = kwargs.get("human_yaw_deg", 0.0)
        degradation_mode = kwargs.get("degradation_mode", "auto")
        seed = kwargs.get("seed", None)
        rng = kwargs.get("rng", None)

        return self.simulate_observation(
            gt_joints=gt_joints,
            camera_pos=camera_pos,
            human_pos=human_pos,
            human_yaw_deg=human_yaw_deg,
            degradation_mode=degradation_mode,
            seed=seed,
            rng=rng,
        )

    def simulate_observation(
        self,
        gt_joints: Dict[str, List[float]],
        camera_pos: List[float],
        human_pos: List[float],
        human_yaw_deg: float = 0.0,
        degradation_mode: str = "auto",
        noise_level: str = "medium_noise",
        occlusion_level: str = "self_occlusion_medium",
        missing_ratio: float = 0.0,
        seed: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> ObservationState:
        """
        根据当前相机视点位置与人体相对空间几何，模拟生成视觉估计结果。
        
        # GT is only used for supervision/evaluation.
        # It must never enter model forward pass.
        
        Args:
            gt_joints: 真实关节点 3D 坐标 (仅供模拟生成估计观测)
            camera_pos: 相机 [x, y, z] 坐标
            human_pos: 人体根部 [x, y, z] 坐标
            human_yaw_deg: 人体朝向角 (度)
            degradation_mode: "auto", "clean", "none", "self_occlusion", "furniture_occlusion", "heavy_noise", "missing_keypoints"
            noise_level: "low_noise", "medium_noise", "high_noise"
            occlusion_level: "self_occlusion_weak", "self_occlusion_medium", "self_occlusion_strong"
            missing_ratio: 强制缺失关键点比例 [0.0, 1.0]
            seed: 随机数种子 (支持 seed=0, 1, 2, 3, 4 严格复现)
            rng: 随机数生成器
        """
        if rng is not None:
            r = rng
        elif seed is not None:
            r = np.random.RandomState(seed)
        else:
            r = np.random.RandomState(42)

        # 1. 计算相机相对人体的几何关系
        dx = camera_pos[0] - human_pos[0]
        dz = camera_pos[2] - human_pos[2]
        dist = math.sqrt(dx * dx + dz * dz)
        cam_angle_rad = math.atan2(dx, dz)
        human_yaw_rad = math.radians(human_yaw_deg)
        rel_angle_deg = math.degrees(cam_angle_rad - human_yaw_rad) % 360.0

        is_front = (rel_angle_deg < 90.0 or rel_angle_deg > 270.0)
        is_right_side = (rel_angle_deg >= 0.0 and rel_angle_deg < 180.0)

        # 2. 距离衰减因子
        dist_factor = max(0.4, 1.0 - max(0.0, dist - 1.8) * 0.18)

        estimated_joints = {}
        joint_confidences = {}
        missing_joints = []

        pelvis_gt = gt_joints.get("pelvis", human_pos)

        # 确定噪声尺度
        if degradation_mode in ("heavy_noise", "severe_noise", "low_confidence") or noise_level == "high_noise":
            base_noise_scale = 0.080  # 8cm
        elif noise_level == "low_noise" or degradation_mode in ("clean", "none"):
            base_noise_scale = 0.010  # 1cm
        else:
            base_noise_scale = 0.025  # 2.5cm

        for j_name in ALL_STANDARD_JOINTS:
            if j_name in gt_joints:
                gx, gy, gz = gt_joints[j_name]
            else:
                gx, gy, gz = pelvis_gt

            base_conf = 0.95 * dist_factor

            # 根据感知退化模式计算关节点置信度
            if degradation_mode in ("clean", "none"):
                conf = float(np.clip(base_conf - r.uniform(0.0, 0.05), 0.85, 1.0))
                noise_scale = base_noise_scale

            elif degradation_mode == "furniture_occlusion":
                # 下肢严重遮挡 (膝盖、脚踝等缺失)
                if any(k in j_name for k in ["knee", "ankle", "hip"]):
                    conf = float(r.uniform(0.05, 0.20))
                    noise_scale = self.noise_std_occ * 2.0
                else:
                    conf = float(np.clip(base_conf - r.uniform(0.0, 0.10), 0.70, 0.95))
                    noise_scale = self.noise_std_vis

            elif degradation_mode in ("heavy_noise", "severe_noise", "low_confidence") or noise_level == "high_noise":
                conf = float(r.uniform(0.18, 0.45))
                noise_scale = base_noise_scale

            elif degradation_mode == "missing_keypoints" or missing_ratio > 0.0:
                # 关键肢体端点严重缺失
                if any(k in j_name for k in ["wrist", "ankle"]) or (missing_ratio > 0.0 and r.uniform(0, 1) < missing_ratio):
                    conf = float(r.uniform(0.02, 0.15))
                    noise_scale = 0.150
                else:
                    conf = float(np.clip(base_conf - r.uniform(0.0, 0.15), 0.55, 0.88))
                    noise_scale = self.noise_std_vis

            elif degradation_mode == "self_occlusion":
                # 遮挡程度分级
                if occlusion_level == "self_occlusion_weak":
                    occ_min, occ_max = 0.30, 0.55
                elif occlusion_level == "self_occlusion_strong":
                    occ_min, occ_max = 0.05, 0.20
                else:  # medium
                    occ_min, occ_max = 0.10, 0.35

                if not is_front and any(k in j_name for k in ["chest", "wrist", "elbow"]):
                    conf = float(r.uniform(occ_min, occ_max))
                    noise_scale = self.noise_std_occ * 1.5
                elif not is_right_side and "right" in j_name:
                    conf = float(r.uniform(occ_min + 0.05, occ_max + 0.05))
                    noise_scale = self.noise_std_occ
                else:
                    conf = float(np.clip(base_conf - r.uniform(0.0, 0.15), 0.65, 0.95))
                    noise_scale = self.noise_std_vis

            else:  # "auto" 几何连续遮挡
                if not is_front and any(k in j_name for k in ["wrist", "elbow", "chest"]):
                    conf = float(r.uniform(0.20, 0.45))
                    noise_scale = self.noise_std_occ
                else:
                    conf = float(np.clip(base_conf - r.uniform(0.0, 0.10), 0.70, 0.98))
                    noise_scale = self.noise_std_vis

            joint_confidences[j_name] = conf

            # 判断缺失关节与合成估计坐标 (p_est = p_gt + epsilon)
            if conf < self.missing_thresh:
                missing_joints.append(j_name)
                ex = pelvis_gt[0] + float(r.normal(0, 0.25))
                ey = pelvis_gt[1] + float(r.normal(0, 0.25))
                ez = pelvis_gt[2] + float(r.normal(0, 0.25))
            else:
                ex = gx + float(r.normal(0, noise_scale))
                ey = gy + float(r.normal(0, noise_scale))
                ez = gz + float(r.normal(0, noise_scale))

            estimated_joints[j_name] = [round(ex, 3), round(ey, 3), round(ez, 3)]

        # 计算 7 大解剖部位可见置信度
        body_part_confs = {}
        for part_name, p_joints in BODY_PART_JOINTS.items():
            vals = [joint_confidences[j] for j in p_joints if j in joint_confidences]
            body_part_confs[part_name] = float(np.mean(vals)) if vals else 0.5

        mean_c = float(np.mean(list(joint_confidences.values())))
        missing_c = len(missing_joints)
        completeness = max(0.0, 1.0 - float(missing_c / len(ALL_STANDARD_JOINTS)))

        return ObservationState(
            estimated_joints_3d=estimated_joints,
            joint_confidences=joint_confidences,
            body_part_confidences=body_part_confs,
            missing_joint_names=missing_joints,
            mean_confidence=mean_c,
            missing_joint_count=missing_c,
            completeness_score=completeness,
        )
