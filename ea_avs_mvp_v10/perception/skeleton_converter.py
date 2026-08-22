"""
COCO-17 3D 骨架生成与置信度融合器 —— skeleton_converter.py
=========================================================

职责：
    1. 保持 Pose Estimator 原生输出格式优先原则，Phase 2 默认原生输出 COCO-17 3D 骨架；
    2. 融合 2D 检测感知置信度 (conf_2d) 与深度空间连续性置信度 (conf_depth)：
       perception_confidence = conf_2d * conf_depth
    3. 基于感知置信度判定感知不确定性掩码 (uncertainty_mask)；
    4. 评估 6 大关键身体部位平均感知置信度 (head, torso, left_arm, right_arm, left_leg, right_leg)；
    5. 输出原生 EstimatedSkeleton3D 实体对象 (17 关键点，默认不强行转换拓扑)。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .depth_projection import DepthProjectionResult
from .pose_estimator import COCO_KEYPOINTS, COCO_SKELETON_PAIRS, Pose2DResult

logger = logging.getLogger(__name__)

# COCO-17 身体部位索引分组
COCO_BODY_PART_GROUPS: Dict[str, List[int]] = {
    "head": [0, 1, 2, 3, 4],            # nose, left_eye, right_eye, left_ear, right_ear
    "torso": [5, 6, 11, 12],             # left_shoulder, right_shoulder, left_hip, right_hip
    "left_arm": [5, 7, 9],               # left_shoulder, left_elbow, left_wrist
    "right_arm": [6, 8, 10],             # right_shoulder, right_elbow, right_wrist
    "left_leg": [11, 13, 15],            # left_hip, left_knee, left_ankle
    "right_leg": [12, 14, 16],           # right_hip, right_knee, right_ankle
}


@dataclass
class EstimatedSkeleton3D:
    """
    估计的 3D 人体骨架与感知质量结构体 (COCO-17 原生拓扑)。

    坐标系属性说明：
        - joints_3d_camera: (17, 3) 机器人局部相机系坐标 (米, +X右, +Y上, +Z前)
        - joints_3d_world: (17, 3) 全局世界系坐标 (米, 仅用于 Habitat 可视化)
        - joints_3d_normalized: (17, 3) 根节点平移与尺度归一化坐标 (ST-GCN 直接输入)
    """
    joint_format: str                     # "COCO17"
    joints_3d_camera: np.ndarray          # 形状 (17, 3) 相机系坐标 (米)
    joints_3d_world: np.ndarray           # 形状 (17, 3) 世界系坐标 (米)
    joints_2d: np.ndarray                 # 形状 (17, 2) 图像系像素坐标
    perception_confidence: np.ndarray     # 形状 (17,) 感知复合置信度 [0.0, 1.0]
    uncertainty_mask: np.ndarray          # 形状 (17,) bool (True=高感知不确定性/遮挡)
    part_confidence: Dict[str, float]     # 6 大部位平均感知置信度
    joints_3d_normalized: Optional[np.ndarray] = None  # 形状 (17, 3) 归一化坐标
    joint_names: List[str] = field(default_factory=lambda: list(COCO_KEYPOINTS))

    # 兼容性别名属性
    @property
    def joints_3d_cam(self) -> np.ndarray:
        return self.joints_3d_camera

    @property
    def confidence(self) -> np.ndarray:
        return self.perception_confidence

    @property
    def occluded_mask(self) -> np.ndarray:
        return self.uncertainty_mask

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_format": self.joint_format,
            "joints_3d_camera": self.joints_3d_camera.tolist(),
            "joints_3d_cam": self.joints_3d_camera.tolist(),
            "joints_3d_world": self.joints_3d_world.tolist(),
            "joints_3d_normalized": self.joints_3d_normalized.tolist() if self.joints_3d_normalized is not None else None,
            "joints_2d": self.joints_2d.tolist(),
            "perception_confidence": self.perception_confidence.tolist(),
            "confidence": self.perception_confidence.tolist(),
            "uncertainty_mask": self.uncertainty_mask.tolist(),
            "occluded_mask": self.uncertainty_mask.tolist(),
            "part_confidence": {k: float(v) for k, v in self.part_confidence.items()},
            "joint_names": self.joint_names,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EstimatedSkeleton3D":
        norm_joints = np.array(data["joints_3d_normalized"], dtype=np.float32) if data.get("joints_3d_normalized") is not None else None
        cam_joints = np.array(data.get("joints_3d_camera", data.get("joints_3d_cam")), dtype=np.float32)
        conf = np.array(data.get("perception_confidence", data.get("confidence")), dtype=np.float32)
        unc_mask = np.array(data.get("uncertainty_mask", data.get("occluded_mask")), dtype=bool)

        return cls(
            joint_format=data.get("joint_format", "COCO17"),
            joints_3d_camera=cam_joints,
            joints_3d_world=np.array(data.get("joints_3d_world", cam_joints), dtype=np.float32),
            joints_3d_normalized=norm_joints,
            joints_2d=np.array(data["joints_2d"], dtype=np.float32),
            perception_confidence=conf,
            uncertainty_mask=unc_mask,
            part_confidence=data.get("part_confidence", {}),
            joint_names=data.get("joint_names", list(COCO_KEYPOINTS)),
        )


class SkeletonConverter:
    """COCO-17 骨架置信度融合与 3D 骨架生成器。"""

    def __init__(self, uncertainty_conf_thresh: float = 0.35):
        self.uncertainty_conf_thresh = float(uncertainty_conf_thresh)

    def convert_and_fuse(
        self,
        pose2d: Pose2DResult,
        depth_res: DepthProjectionResult,
    ) -> EstimatedSkeleton3D:
        """
        融合 2D 姿态检测与 3D 深度逆投影，生成原生 COCO-17 3D 骨架。

        Args:
            pose2d: 2D 姿态检测结果 (COCO-17)
            depth_res: 深度反投影结果 (COCO-17 对应 3D 坐标与深度置信度)

        Returns:
            EstimatedSkeleton3D (COCO-17)
        """
        kpts_2d = pose2d.keypoints                # (17, 2)
        conf_2d = pose2d.confidence               # (17,)
        joints_cam = depth_res.joints_3d_cam      # (17, 3)
        joints_world = depth_res.joints_3d_world  # (17, 3)
        conf_depth = depth_res.depth_confidence  # (17,)

        # 1. 逐关节复合感知置信度: c = c_2d * c_depth
        conf_composite = conf_2d * conf_depth     # (17,)

        # 2. 感知不确定性 / 遮挡掩码判定 (低于阈值标记为不确定)
        uncertainty_mask = conf_composite < self.uncertainty_conf_thresh

        # 3. 计算 6 大身体部位平均感知置信度
        part_conf: Dict[str, float] = {}
        for part_name, joint_indices in COCO_BODY_PART_GROUPS.items():
            sub_confs = [conf_composite[j] for j in joint_indices]
            part_conf[part_name] = float(np.mean(sub_confs))

        return EstimatedSkeleton3D(
            joint_format="COCO17",
            joints_3d_camera=joints_cam.copy(),
            joints_3d_world=joints_world.copy(),
            joints_2d=kpts_2d.copy(),
            perception_confidence=conf_composite.astype(np.float32),
            uncertainty_mask=uncertainty_mask,
            part_confidence=part_conf,
            joint_names=list(COCO_KEYPOINTS),
        )
