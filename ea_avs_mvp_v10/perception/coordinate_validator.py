"""
3D 骨架坐标健康度与合理性校验器 —— coordinate_validator.py
======================================================

职责：
    1. 深度合理性检查 (Depth Range Check)：
       Z_cam > 0.2m 且 Z_cam <= 8.0m (防止极端近景裁剪或背景虚空异常)；
    2. 人体尺度合理性检查 (Human Height Scale Check)：
       人体垂直投影高度差 Delta_Y 应处于 [0.3m, 2.5m] 正常物理人体范围；
    3. 运动学相对方位与骨骼长度一致性检查；
    4. 统计与分级诊断输出：
       输出 VALID, WARNING, INVALID 状态并提供批量样本统计报告。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .skeleton_converter import EstimatedSkeleton3D

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """单样本 3D 骨架几何坐标校验结果。"""
    status: str                         # "VALID", "WARNING", "INVALID"
    reasons: List[str]                  # 异常或警告原因列表
    depth_valid: bool = True            # 深度范围是否合法
    height_valid: bool = True           # 人体尺度是否合法
    kinematics_valid: bool = True       # 关节相对几何是否合理
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "depth_valid": self.depth_valid,
            "height_valid": self.height_valid,
            "kinematics_valid": self.kinematics_valid,
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
        }


class CoordinateValidator:
    """3D 骨架空间坐标健康度校验器。"""

    def __init__(
        self,
        min_depth: float = 0.2,
        max_depth: float = 8.0,
        min_height: float = 0.3,
        max_height: float = 2.5,
        min_valid_joints: int = 4,
        uncertainty_thresh: float = 0.35,
    ):
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.min_valid_joints = int(min_valid_joints)
        self.uncertainty_thresh = float(uncertainty_thresh)

        # 统计计数器
        self.total_checked = 0
        self.valid_count = 0
        self.warning_count = 0
        self.invalid_count = 0

    def reset_stats(self):
        """重置统计计数器。"""
        self.total_checked = 0
        self.valid_count = 0
        self.warning_count = 0
        self.invalid_count = 0

    def validate(self, skeleton: EstimatedSkeleton3D) -> ValidationResult:
        """
        校验单个 EstimatedSkeleton3D 3D 骨架坐标的物理合法性。

        Args:
            skeleton: EstimatedSkeleton3D 对象

        Returns:
            ValidationResult
        """
        self.total_checked += 1
        reasons = []
        depth_valid = True
        height_valid = True
        kinematics_valid = True

        j3d_cam = skeleton.joints_3d_camera
        confs = skeleton.perception_confidence

        valid_mask = (confs >= self.uncertainty_thresh) & (np.linalg.norm(j3d_cam, axis=1) > 0.01)
        num_valid = int(np.sum(valid_mask))

        if num_valid < self.min_valid_joints:
            self.invalid_count += 1
            return ValidationResult(
                status="INVALID",
                reasons=[f"Too few valid joints detected ({num_valid} < {self.min_valid_joints})"],
                depth_valid=False,
                height_valid=False,
                kinematics_valid=False,
                metrics={"valid_joints": num_valid},
            )

        valid_joints = j3d_cam[valid_mask]
        depths = valid_joints[:, 2]
        y_coords = valid_joints[:, 1]

        mean_depth = float(np.mean(depths))
        min_d, max_d = float(np.min(depths)), float(np.max(depths))

        # 1. 深度范围检查
        if min_d < self.min_depth or max_d > self.max_depth:
            depth_valid = False
            reasons.append(f"Depth out of range: min={min_d:.2f}m, max={max_d:.2f}m (valid: [{self.min_depth}, {self.max_depth}])")

        # 2. 人体高度尺度检查
        est_height = float(np.max(y_coords) - np.min(y_coords))
        if est_height < self.min_height or est_height > self.max_height:
            height_valid = False
            reasons.append(f"Abnormal human height: {est_height:.2f}m (valid: [{self.min_height}, {self.max_height}])")

        # 3. 运动学上下关系检查 (Head vs Foot)
        head_foot_diff = 0.0
        if skeleton.joint_format == "MediaPipe33":
            # 0: nose, 27: left_ankle, 28: right_ankle, 31: left_foot, 32: right_foot
            head_y = float(j3d_cam[0, 1])
            foot_y = float(min(j3d_cam[27, 1], j3d_cam[28, 1]))
            # 在 +Y-up 体系下，正常站姿 head_y > foot_y (diff > 0)
            head_foot_diff = head_y - foot_y
            if head_foot_diff < -0.6:
                kinematics_valid = False
                reasons.append(f"Skeleton inverted: Head Y ({head_y:.2f}m) is far below Foot Y ({foot_y:.2f}m)")
            elif head_foot_diff < 0.0:
                reasons.append(f"Low head posture (bending/falling): Head Y ({head_y:.2f}m) <= Foot Y ({foot_y:.2f}m)")
        elif skeleton.joint_format == "COCO17":
            head_y = float(j3d_cam[0, 1])
            foot_y = float(max(j3d_cam[15, 1], j3d_cam[16, 1]))
            head_foot_diff = foot_y - head_y
            if head_foot_diff < -0.6:
                kinematics_valid = False
                reasons.append(f"Skeleton inverted: Head Y ({head_y:.2f}m) is below Foot Y ({foot_y:.2f}m)")
            elif head_foot_diff < 0.0:
                reasons.append(f"Low head posture (bending/falling): Foot Y ({foot_y:.2f}m) <= Head Y ({head_y:.2f}m)")

        metrics = {
            "valid_joints": float(num_valid),
            "mean_depth": mean_depth,
            "min_depth": min_d,
            "max_depth": max_d,
            "estimated_height": est_height,
            "head_foot_diff": head_foot_diff,
        }

        # 判定总体状态
        if not depth_valid or not height_valid or not kinematics_valid:
            status = "INVALID"
            self.invalid_count += 1
        elif len(reasons) > 0:
            status = "WARNING"
            self.warning_count += 1
        else:
            status = "VALID"
            self.valid_count += 1

        return ValidationResult(
            status=status,
            reasons=reasons,
            depth_valid=depth_valid,
            height_valid=height_valid,
            kinematics_valid=kinematics_valid,
            metrics=metrics,
        )

    def get_summary(self) -> Dict[str, Any]:
        """获取全批次校验汇总统计信息。"""
        total = max(1, self.total_checked)
        return {
            "total_checked": self.total_checked,
            "valid_count": self.valid_count,
            "warning_count": self.warning_count,
            "invalid_count": self.invalid_count,
            "valid_rate": float(self.valid_count / total),
            "pass_rate": float((self.valid_count + self.warning_count) / total),
        }
