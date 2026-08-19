"""
人体朝向（Yaw）估计模块 —— orientation_estimator.py
===================================================

功能：
    基于观测到的双侧解剖对称关节对（左右肩、左右髋）的世界 3D 几何，稳健估计人体的水平朝向角（Yaw）。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .depth_lifter import EstimatedJoint3D
from .geometry import normalize_angle


@dataclass
class EstimatedOrientation:
    """人体朝向估计结果。"""
    valid: bool
    yaw_rad: Optional[float]
    yaw_deg: Optional[float]
    confidence: float
    source: str  # "shoulders_and_hips" | "shoulders_only" | "hips_only" | "invalid"
    forward_vector: Optional[np.ndarray] = None


class OrientationEstimator:
    """人体朝向（Yaw）估计器。"""

    def __init__(self, config: dict):
        self.config = config.get("human_state_estimation", {}).get("orientation", {})
        self.yaw_offset_deg = float(self.config.get("yaw_offset_deg", 0.0))
        self.forward_sign = float(self.config.get("forward_sign", 1.0))
        self.min_pair_conf = float(self.config.get("min_bilateral_pair_confidence", 0.30))

    def estimate_orientation(
        self,
        joints_3d: Dict[str, EstimatedJoint3D],
    ) -> EstimatedOrientation:
        """从 3D 关节字典估计人体朝向 Yaw。

        参数：
            joints_3d: 3D 关节字典。

        返回：
            EstimatedOrientation 对象。
        """
        # 1. 提取左右肩
        l_sh = joints_3d.get("left_shoulder")
        r_sh = joints_3d.get("right_shoulder")
        sh_valid = (
            l_sh is not None and r_sh is not None
            and l_sh.observable_3d and r_sh.observable_3d
            and l_sh.position_world is not None and r_sh.position_world is not None
            and min(l_sh.confidence_2d, r_sh.confidence_2d) >= self.min_pair_conf
        )

        # 2. 提取左右髋
        l_hip = joints_3d.get("left_hip")
        r_hip = joints_3d.get("right_hip")
        hip_valid = (
            l_hip is not None and r_hip is not None
            and l_hip.observable_3d and r_hip.observable_3d
            and l_hip.position_world is not None and r_hip.position_world is not None
            and min(l_hip.confidence_2d, r_hip.confidence_2d) >= self.min_pair_conf
        )

        v_lat_sh = None
        if sh_valid:
            d_sh = l_sh.position_world - r_sh.position_world  # right -> left
            xz_sh = np.array([d_sh[0], d_sh[2]], dtype=np.float64)
            norm = np.linalg.norm(xz_sh)
            if norm > 1e-4:
                v_lat_sh = xz_sh / norm

        v_lat_hip = None
        if hip_valid:
            d_hip = l_hip.position_world - r_hip.position_world  # right -> left
            xz_hip = np.array([d_hip[0], d_hip[2]], dtype=np.float64)
            norm = np.linalg.norm(xz_hip)
            if norm > 1e-4:
                v_lat_hip = xz_hip / norm

        if v_lat_sh is None and v_lat_hip is None:
            return EstimatedOrientation(
                valid=False,
                yaw_rad=None,
                yaw_deg=None,
                confidence=0.0,
                source="invalid",
                forward_vector=None,
            )

        if v_lat_sh is not None and v_lat_hip is not None:
            # 检查两组横向向量的一致性（点积）
            agreement = float(np.dot(v_lat_sh, v_lat_hip))
            if agreement < 0.2:
                # 显著冲突时以置信度更高的一组为主或降低置信度
                sh_c = (l_sh.confidence_2d + r_sh.confidence_2d) / 2.0
                hip_c = (l_hip.confidence_2d + r_hip.confidence_2d) / 2.0
                v_lat = v_lat_sh if sh_c >= hip_c else v_lat_hip
                conf = max(sh_c, hip_c) * 0.5
                src = "shoulders_only" if sh_c >= hip_c else "hips_only"
            else:
                v_lat = (v_lat_sh + v_lat_hip) / 2.0
                v_lat = v_lat / np.linalg.norm(v_lat)
                sh_c = (l_sh.confidence_2d + r_sh.confidence_2d) / 2.0
                hip_c = (l_hip.confidence_2d + r_hip.confidence_2d) / 2.0
                conf = float((sh_c + hip_c) / 2.0 * max(0.5, agreement))
                src = "shoulders_and_hips"
        elif v_lat_sh is not None:
            v_lat = v_lat_sh
            conf = float((l_sh.confidence_2d + r_sh.confidence_2d) / 2.0 * 0.8)
            src = "shoulders_only"
        else:
            v_lat = v_lat_hip
            conf = float((l_hip.confidence_2d + r_hip.confidence_2d) / 2.0 * 0.8)
            src = "hips_only"

        # 从横向向量（右指向左）推导前向向量：
        # 从 2D 视觉提升得到的 3D 坐标中，人面向 +Z (yaw=0) 时，其解剖左肩在 +X，解剖右肩在 -X，v_lat = (+1, 0)
        # 对应的人体正面朝向为 +Z (0, 1)，因此：
        # raw_fwd_x = -v_lat.z
        # raw_fwd_z = v_lat.x
        raw_fwd_x = -v_lat[1]
        raw_fwd_z = v_lat[0]

        # 结合配置的符号修正与一次性校准 offset
        raw_yaw = float(np.arctan2(raw_fwd_x, raw_fwd_z))
        calibrated_yaw = normalize_angle(raw_yaw * self.forward_sign + np.deg2rad(self.yaw_offset_deg))

        forward_vec = np.array([np.sin(calibrated_yaw), 0.0, np.cos(calibrated_yaw)], dtype=np.float64)

        return EstimatedOrientation(
            valid=True,
            yaw_rad=calibrated_yaw,
            yaw_deg=float(np.rad2deg(calibrated_yaw)),
            confidence=conf,
            source=src,
            forward_vector=forward_vec,
        )
