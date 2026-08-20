"""
人体视锥可见性约束 —— human_visibility_constraint.py
===================================================

职责：
    1. 校验人体目标是否处于候选视点相机的视场角 (FOV) 与最大有效观察距离内；
    2. 计算相机主光轴与人体中心的水平与俯仰偏角；
    3. 过滤背对人体、偏角过大或超出传感器有效距离的无价值视点。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint

logger = logging.getLogger(__name__)


class HumanVisibilityConstraint:
    """人体是否进入相机视锥 (FOV) 几何可见性检查器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hfov_deg = float(self.config.get("hfov_deg", self.config.get("hfov", 90.0)))
        self.max_distance = float(self.config.get("max_distance", 4.5))
        self.min_distance = float(self.config.get("min_distance", 0.8))

    def evaluate(
        self,
        viewpoint: CandidateViewpoint,
        human_position: Union[List[float], np.ndarray],
    ) -> Tuple[bool, str]:
        """检查人体中心是否在候选视点的相机 FOV 视锥内。

        返回：
            (is_visible: bool, reason: str)
        """
        hx, hy, hz = [float(x) for x in human_position]
        rx, ry, rz = [float(x) for x in viewpoint.position]

        cam_pos = np.array([rx, ry + float(viewpoint.camera_height), rz], dtype=np.float32)
        target_pos = np.array([hx, hy + 0.80, hz], dtype=np.float32)

        vec_to_human = target_pos - cam_pos
        dist = float(np.linalg.norm(vec_to_human))

        # 1. 距离检查
        if dist > self.max_distance:
            return False, "too_far"
        if dist < self.min_distance:
            return False, "too_close"

        # 2. 相机朝向与人体夹角检查 (水平面 FOV)
        vec_horiz = np.array([vec_to_human[0], 0.0, vec_to_human[2]], dtype=np.float32)
        dist_h = float(np.linalg.norm(vec_horiz))
        if dist_h < 1e-4:
            return True, "valid"

        dir_to_human = vec_horiz / dist_h

        # 相机前向向量 (Habitat 坐标系: 0 deg 对应 -Z)
        yaw_rad = math.radians(float(viewpoint.yaw_deg))
        cam_forward = np.array([math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype=np.float32)

        dot = np.clip(np.dot(dir_to_human, cam_forward), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(dot))

        half_fov = self.hfov_deg / 2.0
        if angle_deg > half_fov:
            return False, "human_out_of_fov"

        return True, "valid"
