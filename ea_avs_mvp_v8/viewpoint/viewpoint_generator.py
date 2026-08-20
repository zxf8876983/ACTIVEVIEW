"""
候选观察视角生成器 —— viewpoint_generator.py
============================================

职责：
    1. 围绕人体中心坐标以极坐标规则网格 (Polar Grid) 生成多距离、多水平方位角的候选视点集合；
    2. 计算每个候选视点机器人的空间三维坐标 [x, y, z]；
    3. 自动计算使相机光轴正对人体中心的机器人偏航朝向角 yaw_deg；
    4. 输出标准化 CandidateViewpoint 列表。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint

logger = logging.getLogger(__name__)


class ViewpointGenerator:
    """机器人候选观察视点生成器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # 采样参数
        self.radii = list(self.config.get("radii", [1.5, 2.2, 3.0]))
        self.num_angles = int(self.config.get("num_angles", 8))
        self.camera_height = float(self.config.get("camera_height", 1.2))
        self.face_human = bool(self.config.get("face_human", True))

    def generate_candidates(
        self,
        human_position: Union[List[float], np.ndarray],
        human_yaw_deg: float = 0.0,
        custom_radii: Optional[List[float]] = None,
        custom_num_angles: Optional[int] = None,
    ) -> List[CandidateViewpoint]:
        """围绕人体位置生成候选视点集合。

        参数：
            human_position: 人体基座/中心三维坐标 [hx, hy, hz]
            human_yaw_deg: 人体朝向角 (度)
            custom_radii: 自定义采样半径列表 (可选)
            custom_num_angles: 自定义角度采样数量 (可选)

        返回：
            List[CandidateViewpoint]: 生成的候选视点列表 (包含坐标与朝向)
        """
        hx, hy, hz = [float(x) for x in human_position]
        radii = custom_radii or self.radii
        num_angles = custom_num_angles or self.num_angles

        candidates: List[CandidateViewpoint] = []
        angle_step = 360.0 / float(num_angles)

        view_idx = 0
        for r in radii:
            for a_idx in range(num_angles):
                # 计算水平方位角 (0度为人体后方/Z正向，或相对于人体的相对方位)
                theta_deg = float(a_idx * angle_step)
                theta_rad = math.radians(theta_deg)

                # 机器人底盘在水平面 (X-Z) 的坐标计算
                rx = hx + r * math.sin(theta_rad)
                ry = hy  # 底盘与人体处于同一物理地面基准
                rz = hz + r * math.cos(theta_rad)

                # 计算机器人正对人体的偏航朝向角
                # Habitat 约定: 0 deg 面向 -Z, 90 deg 面向 +X, 180 deg 面向 +Z, 270 deg 面向 -X
                dx = hx - rx
                dz = hz - rz

                if self.face_human:
                    # 偏航角使得机器人看向 (hx, hz)
                    # yaw = atan2(dx, -dz) 转化为度数
                    yaw_rad = math.atan2(dx, -dz)
                    yaw_deg = (math.degrees(yaw_rad) + 360.0) % 360.0
                else:
                    yaw_deg = 0.0

                v_id = f"vp_{view_idx:03d}_r{r:.1f}_a{int(theta_deg):03d}"
                vp = CandidateViewpoint(
                    viewpoint_id=v_id,
                    position=[rx, ry, rz],
                    yaw_deg=yaw_deg,
                    radius=float(r),
                    angle_deg=theta_deg,
                    camera_height=self.camera_height,
                    feasible=True,
                    reason="unverified",
                    metadata={
                        "human_position": [hx, hy, hz],
                        "index": view_idx,
                    },
                )
                candidates.append(vp)
                view_idx += 1

        logger.info(
            "Generated %d candidate viewpoints across %d radii and %d angles around human at %s",
            len(candidates),
            len(radii),
            num_angles,
            [hx, hy, hz],
        )
        return candidates
