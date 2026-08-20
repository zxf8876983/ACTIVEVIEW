"""
候选观察视角生成器 —— viewpoint_generator.py
============================================

职责：
    1. 围绕人体中心坐标以极坐标规则网格 (Polar Grid) 生成多距离、多水平方位角的候选视点集合；
    2. 计算每个候选视点机器人底盘的空间三维坐标 [x, ground_height, z]；
    3. 自动计算使相机光轴正对人体中心的机器人偏航朝向角 yaw_deg；
    4. 独立解耦机器人地面高度 (ground_height) 与相机安装高度 (camera_height)；
    5. 输出标准化 CandidateViewpoint 列表。
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
        self.ground_height = float(self.config.get("ground_height", self.config.get("floor_height", -1.60)))
        self.face_human = bool(self.config.get("face_human", True))

    def generate_candidates(
        self,
        human_position: Union[List[float], np.ndarray],
        human_yaw_deg: float = 0.0,
        ground_height: Optional[float] = None,
        custom_radii: Optional[List[float]] = None,
        custom_num_angles: Optional[int] = None,
    ) -> List[CandidateViewpoint]:
        """围绕人体位置生成候选视点集合。

        参数：
            human_position: 人体基座/中心三维坐标 [hx, hy, hz]
            human_yaw_deg: 人体朝向角 (度)
            ground_height: 机器人底盘所在地面高程 (若为空则从配置读取)
            custom_radii: 自定义采样半径列表 (可选)
            custom_num_angles: 自定义角度采样数量 (可选)

        返回：
            List[CandidateViewpoint]: 生成的候选视点列表 (包含底盘坐标、偏航角与相机空间位姿)
        """
        hx, hy, hz = [float(x) for x in human_position]
        radii = custom_radii or self.radii
        num_angles = custom_num_angles or self.num_angles
        g_height = ground_height if ground_height is not None else self.ground_height

        candidates: List[CandidateViewpoint] = []
        angle_step = 360.0 / float(num_angles)

        view_idx = 0
        for r in radii:
            for a_idx in range(num_angles):
                # 计算水平方位角 (0度为人体正后方/Z正向，相对角度 0-360)
                theta_deg = float(a_idx * angle_step)
                theta_rad = math.radians(theta_deg)

                # 机器人底盘在水平面 (X-Z) 的坐标计算，底盘高度独立由 g_height 决定
                rx = hx + r * math.sin(theta_rad)
                ry = float(g_height)
                rz = hz + r * math.cos(theta_rad)

                # 计算机器人正对人体的偏航朝向角
                # Habitat 约定: 0 deg 面向 -Z, 90 deg 面向 +X, 180 deg 面向 +Z, 270 deg 面向 -X
                dx = hx - rx
                dz = hz - rz

                if self.face_human:
                    yaw_rad = math.atan2(dx, -dz)
                    yaw_deg = (math.degrees(yaw_rad) + 360.0) % 360.0
                else:
                    yaw_deg = 0.0

                # 计算相机空间位姿 (用于存入 candidate_views.json)
                cam_pos = [rx, ry + self.camera_height, rz]
                half_yaw = math.radians(yaw_deg) / 2.0
                cam_rot = [0.0, float(math.sin(half_yaw)), 0.0, float(math.cos(half_yaw))]

                v_id = f"vp_{view_idx:03d}_r{r:.1f}_a{int(theta_deg):03d}"
                vp = CandidateViewpoint(
                    viewpoint_id=v_id,
                    position=[rx, ry, rz],
                    yaw_deg=yaw_deg,
                    radius=float(r),
                    angle_deg=theta_deg,
                    camera_height=self.camera_height,
                    ground_height=ry,
                    camera_pose={
                        "position": cam_pos,
                        "rotation": cam_rot,
                    },
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
            "Generated %d candidate viewpoints (radii=%s, angles=%d, ground_height=%.2f, cam_height=%.2f)",
            len(candidates),
            radii,
            num_angles,
            g_height,
            self.camera_height,
        )
        return candidates
