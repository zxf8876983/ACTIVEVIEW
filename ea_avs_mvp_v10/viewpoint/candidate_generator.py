"""
多视角候选视点生成器 —— candidate_generator.py
============================================

职责：
    1. 围绕人体中心以极坐标网格采样生成多距离、多方位的候选观察视角；
    2. 计算机器人底盘世界空间坐标 [x, ground_height, z] 与相机正对人体的偏航角；
    3. 输出 CandidateViewpointV10 列表；
    4. 严格遵守 Habitat 坐标约定 (Agent 前向为 -Z，偏航角为 atan2(dx, -dz))。
"""

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ea_avs_mvp_v10.core.types import CandidateViewpointV10


class CandidateViewpointGeneratorV10:
    """v10.0 多视角候选视点生成器。"""

    def __init__(
        self,
        radii: Optional[List[float]] = None,
        num_angles: int = 8,
        camera_height: float = 1.20,
        ground_height: float = -1.60,
    ):
        self.radii = radii or [1.5, 2.0, 2.5]
        self.num_angles = num_angles
        self.camera_height = camera_height
        self.ground_height = ground_height

    def generate_candidates(
        self,
        human_position: Union[List[float], np.ndarray],
        human_yaw_deg: float = 0.0,
    ) -> List[CandidateViewpointV10]:
        """
        围绕人体当前位置生成多视角候选视点。

        Args:
            human_position: 人体中心 [x, y, z] 世界坐标
            human_yaw_deg: 人体朝向角 (度)

        Returns:
            candidates: List[CandidateViewpointV10]
        """
        hx = float(human_position[0])
        hy = float(human_position[1])
        hz = float(human_position[2])

        candidates: List[CandidateViewpointV10] = []

        angle_step = 360.0 / float(self.num_angles)
        vp_idx = 0

        for r in self.radii:
            for i in range(self.num_angles):
                rel_angle_deg = i * angle_step
                # 计算世界绝对极角 (由人体朝向叠加相对角)
                world_angle_deg = (human_yaw_deg + rel_angle_deg) % 360.0
                world_angle_rad = math.radians(world_angle_deg)

                # 机器人底盘位置 (以人体为中心在 X-Z 平面环绕，Y 坐标为地面高度)
                rx = hx + r * math.sin(world_angle_rad)
                ry = float(self.ground_height)
                rz = hz + r * math.cos(world_angle_rad)

                # 计算机器人面向人体的注视偏航角
                # Habitat 坐标系中，Agent 默认朝向为 -Z，偏角为 atan2(dx, -dz)
                dx = hx - rx
                dz = hz - rz
                gaze_yaw_rad = math.atan2(dx, -dz)
                gaze_yaw_deg = (math.degrees(gaze_yaw_rad) + 360.0) % 360.0

                vp_id = f"vp_{vp_idx:03d}_r{r:.1f}_a{int(rel_angle_deg):03d}"
                vp_idx += 1

                candidates.append(
                    CandidateViewpointV10(
                        viewpoint_id=vp_id,
                        radius=float(r),
                        angle_deg=float(rel_angle_deg),
                        height=float(self.camera_height),
                        position=[rx, ry, rz],
                        yaw_deg=float(gaze_yaw_deg),
                        feasible=True,
                    )
                )

        return candidates
