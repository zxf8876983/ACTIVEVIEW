"""
多视角候选视点生成器 —— candidate_generator.py
============================================

职责：
    1. 围绕人体中心以极坐标网格采样生成多距离、多方位的候选观察视角；
    2. 计算视点世界空间坐标与相机对准人体的注视偏航角；
    3. 输出 CandidateViewpointV10 列表；
    4. 严格禁止在此处实现任何视点选择策略或评分策略 (Phase 1 仅负责多视点数据采样与观测生成)。
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

        cam_y = self.ground_height + self.camera_height
        candidates: List[CandidateViewpointV10] = []

        angle_step = 360.0 / float(self.num_angles)
        vp_idx = 0

        for r in self.radii:
            for i in range(self.num_angles):
                rel_angle_deg = i * angle_step
                # 计算世界绝对极角 (由人体朝向叠加相对角)
                world_angle_deg = (human_yaw_deg + rel_angle_deg) % 360.0
                world_angle_rad = math.radians(world_angle_deg)

                # 相机位置 (以人体为中心环绕)
                cx = hx + r * math.sin(world_angle_rad)
                cz = hz + r * math.cos(world_angle_rad)

                # 相机朝向人体注视角 (从相机看向人体)
                dx = hx - cx
                dz = hz - cz
                gaze_yaw_rad = math.atan2(dx, dz)
                gaze_yaw_deg = math.degrees(gaze_yaw_rad) % 360.0

                vp_id = f"vp_{vp_idx:03d}_r{r:.1f}_a{int(rel_angle_deg):03d}"
                vp_idx += 1

                candidates.append(
                    CandidateViewpointV10(
                        viewpoint_id=vp_id,
                        radius=r,
                        angle_deg=rel_angle_deg,
                        height=self.camera_height,
                        position=[cx, cam_y, cz],
                        yaw_deg=gaze_yaw_deg,
                        feasible=True,
                    )
                )

        return candidates
