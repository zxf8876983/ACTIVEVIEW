"""
候选主动观察视点生成器 —— candidate_generator.py
=============================================

职责：
    1. 基于人体中心极坐标系 (Polar Coordinates) 采样生成机器人候选观察视点；
    2. 支持灵活配置水平方位角列表 (angles)、观察距离列表 (distances) 与相机高度；
    3. 自动计算机器人相机朝向人体的偏航角 (yaw: face human) 与俯仰角 (pitch)；
    4. 输出标准化的 Viewpoint 候选集合（默认 8 角度 x 4 距离 = 32 个候选点）。
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint

logger = logging.getLogger("candidate_generator")


def load_viewpoint_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """加载视点配置文件。"""
    if config_path is None:
        # 优先查找当前包配置，其次查找根目录配置
        local_path = Path(__file__).resolve().parent.parent / "configs" / "viewpoint_config.yaml"
        top_path = Path(__file__).resolve().parent.parent.parent / "configs" / "viewpoint_config.yaml"
        if local_path.exists():
            config_path = local_path
        elif top_path.exists():
            config_path = top_path

    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return {}


class CandidateViewGenerator:
    """人体中心极坐标候选视点生成器。"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        angles: Optional[List[float]] = None,
        distances: Optional[List[float]] = None,
        camera_height: Optional[float] = None,
        camera_pitch: Optional[float] = None,
    ):
        loaded_cfg = load_viewpoint_config(config_path)
        if config:
            loaded_cfg.update(config)

        sampling_cfg = loaded_cfg.get("sampling", {})

        # 方位角列表 (度)
        self.angles = [float(a) for a in (angles or sampling_cfg.get("angles_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]))]
        # 观察距离列表 (米)
        self.distances = [float(d) for d in (distances or sampling_cfg.get("distances_m", [1.5, 2.0, 2.5, 3.0]))]
        # 相机高度 (米)
        self.camera_height = float(camera_height if camera_height is not None else sampling_cfg.get("camera_height_m", 1.25))
        # 初始俯仰角 (度)
        self.camera_pitch = float(camera_pitch if camera_pitch is not None else sampling_cfg.get("camera_pitch_deg", 0.0))

    def generate(
        self,
        human_position: Union[List[float], np.ndarray],
        robot_current_position: Optional[Union[List[float], np.ndarray]] = None,
    ) -> List[Viewpoint]:
        """
        以人体为中心，在极坐标网格上生成三维候选视点列表。

        参数:
            human_position: 人体世界三维坐标 [hx, hy, hz]
            robot_current_position: 可选，机器人当前位置 [rx, ry, rz]

        返回:
            List[Viewpoint]: 生成的候选视点列表 (理论数量 = len(angles) * len(distances))
        """
        hx, hy, hz = [float(x) for x in human_position]
        candidates: List[Viewpoint] = []
        view_id_counter = 0

        for dist in self.distances:
            for ang_deg in self.angles:
                rad = math.radians(ang_deg)

                # 极坐标转换: x = x_h + r * cos(theta), z = z_h + r * sin(theta)
                vx = hx + dist * math.cos(rad)
                vz = hz + dist * math.sin(rad)
                vy = hy # 机器人底盘与人体处于同一基准地面高程

                # 自动计算朝向人体的偏航角 (Face Human Yaw), 统一归一化到 [0, 360) 度
                dx = hx - vx
                dz = hz - vz
                yaw_deg = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0

                # 计算初始欧氏导航代价 (若提供了机器人当前位置)
                nav_cost = 0.0
                if robot_current_position is not None:
                    rx, ry, rz = [float(p) for p in robot_current_position]
                    nav_cost = math.sqrt((vx - rx) ** 2 + (vz - rz) ** 2)

                vp = Viewpoint(
                    id=view_id_counter,
                    position=[round(vx, 4), round(vy, 4), round(vz, 4)],
                    rotation=[round(yaw_deg, 2), round(self.camera_pitch, 2)],
                    yaw=round(yaw_deg, 2),
                    pitch=round(self.camera_pitch, 2),
                    distance=round(dist, 4),
                    angle=round(ang_deg, 2),
                    camera_height=round(self.camera_height, 4),
                    navigation_cost=round(nav_cost, 4),
                    is_navigable=True,
                    is_reachable=True,
                    is_visible=True,
                    metadata={
                        "view_id": f"vp_r{dist:.1f}_a{int(ang_deg):03d}",
                        "human_position": [round(hx, 4), round(hy, 4), round(hz, 4)],
                        "radius": dist,
                        "angle_deg": ang_deg,
                    },
                )
                candidates.append(vp)
                view_id_counter += 1

        logger.info("Candidate Generation: Generated %d candidate viewpoints (Angles: %s, Distances: %s)",
                    len(candidates), self.angles, self.distances)
        return candidates
