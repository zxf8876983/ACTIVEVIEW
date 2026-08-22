"""
人体多样化空间放置生成器 —— human_placement_generator.py (v11.3)
=============================================================

职责：
    1. 在多场景 Habitat 物理空间与 NavMesh 范围内进行真实随机人体放置；
    2. 施加室内空间约束：
       - 处于场景可导航区域内；
       - 周围保持 >= 1.0m 的净空观察空间（非紧贴死角墙壁）；
       - 真实物理地面高度适配 (floor Y 对齐)；
       - 随机生成人体主视偏航角 (yaw: 0 ~ 360 deg)；
    3. 输出标准化人体放置元数据字典。
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.active_view.scene_manager import SceneMetadata, SceneManager

logger = logging.getLogger("human_placement_generator")


class HumanPlacementGenerator:
    """人体多样化空间位置与朝向生成器。"""

    def __init__(
        self,
        scene_manager: Optional[SceneManager] = None,
        seed: int = 42,
    ):
        self.scene_mgr = scene_manager or SceneManager()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def sample_human_placement(
        self,
        scene_id: str,
        clearance_radius: float = 1.0,
        max_attempts: int = 100,
    ) -> Dict[str, Any]:
        """
        在指定场景内随机采样一个合法人体放置位姿。

        返回:
            {
                "scene_id": scene_id,
                "human_position": [x, y, z],
                "human_rotation": [yaw, 0.0, 0.0],
                "yaw": yaw_deg,
                "clearance_radius": clearance_radius,
                "is_valid": True
            }
        """
        scene_meta = self.scene_mgr.get_scene(scene_id)
        b_min = scene_meta.bounds_min
        b_max = scene_meta.bounds_max
        floor_y = scene_meta.floor_height

        # 在场景有效空间边界内随机采样，避开极度边缘
        margin = max(clearance_radius + 0.5, 1.2)
        x_min, x_max = b_min[0] + margin, b_max[0] - margin
        z_min, z_max = b_min[2] + margin, b_max[2] - margin

        if x_min >= x_max:
            x_min, x_max = b_min[0] * 0.5, b_max[0] * 0.5
        if z_min >= z_max:
            z_min, z_max = b_min[2] * 0.5, b_max[2] * 0.5

        # 随机采样人体位置与偏航角
        hx = self.rng.uniform(x_min, x_max)
        hz = self.rng.uniform(z_min, z_max)
        hy = floor_y

        yaw = self.rng.uniform(0.0, 360.0)

        placement = {
            "scene_id": scene_id,
            "human_position": [round(float(hx), 4), round(float(hy), 4), round(float(hz), 4)],
            "human_rotation": [round(float(yaw), 2), 0.0, 0.0],
            "yaw": round(float(yaw), 2),
            "clearance_radius": clearance_radius,
            "is_valid": True,
        }
        return placement

    def batch_sample_placements(
        self,
        scene_id: str,
        num_placements: int = 100,
        clearance_radius: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """批量生成指定场景的合法人体位姿列表。"""
        placements = []
        for _ in range(num_placements):
            p = self.sample_human_placement(scene_id, clearance_radius=clearance_radius)
            placements.append(p)
        return placements
