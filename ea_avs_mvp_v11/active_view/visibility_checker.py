"""
视线光线追踪通视性检查器 —— visibility_checker.py
=============================================

职责：
    1. 基于光线投射 (Ray Casting) 检查机器人相机视点到人体关键部位的视线通视性；
    2. 检测视线是否被实体墙体、立柱或高大家具阻挡；
    3. 支持 Habitat 仿真器物理 Raycast 与几何包围体/障碍物段检测；
    4. 输出是否完全遮挡与可见性评分。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint

logger = logging.getLogger("visibility_checker")


class VisibilityChecker:
    """机器人相机与人体目标视线通视性检查器。"""

    def __init__(
        self,
        sim: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        obstacles: Optional[List[Dict[str, Any]]] = None,
    ):
        self.sim = sim
        self.config = config or {}
        self.check_enabled = bool(self.config.get("check_visibility", True))
        self.hit_buffer_m = float(self.config.get("hit_buffer_m", 0.20))
        self.target_offsets = self.config.get("human_target_offsets", [
            [0.0, 0.80, 0.0],  # 骨盆/躯干
            [0.0, 1.45, 0.0],  # 头部/肩部
        ])
        # 自定义障碍物几何体列表 (用于无 Habitat 引擎运行环境下的精确几何检测)
        self.obstacles = obstacles or []

    def check_visibility(
        self,
        viewpoint: Viewpoint,
        human_position: Union[List[float], np.ndarray],
    ) -> Tuple[bool, float]:
        """
        检查指定视点对人体的通视性。

        参数:
            viewpoint: 待评估候选视点
            human_position: 人体基座坐标 [hx, hy, hz]

        返回:
            (is_visible: bool, visibility_score: float [0, 1])
        """
        if not self.check_enabled:
            return True, 1.0

        cam_pos = np.array(viewpoint.camera_position, dtype=np.float32)
        hx, hy, hz = [float(x) for x in human_position]

        visible_count = 0
        total_targets = len(self.target_offsets)

        for offset in self.target_offsets:
            target_pt = np.array([hx + offset[0], hy + offset[1], hz + offset[2]], dtype=np.float32)
            is_clear = self._raycast_clear(cam_pos, target_pt)
            if is_clear:
                visible_count += 1

        visibility_score = visible_count / max(total_targets, 1)
        is_visible = (visibility_score > 0.0) # 至少能看见一个关键部位即视为具备可见性

        return is_visible, round(visibility_score, 4)

    def _raycast_clear(self, from_pos: np.ndarray, to_pos: np.ndarray) -> bool:
        """底层光线追踪检测。"""
        # 1. 如果有 Habitat 仿真器物理引擎
        if self.sim is not None:
            try:
                import habitat_sim
                ray = habitat_sim.geo.Ray(from_pos, to_pos - from_pos)
                raycast_results = self.sim.cast_ray(ray)
                if raycast_results.has_hits():
                    dist_to_target = float(np.linalg.norm(to_pos - from_pos))
                    first_hit_dist = float(raycast_results.hits[0].ray_distance)
                    # 如果命中点在目标点之前 (扣除缓冲)，说明被障碍物阻挡
                    if first_hit_dist < dist_to_target - self.hit_buffer_m:
                        return False
            except Exception as e:
                logger.debug("Habitat raycast exception (falling back to geometric check): %s", e)

        # 2. 检查自定义几何障碍物 (AABB 包围盒求交检测)
        for obs in self.obstacles:
            if self._ray_intersects_aabb(from_pos, to_pos, obs):
                return False

        return True

    def _ray_intersects_aabb(self, p1: np.ndarray, p2: np.ndarray, aabb: Dict[str, Any]) -> bool:
        """线段与轴对齐包围盒 (AABB) 求交。"""
        min_pt = np.array(aabb.get("min", [-1, -1, -1]), dtype=np.float32)
        max_pt = np.array(aabb.get("max", [1, 1, 1]), dtype=np.float32)

        dir_vec = p2 - p1
        t_min = 0.0
        t_max = 1.0

        for i in range(3):
            if abs(dir_vec[i]) < 1e-6:
                if p1[i] < min_pt[i] or p1[i] > max_pt[i]:
                    return False
            else:
                t1 = (min_pt[i] - p1[i]) / dir_vec[i]
                t2 = (max_pt[i] - p1[i]) / dir_vec[i]
                t_near = min(t1, t2)
                t_far = max(t1, t2)

                t_min = max(t_min, t_near)
                t_max = min(t_max, t_far)

                if t_min > t_max:
                    return False

        return (t_min <= t_max) and (t_min < 1.0) and (t_max > 0.0)
