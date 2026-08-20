"""
视点空间约束与导航可行性检查器 —— constraint_checker.py
======================================================

职责：
    1. 校验机器人候选观察视点在室内场景中的物理与导航约束；
    2. 基于 Habitat Pathfinder 检查候选点是否处于 NavMesh 可行走区域 (is_navigable)；
    3. 检查从机器人初始位置到候选视点在 NavMesh 上的连通性与可达性 (reachability)；
    4. 过滤剔除穿墙、落入家具遮挡体内部或物理不可达的无效视点。
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """机器人候选视点物理与导航约束检查器。"""

    def __init__(
        self,
        env_adapter: Optional[V8EnvironmentAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.env_adapter = env_adapter
        self.config = config or {}
        self.check_navmesh = bool(self.config.get("check_navmesh", True))
        self.check_reachability = bool(self.config.get("check_path_reachability", True))
        self.max_snap_dist = float(self.config.get("max_snap_distance", 0.5))

    def check_viewpoint(
        self,
        viewpoint: CandidateViewpoint,
        robot_start_pos: Optional[Union[List[float], np.ndarray]] = None,
    ) -> CandidateViewpoint:
        """检查单个候选视点是否满足空间与导航约束。"""
        pos = viewpoint.position

        # 1. 检查是否在 NavMesh 可行走域内
        if self.check_navmesh and self.env_adapter is not None and self.env_adapter.sim is not None:
            if not self.env_adapter.is_navigable(pos, max_snap_dist=self.max_snap_dist):
                viewpoint.feasible = False
                viewpoint.reason = "not_navigable"
                return viewpoint

        # 2. 检查从起点到候选点的路径连通性
        if self.check_reachability and robot_start_pos is not None and self.env_adapter is not None and self.env_adapter.sim is not None:
            reachable, geo_dist = self.env_adapter.check_path_exists(robot_start_pos, pos)
            if not reachable:
                viewpoint.feasible = False
                viewpoint.reason = "unreachable"
                return viewpoint
            viewpoint.metadata["geodesic_distance"] = geo_dist

        viewpoint.feasible = True
        viewpoint.reason = "valid"
        return viewpoint

    def filter_feasible_viewpoints(
        self,
        viewpoints: List[CandidateViewpoint],
        robot_start_pos: Optional[Union[List[float], np.ndarray]] = None,
    ) -> List[CandidateViewpoint]:
        """批量检查并筛选出所有合法的候选视点。"""
        checked_list: List[CandidateViewpoint] = []
        feasible_count = 0

        for vp in viewpoints:
            checked_vp = self.check_viewpoint(vp, robot_start_pos=robot_start_pos)
            checked_list.append(checked_vp)
            if checked_vp.feasible:
                feasible_count += 1

        logger.info(
            "Constraint checking finished: %d / %d viewpoints are feasible",
            feasible_count,
            len(viewpoints),
        )
        return checked_list
