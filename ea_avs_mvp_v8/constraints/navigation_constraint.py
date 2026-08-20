"""
导航与可达性空间约束 —— navigation_constraint.py
==============================================

职责：
    1. 校验机器人候选观察视点在 Habitat NavMesh 可行走区域内的有效性 (is_navigable)；
    2. 校验从机器人初始位置到候选视点在 NavMesh 上的连通性与可达性 (reachability)；
    3. 过滤穿墙、落入不可通行区域或隔断孤岛上的候选视点。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter

logger = logging.getLogger(__name__)


class NavigationConstraint:
    """机器人候选视点 NavMesh 可行性与可达性检查器。"""

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

    def evaluate(
        self,
        viewpoint: CandidateViewpoint,
        robot_start_pos: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Tuple[bool, str]:
        """检查单个候选视点是否满足导航与可达性约束。

        返回：
            (is_feasible: bool, reason: str)
        """
        if not self.check_navmesh or self.env_adapter is None or self.env_adapter.sim is None:
            return True, "valid"

        pos = viewpoint.position

        # 1. 检查是否在 NavMesh 可行域内
        if not self.env_adapter.is_navigable(pos, max_snap_dist=self.max_snap_dist):
            return False, "not_navigable"

        # 2. 检查从起点到候选点的路径连通性与可达性
        if self.check_reachability and robot_start_pos is not None:
            reachable, geo_dist = self.env_adapter.check_path_exists(robot_start_pos, pos)
            if not reachable:
                return False, "unreachable"
            viewpoint.metadata["geodesic_distance"] = geo_dist

        return True, "valid"
