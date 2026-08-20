"""
综合视点空间约束管道 —— constraint_checker.py
============================================

职责：
    1. 串联三大约束检查器：
       - NavigationConstraint (NavMesh 可行域与可达性检查)
       - LineOfSightConstraint (光线追踪视线通视性与遮挡检查)
       - HumanVisibilityConstraint (人体视锥 FOV 与距离范围检查)
    2. 对候选视点执行全量物理与几何过滤；
    3. 输出带有明确有效性标记 (feasible) 与拒绝原因 (reason) 的 CandidateViewpoint 列表。
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.environment.env_adapter import V8EnvironmentAdapter
from .navigation_constraint import NavigationConstraint
from .line_of_sight_constraint import LineOfSightConstraint
from .human_visibility_constraint import HumanVisibilityConstraint

logger = logging.getLogger(__name__)


class ConstraintChecker:
    """机器人候选视点综合约束检查器与过滤器。"""

    def __init__(
        self,
        env_adapter: Optional[V8EnvironmentAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.env_adapter = env_adapter
        self.config = config or {}
        self.nav_constraint = NavigationConstraint(env_adapter, self.config)
        self.los_constraint = LineOfSightConstraint(env_adapter, self.config)
        self.vis_constraint = HumanVisibilityConstraint(self.config)

    def check_viewpoint(
        self,
        viewpoint: CandidateViewpoint,
        human_position: Union[List[float], np.ndarray],
        robot_start_pos: Optional[Union[List[float], np.ndarray]] = None,
    ) -> CandidateViewpoint:
        """对单个候选视点依序执行三大约束检查。"""
        # 1. 导航与可达性约束
        is_nav, reason_nav = self.nav_constraint.evaluate(viewpoint, robot_start_pos=robot_start_pos)
        if not is_nav:
            viewpoint.feasible = False
            viewpoint.reason = reason_nav
            return viewpoint

        # 2. 视线光线追踪通视性约束
        is_los, reason_los = self.los_constraint.evaluate(viewpoint, human_position=human_position)
        if not is_los:
            viewpoint.feasible = False
            viewpoint.reason = reason_los
            return viewpoint

        # 3. 人体进入相机 FOV 视锥约束
        is_fov, reason_fov = self.vis_constraint.evaluate(viewpoint, human_position=human_position)
        if not is_fov:
            viewpoint.feasible = False
            viewpoint.reason = reason_fov
            return viewpoint

        viewpoint.feasible = True
        viewpoint.reason = "valid"
        return viewpoint

    def filter_feasible_viewpoints(
        self,
        viewpoints: List[CandidateViewpoint],
        human_position: Optional[Union[List[float], np.ndarray]] = None,
        robot_start_pos: Optional[Union[List[float], np.ndarray]] = None,
    ) -> List[CandidateViewpoint]:
        """批量检查并筛选出所有合法的候选视点。"""
        h_pos = human_position if human_position is not None else viewpoints[0].metadata.get("human_position", [1.5, -1.6, 4.0])

        checked_list: List[CandidateViewpoint] = []
        feasible_count = 0

        for vp in viewpoints:
            checked_vp = self.check_viewpoint(vp, human_position=h_pos, robot_start_pos=robot_start_pos)
            checked_list.append(checked_vp)
            if checked_vp.feasible:
                feasible_count += 1

        logger.info(
            "Constraint pipeline finished: %d / %d viewpoints passed all constraints",
            feasible_count,
            len(viewpoints),
        )
        return checked_list
