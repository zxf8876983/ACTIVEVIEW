"""
综合视点空间约束管道 —— constraint_checker.py
============================================

职责：
    1. 串联三大约束检查器：
       - NavigationConstraint (NavMesh 可行域与可达性检查)
       - LineOfSightConstraint (光线追踪视线通视性与遮挡检查)
       - HumanVisibilityConstraint (人体视锥 FOV、投影面积与距离范围检查)
    2. 对候选视点执行全量物理与几何过滤；
    3. 收集逐级过滤统计数据 (candidate_statistics)，用于科研消融与实验统计；
    4. 输出带有明确有效性标记 (feasible)、拒绝原因 (reason) 与 evaluation_mode 的 CandidateViewpoint 列表。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
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
        self.evaluation_mode = str(self.config.get("evaluation_mode", "oracle"))
        self.nav_constraint = NavigationConstraint(env_adapter, self.config)
        self.los_constraint = LineOfSightConstraint(env_adapter, self.config)
        self.vis_constraint = HumanVisibilityConstraint(self.config)

    def check_viewpoint(
        self,
        viewpoint: CandidateViewpoint,
        human_position: Union[List[float], np.ndarray],
        human_joints_3d: Optional[Dict[str, List[float]]] = None,
        robot_start_pos: Optional[Union[List[float], np.ndarray]] = None,
    ) -> CandidateViewpoint:
        """对单个候选视点依序执行三大约束检查。"""
        viewpoint.evaluation_mode = self.evaluation_mode

        # 1. 导航与可达性约束
        is_nav, reason_nav = self.nav_constraint.evaluate(viewpoint, robot_start_pos=robot_start_pos)
        viewpoint.metadata["navmesh_valid"] = bool(is_nav)
        if not is_nav:
            viewpoint.feasible = False
            viewpoint.reason = reason_nav
            viewpoint.metadata["line_of_sight_valid"] = False
            viewpoint.metadata["human_visible"] = False
            return viewpoint

        # 2. 视线光线追踪通视性约束
        is_los, reason_los = self.los_constraint.evaluate(viewpoint, human_position=human_position)
        viewpoint.metadata["line_of_sight_valid"] = bool(is_los)
        if not is_los:
            viewpoint.feasible = False
            viewpoint.reason = reason_los
            viewpoint.metadata["human_visible"] = False
            return viewpoint

        # 3. 人体进入相机 FOV 视锥与投影面积约束
        is_fov, reason_fov = self.vis_constraint.evaluate(
            viewpoint,
            human_position=human_position,
            human_joints_3d=human_joints_3d,
        )
        viewpoint.metadata["human_visible"] = bool(is_fov)
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
        human_joints_3d: Optional[Dict[str, List[float]]] = None,
        robot_start_pos: Optional[Union[List[float], np.ndarray]] = None,
    ) -> List[CandidateViewpoint]:
        """批量检查并筛选出所有合法的候选视点。"""
        h_pos = human_position if human_position is not None else viewpoints[0].metadata.get("human_position", [1.5, -1.6, 4.0])

        checked_list: List[CandidateViewpoint] = []
        feasible_count = 0

        for vp in viewpoints:
            checked_vp = self.check_viewpoint(
                vp,
                human_position=h_pos,
                human_joints_3d=human_joints_3d,
                robot_start_pos=robot_start_pos,
            )
            checked_list.append(checked_vp)
            if checked_vp.feasible:
                feasible_count += 1

        logger.info(
            "Constraint pipeline finished: %d / %d viewpoints passed all constraints",
            feasible_count,
            len(viewpoints),
        )
        return checked_list

    @staticmethod
    def compute_statistics(
        viewpoints: List[CandidateViewpoint],
        selected_view_id: Optional[str] = None,
        strategy: str = "geometry_best",
        evaluation_mode: str = "oracle",
    ) -> Dict[str, Any]:
        """计算标准化候选视点过滤统计指标。"""
        total = len(viewpoints)
        nav_valid = sum(1 for v in viewpoints if v.metadata.get("navmesh_valid", False))
        los_valid = sum(1 for v in viewpoints if v.metadata.get("line_of_sight_valid", False))
        vis_valid = sum(1 for v in viewpoints if v.metadata.get("human_visible", False))
        feasible_count = sum(1 for v in viewpoints if v.feasible)

        return {
            "total_candidates": total,
            "navmesh_valid": nav_valid,
            "line_of_sight_valid": los_valid,
            "human_visible": vis_valid,
            "feasible_candidates": feasible_count,
            "filtered_out_candidates": total - feasible_count,
            "feasibility_rate": round(feasible_count / max(1, total), 3),
            "selected_view": selected_view_id,
            "strategy": strategy,
            "evaluation_mode": evaluation_mode,
        }
