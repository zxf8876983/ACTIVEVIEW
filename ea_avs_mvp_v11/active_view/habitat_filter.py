"""
Habitat 环境可行性与可达性过滤器 —— habitat_filter.py
===================================================

职责：
    1. 三阶段过滤候选观察视点：
       - 阶段一: NavMesh 可行走性过滤 (is_navigable: 过滤墙体内、家具内、悬空点)；
       - 阶段二: 测地线路径可达性过滤 (is_reachable: 基于 ShortestPath 连通性检测并计算真实 navigation_cost)；
       - 阶段三: 目标视线通视性过滤 (is_visible: 基于 VisibilityChecker 光线追踪检测)；
    2. 结构化统计与日志汇报；
    3. 输出真正具备可行性、可到达且无完全遮挡的 Feasible Viewpoints 集合。
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from ea_avs_mvp_v11.active_view.candidate_generator import load_viewpoint_config
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.active_view.visibility_checker import VisibilityChecker

logger = logging.getLogger("habitat_filter")


class HabitatViewFilter:
    """Habitat 室内三阶段候选视点过滤器。"""

    def __init__(
        self,
        pathfinder: Optional[Any] = None,
        visibility_checker: Optional[VisibilityChecker] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        nav_bounds: Optional[Dict[str, Any]] = None,
    ):
        loaded_cfg = load_viewpoint_config(config_path)
        if config:
            loaded_cfg.update(config)

        self.config = loaded_cfg
        self.nav_cfg = self.config.get("navigation", {})
        self.vis_cfg = self.config.get("visibility", {})

        self.pathfinder = pathfinder
        self.visibility_checker = visibility_checker or VisibilityChecker(config=self.vis_cfg)

        self.check_navmesh = bool(self.nav_cfg.get("check_navmesh", True))
        self.check_reachability = bool(self.nav_cfg.get("check_reachability", True))
        self.max_snap_dist = float(self.nav_cfg.get("max_snap_distance_m", 0.50))
        self.max_path_cost = float(self.nav_cfg.get("max_path_cost_m", 15.0))

        # 自定义区域边界限制 (用于无 Habitat 引擎时的精准几何验证)
        self.nav_bounds = nav_bounds

    def filter_viewpoints(
        self,
        candidates: List[Viewpoint],
        human_position: Union[List[float], np.ndarray],
        robot_current_position: Optional[Union[List[float], np.ndarray]] = None,
    ) -> List[Viewpoint]:
        """
        三阶段严格过滤候选视点集合。

        参数:
            candidates: 原始生成的候选视点列表
            human_position: 人体三维世界坐标 [hx, hy, hz]
            robot_current_position: 机器人当前起点坐标 [rx, ry, rz]

        返回:
            feasible_viewpoints: 满足全部约束的可行视点列表
        """
        total_generated = len(candidates)
        hx, hy, hz = [float(x) for x in human_position]

        navigable_count = 0
        reachable_count = 0
        visible_count = 0

        feasible_viewpoints: List[Viewpoint] = []

        for vp in candidates:
            pos = np.array(vp.position, dtype=np.float32)

            # =================================================================
            # 阶段一: NavMesh 可行走性检测 (is_navigable)
            # =================================================================
            is_nav = self._check_is_navigable(pos)
            vp.is_navigable = is_nav
            if not is_nav:
                vp.is_reachable = False
                vp.is_visible = False
                continue
            navigable_count += 1

            # =================================================================
            # 阶段二: 测地线路径可达性与导航代价 (is_reachable & navigation_cost)
            # =================================================================
            if robot_current_position is not None:
                r_pos = np.array(robot_current_position, dtype=np.float32)
                is_reach, path_cost = self._check_reachability(r_pos, pos)
                vp.is_reachable = is_reach
                vp.navigation_cost = round(float(path_cost), 4)
                if not is_reach or path_cost > self.max_path_cost:
                    vp.is_reachable = False
                    vp.is_visible = False
                    continue
            else:
                vp.is_reachable = True
            reachable_count += 1

            # =================================================================
            # 阶段三: 目标视线通视性检测 (is_visible)
            # =================================================================
            is_vis, vis_score = self.visibility_checker.check_visibility(vp, human_position)
            vp.is_visible = is_vis
            vp.metadata["visibility_score"] = vis_score
            if not is_vis:
                continue
            visible_count += 1

            feasible_viewpoints.append(vp)

        # 结构化日志汇报
        logger.info("=========================================================")
        logger.info("Candidate Generation:")
        logger.info("  Generated candidates: %d", total_generated)
        logger.info("")
        logger.info("Filtering:")
        logger.info("  Navigable: %d / %d", navigable_count, total_generated)
        logger.info("  Reachable: %d / %d", reachable_count, navigable_count)
        logger.info("  Visible:   %d / %d", visible_count, reachable_count)
        logger.info("")
        logger.info("Final feasible viewpoints: %d / %d", len(feasible_viewpoints), total_generated)
        logger.info("=========================================================")

        return feasible_viewpoints

    def _check_is_navigable(self, pos: np.ndarray) -> bool:
        """检查点是否位于有效地面或 NavMesh。"""
        if not self.check_navmesh:
            return True

        # 1. 如果有 Habitat Pathfinder
        if self.pathfinder is not None:
            try:
                # 检查是否可吸附到导航网格上
                if not self.pathfinder.is_navigable(pos, max_snap_dist=self.max_snap_dist):
                    return False
                snapped_pt = self.pathfinder.snap_point(pos)
                if np.isnan(snapped_pt).any():
                    return False
                dist_to_snap = float(np.linalg.norm(snapped_pt - pos))
                if dist_to_snap > self.max_snap_dist:
                    return False
                return True
            except Exception as e:
                logger.debug("Habitat is_navigable error: %s", e)

        # 2. 如果提供了几何导航边界限制
        if self.nav_bounds is not None:
            min_pt = self.nav_bounds.get("min", [-10, -10, -10])
            max_pt = self.nav_bounds.get("max", [10, 10, 10])
            for i in range(3):
                if pos[i] < min_pt[i] or pos[i] > max_pt[i]:
                    return False

            # 检查不可通行的禁行区 (obstacles / holes)
            forbidden_boxes = self.nav_bounds.get("forbidden_boxes", [])
            for box in forbidden_boxes:
                b_min = box.get("min", [0, 0, 0])
                b_max = box.get("max", [0, 0, 0])
                if (b_min[0] <= pos[0] <= b_max[0] and
                    b_min[1] <= pos[1] <= b_max[1] and
                    b_min[2] <= pos[2] <= b_max[2]):
                    return False

        return True

    def _check_reachability(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple[bool, float]:
        """计算两点间的测地线可达性与路径长度。"""
        if not self.check_reachability:
            euc_dist = float(np.linalg.norm(start_pos - goal_pos))
            return True, euc_dist

        # 1. 如果有 Habitat Pathfinder
        if self.pathfinder is not None:
            try:
                import habitat_sim
                path = habitat_sim.ShortestPath()
                path.requested_start = start_pos
                path.requested_end = goal_pos
                found = self.pathfinder.find_path(path)
                if found and path.geodesic_distance < self.max_path_cost:
                    return True, float(path.geodesic_distance)
                return False, float("inf")
            except Exception as e:
                logger.debug("Habitat shortest path error: %s", e)

        # 2. 默认欧氏几何距离测算
        euc_dist = float(np.linalg.norm(start_pos - goal_pos))
        return True, euc_dist
