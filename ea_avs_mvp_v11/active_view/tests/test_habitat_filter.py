"""
HabitatViewFilter 单元测试 —— test_habitat_filter.py
===================================================
"""

import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.habitat_filter import HabitatViewFilter
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.active_view.visibility_checker import VisibilityChecker


class TestHabitatViewFilter(unittest.TestCase):
    """测试三阶段候选视点过滤器的功能与约束行为。"""

    def setUp(self):
        self.generator = CandidateViewGenerator()
        self.human_pos = [0.0, 0.0, 0.0]
        self.robot_pos = [0.0, 0.0, 3.5]
        self.raw_candidates = self.generator.generate(self.human_pos, robot_current_position=self.robot_pos)

    def test_filter_all_pass_under_unconstrained_environment(self):
        """测试在无障碍、完全开阔环境下所有候选点均可通过过滤。"""
        h_filter = HabitatViewFilter(
            config={"navigation": {"check_navmesh": False, "check_reachability": False},
                    "visibility": {"check_visibility": False}}
        )
        feasible = h_filter.filter_viewpoints(self.raw_candidates, self.human_pos, self.robot_pos)
        self.assertEqual(len(feasible), len(self.raw_candidates))

    def test_filter_non_navigable_obstacles(self):
        """测试位于禁行区/障碍物内部的候选点被准确过滤 (is_navigable=False)。"""
        # 在 X in [1.2, 2.8], Z in [-1.0, 1.0] 设置禁行家具/墙体
        forbidden_boxes = [
            {"min": [1.2, -1.0, -1.0], "max": [2.8, 2.0, 1.0]}
        ]
        h_filter = HabitatViewFilter(
            nav_bounds={"min": [-5, -2, -5], "max": [5, 2, 5], "forbidden_boxes": forbidden_boxes},
            config={"navigation": {"check_navmesh": True, "check_reachability": False},
                    "visibility": {"check_visibility": False}}
        )
        feasible = h_filter.filter_viewpoints(self.raw_candidates, self.human_pos, self.robot_pos)
        self.assertLess(len(feasible), len(self.raw_candidates))

        # 验证被过滤掉的点其 is_navigable 属性为 False
        for vp in self.raw_candidates:
            vx, vy, vz = vp.position
            if 1.2 <= vx <= 2.8 and -1.0 <= vz <= 1.0:
                self.assertFalse(vp.is_navigable)

    def test_filter_occluded_viewpoints(self):
        """测试视线被实体遮挡的候选点被准确过滤 (is_visible=False)。"""
        # 在人体前方 X in [-0.5, 0.5], Z in [0.5, 1.2] 设置一面隔断墙
        obstacles = [
            {"name": "Partition_Wall", "min": [-0.5, -0.5, 0.5], "max": [0.5, 2.0, 1.2]}
        ]
        vis_checker = VisibilityChecker(obstacles=obstacles)
        h_filter = HabitatViewFilter(
            visibility_checker=vis_checker,
            config={"navigation": {"check_navmesh": False, "check_reachability": False},
                    "visibility": {"check_visibility": True}}
        )
        feasible = h_filter.filter_viewpoints(self.raw_candidates, self.human_pos, self.robot_pos)
        self.assertLess(len(feasible), len(self.raw_candidates))

    def test_filter_max_path_cost(self):
        """测试超出最大允许移动路径距离的视点被过滤。"""
        h_filter = HabitatViewFilter(
            config={"navigation": {"check_navmesh": False, "check_reachability": True, "max_path_cost_m": 2.0},
                    "visibility": {"check_visibility": False}}
        )
        # 机器人当前位置在 [0, 0, 10.0]，距离人体很远
        feasible = h_filter.filter_viewpoints(self.raw_candidates, self.human_pos, robot_current_position=[0.0, 0.0, 10.0])
        self.assertEqual(len(feasible), 0)


if __name__ == "__main__":
    unittest.main()
