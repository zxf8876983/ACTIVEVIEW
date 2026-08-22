"""
ACTIVEVIEW v11.4 闭环主动感知单元测试套件 —— test_v114_closed_loop.py
================================================================

测试内容：
    1. NavigationController 导航轨迹生成与测地线计算；
    2. ClosedLoopActivePerceptionRunner 单 Episode 全流程闭环测试；
    3. 各基准策略 (Random, Nearest, Fixed Front, Ours, Oracle) 初始状态严格一致性测试；
    4. 闭环科研指标计算一致性与非负性测试；
    5. 数据集与评测结果 JSON 序列化与结构完整性测试。
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.closed_loop_episode import ClosedLoopActivePerceptionRunner
from ea_avs_mvp_v11.active_view.navigation_controller import NavigationController, NavigationTrajectory
from ea_avs_mvp_v11.core.paths import get_data_root


class TestV114ClosedLoop(unittest.TestCase):
    """v11.4 闭环主动感知单元测试。"""

    def setUp(self):
        self.data_root = get_data_root()
        self.runner = ClosedLoopActivePerceptionRunner(data_root=self.data_root, seed=42)
        self.nav_ctrl = NavigationController(step_size=0.25)

    def test_navigation_controller_planning(self):
        """测试 NavigationController 轨迹规划与参数合法性。"""
        start = [0.0, 0.0, 5.0]
        target = [1.5, 0.0, 0.0]
        human = [0.0, 0.0, 0.0]

        traj = self.nav_ctrl.plan_trajectory(start, target, human_position=human)

        self.assertIsInstance(traj, NavigationTrajectory)
        self.assertTrue(traj.is_success)
        self.assertGreater(traj.total_distance, 0.0)
        self.assertGreaterEqual(traj.num_steps, 2)
        self.assertEqual(len(traj.waypoints), traj.num_steps)
        self.assertAlmostEqual(traj.waypoints[0][0], start[0], places=3)
        self.assertAlmostEqual(traj.waypoints[-1][0], target[0], places=3)

        d = traj.to_dict()
        self.assertIn("total_distance", d)
        self.assertIn("waypoints", d)
        self.assertIn("final_yaw_deg", d)

    def test_closed_loop_single_episode_flow(self):
        """测试单 Episode 全流程闭环执行与各阶段输出。"""
        ep_res = self.runner.run_single_episode(
            episode_idx=0,
            scene_id="apartment_1",
            action_id=0,
            instance_idx=0,
        )

        self.assertEqual(ep_res.scene_id, "apartment_1")
        self.assertEqual(ep_res.action_id, 0)
        self.assertEqual(ep_res.action_label, "standing")

        # 检查初始观察
        init_obs = ep_res.initial_observation
        self.assertIn("entropy", init_obs)
        self.assertIn("confidence", init_obs)
        self.assertGreaterEqual(init_obs["entropy"], 0.0)

        # 检查 5 大策略全部执行
        for p in ["random", "nearest", "fixed_front", "utility_predictor", "oracle"]:
            self.assertIn(p, ep_res.policy_results)
            p_res = ep_res.policy_results[p]
            self.assertGreater(p_res.navigation_distance, 0.0)
            self.assertGreaterEqual(p_res.entropy_after, 0.0)
            self.assertGreaterEqual(p_res.confidence_after, 0.0)
            self.assertLessEqual(p_res.confidence_after, 1.0)
            self.assertIsInstance(p_res.is_correct_after, bool)

    def test_policy_fairness_identical_initial_state(self):
        """验证所有策略基于完全相同的初始状态和候选池。"""
        ep_res = self.runner.run_single_episode(
            episode_idx=1,
            scene_id="apartment_1",
            action_id=1,
            instance_idx=1,
        )

        # 验证初始观察是唯一的
        init_h = ep_res.initial_observation["entropy"]
        init_pos = ep_res.robot_initial_viewpoint["position"]

        for p_name, p_res in ep_res.policy_results.items():
            start_pos = p_res.trajectory["start_position"]
            self.assertEqual(start_pos, init_pos)
            # 验证 Entropy Reduction 计算精准
            expected_reduction = round(init_h - p_res.entropy_after, 4)
            self.assertAlmostEqual(p_res.entropy_reduction, expected_reduction, places=3)

    def test_closed_loop_metrics_consistency(self):
        """验证闭环评价指标的数学一致性与非负性。"""
        ep_res = self.runner.run_single_episode(
            episode_idx=2,
            scene_id="apartment_1",
            action_id=2,
            instance_idx=2,
        )

        oracle_h = ep_res.policy_results["oracle"].entropy_after
        for p_name, p_res in ep_res.policy_results.items():
            # Oracle Gap 必须 >= 0
            self.assertGreaterEqual(p_res.oracle_gap, -1e-5)
            # Efficiency 必须与 distance 成反比
            expected_eff = p_res.entropy_reduction / max(p_res.navigation_distance, 0.1)
            self.assertAlmostEqual(p_res.navigation_efficiency, expected_eff, places=3)

    def test_dataset_serialization_and_summary(self):
        """验证完整字典与 JSON 序列化合法性。"""
        ep_res = self.runner.run_single_episode(
            episode_idx=3,
            scene_id="apartment_1",
            action_id=3,
            instance_idx=3,
        )
        d = ep_res.to_dict()
        s = json.dumps(d)
        self.assertIsInstance(s, str)
        loaded = json.loads(s)
        self.assertEqual(loaded["episode_id"], ep_res.episode_id)
        self.assertEqual(len(loaded["policy_results"]), 5)


if __name__ == "__main__":
    unittest.main()
