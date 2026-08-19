"""
零 GT 信息泄漏静态与运行时检验套件 —— test_no_gt_leakage.py
=========================================================

功能：
    对 v6.0 的估计状态主决策链路执行三重防护审计：
        1. Guard A: 严格 AST 语法树遍历，扫描线上模块禁止出现任何 GT 变量引用；
        2. Guard B: 检查 Estimated 评估模块绝不通过 runner 间接访问 Humanoid Identity；
        3. Guard C: 运行时未来观测拦截 Sentinel，断言在线候选评分与决策过程中 render_at 调用次数为 0。
"""

import ast
import inspect
import os
import sys
import unittest
import numpy as np

from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.estimated_predictive_evaluator import EstimatedPredictiveEvaluator
from ea_avs_v6.policies import EstimatedStateOursPolicy
from ea_avs_v6.candidate_sampler import CandidateSampler, CandidateView
from ea_avs_v6.estimated_human_state import EstimatedHumanState
from ea_avs_v6.action_pose_library import POSE_SKELETONS


class TestNoGTLeakage(unittest.TestCase):

    def setUp(self):
        self.pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ea_avs_v6")

    def test_guard_a_ast_forbidden_names(self):
        """Guard A: 静态 AST 遍历，扫描线上模块绝无 GT 变量名引用。"""
        forbidden_names = {
            "gt_human_pos",
            "gt_human_yaw",
            "gt_skeleton",
            "humanoid_object_ids",
            "target_link_object_ids",
            "keypoint_meta",
            "semantic_mask",
        }

        files_to_scan = [
            "human_state_estimator.py",
            "estimated_predictive_evaluator.py",
            "candidate_sampler.py",
        ]

        for fname in files_to_scan:
            fpath = os.path.join(self.pkg_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=fname)

            for node in ast.walk(tree):
                # 检查变量名、属性名、参数名
                if isinstance(node, ast.Name) and node.id in forbidden_names:
                    self.fail(f"在 {fname} (Line {node.lineno}) 发现直接读取禁止的 GT 变量: '{node.id}'")
                elif isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                    self.fail(f"在 {fname} (Line {node.lineno}) 发现访问禁止的 GT 属性: '{node.attr}'")
                elif isinstance(node, ast.arg) and node.arg in forbidden_names:
                    self.fail(f"在 {fname} (Line {node.lineno}) 发现函数参数包含禁止的 GT 入参: '{node.arg}'")

    def test_guard_b_no_runner_humanoid_access(self):
        """Guard B: 检查 EstimatedPredictiveEvaluator 绝无通过 runner 偷取 Humanoid identity。"""
        fpath = os.path.join(self.pkg_dir, "estimated_predictive_evaluator.py")
        with open(fpath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename="estimated_predictive_evaluator.py")

        forbidden_attributes = {
            "humanoid_manager",
            "_humanoid_manager",
            "get_humanoid_object_ids",
            "get_humanoid_gt_skeleton",
            "get_articulated_object_manager",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
                self.fail(f"在 estimated_predictive_evaluator.py (Line {node.lineno}) 发现非法访问 Humanoid 属性: '{node.attr}'")

    def test_guard_c_future_render_runtime_sentinel(self):
        """Guard C: 运行时 Sentinel 保护，确保在线候选打分与选择过程中绝不调用 render_at。"""
        class SentinelRunner:
            def __init__(self):
                self.render_at_call_count = 0

            def render_at(self, *args, **kwargs):
                self.render_at_call_count += 1
                raise AssertionError("CRITICAL VIOLATION: 在线选择阶段非法调用了未来候选观测 render_at()！")

            def is_navigable(self, pt):
                return True

            def snap_point(self, pt):
                return np.array(pt, dtype=np.float32)

            def geodesic_distance(self, p1, p2):
                return float(np.linalg.norm(np.array(p1) - np.array(p2)))

            def cast_ray(self, origin, direction, max_distance):
                return {"has_hits": False}

        cfg = {
            "camera": {"width": 640, "height": 480, "hfov_deg": 90, "camera_height": 1.2},
            "visibility": {"min_depth": 0.5, "max_depth": 5.0},
            "predictive_score": {"w_action_occ_pred": 0.5, "w_orient_pred": 0.15, "w_center_pred": 0.1, "w_dist_pred": 0.1, "w_move": 0.15},
            "distance_score": {"optimal_distance": 2.0, "sigma": 0.7},
            "orientation_score": {"preferred_angle_deg": 45, "sigma_deg": 35},
            "candidate_sampling": {"radii": [1.5, 2.0], "angles_deg": [-90, 0, 90], "min_distance_to_human": 1.2, "max_distance_to_human": 3.0, "max_geodesic_distance": 6.0},
            "action_part_weights": {"standing": {"torso": 0.35, "lower_body": 0.35, "head": 0.20, "arms": 0.10}},
            "occlusion": {"enabled": True, "ray_epsilon": 0.05, "target_tolerance": 0.12, "min_hit_distance": 0.05},
        }

        sentinel_runner = SentinelRunner()
        evaluator = EstimatedPredictiveEvaluator(cfg)
        sampler = CandidateSampler(cfg)
        policy = EstimatedStateOursPolicy()

        est_state = EstimatedHumanState(
            valid=True,
            human_position_world=np.array([0.0, 0.0, 2.0]),
            human_yaw=0.0,
            proxy_full_skeleton={k: np.array(v) + np.array([0.0, 0.0, 2.0]) for k, v in POSE_SKELETONS["standing"].items()},
        )
        robot_pos = np.array([0.0, 0.0, 0.0])

        # 1. 候选点采样
        cands = sampler.sample(est_state.human_position_world, robot_pos, sentinel_runner)
        curr_view = CandidateView(0, robot_pos, 0.0, 0.0, 2.0, True)

        # 2. 预测打分
        curr_view.pred_score = evaluator.score_view_pred(sentinel_runner, curr_view.position, curr_view.yaw, robot_pos, est_state, 0.0)
        for c in cands:
            c.pred_score = evaluator.score_view_pred(sentinel_runner, c.position, c.yaw, robot_pos, est_state, c.geodesic_distance)

        # 3. 策略决策
        selected = policy.select(curr_view, cands, is_state_valid=True)

        # 校验整个感知预测与决策过程 render_at 调用为 0
        self.assertEqual(
            sentinel_runner.render_at_call_count, 0,
            "在线预测与决策阶段违规调用了 render_at 获取未来图像！"
        )
        self.assertIsNotNone(selected)


if __name__ == "__main__":
    unittest.main()
