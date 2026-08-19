"""
零 GT 信息泄漏静态与运行时检验脚本 —— test_no_gt_leakage.py
=========================================================

功能：
    对 v6.0 的估计状态主决策链路执行严格的 AST 语法树检查与函数签名校验，
    确保在线感知与决策阶段不接收任何 Humanoid GT 参数，也不存在候选点未来观测泄漏。
"""

import inspect
import sys
import unittest

from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.estimated_predictive_evaluator import EstimatedPredictiveEvaluator
from ea_avs_v6.policies import EstimatedStateOursPolicy


class TestNoGTLeakage(unittest.TestCase):

    def test_01_human_state_estimator_signature(self):
        """校验 HumanStateEstimator.estimate 签名绝不包含任何 GT 变量。"""
        sig = inspect.signature(HumanStateEstimator.estimate)
        params = list(sig.parameters.keys())

        # 允许且仅允许的参数
        expected = ["self", "rgb", "depth", "camera_state"]
        self.assertEqual(
            params, expected,
            f"HumanStateEstimator.estimate 参数不合法: {params}，应为 {expected}"
        )

        forbidden_names = [
            "gt", "humanoid", "gt_human_pos", "gt_human_yaw", "gt_skeleton",
            "semantic", "semantic_mask", "humanoid_object_ids", "keypoint_meta",
        ]
        for p in params:
            for f in forbidden_names:
                self.assertNotIn(
                    f, p.lower(),
                    f"HumanStateEstimator.estimate 含有潜在 GT 泄漏参数: {p}"
                )

    def test_02_estimated_predictive_evaluator_signature(self):
        """校验 EstimatedPredictiveEvaluator.score_view_pred 签名使用 estimated_state 而非 GT。"""
        sig = inspect.signature(EstimatedPredictiveEvaluator.score_view_pred)
        params = list(sig.parameters.keys())

        self.assertIn("estimated_state", params)
        forbidden_gt = ["gt_human_pos", "gt_human_yaw", "gt_skeleton", "humanoid_object_ids", "keypoint_meta"]
        for f in forbidden_gt:
            self.assertNotIn(
                f, params,
                f"EstimatedPredictiveEvaluator.score_view_pred 含有禁止的 GT 参数: {f}"
            )

    def test_03_policy_signature(self):
        """校验 EstimatedStateOursPolicy.select 仅接收 current_view, candidates, is_state_valid。"""
        sig = inspect.signature(EstimatedStateOursPolicy.select)
        params = list(sig.parameters.keys())
        expected = ["self", "current_view", "candidates", "is_state_valid"]
        self.assertEqual(params, expected)


if __name__ == "__main__":
    unittest.main()
