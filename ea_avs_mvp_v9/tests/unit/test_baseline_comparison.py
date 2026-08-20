"""
Baseline Comparison 单元测试 —— test_baseline_comparison.py
===========================================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality
from ea_avs_mvp_v9.action.action_encoder import ActionEncoder
from ea_avs_mvp_v9.core.types import ActionViewpointScore, ViewFeature
from ea_avs_mvp_v9.evaluation.baseline_comparison import compare_all_baselines


class TestBaselineComparison(unittest.TestCase):
    def setUp(self):
        self.encoder = ActionEncoder()
        self.action = self.encoder.encode("fall")

        self.vps = [
            CandidateViewpoint("vp1", [1.5, -1.6, 6.0], 0.0, 2.0, 0.0, feasible=True),
            CandidateViewpoint("vp2", [3.5, -1.6, 4.0], 270.0, 2.0, 90.0, feasible=True),
        ]
        self.scores = [
            ActionViewpointScore("vp1", "fall", 0.70, 0.50, 0.62, 0.7, 0.5, 0.8),
            ActionViewpointScore("vp2", "fall", 0.60, 0.85, 0.70, 0.9, 0.8, 0.8),
        ]
        self.feats = [
            ViewFeature("vp1", 2.0, 0.0, 1.0, 0.0, 0.05, {"torso": 1.0, "lower_body": 1.0}, feasible=True),
            ViewFeature("vp2", 2.0, 90.0, 1.0, 0.0, 0.05, {"torso": 1.0, "lower_body": 1.0}, feasible=True),
        ]
        self.geom_qs = [
            ViewpointQuality("vp1", 2.0, 0.0, 16, 0.70),
            ViewpointQuality("vp2", 2.0, 90.0, 16, 0.60),
        ]

    def test_compare_all_baselines_outputs_four_strategies(self):
        report = compare_all_baselines(
            viewpoints=self.vps,
            action_scores=self.scores,
            features=self.feats,
            geometry_qualities=self.geom_qs,
            action=self.action,
        )
        self.assertIn("strategies", report)
        self.assertEqual(len(report["strategies"]), 4)
        self.assertIn("random", report["strategies"])
        self.assertIn("nearest", report["strategies"])
        self.assertIn("geometry_best", report["strategies"])
        self.assertIn("action_conditioned", report["strategies"])
        self.assertTrue(report["preferred_viewpoint_shifted"])
        self.assertGreater(report["action_conditioned_gain_over_v8"], 0.0)


if __name__ == "__main__":
    unittest.main()
