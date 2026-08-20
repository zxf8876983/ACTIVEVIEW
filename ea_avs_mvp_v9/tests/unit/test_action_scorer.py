"""
Action Scorer 单元测试 —— test_action_scorer.py
=============================================
"""

import unittest
from ea_avs_mvp_v9.action.action_encoder import ActionEncoder
from ea_avs_mvp_v9.core.types import ViewFeature
from ea_avs_mvp_v9.scoring.action_scorer import ActionConditionedScorer


class TestActionScorer(unittest.TestCase):
    def setUp(self):
        self.encoder = ActionEncoder()
        self.scorer = ActionConditionedScorer({"w_geometry": 0.60, "w_action": 0.40})

    def test_fall_favors_longer_distance_and_side_front_view(self):
        fall_action = self.encoder.encode("fall")

        # 视点 A: 距离 2.5m, 角度 45 度 (符合 fall 先验)
        feat_a = ViewFeature(
            viewpoint_id="vp_a",
            distance=2.5,
            viewing_angle_deg=45.0,
            pose_coverage=1.0,
            visibility_loss_ratio=0.0,
            projected_area_ratio=0.05,
            region_coverages={"torso": 1.0, "pelvis": 1.0, "head": 1.0, "lower_body": 1.0},
            feasible=True,
        )

        # 视点 B: 距离 1.5m, 角度 0 度 (纯正面贴近，距离过近)
        feat_b = ViewFeature(
            viewpoint_id="vp_b",
            distance=1.5,
            viewing_angle_deg=0.0,
            pose_coverage=1.0,
            visibility_loss_ratio=0.0,
            projected_area_ratio=0.08,
            region_coverages={"torso": 1.0, "pelvis": 1.0, "head": 1.0, "lower_body": 1.0},
            feasible=True,
        )

        # 假设几何基础分相同
        score_a = self.scorer.score_single(feat_a, fall_action, geometry_score=0.70)
        score_b = self.scorer.score_single(feat_b, fall_action, geometry_score=0.70)

        self.assertGreater(score_a.action_delta, score_b.action_delta)
        self.assertGreater(score_a.total_score, score_b.total_score)

    def test_infeasible_viewpoint_scored_zero(self):
        act = self.encoder.encode("standing")
        feat_infeas = ViewFeature(
            viewpoint_id="vp_infeas",
            distance=2.0,
            viewing_angle_deg=0.0,
            pose_coverage=1.0,
            visibility_loss_ratio=0.0,
            projected_area_ratio=0.05,
            region_coverages={"torso": 1.0},
            feasible=False,
        )
        score = self.scorer.score_single(feat_infeas, act, geometry_score=0.0)
        self.assertEqual(score.total_score, 0.0)


if __name__ == "__main__":
    unittest.main()
