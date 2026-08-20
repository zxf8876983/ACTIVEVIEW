"""
View Quality 单元测试 —— test_view_quality.py
============================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator, compute_view_quality_score


class TestViewQuality(unittest.TestCase):
    def setUp(self):
        self.evaluator = ViewQualityEvaluator({
            "optimal_distance": 2.0,
            "max_distance": 4.5,
            "hfov_deg": 90.0,
            "w1_visibility": 0.5,
            "w2_pose_coverage": 0.3,
            "w3_distance": 0.2,
        })
        self.joints = {
            "pelvis": [1.5, -0.80, 4.0],
            "head": [1.5, -0.05, 4.0],
            "left_ankle": [1.4, -1.55, 4.0],
            "right_ankle": [1.6, -1.55, 4.0],
        }

    def test_compute_view_quality_score_formula(self):
        score = compute_view_quality_score(
            human_visibility=1.0,
            pose_coverage=1.0,
            distance=2.0,
            optimal_distance=2.0,
            viewing_angle_deg=0.0,
            w1=0.5,
            w2=0.3,
            w3=0.2,
        )
        self.assertAlmostEqual(score, 0.80, delta=0.01)

    def test_ranking_prefers_frontal_viewpoint(self):
        # Frontal viewpoint at Z=6.0 (looking at human at Z=4.0)
        vp_front = CandidateViewpoint(
            viewpoint_id="front",
            position=[1.5, -1.60, 6.0],
            yaw_deg=0.0,
            radius=2.0,
            angle_deg=0.0,
        )
        # Side viewpoint at X=3.5, Z=4.0 (looking at human at X=1.5, Z=4.0)
        vp_side = CandidateViewpoint(
            viewpoint_id="side",
            position=[3.5, -1.60, 4.0],
            yaw_deg=270.0,
            radius=2.0,
            angle_deg=90.0,
        )

        ranked = self.evaluator.rank_viewpoints([vp_side, vp_front], self.joints, human_yaw_deg=0.0)
        best_vp, best_q = ranked[0]
        self.assertEqual(best_vp.viewpoint_id, "front")
        self.assertGreaterEqual(best_q.visibility_score, ranked[1][1].visibility_score)


if __name__ == "__main__":
    unittest.main()
