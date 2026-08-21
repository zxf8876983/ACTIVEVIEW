import unittest
from ea_avs_mvp_v9.core.types import ViewFeature
from ea_avs_mvp_v9.scoring.human_state_scorer import HumanStateAwareViewScorer


class TestHumanStateScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = HumanStateAwareViewScorer()
        self.feat = ViewFeature(
            viewpoint_id="vp_001",
            distance=2.0,
            viewing_angle_deg=0.0,
            pose_coverage=1.0,
            visibility_loss_ratio=0.0,
            projected_area_ratio=0.05,
            body_part_visibilities={
                "head": 1.0, "torso": 1.0, "pelvis": 1.0,
                "left_hand": 1.0, "right_hand": 1.0,
                "left_leg": 1.0, "right_leg": 1.0,
            },
            feasible=True,
        )

    def test_score_optimal_view(self):
        res = self.scorer.score(self.feat, geom_visibility=1.0)
        self.assertEqual(res["viewpoint_id"], "vp_001")
        self.assertAlmostEqual(res["total_score"], 0.90, places=2)
        self.assertEqual(res["distance_penalty"], 0.0)

    def test_score_infeasible_view(self):
        infeas_feat = ViewFeature(
            viewpoint_id="vp_infeas",
            distance=2.0,
            viewing_angle_deg=0.0,
            pose_coverage=0.0,
            visibility_loss_ratio=1.0,
            projected_area_ratio=0.0,
            feasible=False,
        )
        res = self.scorer.score(infeas_feat)
        self.assertEqual(res["total_score"], 0.0)
        self.assertFalse(res["feasible"])


if __name__ == "__main__":
    unittest.main()
