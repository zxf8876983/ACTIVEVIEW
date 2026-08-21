import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v9.core.types import ActionClass
from ea_avs_mvp_v9.evaluation.oracle_evaluator import OracleViewEvaluator
from ea_avs_mvp_v9.training.dataset import create_mock_joints_for_action


class TestOracleEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = OracleViewEvaluator()
        self.human_pos = [1.5, -1.6, 4.0]
        self.joints = create_mock_joints_for_action(ActionClass.SITTING, self.human_pos, yaw_deg=0.0)
        self.candidates = [
            CandidateViewpoint(viewpoint_id="vp_front", position=[1.5, -0.4, 6.0], yaw_deg=0.0, radius=2.0, angle_deg=0.0, feasible=True),
            CandidateViewpoint(viewpoint_id="vp_back", position=[1.5, -0.4, 2.0], yaw_deg=180.0, radius=2.0, angle_deg=180.0, feasible=True),
            CandidateViewpoint(viewpoint_id="vp_infeas", position=[0.0, 0.0, 0.0], yaw_deg=0.0, radius=1.0, angle_deg=0.0, feasible=False),
        ]

    def test_evaluate_oracle_best_picks_optimal_view(self):
        res = self.evaluator.evaluate_oracle_best(
            candidates=self.candidates,
            gt_joints=self.joints,
            human_pos=self.human_pos,
            human_yaw_deg=0.0,
            initial_quality=0.50,
        )
        self.assertEqual(res.best_viewpoint_id, "vp_front")
        self.assertGreaterEqual(res.oracle_quality_score, 0.70)
        self.assertGreater(res.oracle_information_gain, 0.0)
        self.assertIn("head", res.oracle_body_parts_visibility)


if __name__ == "__main__":
    unittest.main()
