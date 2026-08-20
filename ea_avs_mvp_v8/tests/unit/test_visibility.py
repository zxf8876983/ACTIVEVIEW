"""
Visibility Evaluator 单元测试 —— test_visibility.py
==================================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.visibility.visibility_evaluator import VisibilityEvaluator


class TestVisibilityEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = VisibilityEvaluator({
            "optimal_distance": 2.0,
            "max_distance": 5.0,
            "hfov": 90.0,
        })
        self.dummy_joints = {
            "pelvis": [1.5, -0.80, 4.0],
            "head": [1.5, -0.05, 4.0],
            "left_ankle": [1.4, -1.55, 4.0],
            "right_ankle": [1.6, -1.55, 4.0],
        }

    def test_evaluate_frontal_viewpoint(self):
        # Viewpoint placed at Z=6.0 looking at human at Z=4.0
        vp = CandidateViewpoint(
            viewpoint_id="v_front",
            position=[1.5, -1.60, 6.0],
            yaw_deg=0.0,
            radius=2.0,
            angle_deg=0.0,
            camera_height=1.2,
        )
        quality = self.evaluator.evaluate(vp, self.dummy_joints, human_yaw_deg=0.0)

        self.assertEqual(quality.viewpoint_id, "v_front")
        self.assertAlmostEqual(quality.distance, 2.05, delta=0.2)
        self.assertGreater(quality.visibility_score, 0.5)
        self.assertEqual(quality.visible_joints_count, 4)
        self.assertAlmostEqual(quality.pose_coverage, 1.0)
        self.assertTrue(quality.is_valid)

    def test_evaluate_batch(self):
        vp1 = CandidateViewpoint(
            viewpoint_id="v1",
            position=[1.5, -1.60, 6.0],
            yaw_deg=0.0,
            radius=2.0,
            angle_deg=0.0,
        )
        vp2 = CandidateViewpoint(
            viewpoint_id="v2",
            position=[1.5, -1.60, 2.0],
            yaw_deg=180.0,
            radius=2.0,
            angle_deg=180.0,
        )
        results = self.evaluator.evaluate_batch([vp1, vp2], self.dummy_joints)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.is_valid for r in results))


if __name__ == "__main__":
    unittest.main()
