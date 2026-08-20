"""
Baseline Strategies 单元测试 —— test_baseline_strategies.py
==========================================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality
from ea_avs_mvp_v8.evaluation.baseline_strategies import (
    select_geometry_best_view,
    select_nearest_view,
    select_random_view,
    select_view,
)


class TestBaselineStrategies(unittest.TestCase):
    def setUp(self):
        self.vp1 = CandidateViewpoint(
            viewpoint_id="vp_far",
            position=[1.5, -1.60, 7.0],
            yaw_deg=0.0,
            radius=3.0,
            angle_deg=0.0,
            feasible=True,
        )
        self.vp2 = CandidateViewpoint(
            viewpoint_id="vp_near",
            position=[1.5, -1.60, 5.5],
            yaw_deg=0.0,
            radius=1.5,
            angle_deg=0.0,
            feasible=True,
        )
        self.vp3 = CandidateViewpoint(
            viewpoint_id="vp_optimal",
            position=[1.5, -1.60, 6.0],
            yaw_deg=0.0,
            radius=2.0,
            angle_deg=0.0,
            feasible=True,
        )

        self.q1 = ViewpointQuality(
            viewpoint_id="vp_far",
            distance=3.0,
            viewing_angle_deg=0.0,
            visible_joints_count=16,
            visibility_score=0.65,
            occlusion_ratio=0.0,
            pose_coverage=1.0,
            is_valid=True,
        )
        self.q2 = ViewpointQuality(
            viewpoint_id="vp_near",
            distance=1.5,
            viewing_angle_deg=0.0,
            visible_joints_count=16,
            visibility_score=0.72,
            occlusion_ratio=0.0,
            pose_coverage=1.0,
            is_valid=True,
        )
        self.q3 = ViewpointQuality(
            viewpoint_id="vp_optimal",
            distance=2.0,
            viewing_angle_deg=0.0,
            visible_joints_count=16,
            visibility_score=0.80,
            occlusion_ratio=0.0,
            pose_coverage=1.0,
            is_valid=True,
        )

        self.vps = [self.vp1, self.vp2, self.vp3]
        self.qs = [self.q1, self.q2, self.q3]

    def test_select_geometry_best(self):
        vp, q = select_geometry_best_view(self.vps, self.qs)
        self.assertEqual(vp.viewpoint_id, "vp_optimal")
        self.assertEqual(q.visibility_score, 0.80)

    def test_select_nearest(self):
        vp, q = select_nearest_view(self.vps, self.qs, human_position=[1.5, -1.60, 4.0])
        self.assertEqual(vp.viewpoint_id, "vp_near")

    def test_select_random(self):
        vp, q = select_random_view(self.vps, self.qs, seed=42)
        self.assertIn(vp.viewpoint_id, ["vp_far", "vp_near", "vp_optimal"])

    def test_select_view_dispatcher(self):
        vp_best, _ = select_view(self.vps, self.qs, strategy="geometry_best")
        self.assertEqual(vp_best.viewpoint_id, "vp_optimal")

        vp_near, _ = select_view(self.vps, self.qs, strategy="nearest", human_position=[1.5, -1.60, 4.0])
        self.assertEqual(vp_near.viewpoint_id, "vp_near")


if __name__ == "__main__":
    unittest.main()
