"""
Viewpoint Selector 单元测试 —— test_viewpoint_selector.py
=========================================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality
from ea_avs_mvp_v9.core.types import ActionViewpointScore
from ea_avs_mvp_v9.selection.viewpoint_selector import ViewpointSelector


class TestViewpointSelector(unittest.TestCase):
    def setUp(self):
        self.vps = [
            CandidateViewpoint(viewpoint_id="vp1", position=[1.5, -1.6, 6.0], yaw_deg=0.0, radius=2.0, angle_deg=0.0, feasible=True),
            CandidateViewpoint(viewpoint_id="vp2", position=[3.5, -1.6, 4.0], yaw_deg=270.0, radius=2.0, angle_deg=90.0, feasible=True),
            CandidateViewpoint(viewpoint_id="vp3", position=[1.5, -1.6, 2.5], yaw_deg=180.0, radius=1.5, angle_deg=180.0, feasible=False),
        ]
        self.scores = [
            ActionViewpointScore(viewpoint_id="vp1", action_name="fall", geometry_score=0.70, action_delta=0.60, total_score=0.66, region_score=0.8, aspect_score=0.7, distance_score=0.8),
            ActionViewpointScore(viewpoint_id="vp2", action_name="fall", geometry_score=0.60, action_delta=0.90, total_score=0.72, region_score=0.9, aspect_score=0.9, distance_score=0.9),
            ActionViewpointScore(viewpoint_id="vp3", action_name="fall", geometry_score=0.0, action_delta=0.0, total_score=0.0, region_score=0.0, aspect_score=0.0, distance_score=0.0),
        ]
        self.geom_qs = [
            ViewpointQuality(viewpoint_id="vp1", distance=2.0, viewing_angle_deg=0.0, visible_joints_count=16, visibility_score=0.70),
            ViewpointQuality(viewpoint_id="vp2", distance=2.0, viewing_angle_deg=90.0, visible_joints_count=16, visibility_score=0.60),
            ViewpointQuality(viewpoint_id="vp3", distance=1.5, viewing_angle_deg=180.0, visible_joints_count=0, visibility_score=0.0),
        ]

    def test_select_action_conditioned(self):
        # vp2 has higher total_score (0.72 vs 0.66)
        sel_vp, sel_score = ViewpointSelector.select(
            self.vps, self.scores, geometry_qualities=self.geom_qs, strategy="action_conditioned"
        )
        self.assertEqual(sel_vp.viewpoint_id, "vp2")
        self.assertEqual(sel_score.total_score, 0.72)

    def test_select_geometry_best(self):
        # vp1 has higher geometry_score (0.70 vs 0.60)
        sel_vp, sel_score = ViewpointSelector.select(
            self.vps, self.scores, geometry_qualities=self.geom_qs, strategy="geometry_best"
        )
        self.assertEqual(sel_vp.viewpoint_id, "vp1")

    def test_select_random_and_nearest(self):
        vp_rand, _ = ViewpointSelector.select(self.vps, self.scores, strategy="random", seed=42)
        self.assertTrue(vp_rand.feasible)

        vp_near, _ = ViewpointSelector.select(self.vps, self.scores, strategy="nearest", human_position=[1.5, -1.6, 4.0])
        self.assertTrue(vp_near.feasible)


if __name__ == "__main__":
    unittest.main()
