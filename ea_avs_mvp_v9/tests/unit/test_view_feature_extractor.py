"""
View Feature Extractor 单元测试 —— test_view_feature_extractor.py
===============================================================
"""

import unittest
from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor


class TestViewFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ViewFeatureExtractor({"hfov_deg": 90.0, "max_distance": 4.5})
        self.mock_joints = {
            "pelvis": [1.5, -0.80, 4.0],
            "head": [1.5, -0.05, 4.0],
            "spine": [1.5, -0.50, 4.0],
            "left_hip": [1.4, -0.90, 4.0],
            "right_hip": [1.6, -0.90, 4.0],
            "left_shoulder": [1.3, -0.30, 4.0],
            "right_shoulder": [1.7, -0.30, 4.0],
        }

    def test_extract_frontal_features(self):
        vp_front = CandidateViewpoint(
            viewpoint_id="vp_front",
            position=[1.5, -1.60, 6.0],
            yaw_deg=0.0,
            radius=2.0,
            angle_deg=0.0,
        )
        feat = self.extractor.extract(vp_front, self.mock_joints, human_yaw_deg=0.0)
        self.assertEqual(feat.viewpoint_id, "vp_front")
        self.assertAlmostEqual(feat.distance, 2.04, delta=0.05)
        self.assertAlmostEqual(feat.viewing_angle_deg, 0.0, delta=1.0)
        self.assertGreater(feat.pose_coverage, 0.8)
        self.assertIn("torso", feat.region_coverages)
        self.assertIn("head", feat.region_coverages)
        self.assertIn("pelvis", feat.region_coverages)


if __name__ == "__main__":
    unittest.main()
