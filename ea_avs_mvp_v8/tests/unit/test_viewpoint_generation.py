"""
Viewpoint Generation 单元测试 —— test_viewpoint_generation.py
============================================================
"""

import math
import unittest
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator


class TestViewpointGeneration(unittest.TestCase):
    def setUp(self):
        self.generator = ViewpointGenerator({
            "radii": [1.5, 3.0],
            "num_angles": 4,
            "camera_height": 1.2,
            "face_human": True,
        })

    def test_candidate_generation_count(self):
        candidates = self.generator.generate_candidates(
            human_position=[1.5, -1.60, 4.0],
        )
        # 2 radii * 4 angles = 8 candidates
        self.assertEqual(len(candidates), 8)

    def test_candidate_positions_and_geometry(self):
        candidates = self.generator.generate_candidates(
            human_position=[0.0, 0.0, 0.0],
            custom_radii=[2.0],
            custom_num_angles=4,
        )
        self.assertEqual(len(candidates), 4)

        # Angle 0 deg: (sin 0, cos 0) -> (0, 0, 2)
        c0 = candidates[0]
        self.assertAlmostEqual(c0.position[0], 0.0, places=3)
        self.assertAlmostEqual(c0.position[2], 2.0, places=3)
        self.assertEqual(c0.radius, 2.0)
        self.assertEqual(c0.angle_deg, 0.0)

        # When at (0, 0, 2) looking at (0, 0, 0), direction is -Z, yaw is 0 deg
        self.assertAlmostEqual(c0.yaw_deg, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
