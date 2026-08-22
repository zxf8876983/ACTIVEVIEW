"""
Unit tests for v10 candidate viewpoint generator.
"""

import unittest
from ea_avs_mvp_v10.viewpoint.candidate_generator import CandidateViewpointGeneratorV10


class TestCandidateGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = CandidateViewpointGeneratorV10(
            radii=[1.5, 2.0, 2.5],
            num_angles=8,
            camera_height=1.20,
            ground_height=-1.60,
        )

    def test_generate_candidates_count_and_structure(self):
        h_pos = [1.0, -1.60, 2.0]
        candidates = self.generator.generate_candidates(h_pos, human_yaw_deg=0.0)

        # 3 radii * 8 angles = 24 candidates
        self.assertEqual(len(candidates), 24)

        for vp in candidates:
            self.assertTrue(vp.viewpoint_id.startswith("vp_"))
            self.assertIn(vp.radius, [1.5, 2.0, 2.5])
            self.assertAlmostEqual(vp.height, 1.20)
            self.assertEqual(len(vp.position), 3)
            # y 坐标为 ground_height = -1.60
            self.assertAlmostEqual(vp.position[1], -1.60)


if __name__ == "__main__":
    unittest.main()
