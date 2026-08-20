"""
Human Placement 单元测试 —— test_human_placement.py
==================================================
"""

import unittest
from ea_avs_mvp_v8.human.human_placement import HumanPlacement, DEFAULT_VERIFIED_PLACEMENTS


class TestHumanPlacement(unittest.TestCase):
    def setUp(self):
        self.placer = HumanPlacement()

    def test_default_sampling(self):
        pose = self.placer.sample_position(scene_id="apartment_1")
        self.assertTrue(pose.is_valid)
        self.assertEqual(pose.position, [1.5, -1.60, 4.0])
        self.assertEqual(pose.yaw_deg, 0.0)
        self.assertEqual(pose.scale, 1.0)

    def test_custom_valid_sampling(self):
        pose = self.placer.sample_position(
            scene_id="apartment_1",
            default_position=[2.0, -1.60, 5.0],
            default_yaw_deg=90.0,
        )
        self.assertTrue(pose.is_valid)
        self.assertEqual(pose.position, [2.0, -1.60, 5.0])
        self.assertEqual(pose.yaw_deg, 90.0)

    def test_invalid_elevation_validation(self):
        # Y = +10.0m is in the sky
        is_valid = self.placer.validate_position([1.5, 10.0, 4.0], scene_id="apartment_1")
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
