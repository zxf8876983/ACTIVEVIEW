"""
Unit Tests for Human Spawn & Placement Interface
"""

import unittest
from ea_avs_mvp_v7.human.human_spawn import sample_human_position, get_default_human_orientation


class TestHumanSpawn(unittest.TestCase):
    def test_default_scene_ground_position(self):
        pos = sample_human_position(scene_id="apartment_1")
        self.assertEqual(len(pos), 3)
        self.assertEqual(pos[0], 1.5)
        self.assertEqual(pos[1], -1.60)
        self.assertEqual(pos[2], 4.0)

    def test_user_specified_position_override(self):
        custom_pos = [2.0, -1.60, 5.0]
        pos = sample_human_position(scene_id="apartment_1", default_position=custom_pos)
        self.assertEqual(pos, [2.0, -1.60, 5.0])

    def test_default_orientation(self):
        yaw = get_default_human_orientation("apartment_1", default_yaw_deg=45.0)
        self.assertEqual(yaw, 45.0)


if __name__ == "__main__":
    unittest.main()
