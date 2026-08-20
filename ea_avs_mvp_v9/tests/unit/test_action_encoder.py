"""
Action Encoder 单元测试 —— test_action_encoder.py
=================================================
"""

import unittest
from ea_avs_mvp_v9.action.action_encoder import ActionEncoder, ALL_ACTION_CLASSES
from ea_avs_mvp_v9.core.types import ActionClass


class TestActionEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = ActionEncoder()

    def test_encode_fall_action(self):
        emb = self.encoder.encode("fall to the ground")
        self.assertEqual(emb.action_class, ActionClass.FALL)
        self.assertEqual(emb.action_name, "fall")
        self.assertEqual(len(emb.vector), len(ALL_ACTION_CLASSES))
        self.assertEqual(emb.vector[0], 1.0)
        self.assertIn("torso", emb.critical_regions)
        self.assertEqual(emb.optimal_distance, 2.5)

    def test_encode_sitting_action(self):
        emb = self.encoder.encode("sit down")
        self.assertEqual(emb.action_class, ActionClass.SITTING)
        self.assertEqual(emb.action_name, "sitting")
        self.assertEqual(emb.vector[1], 1.0)
        self.assertIn("lower_body", emb.critical_regions)

    def test_encode_bending_action(self):
        emb = self.encoder.encode("bend forward")
        self.assertEqual(emb.action_class, ActionClass.BENDING)
        self.assertEqual(emb.vector[3], 1.0)

    def test_encode_reaching_action(self):
        emb = self.encoder.encode("reach for glass")
        self.assertEqual(emb.action_class, ActionClass.REACHING)
        self.assertEqual(emb.vector[4], 1.0)
        self.assertIn("upper_body", emb.critical_regions)


if __name__ == "__main__":
    unittest.main()
