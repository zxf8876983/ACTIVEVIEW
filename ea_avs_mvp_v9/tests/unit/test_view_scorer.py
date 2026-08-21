"""
Learnable View Scorer 单元测试 —— test_view_scorer.py
===================================================
"""

import unittest
import torch
from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer


class TestViewScorer(unittest.TestCase):
    def setUp(self):
        self.model = LearnableViewScorer(
            pose_input_dim=49,
            pose_embed_dim=32,
            action_input_dim=5,
            action_embed_dim=16,
            view_input_dim=11,
            view_embed_dim=32,
        )

    def test_forward_multi_view_shape(self):
        pose = torch.randn(2, 49)
        action = torch.zeros(2, 5)
        action[:, 0] = 1.0  # Fall
        views = torch.randn(2, 16, 11)

        scores = self.model(pose, action, views)
        self.assertEqual(scores.shape, (2, 16))
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))

    def test_ablation_switches_functional(self):
        pose = torch.randn(1, 49)
        action = torch.zeros(1, 5)
        action[:, 1] = 1.0  # Sitting
        views = torch.randn(1, 8, 11)

        scores_full = self.model(pose, action, views)
        scores_no_act = self.model(pose, action, views, ablate_action=True)
        scores_no_pose = self.model(pose, action, views, ablate_pose=True)

        self.assertEqual(scores_full.shape, (1, 8))
        self.assertEqual(scores_no_act.shape, (1, 8))
        self.assertEqual(scores_no_pose.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
