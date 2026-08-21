import unittest
import torch
from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer


class TestViewScorer(unittest.TestCase):
    def setUp(self):
        self.model = LearnableViewScorer(
            pose_input_dim=49,
            pose_embed_dim=32,
            view_input_dim=13,
            view_embed_dim=32,
        )

    def test_forward_multi_view_shape(self):
        pose = torch.randn(2, 49)
        views = torch.randn(2, 16, 13)

        scores = self.model(pose, views)
        self.assertEqual(scores.shape, (2, 16))
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))

    def test_ablation_pose_switch(self):
        pose = torch.randn(1, 49)
        views = torch.randn(1, 8, 13)

        scores_full = self.model(pose, views)
        scores_no_pose = self.model(pose, views, ablate_pose=True)

        self.assertEqual(scores_full.shape, (1, 8))
        self.assertEqual(scores_no_pose.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
