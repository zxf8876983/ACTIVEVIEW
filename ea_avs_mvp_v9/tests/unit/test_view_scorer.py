import unittest
import torch
from ea_avs_mvp_v9.models.view_scorer import PerceptionAwareViewScorer


class TestViewScorer(unittest.TestCase):
    def setUp(self):
        self.model = PerceptionAwareViewScorer(
            obs_input_dim=71,
            obs_embed_dim=32,
            view_input_dim=13,
            view_embed_dim=32,
        )

    def test_forward_multi_view_shape(self):
        obs = torch.randn(2, 71)
        views = torch.randn(2, 16, 13)

        scores = self.model(obs, views)
        self.assertEqual(scores.shape, (2, 16))
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))

    def test_ablation_obs_switch(self):
        obs = torch.randn(1, 71)
        views = torch.randn(1, 8, 13)

        scores_full = self.model(obs, views)
        scores_no_obs = self.model(obs, views, ablate_obs=True)

        self.assertEqual(scores_full.shape, (1, 8))
        self.assertEqual(scores_no_obs.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
