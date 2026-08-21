import unittest
from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer
from ea_avs_mvp_v9.training.dataset import generate_scoring_dataset
from ea_avs_mvp_v9.training.losses import CombinedRankingRegressionLoss, PairwiseRankingLoss
from ea_avs_mvp_v9.training.trainer import ViewScorerTrainer
import torch


class TestV91Trainer(unittest.TestCase):
    def setUp(self):
        self.model = LearnableViewScorer(
            pose_input_dim=49,
            pose_embed_dim=32,
            view_input_dim=13,
            view_embed_dim=32,
        )
        self.train_ds, self.val_ds = generate_scoring_dataset(num_episodes=10, seed=42)
        self.trainer = ViewScorerTrainer(self.model, config={"learning_rate": 0.01})

    def test_pairwise_ranking_loss_reduction(self):
        loss_fn = PairwiseRankingLoss(margin=0.1)
        preds = torch.tensor([[0.8, 0.4], [0.2, 0.9]])
        targets = torch.tensor([[0.9, 0.3], [0.1, 0.8]])
        loss = loss_fn(preds, targets)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_trainer_single_epoch_step(self):
        loader = torch.utils.data.DataLoader(self.train_ds, batch_size=4)
        t_loss = self.trainer.train_epoch(loader)
        self.assertGreater(t_loss, 0.0)

    def test_trainer_validation_metrics(self):
        val_loader = torch.utils.data.DataLoader(self.val_ds, batch_size=4)
        metrics = self.trainer.validate(val_loader)
        self.assertIn("val_loss", metrics)
        self.assertIn("top1_acc", metrics)
        self.assertIn("score_ratio", metrics)
        self.assertGreaterEqual(metrics["top1_acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
