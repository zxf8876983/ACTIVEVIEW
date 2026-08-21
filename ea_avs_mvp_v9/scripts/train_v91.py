"""
v9.1 感知驱动主动视角打分模型训练脚本 —— train_v91.py
======================================================

功能：
    1. 自动生成多位姿、空间隔离的当前不完整感知状态与信息增益训练集与验证集；
    2. 基于 Pairwise Ranking Loss 训练 PerceptionAwareViewScorer (G(v | O_curr))；
    3. 保存模型检查点至 data/ActiveView/checkpoints/model_checkpoint.pth；
    4. 生成训练收敛曲线至 data/ActiveView/results/training_curve.png。

运行方式：
    python -m ea_avs_mvp_v9.scripts.train_v91
    python -m ea_avs_mvp_v9.scripts.train_v91 --epochs 40 --lr 0.001
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

from ea_avs_mvp_v9.core.paths import get_data_root, get_repo_root
from ea_avs_mvp_v9.models.view_scorer import PerceptionAwareViewScorer
from ea_avs_mvp_v9.training.dataset import generate_scoring_dataset
from ea_avs_mvp_v9.training.trainer import ViewScorerTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_v91")


def main():
    parser = argparse.ArgumentParser(description="Train v9.1 PerceptionAwareViewScorer")
    parser.add_argument("--config", type=str, default=None, help="Path to training config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of dataset episodes")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Custom checkpoint directory")
    args = parser.parse_args()

    repo_root = get_repo_root()
    default_cfg_path = repo_root / "ea_avs_mvp_v9/configs/v91_training.yaml"
    cfg = {}
    if default_cfg_path.exists():
        with open(default_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    epochs = args.epochs or cfg.get("training", {}).get("epochs", 40)
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 0.001)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 16)
    num_episodes = args.num_episodes or cfg.get("training", {}).get("num_episodes", 200)

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (get_data_root() / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model_checkpoint.pth"

    res_dir = get_data_root() / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    curve_path = res_dir / "training_curve.png"

    logger.info("Generating dataset with %d episodes...", num_episodes)
    train_ds, val_ds = generate_scoring_dataset(num_episodes=num_episodes, seed=42)
    logger.info("Dataset split: Train=%d samples, Val=%d samples", len(train_ds), len(val_ds))

    model = PerceptionAwareViewScorer(
        obs_input_dim=71,
        obs_embed_dim=32,
        view_input_dim=13,
        view_embed_dim=32,
        fusion_hidden_dims=(64, 32),
        dropout=0.1,
    )

    trainer_cfg = {
        "learning_rate": lr,
        "weight_decay": cfg.get("training", {}).get("weight_decay", 1e-4),
        "ranking_margin": cfg.get("training", {}).get("ranking_margin", 0.1),
        "ranking_loss_weight": cfg.get("training", {}).get("ranking_loss_weight", 1.0),
        "regression_loss_weight": cfg.get("training", {}).get("regression_loss_weight", 0.5),
    }
    trainer = ViewScorerTrainer(model=model, config=trainer_cfg)

    results = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=ckpt_path,
        curve_path=curve_path,
    )

    print("\n" + "=" * 65)
    print("  ACTIVEVIEW v9.1 Model Training Summary (Perception-Aware)")
    print("=" * 65)
    print(f"Total Epochs:          {epochs}")
    print(f"Best Val Top-1 Acc:    {results['best_top1_acc'] * 100:.1f}% (Epoch {results['best_epoch']})")
    print(f"Final Val Loss:        {results['final_val_metrics']['val_loss']:.4f}")
    print(f"Target Gain Ratio:     {results['final_val_metrics']['score_ratio'] * 100:.1f}%")
    print(f"Saved Checkpoint:      {ckpt_path}")
    print(f"Training Curve:        {curve_path}")
    print("=" * 65)
    print("PASS:\nACTIVEVIEW v9.1 Perception-aware Model Training Complete\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
