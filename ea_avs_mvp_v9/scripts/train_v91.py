"""
v9.1 学习型视角打分模型训练脚本 —— train_v91.py
=============================================

功能：
    1. 自动生成多动作、多位姿主动视点训练集与验证集；
    2. 基于 Pairwise Ranking Loss 训练 LearnableViewScorer；
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
from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer
from ea_avs_mvp_v9.training.dataset import generate_scoring_dataset
from ea_avs_mvp_v9.training.trainer import ViewScorerTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_v91")


def main():
    parser = argparse.ArgumentParser(description="Train v9.1 LearnableViewScorer")
    parser.add_argument("--config", type=str, default=None, help="Path to training config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of dataset episodes")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Custom checkpoint directory")
    args = parser.parse_args()

    # 1. 加载训练配置
    cfg_p = Path(args.config) if args.config else (get_repo_root() / "ea_avs_mvp_v9" / "configs" / "v91_training.yaml")
    train_cfg = {}
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
            train_cfg = full_cfg.get("training", {})
            model_cfg = full_cfg.get("model", {})
            paths_cfg = full_cfg.get("paths", {})
    else:
        model_cfg = {}
        paths_cfg = {}

    epochs = args.epochs or int(train_cfg.get("num_epochs", 40))
    lr = args.lr or float(train_cfg.get("learning_rate", 0.001))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 16))
    num_episodes = args.num_episodes or int(train_cfg.get("num_episodes", 200))
    seed = int(train_cfg.get("seed", 42))

    # 2. 路径配置
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir) if Path(args.checkpoint_dir).is_absolute() else get_data_root() / args.checkpoint_dir
    else:
        ckpt_dir = get_data_root() / paths_cfg.get("checkpoint_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / paths_cfg.get("checkpoint_name", "model_checkpoint.pth")

    res_dir = get_data_root() / paths_cfg.get("results_dir", "results")
    res_dir.mkdir(parents=True, exist_ok=True)
    curve_path = res_dir / paths_cfg.get("training_curve_name", "training_curve.png")

    logger.info("Generating dataset with %d episodes (seed=%d)...", num_episodes, seed)
    train_dataset, val_dataset = generate_scoring_dataset(num_episodes=num_episodes, seed=seed)
    logger.info("Dataset split: Train=%d samples, Val=%d samples", len(train_dataset), len(val_dataset))

    # 3. 初始化模型与训练器 (Q(v | H))
    model = LearnableViewScorer(
        pose_input_dim=int(model_cfg.get("pose_input_dim", 49)),
        pose_embed_dim=int(model_cfg.get("pose_embed_dim", 32)),
        view_input_dim=int(model_cfg.get("view_input_dim", 13)),
        view_embed_dim=int(model_cfg.get("view_embed_dim", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    trainer = ViewScorerTrainer(
        model=model,
        config={
            "learning_rate": lr,
            "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
            "ranking_margin": float(train_cfg.get("ranking_margin", 0.1)),
            "ranking_loss_weight": float(train_cfg.get("ranking_loss_weight", 1.0)),
            "regression_loss_weight": float(train_cfg.get("regression_loss_weight", 0.5)),
        },
    )

    # 4. 执行训练流程
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=ckpt_path,
        curve_path=curve_path,
    )

    print("\n" + "=" * 65)
    print("  ACTIVEVIEW v9.1 Model Training Summary")
    print("=" * 65)
    print(f"Total Epochs:          {epochs}")
    print(f"Best Val Top-1 Acc:    {results['best_top1_acc'] * 100:.1f}% (Epoch {results['best_epoch']})")
    print(f"Final Val Loss:        {results['final_val_metrics']['val_loss']:.4f}")
    print(f"Target Score Ratio:    {results['final_val_metrics']['score_ratio'] * 100:.1f}%")
    print(f"Saved Checkpoint:      {ckpt_path}")
    print(f"Training Curve:        {curve_path}")
    print("=" * 65)
    print("PASS:\nACTIVEVIEW v9.1 Model Training Complete\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
