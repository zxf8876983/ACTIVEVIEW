"""
模型训练器与评估调度器 —— trainer.py
====================================

职责：
    1. 管理 LearnableViewScorer 的训练生命周期 (优化器、损失函数、学习率调整)；
    2. 计算关键科研评测指标 (Top-1 选点准确率、Pairwise 排序精度、验证集损失)；
    3. 自动保存模型检查点 (model_checkpoint.pth) 与收敛曲线 (training_curve.png)。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ea_avs_mvp_v9.models.view_scorer import LearnableViewScorer
from .dataset import ActiveViewScoringDataset
from .losses import CombinedRankingRegressionLoss

logger = logging.getLogger(__name__)


class ViewScorerTrainer:
    """学习型视角打分模型训练器。"""

    def __init__(
        self,
        model: LearnableViewScorer,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.config = config or {}
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)

        lr = float(self.config.get("learning_rate", 0.001))
        wd = float(self.config.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        margin = float(self.config.get("ranking_margin", 0.1))
        w_rank = float(self.config.get("ranking_loss_weight", 1.0))
        w_reg = float(self.config.get("regression_loss_weight", 0.5))
        self.criterion = CombinedRankingRegressionLoss(
            margin=margin,
            ranking_weight=w_rank,
            regression_weight=w_reg,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_top1_acc": [],
            "val_score_ratio": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """单轮训练。"""
        self.model.train()
        total_loss = 0.0
        batches = 0

        for batch in dataloader:
            pose = batch["pose_vec"].to(self.device)        # (B, 49)
            action = batch["action_vec"].to(self.device)    # (B, 5)
            views = batch["view_vecs"].to(self.device)      # (B, N, 11)
            targets = batch["target_scores"].to(self.device)# (B, N)

            self.optimizer.zero_grad()
            preds = self.model(pose, action, views)         # (B, N)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        return total_loss / max(1, batches)

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证集评估，计算损失与 Top-1 准确率。"""
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        total_episodes = 0
        score_ratios = []

        with torch.no_grad():
            for batch in dataloader:
                pose = batch["pose_vec"].to(self.device)
                action = batch["action_vec"].to(self.device)
                views = batch["view_vecs"].to(self.device)
                targets = batch["target_scores"].to(self.device)
                b_size = pose.size(0)

                preds = self.model(pose, action, views)
                loss = self.criterion(preds, targets)
                total_loss += float(loss.item()) * b_size

                # Top-1 视点匹配度
                pred_best_idx = torch.argmax(preds, dim=-1)     # (B,)
                target_best_idx = torch.argmax(targets, dim=-1) # (B,)

                matched = (pred_best_idx == target_best_idx).sum().item()
                correct_top1 += matched
                total_episodes += b_size

                # 预测最优视点所获取的真实效用比例
                for b_i in range(b_size):
                    p_idx = pred_best_idx[b_i].item()
                    t_max = targets[b_i, target_best_idx[b_i]].item()
                    t_pred = targets[b_i, p_idx].item()
                    if t_max > 1e-4:
                        score_ratios.append(t_pred / t_max)
                    else:
                        score_ratios.append(1.0)

        mean_loss = total_loss / max(1, total_episodes)
        top1_acc = float(correct_top1 / max(1, total_episodes))
        mean_ratio = float(np.mean(score_ratios)) if score_ratios else 0.0

        return {
            "val_loss": mean_loss,
            "top1_acc": top1_acc,
            "score_ratio": mean_ratio,
        }

    def train(
        self,
        train_dataset: ActiveViewScoringDataset,
        val_dataset: ActiveViewScoringDataset,
        num_epochs: int = 40,
        batch_size: int = 16,
        checkpoint_path: Optional[Union[str, Path]] = None,
        curve_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """完整训练流程。"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_top1_acc = -1.0
        best_epoch = -1

        logger.info("Starting training LearnableViewScorer for %d epochs on device '%s'...", num_epochs, self.device)

        for epoch in range(1, num_epochs + 1):
            t_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_top1_acc"].append(val_metrics["top1_acc"])
            self.history["val_score_ratio"].append(val_metrics["score_ratio"])

            if val_metrics["top1_acc"] > best_top1_acc:
                best_top1_acc = val_metrics["top1_acc"]
                best_epoch = epoch
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path, epoch=epoch, metrics=val_metrics)

            if epoch % 5 == 0 or epoch == num_epochs:
                logger.info(
                    "Epoch [%02d/%02d] - Train Loss: %.4f | Val Loss: %.4f | Top-1 Acc: %.1f%% | Target Ratio: %.3f",
                    epoch, num_epochs, t_loss, val_metrics["val_loss"], val_metrics["top1_acc"] * 100, val_metrics["score_ratio"]
                )

        logger.info("Training completed. Best Val Top-1 Acc: %.1f%% (Epoch %d)", best_top1_acc * 100, best_epoch)

        if curve_path:
            self.plot_training_curve(curve_path)

        return {
            "best_top1_acc": best_top1_acc,
            "best_epoch": best_epoch,
            "final_val_metrics": val_metrics,
            "history": self.history,
        }

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """保存模型检查点。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "metrics": metrics or {},
        }, p)
        return p

    def plot_training_curve(self, output_path: Union[str, Path]) -> Optional[Path]:
        """绘制训练与验证损失及 Top-1 准确率收敛曲线。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping training curve plot")
            return None

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        epochs = list(range(1, len(self.history["train_loss"]) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", color="#4A90E2", linewidth=2.0)
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss", color="#E94E77", linewidth=2.0)
        ax1.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Top-1 准确率曲线
        ax2.plot(epochs, [acc * 100 for acc in self.history["val_top1_acc"]], label="Val Top-1 Accuracy (%)", color="#50E3C2", linewidth=2.0)
        ax2.plot(epochs, [r * 100 for r in self.history["val_score_ratio"]], label="Val Target Score Ratio (%)", color="#F5A623", linewidth=2.0, linestyle="--")
        ax2.set_title("Top-1 Selection Accuracy & Utility Ratio", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, 105)
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_p, dpi=150)
        plt.close(fig)
        logger.info("Training curve successfully saved to: %s", out_p)
        return out_p
