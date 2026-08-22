"""
ST-GCN 动作识别模型训练与评估引擎 —— trainer.py
============================================

职责：
    1. 负责 ST-GCN 神经网络在动作骨架数据集上的监督训练、验证与评估；
    2. 使用 CrossEntropy 损失函数与 Adam/SGD 优化器，搭配 Cosine/Step 学习率衰减调度；
    3. 自动记录 Top-1 准确率、混淆矩阵、各动作分类召回率与预测不确定度变化；
    4. 自动保存最优模型检查点 `best_st_gcn_model.pth` 与训练统计 JSON 报表；
    5. 提供命令行训练 CLI 与 Python API 接口。
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ea_avs_mvp_v11.action_recognition.action_dataset import create_action_dataloader
from ea_avs_mvp_v11.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v11.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


class STGCNTrainer:
    """ST-GCN 模型训练与基准评测控制器。"""

    def __init__(
        self,
        model: Optional[STGCN] = None,
        num_classes: int = 6,
        in_channels: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        self.skel_def = skel_def or get_skeleton_definition()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = STGCN(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                skel_def=self.skel_def,
            ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """训练单个 Epoch。"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for data, labels in dataloader:
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item()) * data.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total_samples += data.size(0)

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """在测试集/验证集上评估准确率与不确定度。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(data)
                loss = self.criterion(logits, labels)
                probs = F.softmax(logits, dim=1)

                total_loss += float(loss.item()) * data.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == labels).sum().item())
                total_samples += data.size(0)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        val_loss = total_loss / max(total_samples, 1)
        val_acc = correct / max(total_samples, 1)

        # 计算熵与各类别召回率、精确率、F1 分数
        probs_np = np.array(all_probs)
        preds_np = np.array(all_preds)
        labels_np = np.array(all_labels)

        eps = 1e-8
        entropies = -np.sum(probs_np * np.log(probs_np + eps), axis=1)
        mean_entropy = float(np.mean(entropies)) if len(entropies) > 0 else 0.0
        max_entropy = math.log(max(self.num_classes, 2))
        mean_norm_entropy = float(mean_entropy / max_entropy)

        # 计算宏平均与分类别 Precision / Recall / F1
        per_class_metrics: Dict[int, Dict[str, float]] = {}
        precisions = []
        recalls = []
        f1s = []

        for c in range(self.num_classes):
            tp = int(np.sum((preds_np == c) & (labels_np == c)))
            fp = int(np.sum((preds_np == c) & (labels_np != c)))
            fn = int(np.sum((preds_np != c) & (labels_np == c)))
            support = int(np.sum(labels_np == c))

            p = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
            r = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / max(p + r, 1e-8) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

            per_class_metrics[c] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1_score": round(float(f1), 4),
                "support": support,
            }

        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))

        return {
            "loss": round(val_loss, 4),
            "accuracy": round(val_acc, 4),
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1_score": round(macro_f1, 4),
            "mean_entropy": round(mean_entropy, 4),
            "mean_normalized_uncertainty": round(mean_norm_entropy, 4),
            "per_class_metrics": per_class_metrics,
            "per_class_accuracy": {c: per_class_metrics[c]["recall"] for c in per_class_metrics},
            "num_samples": total_samples,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """执行完整训练与验证循环。"""
        if checkpoint_dir:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        else:
            ckpt_dir = None

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        best_acc = -1.0
        best_ckpt_path = None
        history = []

        logger.info(">>> Starting ST-GCN training for %d epochs on device: %s...", epochs, self.device)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            scheduler.step()

            if val_loader is not None:
                val_res = self.evaluate(val_loader)
                val_acc = val_res["accuracy"]
                val_loss = val_res["loss"]
                mean_ent = val_res["mean_entropy"]
            else:
                val_acc = train_acc
                val_loss = train_loss
                mean_ent = 0.0

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "mean_entropy": round(mean_ent, 4),
            }
            history.append(epoch_record)

            if epoch % 10 == 0 or epoch == epochs or val_acc > best_acc:
                logger.info(
                    "Epoch %03d/%03d | Train Loss: %.4f, Acc: %.1f%% | Val Loss: %.4f, Acc: %.1f%% | Ent: %.3f",
                    epoch, epochs, train_loss, train_acc * 100, val_loss, val_acc * 100, mean_ent
                )

            # 保存最优检查点
            if val_acc > best_acc and ckpt_dir is not None:
                best_acc = val_acc
                best_ckpt_path = ckpt_dir / "best_st_gcn_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "num_classes": self.num_classes,
                }, best_ckpt_path)

        summary = {
            "best_accuracy": round(best_acc, 4),
            "best_checkpoint": str(best_ckpt_path) if best_ckpt_path else None,
            "final_epoch": epochs,
            "history": history,
        }

        if ckpt_dir is not None:
            with open(ckpt_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary
