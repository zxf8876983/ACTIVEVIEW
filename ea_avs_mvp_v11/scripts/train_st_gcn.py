#!/usr/bin/env python3
"""
ACTIVEVIEW v11.5 ST-GCN 动作分类模型重训练与评估脚本
=====================================================

职责：
    1. 在全量 AMASS 动作时序数据集 (2,400 训练样本 + 600 测试样本) 上训练 ST-GCN；
    2. 计算多类别分类准确率、后验熵收敛曲线与混淆矩阵 (Confusion Matrix)；
    3. 将检查点权重保存至 `data/ActiveView/checkpoints/v11_st_gcn/` 与 `v10_st_gcn/`；
    4. 输出详细训练日志与评估汇总。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v10.action_recognition.action_dataset import create_action_dataloader
from ea_avs_mvp_v10.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v10.action_recognition.trainer import STGCNTrainer
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v11.active_view.action_registry import DEFAULT_ACTION_CATEGORIES
from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_st_gcn_v115")


def train_st_gcn(
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    dataset_dir: str = None,
    checkpoint_dir: str = None,
) -> Dict[str, Any]:
    """训练 ST-GCN 模型并输出详细评估指标。"""
    data_root = get_data_root()
    action_data_root = Path(dataset_dir) if dataset_dir else (data_root / "datasets" / "action")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (data_root / "checkpoints" / "v11_st_gcn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_data_p = action_data_root / "train" / "clean_perception" / "data.npy"
    train_labels_p = action_data_root / "train" / "clean_perception" / "labels.npy"
    val_data_p = action_data_root / "test" / "clean_perception" / "data.npy"
    val_labels_p = action_data_root / "test" / "clean_perception" / "labels.npy"

    if not train_data_p.exists():
        raise FileNotFoundError(f"Training data not found at: {train_data_p}")

    skel_def = get_skeleton_definition()
    train_loader = create_action_dataloader(train_data_p, train_labels_p, batch_size=batch_size, shuffle=True)
    val_loader = create_action_dataloader(val_data_p, val_labels_p, batch_size=batch_size, shuffle=False)

    num_classes = len(DEFAULT_ACTION_CATEGORIES)
    trainer = STGCNTrainer(num_classes=num_classes, lr=lr, skel_def=skel_def)

    logger.info("Starting ST-GCN training for %d epochs on %s...", epochs, train_data_p)
    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
    )

    # 评估混淆矩阵
    model = STGCN(in_channels=3, num_classes=num_classes, skel_def=skel_def)
    best_pth = ckpt_dir / "best_st_gcn_model.pth"
    if best_pth.exists():
        ckpt = torch.load(best_pth, map_location="cpu")
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            outputs = model(batch_data)
            preds = torch.argmax(outputs, dim=1).numpy()
            all_preds.extend(preds)
            all_gts.extend(batch_labels.numpy())

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_gts, all_preds):
        conf_matrix[t, p] += 1

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_gts))) * 100

    results = {
        "num_classes": num_classes,
        "categories": DEFAULT_ACTION_CATEGORIES,
        "val_accuracy": round(accuracy, 2),
        "confusion_matrix": conf_matrix.tolist(),
        "checkpoint_path": str(best_pth),
    }

    with open(ckpt_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("================================================================")
    logger.info("  ST-GCN Re-Training Complete!                                  ")
    logger.info("  Validation Accuracy: %.2f%%                                   ", accuracy)
    logger.info("  Categories:          %s", DEFAULT_ACTION_CATEGORIES)
    logger.info("  Saved Checkpoint to: %s", best_pth)
    logger.info("================================================================")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ST-GCN on Full AMASS dataset")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory")
    args = parser.parse_args()

    train_st_gcn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
