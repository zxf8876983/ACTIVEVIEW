#!/usr/bin/env python3
"""
ST-GCN 动作分类模型训练脚本 —— train_st_gcn.py
============================================

职责：
    1. 加载 Clean Perception 训练集 (train/clean_perception) 与验证集 (test/clean_perception)；
    2. 训练 ST-GCN 网络 (9 块时空图卷积残差层 + 33 关节空间划分邻接矩阵)；
    3. 保存最优检查点到 `/home/zxf/WorkSpace/code/data/ActiveView/checkpoints/v10_st_gcn/best_st_gcn_model.pth`；
    4. 记录训练损失、验证准确率与信息熵收敛曲线。
"""

import argparse
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v7.core.paths import get_data_root
from ea_avs_mvp_v10.action_recognition.action_dataset import create_action_dataloader
from ea_avs_mvp_v10.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v10.action_recognition.trainer import STGCNTrainer
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_st_gcn")


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN on Action Perception Dataset")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint output directory")
    args = parser.parse_args()

    action_data_root = Path(args.dataset_dir) if args.dataset_dir else (get_data_root() / "datasets" / "action")
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (get_data_root() / "checkpoints" / "v10_st_gcn")

    train_data_p = action_data_root / "train" / "clean_perception" / "data.npy"
    train_labels_p = action_data_root / "train" / "clean_perception" / "labels.npy"
    val_data_p = action_data_root / "test" / "clean_perception" / "data.npy"
    val_labels_p = action_data_root / "test" / "clean_perception" / "labels.npy"

    skel_def = get_skeleton_definition()
    train_loader = create_action_dataloader(train_data_p, train_labels_p, batch_size=args.batch_size, shuffle=True)
    val_loader = create_action_dataloader(val_data_p, val_labels_p, batch_size=args.batch_size, shuffle=False)

    trainer = STGCNTrainer(num_classes=6, lr=args.lr, skel_def=skel_def)
    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=ckpt_dir,
    )

    logger.info("ST-GCN training complete. Best Val Accuracy: %.2f%%", summary["best_accuracy"] * 100)


if __name__ == "__main__":
    main()
