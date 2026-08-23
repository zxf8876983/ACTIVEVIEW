#!/usr/bin/env python3
"""
ST-GCN Action Recognition Model Trainer —— train_st_gcn_v11_5.py (v11.5)
========================================================================

职责：
    1. 加载 Clean Perception 训练集 (clean_perception_v11_5/train) 与验证集 (val)；
    2. 使用 VideoPose3D 估计骨架 (Human3.6M 17-joint, K=3 空间划分) 训练 ST-GCN；
    3. 严禁使用 AMASS GT 骨架，确保感知一致性；
    4. 训练完成后保存最优权重至 /home/zxf/WorkSpace/code/data/ActiveView/checkpoints/v11_5/stgcn_v11_5_best.pth；
    5. 测试阶段严格冻结该权重，用于后续 Habitat 主动视角评测与不确定度计算。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 保证包路径正确
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_st_gcn_v11_5")

ACTION_CATEGORIES = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]


def load_dataset(dataset_dir: Path) -> Tuple[TensorDataset, TensorDataset]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    train_data = np.load(train_dir / "data.npy")      # (N_tr, 3, 30, 17, 1)
    train_labels = np.load(train_dir / "labels.npy")  # (N_tr,)
    val_data = np.load(val_dir / "data.npy")          # (N_val, 3, 30, 17, 1)
    val_labels = np.load(val_dir / "labels.npy")      # (N_val,)

    logger.info("Loaded Train Data: shape=%s, labels=%s", train_data.shape, train_labels.shape)
    logger.info("Loaded Val Data:   shape=%s, labels=%s", val_data.shape, val_labels.shape)

    train_ds = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long())
    val_ds = TensorDataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_labels).long())
    return train_ds, val_ds


def calculate_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算 Shannon 信息熵 H(p) = -sum p_i log(p_i)."""
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)


def train_st_gcn_v11_5(
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_str: str = "cuda:0",
) -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Training ST-GCN on device: %s...", device)

    data_root = get_data_root()
    dataset_dir = data_root / "datasets" / "clean_perception_v11_5"
    ckpt_dir = data_root / "checkpoints" / "v11_5"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_dataset(dataset_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    skel_def = get_skeleton_definition(backend="h36m_17")
    num_classes = len(ACTION_CATEGORIES)

    model = STGCN(
        in_channels=3,
        num_classes=num_classes,
        graph_args={"strategy": "spatial", "max_hop": 1},
        edge_importance_weighting=True,
        skel_def=skel_def,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_ckpt_path = ckpt_dir / "stgcn_v11_5_best.pth"
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_entropy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_b.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y_b).sum().item()
            train_total += x_b.size(0)

        scheduler.step()
        train_loss = total_train_loss / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_entropies = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                probs = torch.softmax(logits, dim=-1)
                entropy = calculate_entropy(probs)

                total_val_loss += loss.item() * x_b.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_b).sum().item()
                val_total += x_b.size(0)

                all_entropies.extend(entropy.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y_b.cpu().numpy().tolist())

        val_loss = total_val_loss / val_total
        val_acc = val_correct / val_total
        mean_entropy = float(np.mean(all_entropies))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_entropy"].append(mean_entropy)

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "skel_backend": "h36m_17",
                "categories": ACTION_CATEGORIES,
            }, best_ckpt_path)
            logger.info(">>> [Epoch %02d/%02d] New Best Checkpoint Saved! Val Acc: %.2f%% | Entropy: %.4f",
                        epoch, epochs, val_acc * 100, mean_entropy)

        if epoch % 10 == 0 or epoch == epochs:
            logger.info("Epoch [%02d/%02d] - Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%, Ent: %.4f",
                        epoch, epochs, train_loss, train_acc * 100, val_loss, val_acc * 100, mean_entropy)

    logger.info("==========================================================")
    logger.info("  ST-GCN Training Finished!")
    logger.info("  Best Validation Accuracy: %.2f%%", best_val_acc * 100)
    logger.info("  Best Checkpoint Saved To: %s", best_ckpt_path)
    logger.info("==========================================================")

    return {
        "best_val_acc": best_val_acc,
        "best_ckpt_path": str(best_ckpt_path),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_st_gcn_v11_5(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
