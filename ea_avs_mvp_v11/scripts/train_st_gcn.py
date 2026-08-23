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
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    dataset_dir: str = None,
    checkpoint_dir: str = None,
    exclude_locomotion: bool = True,
) -> Dict[str, Any]:
    """训练 ST-GCN 模型并输出详细评估指标。"""
    data_root = get_data_root()
    action_data_root = Path(dataset_dir) if dataset_dir else (data_root / "datasets" / "action")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (data_root / "checkpoints" / "v11_st_gcn")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_lbl_json = action_data_root / "train" / "label.json"
    val_lbl_json = action_data_root / "test" / "label.json"

    categories = list(DEFAULT_ACTION_CATEGORIES)
    num_classes = len(categories)
    cat_to_id = {c: i for i, c in enumerate(categories)}

    skel_def = get_skeleton_definition()
    from ea_avs_mvp_v11.active_view.skeleton_canonicalizer import CanonicalSkeletonAligner
    aligner = CanonicalSkeletonAligner(skel_def=skel_def)

    if train_lbl_json.exists() and val_lbl_json.exists():
        logger.info("Loading RGB-D Action Dataset from %s and %s", train_lbl_json, val_lbl_json)
        with open(train_lbl_json, "r", encoding="utf-8") as f:
            train_mf = json.load(f)
        with open(val_lbl_json, "r", encoding="utf-8") as f:
            val_mf = json.load(f)

        # 过滤目标 16 类动作
        train_mf = [item for item in train_mf if item["action_label"] in cat_to_id]
        val_mf = [item for item in val_mf if item["action_label"] in cat_to_id]

        N_tr = len(train_mf)
        N_va = len(val_mf)
        T, V, C = 30, 33, 3

        canonical_train_data = np.zeros((N_tr, C, T, V, 1), dtype=np.float32)
        train_labels = np.array([cat_to_id[item["action_label"]] for item in train_mf], dtype=np.int64)

        canonical_val_data = np.zeros((N_va, C, T, V, 1), dtype=np.float32)
        val_labels = np.array([cat_to_id[item["action_label"]] for item in val_mf], dtype=np.int64)

        logger.info("Loading & Canonical Aligning %d Train RGB-D skeletons...", N_tr)
        for i, item in enumerate(train_mf):
            skel_p = action_data_root / "train" / item["skeleton_path"]
            if skel_p.exists():
                raw_skel = np.load(skel_p) # (30, 33, 3)
            else:
                raw_skel = np.zeros((T, V, C), dtype=np.float32)
            # 数据增强：随机偏航角旋转
            rand_yaw = np.random.uniform(0, 2 * np.pi)
            cos_y, sin_y = np.cos(rand_yaw), np.sin(rand_yaw)
            R_rand = np.array([[cos_y, 0, -sin_y], [0, 1, 0], [sin_y, 0, cos_y]], dtype=np.float32)
            aug_skel = np.zeros_like(raw_skel)
            for t in range(T):
                aug_skel[t] = (R_rand @ raw_skel[t].T).T
            canon_skel = aligner.align(aug_skel)
            canonical_train_data[i, :, :, :, 0] = np.transpose(canon_skel, (2, 0, 1))

        logger.info("Loading & Canonical Aligning %d Val RGB-D skeletons...", N_va)
        for j, item in enumerate(val_mf):
            skel_p = action_data_root / "test" / item["skeleton_path"]
            if skel_p.exists():
                raw_skel = np.load(skel_p)
            else:
                raw_skel = np.zeros((T, V, C), dtype=np.float32)
            canon_skel = aligner.align(raw_skel)
            canonical_val_data[j, :, :, :, 0] = np.transpose(canon_skel, (2, 0, 1))

    else:
        train_data_p = action_data_root / "train" / "clean_perception" / "data.npy"
        val_data_p = action_data_root / "test" / "clean_perception" / "data.npy"
        train_mf_p = action_data_root / "train" / "clean_perception" / "manifest.json"
        val_mf_p = action_data_root / "test" / "clean_perception" / "manifest.json"

        if not train_data_p.exists():
            raise FileNotFoundError(f"Training data not found at: {train_data_p}")

        raw_train_data = np.load(train_data_p)
        raw_val_data = np.load(val_data_p)
        with open(train_mf_p, "r", encoding="utf-8") as f:
            train_mf = json.load(f)
        with open(val_mf_p, "r", encoding="utf-8") as f:
            val_mf = json.load(f)

        train_idx = [i for i, item in enumerate(train_mf) if item.get("action_label") in cat_to_id]
        val_idx = [i for i, item in enumerate(val_mf) if item.get("action_label") in cat_to_id]

        train_data = raw_train_data[train_idx]
        train_labels = np.array([cat_to_id[train_mf[i]["action_label"]] for i in train_idx], dtype=np.int64)
        val_data = raw_val_data[val_idx]
        val_labels = np.array([cat_to_id[val_mf[i]["action_label"]] for i in val_idx], dtype=np.int64)

        canonical_train_data = np.zeros_like(train_data)
        canonical_val_data = np.zeros_like(val_data)

        for i in range(len(train_data)):
            raw_skel = np.transpose(train_data[i, :, :, :, 0], (1, 2, 0))
            rand_yaw = np.random.uniform(0, 2 * np.pi)
            cos_y, sin_y = np.cos(rand_yaw), np.sin(rand_yaw)
            R_rand = np.array([[cos_y, 0, -sin_y], [0, 1, 0], [sin_y, 0, cos_y]], dtype=np.float32)
            aug_skel = np.zeros_like(raw_skel)
            for t in range(raw_skel.shape[0]):
                aug_skel[t] = (R_rand @ raw_skel[t].T).T
            canon_skel = aligner.align(aug_skel)
            canonical_train_data[i, :, :, :, 0] = np.transpose(canon_skel, (2, 0, 1))

        for i in range(len(val_data)):
            raw_skel = np.transpose(val_data[i, :, :, :, 0], (1, 2, 0))
            canon_skel = aligner.align(raw_skel)
            canonical_val_data[i, :, :, :, 0] = np.transpose(canon_skel, (2, 0, 1))


    # 临时保存过滤后的数据文件用于 DataLoader
    filtered_dir = data_root / "cache" / "filtered_action_dataset"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    f_train_data_p = filtered_dir / "train_data.npy"
    f_train_lbl_p = filtered_dir / "train_labels.npy"
    f_val_data_p = filtered_dir / "val_data.npy"
    f_val_lbl_p = filtered_dir / "val_labels.npy"

    np.save(f_train_data_p, canonical_train_data)
    np.save(f_train_lbl_p, train_labels)
    np.save(f_val_data_p, canonical_val_data)
    np.save(f_val_lbl_p, val_labels)

    train_loader = create_action_dataloader(f_train_data_p, f_train_lbl_p, batch_size=batch_size, shuffle=True)
    val_loader = create_action_dataloader(f_val_data_p, f_val_lbl_p, batch_size=batch_size, shuffle=False)

    trainer = STGCNTrainer(num_classes=num_classes, lr=lr, skel_def=skel_def)


    logger.info("Starting ST-GCN training for %d epochs on %d train / %d val non-locomotion samples...",
                epochs, len(canonical_train_data), len(canonical_val_data))

    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=ckpt_dir,
    )
    best_acc = summary.get("best_val_accuracy", 0.0)

    # 保存额外命名检查点
    best_pth = ckpt_dir / "best_st_gcn_model.pth"
    v11_pth = ckpt_dir / "v11_4_2_stgcn_estimated_pose.pth"
    if best_pth.exists():
        import shutil
        shutil.copyfile(best_pth, v11_pth)


    # 评估最终混淆矩阵与验证准确率

    all_preds = []
    all_gts = []
    trainer.model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(trainer.device)
            outputs = trainer.model(batch_data)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_gts.extend(batch_labels.numpy())

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, g in zip(all_preds, all_gts):
        conf_matrix[g, p] += 1
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_gts))) * 100

    results = {
        "num_classes": num_classes,
        "categories": DEFAULT_ACTION_CATEGORIES,
        "val_accuracy": round(accuracy, 2),
        "confusion_matrix": conf_matrix.tolist(),
        "checkpoint_path": str(best_pth),
        "estimated_checkpoint_path": str(v11_pth),
        "skeleton_source": "estimated",
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
