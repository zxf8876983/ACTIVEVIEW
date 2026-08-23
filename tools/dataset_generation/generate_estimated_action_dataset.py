#!/usr/bin/env python3
"""
Estimated 3D Action Dataset Generator —— generate_estimated_action_dataset.py (v11.4.1)
====================================================================================

职责：
    1. 通过 Pose Estimator 提取全量 16 类非位移动作估计骨架，彻底替代旧的真值骨架；
    2. 生成 Train 训练集 (N=3,200 条) 与 Test 测试集 (N=800 条)；
    3. 严格执行 Motion Instance-Level 隔离 (Train IDs ∩ Test IDs = ∅)；
    4. 输出包含 `"source": "estimated"` 的 manifest 与 `split_statistics.json`。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.action_registry import DEFAULT_ACTION_CATEGORIES
from ea_avs_mvp_v11.active_view.skeleton_canonicalizer import CanonicalSkeletonAligner
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.perception.pose_estimator import get_pose_estimator
from tools.dataset_generation.generate_16class_amass_dataset import synthesize_canonical_motion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_estimated_action_dataset")


def build_estimated_action_dataset() -> Dict[str, Any]:
    data_root = get_data_root()
    action_dir = data_root / "datasets" / "action"
    train_dir = action_dir / "train" / "perception"
    test_dir = action_dir / "test" / "perception"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    pose_estimator = get_pose_estimator()
    aligner = CanonicalSkeletonAligner()

    num_train_per_cat = 200
    num_test_per_cat = 50
    total_train = len(DEFAULT_ACTION_CATEGORIES) * num_train_per_cat
    total_test = len(DEFAULT_ACTION_CATEGORIES) * num_test_per_cat

    train_data = np.zeros((total_train, 3, 30, 33, 1), dtype=np.float32)
    train_labels = np.zeros((total_train,), dtype=np.int64)
    train_manifest = []

    test_data = np.zeros((total_test, 3, 30, 33, 1), dtype=np.float32)
    test_labels = np.zeros((total_test,), dtype=np.int64)
    test_manifest = []

    train_motion_ids = set()
    test_motion_ids = set()

    logger.info("Generating Estimated 3D Action Dataset across %d categories (Train=%d, Test=%d)...",
                len(DEFAULT_ACTION_CATEGORIES), total_train, total_test)

    # 1. 生成训练集 (由 Pose 估计器提取 + 随机视角偏航增强)
    train_idx = 0
    for cat_id, cat_name in enumerate(DEFAULT_ACTION_CATEGORIES):
        for i in range(num_train_per_cat):
            m_id = f"motion_{cat_name}_tr_{i:04d}"
            train_motion_ids.add(m_id)

            base_motion = synthesize_canonical_motion(cat_name, seed=cat_id * 10000 + i)
            rand_yaw = float(np.random.uniform(0.0, 360.0))
            rand_dist = float(np.random.uniform(1.8, 3.2))

            # 模拟相机图像与姿态估计器提取 (Clean / Mild Occlusion)
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            est_skel, conf, meta = pose_estimator.estimate(
                rgb=rgb,
                angle_deg=rand_yaw,
                distance_m=rand_dist,
                occlusion_ratio=0.0,
                base_motion_seq=base_motion,
            )
            assert meta["skeleton_source"] == "estimated"

            # Canonical 对齐
            canon_skel = aligner.align(est_skel)
            train_data[train_idx] = np.transpose(canon_skel, (2, 0, 1))[..., np.newaxis]
            train_labels[train_idx] = cat_id

            train_manifest.append({
                "sample_id": f"train_{train_idx:05d}",
                "motion_id": m_id,
                "action_id": cat_id,
                "action_label": cat_name,
                "split": "train",
                "source": "estimated",
                "confidence": conf,
                "yaw_deg": rand_yaw,
                "distance_m": rand_dist,
                "sequence_length": 30,
                "joint_num": 33,
            })
            train_idx += 1

    # 2. 生成测试集 (独立 Motion IDs)
    test_idx = 0
    for cat_id, cat_name in enumerate(DEFAULT_ACTION_CATEGORIES):
        for i in range(num_test_per_cat):
            m_id = f"motion_{cat_name}_ts_{i:04d}"
            test_motion_ids.add(m_id)

            base_motion = synthesize_canonical_motion(cat_name, seed=cat_id * 20000 + 5000 + i)
            rand_yaw = float(np.random.uniform(0.0, 360.0))
            rand_dist = float(np.random.uniform(1.8, 3.2))

            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            est_skel, conf, meta = pose_estimator.estimate(
                rgb=rgb,
                angle_deg=rand_yaw,
                distance_m=rand_dist,
                occlusion_ratio=0.0,
                base_motion_seq=base_motion,
            )
            assert meta["skeleton_source"] == "estimated"

            canon_skel = aligner.align(est_skel)
            test_data[test_idx] = np.transpose(canon_skel, (2, 0, 1))[..., np.newaxis]
            test_labels[test_idx] = cat_id

            test_manifest.append({
                "sample_id": f"test_{test_idx:05d}",
                "motion_id": m_id,
                "action_id": cat_id,
                "action_label": cat_name,
                "split": "test",
                "source": "estimated",
                "confidence": conf,
                "yaw_deg": rand_yaw,
                "distance_m": rand_dist,
                "sequence_length": 30,
                "joint_num": 33,
            })
            test_idx += 1

    # 检查隔离性
    overlap = train_motion_ids.intersection(test_motion_ids)
    assert len(overlap) == 0, f"Error: Train and Test motion overlap detected: {overlap}"

    # 保存主文件
    np.save(train_dir / "data.npy", train_data)
    np.save(train_dir / "labels.npy", train_labels)
    with open(train_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)

    np.save(test_dir / "data.npy", test_data)
    np.save(test_dir / "labels.npy", test_labels)
    with open(test_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    # 兼容性保存：clean_perception 路径
    clean_tr_dir = action_dir / "train" / "clean_perception"
    clean_ts_dir = action_dir / "test" / "clean_perception"
    clean_tr_dir.mkdir(parents=True, exist_ok=True)
    clean_ts_dir.mkdir(parents=True, exist_ok=True)
    np.save(clean_tr_dir / "data.npy", train_data)
    np.save(clean_tr_dir / "labels.npy", train_labels)
    with open(clean_tr_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)
    np.save(clean_ts_dir / "data.npy", test_data)
    np.save(clean_ts_dir / "labels.npy", test_labels)
    with open(clean_ts_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    # 导出划分统计
    split_stats = {
        "dataset_name": "ACTIVEVIEW v11.4.1 Estimated 3D Pose Dataset",
        "num_categories": len(DEFAULT_ACTION_CATEGORIES),
        "categories": DEFAULT_ACTION_CATEGORIES,
        "total_train_samples": total_train,
        "total_test_samples": total_test,
        "train_motion_ids_count": len(train_motion_ids),
        "test_motion_ids_count": len(test_motion_ids),
        "overlap_count": len(overlap),
        "source": "estimated",
    }
    with open(action_dir / "split_statistics.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    logger.info("Successfully generated Estimated Action Dataset (Train=%d, Test=%d, Overlap=0).",
                total_train, total_test)
    return split_stats


if __name__ == "__main__":
    build_estimated_action_dataset()
