#!/usr/bin/env python3
"""
AMASS Non-Locomotion Motion Dataset Splitter —— dataset_split.py (v11.5)
=======================================================================

职责：
    1. 扫描与索引 /home/zxf/WorkSpace/code/data/ActiveView/assets/motions/converted/ 下所有 AMASS 动作实例；
    2. 按动作类别分层进行严格的 Instance-Level Disjoint 划分：
       - Train: 70%
       - Validation: 15%
       - Test: 15%
    3. 保存元数据清单至 /home/zxf/WorkSpace/code/data/ActiveView/datasets/amass_split/：
       - train.json
       - val.json
       - test.json
    4. 严格校验：train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅。
"""

import json
import logging
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 保证包路径正确
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.core.paths import get_data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_split")


def get_converted_motions_dir() -> Path:
    data_root = get_data_root()
    return data_root / "assets" / "motions" / "converted"


def get_split_output_dir() -> Path:
    data_root = get_data_root()
    out_dir = data_root / "datasets" / "amass_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def scan_and_split_motions(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """扫描所有动作文件并按类别分层拆分。"""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Splits must sum to 1.0"

    converted_dir = get_converted_motions_dir()
    pkl_files = sorted(list(converted_dir.glob("*.pkl")))
    logger.info("Found %d converted motion files in %s", len(pkl_files), converted_dir)

    category_to_records = defaultdict(list)

    for p in pkl_files:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            meta = d.get("metadata", {})
            action_label = meta.get("target_class", "unknown")
            sid = str(meta.get("babel_sid", "unknown"))
            ds = str(meta.get("amass_dataset", "unknown"))
            num_frames = int(d["pose_motion"]["joints_array"].shape[0])
            motion_id = p.stem

            record = {
                "motion_id": motion_id,
                "file_path": str(p.relative_to(get_data_root())),
                "absolute_path": str(p),
                "dataset_source": ds,
                "action_label": action_label,
                "subject_id": sid,
                "num_frames": num_frames,
            }
            category_to_records[action_label].append(record)
        except Exception as e:
            logger.error("Failed to load motion file %s: %s", p.name, e)

    # 随机分层抽样
    rng = random.Random(seed)
    train_records = []
    val_records = []
    test_records = []

    for cat, records in sorted(category_to_records.items()):
        shuffled = records.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        
        # 确保小样本类别至少分配至 train 和 test
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1 if n >= 3 else 0, int(round(n * val_ratio)))
        # 调整保证总数严格等于 n
        if n_train + n_val >= n:
            n_train = max(1, n - 2) if n >= 3 else max(1, n - 1)
            n_val = 1 if (n - n_train) >= 2 else 0
        n_test = n - n_train - n_val
        if n_test <= 0 and n >= 2:
            n_test = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        tr = shuffled[:n_train]
        va = shuffled[n_train : n_train + n_val]
        te = shuffled[n_train + n_val :]

        train_records.extend(tr)
        val_records.extend(va)
        test_records.extend(te)

        logger.info(
            "Category '%s': Total=%d -> Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
            cat, n, len(tr), len(tr)/n*100, len(va), len(va)/n*100, len(te), len(te)/n*100,
        )

    return train_records, val_records, test_records


def verify_disjoint_splits(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> None:
    """验证各集合之间完全互斥且无交集。"""
    train_ids = {r["motion_id"] for r in train}
    val_ids = {r["motion_id"] for r in val}
    test_ids = {r["motion_id"] for r in test}

    assert len(train_ids & val_ids) == 0, f"Leakage detected: train ∩ val = {train_ids & val_ids}"
    assert len(train_ids & test_ids) == 0, f"Leakage detected: train ∩ test = {train_ids & test_ids}"
    assert len(val_ids & test_ids) == 0, f"Leakage detected: val ∩ test = {val_ids & test_ids}"
    logger.info("Strict verification PASSED: train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅!")


def main():
    out_dir = get_split_output_dir()
    train_recs, val_recs, test_recs = scan_and_split_motions()

    verify_disjoint_splits(train_recs, val_recs, test_recs)

    # 保存 JSON 文件
    train_p = out_dir / "train.json"
    val_p = out_dir / "val.json"
    test_p = out_dir / "test.json"

    with open(train_p, "w", encoding="utf-8") as f:
        json.dump(train_recs, f, indent=2, ensure_ascii=False)
    with open(val_p, "w", encoding="utf-8") as f:
        json.dump(val_recs, f, indent=2, ensure_ascii=False)
    with open(test_p, "w", encoding="utf-8") as f:
        json.dump(test_recs, f, indent=2, ensure_ascii=False)

    logger.info("Saved dataset splits to: %s", out_dir)
    logger.info("  Train Samples: %d -> %s", len(train_recs), train_p.name)
    logger.info("  Val Samples:   %d -> %s", len(val_recs), val_p.name)
    logger.info("  Test Samples:  %d -> %s", len(test_recs), test_p.name)


if __name__ == "__main__":
    main()
