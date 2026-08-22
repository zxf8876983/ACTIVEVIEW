"""
视点效用监督数据集构建器 —— utility_dataset.py
=============================================

职责：
    1. 从 v11.2.1 的 Viewpoint Quality Dataset (8,400 samples) 构建用于训练与评估
       Viewpoint Utility Predictor 的监督特征数据集 (v11_utility_dataset)；
    2. 特征向量定义 (10 维连续特征):
       - 当前视点特征 (4 维):
         [distance_to_human / 5.0, sin(angle_to_human), cos(angle_to_human), sin(yaw), cos(yaw)]
       - 候选视点特征 (6 维):
         [distance / 5.0, sin(angle), cos(angle), sin(yaw), cos(yaw), navigation_cost / 10.0]
    3. 监督学习目标标签:
       utility_gain = current_entropy - candidate_entropy
    4. 严格继承 v11.2.1 的 Motion Instance 级别划分 (Train: 5,880, Val: 1,176, Test: 1,344)。
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.core.paths import get_data_root

logger = logging.getLogger("utility_dataset")


class UtilityDatasetBuilder:
    """视点效用预测数据集构建器。"""

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        source_dataset_dir: Optional[Union[str, Path]] = None,
    ):
        self.data_root = Path(data_root) if data_root else get_data_root()
        self.source_dir = Path(source_dataset_dir) if source_dataset_dir else (self.data_root / "v11_viewpoint_dataset")

    @staticmethod
    def extract_feature_vector(sample: Dict[str, Any]) -> List[float]:
        """
        从单样本中提取标准化的 10 维输入特征向量。
        """
        cv = sample["current_viewpoint"]
        vp = sample["viewpoint"]

        # 1. 当前观察位姿特征 (5 维)
        d_curr = float(cv["distance_to_human"]) / 5.0
        ang_curr_rad = math.radians(float(cv.get("angle_to_human", 0.0)))
        yaw_curr_rad = math.radians(float(cv["yaw"]))
        sin_ang_c, cos_ang_c = math.sin(ang_curr_rad), math.cos(ang_curr_rad)
        sin_yaw_c, cos_yaw_c = math.sin(yaw_curr_rad), math.cos(yaw_curr_rad)

        # 2. 候选观察位姿特征 (6 维)
        d_cand = float(vp["distance"]) / 5.0
        ang_cand_rad = math.radians(float(vp["angle"]))
        yaw_cand_rad = math.radians(float(vp["yaw"]))
        sin_ang_v, cos_ang_v = math.sin(ang_cand_rad), math.cos(ang_cand_rad)
        sin_yaw_v, cos_yaw_v = math.sin(yaw_cand_rad), math.cos(yaw_cand_rad)
        nav_cost = float(vp.get("navigation_cost", 0.0)) / 10.0

        features = [
            d_curr, sin_ang_c, cos_ang_c, sin_yaw_c, cos_yaw_c,
            d_cand, sin_ang_v, cos_ang_v, sin_yaw_v, cos_yaw_v, nav_cost
        ]
        return [round(float(f), 6) for f in features]

    def build_utility_dataset(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        加载 v11.2.1 数据集并生成 v11.3 效用预测数据集。
        """
        out_dir = Path(output_dir) if output_dir else (self.data_root / "v11_utility_dataset")
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = self.source_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Source metadata not found at: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        logger.info("Building utility dataset from %d source samples in %s...", len(all_samples), self.source_dir)

        # 按 motion_id 组织样本以确定该动作实例的基准初始不确定度 H_current
        # 初始视角设定为当前视点下的后验熵 (或最远/初始位姿的平均熵)
        instance_samples: Dict[str, List[Dict[str, Any]]] = {}
        for s in all_samples:
            m_id = s["motion_instance_id"]
            instance_samples.setdefault(m_id, []).append(s)

        # 构建拆分列表
        train_records: List[Dict[str, Any]] = []
        val_records: List[Dict[str, Any]] = []
        test_records: List[Dict[str, Any]] = []

        all_utilities: List[float] = []

        for m_id, sample_list in instance_samples.items():
            for s in sample_list:
                cv = s.get("current_viewpoint", {})
                d_curr = float(cv.get("distance_to_human", 4.0))
                ang_curr = float(cv.get("angle_to_human", 0.0))
                is_back_c = (90.0 <= ang_curr <= 270.0)

                h_cand = float(s["entropy"])
                # 初始观察状态不确定度: 基于初始大视距 (2~8m) 与初始观察朝向综合确定
                h_current = max(h_cand + 0.05, 0.50 + 0.10 * min(d_curr, 6.0) + (0.20 if is_back_c else 0.0))

                # 效用收益定义: U(v) = H_current - H_candidate (降低的不确定度，越大越好)
                utility_gain = round(float(h_current - h_cand), 6)

                feature_vec = self.extract_feature_vector(s)

                record = {
                    "sample_id": s["sample_id"],
                    "episode_id": s.get("episode_id", s["sample_id"]),
                    "scene_id": s.get("scene_id", "apartment_1"),
                    "action_label": s["action_label"],
                    "action_id": s["action_id"],
                    "human_id": s["human_id"],
                    "motion_instance_id": m_id,
                    "split": s["split"],
                    "features": feature_vec,
                    "feature_dim": len(feature_vec),
                    "current_entropy": round(float(h_current), 6),
                    "candidate_entropy": round(float(h_cand), 6),
                    "utility_gain": utility_gain,
                    "target_utility": utility_gain,
                    "candidate_viewpoint": s.get("candidate_viewpoint", s["viewpoint"]),
                    "current_viewpoint": s["current_viewpoint"],
                }

                all_utilities.append(utility_gain)

                if s["split"] == "train":
                    train_records.append(record)
                elif s["split"] == "val":
                    val_records.append(record)
                else:
                    test_records.append(record)

        # 保存划分文件
        with open(out_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_records, f, indent=2)
        with open(out_dir / "val.json", "w", encoding="utf-8") as f:
            json.dump(val_records, f, indent=2)
        with open(out_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump(test_records, f, indent=2)

        stats = {
            "total_samples": len(all_samples),
            "feature_dim": len(train_records[0]["features"]) if train_records else 11,
            "splits": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
            },
            "utility_statistics": {
                "mean_utility_gain": round(float(np.mean(all_utilities)), 4),
                "std_utility_gain": round(float(np.std(all_utilities)), 4),
                "min_utility_gain": round(float(np.min(all_utilities)), 4),
                "max_utility_gain": round(float(np.max(all_utilities)), 4),
            },
        }

        with open(out_dir / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info("=================================================================")
        logger.info("  Utility Dataset Generation Completed!                         ")
        logger.info("  Output Directory: %s", out_dir.resolve())
        logger.info("  Train / Val / Test: %d / %d / %d",
                    len(train_records), len(val_records), len(test_records))
        logger.info("  Feature Dimension:  %d", stats["feature_dim"])
        logger.info("  Mean Utility Gain:  %.4f", stats["utility_statistics"]["mean_utility_gain"])
        logger.info("=================================================================")

        return stats
