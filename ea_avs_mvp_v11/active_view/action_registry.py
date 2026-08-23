"""
AMASS 完整动作数据与类别注册管理器 —— action_registry.py (v11.5)
============================================================

职责：
    1. 自动扫描、解析并建立全量 AMASS 动作时序资产索引；
    2. 统一管理 SMPL 标准骨架格式 (30 frames, 33 joints, 3 coords)；
    3. 支持动作类别、动作实例、序列长度与来源元数据映射；
    4. 导出 actions_statistics.json 与提供动作样本检索接口。
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.core.paths import get_data_root

logger = logging.getLogger("action_registry")

DEFAULT_ACTION_CATEGORIES = [
    "standing",
    "sitting",
    "sit_down",
    "stand_up",
    "bending",
    "reaching",
    "picking_up",
    "squatting",
    "jumping",
    "turning",
    "stretching",
    "waving",
    "dancing",
    "kicking",
    "fall_stumble",
    "placing",
]


@dataclass
class MotionMetadata:
    """单个动作实例元数据实体。"""
    action_id: int
    action_label: str
    motion_id: str
    sample_id: str
    category: str
    sequence_length: int = 30
    joint_num: int = 33
    split: str = "train"
    source: str = "amass_clean_perception"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "action_label": str(self.action_label),
            "motion_id": str(self.motion_id),
            "sample_id": str(self.sample_id),
            "category": str(self.category),
            "sequence_length": int(self.sequence_length),
            "joint_num": int(self.joint_num),
            "split": str(self.split),
            "source": str(self.source),
            "metadata": self.metadata,
        }


class ActionRegistry:
    """AMASS 动作资产注册表 (支持排除 walk/run 等位移动作，全面启用其余动作)。"""

    def __init__(self, data_root: Optional[Union[str, Path]] = None, exclude_locomotion: bool = True):
        self.data_root = Path(data_root) if data_root else get_data_root()
        self.action_dataset_dir = self.data_root / "datasets" / "action"
        self.exclude_locomotion = exclude_locomotion
        
        self.categories: List[str] = list(DEFAULT_ACTION_CATEGORIES)
        self.category_to_id: Dict[str, int] = {c: i for i, c in enumerate(self.categories)}
        self.id_to_category: Dict[int, str] = {i: c for i, c in enumerate(self.categories)}
        
        self._motions: Dict[str, MotionMetadata] = {}
        self._category_index: Dict[str, List[str]] = {c: [] for c in self.categories}
        self._train_data: Optional[np.ndarray] = None
        self._test_data: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._test_labels: Optional[np.ndarray] = None
        
        self.scan_and_register_all_motions()


    def scan_and_register_all_motions(self) -> Dict[str, MotionMetadata]:
        """扫描所有可用 AMASS 动作文件并构建全量动作注册表。"""
        self._motions.clear()
        for c in self.categories:
            self._category_index[c] = []

        train_manifest = self.action_dataset_dir / "train" / "clean_perception" / "manifest.json"
        test_manifest = self.action_dataset_dir / "test" / "clean_perception" / "manifest.json"

        all_manifest_items: List[Tuple[Dict[str, Any], str]] = []
        if train_manifest.exists():
            with open(train_manifest, "r", encoding="utf-8") as f:
                all_manifest_items.extend([(item, "train") for item in json.load(f)])
        if test_manifest.exists():
            with open(test_manifest, "r", encoding="utf-8") as f:
                all_manifest_items.extend([(item, "test") for item in json.load(f)])

        # 加载 Numpy 原始数据
        train_data_p = self.action_dataset_dir / "train" / "clean_perception" / "data.npy"
        test_data_p = self.action_dataset_dir / "test" / "clean_perception" / "data.npy"
        train_lbl_p = self.action_dataset_dir / "train" / "clean_perception" / "labels.npy"
        test_lbl_p = self.action_dataset_dir / "test" / "clean_perception" / "labels.npy"

        if train_data_p.exists():
            self._train_data = np.load(train_data_p)
        if test_data_p.exists():
            self._test_data = np.load(test_data_p)
        if train_lbl_p.exists():
            self._train_labels = np.load(train_lbl_p)
        if test_lbl_p.exists():
            self._test_labels = np.load(test_lbl_p)

        for item, split in all_manifest_items:
            m_id = item.get("motion_id", item.get("sample_id", ""))
            act_lbl = item.get("action_label", "standing")
            
            # 排除 walk / run 动作
            if self.exclude_locomotion and any(k in act_lbl.lower() for k in ["walk", "run", "jog"]):
                continue

            if act_lbl not in self.category_to_id:
                self.categories.append(act_lbl)
                self.category_to_id[act_lbl] = len(self.categories) - 1
                self.id_to_category[len(self.categories) - 1] = act_lbl
                self._category_index[act_lbl] = []

            act_id = self.category_to_id[act_lbl]
            meta = MotionMetadata(
                action_id=act_id,
                action_label=act_lbl,
                motion_id=m_id,
                sample_id=item.get("sample_id", m_id),
                category=act_lbl,
                sequence_length=int(item.get("sequence_length", 30)),
                joint_num=int(item.get("joint_num", 33)),
                split=split,
                source=item.get("source", "amass_clean_perception"),
                metadata=item,
            )
            self._motions[m_id] = meta
            self._category_index[act_lbl].append(m_id)

        logger.info("ActionRegistry loaded %d non-locomotion motion sequences across %d categories: %s.",
                    len(self._motions), len(self.categories), self.categories)
        return self._motions

    def get_skeleton_sequence(self, action_id: int, instance_idx: int, split: str = "train") -> np.ndarray:
        """获取指定动作类别与实例的标准 (30, 33, 3) 骨架时序。"""
        data_source = self._train_data if split == "train" else (self._test_data if self._test_data is not None else self._train_data)
        
        if data_source is not None and len(data_source) > 0:
            num_cats = len(self.categories)
            samples_per_action = len(data_source) // num_cats
            if samples_per_action > 0:
                base_idx = (action_id % num_cats) * samples_per_action + (instance_idx % max(samples_per_action, 1))
                if base_idx < len(data_source):
                    raw_sample = data_source[base_idx, :, :, :, 0] # (C, T, V)
                    return np.transpose(raw_sample, (1, 2, 0)).astype(np.float32)

        # 过程化合成回退机制
        try:
            from tools.dataset_generation.generate_16class_amass_dataset import synthesize_canonical_motion
            target_cat = self.id_to_category.get(action_id, self.categories[action_id % len(self.categories)])
            return synthesize_canonical_motion(target_cat, seed=action_id * 1000 + instance_idx)
        except Exception:
            T, V, C = 30, 33, 3
            skel = np.zeros((T, V, C), dtype=np.float32)
            for t in range(T):
                skel[t, 0] = [0.0, 0.50, 0.0]
                skel[t, 11] = [0.20, 0.35, 0.0]
                skel[t, 12] = [-0.20, 0.35, 0.0]
                skel[t, 23] = [0.15, -0.10, 0.0]
                skel[t, 24] = [-0.15, -0.10, 0.0]
            return skel


    def export_statistics(self, output_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """导出动作类别统计字典并持久化。"""
        out_p = Path(output_file) if output_file else (self.data_root / "v11_actions_metadata" / "actions_statistics.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_motion_sequences": len(self._motions),
            "num_categories": len(self.categories),
            "categories": self.categories,
            "category_distribution": {
                c: len(self._category_index[c]) for c in self.categories
            },
            "sequence_length": 30,
            "joint_count": 33,
            "splits": {
                "train_samples": len([m for m in self._motions.values() if m.split == "train"]),
                "test_samples": len([m for m in self._motions.values() if m.split == "test"]),
            },
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info("Exported action statistics to: %s", out_p)
        return stats
