"""文件用途：
    预处理已有 ActiveView 数据并构建缓存。

主要输入：
    - 策略记录、骨架预测和几何/RGB 特征。
主要输出：
    - 训练所需的 feature/cache 对象。
项目角色：
    - 属于 data.preprocessing 预处理模块。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class EpisodeFeatureDataset(Dataset[Dict[str, Any]]):
    """Load one feature-cache JSONL row per Episode."""

    def __init__(self, path: Path, *, current_mean: np.ndarray | None = None, current_std: np.ndarray | None = None, geometry_mean: np.ndarray | None = None, geometry_std: np.ndarray | None = None) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.current_mean = current_mean
        self.current_std = current_std
        self.geometry_mean = geometry_mean
        self.geometry_std = geometry_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        current = np.asarray(row["current_feature"], dtype=np.float32)
        geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        if self.current_mean is not None and self.current_std is not None:
            current = (current - self.current_mean) / self.current_std
        if self.geometry_mean is not None and self.geometry_std is not None:
            geometry = (geometry - self.geometry_mean) / self.geometry_std
        return {
            "current_feature": torch.from_numpy(current),
            "candidate_geometry": torch.from_numpy(geometry),
            "utility_targets": torch.tensor(row["utility_targets"], dtype=torch.float32),
            "candidate_ids": [int(value) for value in row["candidate_viewpoint_ids"]],
            "candidate_geodesic": torch.tensor(row["candidate_geodesic"], dtype=torch.float32),
            "episode_id": str(row["episode_id"]),
            "record_id": str(row["record_id"]),
            "policy_split": str(row["policy_split"]),
            "scene_id": str(row["scene_id"]),
            "region": str(row["region"]),
            "label_id": int(row["label_id"]),
        }


def collate_episode_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    max_candidates = max(len(item["candidate_ids"]) for item in batch)
    geometry_dim = int(batch[0]["candidate_geometry"].shape[1])
    current_dim = int(batch[0]["current_feature"].shape[0])
    geometry = torch.zeros((batch_size, max_candidates, geometry_dim), dtype=torch.float32)
    targets = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    geodesic = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    candidate_ids: List[List[int]] = []
    for row_index, item in enumerate(batch):
        count = len(item["candidate_ids"])
        geometry[row_index, :count] = item["candidate_geometry"]
        targets[row_index, :count] = item["utility_targets"]
        geodesic[row_index, :count] = item["candidate_geodesic"]
        mask[row_index, :count] = True
        candidate_ids.append(list(item["candidate_ids"]))
    return {
        "current_feature": torch.stack([item["current_feature"] for item in batch]),
        "candidate_geometry": geometry,
        "utility_targets": targets,
        "candidate_geodesic": geodesic,
        "candidate_mask": mask,
        "candidate_ids": candidate_ids,
        "episode_id": [str(item["episode_id"]) for item in batch],
        "record_id": [str(item["record_id"]) for item in batch],
        "policy_split": [str(item["policy_split"]) for item in batch],
        "scene_id": [str(item["scene_id"]) for item in batch],
        "region": [str(item["region"]) for item in batch],
        "label_id": torch.tensor([int(item["label_id"]) for item in batch], dtype=torch.long),
        "current_feature_dim": current_dim,
    }


class RecordBalancedSampler(Sampler[int]):
    """Sample a fixed number of Episodes per independent motion record per epoch."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], episodes_per_record: int = 16, seed: int = 42) -> None:
        if episodes_per_record <= 0:
            raise ValueError("episodes_per_record must be positive")
        self.groups: Dict[str, List[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.groups[str(row["record_id"])].append(index)
        self.episodes_per_record = int(episodes_per_record)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        indices: List[int] = []
        for record_id in sorted(self.groups):
            group = np.asarray(self.groups[record_id], dtype=np.int64)
            selected = rng.choice(group, size=self.episodes_per_record, replace=len(group) < self.episodes_per_record)
            indices.extend(int(value) for value in selected)
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.groups) * self.episodes_per_record


class HardRecordAwareSampler(Sampler[int]):
    """Oversample Train records marked hard by a frozen difficulty analysis."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        hard_record_ids: Iterable[str],
        *,
        hard_episodes_per_record: int = 32,
        normal_episodes_per_record: int = 12,
        seed: int = 42,
    ) -> None:
        if hard_episodes_per_record <= 0 or normal_episodes_per_record <= 0:
            raise ValueError("episodes_per_record values must be positive")
        self.groups: Dict[str, List[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.groups[str(row["record_id"])].append(index)
        self.hard_record_ids = {str(record_id) for record_id in hard_record_ids}
        unknown = self.hard_record_ids.difference(self.groups)
        if unknown:
            raise ValueError(
                "Difficulty contains record IDs absent from the Train dataset: "
                + ", ".join(sorted(unknown))
            )
        self.hard_episodes_per_record = int(hard_episodes_per_record)
        self.normal_episodes_per_record = int(normal_episodes_per_record)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        indices: List[int] = []
        for record_id in sorted(self.groups):
            group = np.asarray(self.groups[record_id], dtype=np.int64)
            count = (
                self.hard_episodes_per_record
                if record_id in self.hard_record_ids
                else self.normal_episodes_per_record
            )
            selected = rng.choice(group, size=count, replace=len(group) < count)
            indices.extend(int(value) for value in selected)
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        hard_count = len(self.hard_record_ids)
        normal_count = len(self.groups) - hard_count
        return (
            hard_count * self.hard_episodes_per_record
            + normal_count * self.normal_episodes_per_record
        )


def feature_statistics(rows: Iterable[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    current: List[np.ndarray] = []
    geometry: List[np.ndarray] = []
    for row in rows:
        current.append(np.asarray(row["current_feature"], dtype=np.float64))
        geometry.extend(np.asarray(row["candidate_geometry"], dtype=np.float64))
    if not current or not geometry:
        raise ValueError("Cannot compute feature statistics from empty rows")
    current_array = np.stack(current)
    geometry_array = np.stack(geometry)
    current_std = current_array.std(axis=0)
    geometry_std = geometry_array.std(axis=0)
    current_std[current_std < 1e-6] = 1.0
    geometry_std[geometry_std < 1e-6] = 1.0
    return {
        "current_mean": current_array.mean(axis=0).astype(np.float32),
        "current_std": current_std.astype(np.float32),
        "geometry_mean": geometry_array.mean(axis=0).astype(np.float32),
        "geometry_std": geometry_std.astype(np.float32),
    }


def load_feature_statistics(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: np.asarray(payload[key], dtype=np.float32) for key in ("current_mean", "current_std", "geometry_mean", "geometry_std")}


def save_feature_statistics(path: Path, statistics: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: np.asarray(value).tolist() for key, value in statistics.items()}, indent=2), encoding="utf-8")
