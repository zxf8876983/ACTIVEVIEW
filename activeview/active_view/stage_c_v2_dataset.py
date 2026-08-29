"""Dataset and batching utilities for the Stage C-v2 representation studies."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from activeview.active_view.stage_c_v2_features import JOINT_TOKEN_DIM, SEMANTIC_CONTEXT_DIM


class StageCV2Dataset(Dataset[Dict[str, Any]]):
    """Load metadata JSONL and memory-mapped current-observation arrays."""

    def __init__(self, feature_root: Path, split: str, *, stats: Mapping[str, np.ndarray]) -> None:
        summary = json.loads((feature_root / "stage_c_v2_feature_summary.json").read_text(encoding="utf-8"))
        self.rows = [json.loads(line) for line in (feature_root / "features" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        arrays = summary["arrays"]
        self.joint_tokens = np.load(Path(arrays["joint_tokens"][split]), mmap_mode="r")
        self.skeletons = np.load(Path(arrays["skeletons"][split]), mmap_mode="r")
        self.semantic_mean = np.asarray(stats["semantic_mean"], dtype=np.float32)
        self.semantic_std = np.asarray(stats["semantic_std"], dtype=np.float32)
        self.geometry_mean = np.asarray(stats["geometry_mean"], dtype=np.float32)
        self.geometry_std = np.asarray(stats["geometry_std"], dtype=np.float32)
        if self.semantic_mean.shape != (SEMANTIC_CONTEXT_DIM,) or self.semantic_std.shape != (SEMANTIC_CONTEXT_DIM,):
            raise ValueError("Invalid semantic statistics shape")
        if self.joint_tokens.shape[0] != len(self.rows) or self.skeletons.shape[0] != len(self.rows):
            raise ValueError(f"Array/row count mismatch for {split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        semantic = np.asarray(row["current_semantic_context"], dtype=np.float32)
        geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        semantic = (semantic - self.semantic_mean) / self.semantic_std
        geometry = (geometry - self.geometry_mean) / self.geometry_std
        return {
            "joint_tokens": torch.from_numpy(np.asarray(self.joint_tokens[int(row["joint_tokens_index"])], dtype=np.float32).copy()),
            "skeleton": torch.from_numpy(np.asarray(self.skeletons[int(row["skeleton_index"])], dtype=np.float32).copy()),
            "semantic_context": torch.from_numpy(semantic),
            "candidate_geometry": torch.from_numpy(geometry),
            "utility_targets": torch.tensor(row["utility_targets"], dtype=torch.float32),
            "candidate_geodesic": torch.tensor(row["candidate_geodesic"], dtype=torch.float32),
            "candidate_ids": [int(value) for value in row["candidate_viewpoint_ids"]],
            "episode_id": str(row["episode_id"]), "record_id": str(row["record_id"]),
            "policy_split": str(row["policy_split"]), "scene_id": str(row["scene_id"]),
            "region": str(row["region"]), "label_id": int(row["label_id"]),
            "current_viewpoint_id": int(row["current_viewpoint_id"]),
            "current_entropy": float(row["current_entropy"]),
            "current_margin": float(row["current_margin"]),
            "current_pose_confidence": float(row["current_pose_confidence"]),
        }


def collate_stage_c_v2(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    size = len(batch)
    max_candidates = max(len(item["candidate_ids"]) for item in batch)
    geometry_dim = int(batch[0]["candidate_geometry"].shape[1])
    geometry = torch.zeros((size, max_candidates, geometry_dim), dtype=torch.float32)
    targets = torch.zeros((size, max_candidates), dtype=torch.float32)
    geodesic = torch.zeros((size, max_candidates), dtype=torch.float32)
    mask = torch.zeros((size, max_candidates), dtype=torch.bool)
    ids: list[list[int]] = []
    for row_index, item in enumerate(batch):
        count = len(item["candidate_ids"])
        geometry[row_index, :count] = item["candidate_geometry"]
        targets[row_index, :count] = item["utility_targets"]
        geodesic[row_index, :count] = item["candidate_geodesic"]
        mask[row_index, :count] = True
        ids.append(list(item["candidate_ids"]))
    return {
        "joint_tokens": torch.stack([item["joint_tokens"] for item in batch]),
        "skeleton": torch.stack([item["skeleton"] for item in batch]),
        "semantic_context": torch.stack([item["semantic_context"] for item in batch]),
        "candidate_geometry": geometry, "utility_targets": targets,
        "candidate_geodesic": geodesic, "candidate_mask": mask, "candidate_ids": ids,
        "episode_id": [str(item["episode_id"]) for item in batch],
        "record_id": [str(item["record_id"]) for item in batch],
        "policy_split": [str(item["policy_split"]) for item in batch],
        "scene_id": [str(item["scene_id"]) for item in batch],
        "region": [str(item["region"]) for item in batch],
        "label_id": torch.tensor([int(item["label_id"]) for item in batch], dtype=torch.long),
        "current_viewpoint_id": [int(item["current_viewpoint_id"]) for item in batch],
        "current_entropy": [float(item["current_entropy"]) for item in batch],
        "current_margin": [float(item["current_margin"]) for item in batch],
        "current_pose_confidence": [float(item["current_pose_confidence"]) for item in batch],
    }


class V2RecordBalancedSampler(Sampler[int]):
    """Use the same record-balanced protocol as frozen Stage C-v0."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], episodes_per_record: int = 16, seed: int = 42) -> None:
        if episodes_per_record <= 0:
            raise ValueError("episodes_per_record must be positive")
        self.groups: Dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.groups[str(row["record_id"])].append(index)
        self.episodes_per_record = int(episodes_per_record)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        selected: list[int] = []
        for record_id in sorted(self.groups):
            group = np.asarray(self.groups[record_id], dtype=np.int64)
            values = rng.choice(group, size=self.episodes_per_record, replace=len(group) < self.episodes_per_record)
            selected.extend(int(value) for value in values)
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.groups) * self.episodes_per_record


def compute_v2_statistics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot compute statistics from empty rows")
    semantic = np.asarray([row["current_semantic_context"] for row in rows], dtype=np.float64)
    geometry = np.concatenate([np.asarray(row["candidate_geometry"], dtype=np.float64) for row in rows], axis=0)
    semantic_std = semantic.std(axis=0)
    geometry_std = geometry.std(axis=0)
    semantic_std[semantic_std < 1e-6] = 1.0
    geometry_std[geometry_std < 1e-6] = 1.0
    return {
        "semantic_mean": semantic.mean(axis=0).astype(np.float32), "semantic_std": semantic_std.astype(np.float32),
        "geometry_mean": geometry.mean(axis=0).astype(np.float32), "geometry_std": geometry_std.astype(np.float32),
    }


def save_v2_statistics(path: Path, stats: Mapping[str, np.ndarray]) -> None:
    path.write_text(json.dumps({key: np.asarray(value).tolist() for key, value in stats.items()}, indent=2), encoding="utf-8")


def load_v2_statistics(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: np.asarray(payload[key], dtype=np.float32) for key in ("semantic_mean", "semantic_std", "geometry_mean", "geometry_std")}
