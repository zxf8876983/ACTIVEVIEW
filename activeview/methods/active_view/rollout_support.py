"""文件用途：
    实现主动视角策略运行时逻辑。

主要输入：
    - 策略特征、候选视点和访问历史。
主要输出：
    - 动作、rollout 或 utility 预测。
项目角色：
    - 属于 methods.active_view 方法模块。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from activeview.methods.active_view.geometry import candidate_order
from activeview.perception.rgb_features import RGBObservationKey, load_dinov2
from activeview.data.preprocessing.rgb_cache import build_or_load_spatial_cache

VIEW_COUNT = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_candidate_order(row: Mapping[str, Any], current: int, visited: set[int], pairwise: Mapping[int, Mapping[int, float]], azimuths: Mapping[int, float]) -> list[int]:
    return candidate_order(row, current, visited, pairwise, azimuths)


@dataclass
class Observation:
    skeleton: np.ndarray
    rgb: np.ndarray
    logp: np.ndarray


class VisitedObservationStore:
    """Reveal archived observations only after a viewpoint is selected."""

    def __init__(self, source: Path, archive: Mapping[str, np.ndarray], rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray], key: tuple[str, str, str]) -> None:
        self.source = source
        self.archive = archive
        self.rgb_lookup = rgb_lookup
        self.key = key
        self._revealed: set[int] = set()

    def reveal(self, viewpoint_id: int) -> Observation:
        viewpoint_id = int(viewpoint_id)
        if viewpoint_id < 0 or viewpoint_id >= VIEW_COUNT:
            raise ValueError("invalid viewpoint")
        ids = np.asarray(self.archive["viewpoint_ids"], dtype=np.int64)
        positions = {int(value): index for index, value in enumerate(ids.tolist())}
        if viewpoint_id not in positions:
            raise ValueError("viewpoint alignment failure")
        self._revealed.add(viewpoint_id)
        index = positions[viewpoint_id]
        return Observation(
            np.asarray(self.archive["skeleton"][index], dtype=np.float32),
            np.asarray(self.rgb_lookup[(*self.key, viewpoint_id)], dtype=np.float32),
            np.asarray(self.archive["logp"][index], dtype=np.float32),
        )


def load_sources(source_path: Path, cache: Mapping[str, np.ndarray], context: tuple[str, str, str]) -> dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
    cache_ids = np.asarray(cache.get("viewpoint_ids", ids), dtype=np.int64)
    if skeleton.shape != (VIEW_COUNT, 3, 30, 17) or not np.array_equal(ids, cache_ids):
        raise ValueError(f"skeleton/cache alignment failure for {context}")
    return {"skeleton": skeleton, "viewpoint_ids": ids, "positions": positions, "logp": np.asarray(cache["true_logp"], dtype=np.float32)}


def joint_choice(model: torch.nn.Module, current_logs: tuple[np.ndarray, np.ndarray], imagined: np.ndarray, descriptors: np.ndarray, device: torch.device) -> int | None:
    from activeview.methods.joint_revision.model import choose_next
    return choose_next(model, current_logs, imagined, descriptors, device)


def build_rgb_cache(data_root: Path, rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str], device: torch.device, cache_dir: Path) -> tuple[dict[tuple[str, str, str, int], np.ndarray], dict[str, Any]]:
    del data_root, sources
    keys = [RGBObservationKey(str(row["scene_id"]), str(row["region"]), str(row["record_id"]), view) for row in rows for view in range(VIEW_COUNT)]
    started = time.monotonic()
    values, manifest, info = build_or_load_spatial_cache(
        rgb_root=Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train"),
        cache_dir=cache_dir, keys=keys, model_loader=load_dinov2, device=device, batch_size=64,
    )
    lookup = {(str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"])): np.asarray(values[index], dtype=np.float32) for index, row in enumerate(manifest)}
    return lookup, {**info, "contexts": len(rows), "views": len(keys), "elapsed_sec": time.monotonic() - started, "future_candidate_rgb_used": False}


_candidate_order = frozen_candidate_order
_joint_choice = joint_choice
_load_sources = load_sources

__all__ = ["Observation", "VisitedObservationStore", "build_rgb_cache", "frozen_candidate_order", "joint_choice", "load_sources", "sha256"]
