"""文件用途：
    实现候选观测 world model。

主要输入：
    - 当前观测特征与候选几何。
主要输出：
    - 未来观测表示和训练损失。
项目角色：
    - 属于 methods.world_model 方法模块。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from activeview.methods.active_view.geometry import relative_view_descriptor

SKELETON_SHAPE = (3, 30, 17)
SKELETON_OUTPUT_DIM = int(np.prod(SKELETON_SHAPE))
VIEW_COUNT = 32


def _temporal_token_encoder() -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=2)


class CandidateObservationWorldModel(nn.Module):
    """Small history/candidate-conditioned perceived-skeleton predictor."""

    def __init__(self, *, use_belief: bool = False, use_rgb: bool = False, residual: bool = False, num_classes: int = 16) -> None:
        super().__init__()
        self.use_belief = bool(use_belief)
        self.use_rgb = bool(use_rgb)
        self.residual = bool(residual)
        self.num_classes = int(num_classes)
        self.frame_encoder = nn.Linear(17 * 3, 128)
        self.temporal_position = nn.Parameter(torch.zeros(30, 128))
        self.temporal_encoder = _temporal_token_encoder()
        self.view_descriptor = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, 128))
        self.history_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=2,
        )
        self.belief_encoder = (
            nn.Sequential(nn.Linear(2 * self.num_classes, 64), nn.GELU(), nn.Linear(64, 128))
            if self.use_belief else None
        )
        self.rgb_projector = nn.Sequential(nn.Linear(768, 128), nn.GELU()) if self.use_rgb else None
        self.rgb_encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=128,
                    nhead=4,
                    dim_feedforward=256,
                    dropout=0.1,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                ),
                num_layers=1,
            )
            if self.use_rgb else None
        )
        self.candidate_encoder = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 128))
        self.decoder_queries = nn.Parameter(torch.zeros(30, 128))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output_head = nn.Linear(128, 17 * 3)
        nn.init.normal_(self.temporal_position, std=0.02)
        nn.init.normal_(self.decoder_queries, std=0.02)

    def _encode_view(self, skeleton: torch.Tensor) -> torch.Tensor:
        # skeleton: [B, 3, 30, 17]
        if skeleton.ndim != 4 or tuple(skeleton.shape[1:]) != SKELETON_SHAPE:
            raise ValueError(f"expected skeleton [B,3,30,17], got {tuple(skeleton.shape)}")
        frames = skeleton.permute(0, 2, 1, 3).reshape(skeleton.size(0), 30, 51)
        tokens = self.frame_encoder(frames) + self.temporal_position.unsqueeze(0)
        return self.temporal_encoder(tokens).mean(dim=1)

    def _encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if not self.use_rgb or self.rgb_projector is None or self.rgb_encoder is None:
            return torch.zeros((rgb.size(0), 128), dtype=rgb.dtype, device=rgb.device)
        if rgb.ndim != 3 or rgb.shape[1:] != (16, 768):
            raise ValueError(f"expected RGB spatial tokens [B,16,768], got {tuple(rgb.shape)}")
        return self.rgb_encoder(self.rgb_projector(rgb)).mean(dim=1)

    def forward(
        self,
        history_skeleton: torch.Tensor,
        history_descriptor: torch.Tensor,
        candidate_descriptor: torch.Tensor,
        history_belief: torch.Tensor | None = None,
        history_rgb: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one candidate skeleton; output shape is ``[B,3,30,17]``."""
        if history_skeleton.ndim != 5 or tuple(history_skeleton.shape[2:]) != SKELETON_SHAPE:
            raise ValueError("history_skeleton must have shape [B,H,3,30,17]")
        batch, history_count = history_skeleton.shape[:2]
        flat = history_skeleton.reshape(batch * history_count, *SKELETON_SHAPE)
        view_tokens = self._encode_view(flat).reshape(batch, history_count, 128)
        if history_descriptor.shape != (batch, history_count, 9):
            raise ValueError("history_descriptor must have shape [B,H,9]")
        view_tokens = view_tokens + self.view_descriptor(history_descriptor)
        if self.use_belief:
            if history_belief is None or self.belief_encoder is None or history_belief.shape != (batch, 2 * self.num_classes):
                raise ValueError(f"belief model requires [B,{2 * self.num_classes}] history belief input")
            view_tokens = view_tokens + self.belief_encoder(history_belief).unsqueeze(1)
        if self.use_rgb:
            if history_rgb is None or history_rgb.shape != (batch, history_count, 16, 768):
                raise ValueError("RGB model requires [B,H,16,768] history RGB input")
            rgb = history_rgb.reshape(batch * history_count, 16, 768)
            rgb_tokens = self._encode_rgb(rgb).reshape(batch, history_count, 128)
            view_tokens = view_tokens + rgb_tokens
        memory = self.history_encoder(view_tokens, src_key_padding_mask=None if history_mask is None else ~history_mask.bool())
        history_state = memory.mean(dim=1)
        if candidate_descriptor.ndim == 2:
            if candidate_descriptor.shape != (batch, 9):
                raise ValueError("candidate_descriptor must have shape [B,9]")
            condition = history_state + self.candidate_encoder(candidate_descriptor)
            queries = self.decoder_queries.unsqueeze(0).expand(batch, -1, -1)
            decoded = self.decoder(queries, condition.unsqueeze(1))
            output = self.output_head(decoded).reshape(batch, 3, 30, 17)
            return output + history_skeleton[:, -1] if self.residual else output
        if candidate_descriptor.ndim != 3 or candidate_descriptor.shape[0] != batch or candidate_descriptor.shape[-1] != 9:
            raise ValueError("candidate_descriptor must have shape [B,K,9]")
        candidate_count = candidate_descriptor.size(1)
        condition = history_state.unsqueeze(1) + self.candidate_encoder(candidate_descriptor)
        condition = condition.reshape(batch * candidate_count, 128)
        queries = self.decoder_queries.unsqueeze(0).expand(batch * candidate_count, -1, -1)
        decoded = self.decoder(queries, condition.unsqueeze(1))
        output = self.output_head(decoded).reshape(batch, candidate_count, 3, 30, 17)
        return output + history_skeleton[:, -1].unsqueeze(1) if self.residual else output


@dataclass(frozen=True)
class WorldModelSample:
    context_key: tuple[str, str, str]
    viewpoint_id: int
    source_path: Path


class LazyWorldModelDataset(Dataset[dict[str, Any]]):
    """Lazy context/view dataset; NPZ archives are loaded through a small LRU."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        source_by_context: Mapping[tuple[str, str, str], str],
        *,
        use_belief: bool = False,
        rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray] | None = None,
        target_ids: Sequence[int] = range(VIEW_COUNT),
        cache_size: int = 8,
    ) -> None:
        self.rows = list(rows)
        self.source_by_context = dict(source_by_context)
        self.use_belief = bool(use_belief)
        self.rgb_lookup = rgb_lookup
        self.target_ids = tuple(int(v) for v in target_ids)
        self.samples = [
            WorldModelSample((str(row["scene_id"]), str(row["region"]), str(row["record_id"])), view, Path(self.source_by_context[(str(row["scene_id"]), str(row["region"]), str(row["record_id"]))]))
            for row in self.rows for view in self.target_ids
        ]
        self._archives: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.cache_size = max(1, int(cache_size))
        self._row_by_key = {(
            str(row["scene_id"]), str(row["region"]), str(row["record_id"])
        ): row for row in self.rows}

    def __len__(self) -> int:
        return len(self.samples)

    def _archive(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path)
        cached = self._archives.get(key)
        if cached is not None:
            self._archives.move_to_end(key)
            return cached
        with np.load(path, allow_pickle=False) as archive:
            value = {
                "skeleton": np.asarray(archive["skeleton"], dtype=np.float32),
                "viewpoint_ids": np.asarray(archive["viewpoint_ids"], dtype=np.int64),
                "positions": np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32),
            }
        if value["skeleton"].shape != (32, 3, 30, 17) or value["viewpoint_ids"].shape != (32,):
            raise ValueError(f"invalid skeleton archive {path}")
        self._archives[key] = value
        self._archives.move_to_end(key)
        while len(self._archives) > self.cache_size:
            self._archives.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        row = self._row_by_key[sample.context_key]
        archive = self._archive(sample.source_path)
        ids = archive["viewpoint_ids"]
        by_id = {int(view): pos for pos, view in enumerate(ids.tolist())}
        s0_id = int(row["s0_viewpoint_id"]); s1_id = int(row["s1_viewpoint_id"])
        if s0_id not in by_id or s1_id not in by_id or sample.viewpoint_id not in by_id:
            raise ValueError(f"viewpoint alignment failure for {sample.context_key}")
        positions = archive["positions"]
        current_position = positions[by_id[s1_id]]
        history_ids = (s0_id, s1_id)
        history_skeleton = np.stack([archive["skeleton"][by_id[v]] for v in history_ids], axis=0)
        history_descriptor = np.stack([relative_view_descriptor(positions, current_position, v) for v in history_ids], axis=0)
        candidate_descriptor = relative_view_descriptor(positions, current_position, sample.viewpoint_id)
        result: dict[str, Any] = {
            "history_skeleton": torch.from_numpy(history_skeleton),
            "history_descriptor": torch.from_numpy(history_descriptor),
            "candidate_descriptor": torch.from_numpy(candidate_descriptor),
            "target_skeleton": torch.from_numpy(archive["skeleton"][by_id[sample.viewpoint_id]]),
            "context_key": sample.context_key,
            "viewpoint_id": sample.viewpoint_id,
            "label_id": int(row["label_id"]),
        }
        if self.use_belief:
            s0_feature = np.asarray(row["s0_feature"], dtype=np.float32)
            s1_feature = np.asarray(row["s1_feature"], dtype=np.float32)
            class_count = s0_feature.size - 259
            result["history_belief"] = torch.from_numpy(np.concatenate([s0_feature[256:256 + class_count], s1_feature[256:256 + class_count]]))
        if self.rgb_lookup is not None:
            keys = [(*sample.context_key, int(v)) for v in history_ids]
            try:
                result["history_rgb"] = torch.from_numpy(np.stack([np.asarray(self.rgb_lookup[k], dtype=np.float32) for k in keys]))
            except KeyError as exc:
                raise ValueError(f"missing visited RGB embedding {exc}") from exc
        return result


class LazyWorldModelContextDataset(Dataset[dict[str, Any]]):
    """One item per context with all 32 candidate targets for efficient training."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], source_by_context: Mapping[tuple[str, str, str], str], *, use_belief: bool = False, rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray] | None = None, cache_size: int = 8, target_scope: str = "remaining") -> None:
        if target_scope not in {"remaining", "all"}:
            raise ValueError("target_scope must be remaining or all")
        self.rows = list(rows); self.source_by_context = dict(source_by_context); self.use_belief = bool(use_belief); self.rgb_lookup = rgb_lookup; self.target_scope = target_scope; self.cache_size = max(1, int(cache_size)); self._archives: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _archive(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path); cached = self._archives.get(key)
        if cached is not None:
            self._archives.move_to_end(key); return cached
        with np.load(path, allow_pickle=False) as archive:
            value = {"skeleton": np.asarray(archive["skeleton"], dtype=np.float32), "viewpoint_ids": np.asarray(archive["viewpoint_ids"], dtype=np.int64), "positions": np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)}
        self._archives[key] = value; self._archives.move_to_end(key)
        while len(self._archives) > self.cache_size: self._archives.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]; key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"])); archive = self._archive(Path(self.source_by_context[key])); ids = archive["viewpoint_ids"]; by_id = {int(view): pos for pos, view in enumerate(ids.tolist())}; s0_id = int(row["s0_viewpoint_id"]); s1_id = int(row["s1_viewpoint_id"])
        positions = archive["positions"]; current = positions[by_id[s1_id]]; history_ids = (s0_id, s1_id)
        candidate_ids = tuple(range(VIEW_COUNT)) if self.target_scope == "all" else tuple(int(v) for v in row["remaining_candidate_ids"])
        result: dict[str, Any] = {"history_skeleton": torch.from_numpy(np.stack([archive["skeleton"][by_id[v]] for v in history_ids])), "history_descriptor": torch.from_numpy(np.stack([relative_view_descriptor(positions, current, v) for v in history_ids])), "candidate_descriptor": torch.from_numpy(np.stack([relative_view_descriptor(positions, current, v) for v in candidate_ids])), "target_skeleton": torch.from_numpy(np.stack([archive["skeleton"][by_id[v]] for v in candidate_ids])), "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long), "context_key": key, "label_id": int(row["label_id"])}
        if self.use_belief:
            s0_feature = np.asarray(row["s0_feature"], dtype=np.float32); s1_feature = np.asarray(row["s1_feature"], dtype=np.float32); class_count = s0_feature.size - 259; result["history_belief"] = torch.from_numpy(np.concatenate([s0_feature[256:256 + class_count], s1_feature[256:256 + class_count]]))
        if self.rgb_lookup is not None:
            result["history_rgb"] = torch.from_numpy(np.stack([np.asarray(self.rgb_lookup[(*key, int(v))], dtype=np.float32) for v in history_ids]))
        return result


def collate_world_model(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty world-model batch")
    result: dict[str, Any] = {
        "history_skeleton": torch.stack([item["history_skeleton"] for item in batch]),
        "history_descriptor": torch.stack([item["history_descriptor"] for item in batch]),
        "candidate_descriptor": torch.stack([item["candidate_descriptor"] for item in batch]),
        "target_skeleton": torch.stack([item["target_skeleton"] for item in batch]),
        "context_key": [item["context_key"] for item in batch],
        "viewpoint_id": [int(item["viewpoint_id"]) for item in batch],
        "label_id": torch.tensor([int(item["label_id"]) for item in batch], dtype=torch.long),
    }
    if "history_belief" in batch[0]:
        result["history_belief"] = torch.stack([item["history_belief"] for item in batch])
    if "history_rgb" in batch[0]:
        result["history_rgb"] = torch.stack([item["history_rgb"] for item in batch])
    return result


def collate_world_model_context(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty world-model batch")
    max_candidates = max(int(item["candidate_descriptor"].shape[0]) for item in batch)
    candidate_descriptor = torch.zeros((len(batch), max_candidates, 9), dtype=torch.float32)
    target_skeleton = torch.zeros((len(batch), max_candidates, *SKELETON_SHAPE), dtype=torch.float32)
    candidate_ids = torch.full((len(batch), max_candidates), -1, dtype=torch.long)
    candidate_mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = int(item["candidate_descriptor"].shape[0])
        candidate_descriptor[index, :count] = item["candidate_descriptor"]
        target_skeleton[index, :count] = item["target_skeleton"]
        candidate_ids[index, :count] = item["candidate_ids"]
        candidate_mask[index, :count] = True
    result: dict[str, Any] = {"history_skeleton": torch.stack([item["history_skeleton"] for item in batch]), "history_descriptor": torch.stack([item["history_descriptor"] for item in batch]), "candidate_descriptor": candidate_descriptor, "target_skeleton": target_skeleton, "candidate_ids": candidate_ids, "candidate_mask": candidate_mask, "context_key": [item["context_key"] for item in batch], "label_id": torch.tensor([item["label_id"] for item in batch], dtype=torch.long)}
    if "history_belief" in batch[0]: result["history_belief"] = torch.stack([item["history_belief"] for item in batch])
    if "history_rgb" in batch[0]: result["history_rgb"] = torch.stack([item["history_rgb"] for item in batch])
    return result


def world_model_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed SmoothL1 + 0.25 velocity loss and its components."""
    pose = nn.functional.smooth_l1_loss(prediction, target)
    velocity = nn.functional.smooth_l1_loss(prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
    return pose + 0.25 * velocity, pose, velocity
