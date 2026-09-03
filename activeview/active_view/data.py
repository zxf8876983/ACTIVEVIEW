"""Data and frozen artifact loaders used by the final ActiveView pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from activeview.action_recognition.st_gcn_model import STGCN
from activeview.active_view.geometry import ContextKey, context_key
from activeview.active_view.stage_d_dataset import load_jsonl, load_pairwise_geodesic
from activeview.active_view.stage_d_world_model import CandidateObservationWorldModel
from activeview.core.paths import get_data_root
from activeview.scripts.build_stage_b_utility_labels import _load_model


def rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError("Test rows are only available through the explicit final runner")
    path = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/features" / f"{split}.jsonl"
    values = load_jsonl(path)
    if any(str(row.get("policy_split", "")).lower() != split for row in values):
        raise ValueError(f"explicit policy_split={split} required: {path}")
    return values


def episode_sources(data_root: Path, split: str) -> dict[ContextKey, str]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split}")
    path = data_root / "datasets/policy_v11_5/episodes" / f"{split}_episodes.jsonl"
    values = load_jsonl(path)
    output: dict[ContextKey, str] = {}
    for episode in values:
        if str(episode.get("policy_split", "")).lower() != split:
            raise ValueError(f"explicit policy_split={split} required: {path}")
        key = context_key(episode)
        source = str(episode["current_view"]["skeleton_source_path"])
        if key in output and Path(output[key]).resolve() != Path(source).resolve():
            raise ValueError(f"source path mismatch for {key}")
        output[key] = source
    return output


def load_stage_d_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_stgcn(data_root: Path, device: torch.device) -> STGCN:
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text(encoding="utf-8"))
    mapping = json.loads(Path(summary["label_mapping"]).read_text(encoding="utf-8"))
    model, _ = _load_model(Path(summary["stgcn_checkpoint"]), len(mapping), str(device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def log_probs(model: STGCN, skeletons: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(skeletons), 1024):
            batch = torch.from_numpy(skeletons[start : start + 1024]).float().to(device)
            values.append(torch.log_softmax(model(batch), dim=-1).cpu().numpy())
    return np.concatenate(values, axis=0)


def load_wm_e(checkpoint: Path, device: torch.device) -> CandidateObservationWorldModel:
    model = CandidateObservationWorldModel(use_belief=True, use_rgb=True, residual=False).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def load_observation_archive(
    source_path: Path, cache: Mapping[str, np.ndarray], context: ContextKey,
) -> dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
    cache_ids = np.asarray(cache.get("viewpoint_ids", ids), dtype=np.int64)
    if skeleton.shape != (32, 3, 30, 17) or not np.array_equal(ids, cache_ids):
        raise ValueError(f"skeleton/cache alignment failure for {context}")
    return {"skeleton": skeleton, "viewpoint_ids": ids, "positions": positions, "logp": np.asarray(cache["true_logp"], dtype=np.float32)}


__all__ = ["episode_sources", "get_data_root", "load_jsonl", "load_observation_archive", "load_stage_d_cache", "load_stgcn", "load_wm_e", "log_probs", "rows"]
