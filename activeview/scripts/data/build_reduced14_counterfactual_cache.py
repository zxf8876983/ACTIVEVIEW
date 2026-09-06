#!/usr/bin/env python3
"""Build frozen WM-E counterfactual recognition caches for one split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.rgb_cache import spatial_embedding_index
from activeview.methods.world_model.model import CandidateObservationWorldModel, LazyWorldModelContextDataset, collate_world_model_context
from activeview.recognition.stgcn.model import load_checkpoint

NUM_CLASSES = 14
VIEW_COUNT = 32


def _sources(data_root: Path, rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    return {
        (str(row["scene_id"]), str(row["region"]), str(row["record_id"])): str(
            root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"
        ) for row in rows
    }


def _rgb_lookup(data_root: Path) -> dict[tuple[str, str, str, int], np.ndarray]:
    cache = data_root / "features/dinov2_vitb14_spatial4x4_reduced14_eight_placement/initial_history"
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    rows = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]
    index = spatial_embedding_index(rows)
    return {key: np.asarray(embeddings[pos], dtype=np.float32) for key, pos in index.items()}


def _logp(model: torch.nn.Module, skeleton: torch.Tensor, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, skeleton.shape[0], 512):
            values.append(torch.log_softmax(model(skeleton[start:start + 512].to(device)), dim=-1).cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def build_split(data_root: Path, split: str, wm_path: Path, output: Path, device: torch.device, batch_size: int, workers: int) -> dict[str, Any]:
    policy_root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    rows = load_jsonl(policy_root / "stage_d/features" / f"{split}.jsonl")
    dataset = LazyWorldModelContextDataset(
        rows, _sources(data_root, rows), use_belief=True, rgb_lookup=_rgb_lookup(data_root),
        target_scope="all", cache_size=64,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_world_model_context, pin_memory=True, persistent_workers=workers > 0)
    payload = torch.load(wm_path, map_location=device, weights_only=False)
    model = CandidateObservationWorldModel(use_belief=True, use_rgb=True, residual=False, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    stgcn, _ = load_checkpoint(data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth", NUM_CLASSES, str(device))
    current_s0: list[np.ndarray] = []
    current_s1: list[np.ndarray] = []
    imagined: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    labels: list[int] = []
    episode_ids: list[str] = []
    row_offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch_count = len(batch["context_key"])
            history = batch["history_skeleton"].float()
            history_logp = _logp(stgcn, history.reshape(-1, 3, 30, 17), device).reshape(len(history), 2, NUM_CLASSES)
            current_s0.append(history_logp[:, 0]); current_s1.append(history_logp[:, 1])
            target = batch["target_skeleton"].float()
            truth.append(_logp(stgcn, target.reshape(-1, 3, 30, 17), device).reshape(len(target), VIEW_COUNT, NUM_CLASSES))
            descriptors.append(batch["candidate_descriptor"].numpy().astype(np.float32))
            labels.extend(batch["label_id"].numpy().astype(np.int64).tolist())
            episode_ids.extend(str(rows[row_offset + i]["episode_id"]) for i in range(batch_count))
            row_offset += batch_count
            kwargs = {
                "history_skeleton": history.to(device, non_blocking=True),
                "history_descriptor": batch["history_descriptor"].to(device, non_blocking=True),
                "history_belief": batch["history_belief"].to(device, non_blocking=True),
                "history_rgb": batch["history_rgb"].to(device, non_blocking=True),
            }
            predicted_parts: list[np.ndarray] = []
            for start in range(0, VIEW_COUNT, 8):
                prediction = model(candidate_descriptor=batch["candidate_descriptor"][:, start:start + 8].to(device), **kwargs)
                predicted_parts.append(_logp(stgcn, prediction.reshape(-1, 3, 30, 17), device).reshape(batch_count, -1, NUM_CLASSES))
            imagined.append(np.concatenate(predicted_parts, axis=1))
    result = {
        "episode_ids": np.asarray(episode_ids, dtype="U"),
        "current_logp_s0": np.concatenate(current_s0),
        "current_logp_s1": np.concatenate(current_s1),
        "imagined_logp": np.concatenate(imagined),
        "true_logp": np.concatenate(truth),
        "candidate_descriptor": np.concatenate(descriptors),
        "label_id": np.asarray(labels, dtype=np.int64),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    summary = {"split": split, "contexts": len(episode_ids), "shape_imagined": list(result["imagined_logp"].shape), "wm_checkpoint": str(wm_path.resolve()), "test_used": split == "test"}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; counterfactual cache requires GPU")
    build_split(args.data_root.resolve(), args.split, args.wm_checkpoint.resolve(), args.output.resolve(), device, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
