#!/usr/bin/env python3
"""Build Train/Val counterfactual recognition caches for reduced12 WM-E."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.rgb_cache import spatial_embedding_index
from activeview.methods.active_view.geometry import context_key
from activeview.methods.world_model.model import CandidateObservationWorldModel, LazyWorldModelContextDataset, collate_world_model_context
from activeview.recognition.stgcn.model import load_checkpoint

NUM_CLASSES = 12
VIEW_COUNT = 32


def _sources(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/habitat-train/00006-00087"
    return {context_key(row): str(root / context_key(row)[0] / context_key(row)[1] / f"{context_key(row)[2]}.npz") for row in rows}


def _logp(model: torch.nn.Module, skeleton: torch.Tensor, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, skeleton.shape[0], 512):
            values.append(torch.log_softmax(model(skeleton[start:start + 512].to(device)), dim=-1).cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def _rgb_lookup(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], np.ndarray]:
    cache = data_root / "features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/initial_history"
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    manifest = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]
    index = spatial_embedding_index(manifest)
    required = {
        (*context_key(row), int(row["s0_viewpoint_id"])) for row in rows
    } | {
        (*context_key(row), int(row["s1_viewpoint_id"])) for row in rows
    }
    if not required.issubset(index):
        raise RuntimeError(f"visited DINO cache incomplete: {len(required - set(index))} missing")
    return {key: np.asarray(embeddings[pos], dtype=np.float32) for key, pos in index.items() if key in required}


def build_split(
    data_root: Path,
    split: str,
    wm_path: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    use_rgb: bool = False,
) -> dict[str, Any]:
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    rows = load_jsonl(policy_root / "stage_d/features" / f"{split}.jsonl")
    rgb_lookup = _rgb_lookup(data_root, rows) if use_rgb else None
    dataset = LazyWorldModelContextDataset(rows, _sources(data_root, rows), use_belief=True, rgb_lookup=rgb_lookup, target_scope="all", cache_size=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_world_model_context, pin_memory=True, persistent_workers=workers > 0)
    payload = torch.load(wm_path, map_location=device, weights_only=False)
    model = CandidateObservationWorldModel(use_belief=True, use_rgb=use_rgb, residual=False, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    stgcn, _ = load_checkpoint(data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth", NUM_CLASSES, str(device))
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
            history = batch["history_skeleton"].float()
            batch_count = len(batch["context_key"])
            history_logp = _logp(stgcn, history.reshape(-1, 3, 30, 17), device).reshape(batch_count, 2, NUM_CLASSES)
            current_s0.append(history_logp[:, 0]); current_s1.append(history_logp[:, 1])
            target = batch["target_skeleton"].float()
            truth.append(_logp(stgcn, target.reshape(-1, 3, 30, 17), device).reshape(batch_count, VIEW_COUNT, NUM_CLASSES))
            descriptors.append(batch["candidate_descriptor"].numpy().astype(np.float32))
            labels.extend(batch["label_id"].numpy().astype(np.int64).tolist())
            episode_ids.extend(str(rows[row_offset + i]["episode_id"]) for i in range(batch_count))
            row_offset += batch_count
            kwargs = {
                "history_skeleton": history.to(device, non_blocking=True),
                "history_descriptor": batch["history_descriptor"].to(device, non_blocking=True),
                "history_belief": batch["history_belief"].to(device, non_blocking=True),
            }
            if use_rgb:
                kwargs["history_rgb"] = batch["history_rgb"].to(device, non_blocking=True)
            parts: list[np.ndarray] = []
            for start in range(0, VIEW_COUNT, 8):
                prediction = model(candidate_descriptor=batch["candidate_descriptor"][:, start:start + 8].to(device), **kwargs)
                parts.append(_logp(stgcn, prediction.reshape(-1, 3, 30, 17), device).reshape(batch_count, -1, NUM_CLASSES))
            imagined.append(np.concatenate(parts, axis=1))
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
    summary = {"split": split, "contexts": len(episode_ids), "shape_imagined": list(result["imagined_logp"].shape), "wm_checkpoint": str(wm_path.resolve()), "use_rgb": use_rgb, "target_scope": "all", "test_used": False}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use-rgb", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; counterfactual cache requires GPU")
    if args.output is None and args.output_root is None:
        raise ValueError("provide --output or --output-root")
    output = args.output.resolve() if args.output is not None else Path(".")
    if args.output_root is not None:
        output = args.output_root.resolve() / f"{args.split}.npz"
    build_split(args.data_root.resolve(), args.split, args.wm_checkpoint.resolve(), output, device, args.batch_size, args.workers, args.use_rgb)


if __name__ == "__main__":
    main()
