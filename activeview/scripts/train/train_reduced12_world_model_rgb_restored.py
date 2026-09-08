#!/usr/bin/env python3
"""Train the formal reduced12 RGB-conditioned Recognition-aware WM-E.

This is deliberately separate from the historical no-RGB reduced12 entry point.
It restores the pre-reduced12 protocol without changing that checkpoint or its
cache: belief and visited DINO spatial tokens are required, all 32 candidate
targets are used, and only Train/Val artifacts are read.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.rgb_cache import spatial_embedding_index
from activeview.methods.active_view.geometry import context_key
from activeview.methods.world_model.model import (
    CandidateObservationWorldModel,
    LazyWorldModelContextDataset,
    collate_world_model_context,
    world_model_loss,
)
from activeview.recognition.stgcn.model import load_checkpoint

SEED = 42
NUM_CLASSES = 12
EPOCHS = 12
CANDIDATE_CHUNK = 16


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _source_paths(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/habitat-train/00006-00087"
    result: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = context_key(row)
        path = root / key[0] / key[1] / f"{key[2]}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[key] = str(path)
    return result


def _rgb_lookup(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], np.ndarray]:
    cache = data_root / "features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/initial_history"
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    manifest = [
        json.loads(line)
        for line in (cache / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = spatial_embedding_index(manifest)
    required = {
        (*context_key(row), int(row["s0_viewpoint_id"])) for row in rows
    } | {
        (*context_key(row), int(row["s1_viewpoint_id"])) for row in rows
    }
    if not required.issubset(index):
        missing = sorted(required - set(index))[:5]
        raise RuntimeError(f"visited DINO cache incomplete; missing={missing}")
    return {key: np.asarray(embeddings[position], dtype=np.float32) for key, position in index.items() if key in required}


def _loader(
    data_root: Path,
    split: str,
    batch_size: int,
    workers: int,
) -> DataLoader:
    policy = data_root / "datasets/policy_reduced12_eight_placement_v1"
    rows = load_jsonl(policy / "stage_d/features" / f"{split}.jsonl")
    source_paths = _source_paths(data_root, rows)
    rgb_lookup = _rgb_lookup(data_root, rows)
    dataset = LazyWorldModelContextDataset(
        rows,
        source_paths,
        use_belief=True,
        rgb_lookup=rgb_lookup,
        target_scope="all",
        cache_size=64,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        collate_fn=collate_world_model_context,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _loss(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    backward: bool,
) -> float:
    kwargs = {
        name: batch[name].to(device, non_blocking=True)
        for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")
    }
    kwargs["history_belief"] = batch["history_belief"].to(device, non_blocking=True)
    kwargs["history_rgb"] = batch["history_rgb"].to(device, non_blocking=True)
    target = batch["target_skeleton"].to(device, non_blocking=True)
    mask = batch["candidate_mask"].to(device)
    total = torch.zeros((), device=device)
    chunks = 0
    candidate_count = int(kwargs["candidate_descriptor"].shape[1])
    chunk_total = max(1, math.ceil(candidate_count / CANDIDATE_CHUNK))
    for start in range(0, candidate_count, CANDIDATE_CHUNK):
        stop = min(start + CANDIDATE_CHUNK, candidate_count)
        chunk_kwargs = dict(kwargs)
        chunk_kwargs["candidate_descriptor"] = kwargs["candidate_descriptor"][:, start:stop]
        prediction = model(**chunk_kwargs)
        valid = mask[:, start:stop].reshape(-1)
        if not bool(valid.any()):
            continue
        predicted = prediction.reshape(-1, 3, 30, 17)[valid]
        truth = target[:, start:stop].reshape(-1, 3, 30, 17)[valid]
        pose, _, _ = world_model_loss(predicted, truth)
        with torch.no_grad():
            true_logp = torch.log_softmax(teacher(truth), dim=-1)
        predicted_logp = torch.log_softmax(teacher(predicted), dim=-1)
        recognition = F.kl_div(predicted_logp, true_logp.exp(), reduction="batchmean")
        value = pose + 0.10 * recognition
        if backward:
            (value / float(chunk_total)).backward()
        total = total + value.detach()
        chunks += 1
    return float((total / max(chunks, 1)).cpu())


def train(data_root: Path, device: torch.device, batch_size: int, workers: int) -> dict[str, Any]:
    _seed()
    train_loader = _loader(data_root, "train", batch_size, workers)
    val_loader = _loader(data_root, "val", batch_size, max(0, min(workers, 2)))
    teacher_path = data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth"
    teacher, _ = load_checkpoint(teacher_path, NUM_CLASSES, str(device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = CandidateObservationWorldModel(
        use_belief=True,
        use_rgb=True,
        residual=False,
        num_classes=NUM_CLASSES,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    output = data_root / "checkpoints/activeview_reduced12_eight_placement_v1_rgb_restored/wm_e"
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "wm_e_best.pth"
    best_val = float("inf")
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_values: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            value = _loss(model, teacher, batch, device, True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_values.append(value)
        model.eval()
        val_values: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                val_values.append(_loss(model, teacher, batch, device, False))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_values)), "val_loss": float(np.mean(val_values))}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": "Recognition-aware-WM-E",
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                    "epoch": epoch,
                    "use_belief": True,
                    "use_rgb": True,
                    "residual": False,
                    "target_scope": "all",
                },
                best_path,
            )
    summary = {
        "protocol": "reduced12-eight-placement-v1",
        "variant": "Recognition-aware WM-E RGB restored",
        "train_contexts": len(train_loader.dataset),
        "val_contexts": len(val_loader.dataset),
        "epochs": EPOCHS,
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "use_belief": True,
        "use_rgb": True,
        "residual": False,
        "target_scope": "all",
        "loss": "SmoothL1 + 0.25 velocity + 0.10 frozen ST-GCN recognition KL",
        "best_val_loss": best_val,
        "history": history,
        "checkpoint": str(best_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "test_used": False,
    }
    (output / "wm_e_training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; RGB-restored WM-E requires GPU")
    train(args.data_root.resolve(), device, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
