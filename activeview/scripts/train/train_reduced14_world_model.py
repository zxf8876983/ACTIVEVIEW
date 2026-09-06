#!/usr/bin/env python3
"""Train the frozen-protocol recognition-aware WM-E on reduced14 data."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl, load_stgcn, log_probs
from activeview.data.preprocessing.rgb_cache import spatial_embedding_index
from activeview.methods.world_model.model import (
    CandidateObservationWorldModel,
    LazyWorldModelContextDataset,
    collate_world_model_context,
    world_model_loss,
)
from activeview.recognition.stgcn.model import load_checkpoint

SEED = 42
NUM_CLASSES = 14
EPOCHS = 12
CANDIDATE_CHUNK = 16


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sources(data_root: Path, rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    skeleton_root = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    result: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
        path = skeleton_root / key[0] / key[1] / f"{key[2]}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[key] = str(path)
    return result


def _rgb_lookup(data_root: Path) -> dict[tuple[str, str, str, int], np.ndarray]:
    cache = data_root / "features/dinov2_vitb14_spatial4x4_reduced14_eight_placement/initial_history"
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    rows = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]
    index = spatial_embedding_index(rows)
    return {key: np.asarray(embeddings[pos], dtype=np.float32) for key, pos in index.items()}


def _train_step(model: CandidateObservationWorldModel, teacher: torch.nn.Module, batch: dict[str, Any], device: torch.device, optimizer: torch.optim.Optimizer, grad_scale: int) -> float:
    kwargs = {
        name: batch[name].to(device, non_blocking=True)
        for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")
    }
    kwargs["history_belief"] = batch["history_belief"].to(device, non_blocking=True)
    kwargs["history_rgb"] = batch["history_rgb"].to(device, non_blocking=True)
    target = batch["target_skeleton"].to(device, non_blocking=True)
    mask = batch["candidate_mask"].to(device)
    chunks: list[torch.Tensor] = []
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
        pose_loss, _, _ = world_model_loss(predicted, truth)
        with torch.no_grad():
            truth_logp = torch.log_softmax(teacher(truth), dim=-1)
        predicted_logp = torch.log_softmax(teacher(predicted), dim=-1)
        recognition_loss = torch.nn.functional.kl_div(predicted_logp, truth_logp.exp(), reduction="batchmean")
        loss = pose_loss + 0.10 * recognition_loss
        (loss / float(grad_scale) / float(chunk_total)).backward()
        chunks.append(loss.detach())
    if not chunks:
        return 0.0
    return float(torch.stack(chunks).mean().detach().cpu())


def train(data_root: Path, device: torch.device, batch_size: int, workers: int) -> dict[str, Any]:
    seed_everything()
    policy_root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    rows = load_jsonl(policy_root / "stage_d/features/train.jsonl")
    sources = _sources(data_root, rows)
    rgb_lookup = _rgb_lookup(data_root)
    dataset = LazyWorldModelContextDataset(
        rows, sources, use_belief=True, rgb_lookup=rgb_lookup, target_scope="all", cache_size=64
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        collate_fn=collate_world_model_context, pin_memory=True,
        persistent_workers=workers > 0,
    )
    model = CandidateObservationWorldModel(
        use_belief=True, use_rgb=True, residual=False, num_classes=NUM_CLASSES
    ).to(device)
    teacher, _ = load_checkpoint(
        data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth",
        NUM_CLASSES,
        str(device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    output_dir = data_root / "checkpoints/activeview_reduced14_eight_placement_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    grad_accum = 2
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        count = 0
        for step, batch in enumerate(loader, 1):
            loss = _train_step(model, teacher, batch, device, optimizer, grad_accum)
            if step % grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            size = len(batch["context_key"])
            running += loss * size
            count += size
        result = {"epoch": epoch, "loss": running / max(count, 1)}
        history.append(result)
        torch.save(
            {
                "model_state_dict": model.state_dict(), "variant": "E", "epoch": epoch,
                "seed": SEED, "num_classes": NUM_CLASSES,
            }, output_dir / "wm_e_last.pth"
        )
        print(f"WM-E epoch {epoch}/{EPOCHS} loss={result['loss']:.6f}", flush=True)
    summary = {
        "variant": "E", "train_contexts": len(rows), "train_targets": len(rows) * 32,
        "batch_size": batch_size, "gradient_accumulation": grad_accum,
        "epochs": EPOCHS, "final_loss": history[-1]["loss"], "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str((output_dir / "wm_e_last.pth").resolve()),
        "num_classes": NUM_CLASSES, "test_used": False,
    }
    (output_dir / "wm_e_training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; WM-E requires GPU")
    train(args.data_root.resolve(), device, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
