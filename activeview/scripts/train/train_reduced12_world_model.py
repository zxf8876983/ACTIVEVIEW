#!/usr/bin/env python3
"""Train the frozen Recognition-aware WM-E for the reduced12 protocol.

This entry point intentionally accepts only Train/Val artifacts.  The current
eight-placement skeleton dataset has no RGB cache, so this formal baseline
uses its frozen history skeleton/belief and 9-D candidate geometry inputs.
"""

from __future__ import annotations

import argparse
import json
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
from activeview.methods.active_view.geometry import candidate_order, context_key, load_pairwise_and_azimuths
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


def _sources(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/habitat-train/00006-00087"
    result: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = context_key(row)
        path = root / key[0] / key[1] / f"{key[2]}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[key] = str(path)
    return result


def _loader(data_root: Path, split: str, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    rows = load_jsonl(policy_root / "stage_d/features" / f"{split}.jsonl")
    sources = _sources(data_root, rows)
    pairwise, azimuths = load_pairwise_and_azimuths(
        data_root,
        rows,
        sources,
        pair_root=policy_root / "pairwise_viewpoint_geodesic",
    )
    legal: dict[tuple[str, str, str], tuple[int, ...]] = {}
    for row in rows:
        key = context_key(row)
        legal[key] = tuple(candidate_order(
            row,
            int(row["s1_viewpoint_id"]),
            {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])},
            pairwise[(key[0], key[1])],
            azimuths[(key[0], key[1])],
        ))
    dataset = LazyWorldModelContextDataset(
        rows,
        sources,
        use_belief=True,
        # WM-E is trained on the legal unvisited candidate set used by the
        # policy, rather than spending compute on permanently excluded views.
        target_scope="remaining",
        legal_candidate_ids=legal,
        cache_size=64,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_world_model_context,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _batch_loss(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    train: bool,
) -> float:
    kwargs = {
        name: batch[name].to(device, non_blocking=True)
        for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")
    }
    kwargs["history_belief"] = batch["history_belief"].to(device, non_blocking=True)
    target = batch["target_skeleton"].to(device, non_blocking=True)
    mask = batch["candidate_mask"].to(device)
    total = torch.zeros((), device=device)
    count = 0
    for start in range(0, int(kwargs["candidate_descriptor"].shape[1]), CANDIDATE_CHUNK):
        stop = min(start + CANDIDATE_CHUNK, int(kwargs["candidate_descriptor"].shape[1]))
        chunk_kwargs = dict(kwargs)
        chunk_kwargs["candidate_descriptor"] = kwargs["candidate_descriptor"][:, start:stop]
        prediction = model(**chunk_kwargs)
        valid = mask[:, start:stop].reshape(-1)
        if not bool(valid.any()):
            continue
        predicted = prediction.reshape(-1, 3, 30, 17)[valid]
        truth = target[:, start:stop].reshape(-1, 3, 30, 17)[valid]
        pose_total, _pose, _velocity = world_model_loss(predicted, truth)
        with torch.no_grad():
            true_logp = torch.log_softmax(teacher(truth), dim=-1)
        predicted_logp = torch.log_softmax(teacher(predicted), dim=-1)
        recognition = F.kl_div(predicted_logp, true_logp.exp(), reduction="batchmean")
        total = total + pose_total + 0.10 * recognition
        count += 1
    if count == 0:
        return 0.0
    value = total / count
    if train:
        value.backward()
    return float(value.detach().cpu())


def train(data_root: Path, device: torch.device, batch_size: int, workers: int) -> dict[str, Any]:
    _seed()
    train_loader = _loader(data_root, "train", batch_size, workers, True)
    val_loader = _loader(data_root, "val", batch_size, workers, False)
    checkpoint = data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth"
    teacher, _ = load_checkpoint(checkpoint, NUM_CLASSES, str(device))
    model = CandidateObservationWorldModel(use_belief=True, use_rgb=False, residual=False, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    output_dir = data_root / "checkpoints/activeview_reduced12_eight_placement_v1/wm_e"
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_path = output_dir / "wm_e_best.pth"
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(_batch_loss(model, teacher, batch, device, True))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(_batch_loss(model, teacher, batch, device, False))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "variant": "Recognition-aware-WM-E",
                "num_classes": NUM_CLASSES,
                "seed": SEED,
                "epoch": epoch,
                "use_rgb": False,
                "rgb_dino_available": False,
            }, best_path)
    summary = {
        "protocol": "reduced12-eight-placement-v1",
        "variant": "Recognition-aware WM-E",
        "train_contexts": len(train_loader.dataset),
        "val_contexts": len(val_loader.dataset),
        "epochs": EPOCHS,
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "use_rgb": False,
        "rgb_dino_available": False,
        "best_val_loss": best_val,
        "history": history,
        "checkpoint": str(best_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "test_used": False,
    }
    (output_dir / "wm_e_training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
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
