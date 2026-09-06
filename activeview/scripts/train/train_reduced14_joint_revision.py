#!/usr/bin/env python3
"""Train Multi-Positive Joint Revision on the reduced14 counterfactual cache."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import candidate_order, load_pairwise_and_azimuths
from activeview.methods.joint_revision.model import JointRevision

SEED = 42
NUM_CLASSES = 14
VIEW_COUNT = 32
EPOCHS = 20


def _seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class _Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: tuple[np.ndarray, ...]) -> None:
        self.current, self.candidates, self.mask, self.fallback, self.positive, self.labels = arrays

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "current": torch.from_numpy(self.current[index]),
            "candidates": torch.from_numpy(self.candidates[index]),
            "mask": torch.from_numpy(self.mask[index]),
            "fallback": torch.tensor(self.fallback[index]),
            "positive": torch.from_numpy(self.positive[index]),
            "label": torch.tensor(self.labels[index]),
        }


def _source_map(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    return {
        (str(row["scene_id"]), str(row["region"]), str(row["record_id"])): str(root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz")
        for row in rows
    }


def _orders(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    source = _source_map(data_root, rows)
    pair, azimuth = load_pairwise_and_azimuths(
        data_root, rows, source,
        pair_root=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1/pairwise_viewpoint_geodesic",
    )
    output: dict[str, list[int]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]))
        output[str(row["episode_id"])] = candidate_order(
            row, int(row["s1_viewpoint_id"]),
            {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])},
            pair[key], azimuth[key],
        )
    return output


def _examples(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]]) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    current_all: list[np.ndarray] = []; candidates_all: list[np.ndarray] = []; masks: list[np.ndarray] = []
    fallbacks: list[int] = []; positives: list[np.ndarray] = []; labels: list[int] = []
    positive_counts: list[int] = []; multi = single = none = 0
    max_actions = 1 + VIEW_COUNT - 2
    for row in rows:
        i = index[str(row["episode_id"])]
        candidates = [int(v) for v in orders[str(row["episode_id"])]]
        current_logp = np.asarray(cache["current_logp_s1"][i], dtype=np.float32)
        probs0 = np.exp(cache["current_logp_s0"][i]); probs1 = np.exp(current_logp)
        current = np.asarray([*probs0, *probs1, -np.sum(probs0 * cache["current_logp_s0"][i]), -np.sum(probs1 * current_logp), np.max(probs0), np.max(probs1), np.sort(probs1)[-1] - np.sort(probs1)[-2], np.sort(probs0)[-1] - np.sort(probs0)[-2]], dtype=np.float32)
        values = [np.concatenate([current_logp, np.zeros(9, dtype=np.float32), [1.0]])]
        values.extend(np.concatenate([cache["imagined_logp"][i, c], cache["candidate_descriptor"][i, c], [0.0]]) for c in candidates)
        action_correct = np.asarray([int(np.argmax(current_logp) == int(row["label_id"]))] + [int(np.argmax(cache["true_logp"][i, c]) == int(row["label_id"])) for c in candidates], dtype=np.float32)
        positive_count = int(action_correct.sum())
        if positive_count:
            positive_counts.append(positive_count); multi += int(positive_count > 1); single += int(positive_count == 1)
            fallback = int(np.flatnonzero(action_correct)[0])
        else:
            none += 1
            scores = [float(current_logp[int(row["label_id"])])] + [float(cache["true_logp"][i, c, int(row["label_id"])]) for c in candidates]
            fallback = int(np.argmax(scores))
        padded = np.zeros((max_actions, NUM_CLASSES + 10), dtype=np.float32); padded[:len(values)] = np.asarray(values, dtype=np.float32)
        positive = np.zeros(max_actions, dtype=np.float32); positive[:len(values)] = action_correct
        mask = np.asarray([True] * len(values) + [False] * (max_actions - len(values)), dtype=bool)
        current_all.append(current); candidates_all.append(padded); masks.append(mask); fallbacks.append(fallback); positives.append(positive); labels.append(int(row["label_id"]))
    stats = {"contexts": len(labels), "positive_contexts": multi + single, "multi_positive_contexts": multi, "single_positive_contexts": single, "no_positive_contexts": none, "mean_positive_count": float(np.mean(positive_counts)) if positive_counts else 0.0}
    return (np.asarray(current_all), np.asarray(candidates_all), np.asarray(masks), np.asarray(fallbacks), np.asarray(positives), np.asarray(labels, dtype=np.int64)), stats


def train(data_root: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    _seed()
    root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    rows = load_jsonl(root / "stage_d/features/train.jsonl")
    cache = {k: np.asarray(v) for k, v in np.load(root / "counterfactual_cache/train.npz", allow_pickle=False).items()}
    orders = _orders(data_root, rows)
    arrays, stats = _examples(rows, cache, orders)
    loader = DataLoader(_Dataset(arrays), batch_size=batch_size, shuffle=True)
    model = JointRevision(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses: list[float] = []
        for batch in loader:
            current = batch["current"].float().to(device); candidates = batch["candidates"].float().to(device); mask = batch["mask"].bool().to(device); fallback = batch["fallback"].long().to(device); positive = batch["positive"].float().to(device); labels = batch["label"].long().to(device)
            scores, posterior = model(current, candidates, mask)
            valid_scores = scores.masked_fill(~mask, -1e9); all_lse = torch.logsumexp(valid_scores, dim=1); positive_mask = positive.bool() & mask; positive_lse = torch.logsumexp(scores.masked_fill(~positive_mask, -1e9), dim=1)
            has_positive = positive_mask.any(dim=1); main = torch.where(has_positive, all_lse - positive_lse, nn.functional.cross_entropy(valid_scores, fallback, reduction="none"))
            bce_all = nn.functional.binary_cross_entropy_with_logits(scores, positive, reduction="none"); bce = (bce_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            target = labels.unsqueeze(1).expand(-1, posterior.size(1)); post_all = nn.functional.cross_entropy(posterior.reshape(-1, NUM_CLASSES), target.reshape(-1), reduction="none").reshape_as(scores); post = (post_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            loss = (main + 0.25 * bce + 0.05 * post).mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses))); print(f"JR epoch {epoch}/{EPOCHS} loss={history[-1]:.6f}", flush=True)
    output = data_root / "checkpoints/activeview_reduced14_eight_placement_v1/joint_revision_multi_positive.pth"; output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS, "num_classes": NUM_CLASSES}, output)
    result = {**stats, "final_loss": history[-1], "loss_history": history, "checkpoint": str(output.resolve()), "test_used": False}
    (output.parent / "joint_revision_training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=512); args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; JR requires GPU")
    train(args.data_root.resolve(), device, args.batch_size)


if __name__ == "__main__": main()
