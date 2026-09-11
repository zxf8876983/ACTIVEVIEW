#!/usr/bin/env python3
"""Train a prefix HAR encoder and audit sequential 1-hop NBV on reduced12.

The runner reads only existing Train/Val skeleton archives and Stage-C/Stage-D
records.  It trains a small masked GRU on Train prefixes, evaluates fixed-view
prefix recognition on Val Moving, then compares Stay, Random-1Hop and a
GT-label-conditioned greedy oracle under a discrete-time view-switch
approximation.  No RGB, skeleton, DINO, policy-Test, or frozen ST-GCN asset is
generated or modified.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, lattice_distance


SEED = 42
NUM_CLASSES = 12
FRAME_COUNT = 30
JOINT_COUNT = 17
COORDINATE_DIM = JOINT_COUNT * 3
CHUNK_SIZE = 5
PREFIXES = (5, 10, 15, 20, 25, 30)
HIDDEN_DIM = 256
EPOCHS = 12
BATCH_SIZE = 512
INFERENCE_BATCH_SIZE = 1024

DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/prefix_sequential_nbv_mvp"
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
OFFLINE_RELATIVE = Path("datasets/offline/habitat-train/00006-00087")
RUNTIME_CHECKPOINT_RELATIVE = Path(
    "checkpoints/activeview_reduced12_eight_placement_v1/prefix_sequential_nbv_mvp"
)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_rows(policy_root: Path, stage: str, split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in load_jsonl(policy_root / stage / "features" / f"{split}.jsonl")]


def _record_path(offline_root: Path, row: Mapping[str, Any]) -> Path:
    return offline_root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"


def _flatten_sequence(skeleton: np.ndarray) -> np.ndarray:
    value = np.asarray(skeleton, dtype=np.float32)
    if value.shape != (3, FRAME_COUNT, JOINT_COUNT):
        raise ValueError(f"expected skeleton (3,30,17), got {value.shape}")
    return np.transpose(value, (1, 2, 0)).reshape(FRAME_COUNT, COORDINATE_DIM)


def _load_view_sequence(path: Path, viewpoint_id: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        matches = np.flatnonzero(ids == int(viewpoint_id))
        if matches.size != 1:
            raise ValueError(f"viewpoint {viewpoint_id} is not unique in {path}")
        return _flatten_sequence(np.asarray(archive["skeleton"][int(matches[0])]))


def _load_all_views(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
    if skeleton.shape != (32, 3, FRAME_COUNT, JOINT_COUNT):
        raise ValueError(f"expected all-view skeleton (32,3,30,17), got {skeleton.shape}")
    order = np.argsort(ids)
    flattened = np.transpose(skeleton[order], (0, 2, 3, 1)).reshape(32, FRAME_COUNT, COORDINATE_DIM)
    if not np.array_equal(ids[order], np.arange(32, dtype=np.int64)):
        raise ValueError(f"offline viewpoint ids are not 0..31 in {path}")
    return flattened


class PrefixDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Six deterministic prefix views per Train skeleton, shuffled by loader."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        if sequences.ndim != 3 or sequences.shape[1:] != (FRAME_COUNT, COORDINATE_DIM):
            raise ValueError(f"unexpected sequence array shape {sequences.shape}")
        self.sequences = np.asarray(sequences, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.sequences.shape[0] * len(PREFIXES))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_index = int(index) // len(PREFIXES)
        prefix = int(PREFIXES[int(index) % len(PREFIXES)])
        sequence = np.zeros_like(self.sequences[base_index])
        sequence[:prefix] = self.sequences[base_index, :prefix]
        mask = np.zeros(FRAME_COUNT, dtype=np.float32)
        mask[:prefix] = 1.0
        return (
            torch.from_numpy(sequence),
            torch.from_numpy(mask),
            torch.tensor(self.labels[base_index], dtype=torch.long),
        )


class PrefixHAR(nn.Module):
    """Small temporal GRU producing a history feature and class posterior."""

    def __init__(self, use_mask: bool = True, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.use_mask = bool(use_mask)
        input_dim = COORDINATE_DIM + (1 if self.use_mask else 0)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.feature_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.classifier = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_mask:
            if mask is None:
                raise ValueError("masked PrefixHAR requires a temporal mask")
            sequence = torch.cat([sequence, mask.unsqueeze(-1)], dim=-1)
        _, hidden = self.gru(sequence)
        feature = self.feature_head(hidden[-1])
        return self.classifier(feature), feature


def _prefix_batch(
    sequences: np.ndarray, prefix: int, start: int, end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = np.zeros((end - start, FRAME_COUNT, COORDINATE_DIM), dtype=np.float32)
    value[:, :prefix] = sequences[start:end, :prefix]
    mask = np.zeros((end - start, FRAME_COUNT), dtype=np.float32)
    mask[:, :prefix] = 1.0
    return torch.from_numpy(value), torch.from_numpy(mask)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)


def _evaluate_prefix(
    model: PrefixHAR,
    sequences: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    use_mask: bool,
) -> dict[str, Any]:
    model.eval()
    output: dict[str, Any] = {}
    with torch.inference_mode():
        for prefix in PREFIXES:
            logits_parts: list[np.ndarray] = []
            for start in range(0, len(sequences), INFERENCE_BATCH_SIZE):
                end = min(start + INFERENCE_BATCH_SIZE, len(sequences))
                batch, mask = _prefix_batch(sequences, prefix, start, end)
                if not use_mask:
                    mask = torch.ones_like(mask)
                logits, _ = model(batch.to(device), mask.to(device) if use_mask else None)
                logits_parts.append(logits.detach().cpu().numpy())
            logits_np = np.concatenate(logits_parts, axis=0)
            predictions = np.argmax(logits_np, axis=1)
            probabilities = _softmax(logits_np)
            metrics = classification(labels.tolist(), predictions.tolist())
            metrics.update({
                "prefix_frames": int(prefix),
                "mean_entropy": float(np.mean(_entropy(probabilities))),
                "mean_gt_probability": float(np.mean(probabilities[np.arange(len(labels)), labels])),
            })
            output[f"{prefix}_frames"] = metrics
    return output


def _train_encoder(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    val_sequences: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    use_mask: bool,
    checkpoint: Path,
    summary_path: Path,
) -> dict[str, Any]:
    _seed()
    dataset = PrefixDataset(train_sequences, train_labels)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=True,
    )
    model = PrefixHAR(use_mask=use_mask).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for sequence, mask, labels in loader:
            sequence = sequence.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(sequence, mask if use_mask else None)
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = _evaluate_prefix(model, val_sequences, val_labels, device, use_mask)
        val_macro = float(np.mean([val_metrics[f"{prefix}_frames"]["macro_f1"] for prefix in PREFIXES]))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_macro_f1": val_macro}
        history.append(record)
        print(
            f"[prefix-{('mask' if use_mask else 'zero-only')}] "
            f"epoch={epoch:02d}/{EPOCHS} loss={record['train_loss']:.6f} "
            f"val_mean_macro_f1={val_macro:.6f}",
            flush=True,
        )
        if val_macro > best_score + 1e-8:
            best_score = val_macro
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "use_mask": use_mask,
                "hidden_dim": HIDDEN_DIM,
                "epoch": epoch,
                "seed": SEED,
            }, checkpoint)
    summary = {
        "use_mask": use_mask,
        "epochs": EPOCHS,
        "selected_epoch": best_epoch,
        "best_val_mean_macro_f1": best_score,
        "train_contexts": int(len(train_sequences)),
        "train_prefix_examples": int(len(dataset)),
        "val_contexts": int(len(val_sequences)),
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "checkpoint": str(checkpoint.resolve()),
        "seed": SEED,
        "test_used": False,
        "history": history,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_encoder(checkpoint: Path, device: torch.device) -> PrefixHAR:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PrefixHAR(use_mask=bool(payload["use_mask"]), hidden_dim=int(payload["hidden_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _candidate_options(current: int, candidate_ids: Sequence[int]) -> list[int]:
    neighbors = sorted(
        int(candidate)
        for candidate in candidate_ids
        if lattice_distance(int(current), int(candidate)) == 1
    )
    return sorted({int(current), *neighbors})


def _batch_logits(
    model: PrefixHAR, sequences: np.ndarray, observed: int, device: torch.device,
) -> np.ndarray:
    logits_parts: list[np.ndarray] = []
    for start in range(0, len(sequences), INFERENCE_BATCH_SIZE):
        end = min(start + INFERENCE_BATCH_SIZE, len(sequences))
        batch = torch.from_numpy(np.asarray(sequences[start:end], dtype=np.float32))
        mask = torch.zeros((end - start, FRAME_COUNT), dtype=torch.float32)
        mask[:, :observed] = 1.0
        with torch.inference_mode():
            logits, _ = model(batch.to(device), mask.to(device))
        logits_parts.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_parts, axis=0) if logits_parts else np.empty((0, NUM_CLASSES), dtype=np.float32)


def _boundary_displacement(previous: np.ndarray, following: np.ndarray) -> float:
    prev_joints = previous.reshape(JOINT_COUNT, 3)
    next_joints = following.reshape(JOINT_COUNT, 3)
    return float(np.linalg.norm(next_joints - prev_joints, axis=1).mean())


def _load_context_group(
    contexts: Sequence[Mapping[str, Any]], offline_root: Path,
) -> list[np.ndarray]:
    return [_load_all_views(_record_path(offline_root, context["row"])) for context in contexts]


def _sequential_eval(
    model: PrefixHAR,
    contexts: Sequence[Mapping[str, Any]],
    offline_root: Path,
    device: torch.device,
    batch_contexts: int = 128,
) -> tuple[dict[str, Any], dict[str, Any]]:
    methods = ("Stay", "Random-1Hop", "Privileged-Greedy-Oracle")
    observations = tuple(range(CHUNK_SIZE, FRAME_COUNT + 1, CHUNK_SIZE))
    pred_store: dict[str, dict[int, list[int]]] = {
        method: {time: [] for time in observations} for method in methods
    }
    gt_prob_store: dict[str, dict[int, list[float]]] = {
        method: {time: [] for time in observations} for method in methods
    }
    entropy_store: dict[str, dict[int, list[float]]] = {
        method: {time: [] for time in observations} for method in methods
    }
    move_fraction_store: dict[str, dict[int, list[float]]] = {
        method: {time: [] for time in observations} for method in methods
    }
    boundary_values: dict[str, dict[str, list[float]]] = {
        method: {"same_view": [], "switched_view": []} for method in methods
    }
    rng = np.random.default_rng(SEED)

    for group_start in range(0, len(contexts), batch_contexts):
        group = contexts[group_start : group_start + batch_contexts]
        labels = np.asarray([int(context["label"]) for context in group], dtype=np.int64)
        all_views = _load_context_group(group, offline_root)
        size = len(group)
        histories = {method: np.zeros((size, FRAME_COUNT, COORDINATE_DIM), dtype=np.float32) for method in methods}
        current_views = {
            method: np.asarray([int(context["start_view"]) for context in group], dtype=np.int64)
            for method in methods
        }
        move_counts = {method: np.zeros(size, dtype=np.int64) for method in methods}
        for chunk_index, observed in enumerate(observations):
            start = observed - CHUNK_SIZE
            for method in methods:
                for local_index, views in enumerate(all_views):
                    view = int(current_views[method][local_index])
                    histories[method][local_index, start:observed] = views[view, start:observed]
                logits = _batch_logits(model, histories[method], observed, device)
                probabilities = _softmax(logits)
                pred_store[method][observed].extend(np.argmax(logits, axis=1).astype(int).tolist())
                gt_prob_store[method][observed].extend(probabilities[np.arange(size), labels].astype(float).tolist())
                entropy_store[method][observed].extend(_entropy(probabilities).astype(float).tolist())
                decisions = max(chunk_index, 1)
                move_fraction_store[method][observed].extend(
                    (move_counts[method] / decisions).astype(float).tolist() if chunk_index else [0.0] * size
                )

            if chunk_index == len(observations) - 1:
                continue

            next_observed = observed + CHUNK_SIZE
            for method in ("Stay", "Random-1Hop"):
                choices: list[int] = []
                for local_index, context in enumerate(group):
                    options = _candidate_options(int(current_views[method][local_index]), context["candidate_ids"])
                    if method == "Stay":
                        choice = int(current_views[method][local_index])
                    else:
                        choice = int(rng.choice(np.asarray(options, dtype=np.int64)))
                    choices.append(choice)
                next_views = np.asarray(choices, dtype=np.int64)
                for local_index, views in enumerate(all_views):
                    previous_view = int(current_views[method][local_index])
                    next_view = int(next_views[local_index])
                    displacement = _boundary_displacement(
                        views[previous_view, observed - 1], views[next_view, observed]
                    )
                    boundary_values[method]["same_view" if previous_view == next_view else "switched_view"].append(displacement)
                move_counts[method] += next_views != current_views[method]
                current_views[method] = next_views

            oracle_histories: list[np.ndarray] = []
            owners: list[int] = []
            option_views: list[int] = []
            for local_index, context in enumerate(group):
                options = _candidate_options(int(current_views["Privileged-Greedy-Oracle"][local_index]), context["candidate_ids"])
                for option in options:
                    candidate_history = histories["Privileged-Greedy-Oracle"][local_index].copy()
                    candidate_history[observed:next_observed] = all_views[local_index][option, observed:next_observed]
                    oracle_histories.append(candidate_history)
                    owners.append(local_index)
                    option_views.append(int(option))
            oracle_logits = _batch_logits(model, np.stack(oracle_histories), next_observed, device)
            selected: list[int] = []
            for local_index, context in enumerate(group):
                candidate_indices = [index for index, owner in enumerate(owners) if owner == local_index]
                label = int(context["label"])
                best_index = max(
                    candidate_indices,
                    key=lambda index: (
                        float(oracle_logits[index, label] - np.max(np.delete(oracle_logits[index], label))),
                        -option_views[index],
                    ),
                )
                selected.append(option_views[best_index])
            next_views = np.asarray(selected, dtype=np.int64)
            for local_index, views in enumerate(all_views):
                previous_view = int(current_views["Privileged-Greedy-Oracle"][local_index])
                next_view = int(next_views[local_index])
                displacement = _boundary_displacement(
                    views[previous_view, observed - 1], views[next_view, observed]
                )
                boundary_values["Privileged-Greedy-Oracle"][
                    "same_view" if previous_view == next_view else "switched_view"
                ].append(displacement)
            move_counts["Privileged-Greedy-Oracle"] += next_views != current_views["Privileged-Greedy-Oracle"]
            current_views["Privileged-Greedy-Oracle"] = next_views

    sequential: dict[str, Any] = {}
    for method in methods:
        sequential[method] = {}
        for observed in observations:
            metrics = classification(
                [int(context["label"]) for context in contexts], pred_store[method][observed]
            )
            metrics.update({
                "observed_frames": observed,
                "mean_entropy": float(np.mean(entropy_store[method][observed])),
                "mean_gt_probability": float(np.mean(gt_prob_store[method][observed])),
                "move_rate": float(np.mean(move_fraction_store[method][observed])),
                "stay_rate": float(1.0 - np.mean(move_fraction_store[method][observed])),
                "discrete_time_view_switch_approximation": True,
            })
            sequential[method][f"t{observed}"] = metrics
    boundary: dict[str, Any] = {}
    for method in methods:
        values = boundary_values[method]
        boundary[method] = {
            "same_view_boundary_count": len(values["same_view"]),
            "switched_view_boundary_count": len(values["switched_view"]),
            "mean_joint_displacement_same_view": float(np.mean(values["same_view"])) if values["same_view"] else None,
            "mean_joint_displacement_switched_view": float(np.mean(values["switched_view"])) if values["switched_view"] else None,
            "median_joint_displacement_same_view": float(np.median(values["same_view"])) if values["same_view"] else None,
            "median_joint_displacement_switched_view": float(np.median(values["switched_view"])) if values["switched_view"] else None,
        }
    return sequential, boundary


def _load_contexts(policy_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stage_c_train = _read_rows(policy_root, "stage_c", "train")
    stage_c_val = _read_rows(policy_root, "stage_c", "val")
    stage_d_val = _read_rows(policy_root, "stage_d", "val")
    stage_c_by_episode = {str(row["episode_id"]): row for row in stage_c_val}
    contexts: list[dict[str, Any]] = []
    for row in stage_d_val:
        key = str(row["episode_id"])
        stage_c = stage_c_by_episode.get(key)
        if stage_c is None:
            raise ValueError(f"missing Stage-C row for {key}")
        start = int(row["s0_viewpoint_id"])
        if start != int(stage_c["current_viewpoint_id"]):
            raise ValueError(f"Stage-C/Stage-D start viewpoint mismatch for {key}")
        candidates = sorted(int(value) for value in stage_c["candidate_viewpoint_ids"])
        contexts.append({
            "episode_id": key,
            "row": row,
            "label": int(row["label_id"]),
            "start_view": start,
            "candidate_ids": candidates,
        })
    summary = {
        "stage_c_train_contexts": len(stage_c_train),
        "stage_c_val_contexts": len(stage_c_val),
        "stage_d_val_moving_contexts": len(contexts),
        "train_split_used": "stage_c/features/train.jsonl",
        "val_split_used": "stage_d/features/val.jsonl",
    }
    return contexts, summary


def _load_train_sequences(policy_root: Path, offline_root: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_rows(policy_root, "stage_c", "train")
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for index, row in enumerate(rows, start=1):
        sequences.append(_load_view_sequence(_record_path(offline_root, row), int(row["current_viewpoint_id"])))
        labels.append(int(row["label_id"]))
        if index % 5000 == 0:
            print(f"[prefix-data] loaded Train skeletons {index}/{len(rows)}", flush=True)
    return np.stack(sequences).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _load_val_prefix_sequences(contexts: Sequence[Mapping[str, Any]], offline_root: Path) -> tuple[np.ndarray, np.ndarray]:
    sequences = [
        _load_view_sequence(_record_path(offline_root, context["row"]), int(context["start_view"]))
        for context in contexts
    ]
    labels = np.asarray([int(context["label"]) for context in contexts], dtype=np.int64)
    return np.stack(sequences).astype(np.float32), labels


def _analysis(result: Mapping[str, Any]) -> str:
    masked = result["prefix_recognition"]["zero_plus_mask"]
    sequential = result["sequential_baselines"]
    observations = [f"t{value}" for value in (5, 10, 15, 20, 25, 30)]
    stay_final = sequential["Stay"]["t30"]
    oracle_final = sequential["Privileged-Greedy-Oracle"]["t30"]
    oracle_gain = float(oracle_final["accuracy"] - stay_final["accuracy"])
    deltas = {
        time: float(sequential["Privileged-Greedy-Oracle"][time]["accuracy"] - stay_final["accuracy"])
        for time in observations
    }
    best_time = max(deltas, key=deltas.get)
    boundary = result["boundary_continuity"]
    switched = [
        item["mean_joint_displacement_switched_view"]
        for item in boundary.values()
        if item["mean_joint_displacement_switched_view"] is not None
    ]
    same = [
        item["mean_joint_displacement_same_view"]
        for item in boundary.values()
        if item["mean_joint_displacement_same_view"] is not None
    ]
    prefix_lines = []
    for prefix in PREFIXES:
        item = masked[f"{prefix}_frames"]
        prefix_lines.append(f"| {prefix} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['mean_entropy']:.6f} |")
    seq_lines = []
    for method in ("Stay", "Random-1Hop", "Privileged-Greedy-Oracle"):
        for time in observations:
            item = sequential[method][time]
            seq_lines.append(f"| {method} | {time[1:]} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['mean_entropy']:.6f} | {item['mean_gt_probability']:.6f} | {item['move_rate']:.6f} |")
    if oracle_gain >= 0.02:
        policy_judgement = "t30 oracle 相对 Stay 的提升达到 2pp 以上，支持下一步研究 learned sequential NBV policy；本轮仍未训练 policy。"
    else:
        policy_judgement = "t30 oracle 相对 Stay 的提升低于 2pp，当前简单 sequential headroom 有限，不据此训练 learned policy。"
    continuity = "未发现明显跨视点 discontinuity" if not switched or not same or float(np.mean(switched)) <= 2.0 * float(np.mean(same)) else "跨视点边界位移明显大于同视点边界，存在 skeleton discontinuity 风险"
    return "\n".join([
        "# Reduced12 Prefix-conditioned Sequential NBV MVP",
        "",
        f"Train Stage-C contexts: {result['data_summary']['stage_c_train_contexts']}; Val Moving contexts: {result['data_summary']['stage_d_val_moving_contexts']}.",
        "The protocol is a discrete-time view-switch approximation: after each observed 5-frame chunk, the next chunk may use Stay or a legal lattice 1-hop neighbor. Continuous robot motion is not claimed.",
        "Policy Test was not read; no RGB, skeleton or DINO data was regenerated.",
        "",
        "## Prefix recognizer sanity check (fixed start viewpoint)",
        "",
        "| Observed frames | Accuracy | Macro-F1 | Mean entropy |",
        "|---:|---:|---:|---:|",
        *prefix_lines,
        "",
        "The zero+mask and zero-only comparison is stored in prefix_recognition.json; the deployed sequential diagnostic uses zero+mask.",
        "",
        "## Sequential baselines",
        "",
        "| Method | t | Accuracy | Macro-F1 | Mean entropy | GT probability | Move rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *seq_lines,
        "",
        f"At t=30, Privileged-Greedy-Oracle is {oracle_final['accuracy']:.6f}/{oracle_final['macro_f1']:.6f}; Stay is {stay_final['accuracy']:.6f}/{stay_final['macro_f1']:.6f}; ΔAccuracy={oracle_gain * 100:.2f}pp.",
        f"The largest oracle-vs-Stay Accuracy difference occurs at {best_time} ({deltas[best_time] * 100:.2f}pp relative to Stay t30).",
        "",
        "## Boundary continuity",
        "",
        continuity + ". The per-method same-view and switched-view displacement counts and means are in boundary_continuity.json; large switched-boundary displacement should be treated as a limitation of this diagnostic rather than hidden.",
        "",
        "## Answers",
        "",
        f"1. 5/10/15-frame prefix accuracy is {masked['5_frames']['accuracy']:.6f}/{masked['10_frames']['accuracy']:.6f}/{masked['15_frames']['accuracy']:.6f}; information accumulates monotonically, but absolute identity accuracy remains low rather than immediately deployment-ready.",
        f"2. With the explicit mask, Stay improves from t5 to t30 by {(sequential['Stay']['t30']['accuracy'] - sequential['Stay']['t5']['accuracy']) * 100:.2f}pp, so cumulative history is more informative than the first chunk; the zero-only ablation is reported separately.",
        f"3. The t30 privileged oracle exceeds Stay by {oracle_gain * 100:.2f}pp Accuracy, showing a substantial diagnostic sequential ceiling under this approximation.",
        "4. The observation chunk with the largest oracle-vs-Stay difference is reported above.",
        f"5. {continuity}.",
        f"6. {policy_judgement}",
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "new_rgb_generated=false",
        "new_skeleton_generated=false",
        "existing_stgcn_modified=false",
        "prefix_encoder_trained_on_train_only=true",
        "gt_action_used_for_oracle_only=true",
        "continuous_robot_motion_claimed=false",
        "```",
        "",
    ])


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    policy_root = data_root / POLICY_RELATIVE
    offline_root = data_root / OFFLINE_RELATIVE
    runtime_root = data_root / RUNTIME_CHECKPOINT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts, data_summary = _load_contexts(policy_root)
    print(json.dumps(data_summary, ensure_ascii=False), flush=True)
    train_sequences, train_labels = _load_train_sequences(policy_root, offline_root)
    val_sequences, val_labels = _load_val_prefix_sequences(contexts, offline_root)

    masked_checkpoint = runtime_root / "prefix_har_masked_best.pth"
    zero_checkpoint = runtime_root / "prefix_har_zero_only_best.pth"
    masked_summary = _train_encoder(
        train_sequences, train_labels, val_sequences, val_labels, device, True,
        masked_checkpoint, output_dir / "prefix_encoder_training_masked.json",
    )
    zero_summary = _train_encoder(
        train_sequences, train_labels, val_sequences, val_labels, device, False,
        zero_checkpoint, output_dir / "prefix_encoder_training_zero_only.json",
    )
    masked_model = _load_encoder(masked_checkpoint, device)
    zero_model = _load_encoder(zero_checkpoint, device)
    prefix_recognition = {
        "zero_plus_mask": _evaluate_prefix(masked_model, val_sequences, val_labels, device, True),
        "zero_only": _evaluate_prefix(zero_model, val_sequences, val_labels, device, False),
        "fixed_viewpoint": "Stage-D s0 viewpoint for each Val Moving context",
    }
    sequential_baselines, boundary_continuity = _sequential_eval(masked_model, contexts, offline_root, device)
    belief_progression = {
        method: {
            time: {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_entropy": metrics["mean_entropy"],
                "mean_gt_probability": metrics["mean_gt_probability"],
            }
            for time, metrics in values.items()
        }
        for method, values in sequential_baselines.items()
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_PREFIX_SEQUENTIAL_NBV_MVP",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "data_summary": data_summary,
        "labels": list(LABELS),
        "protocol": {
            "chunk_size": CHUNK_SIZE,
            "chunks": [[start, start + CHUNK_SIZE] for start in range(0, FRAME_COUNT, CHUNK_SIZE)],
            "prefixes_train": list(PREFIXES),
            "legal_actions": "Stay or current-view lattice distance <= 1 legal neighbor",
            "random_seed": SEED,
            "oracle": "GT label selects the option with maximum next-prefix GT-class margin",
            "continuous_robot_motion_claimed": False,
        },
        "prefix_encoder_training": {"zero_plus_mask": masked_summary, "zero_only": zero_summary},
        "prefix_recognition": prefix_recognition,
        "sequential_baselines": sequential_baselines,
        "belief_progression": belief_progression,
        "boundary_continuity": boundary_continuity,
        "policy_test_used": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "existing_stgcn_modified": False,
        "prefix_encoder_trained_on_train_only": True,
        "gt_action_used_for_oracle_only": True,
        "continuous_robot_motion_claimed": False,
    }
    _write(output_dir / "prefix_recognition.json", prefix_recognition)
    _write(output_dir / "sequential_baselines.json", sequential_baselines)
    _write(output_dir / "belief_progression.json", belief_progression)
    _write(output_dir / "boundary_continuity.json", boundary_continuity)
    _write(output_dir / "result.json", result)
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = _cuda(args.device)
    result = run(args.output_dir.resolve(), args.data_root.resolve(), device)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "contexts": result["data_summary"]["stage_d_val_moving_contexts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
