#!/usr/bin/env python3
"""Train/Val-only Top-3 sequential hypothesis verification for reduced12.

The existing Candidate-Conditioned Spatial H1 proposal is frozen.  A real
archived H1 observation is then exposed to a tiny listwise verifier whose
only actions are STOP, proposal rank2, and proposal rank3.  Rank2/rank3
observations are used only as Train targets and terminal evaluation, never as
verifier inputs.  Policy Test and perception regeneration are intentionally
outside this runner.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.policy_data import load_feature_statistics
from activeview.scripts.experiments.run_reduced12_h1_stay_aware_batch import (
    StayDataset,
    _collate as stay_collate,
    _load_baseline,
    _scores as stay_scores,
)
from activeview.scripts.experiments.run_reduced12_h1_visual_context_batch import (
    VisualDataset,
    _collate as visual_collate,
    _load_dino_store,
    _model as visual_model,
    _score as visual_score,
    _covered_rows,
)

SEED = 42
NUM_CLASSES = 12
STGCN_FEATURE_DIM = 256
CURRENT_FEATURE_DIM = 271
GEOMETRY_DIM = 11
VERIFIER_INPUT_DIM = 550
HIDDEN_DIM = 128
EPOCHS = 20
BATCH_SIZE = 512
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
EXPERIMENT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/top3_sequential_hypothesis_verification"


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


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _subset(cache: Mapping[str, np.ndarray], indices: Sequence[int]) -> dict[str, np.ndarray]:
    index = np.asarray(indices, dtype=np.int64)
    return {key: value[index] if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == len(cache["labels"]) else value for key, value in cache.items()}


def _classification(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, name in enumerate(LABELS):
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        tp = int(confusion[index, index])
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": support, "recall": recall, "f1": f1}
    return {
        "n": int(labels.size),
        "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _corr(left: Sequence[float], right: Sequence[float], spearman: bool = False) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if spearman:
        x, y = _rank(x), _rank(y)
    return float(np.corrcoef(x, y)[0, 1])


def _utility(logp: np.ndarray, label: int) -> float:
    others = np.delete(logp, int(label))
    return float(logp[int(label)] - np.max(others))


def _load_data(data_root: Path, device: torch.device) -> dict[str, Any]:
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    stage_c = policy_root / "stage_c"
    train_all = load_jsonl(stage_c / "features/train.jsonl")
    val_all = load_jsonl(stage_c / "features/val.jsonl")
    train_d_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/train.jsonl")}
    val_d_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/val.jsonl")}
    dino_lookup, dino_embeddings, dino_summary = _load_dino_store(data_root)
    train_rows, train_indices = _covered_rows(train_all, train_d_ids, dino_lookup)
    val_rows, val_indices = _covered_rows(val_all, val_d_ids, dino_lookup)
    train_cache = _subset(_read_npz(data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/train_candidate_true_logp.npz"), train_indices)
    val_cache = _subset(_read_npz(data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"), val_indices)
    for rows, cache, name in ((train_rows, train_cache, "Train"), (val_rows, val_cache, "Val")):
        labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
        if not np.array_equal(labels, cache["labels"]):
            raise ValueError(f"{name} labels are misaligned with true evidence cache")
        for row_index, row in enumerate(rows):
            cached_ids = cache["candidate_ids"][row_index, cache["candidate_mask"][row_index]].astype(int).tolist()
            if cached_ids != [int(value) for value in row["candidate_viewpoint_ids"]]:
                raise ValueError(f"{name} candidate identity mismatch at {row['episode_id']}")
    target_root = data_root / "diagnostics/reduced12_future_recognition_evidence_prediction"
    train_target = np.asarray(np.load(target_root / "train_target_features.npz", allow_pickle=False)["candidate_feature"], dtype=np.float32)
    val_target = np.asarray(np.load(target_root / "val_moving_target_features.npz", allow_pickle=False)["candidate_feature"], dtype=np.float32)
    if train_target.shape[0] != len(train_rows) or val_target.shape[0] != len(val_rows):
        raise ValueError("target feature cache is not aligned with DINO-covered Stage-D rows")
    stats = load_feature_statistics(stage_c / "stage_c_feature_stats.json")
    summary = json.loads((stage_c / "stage_c_feature_summary.json").read_text(encoding="utf-8"))
    if int(summary["current_feature_dim"]) != CURRENT_FEATURE_DIM or int(summary["candidate_geometry_dim"]) != GEOMETRY_DIM:
        raise ValueError("unexpected Stage-C feature schema")
    return {
        "train_rows": train_rows, "val_rows": val_rows, "train_cache": train_cache, "val_cache": val_cache,
        "train_target": train_target, "val_target": val_target, "stats": stats, "summary": summary,
        "dino_lookup": dino_lookup, "dino_embeddings": dino_embeddings, "dino_summary": dino_summary,
        "val_full": val_all, "policy_root": policy_root,
    }


def _proposal_scores(data: Mapping[str, Any], device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = get_data_root() / "checkpoints/policy_reduced12_eight_placement_v1/h1_visual_context_batch/candidate_conditioned_spatial_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    train_set = VisualDataset(data["train_rows"], data["train_cache"], data["stats"], data["dino_lookup"], data["dino_embeddings"], "candidate_conditioned_spatial")
    val_set = VisualDataset(data["val_rows"], data["val_cache"], data["stats"], data["dino_lookup"], data["dino_embeddings"], "candidate_conditioned_spatial")
    model = visual_model("candidate_conditioned_spatial").to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    train_stay, train_candidate = visual_score(model, train_set, device, batch_size)
    val_stay, val_candidate = visual_score(model, val_set, device, batch_size)
    return np.concatenate([train_stay[:, None], train_candidate], axis=1), np.concatenate([val_stay[:, None], val_candidate], axis=1)


def _action_ranks(scores: np.ndarray, cache: Mapping[str, np.ndarray]) -> np.ndarray:
    ranks = np.full((scores.shape[0], 3), -1, dtype=np.int64)
    for row_index in range(scores.shape[0]):
        valid = np.flatnonzero(cache["candidate_mask"][row_index])
        actions = np.concatenate([np.zeros(1, dtype=np.int64), valid + 1])
        if actions.size < 3:
            raise ValueError("Top-3 protocol requires at least rank1/rank2 actions")
        order = np.argsort(-scores[row_index, actions], kind="mergesort")
        ranks[row_index] = actions[order[:3]]
    return ranks


def _evidence(row_index: int, action: int, cache: Mapping[str, np.ndarray], targets: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if int(action) == 0:
        return np.asarray(cache["current_logp"][row_index], dtype=np.float32), np.asarray(rows[row_index]["current_feature"][:STGCN_FEATURE_DIM], dtype=np.float32)
    candidate_index = int(action) - 1
    return np.asarray(cache["candidate_logp"][row_index, candidate_index], dtype=np.float32), np.asarray(targets[row_index, candidate_index], dtype=np.float32)


def _build_sequential_arrays(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], targets: np.ndarray, proposal_scores: np.ndarray, stats: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(rows)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    s0_logp = np.asarray(cache["current_logp"], dtype=np.float32)
    s0_feature = np.asarray([np.asarray(row["current_feature"][:STGCN_FEATURE_DIM], dtype=np.float32) for row in rows], dtype=np.float32)
    geometry = np.zeros((n, proposal_scores.shape[1], GEOMETRY_DIM), dtype=np.float32)
    gmean, gstd = stats["geometry_mean"], stats["geometry_std"]
    for row_index, row in enumerate(rows):
        row_geometry = (np.asarray(row["candidate_geometry"], dtype=np.float32) - gmean) / gstd
        valid = np.flatnonzero(cache["candidate_mask"][row_index])
        geometry[row_index, valid + 1] = row_geometry[:len(valid)]
    ranks = _action_ranks(proposal_scores, cache)
    h1_logp = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    h1_feature = np.zeros((n, STGCN_FEATURE_DIM), dtype=np.float32)
    action_logp = np.zeros((n, 3, NUM_CLASSES), dtype=np.float32)
    action_feature = np.zeros((n, 3, STGCN_FEATURE_DIM), dtype=np.float32)
    utilities = np.zeros((n, 3), dtype=np.float32)
    for row_index in range(n):
        h1_logp[row_index], h1_feature[row_index] = _evidence(row_index, int(ranks[row_index, 0]), cache, targets, rows)
        action_logp[row_index, 0], action_feature[row_index, 0] = h1_logp[row_index], h1_feature[row_index]
        utilities[row_index, 0] = _utility(h1_logp[row_index], int(labels[row_index]))
        for action_slot in (1, 2):
            action = int(ranks[row_index, action_slot])
            action_logp[row_index, action_slot], action_feature[row_index, action_slot] = _evidence(row_index, action, cache, targets, rows)
            utilities[row_index, action_slot] = _utility(action_logp[row_index, action_slot], int(labels[row_index]))
    return {"labels": labels, "s0_logp": s0_logp, "s0_feature": s0_feature, "h1_logp": h1_logp, "h1_feature": h1_feature, "action_logp": action_logp, "action_feature": action_feature, "utilities": utilities, "ranks": ranks, "geometry": geometry, "proposal_scores": proposal_scores, "all_candidate_logp": np.asarray(cache["candidate_logp"], dtype=np.float32)}


def _verifier_inputs(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    n = arrays["labels"].size
    output = np.zeros((n, 3, VERIFIER_INPUT_DIM), dtype=np.float32)
    for row_index in range(n):
        for slot in range(3):
            stop = 1.0 if slot == 0 else 0.0
            action = int(arrays["ranks"][row_index, slot])
            rank_value = float(slot + 1) if slot else 0.0
            score = float(arrays["proposal_scores"][row_index, action]) if action >= 0 else 0.0
            parts = [arrays["s0_logp"][row_index], arrays["s0_feature"][row_index], arrays["h1_logp"][row_index], arrays["h1_feature"][row_index], arrays["geometry"][row_index, action] if action >= 0 else np.zeros(GEOMETRY_DIM, dtype=np.float32), np.asarray([score, rank_value, stop], dtype=np.float32)]
            value = np.concatenate(parts).astype(np.float32)
            if value.size != VERIFIER_INPUT_DIM:
                raise ValueError(f"verifier input size mismatch: {value.size}")
            output[row_index, slot] = value
    return output


class SequentialVerifier(nn.Module):
    def __init__(self, input_dim: int = VERIFIER_INPUT_DIM) -> None:
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.scorer(inputs).squeeze(-1)


def _train_verifier(train_inputs: np.ndarray, train_targets: np.ndarray, val_inputs: np.ndarray, val_targets: np.ndarray, device: torch.device, checkpoint: Path, summary_path: Path, batch_size: int) -> dict[str, Any]:
    _seed()
    mean = train_inputs.reshape(-1, VERIFIER_INPUT_DIM).mean(axis=0).astype(np.float32)
    std = train_inputs.reshape(-1, VERIFIER_INPUT_DIM).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_norm = ((train_inputs - mean) / std).astype(np.float32)
    val_norm = ((val_inputs - mean) / std).astype(np.float32)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_norm), torch.from_numpy(train_targets)), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_norm), torch.from_numpy(val_targets)), batch_size=batch_size, shuffle=False, num_workers=0)
    model = SequentialVerifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for inputs, targets in train_loader:
            scores = model(inputs.to(device))
            target_dist = torch.softmax(targets.to(device), dim=1)
            loss = -(target_dist * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, targets in val_loader:
                scores = model(inputs.to(device))
                target_dist = torch.softmax(targets.to(device), dim=1)
                val_losses.append(float((-(target_dist * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        print(f"[sequential-verifier] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss = record["val_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "input_mean": mean, "input_std": std, "epoch": epoch, "seed": SEED}, checkpoint)
    summary = {"epochs": EPOCHS, "selected_epoch": best_epoch, "best_val_loss": best_loss, "batch_size": batch_size, "input_dim": VERIFIER_INPUT_DIM, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "checkpoint": str(checkpoint.resolve()), "history": history, "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_verifier(checkpoint: Path, device: torch.device) -> tuple[SequentialVerifier, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SequentialVerifier().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["input_mean"], dtype=np.float32), np.asarray(payload["input_std"], dtype=np.float32)


def _terminal_metrics(arrays: Mapping[str, np.ndarray], selected_slots: np.ndarray, fusion: bool = False, name: str = "") -> dict[str, Any]:
    labels = arrays["labels"]
    predictions: list[int] = []
    for row_index, slot in enumerate(selected_slots):
        slot = int(slot)
        if fusion:
            if slot == 0:
                final = np.mean([arrays["s0_logp"][row_index], arrays["h1_logp"][row_index]], axis=0)
            else:
                final = np.mean([arrays["s0_logp"][row_index], arrays["h1_logp"][row_index], arrays["action_logp"][row_index, slot]], axis=0)
            predictions.append(int(np.argmax(final)))
        else:
            predictions.append(int(np.argmax(arrays["action_logp"][row_index, slot])))
    result = _classification(labels, np.asarray(predictions, dtype=np.int64))
    result["selector"] = name
    result["h2_move_rate"] = float(np.mean(selected_slots != 0))
    result["stop_rate"] = float(np.mean(selected_slots == 0))
    result["selected_action_positive_rate"] = float(np.mean(np.asarray(predictions) == labels))
    return result


def _terminal_action_metrics(arrays: Mapping[str, np.ndarray], actions: np.ndarray, name: str) -> dict[str, Any]:
    labels = arrays["labels"]
    predictions = []
    for row_index, action in enumerate(actions):
        if int(action) == 0:
            logp = arrays["s0_logp"][row_index]
        else:
            logp = arrays["action_logp"][row_index, 0] if int(action) == int(arrays["ranks"][row_index, 0]) else None
            if logp is None:
                # The full FrozenStageCv0 action may be outside the Top-3
                # shortlist; recover it from the complete candidate cache.
                candidate_index = int(action) - 1
                logp = arrays["all_candidate_logp"][row_index, candidate_index]
        predictions.append(int(np.argmax(logp)))
    result = _classification(labels, np.asarray(predictions, dtype=np.int64))
    result["selector"] = name
    result["move_rate"] = float(np.mean(actions != 0))
    result["stay_rate"] = float(np.mean(actions == 0))
    return result


def _proposal_baselines(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    ranks = arrays["ranks"]
    result: dict[str, Any] = {}
    result["Proposal-H1"] = _terminal_metrics(arrays, np.zeros(len(ranks), dtype=np.int64), name="Proposal-H1")
    result["Proposal-H1-H2Fixed"] = _terminal_metrics(arrays, np.ones(len(ranks), dtype=np.int64), name="Proposal-H1-H2Fixed")
    result["Proposal-H1-H2Fixed-Fusion"] = _terminal_metrics(arrays, np.ones(len(ranks), dtype=np.int64), fusion=True, name="Proposal-H1-H2Fixed-Fusion")
    result["S0-only"] = _classification(arrays["labels"], np.argmax(arrays["s0_logp"], axis=1))
    result["S0-only"]["selector"] = "S0-only"
    return result


def _oracle_metrics(arrays: Mapping[str, np.ndarray], cache: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = arrays["labels"]
    ranks = arrays["ranks"]
    selected_top2 = np.zeros(len(ranks), dtype=np.int64)
    selected_top3 = np.zeros(len(ranks), dtype=np.int64)
    gt_slots = np.zeros(len(ranks), dtype=np.int64)
    any2 = np.zeros(len(ranks), dtype=np.int64)
    any3 = np.zeros(len(ranks), dtype=np.int64)
    for row_index, label in enumerate(labels):
        correct_slots = [slot for slot in range(3) if int(np.argmax(arrays["action_logp"][row_index, slot])) == int(label)]
        any2[row_index] = 1 if any(slot < 2 for slot in correct_slots) else 0
        any3[row_index] = 1 if correct_slots else 0
        selected_top2[row_index] = next((slot for slot in (0, 1) if slot in correct_slots), 0)
        selected_top3[row_index] = next((slot for slot in (0, 1, 2) if slot in correct_slots), 0)
        utilities = arrays["utilities"][row_index]
        gt_slots[row_index] = int(np.argmax(utilities))
    full_any = np.zeros(len(ranks), dtype=np.int64)
    full_gt = np.zeros(len(ranks), dtype=np.int64)
    for row_index, label in enumerate(labels):
        current_utility = _utility(arrays["s0_logp"][row_index], int(label))
        candidate_utilities = np.asarray([_utility(value, int(label)) for value in arrays["all_candidate_logp"][row_index, cache["candidate_mask"][row_index]]], dtype=np.float32)
        options = np.concatenate([[current_utility], candidate_utilities])
        full_gt[row_index] = int(np.argmax(options))
        full_any[row_index] = int(np.any(np.argmax(np.vstack([arrays["s0_logp"][row_index], arrays["all_candidate_logp"][row_index, cache["candidate_mask"][row_index]]]), axis=1) == label))
    any_predictions = np.where(full_any == 1, labels, np.argmax(arrays["s0_logp"], axis=1))
    gt_actions = np.zeros(len(ranks), dtype=np.int64)
    for row_index in range(len(ranks)):
        gt_actions[row_index] = 0 if full_gt[row_index] == 0 else int(np.flatnonzero(cache["candidate_mask"][row_index])[full_gt[row_index] - 1]) + 1
    result = {
        "AnyCorrect Oracle": {**_classification(labels, any_predictions), "any_correct_rate": float(np.mean(full_any)), "selector": "AnyCorrect Oracle"},
        "Real-GTMargin Oracle": _terminal_action_metrics({**arrays, "all_candidate_logp": arrays["all_candidate_logp"]}, gt_actions, "Real-GTMargin Oracle"),
        "Top2-AnyCorrect": {**_classification(labels, np.where(any2 == 1, labels, np.argmax(arrays["action_logp"][:, 0], axis=1))), "any_correct_rate": float(np.mean(any2)), "selector": "Top2-AnyCorrect"},
        "Top3-AnyCorrect": {**_classification(labels, np.where(any3 == 1, labels, np.argmax(arrays["action_logp"][:, 0], axis=1))), "any_correct_rate": float(np.mean(any3)), "selector": "Top3-AnyCorrect"},
        "Top3-GTSequentialVerifier": _terminal_metrics(arrays, gt_slots, name="Top3-GTSequentialVerifier"),
        "Top3-GTSequentialVerifier-Fusion": _terminal_metrics(arrays, gt_slots, fusion=True, name="Top3-GTSequentialVerifier-Fusion"),
    }
    return result


def _diagnostics(arrays: Mapping[str, np.ndarray], selected_slots: np.ndarray, verifier_scores: np.ndarray) -> dict[str, Any]:
    labels = arrays["labels"]
    s0_pred = np.argmax(arrays["s0_logp"], axis=1)
    h1_pred = np.argmax(arrays["h1_logp"], axis=1)
    entropy = lambda values: -np.sum(np.exp(values) * values, axis=1)
    mean_logp = np.mean(np.stack([arrays["s0_logp"], arrays["h1_logp"]], axis=1), axis=1)
    h1_wrong = h1_pred != labels
    oracle_correct = np.any(np.argmax(arrays["action_logp"], axis=2) == labels[:, None], axis=1)
    rescue = int(np.sum(h1_wrong & (np.asarray([np.argmax(arrays["action_logp"][i, selected_slots[i]]) for i in range(len(labels))]) == labels)))
    harm = int(np.sum(~h1_wrong & (np.asarray([np.argmax(arrays["action_logp"][i, selected_slots[i]]) for i in range(len(labels))]) != labels)))
    remaining = h1_wrong & oracle_correct
    verifier_selected = np.asarray([np.argmax(arrays["action_logp"][i, selected_slots[i]]) for i in range(len(labels))])
    gt_order = np.argsort(-arrays["utilities"], axis=1)
    return {
        "mean_entropy_s0": float(np.mean(entropy(arrays["s0_logp"]))),
        "mean_entropy_h1": float(np.mean(entropy(arrays["h1_logp"]))),
        "mean_entropy_s0_h1_fusion": float(np.mean(entropy(mean_logp))),
        "h1_entropy_change": float(np.mean(entropy(arrays["h1_logp"]) - entropy(arrays["s0_logp"]))),
        "h1_true_class_logp_gain": float(np.mean(arrays["h1_logp"][np.arange(len(labels)), labels] - arrays["s0_logp"][np.arange(len(labels)), labels])),
        "h1_corrects_s0_errors": int(np.sum((s0_pred != labels) & (h1_pred == labels))),
        "h1_wrong_contexts": int(np.sum(h1_wrong)),
        "verifier_rescue_count": rescue,
        "verifier_harm_count": harm,
        "net_rescue": rescue - harm,
        "gt_verifier_rescue_ceiling": int(np.sum(remaining & np.any(np.argmax(arrays["action_logp"], axis=2) == labels[:, None], axis=1))),
        "h1_wrong_top3_contains_correct": int(np.sum(remaining)),
        "learned_selected_correct_remaining": float(np.mean(verifier_selected[remaining] == labels[remaining])) if np.any(remaining) else 0.0,
        "learned_stop_on_remaining": float(np.mean(selected_slots[remaining] == 0)) if np.any(remaining) else 0.0,
        "learned_other_wrong_on_remaining": float(np.mean((selected_slots[remaining] != 0) & (verifier_selected[remaining] != labels[remaining]))) if np.any(remaining) else 0.0,
        "verifier_gt_utility_spearman": float(np.mean([_corr(verifier_scores[i], arrays["utilities"][i], spearman=True) for i in range(len(labels))])),
        "verifier_selected_gt_best_rate": float(np.mean(selected_slots == gt_order[:, 0])),
        "verifier_selected_gt_top2_rate": float(np.mean(np.any(selected_slots[:, None] == gt_order[:, :2], axis=1))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root()
    device = _cuda(args.device)
    data = _load_data(data_root, device)
    proposal_train, proposal_val = _proposal_scores(data, device, args.inference_batch_size)
    train_arrays = _build_sequential_arrays(data["train_rows"], data["train_cache"], data["train_target"], proposal_train, data["stats"])
    val_arrays = _build_sequential_arrays(data["val_rows"], data["val_cache"], data["val_target"], proposal_val, data["stats"])
    train_inputs = _verifier_inputs(train_arrays)
    val_inputs = _verifier_inputs(val_arrays)
    train_targets = train_arrays["utilities"]
    val_targets = val_arrays["utilities"]
    checkpoint = data_root / "checkpoints/reduced12_eight_placement_v1/top3_sequential_hypothesis_verification/verifier_best.pth"
    summary_path = EXPERIMENT / "verifier_training.json"
    if checkpoint.exists() and summary_path.exists():
        training = json.loads(summary_path.read_text(encoding="utf-8"))
        print("[sequential-verifier] skip completed checkpoint", flush=True)
    else:
        training = _train_verifier(train_inputs, train_targets, val_inputs, val_targets, device, checkpoint, summary_path, args.batch_size)
    verifier, input_mean, input_std = _load_verifier(checkpoint, device)
    with torch.inference_mode():
        verifier_scores = verifier(torch.from_numpy(((val_inputs - input_mean) / input_std).astype(np.float32)).to(device)).cpu().numpy()
    selected_slots = np.argmax(verifier_scores, axis=1).astype(np.int64)
    moving = _proposal_baselines(val_arrays)
    moving["Candidate-Conditioned Spatial"] = moving["Proposal-H1"]
    moving["Top3-SequentialVerifier-H2Only"] = _terminal_metrics(val_arrays, selected_slots, name="Top3-SequentialVerifier-H2Only")
    moving["Top3-SequentialVerifier-Fusion"] = _terminal_metrics(val_arrays, selected_slots, fusion=True, name="Top3-SequentialVerifier-Fusion")
    moving.update(_oracle_metrics(val_arrays, data["val_cache"], data["val_rows"]))
    frozen_checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/stage_c/set_ranker_best.pth"
    frozen_model = _load_baseline(frozen_checkpoint, data["summary"], device)
    frozen_set = StayDataset(data["val_rows"], data["val_cache"], data["stats"])
    frozen_stay, frozen_candidate = stay_scores(frozen_model, frozen_set, device, args.inference_batch_size, baseline=True)
    frozen_action = np.zeros(len(val_arrays["labels"]), dtype=np.int64)
    for i in range(len(frozen_action)):
        valid = np.flatnonzero(data["val_cache"]["candidate_mask"][i])
        action_values = np.concatenate([[frozen_stay[i]], frozen_candidate[i, valid]])
        chosen = int(np.argmax(action_values))
        frozen_action[i] = 0 if chosen == 0 else int(valid[chosen - 1]) + 1
    moving["FrozenStageCv0"] = _terminal_action_metrics(val_arrays, frozen_action, "FrozenStageCv0")
    diagnostics = _diagnostics(val_arrays, selected_slots, verifier_scores)
    diagnostics["proposal_h1_accuracy"] = float(moving["Proposal-H1"]["accuracy"])
    diagnostics["proposal_h1_macro_f1"] = float(moving["Proposal-H1"]["macro_f1"])
    diagnostics["train_contexts"] = len(train_arrays["labels"])
    diagnostics["val_moving_contexts"] = len(val_arrays["labels"])
    diagnostics["top3_eligible_train"] = len(train_arrays["labels"])
    diagnostics["top3_eligible_val"] = len(val_arrays["labels"])
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": "REDUCED12_TOP3_SEQUENTIAL_HYPOTHESIS_VERIFICATION",
        "labels": list(LABELS),
        "population": {"train_contexts": len(train_arrays["labels"]), "val_moving_contexts": len(val_arrays["labels"]), "val_full_contexts": len(data["val_full"])},
        "metrics_moving": moving,
        "training": training,
        "diagnostics": diagnostics,
        "protocol": {"proposal": "Candidate-Conditioned Spatial with stay + legal candidates", "h2_actions": ["STOP at H1", "proposal rank2", "proposal rank3"], "terminal_observation": "real archived skeleton through frozen reduced12 ST-GCN", "verifier_input_excludes_rank2_rank3_evidence": True},
        "test_used": False, "test_read": False, "future_rgb_used": False, "future_dino_used": False, "skeleton_regenerated": False,
        "train_used_for_h2_verifier": True, "h1_real_observation_used_at_h2": True, "future_rank2_rank3_skeleton_used_for_train_target_and_terminal_eval_only": True, "gt_action_used_for_train_target_and_privileged_oracle_only": True, "deployable_learned_verifier": True,
    }
    (EXPERIMENT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "sequential_baselines.json").write_text(json.dumps({key: moving[key] for key in ("S0-only", "FrozenStageCv0", "Candidate-Conditioned Spatial", "Proposal-H1", "Proposal-H1-H2Fixed", "Proposal-H1-H2Fixed-Fusion")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "verifier_metrics.json").write_text(json.dumps({key: moving[key] for key in ("Top3-SequentialVerifier-H2Only", "Top3-SequentialVerifier-Fusion")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "oracle_ceiling.json").write_text(json.dumps({key: moving[key] for key in ("AnyCorrect Oracle", "Top2-AnyCorrect", "Top3-AnyCorrect", "Top3-GTSequentialVerifier", "Top3-GTSequentialVerifier-Fusion", "Real-GTMargin Oracle")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_class = {label: {method: metric["per_class"][label] for method, metric in moving.items() if isinstance(metric, Mapping) and "per_class" in metric} for label in LABELS}
    (EXPERIMENT / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    spatial = moving["Candidate-Conditioned Spatial"]
    learned = moving["Top3-SequentialVerifier-H2Only"]
    gt = moving["Top3-GTSequentialVerifier"]
    lines = ["# Reduced12 Top-3 sequential hypothesis verification", "", "Train/Val only; policy Test was not read and no RGB/DINO/skeleton data was regenerated.", "", "## Moving Val", "", "| Method | Accuracy | Macro-F1 | H2 move rate | STOP rate | ΔAcc vs Candidate Spatial |", "|---|---:|---:|---:|---:|---:|"]
    for name, metric in moving.items():
        if "accuracy" not in metric:
            continue
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('h2_move_rate', 0.0):.6f} | {metric.get('stop_rate', 0.0):.6f} | {100.0 * (metric['accuracy'] - spatial['accuracy']):+.3f} pp |")
    lines.extend(["", "## Sequential diagnostics", "", f"H1 proposal Accuracy={moving['Proposal-H1']['accuracy']:.6f}; H1→rank2 fixed Accuracy={moving['Proposal-H1-H2Fixed']['accuracy']:.6f}; learned verifier H2-only={learned['accuracy']:.6f}; GT verifier ceiling={gt['accuracy']:.6f}.", f"H1 entropy change (H1 - s0)={diagnostics['h1_entropy_change']:.6f}; H1 true-class logp gain={diagnostics['h1_true_class_logp_gain']:.6f}.", f"On H1-wrong contexts with a correct Top-3 option, learned verifier selected a correct remaining action at {diagnostics['learned_selected_correct_remaining']:.6f}, STOP at {diagnostics['learned_stop_on_remaining']:.6f}; net rescue-harm={diagnostics['net_rescue']}.", f"Verifier utility Spearman (mean within-context)={diagnostics['verifier_gt_utility_spearman']:.6f}; selected GT-best={diagnostics['verifier_selected_gt_best_rate']:.6f}; selected GT-top2={diagnostics['verifier_selected_gt_top2_rate']:.6f}.", "", "## Scientific decision", "", ("The learned sequential verifier reaches at least 55% Moving Accuracy; real H1 observation plus shortlist verification is a promising direction." if learned["accuracy"] >= 0.55 else "The learned sequential verifier remains below 55% Moving Accuracy; this run does not establish a large deployable gain from shortlist verification."), ("The GT sequential ceiling is at least 60%, while the learned verifier is substantially lower, indicating that the Top-3 shortlist is useful but H2 hypothesis discrimination remains the bottleneck." if gt["accuracy"] >= 0.60 and learned["accuracy"] < 0.55 else "The GT sequential ceiling does not by itself establish a large recoverable Top-3 gap; do not infer a new method from this diagnostic alone."), "No subsequent method was started automatically.", "", "test_used=false; future_rgb_used=false; future_dino_used=false; deployable_learned_verifier=true."])
    (EXPERIMENT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.inference_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    result = run(args)
    print(json.dumps({"output": str((EXPERIMENT / "result.json").resolve()), "val_moving": result["population"]["val_moving_contexts"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
