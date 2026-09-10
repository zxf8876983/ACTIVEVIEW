#!/usr/bin/env python3
"""Train/Val-only single-step set-level real-evidence ceiling study."""

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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.generation.utility_labels import file_sha256
from activeview.scripts.experiments.run_reduced12_future_recognition_evidence_prediction import (
    _candidate_alignment,
    _covered_rows,
    _load_dino,
    _read_npz,
    _subset_cache,
)


SEED = 42
NUM_CLASSES = 12
GEOMETRY_DIM = 11
HIDDEN_DIM = 128
EPOCHS = 20
BATCH_SIZE = 2048
SET_BATCH_SIZE = 512
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
EXPERIMENT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/single_step_set_level_real_evidence"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA")
    return device


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        out = np.empty(v.size, dtype=np.float64)
        sorted_v = v[order]
        start = 0
        while start < v.size:
            end = start + 1
            while end < v.size and sorted_v[end] == sorted_v[start]:
                end += 1
            out[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return out
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for index, name in enumerate(LABELS):
        tp = float(confusion[index, index])
        support = float(confusion[index].sum())
        predicted = float(confusion[:, index].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": int(support), "recall": recall, "f1": f1}
    return {"n": int(labels.size), "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0, "macro_f1": float(np.mean(f1_values)), "per_class": per_class, "confusion_matrix": confusion.tolist()}


def _action_set(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Construct one common action axis: index 0 is stay, then legal moves."""
    n_rows, max_candidates = cache["candidate_logp"].shape[:2]
    action_count = max_candidates + 1
    logp = np.zeros((n_rows, action_count, NUM_CLASSES), dtype=np.float32)
    logp[:, 0] = cache["current_logp"]
    logp[:, 1:] = cache["candidate_logp"]
    mask = np.zeros((n_rows, action_count), dtype=bool)
    mask[:, 0] = True
    mask[:, 1:] = cache["candidate_mask"]
    geometry = np.zeros((n_rows, action_count, GEOMETRY_DIM), dtype=np.float32)
    action_ids = np.full((n_rows, action_count), -1, dtype=np.int16)
    action_ids[:, 1:] = cache["candidate_ids"]
    for row_index, row in enumerate(rows):
        ids = [int(value) for value in row["candidate_viewpoint_ids"]]
        row_geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        for candidate_index in np.flatnonzero(cache["candidate_mask"][row_index]):
            viewpoint = int(cache["candidate_ids"][row_index, candidate_index])
            geometry_index = ids.index(viewpoint)
            geometry[row_index, candidate_index + 1] = row_geometry[geometry_index]
    labels = np.asarray(cache["labels"], dtype=np.int64)
    probabilities = _softmax(np.where(mask[..., None], logp, -1e9))
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=-1)
    action_correct = np.argmax(logp, axis=-1) == labels[:, None]
    utilities = np.zeros((n_rows, action_count), dtype=np.float32)
    for row_index, label in enumerate(labels):
        for action_index in np.flatnonzero(mask[row_index]):
            values = logp[row_index, action_index]
            utilities[row_index, action_index] = values[label] - np.max(np.delete(values, label))
    return {"logp": logp, "probabilities": probabilities, "entropy": entropy, "mask": mask, "geometry": geometry, "action_ids": action_ids, "labels": labels, "correct": action_correct, "utility": utilities}


def _pick(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.where(mask, scores, -np.inf)
    return np.argmax(values, axis=1).astype(np.int64)


def _selection_metrics(name: str, actions: Mapping[str, np.ndarray], selected: np.ndarray, score_matrix: np.ndarray | None = None) -> dict[str, Any]:
    labels = actions["labels"]
    logp = actions["logp"]
    predictions = np.asarray([int(np.argmax(logp[i, action])) for i, action in enumerate(selected)], dtype=np.int64)
    metric = _metrics(labels, predictions)
    s0_correct = np.argmax(logp[:, 0], axis=1) == labels
    selected_correct = predictions == labels
    move_rate = float(np.mean(selected > 0)) if len(selected) else 0.0
    gt_scores = np.take_along_axis(actions["logp"], labels[:, None, None], axis=2).squeeze(-1)
    gt_oracle = _pick(gt_scores, actions["mask"])
    metric.update({
        "selector": name,
        "move_rate": move_rate,
        "stay_rate": 1.0 - move_rate,
        "selected_candidate_correct_rate": float(np.mean(selected_correct)) if len(selected_correct) else 0.0,
        "s0_error_correction_rate": float(np.sum(~s0_correct & selected_correct) / max(np.sum(~s0_correct), 1)),
        "s0_correct_harm_rate": float(np.sum(s0_correct & ~selected_correct) / max(np.sum(s0_correct), 1)),
        "mean_selected_gt_true_logp": float(np.mean([logp[i, selected[i], labels[i]] for i in range(len(labels))])) if len(labels) else 0.0,
        "mean_selected_gt_margin": float(np.mean([actions["utility"][i, selected[i]] for i in range(len(labels))])) if len(labels) else 0.0,
        "selected_action_ids": [None if int(action) == 0 else int(actions["action_ids"][i, action]) for i, action in enumerate(selected)],
        "selected_correct": selected_correct.tolist(),
        "overlap_gt_true_logp_oracle": float(np.mean(selected == gt_oracle)) if len(labels) else 0.0,
    })
    if score_matrix is not None:
        ranks = [_spearman(score_matrix[i, actions["mask"][i]], actions["utility"][i, actions["mask"][i]]) for i in range(len(labels))]
        metric["context_ranking_spearman"] = float(np.mean(ranks)) if ranks else 0.0
    return metric


def _oracle_and_consensus(actions: Mapping[str, np.ndarray]) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    p, mask, labels = actions["probabilities"], actions["mask"], actions["labels"]
    n_rows = len(labels)
    valid_count = np.maximum(mask.sum(axis=1), 1)
    mean_consensus = (p * mask[..., None]).sum(axis=1) / valid_count[:, None]
    median_consensus = np.asarray([np.median(p[i, mask[i]], axis=0) for i in range(n_rows)])
    vote_consensus = np.zeros((n_rows, NUM_CLASSES), dtype=np.float32)
    weighted_consensus = np.zeros((n_rows, NUM_CLASSES), dtype=np.float32)
    for i in range(n_rows):
        valid = np.flatnonzero(mask[i])
        votes = np.argmax(p[i, valid], axis=1)
        vote_consensus[i, int(np.bincount(votes, minlength=NUM_CLASSES).argmax())] = 1.0
        weights = 1.0 - actions["entropy"][i, valid] / np.log(NUM_CLASSES)
        if float(weights.sum()) <= 1e-8:
            weights = np.ones_like(weights)
        weighted_consensus[i] = (p[i, valid] * weights[:, None]).sum(axis=0) / weights.sum()
    def class_scores(consensus: np.ndarray) -> np.ndarray:
        class_index = np.argmax(consensus, axis=1)[:, None, None]
        expanded = np.broadcast_to(class_index, (n_rows, p.shape[1], 1))
        return np.take_along_axis(p, expanded, axis=2).squeeze(-1)

    scores: dict[str, np.ndarray] = {
        "ConsensusMean": class_scores(mean_consensus),
        "ConsensusMedian": class_scores(median_consensus),
        "VoteConsensus": class_scores(vote_consensus),
        "ConfidenceWeightedConsensus": class_scores(weighted_consensus),
    }
    # The advanced indexing expression above is intentionally reshaped to [N,A].
    scores = {name: np.asarray(value).reshape(n_rows, -1) for name, value in scores.items()}
    selected: dict[str, np.ndarray] = {name: _pick(value, mask) for name, value in scores.items()}
    # Aligned oracle definitions: stay is a real action and is never excluded.
    s0_pred = np.argmax(actions["logp"][:, 0], axis=1)
    any_selected = np.zeros(n_rows, dtype=np.int64)
    for i in range(n_rows):
        if int(s0_pred[i]) == int(labels[i]):
            any_selected[i] = 0
        else:
            correct_moves = np.flatnonzero(actions["correct"][i] & actions["mask"][i])
            any_selected[i] = int(correct_moves[0]) if len(correct_moves) else 0
    selected["AnyCorrect Oracle"] = any_selected
    gt_scores = np.take_along_axis(actions["logp"], labels[:, None, None], axis=2).squeeze(-1)
    selected["Real-GTTrueLogP Oracle"] = _pick(gt_scores, mask)
    selected["Real-GTMargin Oracle"] = _pick(actions["utility"], mask)
    metrics = {name: _selection_metrics(name, actions, value, scores.get(name)) for name, value in selected.items()}
    for name in scores:
        metrics[name]["candidate_score_spearman_vs_gt_true_logp"] = float(np.mean([_spearman(scores[name][i, mask[i]], actions["logp"][i, mask[i], labels[i]]) for i in range(n_rows)]))
    return metrics, scores


class _CandidateDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class _SetDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: np.ndarray, mask: np.ndarray, labels: np.ndarray, utilities: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(bool))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.utilities = torch.from_numpy(utilities.astype(np.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.mask[index], self.labels[index], self.utilities[index]


class _Ranker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


class _SetEncoder(nn.Module):
    def __init__(self, input_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.encoder = nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU())
        if mode == "action":
            self.head = nn.Sequential(nn.Linear(HIDDEN_DIM + NUM_CLASSES, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, NUM_CLASSES))
        else:
            self.head = nn.Sequential(nn.Linear(2 * HIDDEN_DIM + NUM_CLASSES, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))

    def forward(self, action_inputs: torch.Tensor, mask: torch.Tensor, s0_logp: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(action_inputs)
        masked = embeddings * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        if self.mode == "action":
            return self.head(torch.cat([pooled, s0_logp], dim=-1))
        global_context = pooled.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        s0_context = s0_logp.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        return self.head(torch.cat([embeddings, global_context, s0_context], dim=-1)).squeeze(-1)


def _flatten_actions(actions: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = actions["probabilities"]
    logp = actions["logp"]
    stats = np.stack([actions["entropy"], np.sort(p, axis=-1)[:, :, -1] - np.sort(p, axis=-1)[:, :, -2], p.max(axis=-1)], axis=-1)
    features = np.concatenate([actions["logp"], stats, actions["geometry"]], axis=-1)
    return features, actions["mask"], actions["utility"]


def _train_ranker(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, checkpoint: Path, summary_path: Path, device: torch.device, batch_size: int, kind: str) -> dict[str, Any]:
    _seed()
    model = _Ranker(train_x.shape[1]).to(device)
    loader = DataLoader(_CandidateDataset(train_x[train_mask_global], train_y[train_mask_global]), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(_CandidateDataset(val_x[val_mask_global], val_y[val_mask_global]), batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_epoch = float("inf"), 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); train_losses: list[float] = []
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            if kind == "correctness_bce":
                loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.to(device))
            else:
                loss = nn.functional.smooth_l1_loss(logits, targets.to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, targets in val_loader:
                logits = model(inputs.to(device))
                if kind == "correctness_bce":
                    value = nn.functional.binary_cross_entropy_with_logits(logits, targets.to(device))
                else:
                    value = nn.functional.smooth_l1_loss(logits, targets.to(device))
                val_losses.append(float(value.cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}; history.append(record)
        print(f"[aligned ranker {kind}] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss, best_epoch = record["val_loss"], epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train_x.shape[1]), "kind": kind, "epoch": epoch, "seed": SEED}, checkpoint)
    summary = {"kind": kind, "epochs": EPOCHS, "hidden_dim": HIDDEN_DIM, "best_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started, "history": history, "test_used": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _train_ranker_listwise(
    train_x: np.ndarray,
    train_mask: np.ndarray,
    train_utility: np.ndarray,
    val_x: np.ndarray,
    val_mask: np.ndarray,
    val_utility: np.ndarray,
    checkpoint: Path,
    summary_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train a context-level ListNet ranker over the full action set."""
    _seed()
    model = _Ranker(train_x.shape[-1]).to(device)
    dummy_labels = np.zeros(len(train_x), dtype=np.int64)
    dummy_val_labels = np.zeros(len(val_x), dtype=np.int64)
    loader = DataLoader(_SetDataset(train_x, train_mask, dummy_labels, train_utility), batch_size=SET_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(_SetDataset(val_x, val_mask, dummy_val_labels, val_utility), batch_size=SET_BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_epoch = float("inf"), 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); train_losses: list[float] = []
        for inputs, mask, _, utilities in loader:
            inputs, mask, utilities = inputs.to(device), mask.to(device), utilities.to(device)
            scores = model(inputs).masked_fill(~mask, -1e9)
            target_distribution = torch.softmax(utilities.masked_fill(~mask, -1e9), dim=-1)
            loss = -(target_distribution * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, mask, _, utilities in val_loader:
                inputs, mask, utilities = inputs.to(device), mask.to(device), utilities.to(device)
                scores = model(inputs).masked_fill(~mask, -1e9)
                target_distribution = torch.softmax(utilities.masked_fill(~mask, -1e9), dim=-1)
                val_losses.append(float((-(target_distribution * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}; history.append(record)
        print(f"[aligned ranker gt-margin listwise] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss, best_epoch = record["val_loss"], epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train_x.shape[-1]), "kind": "gt_margin_listwise", "epoch": epoch, "seed": SEED}, checkpoint)
    summary = {"kind": "gt_margin_listwise", "epochs": EPOCHS, "hidden_dim": HIDDEN_DIM, "best_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started, "history": history, "test_used": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _train_set_model(train_x: np.ndarray, train_mask: np.ndarray, train_labels: np.ndarray, train_utilities: np.ndarray, val_x: np.ndarray, val_mask: np.ndarray, val_labels: np.ndarray, val_utilities: np.ndarray, s0_train: np.ndarray, s0_val: np.ndarray, checkpoint: Path, summary_path: Path, device: torch.device, mode: str) -> dict[str, Any]:
    _seed()
    model = _SetEncoder(train_x.shape[-1], mode).to(device)
    train_loader = DataLoader(_SetDataset(train_x, train_mask, train_labels, train_utilities), batch_size=SET_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(_SetDataset(val_x, val_mask, val_labels, val_utilities), batch_size=SET_BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss, best_epoch = float("inf"), 0; history: list[dict[str, float | int]] = []; started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); train_losses: list[float] = []
        for inputs, mask, labels, utilities in train_loader:
            s0 = inputs[:, 0, :NUM_CLASSES].to(device)
            output = model(inputs.to(device), mask.to(device), s0)
            if mode == "action":
                loss = nn.functional.cross_entropy(output, labels.to(device))
            else:
                target = actions_correct_from_utility(utilities).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(output.masked_fill(~mask.to(device), 0.0), target.masked_fill(~mask.to(device), 0.0), weight=mask.to(device).float())
                loss = loss * (mask.to(device).numel() / mask.to(device).sum().clamp_min(1.0))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, mask, labels, utilities in val_loader:
                s0 = inputs[:, 0, :NUM_CLASSES].to(device)
                output = model(inputs.to(device), mask.to(device), s0)
                if mode == "action":
                    value = nn.functional.cross_entropy(output, labels.to(device))
                else:
                    target = actions_correct_from_utility(utilities).to(device); mask_device = mask.to(device)
                    value = nn.functional.binary_cross_entropy_with_logits(output.masked_fill(~mask_device, 0.0), target.masked_fill(~mask_device, 0.0), weight=mask_device.float())
                    value = value * (mask_device.numel() / mask_device.sum().clamp_min(1.0))
                val_losses.append(float(value.cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}; history.append(record)
        print(f"[set {mode}] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss, best_epoch = record["val_loss"], epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train_x.shape[-1]), "mode": mode, "epoch": epoch, "seed": SEED}, checkpoint)
    summary = {"mode": mode, "epochs": EPOCHS, "hidden_dim": HIDDEN_DIM, "best_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started, "history": history, "test_used": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def actions_correct_from_utility(utilities: torch.Tensor) -> torch.Tensor:
    # Utility is GT-class logp minus the best competing class; positive means
    # the frozen recognizer's argmax is the GT class.
    return (utilities > 0.0).float()


def _predict_ranker(model: _Ranker, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    out = np.zeros(len(inputs), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(inputs), BATCH_SIZE):
            out[start:start + BATCH_SIZE] = model(torch.from_numpy(inputs[start:start + BATCH_SIZE]).to(device)).cpu().numpy()
    return out


def _load_ranker(path: Path, device: torch.device) -> _Ranker:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = _Ranker(int(payload["input_dim"])).to(device); model.load_state_dict(payload["state_dict"]); model.eval(); return model


def _load_set_model(path: Path, device: torch.device) -> _SetEncoder:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = _SetEncoder(int(payload["input_dim"]), str(payload["mode"])).to(device); model.load_state_dict(payload["state_dict"]); model.eval(); return model


def _set_predictions(model: _SetEncoder, inputs: np.ndarray, mask: np.ndarray, s0_logp: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), SET_BATCH_SIZE):
            batch_inputs = torch.from_numpy(inputs[start:start + SET_BATCH_SIZE]).to(device)
            batch_mask = torch.from_numpy(mask[start:start + SET_BATCH_SIZE]).to(device)
            batch_s0 = torch.from_numpy(s0_logp[start:start + SET_BATCH_SIZE]).to(device)
            values.append(model(batch_inputs, batch_mask, batch_s0).cpu().numpy())
    return np.concatenate(values, axis=0)


def _reference_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
    previous = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").read_text(encoding="utf-8"))
    visual = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_visual_context_batch/result.json").read_text(encoding="utf-8"))
    moving = {"S0-only": previous["metrics_moving"]["S0-only"], "FrozenStageCv0": previous["metrics_moving"]["FrozenStageCv0"], "Candidate-Conditioned Spatial": visual["metrics_moving"]["candidate_conditioned_spatial"]}
    full = {name: previous["metrics_full"][name] for name in ("S0-only", "FrozenStageCv0", "AnyCorrect Oracle")}
    return moving, full


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root().resolve(); device = _require_cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"; diag_root = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"
    train_rows = load_jsonl(policy_root / "stage_c/features/train.jsonl"); val_rows_all = load_jsonl(policy_root / "stage_c/features/val.jsonl")
    stage_d_val_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/val.jsonl")}
    dino_lookup, _, _ = _load_dino(data_root)
    val_rows, val_indices = _covered_rows(val_rows_all, stage_d_val_ids, dino_lookup)
    train_cache = _read_npz(diag_root / "train_candidate_true_logp.npz"); val_all_cache = _read_npz(diag_root / "val_all_candidate_true_logp.npz")
    val_cache = _subset_cache(val_all_cache, val_indices); _candidate_alignment(train_rows, train_cache); _candidate_alignment(val_rows, val_cache)
    train_actions = _action_set(train_rows, train_cache); val_actions = _action_set(val_rows, val_cache)
    output = args.output_dir.resolve() if args.output_dir else EXPERIMENT_ROOT; branch_root = output / "branch_results"; output.mkdir(parents=True, exist_ok=True); branch_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = data_root / "checkpoints/policy_reduced12_eight_placement_v1/single_step_set_level_real_evidence"

    # Batch A: all set-level non-parametric baselines plus aligned privileged oracles.
    batch_a, score_a = _oracle_and_consensus(val_actions)
    (branch_root / "consensus.json").write_text(json.dumps({k: v for k, v in batch_a.items() if k.startswith("Consensus")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "oracle_alignment.json").write_text(json.dumps({k: batch_a[k] for k in ("AnyCorrect Oracle", "Real-GTTrueLogP Oracle", "Real-GTMargin Oracle")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Candidate ranker input: logp + entropy/margin/maxprob + geometry.
    train_rank_raw, train_rank_mask, train_rank_utility = _flatten_actions(train_actions); val_rank_raw, val_rank_mask, val_rank_utility = _flatten_actions(val_actions)
    train_flat_mask = train_rank_mask.reshape(-1); val_flat_mask = val_rank_mask.reshape(-1)
    global train_mask_global, val_mask_global
    train_mask_global, val_mask_global = train_flat_mask, val_flat_mask
    train_rank_flat_raw = train_rank_raw.reshape(-1, train_rank_raw.shape[-1])
    val_rank_flat_raw = val_rank_raw.reshape(-1, val_rank_raw.shape[-1])
    rank_mean = train_rank_flat_raw[train_flat_mask].mean(axis=0); rank_std = train_rank_flat_raw[train_flat_mask].std(axis=0); rank_std[rank_std < 1e-6] = 1.0
    train_rank = ((train_rank_raw - rank_mean) / rank_std).reshape(train_rank_raw.shape); val_rank = ((val_rank_raw - rank_mean) / rank_std).reshape(val_rank_raw.shape)
    train_flat, val_flat = train_rank.reshape(-1, train_rank.shape[-1]), val_rank.reshape(-1, val_rank.shape[-1])
    train_correct_flat, val_correct_flat = train_actions["correct"].reshape(-1).astype(np.float32), val_actions["correct"].reshape(-1).astype(np.float32)
    train_util_flat, val_util_flat = train_actions["utility"].reshape(-1), val_actions["utility"].reshape(-1)
    training: dict[str, Any] = {}; ranker_metrics: dict[str, Any] = {}
    rank_specs = (("real_correctness_bce", train_correct_flat, val_correct_flat, "correctness_bce"), ("real_gt_margin_listwise", train_util_flat, val_util_flat, "gt_margin_listwise"))
    for name, train_target, val_target, kind in rank_specs:
        checkpoint = checkpoint_root / f"{name}_best.pth"; summary_path = output / f"{name}_training.json"
        if checkpoint.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8")); print(f"[{name}] skip existing checkpoint", flush=True)
        else:
            if kind == "correctness_bce":
                summary = _train_ranker(train_flat, train_target, val_flat, val_target, checkpoint, summary_path, device, args.batch_size, kind)
            else:
                summary = _train_ranker_listwise(train_rank, train_rank_mask, train_rank_utility, val_rank, val_rank_mask, val_rank_utility, checkpoint, summary_path, device)
        training[name] = summary; model = _load_ranker(checkpoint, device); val_scores = _predict_ranker(model, val_flat, device).reshape(len(val_rows), -1)
        selected = _pick(val_scores, val_actions["mask"]); metric = _selection_metrics("RealEvidence-CorrectnessBCE" if kind == "correctness_bce" else "RealEvidence-GTMarginListwise", val_actions, selected, val_scores)
        metric["candidate_score_spearman_vs_gt_true_logp"] = float(np.mean([_spearman(val_scores[i, val_actions["mask"][i]], val_actions["logp"][i, val_actions["mask"][i], val_actions["labels"][i]]) for i in range(len(val_rows))]))
        ranker_metrics[metric["selector"]] = metric; np.savez_compressed(branch_root / f"{name}_scores.npz", scores=val_scores)
    (branch_root / "real_rankers.json").write_text(json.dumps(ranker_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Batch B: set-level action inference and hard/soft selectors.
    set_input_dim = NUM_CLASSES + GEOMETRY_DIM
    train_set_raw = np.concatenate([train_actions["logp"], train_actions["geometry"]], axis=-1); val_set_raw = np.concatenate([val_actions["logp"], val_actions["geometry"]], axis=-1)
    set_mean = train_set_raw[train_actions["mask"]].mean(axis=0); set_std = train_set_raw[train_actions["mask"]].std(axis=0); set_std[set_std < 1e-6] = 1.0
    train_set = ((train_set_raw - set_mean) / set_std).astype(np.float32); val_set = ((val_set_raw - set_mean) / set_std).astype(np.float32)
    action_checkpoint = checkpoint_root / "set_action_best.pth"; action_summary_path = output / "set_action_training.json"
    if action_checkpoint.exists() and action_summary_path.exists():
        action_summary = json.loads(action_summary_path.read_text(encoding="utf-8")); print("[set action] skip existing checkpoint", flush=True)
    else:
        action_summary = _train_set_model(train_set, train_actions["mask"], train_actions["labels"], train_actions["utility"], val_set, val_actions["mask"], val_actions["labels"], val_actions["utility"], train_actions["logp"][:, 0], val_actions["logp"][:, 0], action_checkpoint, action_summary_path, device, "action")
    action_model = _load_set_model(action_checkpoint, device); action_logits = _set_predictions(action_model, val_set, val_actions["mask"], val_actions["logp"][:, 0], device); action_posterior = _softmax(action_logits)
    action_classifier_metric = _metrics(val_actions["labels"], np.argmax(action_logits, axis=1)); action_classifier_metric["confusion_matrix"] = action_classifier_metric["confusion_matrix"]
    inferred = np.argmax(action_logits, axis=1); hard_scores = np.take_along_axis(val_actions["logp"], inferred[:, None, None], axis=2).squeeze(-1); soft_scores = (action_posterior[:, None, :] * val_actions["probabilities"]).sum(axis=2)
    set_action_metrics = {"RealEvidence-SetAction": _selection_metrics("RealEvidence-SetAction", val_actions, _pick(hard_scores, val_actions["mask"]), hard_scores), "RealEvidence-SetActionSoft": _selection_metrics("RealEvidence-SetActionSoft", val_actions, _pick(soft_scores, val_actions["mask"]), soft_scores)}
    for metric in set_action_metrics.values():
        metric["candidate_score_spearman_vs_gt_true_logp"] = float(np.mean([_spearman((hard_scores if metric["selector"] == "RealEvidence-SetAction" else soft_scores)[i, val_actions["mask"][i]], val_actions["logp"][i, val_actions["mask"][i], val_actions["labels"][i]]) for i in range(len(val_rows))]))
    set_action_metrics["SetActionClassifier"] = action_classifier_metric
    (branch_root / "set_action.json").write_text(json.dumps(set_action_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Batch C: set-level correctness selector.
    correct_checkpoint = checkpoint_root / "set_correctness_best.pth"; correct_summary_path = output / "set_correctness_training.json"
    if correct_checkpoint.exists() and correct_summary_path.exists():
        correct_summary = json.loads(correct_summary_path.read_text(encoding="utf-8")); print("[set correctness] skip existing checkpoint", flush=True)
    else:
        correct_summary = _train_set_model(train_set, train_actions["mask"], train_actions["labels"], train_actions["utility"], val_set, val_actions["mask"], val_actions["labels"], val_actions["utility"], train_actions["logp"][:, 0], val_actions["logp"][:, 0], correct_checkpoint, correct_summary_path, device, "correctness")
    correct_model = _load_set_model(correct_checkpoint, device); correctness_scores = _set_predictions(correct_model, val_set, val_actions["mask"], val_actions["logp"][:, 0], device); set_correct_metrics = {"RealEvidence-SetCorrectness": _selection_metrics("RealEvidence-SetCorrectness", val_actions, _pick(correctness_scores, val_actions["mask"]), correctness_scores)}
    set_correct_metrics["RealEvidence-SetCorrectness"]["candidate_score_spearman_vs_gt_true_logp"] = float(np.mean([_spearman(correctness_scores[i, val_actions["mask"][i]], val_actions["logp"][i, val_actions["mask"][i], val_actions["labels"][i]]) for i in range(len(val_rows))]))
    (branch_root / "set_correctness.json").write_text(json.dumps(set_correct_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    moving_ref, full_ref = _reference_metrics(); metrics = {**moving_ref, **batch_a, **ranker_metrics, **{k: v for k, v in set_action_metrics.items() if k != "SetActionClassifier"}, **set_correct_metrics}
    frozen_acc = float(moving_ref["FrozenStageCv0"]["accuracy"]); frozen_f1 = float(moving_ref["FrozenStageCv0"]["macro_f1"]); spatial_acc = float(moving_ref["Candidate-Conditioned Spatial"]["accuracy"]); spatial_f1 = float(moving_ref["Candidate-Conditioned Spatial"]["macro_f1"])
    for metric in metrics.values():
        if "accuracy" not in metric: continue
        metric["delta_vs_frozen_accuracy_pp"] = 100.0 * (float(metric["accuracy"]) - frozen_acc); metric["delta_vs_frozen_macro_f1_pp"] = 100.0 * (float(metric["macro_f1"]) - frozen_f1); metric["delta_vs_candidate_conditioned_spatial_accuracy_pp"] = 100.0 * (float(metric["accuracy"]) - spatial_acc); metric["delta_vs_candidate_conditioned_spatial_macro_f1_pp"] = 100.0 * (float(metric["macro_f1"]) - spatial_f1)
    set_models = {**set_action_metrics, **set_correct_metrics}; best_set = max((name for name in metrics if name.startswith("RealEvidence-Set")), key=lambda name: float(metrics[name]["accuracy"]))
    result = {"experiment_id": "REDUCED12_SINGLE_STEP_SET_LEVEL_REAL_EVIDENCE", "status": "COMPLETED", "labels": list(LABELS), "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "moving_val_actions": int(val_actions["mask"].sum()), "action_set": "stay + legal move candidates", "max_actions": int(val_actions["mask"].shape[1])}, "metrics_moving": metrics, "metrics_full": {**full_ref, "set_level_methods": "NOT_EVALUATED_BY_DESIGN"}, "set_action_classifier": action_classifier_metric, "training": {**training, "set_action": action_summary, "set_correctness": correct_summary}, "best_set_level_selector": {"name": best_set, "accuracy": metrics[best_set]["accuracy"], "macro_f1": metrics[best_set]["macro_f1"]}, "gaps": {"action_inference_gap_accuracy_pp": 100.0 * (metrics["Real-GTTrueLogP Oracle"]["accuracy"] - max(metrics["RealEvidence-SetAction"]["accuracy"], metrics["RealEvidence-SetActionSoft"]["accuracy"])), "set_decision_gap_accuracy_pp": 100.0 * (metrics["Real-GTTrueLogP Oracle"]["accuracy"] - metrics["RealEvidence-SetCorrectness"]["accuracy"]), "candidate_independent_gain_accuracy_pp": 100.0 * (metrics[best_set]["accuracy"] - metrics["RealEvidence-CorrectnessBCE"]["accuracy"])}, "oracle_alignment": {name: metrics[name]["accuracy"] for name in ("AnyCorrect Oracle", "Real-GTTrueLogP Oracle", "Real-GTMargin Oracle")}, "test_used": False, "train_used_for_set_models": True, "future_candidate_skeleton_used_only_to_obtain_real_evidence_and_terminal_eval": True, "gt_action_used_only_for_train_supervision_and_privileged_oracles": True, "predicted_future_evidence_used": False, "deployable": False}
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    diagnostics = {"oracle_alignment": result["oracle_alignment"], "set_action_classifier": action_classifier_metric, "set_level_selectors": {name: metrics[name] for name in metrics if name.startswith("RealEvidence-Set")}, "ranker_selectors": {name: metrics[name] for name in ranker_metrics}, "gaps": result["gaps"]}
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_class = {name: metric["per_class"] for name, metric in metrics.items() if isinstance(metric, dict) and "per_class" in metric}; (output / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Single-Step Set-Level Real-Evidence Upper-Bound Study", "", "Train/Val only. The common action set is stay + legal move candidates; policy Test was not read. No RGB, skeleton, DINO, ST-GCN, WM-E or JR artifact was regenerated or modified.", "", "## Moving Val", "", "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen | Move rate |", "|---|---:|---:|---:|---:|---:|"]
    for name, metric in metrics.items():
        if "accuracy" not in metric: continue
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('delta_vs_frozen_accuracy_pp', 0.0):+.3f}pp | {metric.get('delta_vs_frozen_macro_f1_pp', 0.0):+.3f}pp | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend(["", "## Oracle alignment", "", f"AnyCorrect={result['oracle_alignment']['AnyCorrect Oracle']:.6f}; GTTrueLogP={result['oracle_alignment']['Real-GTTrueLogP Oracle']:.6f}; GTMargin={result['oracle_alignment']['Real-GTMargin Oracle']:.6f}.", "", "## Set diagnostics", "", f"SetActionClassifier Accuracy/Macro-F1: {action_classifier_metric['accuracy']:.6f}/{action_classifier_metric['macro_f1']:.6f}.", f"Best set selector: **{best_set}**, {metrics[best_set]['accuracy']:.6f}/{metrics[best_set]['macro_f1']:.6f}.", f"Action-inference gap: {result['gaps']['action_inference_gap_accuracy_pp']:+.3f}pp; set-decision gap: {result['gaps']['set_decision_gap_accuracy_pp']:+.3f}pp; gain over aligned candidate-independent BCE: {result['gaps']['candidate_independent_gain_accuracy_pp']:+.3f}pp.", ""])
    best_acc = float(metrics[best_set]["accuracy"])
    if best_acc >= 0.60:
        lines.append("The set-level real-evidence selector reaches the 60% range, strongly supporting joint candidate-set action inference before further evidence prediction work.")
    elif best_acc >= 0.55:
        lines.append("The set-level real-evidence selector reaches the 55% range, indicating meaningful remaining potential in a set-level/hypothesis-conditioned selector.")
    else:
        lines.append("All set-level real-evidence selectors remain below 55%; even jointly observing true future evidence does not reliably recover the correct action in this single-step formulation. Further set-model scaling is not justified by this batch.")
    lines.extend(["", "The aligned stay-inclusive oracle values are reported explicitly; any residual differences from older candidate-only oracle numbers are due to the corrected action set, not a protocol change.", "", "`test_used=false`; `train_used_for_set_models=true`; `predicted_future_evidence_used=false`; `deployable=false`."])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=BATCH_SIZE); parser.add_argument("--output-dir", type=Path, default=None); args = parser.parse_args(); result = run(args); print(json.dumps({"status": result["status"], "test_used": result["test_used"], "best_set_level_selector": result["best_set_level_selector"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
