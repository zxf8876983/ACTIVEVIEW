#!/usr/bin/env python3
"""Train/Val-only decomposition of future recognition-evidence selection.

The experiment deliberately separates three effects: selectors that see real
candidate recognition evidence, small rankers trained on that real evidence,
and the same rankers trained on the existing predicted-evidence checkpoint.
No policy Test data are read and no frozen perception model is modified.
"""

from __future__ import annotations

import argparse
import hashlib
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
from activeview.data.generation.utility_labels import file_sha256
from activeview.data.preprocessing.cache import load_jsonl
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.experiments.run_reduced12_future_recognition_evidence_prediction import (
    EvidencePredictor,
    _candidate_alignment,
    _covered_rows,
    _flatten_inputs,
    _load_dino,
    _load_model,
    _predict_candidates,
    _read_npz,
    _signature,
    _subset_cache,
)


SEED = 42
NUM_CLASSES = 12
GEOMETRY_DIM = 11
HIDDEN_DIM = 128
EPOCHS = 20
BATCH_SIZE = 2048
LISTWISE_BATCH_SIZE = 512
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
FOCUS_CLASSES = ("bend", "stumble", "knock", "touching face")
EXPERIMENT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/recognition_evidence_decision_decomposition"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA")
    return device


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    # Average ranks make ties (common for legal candidate scores) well-defined.
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(LABELS):
        tp = float(confusion[class_id, class_id])
        support = float(confusion[class_id].sum())
        predicted = float(confusion[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": int(support), "recall": recall, "f1": f1}
    return {
        "n": int(labels.size),
        "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.clip(values.sum(axis=-1, keepdims=True), 1e-12, None)


def _evidence_features(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    evidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Flatten candidate evidence into ranker inputs and supervision arrays."""
    inputs: list[np.ndarray] = []
    correctness: list[float] = []
    utilities: list[float] = []
    refs: list[tuple[int, int]] = []
    labels = np.asarray(cache["labels"], dtype=np.int64)
    for row_index, row in enumerate(rows):
        valid = np.flatnonzero(cache["candidate_mask"][row_index])
        row_ids = [int(value) for value in row["candidate_viewpoint_ids"]]
        geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        current = np.asarray(cache["current_logp"][row_index], dtype=np.float32)
        for candidate_index in valid:
            candidate = np.asarray(evidence[row_index, candidate_index], dtype=np.float32)
            probabilities = _stable_softmax(candidate)
            entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
            top_two = np.partition(probabilities, -2)[-2:]
            margin = float(top_two[-1] - top_two[-2])
            max_probability = float(probabilities.max())
            viewpoint_id = int(cache["candidate_ids"][row_index, candidate_index])
            geometry_index = row_ids.index(viewpoint_id)
            inputs.append(np.concatenate([
                current,
                candidate,
                np.asarray([entropy, margin, max_probability], dtype=np.float32),
                geometry[geometry_index],
            ]))
            predicted_class = int(np.argmax(candidate))
            label = int(labels[row_index])
            correctness.append(float(predicted_class == label))
            other = np.delete(candidate, label)
            utilities.append(float(candidate[label] - np.max(other)))
            refs.append((row_index, int(candidate_index)))
    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(correctness, dtype=np.float32),
        np.asarray(utilities, dtype=np.float32),
        refs,
    )


class _CandidateDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


class _ListwiseDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, context_inputs: np.ndarray, utilities: np.ndarray, mask: np.ndarray) -> None:
        self.inputs = torch.from_numpy(context_inputs.astype(np.float32))
        self.utilities = torch.from_numpy(utilities.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(bool))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.utilities[index], self.mask[index]


class EvidenceRanker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def _context_tensors(
    flat_inputs: np.ndarray,
    utilities: np.ndarray,
    refs: Sequence[tuple[int, int]],
    row_count: int,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context_inputs = np.zeros((row_count, max_candidates, flat_inputs.shape[1]), dtype=np.float32)
    context_utilities = np.zeros((row_count, max_candidates), dtype=np.float32)
    context_mask = np.zeros((row_count, max_candidates), dtype=bool)
    for flat_index, (row_index, candidate_index) in enumerate(refs):
        context_inputs[row_index, candidate_index] = flat_inputs[flat_index]
        context_utilities[row_index, candidate_index] = utilities[flat_index]
        context_mask[row_index, candidate_index] = True
    return context_inputs, context_utilities, context_mask


def _train_bce(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    checkpoint: Path,
    summary_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    _seed()
    model = EvidenceRanker(train_inputs.shape[1]).to(device)
    loader = DataLoader(_CandidateDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(_CandidateDataset(val_inputs, val_targets), batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    started = time.time()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_values: list[float] = []
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        model.eval()
        val_values: list[float] = []
        with torch.inference_mode():
            for inputs, targets in val_loader:
                val_values.append(float(nn.functional.binary_cross_entropy_with_logits(model(inputs.to(device)), targets.to(device)).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_values)), "val_loss": float(np.mean(val_values))}
        history.append(record)
        print(f"[real/pred correctness BCE] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss, best_epoch = record["val_loss"], epoch
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train_inputs.shape[1]), "kind": "correctness_bce", "epoch": epoch, "seed": SEED}, checkpoint)
    summary = {"kind": "correctness_bce", "epochs": EPOCHS, "hidden_dim": HIDDEN_DIM, "learning_rate": 1e-3, "best_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started, "history": history, "test_used": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _train_listwise(
    train_inputs: np.ndarray,
    train_utilities: np.ndarray,
    train_mask: np.ndarray,
    val_inputs: np.ndarray,
    val_utilities: np.ndarray,
    val_mask: np.ndarray,
    checkpoint: Path,
    summary_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    _seed()
    model = EvidenceRanker(train_inputs.shape[-1]).to(device)
    loader = DataLoader(_ListwiseDataset(train_inputs, train_utilities, train_mask), batch_size=LISTWISE_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(_ListwiseDataset(val_inputs, val_utilities, val_mask), batch_size=LISTWISE_BATCH_SIZE, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    started = time.time()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_values: list[float] = []
        for inputs, utilities, mask in loader:
            inputs, utilities, mask = inputs.to(device), utilities.to(device), mask.to(device)
            scores = model(inputs)
            masked_scores = scores.masked_fill(~mask, -1e9)
            target_distribution = torch.softmax(utilities.masked_fill(~mask, -1e9), dim=-1)
            loss = -(target_distribution * torch.log_softmax(masked_scores, dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        model.eval()
        val_values: list[float] = []
        with torch.inference_mode():
            for inputs, utilities, mask in val_loader:
                inputs, utilities, mask = inputs.to(device), utilities.to(device), mask.to(device)
                scores = model(inputs).masked_fill(~mask, -1e9)
                target_distribution = torch.softmax(utilities.masked_fill(~mask, -1e9), dim=-1)
                val_values.append(float((-(target_distribution * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_values)), "val_loss": float(np.mean(val_values))}
        history.append(record)
        print(f"[real/pred GT-margin listwise] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss, best_epoch = record["val_loss"], epoch
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train_inputs.shape[-1]), "kind": "gt_margin_listwise", "epoch": epoch, "seed": SEED}, checkpoint)
    summary = {"kind": "gt_margin_listwise", "epochs": EPOCHS, "hidden_dim": HIDDEN_DIM, "learning_rate": 1e-3, "best_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.time() - started, "history": history, "test_used": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _predict_ranker(model: EvidenceRanker, inputs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values = np.zeros(len(inputs), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            values[start:start + batch_size] = model(torch.from_numpy(inputs[start:start + batch_size]).to(device)).cpu().numpy()
    return values


def _load_ranker(checkpoint: Path, device: torch.device) -> EvidenceRanker:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = EvidenceRanker(int(payload["input_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels > 0.5
    negatives = ~positives
    if not positives.any() or not negatives.any():
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2.0) / (positives.sum() * negatives.sum()))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels > 0.5
    count = int(positives.sum())
    if count == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    hits = positives[order]
    cumulative = np.cumsum(hits)
    return float(np.sum((cumulative / np.arange(1, len(scores) + 1)) * hits) / count)


def _oracle_and_heuristics(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    labels = np.asarray(cache["labels"], dtype=np.int64)
    candidate_logp = np.asarray(cache["candidate_logp"], dtype=np.float32)
    current_prob = _stable_softmax(np.asarray(cache["current_logp"], dtype=np.float32))
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    score_names = ("Real-MaxProb", "Real-Entropy", "Real-Margin", "Real-S0Agreement", "Real-MeanFusion", "Real-ProductFusion", "Real-GTTrueLogP", "Real-GTMargin")
    score_vectors: dict[str, list[float]] = {name: [] for name in score_names}
    gt_values: list[float] = []
    refs: list[tuple[int, int]] = []
    selected: dict[str, list[int]] = {name: [] for name in score_names}
    predictions: dict[str, list[int]] = {name: [] for name in score_names}
    selected_scores: dict[str, list[float]] = {name: [] for name in score_names}
    for row_index, label in enumerate(labels):
        valid = np.flatnonzero(mask[row_index])
        probabilities = _stable_softmax(candidate_logp[row_index, valid])
        gt_logp = candidate_logp[row_index, valid, label]
        margins = np.asarray([candidate_logp[row_index, candidate, label] - np.max(np.delete(candidate_logp[row_index, candidate], label)) for candidate in valid])
        values = {
            "Real-MaxProb": probabilities.max(axis=1),
            "Real-Entropy": -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1),
            "Real-Margin": np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2],
            "Real-S0Agreement": probabilities @ current_prob[row_index],
            "Real-MeanFusion": -np.sum((0.5 * (probabilities + current_prob[row_index])) * np.log(np.clip(0.5 * (probabilities + current_prob[row_index]), 1e-12, 1.0)), axis=1),
            "Real-ProductFusion": [],
            "Real-GTTrueLogP": gt_logp,
            "Real-GTMargin": margins,
        }
        product = probabilities * current_prob[row_index][None, :]
        product /= np.clip(product.sum(axis=1, keepdims=True), 1e-12, None)
        values["Real-ProductFusion"] = -np.sum(product * np.log(np.clip(product, 1e-12, 1.0)), axis=1)
        for name in score_names:
            scores = np.asarray(values[name], dtype=np.float64)
            # Entropy is represented as negative entropy above; all selectors maximize.
            best_position = max(range(len(valid)), key=lambda position: (float(scores[position]), -int(cache["candidate_ids"][row_index, valid[position]])))
            candidate_index = int(valid[best_position])
            selected[name].append(int(cache["candidate_ids"][row_index, candidate_index]))
            predictions[name].append(int(np.argmax(candidate_logp[row_index, candidate_index])))
            selected_scores[name].append(float(scores[best_position]))
            score_vectors[name].extend(scores.tolist())
        gt_values.extend(gt_logp.tolist())
        refs.extend((row_index, int(candidate)) for candidate in valid)
    result: dict[str, dict[str, Any]] = {}
    for name in score_names:
        metric = _metrics(labels, np.asarray(predictions[name], dtype=np.int64))
        metric.update({
            "selector": name,
            "move_rate": 1.0,
            "stay_rate": 0.0,
            "selected_candidate_correct_rate": float(np.mean(np.asarray(predictions[name]) == labels)),
            "mean_selected_score": float(np.mean(selected_scores[name])),
            "mean_selected_gt_true_logp": float(np.mean([candidate_logp[i, np.flatnonzero(cache["candidate_ids"][i] == selected[name][i])[0], labels[i]] for i in range(len(labels))])),
            "candidate_score_spearman_vs_gt_true_logp": _spearman(score_vectors[name], gt_values),
            "selected_viewpoint_ids": selected[name],
            "selected_correct": [bool(predictions[name][i] == int(labels[i])) for i in range(len(labels))],
        })
        result[name] = metric
    return result, {name: np.asarray(values, dtype=np.float32) for name, values in score_vectors.items()}


def _ranker_result(
    name: str,
    labels: np.ndarray,
    val_evidence: np.ndarray,
    cache: Mapping[str, np.ndarray],
    refs: Sequence[tuple[int, int]],
    predicted_scores: np.ndarray,
    utilities: np.ndarray,
    correctness: np.ndarray,
) -> dict[str, Any]:
    selected_ids: list[int] = []
    selected_predictions: list[int] = []
    selected_values: list[float] = []
    selected_true_logp: list[float] = []
    context_ranks: list[float] = []
    cursor = 0
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    for row_index, label in enumerate(labels):
        valid = np.flatnonzero(mask[row_index])
        count = len(valid)
        scores = predicted_scores[cursor:cursor + count]
        utility = utilities[cursor:cursor + count]
        context_ranks.append(_spearman(scores, utility))
        position = max(range(count), key=lambda item: (float(scores[item]), -int(cache["candidate_ids"][row_index, valid[item]])))
        candidate_index = int(valid[position])
        selected_ids.append(int(cache["candidate_ids"][row_index, candidate_index]))
        selected_predictions.append(int(np.argmax(cache["candidate_logp"][row_index, candidate_index])))
        selected_values.append(float(scores[position]))
        selected_true_logp.append(float(cache["candidate_logp"][row_index, candidate_index, label]))
        cursor += count
    metric = _metrics(labels, np.asarray(selected_predictions, dtype=np.int64))
    metric.update({
        "selector": name,
        "move_rate": 1.0,
        "stay_rate": 0.0,
        "selected_candidate_correct_rate": float(np.mean(np.asarray(selected_predictions) == labels)),
        "mean_selected_score": float(np.mean(selected_values)),
        "mean_selected_gt_true_logp": float(np.mean(selected_true_logp)),
        "candidate_score_spearman_vs_gt_true_logp": _spearman(predicted_scores, np.asarray([cache["candidate_logp"][row, candidate, labels[row]] for row, candidate in refs], dtype=np.float32)),
        "candidate_correctness_auroc": _auroc(correctness, predicted_scores),
        "candidate_correctness_ap": _average_precision(correctness, predicted_scores),
        "context_ranking_spearman": float(np.mean(context_ranks)),
        "selected_viewpoint_ids": selected_ids,
        "selected_correct": [bool(selected_predictions[i] == int(labels[i])) for i in range(len(labels))],
    })
    return metric


def _reference_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
    previous = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").read_text(encoding="utf-8"))
    visual = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_visual_context_batch/result.json").read_text(encoding="utf-8"))
    scene = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/result.json").read_text(encoding="utf-8"))
    moving = {
        "S0-only": previous["metrics_moving"]["S0-only"],
        "FrozenStageCv0": previous["metrics_moving"]["FrozenStageCv0"],
        "Candidate-Conditioned Spatial": visual["metrics_moving"]["candidate_conditioned_spatial"],
        "SceneVisibility": scene["methods"]["SceneVisibility"],
        "AnyCorrect Oracle": previous["metrics_moving"]["AnyCorrect Oracle"],
    }
    full = {name: previous["metrics_full"][name] for name in ("S0-only", "FrozenStageCv0", "AnyCorrect Oracle")}
    return moving, full


def _predict_evidence_train_val(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    target_features: np.ndarray,
    dino_lookup: Mapping[tuple[str, str, str, int], int],
    dino_embeddings: np.ndarray,
    predictor_checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    inputs_raw, _, _, refs = _flatten_inputs(rows, cache, target_features, dino_lookup, dino_embeddings)
    payload = torch.load(predictor_checkpoint, map_location=device, weights_only=False)
    input_mean = np.asarray(payload["input_mean"], dtype=np.float32)
    input_std = np.asarray(payload["input_std"], dtype=np.float32)
    inputs = ((inputs_raw - input_mean) / np.where(input_std < 1e-6, 1.0, input_std)).astype(np.float32)
    model = _load_model(predictor_checkpoint, "feature_logp", inputs.shape[1], device)
    predictions, _ = _predict_candidates(model, inputs, refs, len(rows), int(cache["candidate_logp"].shape[1]), device, batch_size)
    return predictions


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root().resolve()
    device = _require_cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    diagnostics_root = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"
    train_rows = load_jsonl(policy_root / "stage_c/features/train.jsonl")
    val_rows_all = load_jsonl(policy_root / "stage_c/features/val.jsonl")
    stage_d_train_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/train.jsonl")}
    stage_d_val_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/val.jsonl")}
    dino_lookup, dino_embeddings, dino_summary = _load_dino(data_root)
    val_rows, val_indices = _covered_rows(val_rows_all, stage_d_val_ids, dino_lookup)
    train_pred_rows, train_pred_indices = _covered_rows(train_rows, stage_d_train_ids, dino_lookup)
    train_cache = _read_npz(diagnostics_root / "train_candidate_true_logp.npz")
    val_cache_all = _read_npz(diagnostics_root / "val_all_candidate_true_logp.npz")
    train_pred_cache = _subset_cache(train_cache, train_pred_indices)
    val_cache = _subset_cache(val_cache_all, val_indices)
    _candidate_alignment(train_rows, train_cache)
    _candidate_alignment(train_pred_rows, train_pred_cache)
    _candidate_alignment(val_rows, val_cache)
    archive_root = data_root / "datasets/offline/habitat-train/00006-00087"
    stgcn_checkpoint = data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth"
    predictor_checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/future_recognition_evidence_prediction/feature_logp_best.pth"
    target_root = data_root / "diagnostics/reduced12_future_recognition_evidence_prediction"
    train_target = np.asarray(np.load(target_root / "train_target_features.npz", allow_pickle=False)["candidate_feature"])
    val_target = np.asarray(np.load(target_root / "val_moving_target_features.npz", allow_pickle=False)["candidate_feature"])
    output = args.output_dir.resolve() if args.output_dir else EXPERIMENT_ROOT
    branch_root = output / "branch_results"
    checkpoint_root = data_root / "checkpoints/policy_reduced12_eight_placement_v1/recognition_evidence_decision_decomposition"
    output.mkdir(parents=True, exist_ok=True)
    branch_root.mkdir(parents=True, exist_ok=True)

    # Batch A: direct selectors over true archived recognition evidence.
    real_metrics, real_score_vectors = _oracle_and_heuristics(val_rows, val_cache)
    (branch_root / "real_heuristics.json").write_text(json.dumps(real_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "real_heuristic_candidate_scores.npz", **real_score_vectors)

    # Batch C predicted evidence is generated before ranker training and is cached locally.
    train_pred_evidence = _predict_evidence_train_val(train_pred_rows, train_pred_cache, train_target, dino_lookup, dino_embeddings, predictor_checkpoint, device, args.inference_batch_size)
    val_pred_evidence = _predict_evidence_train_val(val_rows, val_cache, val_target, dino_lookup, dino_embeddings, predictor_checkpoint, device, args.inference_batch_size)
    np.savez_compressed(output / "predicted_evidence_cache.npz", train_predicted_logp=train_pred_evidence, val_predicted_logp=val_pred_evidence)

    # Build candidate-level and context-level data for real and predicted evidence.
    train_real_inputs_raw, train_real_correctness, train_real_utility, train_real_refs = _evidence_features(train_rows, train_cache, train_cache["candidate_logp"])
    val_real_inputs_raw, val_real_correctness, val_real_utility, val_real_refs = _evidence_features(val_rows, val_cache, val_cache["candidate_logp"])
    train_pred_inputs_raw, train_pred_correctness, train_pred_utility, train_pred_refs = _evidence_features(train_pred_rows, train_pred_cache, train_pred_evidence)
    val_pred_inputs_raw, val_pred_correctness, val_pred_utility, val_pred_refs = _evidence_features(val_rows, val_cache, val_pred_evidence)
    input_mean = train_real_inputs_raw.mean(axis=0).astype(np.float32)
    input_std = train_real_inputs_raw.std(axis=0).astype(np.float32)
    input_std[input_std < 1e-6] = 1.0
    # Real and predicted rankers use independently standardized features, as their
    # evidence distributions have different scales.
    pred_input_mean = train_pred_inputs_raw.mean(axis=0).astype(np.float32)
    pred_input_std = train_pred_inputs_raw.std(axis=0).astype(np.float32)
    pred_input_std[pred_input_std < 1e-6] = 1.0
    train_real_inputs = (train_real_inputs_raw - input_mean) / input_std
    val_real_inputs = (val_real_inputs_raw - input_mean) / input_std
    train_pred_inputs = (train_pred_inputs_raw - pred_input_mean) / pred_input_std
    val_pred_inputs = (val_pred_inputs_raw - pred_input_mean) / pred_input_std
    max_candidates = int(train_cache["candidate_logp"].shape[1])
    train_real_context = _context_tensors(train_real_inputs, train_real_utility, train_real_refs, len(train_rows), max_candidates)
    val_real_context = _context_tensors(val_real_inputs, val_real_utility, val_real_refs, len(val_rows), max_candidates)
    train_pred_context = _context_tensors(train_pred_inputs, train_pred_utility, train_pred_refs, len(train_pred_rows), max_candidates)
    val_pred_context = _context_tensors(val_pred_inputs, val_pred_utility, val_pred_refs, len(val_rows), max_candidates)

    training: dict[str, Any] = {}
    learned: dict[str, dict[str, Any]] = {}
    branch_specs = (
        ("real", train_real_inputs, train_real_correctness, val_real_inputs, val_real_correctness, train_real_context, val_real_context, train_real_refs, val_real_refs, train_real_utility, val_real_utility, train_cache, val_cache, val_rows),
        ("predicted", train_pred_inputs, train_pred_correctness, val_pred_inputs, val_pred_correctness, train_pred_context, val_pred_context, train_pred_refs, val_pred_refs, train_pred_utility, val_pred_utility, train_pred_cache, val_cache, val_rows),
    )
    for prefix, tr_inputs, tr_correct, va_inputs, va_correct, tr_context, va_context, tr_refs, va_refs, tr_utility, va_utility, _, va_cache, va_rows in branch_specs:
        bce_checkpoint = checkpoint_root / f"{prefix}_correctness_bce_best.pth"
        list_checkpoint = checkpoint_root / f"{prefix}_gt_margin_listwise_best.pth"
        bce_summary_path = output / f"{prefix}_correctness_bce_training.json"
        list_summary_path = output / f"{prefix}_gt_margin_listwise_training.json"
        if bce_checkpoint.exists() and bce_summary_path.exists():
            bce_summary = json.loads(bce_summary_path.read_text(encoding="utf-8"))
            print(f"[{prefix} correctness BCE] skip existing checkpoint", flush=True)
        else:
            bce_summary = _train_bce(tr_inputs, tr_correct, va_inputs, va_correct, bce_checkpoint, bce_summary_path, device, args.batch_size)
        if list_checkpoint.exists() and list_summary_path.exists():
            list_summary = json.loads(list_summary_path.read_text(encoding="utf-8"))
            print(f"[{prefix} GT-margin listwise] skip existing checkpoint", flush=True)
        else:
            list_summary = _train_listwise(*tr_context, *va_context, list_checkpoint, list_summary_path, device)
        training[f"{prefix}_correctness_bce"] = bce_summary
        training[f"{prefix}_gt_margin_listwise"] = list_summary
        bce_model = _load_ranker(bce_checkpoint, device)
        list_model = _load_ranker(list_checkpoint, device)
        bce_scores = _predict_ranker(bce_model, va_inputs, device, args.batch_size)
        list_scores = _predict_ranker(list_model, va_inputs, device, args.batch_size)
        learned[f"{prefix}_correctness_bce"] = _ranker_result(f"{prefix.title()}Evidence-CorrectnessBCE", np.asarray([int(row["label_id"]) for row in va_rows]), va_cache["candidate_logp"], va_cache, va_refs, bce_scores, va_utility, va_correct)
        learned[f"{prefix}_gt_margin_listwise"] = _ranker_result(f"{prefix.title()}Evidence-GTMarginListwise", np.asarray([int(row["label_id"]) for row in va_rows]), va_cache["candidate_logp"], va_cache, va_refs, list_scores, va_utility, va_correct)
        learned[f"{prefix}_correctness_bce"]["score_standardization"] = {"mean": input_mean.tolist() if prefix == "real" else pred_input_mean.tolist(), "std": input_std.tolist() if prefix == "real" else pred_input_std.tolist()}
        learned[f"{prefix}_gt_margin_listwise"]["score_standardization"] = learned[f"{prefix}_correctness_bce"]["score_standardization"]
        np.savez_compressed(branch_root / f"{prefix}_ranker_candidate_scores.npz", correctness_bce=bce_scores, gt_margin_listwise=list_scores)
        (branch_root / ("real_learned.json" if prefix == "real" else "predicted_learned.json")).write_text(json.dumps({k: v for k, v in learned.items() if k.startswith(prefix + "_")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    moving_references, full_references = _reference_metrics()
    frozen_accuracy = float(moving_references["FrozenStageCv0"]["accuracy"])
    frozen_f1 = float(moving_references["FrozenStageCv0"]["macro_f1"])
    candidate_spatial_accuracy = float(moving_references["Candidate-Conditioned Spatial"]["accuracy"])
    candidate_spatial_f1 = float(moving_references["Candidate-Conditioned Spatial"]["macro_f1"])
    metrics_moving: dict[str, Any] = {**moving_references, **real_metrics, **learned}
    for metric in metrics_moving.values():
        metric["delta_vs_frozen_accuracy_pp"] = 100.0 * (float(metric["accuracy"]) - frozen_accuracy)
        metric["delta_vs_frozen_macro_f1_pp"] = 100.0 * (float(metric["macro_f1"]) - frozen_f1)
        metric["delta_vs_candidate_conditioned_spatial_accuracy_pp"] = 100.0 * (float(metric["accuracy"]) - candidate_spatial_accuracy)
        metric["delta_vs_candidate_conditioned_spatial_macro_f1_pp"] = 100.0 * (float(metric["macro_f1"]) - candidate_spatial_f1)
    deployable_names = [name for name in learned if name.startswith("predicted_")]
    best_predicted = max(deployable_names, key=lambda name: float(metrics_moving[name]["accuracy"]))
    best_real_non_gt = max((name for name in real_metrics if name not in {"Real-GTTrueLogP", "Real-GTMargin"}), key=lambda name: float(real_metrics[name]["accuracy"]))
    result = {
        "experiment_id": "REDUCED12_RECOGNITION_EVIDENCE_DECISION_DECOMPOSITION",
        "status": "COMPLETED",
        "labels": list(LABELS),
        "focus_classes": list(FOCUS_CLASSES),
        "population": {"train_contexts": len(train_rows), "predicted_ranker_train_contexts": len(train_pred_rows), "moving_val_contexts": len(val_rows), "train_real_candidate_examples": len(train_real_inputs), "train_predicted_candidate_examples": len(train_pred_inputs), "moving_val_candidate_examples": len(val_real_inputs)},
        "metrics_moving": metrics_moving,
        "metrics_full": {**full_references, "real_evidence_selectors": "NOT_EVALUATED_BY_DESIGN", "predicted_evidence_selectors": "NOT_AVAILABLE_DINO_COVERAGE"},
        "best_real_non_gt_selector": {"name": best_real_non_gt, "accuracy": real_metrics[best_real_non_gt]["accuracy"], "macro_f1": real_metrics[best_real_non_gt]["macro_f1"]},
        "best_predicted_selector": {"name": best_predicted, "accuracy": metrics_moving[best_predicted]["accuracy"], "macro_f1": metrics_moving[best_predicted]["macro_f1"]},
        "gaps": {
            "decision_gap_real_gt_true_logp_minus_best_real_non_gt_accuracy_pp": 100.0 * (real_metrics["Real-GTTrueLogP"]["accuracy"] - real_metrics[best_real_non_gt]["accuracy"]),
            "decision_gap_real_gt_true_logp_minus_best_real_non_gt_macro_f1_pp": 100.0 * (real_metrics["Real-GTTrueLogP"]["macro_f1"] - real_metrics[best_real_non_gt]["macro_f1"]),
            "prediction_gap_best_real_learned_minus_best_predicted_accuracy_pp": 100.0 * (max(metrics_moving["real_correctness_bce"]["accuracy"], metrics_moving["real_gt_margin_listwise"]["accuracy"]) - metrics_moving[best_predicted]["accuracy"]),
            "prediction_gap_best_real_learned_minus_best_predicted_macro_f1_pp": 100.0 * (max(metrics_moving["real_correctness_bce"]["macro_f1"], metrics_moving["real_gt_margin_listwise"]["macro_f1"]) - metrics_moving[best_predicted]["macro_f1"]),
            "remaining_h1_gap_any_correct_minus_best_predicted_accuracy_pp": 100.0 * (moving_references["AnyCorrect Oracle"]["accuracy"] - metrics_moving[best_predicted]["accuracy"]),
            "remaining_h1_gap_any_correct_minus_best_predicted_macro_f1_pp": 100.0 * (moving_references["AnyCorrect Oracle"]["macro_f1"] - metrics_moving[best_predicted]["macro_f1"]),
        },
        "training": training,
        "frozen_stgcn_checkpoint": str(stgcn_checkpoint.resolve()),
        "frozen_stgcn_checkpoint_sha256": file_sha256(stgcn_checkpoint),
        "predicted_evidence_checkpoint": str(predictor_checkpoint.resolve()),
        "predicted_evidence_checkpoint_sha256": file_sha256(predictor_checkpoint),
        "dino_cache": {**dino_summary, "regenerated": False, "future_candidate_dino_used": False},
        "protocol": {"real_candidate_evidence": "frozen ST-GCN on archived candidate skeleton", "predicted_candidate_evidence": "existing Feature+LogP predictor", "terminal_observation": "selected real archived candidate through frozen ST-GCN", "gt_action_used_only_for_train_targets_and_privileged_oracles": True, "future_candidate_skeleton_used_for_target_and_terminal_eval_only": True},
        "test_used": False,
        "train_used_for_ranker_training": True,
        "future_candidate_rgb_used": False,
        "future_candidate_dino_used": False,
        "gt_action_used_only_for_train_targets_and_privileged_oracles": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "diagnostics.json").write_text(json.dumps({"real_heuristics": real_metrics, "real_learned": {k: v for k, v in learned.items() if k.startswith("real_")}, "predicted_learned": {k: v for k, v in learned.items() if k.startswith("predicted_")}, "gaps": result["gaps"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_class = {name: metrics_moving[name]["per_class"] for name in metrics_moving if isinstance(metrics_moving[name], dict) and "per_class" in metrics_moving[name]}
    (output / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Real / Predicted Recognition Evidence Decision Decomposition", "", "Train/Val only; policy Test was not read. Existing archived skeleton, true candidate evidence, visited s0 DINO and Feature+LogP predicted-evidence assets were reused; no RGB/skeleton/DINO regeneration was performed.", "", "## Moving Val", "", "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen |", "|---|---:|---:|---:|---:|"]
    for name, metric in metrics_moving.items():
        lines.append(f"| {name} | {float(metric['accuracy']):.6f} | {float(metric['macro_f1']):.6f} | {float(metric.get('delta_vs_frozen_accuracy_pp', 0.0)):+.3f}pp | {float(metric.get('delta_vs_frozen_macro_f1_pp', 0.0)):+.3f}pp |")
    lines.extend(["", "## Full Val references", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name, metric in full_references.items():
        lines.append(f"| {name} | {float(metric['accuracy']):.6f} | {float(metric['macro_f1']):.6f} |")
    lines.extend(["", "## Evidence/ranker diagnostics", ""])
    for name in real_metrics:
        lines.append(f"- **{name}** candidate-score Spearman vs GT true-logp: {real_metrics[name]['candidate_score_spearman_vs_gt_true_logp']:.6f}.")
    for name in learned:
        metric = learned[name]
        lines.append(f"- **{metric['selector']}** AUROC/AP={metric['candidate_correctness_auroc']:.6f}/{metric['candidate_correctness_ap']:.6f}; context ranking Spearman={metric['context_ranking_spearman']:.6f}; selected correctness={metric['selected_candidate_correct_rate']:.6f}.")
    lines.extend(["", "## Gap decomposition", "", f"- Real GTTrueLogP minus best real non-GT selector: {result['gaps']['decision_gap_real_gt_true_logp_minus_best_real_non_gt_accuracy_pp']:+.3f}pp Accuracy; {result['gaps']['decision_gap_real_gt_true_logp_minus_best_real_non_gt_macro_f1_pp']:+.3f}pp Macro-F1.", f"- Best real learned minus best predicted learned: {result['gaps']['prediction_gap_best_real_learned_minus_best_predicted_accuracy_pp']:+.3f}pp Accuracy; {result['gaps']['prediction_gap_best_real_learned_minus_best_predicted_macro_f1_pp']:+.3f}pp Macro-F1.", f"- AnyCorrect Oracle minus best predicted selector: {result['gaps']['remaining_h1_gap_any_correct_minus_best_predicted_accuracy_pp']:+.3f}pp Accuracy; {result['gaps']['remaining_h1_gap_any_correct_minus_best_predicted_macro_f1_pp']:+.3f}pp Macro-F1.", "", "## Scientific judgment", ""])
    real_non_gt_acc = float(real_metrics[best_real_non_gt]["accuracy"])
    pred_acc = float(metrics_moving[best_predicted]["accuracy"])
    gt_acc = float(real_metrics["Real-GTTrueLogP"]["accuracy"])
    if gt_acc >= 0.70 and real_non_gt_acc < 0.50:
        lines.append("Real future recognition evidence contains substantial recoverable information, but non-GT utility selection remains the dominant decision bottleneck.")
    elif max(metrics_moving["real_correctness_bce"]["accuracy"], metrics_moving["real_gt_margin_listwise"]["accuracy"]) - pred_acc >= 0.05:
        lines.append("A learned decision rule is viable with real evidence, while the large real-to-predicted gap makes future evidence prediction the leading limitation.")
    elif max(metrics_moving["real_correctness_bce"]["accuracy"], metrics_moving["real_gt_margin_listwise"]["accuracy"]) < 0.50:
        lines.append("Even real future posterior evidence gives limited correctness-selection performance; the posterior/evidence representation itself is insufficient for reliable utility decisions.")
    elif pred_acc >= 0.50:
        lines.append("Predicted recognition evidence reaches the requested 50% range, supporting further development of the future-evidence route.")
    else:
        lines.append("Predicted evidence selectors remain below 50%; current s0 information and geometry do not yet predict candidate-specific recognition evidence reliably.")
    lines.extend(["", "No formal WM/JR/ST-GCN checkpoint, taxonomy, or split was changed. No Test data were read.", "", "`test_used=false`; `train_used_for_ranker_training=true`; `future_candidate_skeleton_used_for_target_and_terminal_eval_only=true`; `future_candidate_rgb_used=false`; `future_candidate_dino_used=false`."])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "test_used": result["test_used"], "best_predicted_selector": result["best_predicted_selector"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
