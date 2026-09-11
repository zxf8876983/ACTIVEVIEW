#!/usr/bin/env python3
"""Train and evaluate a minimal direct candidate-correctness H1 ranker.

The selector is trained on Train contexts with a stay-inclusive multi-positive
listwise objective.  Candidate correctness targets come from the frozen
reduced12 recognizer and are never consumed by the deployable scorer.
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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.generation.utility_labels import file_sha256
from activeview.data.preprocessing.policy_data import load_feature_statistics
from activeview.scripts.experiments.run_reduced12_h1_discriminative_batch import build_true_logp_cache
from activeview.scripts.experiments.run_reduced12_h1_stay_aware_batch import (
    StayDataset,
    _load_baseline,
    _load_stay_checkpoint,
    _scores as _stay_scores,
)


NUM_CLASSES = 12
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA")
    return device


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CorrectnessDataset(Dataset[dict[str, Any]]):
    """Stage-C rows plus frozen-recognizer correctness targets."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray]) -> None:
        self.rows = list(rows)
        self.cache = cache
        self.current_mean = stats["current_mean"]
        self.current_std = stats["current_std"]
        self.geometry_mean = stats["geometry_mean"]
        self.geometry_std = stats["geometry_std"]
        self.max_candidates = int(cache["candidate_logp"].shape[1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        current = (np.asarray(row["current_feature"], dtype=np.float32) - self.current_mean) / self.current_std
        geometry = (np.asarray(row["candidate_geometry"], dtype=np.float32) - self.geometry_mean) / self.geometry_std
        count = len(row["candidate_viewpoint_ids"])
        padded_geometry = np.zeros((self.max_candidates, geometry.shape[1]), dtype=np.float32)
        padded_geometry[:count] = geometry
        mask = np.asarray(self.cache["candidate_mask"][index], dtype=bool)
        candidate_logp = np.asarray(self.cache["candidate_logp"][index], dtype=np.float32)
        label = int(row["label_id"])
        candidate_prediction = np.argmax(candidate_logp, axis=1)
        candidate_positive = (candidate_prediction == label) & mask
        current_logp = np.asarray(self.cache["current_logp"][index], dtype=np.float32)
        stay_positive = bool(int(np.argmax(current_logp)) == label)
        other = candidate_logp.copy()
        other[:, label] = -np.inf
        candidate_margin = candidate_logp[:, label] - np.max(other, axis=1)
        return {
            "current_feature": torch.from_numpy(current),
            "candidate_geometry": torch.from_numpy(padded_geometry),
            "candidate_mask": torch.from_numpy(mask),
            "candidate_ids": torch.from_numpy(np.asarray(self.cache["candidate_ids"][index], dtype=np.int64)),
            "current_viewpoint_id": torch.tensor(int(row["current_viewpoint_id"]), dtype=torch.float32),
            "target": torch.from_numpy(np.concatenate(([float(stay_positive)], candidate_positive.astype(np.float32)))),
            "fallback_margin": torch.from_numpy(candidate_margin.astype(np.float32)),
            "label_id": label,
            "episode_id": str(row["episode_id"]),
        }


def _collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("current_feature", "candidate_geometry", "candidate_mask", "candidate_ids", "current_viewpoint_id", "target", "fallback_margin")
    return {
        **{key: torch.stack([item[key] for item in items]) for key in keys},
        "label_id": torch.tensor([int(item["label_id"]) for item in items], dtype=torch.long),
        "episode_id": [str(item["episode_id"]) for item in items],
    }


class CandidateCorrectnessRanker(nn.Module):
    """A per-action MLP scorer with no candidate future evidence input."""

    def __init__(self, current_dim: int, geometry_dim: int) -> None:
        super().__init__()
        self.current_dim = int(current_dim)
        self.geometry_dim = int(geometry_dim)
        self.input_dim = self.current_dim + self.geometry_dim + 3
        self.network = nn.Sequential(nn.Linear(self.input_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, current: torch.Tensor, geometry: torch.Tensor, candidate_ids: torch.Tensor, current_viewpoint_id: torch.Tensor) -> torch.Tensor:
        batch, candidates, _ = geometry.shape
        zero_geometry = torch.zeros((batch, 1, self.geometry_dim), dtype=geometry.dtype, device=geometry.device)
        all_geometry = torch.cat((zero_geometry, geometry), dim=1)
        stay_flag = torch.cat((torch.ones((batch, 1), device=geometry.device), torch.zeros((batch, candidates), device=geometry.device)), dim=1)
        candidate_view = torch.cat((torch.zeros((batch, 1), device=geometry.device), candidate_ids.float() / 31.0), dim=1)
        current_view = (current_viewpoint_id / 31.0).unsqueeze(1).expand(-1, candidates + 1)
        current_tokens = current.unsqueeze(1).expand(-1, candidates + 1, -1)
        features = torch.cat((current_tokens, all_geometry, current_view.unsqueeze(-1), candidate_view.unsqueeze(-1), stay_flag.unsqueeze(-1)), dim=-1)
        return self.network(features).squeeze(-1)


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _valid_mask(candidate_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.ones((candidate_mask.size(0), 1), dtype=torch.bool, device=candidate_mask.device), candidate_mask), dim=1)


def _set_loss(scores: torch.Tensor, targets: torch.Tensor, candidate_mask: torch.Tensor, fallback_margin: torch.Tensor) -> tuple[torch.Tensor, int]:
    valid = _valid_mask(candidate_mask)
    losses: list[torch.Tensor] = []
    fallback_count = 0
    for index in range(scores.size(0)):
        actions = valid[index]
        positives = actions & (targets[index] > 0.5)
        if not bool(positives.any()):
            fallback_count += 1
            candidate_valid = candidate_mask[index]
            positives = torch.zeros_like(actions)
            if bool(candidate_valid.any()):
                best = torch.where(candidate_valid)[0][torch.argmax(fallback_margin[index, candidate_valid])]
                positives[best + 1] = True
            else:
                positives[0] = True
        losses.append(torch.logsumexp(scores[index, actions], dim=0) - torch.logsumexp(scores[index, positives], dim=0))
    if not losses:
        return scores.sum() * 0.0, fallback_count
    return torch.stack(losses).mean(), fallback_count


def _forward_model(model: CandidateCorrectnessRanker, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    return model(batch["current_feature"], batch["candidate_geometry"], batch["candidate_ids"], batch["current_viewpoint_id"])


def _save_checkpoint(model: CandidateCorrectnessRanker, path: Path, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_type": "direct_candidate_correctness_mlp", "epoch": epoch, "current_dim": model.current_dim, "geometry_dim": model.geometry_dim}, path)


def _load_model(path: Path, device: torch.device) -> CandidateCorrectnessRanker:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = CandidateCorrectnessRanker(int(payload["current_dim"]), int(payload["geometry_dim"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _train(
    train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], train_cache: Mapping[str, np.ndarray], val_cache: Mapping[str, np.ndarray],
    stats: Mapping[str, np.ndarray], checkpoint: Path, device: torch.device, batch_size: int, max_epochs: int,
) -> dict[str, Any]:
    _seed(42)
    train_set = CorrectnessDataset(train_rows, train_cache, stats)
    val_set = CorrectnessDataset(val_rows, val_cache, stats)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0, pin_memory=True)
    model = CandidateCorrectnessRanker(train_set[0]["current_feature"].numel(), train_set[0]["candidate_geometry"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_fallback = 0
        train_steps = 0
        for raw_batch in train_loader:
            batch = _device_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            scores = _forward_model(model, batch, device)
            loss, fallback = _set_loss(scores, batch["target"], batch["candidate_mask"], batch["fallback_margin"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach().cpu())
            train_fallback += fallback
            train_steps += 1
        model.eval()
        val_total = 0.0
        val_fallback = 0
        val_steps = 0
        with torch.inference_mode():
            for raw_batch in val_loader:
                batch = _device_batch(raw_batch, device)
                scores = _forward_model(model, batch, device)
                loss, fallback = _set_loss(scores, batch["target"], batch["candidate_mask"], batch["fallback_margin"])
                val_total += float(loss.cpu())
                val_fallback += fallback
                val_steps += 1
        record = {
            "epoch": epoch,
            "train_loss": train_total / max(train_steps, 1),
            "val_loss": val_total / max(val_steps, 1),
            "train_fallback_batches": train_fallback,
            "val_fallback_batches": val_fallback,
        }
        history.append(record)
        print(f"[direct-correctness] epoch={epoch:03d} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss = record["val_loss"]
            best_epoch = epoch
            _save_checkpoint(model, checkpoint, epoch)
    return {
        "branch": "direct_correctness",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "seed": 42,
        "max_epochs": max_epochs,
        "selected_epoch": best_epoch,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_selection_metric": "Val multi-positive action-set loss",
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "elapsed_seconds": time.time() - started,
        "history": history,
        "test_used": False,
    }


def _classification(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
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
    return {"n": int(labels.size), "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0, "macro_f1": float(np.mean(f1_values)), "per_class": per_class, "confusion_matrix": confusion.tolist()}


def _direct_metrics(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], scores: np.ndarray, name: str, seed: int = 42) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    current = np.asarray(cache["current_logp"], dtype=np.float64)
    truth = np.asarray(cache["candidate_logp"], dtype=np.float64)
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    ids = np.asarray(cache["candidate_ids"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    predictions: list[int] = []
    selected_ids: list[int | None] = []
    move_count = 0
    selected_candidate_correct = 0
    candidate_any = 0
    candidate_hit_given_any = 0
    s0_correct = np.argmax(current, axis=1) == labels
    for index, label in enumerate(labels):
        valid = np.flatnonzero(mask[index])
        candidate_correct = np.argmax(truth[index, valid], axis=1) == label
        candidate_any += int(candidate_correct.any())
        if name == "S0-only":
            chosen = 0
        elif name == "Random candidate":
            chosen = 1 + int(rng.choice(valid))
        elif name == "AnyCorrect Oracle":
            correct = valid[candidate_correct]
            chosen = 0 if s0_correct[index] else (1 + int(np.flatnonzero(valid == correct[0])[0]) if correct.size else 0)
        elif name == "BestSingle Oracle":
            chosen = int(np.argmax(np.concatenate(([current[index, label]], truth[index, valid, label]))))
        else:
            action_scores = np.concatenate(([scores[index, 0]], scores[index, valid + 1]))
            chosen = int(np.argmax(action_scores))
        if chosen == 0:
            prediction = int(np.argmax(current[index]))
            selected_ids.append(None)
        else:
            candidate_index = int(valid[chosen - 1])
            prediction = int(np.argmax(truth[index, candidate_index]))
            selected_ids.append(int(ids[index, candidate_index]))
            move_count += 1
            selected_candidate_correct += int(prediction == label)
            if candidate_any and prediction == label:
                candidate_hit_given_any += 1
        predictions.append(prediction)
    selected_correct = np.asarray(predictions, dtype=np.int64) == labels
    result = _classification(labels, np.asarray(predictions, dtype=np.int64))
    no_candidate = len(labels) - candidate_any
    result.update({
        "selector": name,
        "move_rate": float(move_count / len(labels)) if labels.size else 0.0,
        "stay_rate": float(1.0 - move_count / len(labels)) if labels.size else 0.0,
        "selected_candidate_correct_rate": float(selected_candidate_correct / move_count) if move_count else 0.0,
        "correct_candidate_contexts": int(candidate_any),
        "no_correct_candidate_contexts": int(no_candidate),
        "selected_correct_candidate_rate_given_any": float(candidate_hit_given_any / candidate_any) if candidate_any else 0.0,
        "s0_error_correction_rate": float(np.sum(~s0_correct & selected_correct) / max(np.sum(~s0_correct), 1)),
        "s0_correct_harm_rate": float(np.sum(s0_correct & ~selected_correct) / max(np.sum(s0_correct), 1)),
        "test_used": False,
    })
    return result


def _subset_cache(cache: Mapping[str, np.ndarray], indices: np.ndarray, total: int) -> dict[str, np.ndarray]:
    return {key: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == total else value for key, value in cache.items()}


def _predict(model: CandidateCorrectnessRanker, rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray], device: torch.device, batch_size: int) -> np.ndarray:
    dataset = CorrectnessDataset(rows, cache, stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0, pin_memory=True)
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _device_batch(raw_batch, device)
            output.append(_forward_model(model, batch, device).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, dataset.max_candidates + 1), dtype=np.float32)


def _evaluate_existing(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray], feature_summary: Mapping[str, Any], checkpoint: Path, device: torch.device, batch_size: int, stay_aware: bool, name: str) -> dict[str, Any]:
    dataset = StayDataset(rows, cache, stats)
    if stay_aware:
        model = _load_stay_checkpoint(checkpoint, feature_summary, device)
    else:
        model = _load_baseline(checkpoint, feature_summary, device)
    stays, candidates = _stay_scores(model, dataset, device, batch_size, baseline=not stay_aware)
    return _direct_metrics(rows, cache, np.concatenate((stays[:, None], candidates), axis=1), name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root()
    device = _require_cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    feature_root = policy_root / "stage_c"
    archive_root = data_root / "datasets/offline/habitat-train/00006-00087"
    runtime_cache = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"
    output_root = REPO_ROOT / "experiments/reduced12_eight_placement_v1/direct_candidate_correctness_v1"
    checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/direct_candidate_correctness_v1/direct_correctness_best.pth"
    output_root.mkdir(parents=True, exist_ok=True)
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    feature_summary_path = feature_root / "stage_c_feature_summary.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    train_rows = _read_jsonl(feature_root / "features/train.jsonl")
    val_all = _read_jsonl(feature_root / "features/val.jsonl")
    moving_ids = {str(row["episode_id"]) for row in _read_jsonl(policy_root / "stage_d/features/val.jsonl")}
    val_moving = [row for row in val_all if str(row["episode_id"]) in moving_ids]
    if not val_moving:
        raise RuntimeError("Stage-D moving Val is empty")
    print(f"Train rows={len(train_rows)} Val full={len(val_all)} Val moving={len(val_moving)} device={device}", flush=True)
    train_cache = build_true_logp_cache(train_rows, split="train", archive_root=archive_root, checkpoint=args.stgcn_checkpoint, cache_root=runtime_cache, device=device, batch_size=args.inference_batch_size)
    val_cache_all = build_true_logp_cache(val_all, split="val_all", archive_root=archive_root, checkpoint=args.stgcn_checkpoint, cache_root=runtime_cache, device=device, batch_size=args.inference_batch_size)
    if not np.array_equal(train_cache["labels"], np.asarray([int(row["label_id"]) for row in train_rows])) or not np.array_equal(val_cache_all["labels"], np.asarray([int(row["label_id"]) for row in val_all])):
        raise RuntimeError("Cache labels do not match Stage-C rows")
    val_indices = np.asarray([index for index, row in enumerate(val_all) if str(row["episode_id"]) in moving_ids], dtype=np.int64)
    val_cache = _subset_cache(val_cache_all, val_indices, len(val_all))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    training_summary_path = output_root / "training_summary.json"
    if args.reuse_checkpoint and checkpoint.exists() and training_summary_path.exists():
        training = json.loads(training_summary_path.read_text(encoding="utf-8"))
        print(f"Reusing direct-correctness checkpoint: {checkpoint}", flush=True)
    else:
        training = _train(train_rows, val_all, train_cache, val_cache_all, stats, checkpoint, device, args.batch_size, args.max_epochs)
        training_summary_path.write_text(json.dumps(training, indent=2, ensure_ascii=False), encoding="utf-8")
    model = _load_model(checkpoint, device)
    direct_full_scores = _predict(model, val_all, val_cache_all, stats, device, args.inference_batch_size)
    direct_moving_scores = direct_full_scores[val_indices]

    baseline_checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_discriminative_objective_batch/baseline/set_ranker_best.pth"
    old_checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_stay_aware_objective_batch/margin_listwise/margin_listwise_best.pth"
    if not baseline_checkpoint.exists() or not old_checkpoint.exists():
        raise FileNotFoundError("Required FrozenStageCv0 or old best one-shot selector checkpoint is missing")
    metrics_moving: dict[str, Any] = {
        "S0-only": _direct_metrics(val_moving, val_cache, np.zeros((len(val_moving), val_cache["candidate_logp"].shape[1] + 1), dtype=np.float32), "S0-only"),
        "FrozenStageCv0": _evaluate_existing(val_moving, val_cache, stats, feature_summary, baseline_checkpoint, device, args.inference_batch_size, False, "FrozenStageCv0"),
        "Random candidate": _direct_metrics(val_moving, val_cache, np.zeros((len(val_moving), val_cache["candidate_logp"].shape[1] + 1), dtype=np.float32), "Random candidate"),
        "OldBestOneShot": _evaluate_existing(val_moving, val_cache, stats, feature_summary, old_checkpoint, device, args.inference_batch_size, True, "OldBestOneShot"),
        "Direct-Correctness": _direct_metrics(val_moving, val_cache, direct_moving_scores, "Direct-Correctness"),
        "AnyCorrect Oracle": _direct_metrics(val_moving, val_cache, np.zeros_like(direct_moving_scores), "AnyCorrect Oracle"),
        "BestSingle Oracle": _direct_metrics(val_moving, val_cache, np.zeros_like(direct_moving_scores), "BestSingle Oracle"),
    }
    metrics_full: dict[str, Any] = {
        "S0-only": _direct_metrics(val_all, val_cache_all, np.zeros((len(val_all), val_cache_all["candidate_logp"].shape[1] + 1), dtype=np.float32), "S0-only"),
        "FrozenStageCv0": _evaluate_existing(val_all, val_cache_all, stats, feature_summary, baseline_checkpoint, device, args.inference_batch_size, False, "FrozenStageCv0"),
        "Random candidate": _direct_metrics(val_all, val_cache_all, np.zeros((len(val_all), val_cache_all["candidate_logp"].shape[1] + 1), dtype=np.float32), "Random candidate"),
        "OldBestOneShot": _evaluate_existing(val_all, val_cache_all, stats, feature_summary, old_checkpoint, device, args.inference_batch_size, True, "OldBestOneShot"),
        "Direct-Correctness": _direct_metrics(val_all, val_cache_all, direct_full_scores, "Direct-Correctness"),
        "AnyCorrect Oracle": _direct_metrics(val_all, val_cache_all, np.zeros_like(direct_full_scores), "AnyCorrect Oracle"),
        "BestSingle Oracle": _direct_metrics(val_all, val_cache_all, np.zeros_like(direct_full_scores), "BestSingle Oracle"),
    }
    leakage = {
        "inference_inputs": ["current_feature", "current_soft_posterior_in_current_feature", "current_viewpoint_id", "candidate_geometry", "candidate_viewpoint_id", "stay_flag"],
        "forbidden_inputs": ["gt_label", "candidate_true_correctness", "candidate_future_skeleton", "candidate_future_rgb", "candidate_future_feature", "candidate_future_logp", "hard_predicted_action"],
        "forbidden_inputs_present": False,
        "target_only_fields": ["candidate_true_correctness", "candidate_true_logp", "gt_label"],
        "test_used": False,
    }
    result = {
        "protocol": "reduced12 direct candidate correctness ranking",
        "labels": list(LABELS),
        "train_stage_c_rows": len(train_rows),
        "val_stage_c_rows": len(val_all),
        "val_moving_contexts": len(val_moving),
        "input_schema": {"current_feature_dim": 271, "candidate_geometry_dim": 11, "current_viewpoint_scalars": 1, "candidate_viewpoint_scalar": 1, "stay_flag": 1, "total_dim": model.input_dim, "model": "Linear(input_dim,256)->GELU->Linear(256,1)"},
        "target": {"definition": "argmax(frozen candidate logp)==ground-truth label", "positive_candidates_are_all_correct": True, "loss": "stay-inclusive multi-positive set/listwise loss", "no_positive_fallback": "candidate with maximum true-class margin", "train_no_positive_candidate_contexts": int(np.sum(~np.any(np.argmax(train_cache["candidate_logp"], axis=2) == np.asarray([int(row["label_id"]) for row in train_rows])[:, None], axis=1))), "val_no_positive_candidate_contexts": int(np.sum(~np.any(np.argmax(val_cache_all["candidate_logp"], axis=2) == np.asarray([int(row["label_id"]) for row in val_all])[:, None], axis=1)))},
        "metrics_moving": metrics_moving,
        "metrics_full": metrics_full,
        "training": training,
        "frozen_stgcn_checkpoint": str(args.stgcn_checkpoint.resolve()),
        "frozen_stgcn_checkpoint_sha256": file_sha256(args.stgcn_checkpoint),
        "old_best_selector_checkpoint": str(old_checkpoint.resolve()),
        "leakage_audit": leakage,
        "test_used": False,
        "test_read": False,
        "new_perception_generated": False,
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    per_class = {action: {method: metric["per_class"][action] for method, metric in metrics_moving.items()} for action in LABELS}
    (output_root / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8")
    frozen = metrics_moving["FrozenStageCv0"]
    direct = metrics_moving["Direct-Correctness"]
    direct_acc = float(direct["accuracy"])
    if direct_acc <= 0.49:
        decision = "FAILURE: Accuracy <= 49%; stop this direction."
    elif direct_acc <= 0.52:
        decision = "WEAK SIGNAL: Accuracy is in 50-52%; do not complexify."
    elif direct_acc < 0.55:
        decision = "PROMISING: Accuracy exceeds 52%; further work requires explicit approval."
    else:
        decision = "STRONG CANDIDATE: Accuracy reaches at least 55%."
    lines = [
        "# Direct Candidate Correctness Ranking", "", f"Moving Val contexts: {len(val_moving)}", "", "## Protocol", "", "The scorer uses only current Stage-C features/posterior, current viewpoint id, candidate geometry/id and an explicit stay flag. Candidate frozen-recognizer outputs and labels are Train supervision only.", "", "## Moving Val", "", "| Selector | Accuracy | Macro-F1 | Correct-candidate hit | Any-correct contexts | No-correct contexts | Move rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in metrics_moving.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('selected_candidate_correct_rate', 0.0):.6f} | {metric.get('correct_candidate_contexts', 0)} | {metric.get('no_correct_candidate_contexts', 0)} | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend(["", f"Direct-Correctness Δ vs FrozenStageCv0: {direct_acc - float(frozen['accuracy']):+.6f} accuracy points; Δ Macro-F1: {float(direct['macro_f1']) - float(frozen['macro_f1']):+.6f}.", f"{decision}", "", "## Leakage audit", "", "No GT label, future candidate observation/recognizer output, candidate correctness, or hard predicted action is used by the inference scorer. Test was not read.", "", "## Scientific decision", "", "The direct-correctness target is task-aligned and preserves all correct candidates as positives. The result is a single no-tuning experiment; no second version or additional method is started automatically."])
    best_record = min(training["history"], key=lambda item: float(item["val_loss"]))
    lines.extend([
        "",
        f"Train convergence: selected epoch {training['selected_epoch']} with best Val loss {float(best_record['val_loss']):.6f}; corresponding Train loss {float(best_record['train_loss']):.6f}.",
        f"No-correct-candidate fallback was used for {result['target']['train_no_positive_candidate_contexts']} Train and {result['target']['val_no_positive_candidate_contexts']} Val contexts; fallback supervision used the maximum true-class margin candidate.",
    ])
    (output_root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"seed": 42, "batch_size": args.batch_size, "max_epochs": args.max_epochs, "device": str(device), "test_used": False}, indent=2), encoding="utf-8")
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    parser.add_argument("--stgcn-checkpoint", type=Path, default=data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.inference_batch_size <= 0 or args.max_epochs <= 0:
        raise ValueError("batch sizes and max epochs must be positive")
    run(args)


if __name__ == "__main__":
    main()
