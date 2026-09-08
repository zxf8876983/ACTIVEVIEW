#!/usr/bin/env python3
"""Train and evaluate stay-aware reduced12 H1 objectives on Train/Val only."""

from __future__ import annotations

import argparse
import json
import random
import shutil
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
from activeview.data.generation.utility_labels import file_sha256
from activeview.data.preprocessing.policy_data import RecordBalancedSampler, load_feature_statistics
from activeview.methods.active_view.geometry import order_candidates
from activeview.methods.active_view.utility_predictor import SetUtilityRanker, build_utility_predictor, count_parameters
from activeview.scripts.experiments.run_reduced12_h1_discriminative_batch import build_true_logp_cache


NUM_CLASSES = 12
LABELS = ("walk", "sit", "stand up", "bend", "crawl", "stumble", "clap", "throw", "kick", "knock", "punch", "touching face")
BRANCHES = ("delta_logp", "delta_margin", "stay_multi_positive", "logp_listwise", "margin_listwise", "hybrid")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA")
    return device


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class StayDataset(Dataset[dict[str, Any]]):
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
        target_utility = np.zeros(self.max_candidates, dtype=np.float32)
        target_utility[:count] = np.asarray(row["utility_targets"], dtype=np.float32)
        return {
            "current_feature": torch.from_numpy(current),
            "candidate_geometry": torch.from_numpy(padded_geometry),
            "candidate_mask": torch.from_numpy(np.asarray(self.cache["candidate_mask"][index], dtype=bool)),
            "candidate_ids": torch.from_numpy(np.asarray(self.cache["candidate_ids"][index], dtype=np.int64)),
            "candidate_geodesic": torch.from_numpy(np.asarray(self.cache["candidate_geodesic"][index], dtype=np.float32)),
            "target_utility": torch.from_numpy(target_utility),
            "true_logp": torch.from_numpy(np.asarray(self.cache["candidate_logp"][index], dtype=np.float32)),
            "current_logp": torch.from_numpy(np.asarray(self.cache["current_logp"][index], dtype=np.float32)),
            "label_id": int(row["label_id"]),
            "episode_id": str(row["episode_id"]),
        }


def _collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("current_feature", "candidate_geometry", "candidate_mask", "candidate_ids", "candidate_geodesic", "target_utility", "true_logp", "current_logp")
    return {**{key: torch.stack([item[key] for item in items]) for key in keys}, "label_id": torch.tensor([int(item["label_id"]) for item in items], dtype=torch.long), "episode_id": [str(item["episode_id"]) for item in items]}


class StayAwareSetUtilityRanker(nn.Module):
    """The frozen SetUtilityRanker with one explicit current-STAY score."""

    def __init__(self, current_dim: int, geometry_dim: int) -> None:
        super().__init__()
        self.ranker = SetUtilityRanker(current_dim=current_dim, geometry_dim=geometry_dim)
        self.stay_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, current_feature: torch.Tensor, candidate_geometry: torch.Tensor, candidate_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = self.ranker(current_feature, candidate_geometry, candidate_mask)
        stay = self.stay_head(self.ranker.current_encoder(current_feature)).squeeze(-1)
        return stay, candidate


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _action_targets(batch: Mapping[str, Any], mode: str) -> torch.Tensor:
    current = batch["current_logp"]
    candidate = batch["true_logp"]
    labels = batch["label_id"]
    candidate_values = candidate.gather(2, labels[:, None, None].expand(-1, candidate.size(1), 1)).squeeze(-1)
    current_values = current.gather(1, labels[:, None]).squeeze(-1)
    if mode in ("delta_logp", "logp_listwise"):
        return torch.cat([current_values[:, None] * 0.0, candidate_values - current_values[:, None]], dim=1) if mode == "delta_logp" else torch.cat([current_values[:, None], candidate_values], dim=1)
    other = candidate.clone()
    other.scatter_(2, labels[:, None, None].expand(-1, candidate.size(1), 1), float("-inf"))
    current_margin = current_values - torch.max(torch.cat([current[:, :labels.size(0) * 0].reshape(current.size(0), 0), current], dim=1), dim=1).values if False else current_values - torch.where(torch.arange(NUM_CLASSES, device=current.device)[None, :] == labels[:, None], float("-inf"), current).max(dim=1).values
    candidate_margin = candidate_values - other.max(dim=2).values
    if mode == "delta_margin":
        return torch.cat([current_values[:, None] * 0.0, candidate_margin - current_margin[:, None]], dim=1)
    if mode == "margin_listwise":
        return torch.cat([current_margin[:, None], candidate_margin], dim=1)
    if mode == "hybrid":
        return torch.cat([current_values[:, None] * 0.0, candidate_margin - current_margin[:, None]], dim=1)
    if mode == "stay_multi_positive":
        stay_positive = (current.argmax(dim=1) == labels).to(torch.float32)
        candidate_positive = (candidate.argmax(dim=2) == labels[:, None]).to(torch.float32)
        return torch.cat([stay_positive[:, None], candidate_positive], dim=1)
    raise ValueError(f"Unknown mode {mode}")


def _valid_action_mask(candidate_mask: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.ones((candidate_mask.size(0), 1), dtype=torch.bool, device=candidate_mask.device), candidate_mask], dim=1)


def _listwise_loss(scores: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    masked_target = targets.masked_fill(~valid, float("-inf"))
    target_distribution = torch.softmax(masked_target / tau, dim=1)
    log_prediction = torch.log_softmax(scores.masked_fill(~valid, float("-inf")) / tau, dim=1)
    return -(target_distribution * log_prediction.masked_fill(~valid, 0.0)).sum(dim=1).mean()


def _set_loss(scores: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for index in range(scores.size(0)):
        actions = valid[index]
        positives = actions & (targets[index] > 0.5)
        if not bool(positives.any()):
            fallback = torch.where(actions)[0][torch.argmax(targets[index, actions])]
            positives = torch.zeros_like(actions)
            positives[fallback] = True
        losses.append(torch.logsumexp(scores[index, actions], dim=0) - torch.logsumexp(scores[index, positives], dim=0))
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def _loss(stay: torch.Tensor, candidate: torch.Tensor, batch: Mapping[str, Any], mode: str) -> dict[str, torch.Tensor]:
    scores = torch.cat([stay[:, None], candidate], dim=1)
    valid = _valid_action_mask(batch["candidate_mask"])
    targets = _action_targets(batch, mode)
    if mode == "stay_multi_positive":
        ranking = _set_loss(scores, targets, valid)
        return {"total": ranking, "regression": ranking * 0.0, "ranking": ranking}
    if mode == "hybrid":
        ranking = _set_loss(scores, _action_targets(batch, "stay_multi_positive"), valid)
        delta = _action_targets(batch, "delta_margin")
        regression = nn.functional.smooth_l1_loss(scores[:, 1:][batch["candidate_mask"]], delta[:, 1:][batch["candidate_mask"]])
        return {"total": ranking + 0.25 * regression, "regression": regression, "ranking": ranking}
    regression = nn.functional.smooth_l1_loss(scores[valid], targets[valid])
    ranking = _listwise_loss(scores, targets, valid)
    return {"total": regression + ranking, "regression": regression, "ranking": ranking}


def _make_model(feature_summary: Mapping[str, Any], device: torch.device) -> StayAwareSetUtilityRanker:
    return StayAwareSetUtilityRanker(int(feature_summary["current_feature_dim"]), int(feature_summary["candidate_geometry_dim"])).to(device)


def _save_checkpoint(model: nn.Module, path: Path, mode: str, epoch: int, feature_summary: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_type": "stay_aware_set_ranker", "target_mode": mode, "epoch": epoch, "feature_summary_sha256": file_sha256(feature_summary)}, path)


def train_branch(mode: str, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], train_cache: Mapping[str, np.ndarray], val_cache: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray], feature_summary: Mapping[str, Any], feature_summary_path: Path, output_dir: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    _seed(42)
    train_set = StayDataset(train_rows, train_cache, stats)
    val_set = StayDataset(val_rows, val_cache, stats)
    sampler = RecordBalancedSampler(train_rows, episodes_per_record=16, seed=42)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, collate_fn=_collate, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    model = _make_model(feature_summary, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    checkpoint = output_dir / f"{mode}_best.pth"
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 101):
        sampler.set_epoch(epoch)
        model.train()
        totals = {"total": 0.0, "regression": 0.0, "ranking": 0.0}
        steps = 0
        for batch in train_loader:
            batch = _device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            stay, candidate = model(batch["current_feature"], batch["candidate_geometry"], batch["candidate_mask"])
            losses = _loss(stay, candidate, batch, mode)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            steps += 1
        model.eval()
        validation = 0.0
        val_steps = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = _device_batch(batch, device)
                stay, candidate = model(batch["current_feature"], batch["candidate_geometry"], batch["candidate_mask"])
                validation += float(_loss(stay, candidate, batch, mode)["total"].cpu())
                val_steps += 1
        record = {"epoch": epoch, "train_loss": totals["total"] / max(steps, 1), "train_regression_loss": totals["regression"] / max(steps, 1), "train_ranking_loss": totals["ranking"] / max(steps, 1), "val_loss": validation / max(val_steps, 1)}
        history.append(record)
        print(f"[{mode}] epoch={epoch:03d} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss = record["val_loss"]
            best_epoch = epoch
            stale = 0
            _save_checkpoint(model, checkpoint, mode, epoch, feature_summary_path)
        else:
            stale += 1
        if stale >= 10:
            break
    summary = {"branch": mode, "parameter_count": count_parameters(model), "device": str(device), "max_epochs": 100, "selected_epoch": best_epoch, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint), "checkpoint_selection_metric": "Val action-set objective loss", "history": history, "test_used": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{mode}_training.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _load_stay_checkpoint(path: Path, feature_summary: Mapping[str, Any], device: torch.device) -> StayAwareSetUtilityRanker:
    model = _make_model(feature_summary, device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _load_baseline(path: Path, feature_summary: Mapping[str, Any], device: torch.device) -> SetUtilityRanker:
    model = build_utility_predictor("set_ranker", current_dim=int(feature_summary["current_feature_dim"]), geometry_dim=int(feature_summary["candidate_geometry_dim"])).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _scores(model: nn.Module, dataset: StayDataset, device: torch.device, batch_size: int, baseline: bool = False) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=0)
    stays: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch_device = _device_batch(batch, device)
            if baseline:
                candidate = model(batch_device["current_feature"], batch_device["candidate_geometry"], batch_device["candidate_mask"])
                stay = torch.zeros(candidate.size(0), device=device)
            else:
                stay, candidate = model(batch_device["current_feature"], batch_device["candidate_geometry"], batch_device["candidate_mask"])
            stays.append(stay.cpu().numpy())
            candidates.append(candidate.cpu().numpy())
    return np.concatenate(stays), np.concatenate(candidates)


def _f1(confusion: np.ndarray) -> float:
    values = []
    for index in range(NUM_CLASSES):
        tp = float(confusion[index, index])
        predicted = float(confusion[:, index].sum())
        support = float(confusion[index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def _classification(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    for index, name in enumerate(LABELS):
        support = int(confusion[index].sum())
        tp = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        per_class[name] = {"support": support, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {"n": int(labels.size), "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0, "macro_f1": _f1(confusion), "per_class": per_class, "confusion_matrix": confusion.tolist()}


def evaluate_selector(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], stays: np.ndarray, candidates: np.ndarray, name: str) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    current = np.asarray(cache["current_logp"], dtype=np.float64)
    truth = np.asarray(cache["candidate_logp"], dtype=np.float64)
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    ids = np.asarray(cache["candidate_ids"], dtype=np.int64)
    geodesic = np.asarray(cache["candidate_geodesic"], dtype=np.float64)
    predictions: list[int] = []
    selected_keys: list[int | None] = []
    any_keys: list[int | None] = []
    gt_keys: list[int | None] = []
    selected_scores: list[float] = []
    selected_margins: list[float] = []
    gt_regrets: list[float] = []
    move_count = 0
    for index, label in enumerate(labels):
        valid = np.flatnonzero(mask[index])
        if name == "AnyCorrect Oracle":
            if int(np.argmax(current[index])) == int(label):
                chosen_action = 0
            else:
                correct = [int(candidate) for candidate in valid if int(np.argmax(truth[index, candidate])) == int(label)]
                chosen_action = 1 + int(np.flatnonzero(valid == correct[0])[0]) if correct else 0
        elif name == "GTTrueLogP Oracle":
            oracle_scores = np.concatenate([[float(current[index, label])], truth[index, valid, label]])
            chosen_action = int(np.argmax(oracle_scores))
        else:
            action_values = np.concatenate([[stays[index]], candidates[index, valid]])
            chosen_action = int(np.argmax(action_values))
        if chosen_action == 0:
            chosen_key = None
            predicted = int(np.argmax(current[index]))
            score = float(current[index, label])
            margin = float(current[index, label] - np.max(np.delete(current[index], label)))
        else:
            candidate_index = int(valid[chosen_action - 1])
            chosen_key = int(ids[index, candidate_index])
            predicted = int(np.argmax(truth[index, candidate_index]))
            score = float(truth[index, candidate_index, label])
            margin = float(truth[index, candidate_index, label] - np.max(np.delete(truth[index, candidate_index], label)))
            move_count += 1
        predictions.append(predicted)
        selected_keys.append(chosen_key)
        selected_scores.append(score)
        selected_margins.append(margin)
        current_correct = int(np.argmax(current[index])) == int(label)
        any_correct = [int(ids[index, candidate]) for candidate in valid if int(np.argmax(truth[index, candidate])) == int(label)]
        any_keys.append(None if current_correct else (any_correct[0] if any_correct else None))
        oracle_options = [(float(current[index, label]), None)] + [(float(truth[index, candidate, label]), int(ids[index, candidate])) for candidate in valid]
        oracle_key = max(oracle_options, key=lambda item: (item[0], -(item[1] if item[1] is not None else -1)))[1]
        gt_keys.append(oracle_key)
        gt_regrets.append(max(item[0] for item in oracle_options) - score)
    classification = _classification(labels, np.asarray(predictions, dtype=np.int64))
    s0_correct = np.argmax(current, axis=1) == labels
    selected_correct = np.asarray(predictions) == labels
    return {**classification, "selector": name, "stay_rate": float(1.0 - move_count / len(labels)) if len(labels) else 0.0, "move_rate": float(move_count / len(labels)) if len(labels) else 0.0, "selected_action_positive_rate": float(np.mean(selected_correct)) if len(labels) else 0.0, "s0_error_correction_rate": float(np.sum(~s0_correct & selected_correct) / max(np.sum(~s0_correct), 1)), "s0_correct_harm_rate": float(np.sum(s0_correct & ~selected_correct) / max(np.sum(s0_correct), 1)), "mean_selected_gt_true_logp": float(np.mean(selected_scores)) if selected_scores else 0.0, "mean_selected_gt_margin": float(np.mean(selected_margins)) if selected_margins else 0.0, "overlap_any_correct_oracle": float(np.mean([a == b for a, b in zip(selected_keys, any_keys)])) if labels.size else 0.0, "overlap_gt_true_logp_oracle": float(np.mean([a == b for a, b in zip(selected_keys, gt_keys)])) if labels.size else 0.0, "mean_regret_to_gt_logp_oracle": float(np.mean(gt_regrets)) if gt_regrets else 0.0, "p90_regret_to_gt_logp_oracle": float(np.percentile(gt_regrets, 90)) if gt_regrets else 0.0}


def _evaluate_population(model: nn.Module, rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], stats: Mapping[str, np.ndarray], feature_summary: Mapping[str, Any], device: torch.device, batch_size: int, name: str, baseline: bool = False) -> dict[str, Any]:
    stays, candidates = _scores(model, StayDataset(rows, cache, stats), device, batch_size, baseline=baseline)
    return evaluate_selector(rows, cache, stays, candidates, name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root()
    device = _cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    feature_root = policy_root / "stage_c"
    archive_root = data_root / "datasets/offline/habitat-train/00006-00087"
    stgcn_checkpoint = args.stgcn_checkpoint
    output_root = REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch"
    checkpoint_root = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_stay_aware_objective_batch"
    cache_root = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"
    output_root.mkdir(parents=True, exist_ok=True)
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    summary_path = feature_root / "stage_c_feature_summary.json"
    feature_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_rows = _read_jsonl(feature_root / "features/train.jsonl")
    val_all = _read_jsonl(feature_root / "features/val.jsonl")
    moving_ids = {str(row["episode_id"]) for row in _read_jsonl(policy_root / "stage_d/features/val.jsonl")}
    val_moving = [row for row in val_all if str(row["episode_id"]) in moving_ids]
    if not val_moving:
        raise RuntimeError("Stage-D moving Val is empty")
    train_cache = build_true_logp_cache(train_rows, split="train", archive_root=archive_root, checkpoint=stgcn_checkpoint, cache_root=cache_root, device=device, batch_size=args.inference_batch_size)
    val_cache_all = build_true_logp_cache(val_all, split="val_all", archive_root=archive_root, checkpoint=stgcn_checkpoint, cache_root=cache_root, device=device, batch_size=args.inference_batch_size)
    val_index = {str(row["episode_id"]): index for index, row in enumerate(val_all)}
    indices = np.asarray([val_index[str(row["episode_id"])] for row in val_moving], dtype=np.int64)
    val_cache = {key: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == len(val_all) else value for key, value in val_cache_all.items()}
    print(f"Train rows={len(train_rows)} Val full={len(val_all)} Val moving={len(val_moving)} device={device}", flush=True)

    previous_baseline = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_discriminative_objective_batch/baseline/set_ranker_best.pth"
    if not previous_baseline.exists():
        raise FileNotFoundError(f"Frozen baseline reproduction checkpoint missing: {previous_baseline}")
    baseline_copy = checkpoint_root / "baseline" / "set_ranker_best.pth"
    baseline_copy.parent.mkdir(parents=True, exist_ok=True)
    if not baseline_copy.exists():
        shutil.copy2(previous_baseline, baseline_copy)
    baseline_model = _load_baseline(baseline_copy, feature_summary, device)
    baseline_moving = _evaluate_population(baseline_model, val_moving, val_cache, stats, feature_summary, device, args.batch_size, "Baseline reproduction", baseline=True)
    frozen_reference = 0.454265873015873
    if abs(float(baseline_moving["accuracy"]) - frozen_reference) > 0.005:
        raise RuntimeError(f"Baseline reproduction differs by {baseline_moving['accuracy'] - frozen_reference:.6f}; stopped")
    baseline_summary = {"branch": "baseline", "reused_checkpoint": str(previous_baseline.resolve()), "checkpoint_sha256": file_sha256(previous_baseline), "moving_metrics": baseline_moving, "test_used": False}
    (output_root / "baseline_training.json").write_text(json.dumps(baseline_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_moving: dict[str, Any] = {"S0-only": _evaluate_population(baseline_model, val_moving, val_cache, stats, feature_summary, device, args.batch_size, "S0-only", baseline=True), "FrozenStageCv0": baseline_moving, "Baseline reproduction": baseline_moving}
    # Replace the S0-only row with an explicit zero candidate score.
    s0_dataset = StayDataset(val_moving, val_cache, stats)
    zeros = np.zeros((len(val_moving), s0_dataset.max_candidates), dtype=np.float32)
    metrics_moving["S0-only"] = evaluate_selector(val_moving, val_cache, np.zeros(len(val_moving), dtype=np.float32), zeros, "S0-only")
    training_summaries: dict[str, Any] = {}
    for mode in BRANCHES:
        branch_dir = checkpoint_root / mode
        summary_file = branch_dir / f"{mode}_training.json"
        if summary_file.exists() and (branch_dir / f"{mode}_best.pth").exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        else:
            summary = train_branch(mode, train_rows, val_all, train_cache, val_cache_all, stats, feature_summary, summary_path, branch_dir, device, args.batch_size)
        training_summaries[mode] = summary
        model = _load_stay_checkpoint(Path(summary["checkpoint"]), feature_summary, device)
        metrics_moving[mode] = _evaluate_population(model, val_moving, val_cache, stats, feature_summary, device, args.batch_size, mode)
        (output_root / "branch_results.json").write_text(json.dumps({"completed_branch": mode, "metrics_moving": metrics_moving, "branch_training": training_summaries, "test_used": False}, indent=2, ensure_ascii=False), encoding="utf-8")
    zeros = np.zeros((len(val_moving), s0_dataset.max_candidates), dtype=np.float32)
    metrics_moving["AnyCorrect Oracle"] = evaluate_selector(val_moving, val_cache, zeros[:, 0], zeros, "AnyCorrect Oracle")
    metrics_moving["GTTrueLogP Oracle"] = evaluate_selector(val_moving, val_cache, zeros[:, 0], zeros, "GTTrueLogP Oracle")
    full_metrics: dict[str, Any] = {}
    baseline_full = _evaluate_population(baseline_model, val_all, val_cache_all, stats, feature_summary, device, args.batch_size, "FrozenStageCv0", baseline=True)
    full_metrics["FrozenStageCv0"] = baseline_full
    full_metrics["Baseline reproduction"] = baseline_full
    full_metrics["S0-only"] = evaluate_selector(val_all, val_cache_all, np.zeros(len(val_all), dtype=np.float32), np.zeros((len(val_all), s0_dataset.max_candidates), dtype=np.float32), "S0-only")
    for mode, summary in training_summaries.items():
        full_model = _load_stay_checkpoint(Path(summary["checkpoint"]), feature_summary, device)
        full_metrics[mode] = _evaluate_population(full_model, val_all, val_cache_all, stats, feature_summary, device, args.batch_size, mode)
    full_zeros = np.zeros((len(val_all), s0_dataset.max_candidates), dtype=np.float32)
    full_metrics["AnyCorrect Oracle"] = evaluate_selector(val_all, val_cache_all, full_zeros[:, 0], full_zeros, "AnyCorrect Oracle")
    full_metrics["GTTrueLogP Oracle"] = evaluate_selector(val_all, val_cache_all, full_zeros[:, 0], full_zeros, "GTTrueLogP Oracle")
    result = {"protocol": "reduced12 stay-aware H1 objective batch", "labels": list(LABELS), "train_stage_c_rows": len(train_rows), "val_full_contexts": len(val_all), "val_moving_contexts": len(val_moving), "metrics_moving": metrics_moving, "metrics_full": full_metrics, "oracle_audit": {key: metrics_moving[key] for key in ("S0-only", "AnyCorrect Oracle", "GTTrueLogP Oracle")}, "branch_training": training_summaries, "frozen_stage_c_accuracy_reference": frozen_reference, "baseline_reproduction_delta_accuracy": float(baseline_moving["accuracy"] - frozen_reference), "test_used": False, "test_read": False, "skeleton_regenerated": False, "perception_regenerated": False}
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    per_class = {action: {method: metric["per_class"][action] for method, metric in metrics_moving.items()} for action in LABELS}
    (output_root / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8")
    for mode, summary in training_summaries.items():
        (output_root / f"{mode}_training.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Reduced12 Stay-aware H1 objective batch", "", f"Val Moving contexts: {len(val_moving)}", "", "| H1 | Moving Acc | Moving Macro-F1 | Move rate | ΔAcc vs Frozen |", "|---|---:|---:|---:|---:|"]
    frozen_accuracy = float(metrics_moving["FrozenStageCv0"]["accuracy"])
    for name, metric in metrics_moving.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} | {metric['accuracy'] - frozen_accuracy:+.6f} |")
    best = max((name for name in BRANCHES), key=lambda name: float(metrics_moving[name]["accuracy"]))
    gain = float(metrics_moving[best]["accuracy"] - frozen_accuracy)
    oracle_gap = float(metrics_moving["AnyCorrect Oracle"]["accuracy"] - frozen_accuracy)
    lines.extend(["", "## Required diagnostics", "", "Every branch includes stay/move rate, s0-error correction, s0-correct harm, selected-action positive rate, oracle-action overlap, and GTLogP regret in `result.json`.", "", "## Per-class Recall/F1", "", "| Action | Frozen | DeltaLogP | DeltaMargin | Stay-Multi | LogP-Listwise | Margin-Listwise | Hybrid |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for action in LABELS:
        cells = []
        for method in ("FrozenStageCv0", "delta_logp", "delta_margin", "stay_multi_positive", "logp_listwise", "margin_listwise", "hybrid"):
            item = metrics_moving[method]["per_class"][action]
            cells.append(f"{item['recall']:.4f}/{item['f1']:.4f}")
        lines.append(f"| {action} | " + " | ".join(cells) + " |")
    lines.extend(["", f"Best Stay-aware branch: `{best}` ({metrics_moving[best]['accuracy']:.6f} Acc, {metrics_moving[best]['macro_f1']:.6f} Macro-F1; ΔAcc {gain:+.6f}).", f"AnyCorrect Oracle ceiling: {metrics_moving['AnyCorrect Oracle']['accuracy']:.6f}; remaining gap from Frozen: {oracle_gap:.6f}.", "The baseline reproduction passed the ±0.5pp gate.", "", "Scientific decision: if no Stay-aware branch exceeds FrozenStageCv0 by 2pp, the previous failure is not explained by stay/candidate calibration alone; the remaining bottleneck is observable H1 representation. No new method is started automatically.", "", "No Test files were read and no skeleton/RGB/DINO/perception data were regenerated."])
    (output_root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--stgcn-checkpoint", type=Path, default=data_root / "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.inference_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    result = run(args)
    print(json.dumps({"output": str((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").resolve()), "val_moving": result["val_moving_contexts"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
