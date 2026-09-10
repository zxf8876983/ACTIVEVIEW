#!/usr/bin/env python3
"""Train/Val-only action-agnostic transition-utility diagnostics (Tasks 3--5)."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    candidate_logp,
    classification,
    correlation,
    gt_margin,
    load_train_val,
)

SEED = 42
EPOCHS = 20
BATCH_SIZE = 512
HIDDEN = 256
MAX_CANDIDATES = 2
OUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/action_agnostic_transition_utility"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _raw_arrays(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], *, include_h1: bool, stay_aware: bool) -> dict[str, np.ndarray]:
    n = len(rows)
    context_dim = 271 * (2 if include_h1 else 1)
    extra_dim = 4 if include_h1 else 1
    input_dim = context_dim + 11 + extra_dim
    slots = MAX_CANDIDATES + (1 if stay_aware else 0)
    features = np.zeros((n, slots, input_dim), dtype=np.float32)
    margins = np.zeros((n, slots), dtype=np.float32)
    correctness = np.zeros((n, slots), dtype=np.float32)
    mask = np.zeros((n, slots), dtype=bool)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    candidate_ids = np.full((n, MAX_CANDIDATES), -1, dtype=np.int64)
    for row_index, row in enumerate(rows):
        s0 = np.asarray(row["s0_feature"], dtype=np.float32)
        h1 = np.asarray(row["s1_feature"], dtype=np.float32)
        if s0.size != 271 or h1.size != 271:
            raise ValueError("unexpected Stage-D feature width")
        h1_id = int(row["s1_viewpoint_id"])
        h1_context = np.asarray([float(h1_id // 8) / 3.0, np.sin((h1_id % 8) * np.pi / 4.0), np.cos((h1_id % 8) * np.pi / 4.0), float(row["first_step_predicted_utility"])], dtype=np.float32)
        base = np.concatenate([s0, h1] if include_h1 else [s0])
        if stay_aware:
            stay_extra = np.zeros(extra_dim, dtype=np.float32)
            if include_h1:
                stay_extra[:3] = h1_context[:3]
            features[row_index, 0] = np.concatenate([base, np.zeros(11, dtype=np.float32), stay_extra])
            h1_logp = h1[256:268]
            margins[row_index, 0] = gt_margin(h1_logp, labels[row_index])
            correctness[row_index, 0] = float(np.argmax(h1_logp) == labels[row_index])
            mask[row_index, 0] = True
        candidate_geometries = row["second_step_candidate_geometry"]
        ids = [int(value) for value in row["remaining_candidate_ids"]]
        if len(ids) > MAX_CANDIDATES:
            raise ValueError("unexpected candidate count")
        for candidate_slot, (viewpoint_id, geometry) in enumerate(zip(ids, candidate_geometries)):
            slot = candidate_slot + (1 if stay_aware else 0)
            extra = h1_context if include_h1 else np.asarray([float(row["first_step_predicted_utility"])], dtype=np.float32)
            features[row_index, slot] = np.concatenate([base, np.asarray(geometry, dtype=np.float32), extra])
            logp = candidate_logp(cache, int(row["cache_index"]), viewpoint_id)
            margins[row_index, slot] = gt_margin(logp, labels[row_index])
            correctness[row_index, slot] = float(np.argmax(logp) == labels[row_index])
            mask[row_index, slot] = True
            candidate_ids[row_index, candidate_slot] = viewpoint_id
    return {"features": features, "margins": margins, "correctness": correctness, "mask": mask, "labels": labels, "candidate_ids": candidate_ids, "input_dim": input_dim}


def _standardize(train: dict[str, np.ndarray], *others: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], ...]:
    values = train["features"][train["mask"]]
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    output = []
    for item in (train, *others):
        copied = dict(item)
        copied["features"] = ((item["features"] - mean) / std).astype(np.float32)
        output.append(copied)
    return tuple(output)


class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, HIDDEN), nn.GELU(), nn.Dropout(0.1), nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class SetTransitionRanker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.score = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(features)
        valid = mask.unsqueeze(-1).to(embedded.dtype)
        count = valid.sum(dim=1).clamp_min(1.0)
        mean = (embedded * valid).sum(dim=1) / count
        maximum = embedded.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        pooled = torch.cat([mean, maximum], dim=1)
        pooled = pooled.unsqueeze(1).expand(-1, embedded.size(1), -1)
        return self.score(torch.cat([embedded, pooled], dim=2)).squeeze(-1)


def _loss(scores: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "bce":
        return nn.functional.binary_cross_entropy_with_logits(scores[mask], targets[mask])
    target = targets.masked_fill(~mask, float("-inf"))
    target_distribution = torch.softmax(target / 0.5, dim=1)
    log_prediction = torch.log_softmax(scores.masked_fill(~mask, float("-inf")) / 0.5, dim=1)
    return -(target_distribution * log_prediction.masked_fill(~mask, 0.0)).sum(dim=1).mean()


def _split_indices(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    held_out = np.asarray([int(hashlib.sha1(str(row["record_id"]).encode()).hexdigest()[:8], 16) % 5 == 0 for row in rows], dtype=bool)
    return np.flatnonzero(~held_out), np.flatnonzero(held_out)


def _train_model(arrays: dict[str, np.ndarray], mode: str, device: torch.device, checkpoint: Path) -> dict[str, Any]:
    _seed()
    train_index, holdout_index = _split_indices(arrays["rows"])
    train = {key: value[train_index] if isinstance(value, np.ndarray) and value.shape[0] == len(arrays["rows"]) else value for key, value in arrays.items() if key != "rows"}
    holdout = {key: value[holdout_index] if isinstance(value, np.ndarray) and value.shape[0] == len(arrays["rows"]) else value for key, value in arrays.items() if key != "rows"}
    raw_values = train["features"][train["mask"]]
    normalization_mean = raw_values.mean(axis=0).astype(np.float32)
    normalization_std = raw_values.std(axis=0).astype(np.float32)
    normalization_std[normalization_std < 1e-6] = 1.0
    train, holdout = _standardize(train, holdout)
    model = CandidateMLP(int(train["input_dim"])) .to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(train["features"]), torch.from_numpy(train["margins"] if mode == "listwise" else train["correctness"]), torch.from_numpy(train["mask"])), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    holdout_features = torch.from_numpy(holdout["features"]).to(device)
    holdout_targets = torch.from_numpy(holdout["margins"] if mode == "listwise" else holdout["correctness"]).to(device)
    holdout_mask = torch.from_numpy(holdout["mask"]).to(device)
    best = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for features, targets, mask in loader:
            scores = model(features.to(device))
            loss = _loss(scores, targets.to(device), mask.to(device), mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(_loss(model(holdout_features), holdout_targets, holdout_mask, mode).cpu())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "internal_holdout_loss": val_loss}
        history.append(record)
        print(f"[{mode}] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} holdout={val_loss:.6f}", flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train["input_dim"]), "mean": normalization_mean, "std": normalization_std, "mode": mode, "seed": SEED}, checkpoint)
    return {"mode": mode, "epochs": EPOCHS, "selected_epoch": best_epoch, "internal_holdout_loss": best, "checkpoint": str(checkpoint.resolve()), "history": history, "train_contexts": int(train_index.size), "internal_holdout_contexts": int(holdout_index.size), "test_used": False}


def _load_model(summary: Mapping[str, Any], device: torch.device) -> CandidateMLP:
    payload = torch.load(summary["checkpoint"], map_location=device, weights_only=False)
    model = CandidateMLP(int(payload["input_dim"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _predict(model: CandidateMLP, arrays: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    features = torch.from_numpy(arrays["features"]).to(device)
    # The checkpoint stores normalization statistics over flattened features.
    payload = torch.load(model._checkpoint_path, map_location=device, weights_only=False) if hasattr(model, "_checkpoint_path") else None
    if payload is not None:
        features = (features - torch.from_numpy(payload["mean"]).to(device)) / torch.from_numpy(payload["std"]).to(device).clamp_min(1e-6)
    with torch.inference_mode():
        return model(features).cpu().numpy()


def _attach_norm(model: CandidateMLP, summary: Mapping[str, Any], device: torch.device) -> CandidateMLP:
    setattr(model, "_checkpoint_path", summary["checkpoint"])
    return model


def _metric_from_scores(rows: Sequence[Mapping[str, Any]], arrays: dict[str, np.ndarray], scores: np.ndarray, *, stay_aware: bool, name: str) -> tuple[dict[str, Any], dict[str, float]]:
    cache = _CACHE
    labels: list[int] = []
    predictions: list[int] = []
    selected_margins: list[float] = []
    moves = 0
    ranks: list[float] = []
    top_coverage = {1: [], 3: [], 5: []}
    for i, row in enumerate(rows):
        label = int(row["label_id"])
        labels.append(label)
        valid = np.flatnonzero(arrays["mask"][i])
        order = valid[np.argsort(-scores[i, valid], kind="mergesort")]
        if stay_aware:
            selected_slot = int(order[0])
            if selected_slot == 0:
                logp = np.asarray(row["s1_feature"][256:268], dtype=np.float32)
                selected_margins.append(gt_margin(logp, label))
            else:
                candidate = int(arrays["candidate_ids"][i, selected_slot - 1])
                logp = candidate_logp(cache, int(row["cache_index"]), candidate)
                selected_margins.append(gt_margin(logp, label))
                moves += 1
        else:
            selected_slot = int(order[0])
            candidate = int(arrays["candidate_ids"][i, selected_slot])
            logp = candidate_logp(cache, int(row["cache_index"]), candidate)
            selected_margins.append(gt_margin(logp, label))
            moves += 1
        predictions.append(int(np.argmax(logp)))
        candidate_slots = [slot for slot in valid if (not stay_aware or slot != 0)]
        candidate_scores = [scores[i, slot] for slot in candidate_slots]
        candidate_targets = [arrays["margins"][i, slot] for slot in candidate_slots]
        if len(candidate_slots) >= 2:
            ranks.append(correlation(candidate_scores, candidate_targets, spearman=True))
        correct = [bool(arrays["correctness"][i, slot] > 0.5) for slot in candidate_slots]
        order_candidates = np.argsort(-np.asarray(candidate_scores))
        for k in (1, 3, 5):
            top_coverage[k].append(bool(np.any(np.asarray(correct)[order_candidates[:k]])) if correct else False)
    result = classification(labels, predictions)
    result.update({"selector": name, "move_rate": float(moves / len(rows)), "stay_rate": float(1.0 - moves / len(rows)), "mean_selected_gt_margin": float(np.mean(selected_margins)), "mean_within_context_margin_spearman": float(np.mean(ranks)) if ranks else 0.0})
    diagnostics = {f"top{k}_correct_coverage": float(np.mean(values)) for k, values in top_coverage.items()}
    diagnostics["ranking_spearman_median"] = float(np.median(ranks)) if ranks else 0.0
    diagnostics["ranking_fraction_positive"] = float(np.mean(np.asarray(ranks) > 0.0)) if ranks else 0.0
    diagnostics["ranking_fraction_above_0_3"] = float(np.mean(np.asarray(ranks) > 0.3)) if ranks else 0.0
    return result, diagnostics


def _baseline_h1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["label_id"]) for row in rows]
    predictions = [int(np.argmax(np.asarray(row["s1_feature"][256:268]))) for row in rows]
    result = classification(labels, predictions)
    result.update({"selector": "H1 baseline", "move_rate": 1.0, "stay_rate": 0.0})
    return result


def _oracle(rows: Sequence[Mapping[str, Any]], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    moves = 0
    for i, row in enumerate(rows):
        label = int(row["label_id"])
        options = [(gt_margin(np.asarray(row["s1_feature"][256:268]), label), np.asarray(row["s1_feature"][256:268]))]
        for slot in np.flatnonzero(arrays["mask"][i]):
            candidate = int(arrays["candidate_ids"][i, slot])
            if candidate < 0:
                continue
            logp = candidate_logp(_CACHE, int(row["cache_index"]), candidate)
            options.append((gt_margin(logp, label), logp))
        selected = max(options, key=lambda item: item[0])
        labels.append(label)
        predictions.append(int(np.argmax(selected[1])))
        moves += int(selected[1] is not options[0][1])
    result = classification(labels, predictions)
    result.update({"selector": "Full GT-margin Oracle", "move_rate": float(moves / len(rows)), "stay_rate": float(1.0 - moves / len(rows))})
    return result


def _make_set_model(arrays: dict[str, np.ndarray], device: torch.device) -> tuple[SetTransitionRanker, dict[str, Any]]:
    train_index, holdout_index = _split_indices(arrays["rows"])
    train = {key: value[train_index] if isinstance(value, np.ndarray) and value.shape[0] == len(arrays["rows"]) else value for key, value in arrays.items() if key != "rows"}
    holdout = {key: value[holdout_index] if isinstance(value, np.ndarray) and value.shape[0] == len(arrays["rows"]) else value for key, value in arrays.items() if key != "rows"}
    raw_values = train["features"][train["mask"]]
    normalization_mean = raw_values.mean(axis=0).astype(np.float32)
    normalization_std = raw_values.std(axis=0).astype(np.float32)
    normalization_std[normalization_std < 1e-6] = 1.0
    train, holdout = _standardize(train, holdout)
    model = SetTransitionRanker(int(train["input_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(train["features"]), torch.from_numpy(train["margins"]), torch.from_numpy(train["mask"])), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    holdout_x = torch.from_numpy(holdout["features"]).to(device)
    holdout_y = torch.from_numpy(holdout["margins"]).to(device)
    holdout_m = torch.from_numpy(holdout["mask"]).to(device)
    best, selected = float("inf"), 0
    history = []
    checkpoint = data_root_path() / "checkpoints/reduced12_eight_placement_v1/action_agnostic_transition_utility/set_ranker_best.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for x, y, m in loader:
            score = model(x.to(device), m.to(device)); loss = _loss(score, y.to(device), m.to(device), "listwise")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val = float(_loss(model(holdout_x, holdout_m), holdout_y, holdout_m, "listwise").cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "internal_holdout_loss": val})
        print(f"[set-ranker] epoch={epoch:02d}/{EPOCHS} train_loss={history[-1]['train_loss']:.6f} holdout={val:.6f}", flush=True)
        if val < best:
            best, selected = val, epoch
            torch.save({"state_dict": model.state_dict(), "input_dim": int(train["input_dim"]), "mean": normalization_mean, "std": normalization_std, "epoch": epoch}, checkpoint)
    return model, {"selected_epoch": selected, "internal_holdout_loss": best, "checkpoint": str(checkpoint.resolve()), "history": history, "test_used": False}


def data_root_path() -> Path:
    return Path(__import__("activeview.core.paths", fromlist=["get_data_root"]).get_data_root())


_CACHE: Mapping[str, np.ndarray]


def run(device_name: str = "cuda:0") -> dict[str, Any]:
    global _CACHE
    data = load_train_val()
    device = _device(device_name)
    train_rows = data["stage_d_rows"]["train"]
    val_rows = data["stage_d_rows"]["val"]
    _CACHE = data["caches"]["val"]
    train_cache = data["caches"]["train"]
    val_cache = data["caches"]["val"]
    branches: dict[str, tuple[bool, bool, str]] = {
        "BCE-H2": (True, False, "bce"), "BCE-StayAware": (True, True, "bce"),
        "Listwise-H2": (True, False, "listwise"), "Listwise-StayAware": (True, True, "listwise"),
        "Listwise-S0-only": (False, False, "listwise"),
    }
    summaries: dict[str, Any] = {}
    predictions: dict[str, tuple[np.ndarray, bool, dict[str, np.ndarray]]] = {}
    for name, (include_h1, stay_aware, mode) in branches.items():
        train_arrays = _raw_arrays(train_rows, train_cache, include_h1=include_h1, stay_aware=stay_aware); train_arrays["rows"] = train_rows
        val_arrays = _raw_arrays(val_rows, val_cache, include_h1=include_h1, stay_aware=stay_aware); val_arrays["rows"] = val_rows
        checkpoint = data_root_path() / "checkpoints/reduced12_eight_placement_v1/action_agnostic_transition_utility" / (name.lower().replace("-", "_") + ".pth")
        summary_path = OUT / (name.lower().replace("-", "_") + "_training.json")
        if checkpoint.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = _train_model(train_arrays, mode, device, checkpoint)
            summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        model = _attach_norm(_load_model(summary, device), summary, device)
        # Standardize validation with the training distribution recorded in checkpoint.
        payload = torch.load(summary["checkpoint"], map_location=device, weights_only=False)
        val_arrays["features"] = ((val_arrays["features"] - payload["mean"]) / np.maximum(payload["std"], 1e-6)).astype(np.float32)
        with torch.inference_mode():
            score = model(torch.from_numpy(val_arrays["features"]).to(device)).cpu().numpy()
        summaries[name] = summary
        predictions[name] = (score, stay_aware, val_arrays)
    metrics: dict[str, Any] = {"H1 baseline": _baseline_h1(val_rows)}
    diagnostics: dict[str, Any] = {}
    for name, (score, stay_aware, arrays) in predictions.items():
        metrics[name], diagnostics[name] = _metric_from_scores(val_rows, arrays, score, stay_aware=stay_aware, name=name)
    # Full GT-margin oracle uses the candidate-only arrays (same legal set).
    oracle_arrays = _raw_arrays(val_rows, val_cache, include_h1=True, stay_aware=False)
    metrics["Full GT-margin Oracle"] = _oracle(val_rows, oracle_arrays)
    best_name = max((name for name in predictions), key=lambda name: metrics[name]["accuracy"])
    real_h1_gain = float(metrics["Listwise-H2"]["accuracy"] - metrics["Listwise-S0-only"]["accuracy"])
    condition = float(metrics[best_name]["accuracy"]) >= 0.50 or real_h1_gain >= 0.02
    set_summary = None
    if condition:
        set_train = _raw_arrays(train_rows, train_cache, include_h1=True, stay_aware=False); set_train["rows"] = train_rows
        set_val = _raw_arrays(val_rows, val_cache, include_h1=True, stay_aware=False); set_val["rows"] = val_rows
        set_model, set_summary = _make_set_model(set_train, device)
        payload = torch.load(set_summary["checkpoint"], map_location=device, weights_only=False)
        set_val["features"] = ((set_val["features"] - payload["mean"]) / np.maximum(payload["std"], 1e-6)).astype(np.float32)
        with torch.inference_mode():
            set_scores = set_model(torch.from_numpy(set_val["features"]).to(device), torch.from_numpy(set_val["mask"]).to(device)).cpu().numpy()
        metrics["Tiny Set-Level Transition Ranker"], diagnostics["Tiny Set-Level Transition Ranker"] = _metric_from_scores(val_rows, set_val, set_scores, stay_aware=False, name="Tiny Set-Level Transition Ranker")
    else:
        set_summary = {"skipped_due_to_insufficient_transition_learnability": True, "condition_best_accuracy_ge_0_50": float(metrics[best_name]["accuracy"]), "condition_real_h1_gain": real_h1_gain}
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": "REDUCED12_ACTION_AGNOSTIC_TRANSITION_UTILITY",
        "population": {"train_contexts": len(train_rows), "val_moving_contexts": len(val_rows)},
        "metrics_moving": metrics,
        "selector_diagnostics": diagnostics,
        "training": summaries,
        "set_ranker": set_summary,
        "h1_observation_ablation": {"Listwise-S0-only": metrics["Listwise-S0-only"], "Listwise-H2-Real-H1": metrics["Listwise-H2"], "ranking_diagnostics": {"S0-only": diagnostics["Listwise-S0-only"], "Real-H1": diagnostics["Listwise-H2"]}},
        "labels": list(LABELS),
        "policy_test_used": False,
        "policy_train_used_for_model_training_only": True,
        "policy_val_used_for_evaluation_only": True,
        "future_candidate_observation_used_at_inference": False,
        "future_candidate_skeleton_used_as_train_target_or_terminal_eval_only": True,
        "gt_action_predicted_at_inference": False,
        "gt_action_used_for_supervision_or_posthoc_only": True,
        "scene_visibility_used_as_posthoc_only": True,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "frozen_stgcn_modified": False,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "selector_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "h1_ablation.json").write_text(json.dumps(result["h1_observation_ablation"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Reduced12 action-agnostic transition utility", "", f"Train contexts: {len(train_rows)}; Moving Val contexts: {len(val_rows)}", "", "| Selector | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|"]
    for name, item in metrics.items():
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item.get('move_rate', 0.0):.6f} |")
    lines.extend(["", f"Real-H1 vs S0-only Listwise accuracy gain: {real_h1_gain * 100.0:+.3f} pp.", f"Best independent scorer: {best_name} ({metrics[best_name]['accuracy']:.6f}); set-level condition: {condition}.", "A true H1 observation is used as an allowed sequential input; future candidates are targets/terminal evaluation only.", "", "policy_test_used=false; no RGB/skeleton/DINO regeneration; frozen ST-GCN unchanged."])
    if isinstance(set_summary, Mapping) and set_summary.get("skipped_due_to_insufficient_transition_learnability"):
        lines.append("Tiny set-level ranker skipped because the predefined learnability gate was not met.")
    elif set_summary is not None:
        lines.append(f"Tiny set-level ranker evaluated at {metrics['Tiny Set-Level Transition Ranker']['accuracy']:.6f} Accuracy.")
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.device), ensure_ascii=False))
