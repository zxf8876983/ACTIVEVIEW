#!/usr/bin/env python3
"""EXP049--EXP051 counterfactual view-revision campaign.

This campaign consumes only the frozen EXP046 all-view imagined-recognition
cache.  It never reads Test and never regenerates perception or world-model
predictions.  EXP050 trains a small set-aware revision model on Train; Val is
used once for the registered comparisons and the EXP051 gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import binom
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.active_view.stage_d_dataset import load_jsonl, load_pairwise_geodesic
from activeview.active_view.stage_d_dense_campaign import context_key
from activeview.active_view.stage_d_evaluation import (
    build_baseline_trajectories,
    build_fixed_first_oracle,
    build_stage_d_trajectories,
    summarize_trajectory_rows,
)
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _load_rgb_lookup, _rows
from activeview.scripts.run_stage_d_exp046_048 import _load_cache
from activeview.core.paths import get_data_root

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP049_ROOT = REPO_ROOT / "experiments/stage_d/EXP049_counterfactual_candidate_scaling"
EXP050_ROOT = REPO_ROOT / "experiments/stage_d/EXP050_joint_rollout_revision"
EXP051_ROOT = REPO_ROOT / "experiments/stage_d/EXP051_closed_loop_world_model_revision"
VIEW_COUNT = 32
N_CLASSES = 16
SEED = 42
M_VALUES = (2, 4, 8, 16, "ALL_LEGAL")


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pairwise_and_azimuths(data_root: Path, rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str]) -> tuple[dict[tuple[str, str], dict[int, dict[int, float]]], dict[tuple[str, str], dict[int, float]]]:
    pair_root = data_root / "datasets/policy_v11_5/pairwise_viewpoint_geodesic"
    pairwise: dict[tuple[str, str], dict[int, dict[int, float]]] = {}
    azimuths: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]))
        if key in pairwise:
            continue
        pairwise[key] = load_pairwise_geodesic(pair_root / key[0] / f"{key[1]}.json")
        source = Path(sources[context_key(row)])
        manifest = source.parents[1] / "candidate_metadata" / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        placements = [p for p in payload["placements_data"] if str(p["region"]) == key[1]]
        if len(placements) != 1:
            raise ValueError(f"candidate metadata placement mismatch for {key}")
        azimuths[key] = {int(v["viewpoint_id"]): float(v["azimuth_deg"]) for v in placements[0]["viewpoints"]}
        if len(azimuths[key]) != VIEW_COUNT:
            raise ValueError(f"expected 32 azimuths for {key}")
    return pairwise, azimuths


def _legal_order(row: Mapping[str, Any], pairwise: Mapping[tuple[str, str], Mapping[int, Mapping[int, float]]], azimuths: Mapping[tuple[str, str], Mapping[int, float]], v0: Mapping[str, Mapping[str, Any]]) -> list[int]:
    first = v0.get(str(row["episode_id"]))
    if first is None or bool(first["predicted_stays"]):
        return []
    scene_region = (str(row["scene_id"]), str(row["region"]))
    s0, s1 = int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])
    distances = pairwise[scene_region].get(s1, {})
    current_azimuth = azimuths[scene_region][s1]
    values: list[tuple[float, float, int]] = []
    for candidate in range(VIEW_COUNT):
        if candidate in {s0, s1} or candidate not in distances or not np.isfinite(distances[candidate]):
            continue
        delta = (azimuths[scene_region][candidate] - current_azimuth + 180.0) % 360.0 - 180.0
        values.append((float(distances[candidate]), abs(float(delta)), candidate))
    values.sort()
    return [candidate for _, _, candidate in values]


def _budget(order: Sequence[int], budget: int | str) -> list[int]:
    return list(order if budget == "ALL_LEGAL" else order[: int(budget)])


def _candidate_cache_rows(rows: Sequence[Mapping[str, Any]], orders: Mapping[str, Sequence[int]], budget: int | str, pairwise: Mapping[tuple[str, str], Mapping[int, Mapping[int, float]]], v0: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        value = copy.deepcopy(dict(row))
        candidates = _budget(orders[str(row["episode_id"])], budget)
        s1 = int(row["s1_viewpoint_id"])
        distances = pairwise[(str(row["scene_id"]), str(row["region"]))].get(s1, {})
        value["remaining_candidate_ids"] = candidates
        value["second_step_candidate_geodesic"] = [float(distances[c]) for c in candidates]
        output.append(value)
    return output


def _expanded_stage_b_rows(
    stage_b_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    cache_rows: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Add all legal imagined candidates to evaluator records.

    Stage-B utility records contain only the original two-candidate pool.  The
    EXP049 scaling protocol evaluates additional legal viewpoints from the
    frozen all-view cache, so those candidates need trajectory metadata (the
    frozen oracle fields remain unchanged for comparability).
    """
    cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    cache_by_episode = {str(row["episode_id"]): row for row in cache_rows}
    expanded: list[dict[str, Any]] = []
    for record in stage_b_rows:
        episode_id = str(record["episode_id"])
        cached = cache_by_episode.get(episode_id)
        if cached is None or not orders.get(episode_id):
            expanded.append(copy.deepcopy(dict(record)))
            continue
        index = cache_index[episode_id]
        current_id = int(record["current"]["viewpoint_id"])
        current_logp = cache["current_logp_s1"][index]
        by_id = {int(item["viewpoint_id"]): dict(item) for item in record["candidates"]}
        by_id[current_id] = dict(record["current"])
        candidate_items = [by_id[current_id]]
        # Keep the frozen first-step proposal (p1) so the canonical evaluator
        # can account for its first-leg geodesic cost.
        for item_id, item in by_id.items():
            if item_id != current_id and item_id not in {int(v) for v in cached["remaining_candidate_ids"]}:
                candidate_items.append(item)
        for candidate_id in cached["remaining_candidate_ids"]:
            candidate_id = int(candidate_id)
            if candidate_id in {int(item["viewpoint_id"]) for item in candidate_items}:
                continue
            utility = float(cache["true_logp"][index, candidate_id, int(record["label_id"])] - current_logp[int(record["label_id"])])
            candidate_items.append({
                "viewpoint_id": candidate_id,
                "predicted_label_id": int(np.argmax(cache["imagined_logp"][index, candidate_id])),
                "utility": utility,
                "geodesic_distance_m": float(cached["second_step_candidate_geodesic"][list(cached["remaining_candidate_ids"]).index(candidate_id)]),
            })
        value = copy.deepcopy(dict(record)); value["current"] = candidate_items[0]; value["candidates"] = candidate_items
        expanded.append(value)
    return expanded


def _tie_argmax(values: Sequence[float]) -> int:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array)
    return int(np.flatnonzero(np.isclose(array, maximum, rtol=0.0, atol=1e-12))[0])


def _tie_argmin(values: Sequence[float]) -> int:
    array = np.asarray(values, dtype=np.float64)
    minimum = np.min(array)
    return int(np.flatnonzero(np.isclose(array, minimum, rtol=0.0, atol=1e-12))[0])


def _decision_rows(rows: Sequence[Mapping[str, Any]], v0: Mapping[str, Mapping[str, Any]], selections: Mapping[str, Sequence[int | None]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row, selected in zip(rows, selections["selected"]):
        candidates = list(row["remaining_candidate_ids"])
        if bool(v0[str(row["episode_id"])]["predicted_stays"]):
            decisions.append({"episode_id": str(row["episode_id"]), "remaining_candidate_ids": candidates, "predicted_utilities": [0.0] * len(candidates), "predicted_stays": True, "predicted_candidate_viewpoint_id": None, "max_predicted_utility": 0.0})
            continue
        decisions.append({"episode_id": str(row["episode_id"]), "remaining_candidate_ids": candidates, "predicted_utilities": [0.0] * len(candidates), "predicted_stays": selected is None, "predicted_candidate_viewpoint_id": None if selected is None else int(selected), "max_predicted_utility": 0.0})
    return decisions


def _features(cache: Mapping[str, np.ndarray], index: int, candidates: Sequence[int]) -> np.ndarray:
    base = np.concatenate([cache["current_logp_s0"][index], cache["current_logp_s1"][index]])
    values = [np.concatenate([base, cache["current_logp_s1"][index], np.zeros(9, dtype=np.float32)])]
    values.extend(np.concatenate([base, cache["imagined_logp"][index, c], cache["candidate_descriptor"][index, c]]) for c in candidates)
    return np.asarray(values, dtype=np.float32)


def _score_independent(cache: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], orders: Mapping[str, Sequence[int]], model: nn.Module | None, device: torch.device, method: str, budget: int | str) -> list[int | None]:
    episode_index = {str(e): i for i, e in enumerate(cache["episode_ids"].tolist())}
    selected: list[int | None] = []
    for row in rows:
        if not orders[str(row["episode_id"])]:
            selected.append(None)
            continue
        i = episode_index[str(row["episode_id"])]
        candidates = _budget(orders[str(row["episode_id"])], budget)
        logs = np.vstack([cache["current_logp_s1"][i], cache["imagined_logp"][i, candidates]])
        probs = np.exp(logs)
        if method == "PREDICTED_ENTROPY":
            choice = _tie_argmin(-np.sum(probs * logs, axis=1))
        elif method == "PREDICTED_TOP1_CONFIDENCE":
            choice = _tie_argmax(np.max(probs, axis=1))
        elif method == "PREDICTED_BELIEF_CROSS_ENTROPY":
            belief = probs[0] / max(float(probs[0].sum()), 1e-12)
            choice = _tie_argmin(-np.sum(probs * np.log(np.maximum(belief, 1e-12)), axis=1))
        elif method == "IMAGINED_GT_LABEL_ORACLE":
            label = int(cache["label_id"][i])
            choice = _tie_argmax(logs[:, label])
        elif model is not None:
            features = torch.from_numpy(_features(cache, i, candidates)).to(device)
            with torch.inference_mode():
                values = torch.sigmoid(model(features)).detach().cpu().numpy().reshape(-1)
            choice = _tie_argmax(values)
        else:
            raise ValueError(method)
        selected.append(None if choice == 0 else candidates[choice - 1])
    return selected


class _JointDataset(Dataset[dict[str, Any]]):
    def __init__(self, current: np.ndarray, candidates: np.ndarray, masks: np.ndarray, target: np.ndarray, correctness: np.ndarray, labels: np.ndarray) -> None:
        self.current, self.candidates, self.masks = current, candidates, masks
        self.target, self.correctness, self.labels = target, correctness, labels

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"current": self.current[index], "candidates": self.candidates[index], "mask": self.masks[index], "target": self.target[index], "correctness": self.correctness[index], "label": self.labels[index]}


class _JointRevision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 16 s0 probabilities + 16 s1 probabilities + six scalar statistics.
        self.current_projector = nn.Sequential(nn.Linear(38, 128), nn.GELU())
        self.candidate_projector = nn.Sequential(nn.Linear(26, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.score = nn.Linear(128, 1)
        self.posterior = nn.Linear(128, N_CLASSES)

    def forward(self, current: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_token = self.current_projector(current).unsqueeze(1)
        candidate_tokens = self.candidate_projector(candidates)
        tokens = torch.cat([current_token, candidate_tokens], dim=1)
        full_mask = torch.cat([torch.ones((mask.size(0), 1), dtype=torch.bool, device=mask.device), mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~full_mask)
        return self.score(encoded[:, 1:]).squeeze(-1), self.posterior(encoded[:, 1:])


def _joint_examples(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], budget: int | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = {str(e): i for i, e in enumerate(cache["episode_ids"].tolist())}
    current_values: list[np.ndarray] = []; candidate_values: list[np.ndarray] = []; masks: list[np.ndarray] = []; targets: list[int] = []; correctness: list[np.ndarray] = []; labels: list[int] = []
    max_candidates = int(budget) if budget != "ALL_LEGAL" else 30
    for row in rows:
        i = indices[str(row["episode_id"])]
        candidates = _budget(orders[str(row["episode_id"])], budget)
        if not candidates:
            continue
        probs0, probs1 = np.exp(cache["current_logp_s0"][i]), np.exp(cache["current_logp_s1"][i])
        stats = np.asarray([*probs0, *probs1, -np.sum(probs0 * cache["current_logp_s0"][i]), -np.sum(probs1 * cache["current_logp_s1"][i]), np.max(probs0), np.max(probs1), np.sort(probs1)[-1] - np.sort(probs1)[-2], np.sort(probs0)[-1] - np.sort(probs0)[-2]], dtype=np.float32)
        candidate_rows = [np.concatenate([cache["imagined_logp"][i, c], cache["candidate_descriptor"][i, c], [0.0]]) for c in candidates]
        candidate_rows.insert(0, np.concatenate([cache["current_logp_s1"][i], np.zeros(9, dtype=np.float32), [1.0]]))
        action_correct = [int(np.argmax(cache["current_logp_s1"][i]) == int(cache["label_id"][i]))] + [int(np.argmax(cache["true_logp"][i, c]) == int(cache["label_id"][i])) for c in candidates]
        if any(action_correct):
            target = next(j for j, value in enumerate(action_correct) if value)
        else:
            true_scores = [float(cache["current_logp_s1"][i, int(cache["label_id"][i])])] + [float(cache["true_logp"][i, c, int(cache["label_id"][i])]) for c in candidates]
            target = _tie_argmax(true_scores)
        padded = np.zeros((max_candidates + 1, 26), dtype=np.float32); padded[: len(candidate_rows)] = np.asarray(candidate_rows, dtype=np.float32)
        padded_correctness = np.zeros(max_candidates + 1, dtype=np.float32); padded_correctness[: len(action_correct)] = np.asarray(action_correct, dtype=np.float32)
        current_values.append(stats); candidate_values.append(padded); masks.append(np.asarray([True] * len(candidate_rows) + [False] * (max_candidates - len(candidates)), dtype=bool)); targets.append(target); correctness.append(padded_correctness); labels.append(int(cache["label_id"][i]))
    return np.asarray(current_values), np.asarray(candidate_values), np.asarray(masks), np.asarray(targets), np.asarray(correctness), np.asarray(labels)


def _fit_joint(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], device: torch.device) -> tuple[_JointRevision, dict[str, float]]:
    model = _JointRevision().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); final_loss = 0.0; final_correctness = 0.0
    examples = [_joint_examples(rows, cache, orders, budget) for budget in M_VALUES]
    for epoch in range(20):
        losses: list[float] = []; correctness_losses: list[float] = []
        for current, candidate, mask, target, action_correct, labels in examples:
            if len(target) == 0:
                continue
            loader = DataLoader(_JointDataset(current, candidate, mask, target, action_correct, labels), batch_size=512, shuffle=True)
            model.train()
            for batch in loader:
                current_t = batch["current"].float().to(device); candidate_t = batch["candidates"].float().to(device); mask_t = batch["mask"].bool().to(device)
                scores, posterior = model(current_t, candidate_t, mask_t)
                ce = nn.functional.cross_entropy(scores.masked_fill(~mask_t, -1e9), batch["target"].to(device))
                corr_target = torch.zeros_like(scores); corr_target[:, : batch["correctness"].shape[1]] = batch["correctness"].float().to(device)
                bce = nn.functional.binary_cross_entropy_with_logits(scores, corr_target, reduction="none")
                bce = bce.masked_select(mask_t).mean()
                posterior_target = batch["label"].to(device).unsqueeze(1).expand(-1, posterior.size(1))
                post = nn.functional.cross_entropy(posterior.reshape(-1, N_CLASSES), posterior_target.reshape(-1), reduction="none").reshape_as(scores)
                post = post.masked_select(mask_t).mean()
                loss = ce + 0.25 * bce + 0.05 * post
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach())); correctness_losses.append(float(bce.detach()))
        final_loss = float(np.mean(losses)) if losses else 0.0; final_correctness = float(np.mean(correctness_losses)) if correctness_losses else 0.0
    return model.eval(), {"final_loss": final_loss, "final_correctness_loss": final_correctness, "epochs": 20, "batch_size": 512}


def _joint_select(model: _JointRevision, cache: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], orders: Mapping[str, Sequence[int]], budget: int | str, device: torch.device, posterior_mode: bool = False) -> list[int | None]:
    indices = {str(e): i for i, e in enumerate(cache["episode_ids"].tolist())}; selected: list[int | None] = []
    for row in rows:
        order = orders[str(row["episode_id"])]
        if not order:
            selected.append(None); continue
        i = indices[str(row["episode_id"])]
        candidates = _budget(order, budget); current = np.exp(cache["current_logp_s1"][i]); probs0 = np.exp(cache["current_logp_s0"][i]); stats = np.asarray([*probs0, *current, -np.sum(probs0 * cache["current_logp_s0"][i]), -np.sum(current * cache["current_logp_s1"][i]), np.max(probs0), np.max(current), np.sort(current)[-1] - np.sort(current)[-2], np.sort(probs0)[-1] - np.sort(probs0)[-2]], dtype=np.float32)[None]
        values = [np.concatenate([cache["current_logp_s1"][i], np.zeros(9, dtype=np.float32), [1.0]])]
        values.extend(np.concatenate([cache["imagined_logp"][i, c], cache["candidate_descriptor"][i, c], [0.0]]) for c in candidates)
        candidates_t = torch.zeros((1, max(31, len(candidates) + 1), 26), dtype=torch.float32); candidates_t[0, : len(values)] = torch.from_numpy(np.asarray(values, dtype=np.float32)); mask = torch.zeros((1, candidates_t.shape[1]), dtype=torch.bool); mask[0, : len(values)] = True
        with torch.inference_mode():
            scores, posterior = model(torch.from_numpy(stats).float().to(device), candidates_t.to(device), mask.to(device))
        if posterior_mode:
            p = torch.softmax(posterior[0], dim=-1).cpu().numpy(); choice = _tie_argmin(1.0 - p.max(axis=1)[: len(values)])
        else:
            choice = _tie_argmax(scores[0, : len(values)].cpu().numpy())
        selected.append(None if choice == 0 else candidates[choice - 1])
    return selected


def _paired(candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("paired rows must have identical length")
    left_correct = np.asarray([int(r["predicted_label_id"]) == int(r["label_id"]) for r in candidate], dtype=bool)
    right_correct = np.asarray([int(r["predicted_label_id"]) == int(r["label_id"]) for r in baseline], dtype=bool)
    left_labels = np.asarray([int(r["predicted_label_id"]) for r in candidate], dtype=np.int64)
    right_labels = np.asarray([int(r["predicted_label_id"]) for r in baseline], dtype=np.int64)
    truth = np.asarray([int(r["label_id"]) for r in candidate], dtype=np.int64)
    n01 = int(np.sum(~left_correct & right_correct)); n10 = int(np.sum(left_correct & ~right_correct)); n = n01 + n10
    p = float(1.0 if n == 0 else min(1.0, 2.0 * binom.cdf(min(n01, n10), n, 0.5)))

    def macro_f1(pred: np.ndarray, target: np.ndarray) -> float:
        confusion = np.bincount(target * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
        values = []
        for cls in range(N_CLASSES):
            tp = float(confusion[cls, cls]); precision_den = float(confusion[:, cls].sum()); recall_den = float(confusion[cls].sum())
            precision = tp / precision_den if precision_den else 0.0; recall = tp / recall_den if recall_den else 0.0
            values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return float(np.mean(values))

    delta = left_correct.astype(np.float64) - right_correct.astype(np.float64)
    left_f1, right_f1 = macro_f1(left_labels, truth), macro_f1(right_labels, truth)
    rng = np.random.default_rng(SEED); acc_boot = np.empty(10000, dtype=np.float64); f1_delta_boot = np.empty(10000, dtype=np.float64)
    for start in range(0, 10000, 500):
        count = min(500, 10000 - start); indices = rng.integers(0, len(delta), size=(count, len(delta)))
        acc_boot[start : start + count] = np.mean(delta[indices], axis=1)
        for offset, sample in enumerate(indices):
            f1_delta_boot[start + offset] = macro_f1(left_labels[sample], truth[sample]) - macro_f1(right_labels[sample], truth[sample])
    return {
        "n01": n01, "n10": n10, "mcnemar_p": p,
        "delta_accuracy": float(delta.mean()),
        "delta_accuracy_ci95": [float(np.quantile(acc_boot, .025)), float(np.quantile(acc_boot, .975))],
        "delta_macro_f1": float(left_f1 - right_f1),
        "delta_macro_f1_ci95": [float(np.quantile(f1_delta_boot, .025)), float(np.quantile(f1_delta_boot, .975))],
        "bootstrap_resamples": 10000, "seed": SEED,
    }


def run_campaign(data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed(); train_rows, val_rows = _rows(data_root, "train"), _rows(data_root, "val")
    if (len(train_rows), len(val_rows)) != (29133, 9742):
        raise RuntimeError("canonical Train/Val population mismatch")
    train_sources, val_sources = _episode_sources(data_root, "train"), _episode_sources(data_root, "val")
    train_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz")
    val_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz")
    if str(train_cache["policy_split"][0]).lower() != "train" or str(val_cache["policy_split"][0]).lower() != "val":
        raise RuntimeError("cache split metadata failure")
    stage_b = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl"); v0_rows = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"); v0 = {str(r["episode_id"]): r for r in v0_rows}; summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text()); mapping = json.loads(Path(summary["label_mapping"]).read_text()); categories = [n for n, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    pairwise, azimuths = _load_pairwise_and_azimuths(data_root, [*train_rows, *val_rows], {**train_sources, **val_sources})
    train_v0_rows = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/train_predictions.jsonl")
    train_v0 = {str(x["episode_id"]): x for x in train_v0_rows}
    if len(train_v0) != len(train_v0_rows):
        raise ValueError("duplicate Train frozen-v0 prediction episode_id")
    train_orders = {str(r["episode_id"]): _legal_order(r, pairwise, azimuths, train_v0) for r in train_rows}
    val_orders = {str(r["episode_id"]): _legal_order(r, pairwise, azimuths, v0) for r in val_rows}
    # Frozen CF correctness MLP from EXP047; no retraining for EXP049 scaling.
    from activeview.scripts.run_stage_d_exp046_048 import _BinaryMLP
    mlp = _BinaryMLP(); payload = torch.load(REPO_ROOT / "experiments/stage_d/EXP047_counterfactual_decision_layer/cf_correctness_mlp.pth", map_location="cpu", weights_only=False); mlp.load_state_dict(payload["state_dict"]); mlp = mlp.to(device).eval()
    logistic = nn.Linear(57, 1)
    logistic_payload = torch.load(REPO_ROOT / "experiments/stage_d/EXP047_counterfactual_decision_layer/cf_correctness_logistic.pth", map_location="cpu", weights_only=False)
    logistic.load_state_dict(logistic_payload["state_dict"]); logistic = logistic.to(device).eval()
    methods = ("PREDICTED_ENTROPY", "PREDICTED_TOP1_CONFIDENCE", "PREDICTED_BELIEF_CROSS_ENTROPY", "CF_CORRECTNESS_LOGISTIC", "CF_CORRECTNESS_MLP", "IMAGINED_GT_LABEL_ORACLE")
    scaling: dict[str, Any] = {}; val_method_rows: dict[str, list[dict[str, Any]]] = {}
    for budget in M_VALUES:
        key = str(budget); scaling[key] = {}
        cache_rows = _candidate_cache_rows(val_rows, val_orders, budget, pairwise, v0)
        trajectory_records = _expanded_stage_b_rows(stage_b, val_cache, cache_rows, val_orders)
        for method in methods:
            selected = _score_independent(val_cache, val_rows, val_orders, logistic if method == "CF_CORRECTNESS_LOGISTIC" else (mlp if method == "CF_CORRECTNESS_MLP" else None), device, method, budget)
            decisions = _decision_rows(cache_rows, v0, {"selected": selected}); trajectory_rows = build_stage_d_trajectories(trajectory_records, v0_rows, cache_rows, decisions); trajectory = summarize_trajectory_rows(trajectory_rows, categories); scaling[key][method] = trajectory
            if budget == "ALL_LEGAL":
                val_method_rows[method] = trajectory_rows
    # EXP050 joint revision trained only on Train and evaluated at each prefix.
    joint, joint_train = _fit_joint(train_rows, train_cache, train_orders, device)
    joint_scaling: dict[str, Any] = {}; joint_decisions: dict[str, list[dict[str, Any]]] = {}
    for budget in M_VALUES:
        key = str(budget); cache_rows = _candidate_cache_rows(val_rows, val_orders, budget, pairwise, v0); trajectory_records = _expanded_stage_b_rows(stage_b, val_cache, cache_rows, val_orders); selected = _joint_select(joint, val_cache, val_rows, val_orders, budget, device); decisions = _decision_rows(cache_rows, v0, {"selected": selected}); rows_joint = build_stage_d_trajectories(trajectory_records, v0_rows, cache_rows, decisions); joint_decisions[key] = rows_joint; joint_scaling[key] = summarize_trajectory_rows(rows_joint, categories)
    exp014_second = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/runtime/val_second_step_predictions.jsonl")
    exp014_rows = build_stage_d_trajectories(stage_b, v0_rows, val_rows, exp014_second)
    oracle_rows = build_fixed_first_oracle(stage_b, v0_rows, val_rows)
    all_joint = joint_decisions["ALL_LEGAL"]; all_cf = val_method_rows["CF_CORRECTNESS_MLP"]; paired = _paired(all_joint, all_cf)
    EXP049_ROOT.mkdir(parents=True, exist_ok=True); EXP050_ROOT.mkdir(parents=True, exist_ok=True); EXP051_ROOT.mkdir(parents=True, exist_ok=True)
    scaling_payload = {"experiment_id": "EXP049", "status": "COMPLETED", "budgets": list(M_VALUES), "methods": methods, "val_episode_count": len(val_rows), "results": scaling, "test_used": False, "wm_e_frozen": True, "stgcn_frozen": True}
    scaling_payload["provenance"] = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "wm_e_checkpoint_sha256": "db2573a013ed9a7fab87561ad26800334556894b96e69dd3d498464794d9b5e6",
        "train_cache_sha256": _sha256(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz"),
        "val_cache_sha256": _sha256(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz"),
        "train_context_count": len(train_rows), "val_context_count": len(val_rows),
    }
    (EXP049_ROOT / "result.json").write_text(json.dumps(scaling_payload, indent=2), encoding="utf-8")
    (EXP049_ROOT / "candidate_graph_audit.json").write_text(json.dumps({"train_contexts": len(train_rows), "val_contexts": len(val_rows), "train_legal_mean": float(np.mean([len(v) for v in train_orders.values()])), "val_legal_mean": float(np.mean([len(v) for v in val_orders.values()])), "proposal_order": "geodesic, absolute radial azimuth, viewpoint_id", "visited_excluded": True, "test_used": False}, indent=2), encoding="utf-8")
    exp050_payload = {"experiment_id": "EXP050", "status": "COMPLETED", "training": joint_train, "results": joint_scaling, "all_legal_vs_independent_cf": paired, "test_used": False, "wm_e_frozen": True, "stgcn_frozen": True, "fallback_privileged_train_target": True}
    exp050_payload["provenance"] = scaling_payload["provenance"]
    (EXP050_ROOT / "result.json").write_text(json.dumps(exp050_payload, indent=2), encoding="utf-8")
    (EXP050_ROOT / "paired_statistics.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    best_joint_acc = joint_scaling["ALL_LEGAL"]["recognition"]["accuracy"]; best_joint_f1 = joint_scaling["ALL_LEGAL"]["recognition"]["macro_f1"]; cf_acc = scaling["ALL_LEGAL"]["CF_CORRECTNESS_MLP"]["recognition"]["accuracy"]
    monotonic = all(scaling[str(M_VALUES[i + 1])]["CF_CORRECTNESS_MLP"]["recognition"]["accuracy"] >= scaling[str(M_VALUES[i])]["CF_CORRECTNESS_MLP"]["recognition"]["accuracy"] - 1e-12 for i in range(3))
    joint_gate = best_joint_acc >= cf_acc + 0.005 and best_joint_f1 >= scaling["ALL_LEGAL"]["CF_CORRECTNESS_MLP"]["recognition"]["macro_f1"] - 0.002
    gate = {"exp049_monotonic_three_steps": monotonic, "exp050_joint_threshold": joint_gate, "exp051_authorized": bool(monotonic or joint_gate)}
    exp051_payload: dict[str, Any] = {
        "experiment_id": "EXP051",
        "status": "BLOCKED_MISSING_RECURRENT_RGB_HISTORY_ARTIFACT",
        "gate": gate,
        "test_used": False,
        "training_performed": False,
        "rolling_two_view_history": True,
        "reason": "Frozen WM-E requires RGB history for newly visited views, but the approved EXP046 cache contains no RGB embeddings for p2/p3; no substitute inputs were used.",
    }
    (EXP051_ROOT / "result.json").write_text(json.dumps(exp051_payload, indent=2), encoding="utf-8")
    exp014_summary = summarize_trajectory_rows(exp014_rows, categories)
    oracle_summary = summarize_trajectory_rows(oracle_rows, categories)
    return {"experiment_id": "EXP049-051", "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "val_contexts": len(val_rows)}, "exp049": scaling_payload, "exp050": exp050_payload, "exp051": exp051_payload, "exp014": exp014_summary, "fixed_first_oracle": oracle_summary, "test_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP049-EXP051 Train/Val-only campaign")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    print(json.dumps(run_campaign(args.data_root.resolve(), torch.device(args.device)), indent=2))


if __name__ == "__main__":
    main()
