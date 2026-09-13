#!/usr/bin/env python3
"""Train/Val-only audit of the value of actually acquired observations.

The sweep intentionally does not predict unobserved candidate utility.  It
reuses the frozen reduced12 ST-GCN feature cache and the frozen shared head,
then evaluates random and privileged combinations of observations.  The only
trained component is a small Train-only current-observation verifier used to
measure stop/continue predictability.  Policy Test is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import time
from collections import Counter, defaultdict
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
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    head_logits,
    load_rows,
    row_signature,
)

NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
MAX_OPTIONS = 22
SEED = 42
TRAIN_BATCH = 1024
VERIFIER_EPOCHS = 12
VERIFIER_LR = 1e-3
WEIGHT_DECAY = 1e-4
OBSERVATION_BUDGETS = (1, 2, 3, 4)
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "overnight_actual_observation_sweep"
)
RUNTIME_RELATIVE = Path("diagnostics/reduced12_dual_route_overnight")
HEAD_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "view_agnostic_frozen_encoder_head/shared_head_best.pth"
)
VERIFIER_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "overnight_actual_observation_sweep"
)
VISIBILITY_RELATIVE = Path("diagnostics/frame0_visibility_predictor_v1/val.npz")
STGCN_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate_cache_metadata(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    all32: bool,
    stgcn_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Require cache provenance to match the rows and frozen ST-GCN asset."""
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing cache provenance: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "rows": len(rows),
        "signature": row_signature(rows),
        "all32": bool(all32),
        "checkpoint_sha256": stgcn_checkpoint_sha256,
        "test_used": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{path.name} provenance mismatch for {key}: "
                f"expected={value!r} actual={metadata.get(key)!r}"
            )
    return metadata


def log_softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array, axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)


def entropy(logp: np.ndarray) -> np.ndarray:
    probabilities = np.exp(np.asarray(logp, dtype=np.float64))
    return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)), axis=-1)


def margin(logp: np.ndarray) -> np.ndarray:
    values = np.asarray(logp, dtype=np.float64)
    order = np.sort(values, axis=-1)
    return order[..., -1] - order[..., -2]


def validate_options(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], name: str) -> None:
    ids = np.asarray(cache["ids"], dtype=np.int64)
    mask = np.asarray(cache["mask"], dtype=bool)
    if ids.shape != (len(rows), MAX_OPTIONS) or mask.shape != ids.shape:
        raise ValueError(f"{name} cache shape mismatch: {ids.shape}/{mask.shape}")
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index]).tolist()
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if ids[index, valid].tolist() != expected:
            raise ValueError(f"{name} action-set mismatch at {row['episode_id']}")


def recognizer_arrays(
    cache: Mapping[str, np.ndarray], head: SharedHead, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(cache["features"], dtype=np.float32)
    raw = head_logits(head, features, device)
    return features, log_softmax(raw)


def selected_slot(ids: np.ndarray, mask: np.ndarray, row_index: int, viewpoint: int) -> int:
    slots = np.flatnonzero(mask[row_index] & (ids[row_index] == int(viewpoint)))
    if slots.size != 1:
        raise ValueError(f"viewpoint {viewpoint} is not uniquely available at row {row_index}")
    return int(slots[0])


def metric(labels: np.ndarray, predictions: Sequence[int], name: str, moves: Sequence[float] | None = None) -> dict[str, Any]:
    result = classification(labels, np.asarray(predictions, dtype=np.int64))
    result["method"] = name
    if moves is not None:
        result["move_rate"] = float(np.mean(np.asarray(moves, dtype=np.float64)))
        result["stay_rate"] = 1.0 - result["move_rate"]
    result["test_used"] = False
    return result


def option_lists(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    values: list[list[int]] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        expected = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
        actual = ids[index, valid].astype(int).tolist()
        if actual != expected:
            raise ValueError(f"option list mismatch at {row['episode_id']}")
        values.append(actual)
    return values


def predictions_for_views(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, views: Sequence[int]) -> np.ndarray:
    output = np.empty(len(views), dtype=np.int64)
    for index, view in enumerate(views):
        output[index] = int(np.argmax(logp[index, selected_slot(ids, mask, index, int(view))]))
    return output


def fused_logp(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, row_index: int, views: Sequence[int]) -> np.ndarray:
    slots = [selected_slot(ids, mask, row_index, int(view)) for view in views]
    return np.mean(logp[row_index, slots], axis=0)


def fused_feature_logits(
    features: np.ndarray, ids: np.ndarray, mask: np.ndarray, row_index: int,
    views: Sequence[int], head: SharedHead, device: torch.device,
) -> np.ndarray:
    slots = [selected_slot(ids, mask, row_index, int(view)) for view in views]
    value = np.mean(features[row_index, slots], axis=0, keepdims=True).astype(np.float32)
    with torch.inference_mode():
        output = head(torch.from_numpy(value).to(device)).cpu().numpy()[0]
    return log_softmax(output[None, :])[0]


def fused_feature_logits_batch(
    features: np.ndarray, ids: np.ndarray, mask: np.ndarray,
    row_indices: Sequence[int], view_sets: Sequence[Sequence[int]],
    head: SharedHead, device: torch.device,
) -> np.ndarray:
    """Run the frozen shared head once for many fused feature observations."""
    means: list[np.ndarray] = []
    for row_index, views in zip(row_indices, view_sets):
        slots = [selected_slot(ids, mask, int(row_index), int(view)) for view in views]
        means.append(np.mean(features[int(row_index), slots], axis=0))
    values = np.asarray(means, dtype=np.float32)
    with torch.inference_mode():
        logits = head(torch.from_numpy(values).to(device, non_blocking=True)).cpu().numpy()
    return log_softmax(logits)


def moves_for(rows: Sequence[Mapping[str, Any]], views: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [int(int(view) != int(row["current_viewpoint_id"])) for row, view in zip(rows, views)],
        dtype=np.float32,
    )


def random_candidate_views(rows: Sequence[Mapping[str, Any]], seed: int = SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.choice([int(x) for x in row["candidate_ids"]])) for row in rows]


def random_observation_sets(rows: Sequence[Mapping[str, Any]], budgets: Sequence[int]) -> dict[int, list[list[int]]]:
    rng = np.random.default_rng(SEED)
    output: dict[int, list[list[int]]] = {}
    for budget in budgets:
        sets: list[list[int]] = []
        for row in rows:
            current = int(row["current_viewpoint_id"])
            candidates = np.asarray([int(x) for x in row["candidate_ids"]], dtype=np.int64)
            count = min(max(int(budget) - 1, 0), len(candidates))
            picked = rng.choice(candidates, size=count, replace=False) if count else np.asarray([], dtype=np.int64)
            sets.append([current] + [int(x) for x in picked.tolist()])
        output[int(budget)] = sets
    return output


def candidate_geodesic(row: Mapping[str, Any], viewpoint: int) -> float:
    ids = [int(x) for x in row["candidate_ids"]]
    geometry = np.asarray(row.get("candidate_geodesic", []), dtype=np.float32)
    index = ids.index(int(viewpoint))
    return float(geometry[index]) if index < len(geometry) else 0.0


def budget_metrics(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray,
    logp: np.ndarray, features: np.ndarray, head: SharedHead, device: torch.device,
    observation_sets: Mapping[int, Sequence[Sequence[int]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    paths: dict[str, Any] = {}
    for budget, sets in observation_sets.items():
        confidence_predictions: list[int] = []
        confidence_moves: list[float] = []
        any_correct: list[bool] = []
        for fusion_name in ("MeanLogP", "MeanFeature"):
            predictions: list[int] = []
            gt_prob: list[float] = []
            entropies: list[float] = []
            selected_cost: list[float] = []
            feature_fused = (
                fused_feature_logits_batch(features, ids, mask, list(range(len(rows))), sets, head, device)
                if fusion_name == "MeanFeature" else None
            )
            for index, views in enumerate(sets):
                fused = fused_logp(logp, ids, mask, index, views) if fusion_name == "MeanLogP" else feature_fused[index]
                probabilities = np.exp(fused)
                predictions.append(int(np.argmax(fused)))
                gt_prob.append(float(probabilities[int(labels[index])]))
                entropies.append(float(entropy(fused[None])[0]))
                selected_cost.append(float(sum(candidate_geodesic(rows[index], view) for view in views if view != rows[index]["current_viewpoint_id"])))
            moves = np.asarray([len(views) - 1 for views in sets], dtype=np.float32)
            item = metric(labels, predictions, f"Random-B{budget}-{fusion_name}", moves > 0)
            item.update({
                "budget": int(budget),
                "mean_observations": float(np.mean([len(x) for x in sets])),
                "mean_gt_probability": float(np.mean(gt_prob)),
                "mean_entropy": float(np.mean(entropies)),
                "mean_path_length_proxy": float(np.mean(selected_cost)),
            })
            metrics[item["method"]] = item
        # BestObservedConfidence and AnyCorrect are reported for every budget
        # alongside the two fusion rules.  Both operate only on observations
        # in the sampled set; no unobserved candidate evidence is consulted.
        for index, views in enumerate(sets):
            slots = [selected_slot(ids, mask, index, int(view)) for view in views]
            best_position = max(
                range(len(slots)),
                key=lambda position: (
                    float(np.exp(logp[index, slots[position]]).max()),
                    -int(views[position]),
                ),
            )
            best_slot = slots[best_position]
            confidence_predictions.append(int(np.argmax(logp[index, best_slot])))
            confidence_moves.append(float(int(views[best_position]) != int(rows[index]["current_viewpoint_id"])))
            any_correct.append(bool(np.any(np.argmax(logp[index, slots], axis=1) == labels[index])))
        confidence_item = metric(
            labels,
            confidence_predictions,
            f"Random-B{budget}-BestObservedConfidence",
            confidence_moves,
        )
        confidence_item["budget"] = int(budget)
        confidence_item["mean_observations"] = float(np.mean([len(x) for x in sets]))
        metrics[confidence_item["method"]] = confidence_item
        metrics[f"Random-B{budget}-AnyCorrect"] = {
            "method": f"Random-B{budget}-AnyCorrect",
            "budget": int(budget),
            "contexts": int(len(rows)),
            "count": int(np.sum(any_correct)),
            "coverage": float(np.mean(any_correct)),
            "test_used": False,
        }
        paths[f"B{budget}"] = [list(map(int, views)) for views in sets]
    return metrics, paths


def best_single_oracle(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray, logp: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    selected: list[int] = []
    predictions: list[int] = []
    any_correct = np.zeros(len(rows), dtype=bool)
    selected_scores: list[float] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        views = ids[index, valid].astype(int).tolist()
        scores = logp[index, valid, labels[index]]
        best = int(valid[int(np.argmax(scores))])
        selected.append(int(ids[index, best]))
        predictions.append(int(np.argmax(logp[index, best])))
        any_correct[index] = bool(np.any(np.argmax(logp[index, valid], axis=1) == labels[index]))
        selected_scores.append(float(scores[int(np.argmax(scores))]))
    result = metric(labels, predictions, "GT-TrueLogP Oracle", np.asarray(selected) != np.asarray([r["current_viewpoint_id"] for r in rows]))
    result["mean_selected_gt_logp"] = float(np.mean(selected_scores))
    coverage = {
        "method": "Stay+candidate AnyCorrect Coverage",
        "contexts": int(len(rows)),
        "count": int(any_correct.sum()),
        "rate": float(any_correct.mean()),
        "test_used": False,
    }
    return result, np.asarray(selected, dtype=np.int64), any_correct


def best_b_fusion(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray,
    logp: np.ndarray, features: np.ndarray, head: SharedHead, device: torch.device, budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Enumerate legal states but evaluate both fusion functions in batched
    # chunks.  This avoids one CUDA launch per candidate pair on 10k contexts.
    state_rows: list[int] = []
    state_views: list[tuple[int, ...]] = []
    for index, row in enumerate(rows):
        current = int(row["current_viewpoint_id"])
        candidates = [int(x) for x in row["candidate_ids"]]
        count = min(max(int(budget) - 1, 0), len(candidates))
        extras = [()] if count == 0 else itertools.combinations(candidates, count)
        for extra in extras:
            state_rows.append(index)
            state_views.append((current,) + tuple(extra))
    lp_scores = np.full(len(rows), -np.inf, dtype=np.float64)
    feat_scores = np.full(len(rows), -np.inf, dtype=np.float64)
    selected_lp: list[list[int] | None] = [None] * len(rows)
    selected_feat: list[list[int] | None] = [None] * len(rows)
    for start in range(0, len(state_rows), 4096):
        end = min(start + 4096, len(state_rows))
        batch_rows = state_rows[start:end]
        batch_views = state_views[start:end]
        lp_values = np.asarray([fused_logp(logp, ids, mask, row_index, views) for row_index, views in zip(batch_rows, batch_views)])
        feat_values = fused_feature_logits_batch(features, ids, mask, batch_rows, batch_views, head, device)
        for offset, row_index in enumerate(batch_rows):
            lp_value = float(lp_values[offset, labels[row_index]])
            feat_value = float(feat_values[offset, labels[row_index]])
            state = batch_views[offset]
            if lp_value > lp_scores[row_index] or (lp_value == lp_scores[row_index] and tuple(-x for x in state) > tuple(-x for x in selected_lp[row_index] or ())):
                lp_scores[row_index] = lp_value
                selected_lp[row_index] = list(map(int, state))
            if feat_value > feat_scores[row_index] or (feat_value == feat_scores[row_index] and tuple(-x for x in state) > tuple(-x for x in selected_feat[row_index] or ())):
                feat_scores[row_index] = feat_value
                selected_feat[row_index] = list(map(int, state))
    selected_lp = [value or [int(row["current_viewpoint_id"])] for value, row in zip(selected_lp, rows)]
    selected_feat = [value or [int(row["current_viewpoint_id"])] for value, row in zip(selected_feat, rows)]
    preds_lp = [int(np.argmax(fused_logp(logp, ids, mask, i, state))) for i, state in enumerate(selected_lp)]
    feat_final = fused_feature_logits_batch(features, ids, mask, list(range(len(rows))), selected_feat, head, device)
    preds_feat = [int(np.argmax(value)) for value in feat_final]
    moves_lp = np.asarray([int(len(x) > 1) for x in selected_lp], dtype=np.float32)
    moves_feat = np.asarray([int(len(x) > 1) for x in selected_feat], dtype=np.float32)
    return {
        f"Best-B{budget}-MeanLogP": metric(labels, preds_lp, f"Best-B{budget}-MeanLogP", moves_lp),
        f"Best-B{budget}-MeanFeature": metric(labels, preds_feat, f"Best-B{budget}-MeanFeature", moves_feat),
    }, {f"B{budget}-MeanLogP": selected_lp, f"B{budget}-MeanFeature": selected_feat}


def pair_oracles(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray,
    logp: np.ndarray, features: np.ndarray, head: SharedHead, device: torch.device,
) -> dict[str, Any]:
    any_covered = 0
    pair_lp_predictions: list[int] = []
    pair_feat_predictions: list[int] = []
    best_pair_ids: list[int] = []
    pair_rows: list[int] = []
    pair_views: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        current = int(row["current_viewpoint_id"])
        current_slot = selected_slot(ids, mask, index, current)
        current_correct = int(np.argmax(logp[index, current_slot])) == int(labels[index])
        candidates = [int(x) for x in row["candidate_ids"]]
        candidate_correct = [int(np.argmax(logp[index, selected_slot(ids, mask, index, c)])) == int(labels[index]) for c in candidates]
        any_covered += int(current_correct or any(candidate_correct))
        for candidate in candidates:
            pair_rows.append(index)
            pair_views.append((current, candidate))
    lp_values = np.asarray([fused_logp(logp, ids, mask, i, views) for i, views in zip(pair_rows, pair_views)])
    feat_values = fused_feature_logits_batch(features, ids, mask, pair_rows, pair_views, head, device)
    best_positions: dict[int, int] = {}
    for position, row_index in enumerate(pair_rows):
        score = float(lp_values[position, labels[row_index]])
        current_best = best_positions.get(row_index)
        if current_best is None or (score, -pair_views[position][1]) > (float(lp_values[current_best, labels[row_index]]), -pair_views[current_best][1]):
            best_positions[row_index] = position
    best_feat_positions: dict[int, int] = {}
    for position, row_index in enumerate(pair_rows):
        score = float(feat_values[position, labels[row_index]])
        current_best = best_feat_positions.get(row_index)
        if current_best is None or (score, -pair_views[position][1]) > (float(feat_values[current_best, labels[row_index]]), -pair_views[current_best][1]):
            best_feat_positions[row_index] = position
    for row_index in range(len(rows)):
        position = best_positions[row_index]
        position_f = best_feat_positions[row_index]
        pair_lp_predictions.append(int(np.argmax(lp_values[position])))
        pair_feat_predictions.append(int(np.argmax(feat_values[position_f])))
        best_pair_ids.append(int(pair_views[position][1]))
    anycorrect = {
        "method": "BestPair AnyCorrect Oracle",
        "contexts": int(len(rows)),
        "count": int(any_covered),
        "coverage": float(any_covered / len(rows)),
        "test_used": False,
    }
    return {
        "pair_count": int(len(pair_rows)),
        "pair_anycorrect_coverage": float(any_covered / len(rows)),
        "pair_anycorrect_contexts": int(any_covered),
        "best_pair_anycorrect_oracle": anycorrect,
        "pair_mean_logp_oracle": metric(labels, pair_lp_predictions, "Pair GT-TrueLogP fusion Oracle", np.ones(len(rows))),
        "pair_mean_feature_oracle": metric(labels, pair_feat_predictions, "Pair MeanFeature fusion Oracle", np.ones(len(rows))),
        "best_pair_viewpoints": best_pair_ids,
        "test_used": False,
    }


class Verifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def current_evidence(features: np.ndarray, logp: np.ndarray) -> np.ndarray:
    current_feature = np.asarray(features[:, 0], dtype=np.float32)
    current_logp = np.asarray(logp[:, 0], dtype=np.float32)
    probabilities = np.exp(current_logp)
    extras = np.concatenate(
        (
            entropy(current_logp).reshape(-1, 1),
            probabilities.max(axis=1, keepdims=True),
            margin(current_logp).reshape(-1, 1),
            np.linalg.norm(current_feature, axis=1, keepdims=True),
        ),
        axis=1,
    )
    return np.concatenate((current_feature, current_logp, extras), axis=1).astype(np.float32)


def roc_auc(scores: np.ndarray, target: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranked = target[order]
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    positive_rank_sum = float(np.flatnonzero(ranked > 0.5).sum() + positives)
    return float((positive_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    truth = target[order]
    positives = max(float(truth.sum()), 1.0)
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float(np.sum(precision * truth) / positives)


def ece(scores: np.ndarray, target: np.ndarray) -> float:
    result = 0.0
    for low, high in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        active = (scores >= low) & (scores < high if high < 1.0 else scores <= high)
        if active.any():
            result += float(active.mean()) * abs(float(scores[active].mean()) - float(target[active].mean()))
    return result


def train_verifier(
    train_features: np.ndarray, train_logp: np.ndarray, train_rows: Sequence[Mapping[str, Any]],
    train_ids: np.ndarray, train_mask: np.ndarray, device: torch.device, checkpoint: Path,
) -> tuple[Verifier, Verifier, dict[str, Any]]:
    train_x = current_evidence(train_features, train_logp)
    labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    current_correct = (np.argmax(train_logp[:, 0], axis=1) == labels).astype(np.float32)
    candidate_correct = np.zeros(len(train_rows), dtype=bool)
    for index, label in enumerate(labels):
        valid = np.flatnonzero(train_mask[index])[1:]
        candidate_correct[index] = bool(np.any(np.argmax(train_logp[index, valid], axis=1) == label))
    need_move = ((current_correct < 0.5) & candidate_correct).astype(np.float32)
    model = Verifier(train_x.shape[1]).to(device)
    need_model = Verifier(train_x.shape[1]).to(device)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    target_ckpt = checkpoint.with_suffix(".pth")
    summary_path = checkpoint.with_suffix(".json")
    if target_ckpt.is_file() and summary_path.is_file():
        payload = torch.load(target_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["correct_state"])
        need_model.load_state_dict(payload["need_state"])
        return model.eval(), need_model.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(current_correct), torch.from_numpy(need_move))
    loader = DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True, pin_memory=True)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(need_model.parameters()), lr=VERIFIER_LR, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, float]] = []
    for epoch in range(1, VERIFIER_EPOCHS + 1):
        model.train(); need_model.train(); losses: list[float] = []
        for xb, yb, nb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True); nb = nb.to(device, non_blocking=True)
            pred = model(xb); need_pred = need_model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, yb) + nn.functional.binary_cross_entropy_with_logits(need_pred, nb)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        record = {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "optimizer_steps": float(len(losses))}
        history.append(record)
        print(f"[verifier] epoch={epoch:02d} loss={record['train_loss']:.6f}", flush=True)
    torch.save({"correct_state": model.state_dict(), "need_state": need_model.state_dict(), "input_dim": int(train_x.shape[1]), "test_used": False}, target_ckpt)
    summary = {
        "architecture": "Linear(272,256)-GELU-Linear(256,1)",
        "input": "current frozen feature + current logp + entropy/maxprob/margin/norm",
        "epochs": VERIFIER_EPOCHS, "lr": VERIFIER_LR, "weight_decay": WEIGHT_DECAY,
        "history": history, "train_contexts": len(train_rows),
        "positive_current_correct": int(current_correct.sum()), "positive_need_move": int(need_move.sum()),
        "checkpoint": str(target_ckpt.resolve()), "test_used": False,
    }
    write_json(summary_path, summary)
    return model.eval(), need_model.eval(), summary


def verifier_metrics(
    model: Verifier, need_model: Verifier, val_features: np.ndarray, val_logp: np.ndarray,
    val_rows: Sequence[Mapping[str, Any]], val_ids: np.ndarray, val_mask: np.ndarray, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x = current_evidence(val_features, val_logp)
    labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    current_correct = (np.argmax(val_logp[:, 0], axis=1) == labels).astype(np.float32)
    candidate_correct = np.zeros(len(val_rows), dtype=bool)
    for index, label in enumerate(labels):
        valid = np.flatnonzero(val_mask[index])[1:]
        candidate_correct[index] = bool(np.any(np.argmax(val_logp[index, valid], axis=1) == label))
    need_move = ((current_correct < 0.5) & candidate_correct).astype(np.float32)
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()
        need_scores = torch.sigmoid(need_model(torch.from_numpy(x).to(device))).cpu().numpy()
    quality = {
        "current_correct_verifier": {"auroc": roc_auc(scores, current_correct), "auprc": average_precision(scores, current_correct), "ece": ece(scores, current_correct), "accuracy_at_0.5": float(np.mean((scores >= 0.5) == current_correct))},
        "need_move_verifier": {"auroc": roc_auc(need_scores, need_move), "auprc": average_precision(need_scores, need_move), "ece": ece(need_scores, need_move), "positive_rate": float(need_move.mean())},
        "test_used": False,
    }
    diagnostics = {"current_correct": current_correct.tolist(), "need_move": need_move.tolist(), "current_correct_score": scores.tolist(), "need_move_score": need_scores.tolist()}
    return quality, diagnostics


def threshold_policy(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray, logp: np.ndarray,
    need_scores: np.ndarray,
) -> dict[str, Any]:
    # A fixed train-independent threshold keeps this diagnostic descriptive;
    # it is only executed when NeedMove has useful Val AUROC.
    threshold = 0.5
    rng = np.random.default_rng(SEED)
    actions: list[list[int]] = []
    predictions: list[int] = []
    moves: list[int] = []
    for index, row in enumerate(rows):
        current = int(row["current_viewpoint_id"])
        if need_scores[index] > threshold and row["candidate_ids"]:
            candidate = int(rng.choice(np.asarray(row["candidate_ids"], dtype=np.int64)))
            views = [current, candidate]
            prediction = int(np.argmax(fused_logp(logp, ids, mask, index, views)))
            moves.append(1)
        else:
            views = [current]
            prediction = int(np.argmax(logp[index, selected_slot(ids, mask, index, current)]))
            moves.append(0)
        actions.append(views); predictions.append(prediction)
    item = metric(labels, predictions, "NeedMove threshold policy + random O1 MeanLogP", moves)
    item.update({"threshold": threshold, "mean_observations": float(np.mean([len(x) for x in actions])), "actions": actions})
    return item


def observed_evidence_metrics(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray, logp: np.ndarray,
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    correct: list[int] = []
    gt_scores: list[float] = []
    for index, label in enumerate(labels):
        for slot in np.flatnonzero(mask[index])[1:]:
            view_logp = logp[index, slot]
            values["max_confidence"].append(float(np.exp(view_logp).max()))
            values["negative_entropy"].append(float(-entropy(view_logp[None])[0]))
            values["margin"].append(float(margin(view_logp[None])[0]))
            correct.append(int(np.argmax(view_logp) == label))
            gt_scores.append(float(view_logp[label]))
    target = np.asarray(correct, dtype=np.float64)
    result: dict[str, Any] = {"candidate_samples": int(len(target)), "test_used": False}
    for name, array in values.items():
        scores = np.asarray(array, dtype=np.float64)
        result[name] = {"auroc": roc_auc(scores, target), "mean_correct": float(scores[target > 0.5].mean()) if np.any(target > 0.5) else 0.0, "mean_wrong": float(scores[target <= 0.5].mean()) if np.any(target <= 0.5) else 0.0, "gt_true_logp_spearman": correlation(scores, gt_scores, spearman=True)}
    result["historical_unseen_predictor_spearman_reference"] = 0.5526
    result["pose_confidence"] = {"status": "NOT_AVAILABLE", "reason": "current archive cache has no candidate-level viewpoint confidence field; no proxy was invented"}
    return result


def complementarity_metrics(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray, logp: np.ndarray,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rescue = harm = both_wrong_fusion_correct = 0
    pair_total = 0
    for index, row in enumerate(rows):
        current = int(row["current_viewpoint_id"])
        current_slot = selected_slot(ids, mask, index, current)
        current_correct = bool(np.argmax(logp[index, current_slot]) == labels[index])
        for candidate in [int(x) for x in row["candidate_ids"]]:
            pair_total += 1
            cand_slot = selected_slot(ids, mask, index, candidate)
            candidate_correct = bool(np.argmax(logp[index, cand_slot]) == labels[index])
            counts[f"current_{int(current_correct)}_candidate_{int(candidate_correct)}"] += 1
            fused = fused_logp(logp, ids, mask, index, (current, candidate))
            fusion_correct = bool(np.argmax(fused) == labels[index])
            rescue += int((not current_correct) and candidate_correct)
            harm += int(current_correct and not fusion_correct)
            both_wrong_fusion_correct += int((not current_correct) and (not candidate_correct) and fusion_correct)
    return {"pair_count": pair_total, "patterns": dict(counts), "current_wrong_candidate_correct": rescue, "current_correct_fusion_wrong": harm, "both_single_wrong_fusion_correct": both_wrong_fusion_correct, "test_used": False}


def per_class_report(
    labels: np.ndarray, methods: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {name: classification(labels, predictions)["per_class"] for name, predictions in methods.items()}


def occlusion_report(
    data_root: Path, rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray,
    logp: np.ndarray, random_sets: Sequence[int], random_metric_cache: Mapping[str, Any],
) -> dict[str, Any]:
    path = data_root / VISIBILITY_RELATIVE
    if not path.is_file():
        return {"status": "NOT_AVAILABLE", "reason": str(path), "test_used": False}
    visibility = read_cache(path)
    if not np.array_equal(visibility["ids"], ids) or not np.array_equal(visibility["mask"], mask):
        raise ValueError("frame-0 visibility action-set mismatch")
    current_scores = visibility["scores"][:, 0]
    subset = current_scores < np.quantile(current_scores, 1.0 / 3.0)
    indices = np.flatnonzero(subset)
    output: dict[str, Any] = {"definition": "bottom tertile of current frame-0 visibility", "contexts": int(len(indices)), "methods": {}, "test_used": False}
    for name, predictions in random_metric_cache.items():
        if not isinstance(predictions, np.ndarray) or len(predictions) != len(rows):
            continue
        output["methods"][name] = metric(labels[subset], predictions[subset], name)
    output["methods"]["Stay"] = metric(labels[subset], np.argmax(logp[subset, 0], axis=1), "Stay")
    output["current_visibility_mean"] = float(current_scores[subset].mean()) if indices.size else 0.0
    return output


def navigation_cost_curve(rows: Sequence[Mapping[str, Any]], random_sets: Mapping[int, Sequence[Sequence[int]]]) -> dict[str, Any]:
    curve: dict[str, Any] = {}
    for budget, sets in random_sets.items():
        path_cost = [sum(candidate_geodesic(row, view) for view in views if view != row["current_viewpoint_id"]) for row, views in zip(rows, sets)]
        curve[f"B{budget}"] = {"mean_observations": float(np.mean([len(x) for x in sets])), "mean_path_length_proxy": float(np.mean(path_cost)), "p90_path_length_proxy": float(np.quantile(path_cost, 0.9)), "test_used": False}
    return curve


def protocol_gate(rows: Sequence[Mapping[str, Any]], labels: np.ndarray, ids: np.ndarray, mask: np.ndarray, logp: np.ndarray) -> dict[str, Any]:
    stay_pred = np.argmax(logp[:, 0], axis=1)
    stay = metric(labels, stay_pred, "Stay")
    oracle, _, any_correct = best_single_oracle(rows, labels, ids, mask, logp)
    coverage = {
        "contexts": int(len(rows)),
        "count": int(any_correct.sum()),
        "rate": float(any_correct.mean()),
        "test_used": False,
    }
    candidate_count = mask[:, 1:].sum(axis=1)
    return {
        "status": "PASSED",
        "contexts": len(rows),
        "stay": stay,
        "candidate_count": {"mean": float(candidate_count.mean()), "min": int(candidate_count.min()), "max": int(candidate_count.max())},
        "full_legal_gt_true_logp_oracle": oracle,
        "stay_plus_candidate_anycorrect": coverage,
        "historical_references": {"stay_accuracy": 0.302579, "stay_macro_f1": 0.292976, "oracle_accuracy": 0.753175, "oracle_macro_f1": 0.755358, "anycorrect": 0.791964},
        "sanity_deltas_pp": {"stay_accuracy": 100.0 * (stay["accuracy"] - 0.302579), "oracle_accuracy": 100.0 * (oracle["accuracy"] - 0.753175), "anycorrect": 100.0 * (coverage["rate"] - 0.791964)},
        "test_used": False,
    }


def make_analysis(result: Mapping[str, Any]) -> str:
    gate = result["protocol_reproduction"]
    second = result["real_second_observation"]
    budget = result["observation_budget"]
    verifier = result["verifier"]["quality"]
    b2 = budget.get("Random-B2-MeanLogP", {})
    b3 = budget.get("Random-B3-MeanLogP", {})
    full = result["fusion_oracle"].get("Best-B4-MeanLogP", {})
    stay_acc = float(gate["stay"]["accuracy"])
    b3_gain = float(b3.get("accuracy", stay_acc)) - stay_acc
    b4_gain = float(full.get("accuracy", stay_acc)) - stay_acc
    if b3_gain >= 0.05 or b4_gain >= 0.05:
        decision = "STRONG KEEP finite-observation active perception"
    elif max(b3_gain, b4_gain) >= 0.02:
        decision = "KEEP finite-observation active perception"
    elif float(second.get("Random O1 + MeanLogP", {}).get("accuracy", stay_acc)) > stay_acc + 0.01:
        decision = "OBSERVATIONS VALUABLE BUT ACQUISITION UNSOLVED"
    else:
        decision = "KILL finite-observation active perception"
    lines = [
        "# Reduced12 Overnight Actual Observation Research Sweep", "",
        "Only frozen reduced12 ST-GCN/shared-head caches and Policy Train/Moving-Val artifacts were read. Policy Test, VLM, new RGB/skeleton/DINO generation and recognizer fine-tuning were not used.", "",
        "## Protocol reproduction gate", "",
        f"Moving Val contexts: **{gate['contexts']}**. Action set is exactly `Stay/current + Stage-A legal candidate_pool`; candidate count mean/min/max={gate['candidate_count']['mean']:.3f}/{gate['candidate_count']['min']}/{gate['candidate_count']['max']}.",
        f"Stay: {gate['stay']['accuracy']:.6f} Acc / {gate['stay']['macro_f1']:.6f} F1. Full legal GT-TrueLogP oracle: {gate['full_legal_gt_true_logp_oracle']['accuracy']:.6f} / {gate['full_legal_gt_true_logp_oracle']['macro_f1']:.6f}; AnyCorrect coverage={gate['stay_plus_candidate_anycorrect']['rate']:.6f}.",
        "The gate reproduces the historical Stay/oracle references within the recorded cache-order protocol; no Test artifact was opened.", "",
        "## Real second observation", "",
        "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    for name, item in second.items():
        if isinstance(item, dict) and "accuracy" in item:
            lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} |")
    lines += ["", "## Observation budget", "", "| Method | Accuracy | Macro-F1 | Mean observations |", "|---|---:|---:|---:|"]
    for name, item in budget.items():
        if isinstance(item, dict) and "accuracy" in item:
            lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item.get('mean_observations', 0.0):.3f} |")
    lines += [
        "", "## Complementarity and stop/continue", "",
        f"Random-B2 MeanLogP gain over Stay: {100.0 * (float(b2.get('accuracy', stay_acc)) - stay_acc):+.2f}pp; Random-B3 gain: {100.0 * b3_gain:+.2f}pp.",
        f"NeedMove verifier AUROC/AUPRC: {verifier['need_move_verifier']['auroc']:.6f}/{verifier['need_move_verifier']['auprc']:.6f}. The threshold policy is reported only when the AUROC gate is met; it does not use future evidence before acquisition.",
        "Observed candidate confidence/entropy/margin diagnostics are post-acquisition evidence and are not unseen-candidate predictors.", "",
        "## Scientific answers", "",
        "A. A second real observation is evaluated as a finite-observation budget abstraction: each full candidate observation is independently acquired; this report does not claim continuous synchronized navigation.",
        f"B. B=2 and B=3 random fusion gains are {100.0 * (float(budget.get('Random-B2-MeanFeature', {}).get('accuracy', stay_acc)) - stay_acc):+.2f}pp and {100.0 * (float(budget.get('Random-B3-MeanFeature', {}).get('accuracy', stay_acc)) - stay_acc):+.2f}pp for MeanFeature.",
        f"C. The privileged unrestricted B4 reference reaches {float(full.get('accuracy', 0.0)):.6f} Acc; the preregistered sweep decision is **{decision}**.",
        "D. Complementarity, per-class and occlusion-stratified counts are retained in their dedicated JSON files; rescue and harm are reported rather than hidden.",
        "E. No pre-action unseen-candidate utility predictor or new policy was trained. The only Train-only model is the small current-observation verifier.", "",
        "## Flags", "", "```text",
        "policy_test_used=false", "training_used=true_for_current_observation_verifier_only", "new_rgb_generated=false", "new_skeleton_generated=false", "new_dino_generated=false", "frozen_stgcn_modified=false", "future_candidate_observation_read_before_selection=0", "gt_label_used_for_privileged_oracle_only=true", "observation_budget_is_discrete_full_view_abstraction=true", "```", "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything()
    started = time.time()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    device = require_cuda(args.device)
    train_rows, val_rows = load_rows(data_root)
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    runtime = data_root / RUNTIME_RELATIVE
    train_cache = read_cache(runtime / "train_options.npz")
    val_cache = read_cache(runtime / "val_options.npz")
    all_cache = read_cache(runtime / "val_all32.npz")
    stgcn_checkpoint = data_root / STGCN_RELATIVE
    if not stgcn_checkpoint.is_file():
        raise FileNotFoundError(stgcn_checkpoint)
    stgcn_sha256 = sha256_file(stgcn_checkpoint)
    cache_provenance = {
        "train_options": validate_cache_metadata(
            runtime / "train_options.npz", train_rows, False, stgcn_sha256
        ),
        "val_options": validate_cache_metadata(
            runtime / "val_options.npz", val_rows, False, stgcn_sha256
        ),
        "val_all32": validate_cache_metadata(
            runtime / "val_all32.npz", val_rows, True, stgcn_sha256
        ),
    }
    validate_options(train_rows, train_cache, "train")
    validate_options(val_rows, val_cache, "val")
    all_ids = np.asarray(all_cache["ids"], dtype=np.int64)
    all_mask = np.asarray(all_cache["mask"], dtype=bool)
    if all_ids.shape != (len(val_rows), NUM_VIEWS):
        raise ValueError("all32 cache shape mismatch")
    expected_all_ids = np.broadcast_to(np.arange(NUM_VIEWS, dtype=np.int64), all_ids.shape)
    if not np.array_equal(all_ids, expected_all_ids):
        raise ValueError("all32 cache viewpoint order mismatch")
    if all_mask.shape != all_ids.shape or not np.all(all_mask):
        raise ValueError("all32 cache mask mismatch")
    head_path = data_root / HEAD_RELATIVE
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    head_sha256 = sha256_file(head_path)
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    train_features, train_logp = recognizer_arrays(train_cache, head, device)
    val_features, val_logp = recognizer_arrays(val_cache, head, device)
    ids, mask = np.asarray(val_cache["ids"]), np.asarray(val_cache["mask"])
    val_options = option_lists(val_rows, ids, mask)
    output.mkdir(parents=True, exist_ok=True)

    gate = protocol_gate(val_rows, val_labels, ids, mask, val_logp)
    write_json(output / "protocol_reproduction.json", gate)
    if abs(gate["sanity_deltas_pp"]["stay_accuracy"]) > 0.5 or abs(gate["sanity_deltas_pp"]["oracle_accuracy"]) > 1.0:
        gate["status"] = "STOP_PROTOCOL_MISMATCH"
        result = {"status": gate["status"], "protocol_reproduction": gate, "test_used": False}
        write_json(output / "result.json", result)
        (output / "analysis.md").write_text("# Protocol gate stopped\n\nAction-set or recognizer cache mismatch exceeded the preregistered tolerance; no downstream sweep was run.\n", encoding="utf-8")
        return result

    current_views = [int(row["current_viewpoint_id"]) for row in val_rows]
    random_views = random_candidate_views(val_rows)
    random_candidate_predictions = predictions_for_views(val_logp, ids, mask, random_views)
    second: dict[str, Any] = {
        "Random candidate only (O1)": metric(val_labels, random_candidate_predictions, "Random candidate only (O1)", np.ones(len(val_rows))),
    }
    best_conf_predictions: list[int] = []
    mean_prob_predictions: list[int] = []
    mean_logp_predictions: list[int] = []
    mean_feat_predictions: list[int] = []
    for index, candidate in enumerate(random_views):
        current_slot = selected_slot(ids, mask, index, current_views[index]); candidate_slot = selected_slot(ids, mask, index, candidate)
        pair = logp = np.stack([val_logp[index, current_slot], val_logp[index, candidate_slot]])
        best_slot = int(np.argmax(np.exp(pair).max(axis=1)))
        best_conf_predictions.append(int(np.argmax(pair[best_slot])))
        mean_prob = np.mean(np.exp(pair), axis=0); mean_prob_predictions.append(int(np.argmax(mean_prob)))
        mean_logp_predictions.append(int(np.argmax(np.mean(pair, axis=0))))
        mean_feat_predictions.append(int(np.argmax(fused_feature_logits(val_features, ids, mask, index, (current_views[index], candidate), head, device))))
    second["BestObservedConfidence(O0,O1)"] = metric(val_labels, best_conf_predictions, "BestObservedConfidence(O0,O1)")
    second["Posterior average(O0,O1)"] = metric(val_labels, mean_prob_predictions, "Posterior average(O0,O1)")
    second["MeanLogP(O0,O1)"] = metric(val_labels, mean_logp_predictions, "MeanLogP(O0,O1)")
    second["MeanFeature(O0,O1)"] = metric(val_labels, mean_feat_predictions, "MeanFeature(O0,O1)")
    oracle, oracle_views, any_correct = best_single_oracle(val_rows, val_labels, ids, mask, val_logp)
    second["Privileged Best-of-Two AnyCorrect"] = {"coverage": float(np.mean([bool(np.argmax(val_logp[i, selected_slot(ids, mask, i, current_views[i])]) == val_labels[i]) or bool(np.argmax(val_logp[i, selected_slot(ids, mask, i, random_views[i])]) == val_labels[i]) for i in range(len(val_rows))])), "test_used": False}
    gt_pair_predictions: list[int] = []
    gt_pair_selected_views: list[int] = []
    for index, candidate in enumerate(random_views):
        current_slot = selected_slot(ids, mask, index, current_views[index])
        candidate_slot = selected_slot(ids, mask, index, candidate)
        choices = [current_slot, candidate_slot]
        best = max(choices, key=lambda slot: (float(val_logp[index, slot, val_labels[index]]), -int(ids[index, slot])))
        gt_pair_predictions.append(int(np.argmax(val_logp[index, best])))
        gt_pair_selected_views.append(int(ids[index, best]))
    second["GT-TrueLogP best-of-two"] = metric(
        val_labels,
        gt_pair_predictions,
        "GT-TrueLogP best-of-two",
        moves_for(val_rows, gt_pair_selected_views),
    )
    write_json(output / "real_second_observation_metrics.json", second)

    sets = random_observation_sets(val_rows, OBSERVATION_BUDGETS)
    budget, _trajectories = budget_metrics(val_rows, val_labels, ids, mask, val_logp, val_features, head, device, sets)
    write_json(output / "observation_budget_metrics.json", budget)

    pair = pair_oracles(val_rows, val_labels, ids, mask, val_logp, val_features, head, device)
    write_json(output / "pair_oracle_metrics.json", pair)

    fusion_oracle: dict[str, Any] = {}
    for budget_value in (2, 3, 4):
        current, _paths = best_b_fusion(val_rows, val_labels, ids, mask, val_logp, val_features, head, device, budget_value)
        fusion_oracle.update(current)
    # Full legal means the current view plus exactly the Stage-A legal pool,
    # never the unrestricted all-32 archive.
    full_valid = [list(map(int, values)) for values in val_options]
    full_metrics: dict[str, Any] = {}
    full_lp_preds: list[int] = []
    full_any = []
    for i, views in enumerate(full_valid):
        fused = fused_logp(val_logp, ids, mask, i, views)
        full_lp_preds.append(int(np.argmax(fused)))
        full_any.append(bool(np.any(np.argmax(val_logp[i, [selected_slot(ids, mask, i, view) for view in views]], axis=1) == val_labels[i])))
    full_metrics["Full legal MeanLogP"] = metric(val_labels, full_lp_preds, "Full legal MeanLogP", np.ones(len(val_rows)))
    full_metrics["Full legal AnyCorrect"] = {"coverage": float(np.mean(full_any)), "count": int(np.sum(full_any)), "contexts": len(val_rows), "test_used": False}
    full_conf_preds: list[int] = []
    for i, views in enumerate(full_valid):
        slots = [selected_slot(ids, mask, i, view) for view in views]
        best = max(slots, key=lambda slot: (float(np.exp(val_logp[i, slot]).max()), -int(ids[i, slot])))
        full_conf_preds.append(int(np.argmax(val_logp[i, best])))
    full_metrics["Full legal BestObservedConfidence"] = metric(val_labels, full_conf_preds, "Full legal BestObservedConfidence", np.ones(len(val_rows)))
    full_metrics["Full legal GT-TrueLogP"] = gate["full_legal_gt_true_logp_oracle"]
    fusion_oracle.update(full_metrics)
    # Keep the committed report compact; per-context combinations are runtime
    # diagnostics rather than a reusable cache and are intentionally omitted.
    write_json(output / "fusion_oracle_metrics.json", {"metrics": fusion_oracle})
    write_json(output / "multiview_fusion_metrics.json", budget)

    observed = observed_evidence_metrics(val_rows, val_labels, ids, mask, val_logp)
    write_json(output / "observed_evidence_metrics.json", observed)

    verifier_ckpt = data_root / VERIFIER_RELATIVE / "current_evidence_verifier"
    verifier, need_model, verifier_training = train_verifier(train_features, train_logp, train_rows, train_cache["ids"], train_cache["mask"], device, verifier_ckpt)
    quality, verifier_diag = verifier_metrics(verifier, need_model, val_features, val_logp, val_rows, ids, mask, device)
    verifier_result: dict[str, Any] = {"training": verifier_training, "quality": quality}
    if quality["need_move_verifier"]["auroc"] >= 0.75:
        verifier_result["threshold_policy"] = threshold_policy(val_rows, val_labels, ids, mask, val_logp, np.asarray(verifier_diag["need_move_score"]))
    else:
        verifier_result["threshold_policy"] = {"status": "SKIPPED_AUROC_BELOW_0.75", "test_used": False}
    write_json(output / "verifier_metrics.json", verifier_result)

    comp = complementarity_metrics(val_rows, val_labels, ids, mask, val_logp)
    write_json(output / "complementarity_metrics.json", comp)

    per_class_predictions = {
        "Stay": np.asarray(np.argmax(val_logp[:, 0], axis=1)),
        "Random candidate O1": random_candidate_predictions,
        "Random B2 MeanLogP": np.asarray([int(np.argmax(fused_logp(val_logp, ids, mask, i, sets[2][i]))) for i in range(len(val_rows))]),
        "Random B3 MeanLogP": np.asarray([int(np.argmax(fused_logp(val_logp, ids, mask, i, sets[3][i]))) for i in range(len(val_rows))]),
        "BestPair MeanLogP": np.asarray([
            int(np.argmax(
                fused_logp(
                    val_logp,
                    ids,
                    mask,
                    i,
                    (current_views[i], pair["best_pair_viewpoints"][i]),
                )
            ))
            for i in range(len(val_rows))
        ]),
        "Full legal MeanLogP": np.asarray(full_lp_preds),
    }
    write_json(output / "per_class_metrics.json", per_class_report(val_labels, per_class_predictions))

    best_conf_b2: list[int] = []
    best_pair_preds: list[int] = []
    for i, views in enumerate(sets[2]):
        slots = [selected_slot(ids, mask, i, view) for view in views]
        best = max(slots, key=lambda slot: (float(np.exp(val_logp[i, slot]).max()), -int(ids[i, slot])))
        best_conf_b2.append(int(np.argmax(val_logp[i, best])))
    for i, candidate in enumerate(pair["best_pair_viewpoints"]):
        best_pair_preds.append(int(np.argmax(fused_logp(val_logp, ids, mask, i, (current_views[i], int(candidate))))))
    gt_full_preds = predictions_for_views(val_logp, ids, mask, oracle_views)
    occlusion_predictions = {
        "Stay": per_class_predictions["Stay"],
        "Random B2 MeanLogP": per_class_predictions["Random B2 MeanLogP"],
        "Random B3 MeanLogP": per_class_predictions["Random B3 MeanLogP"],
        "BestObservedConfidence B2": np.asarray(best_conf_b2),
        "BestPair MeanLogP Oracle": np.asarray(best_pair_preds),
        "Full legal GT-TrueLogP Oracle": gt_full_preds,
        "Full legal MeanLogP": per_class_predictions["Full legal MeanLogP"],
    }
    occlusion = occlusion_report(data_root, val_rows, val_labels, ids, mask, val_logp, (2, 3), occlusion_predictions)
    write_json(output / "occlusion_metrics.json", occlusion)
    costs = navigation_cost_curve(val_rows, sets)
    write_json(output / "navigation_cost_metrics.json", costs)
    # Keep the attachment's descriptive filename while retaining the original
    # navigation-cost filename used by earlier sweep tooling.
    write_json(output / "observation_cost_curve.json", costs)

    coverage = {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(r['record_id']) for r in train_rows}), "moving_val_records": len({str(r['record_id']) for r in val_rows}), "train_signature": row_signature(train_rows), "val_signature": row_signature(val_rows), "candidate_samples": int(mask[:, 1:].sum()), "action_set": "Stay/current + Stage-A legal candidate_pool", "stgcn_checkpoint": str(stgcn_checkpoint.resolve()), "stgcn_checkpoint_sha256": stgcn_sha256, "shared_head_checkpoint": str(head_path.resolve()), "shared_head_checkpoint_sha256": head_sha256, "cache_provenance": cache_provenance, "test_used": False}
    write_json(output / "coverage_audit.json", coverage)
    leakage = {"policy_test_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "new_dino_generated": False, "frozen_stgcn_modified": False, "future_candidate_observation_read_before_selection": 0, "future_candidate_evidence_used_as_deployable_input": False, "gt_label_used_for_privileged_oracle_only": True, "verifier_train_split_only": True, "observation_budget_is_discrete_full_view_abstraction": True}
    write_json(output / "leakage_audit.json", leakage)

    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_OVERNIGHT_ACTUAL_OBSERVATION_SWEEP", "status": "COMPLETED", "population": {"moving_val_contexts": len(val_rows), "train_contexts": len(train_rows)}, "labels": list(LABELS), "protocol_reproduction": gate, "real_second_observation": second, "observation_budget": budget, "pair_oracle": pair, "fusion_oracle": fusion_oracle, "verifier": verifier_result, "observed_evidence": observed, "complementarity": comp, "occlusion": occlusion, "navigation_cost": costs, "coverage_audit": coverage, "cache_provenance": cache_provenance, "leakage_audit": leakage, "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.time() - started}, "flags": leakage,
    }
    write_json(output / "result.json", result)
    (output / "analysis.md").write_text(make_analysis(result), encoding="utf-8")
    write_json(output / "config.json", {"seed": SEED, "device": str(device), "budgets": list(OBSERVATION_BUDGETS), "verifier_epochs": VERIFIER_EPOCHS, "action_set": "Stay/current + Stage-A legal candidate_pool", "test_used": False})
    write_json(output / "runtime_summary.json", result["runtime"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve()), "gate": result["protocol_reproduction"], "test_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
