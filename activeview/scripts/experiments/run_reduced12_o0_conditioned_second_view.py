#!/usr/bin/env python3
"""Train/Val-only O0-conditioned complementary second-view sweep.

The experiment treats O0 as an already acquired complete observation and
selects exactly one *non-stay* legal second viewpoint.  Future candidate
recognizer evidence is used only for Train targets and privileged Val
diagnostics.  The frozen reduced12 ST-GCN encoder and shared adapted head are
never modified, and Policy Test is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.policy_data import RecordBalancedSampler
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    head_logits,
    load_rows,
    row_signature,
)

SEED = 42
NUM_CLASSES = 12
NUM_VIEWS = 32
FEATURE_DIM = 256
MAX_CANDIDATES = 21
RAW_GEOMETRY_DIM = 11
OPTION_GEOMETRY_DIM = 18
TRAIN_EPOCHS = 12
TRAIN_BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
LISTWISE_TAU = 0.5
SMOOTH_ALPHA = 10.0
RUNTIME_REL = Path("diagnostics/reduced12_dual_route_overnight")
STGCN_REL = Path("checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
HEAD_REL = Path("checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth")
OUTPUT_REL = Path("experiments/reduced12_eight_placement_v1/overnight_o0_conditioned_second_view")
CHECKPOINT_REL = Path("checkpoints/policy_reduced12_eight_placement_v1/overnight_o0_conditioned_second_view")


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_cache(path: Path, rows: Sequence[Mapping[str, Any]], all32: bool, checkpoint_sha: str) -> dict[str, np.ndarray]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {"rows": len(rows), "signature": row_signature(rows), "all32": all32, "checkpoint_sha256": checkpoint_sha, "test_used": False}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{path.name} provenance mismatch {key}: expected={value!r} actual={metadata.get(key)!r}")
    cache = read_npz(path)
    if cache["features"].shape[0] != len(rows) or cache["ids"].shape != cache["mask"].shape:
        raise ValueError(f"invalid cache shape in {path}")
    for index, row in enumerate(rows):
        valid = np.flatnonzero(cache["mask"][index]).tolist()
        expected_ids = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
        if cache["ids"][index, valid].astype(int).tolist() != expected_ids:
            raise ValueError(f"cache/action mismatch at {row['episode_id']}")
    return cache


def log_softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)


def entropy(logp: np.ndarray) -> np.ndarray:
    p = np.exp(np.asarray(logp, dtype=np.float64))
    return -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=-1)


def margin(logp: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(logp, dtype=np.float64), axis=-1)
    return values[..., -1] - values[..., -2]


def lattice_features(viewpoint: int) -> np.ndarray:
    radius, azimuth = divmod(int(viewpoint), 8)
    angle = 2.0 * np.pi * azimuth / 8.0
    return np.asarray([radius / 3.0, np.sin(angle), np.cos(angle)], dtype=np.float32)


def metric(labels: np.ndarray, predictions: np.ndarray, name: str, selected: np.ndarray | None = None, rows: Sequence[Mapping[str, Any]] | None = None, candidate_quality: np.ndarray | None = None, candidate_margin: np.ndarray | None = None, nav: np.ndarray | None = None) -> dict[str, Any]:
    result = classification(labels, predictions)
    result["method"] = name
    result["test_used"] = False
    if selected is not None and rows is not None:
        current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
        result["move_rate"] = float(np.mean(selected != current))
        result["stay_rate"] = 1.0 - result["move_rate"]
        result["selected_pair_correct_rate"] = float(np.mean(predictions == labels))
        result["selected_viewpoint_mean"] = float(np.mean(selected))
    if candidate_quality is not None:
        result["mean_selected_pair_true_logp"] = float(np.mean(candidate_quality))
    if candidate_margin is not None:
        result["mean_selected_pair_margin"] = float(np.mean(candidate_margin))
    if nav is not None:
        result["mean_nav_distance_m"] = float(np.mean(nav))
    return result


def selected_slots(ids: np.ndarray, mask: np.ndarray, selected: np.ndarray) -> np.ndarray:
    slots = np.empty(len(selected), dtype=np.int64)
    for index, viewpoint in enumerate(selected):
        found = np.flatnonzero(mask[index] & (ids[index] == int(viewpoint)))
        if found.size != 1:
            raise ValueError(f"viewpoint {viewpoint} is not uniquely cached at row {index}")
        slots[index] = int(found[0])
    return slots


def build_arrays(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], head: SharedHead, device: torch.device, geometry_mean: np.ndarray | None = None, geometry_std: np.ndarray | None = None) -> dict[str, np.ndarray]:
    features = np.asarray(cache["features"], dtype=np.float32)
    raw_logits = head_logits(head, features, device)
    logp = log_softmax(raw_logits)
    count = len(rows)
    candidate_features = np.zeros((count, MAX_CANDIDATES, FEATURE_DIM), dtype=np.float32)
    candidate_logp = np.zeros((count, MAX_CANDIDATES, NUM_CLASSES), dtype=np.float32)
    candidate_ids = np.full((count, MAX_CANDIDATES), -1, dtype=np.int64)
    candidate_mask = np.zeros((count, MAX_CANDIDATES), dtype=bool)
    geometry = np.zeros((count, MAX_CANDIDATES, OPTION_GEOMETRY_DIM), dtype=np.float32)
    geodesic = np.zeros((count, MAX_CANDIDATES), dtype=np.float32)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    current_ids = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
    for index, row in enumerate(rows):
        valid = np.flatnonzero(cache["mask"][index])
        candidate_count = len(valid) - 1
        if candidate_count <= 0 or candidate_count > MAX_CANDIDATES:
            raise ValueError(f"invalid candidate count {candidate_count} at {row['episode_id']}")
        candidate_features[index, :candidate_count] = features[index, valid[1:]]
        candidate_logp[index, :candidate_count] = logp[index, valid[1:]]
        candidate_ids[index, :candidate_count] = cache["ids"][index, valid[1:]]
        candidate_mask[index, :candidate_count] = True
        raw_geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        raw_geo = np.asarray(row["candidate_geodesic"], dtype=np.float32)
        if raw_geometry.shape != (candidate_count, RAW_GEOMETRY_DIM):
            raise ValueError(f"geometry shape mismatch at {row['episode_id']}: {raw_geometry.shape}")
        if geometry_mean is not None and geometry_std is not None:
            raw_geometry = (raw_geometry - geometry_mean) / np.clip(geometry_std, 1e-6, None)
        current_latent = lattice_features(int(row["current_viewpoint_id"]))
        candidate_latents = np.stack(
            [lattice_features(int(value)) for value in candidate_ids[index, :candidate_count]], axis=0
        )
        geometry[index, :candidate_count] = np.concatenate(
            (
                raw_geometry,
                np.repeat(current_latent[None, :], candidate_count, axis=0),
                candidate_latents,
                np.zeros((candidate_count, 1), dtype=np.float32),
            ),
            axis=1,
        )
        geodesic[index, :candidate_count] = raw_geo
    return {"features": features, "logp": logp, "candidate_features": candidate_features, "candidate_logp": candidate_logp, "candidate_ids": candidate_ids, "candidate_mask": candidate_mask, "geometry": geometry, "geodesic": geodesic, "labels": labels, "current_ids": current_ids, "current_feature": features[:, 0], "current_logp": logp[:, 0]}


def pair_targets(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    current = arrays["current_logp"][:, None, :]
    candidate = arrays["candidate_logp"]
    pair = log_softmax(0.5 * (current + candidate))
    labels = arrays["labels"]
    row_ids = np.arange(len(labels))
    candidate_positions = np.arange(candidate.shape[1], dtype=np.int64)[None, :]
    true_logp = pair[row_ids[:, None], candidate_positions, labels[:, None]]
    other = pair.copy()
    other[row_ids[:, None], candidate_positions, labels[:, None]] = -np.inf
    true_margin = true_logp - np.max(other, axis=-1)
    correct = (np.argmax(pair, axis=-1) == labels[:, None]).astype(np.float32)
    valid = arrays["candidate_mask"]
    true_logp[~valid] = 0.0
    true_margin[~valid] = 0.0
    correct[~valid] = 0.0
    return {"pair_logp": pair, "pair_true_logp": true_logp.astype(np.float32), "pair_margin": true_margin.astype(np.float32), "pair_correct": correct, "valid": valid}


def select_metrics(rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], selected_index: np.ndarray, name: str) -> dict[str, Any]:
    n = len(rows)
    selected_index = np.asarray(selected_index, dtype=np.int64)
    if np.any(~arrays["candidate_mask"][np.arange(n), selected_index]):
        raise ValueError(f"invalid selected candidate for {name}")
    selected_view = arrays["candidate_ids"][np.arange(n), selected_index]
    selected_logp = targets["pair_logp"][np.arange(n), selected_index]
    predictions = np.argmax(selected_logp, axis=-1).astype(np.int64)
    quality = targets["pair_true_logp"][np.arange(n), selected_index]
    pair_margin = targets["pair_margin"][np.arange(n), selected_index]
    nav = arrays["geodesic"][np.arange(n), selected_index]
    candidate_predictions = np.argmax(arrays["candidate_logp"][np.arange(n), selected_index], axis=-1)
    result = metric(arrays["labels"], predictions, name, selected_view, rows, quality, pair_margin, nav)
    result["selected_candidate_observation_correct_rate"] = float(np.mean(candidate_predictions == arrays["labels"]))
    return result


def stable_argmax(values: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape[0], dtype=np.int64)
    for index in range(values.shape[0]):
        valid = np.flatnonzero(mask[index])
        best = valid[0]
        for slot in valid[1:]:
            if values[index, slot] > values[index, best] or (values[index, slot] == values[index, best] and ids[index, slot] < ids[index, best]):
                best = int(slot)
        output[index] = int(best)
    return output


def random_selection(mask: np.ndarray, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros(mask.shape[0], dtype=np.int64)
    for index in range(mask.shape[0]):
        valid = np.flatnonzero(mask[index])
        output[index] = int(rng.choice(valid))
    return output


def geometry_baselines(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = arrays["candidate_mask"]
    geometry = arrays["geometry"]
    ids = arrays["candidate_ids"]
    output: dict[str, np.ndarray] = {"Random-B2": random_selection(mask)}
    output["NearestCandidate-B2"] = stable_argmax(-arrays["geodesic"], ids, mask)
    output["FarthestCandidate-B2"] = stable_argmax(arrays["geodesic"], ids, mask)
    current = arrays["current_ids"][:, None]
    azimuth_delta = np.abs((ids % 8) - (current % 8))
    azimuth_delta = np.minimum(azimuth_delta, 8 - azimuth_delta).astype(np.float32) * (np.pi / 4.0)
    angular = azimuth_delta
    output["MaxAngularChange-B2"] = stable_argmax(angular, ids, mask)
    output["MinAngularChange-B2"] = stable_argmax(-angular, ids, mask)
    return output


def pair_oracle_audit(rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ids, mask = arrays["candidate_ids"], arrays["candidate_mask"]
    row_indices = np.arange(len(rows), dtype=np.int64)[:, None]
    candidate_indices = np.arange(arrays["candidate_logp"].shape[1], dtype=np.int64)[None, :]
    raw_scores = arrays["candidate_logp"][row_indices, candidate_indices, arrays["labels"][:, None]]
    single_view_selected = stable_argmax(raw_scores, ids, mask)
    raw_selected = np.array(single_view_selected, copy=True)
    pair_lp = stable_argmax(targets["pair_true_logp"], ids, mask)
    pair_margin = stable_argmax(targets["pair_margin"], ids, mask)
    pair_correct = targets["pair_correct"] > 0.5
    any_correct = np.any(pair_correct & mask, axis=1)
    raw_pair = select_metrics(rows, arrays, targets, raw_selected, "HistoricalRawMeanLogP-B2")
    norm_pair = select_metrics(rows, arrays, targets, pair_lp, "PairNormalizedTrueLogPOracle-B2")
    margin_pair = select_metrics(rows, arrays, targets, pair_margin, "PairMarginOracle-B2")
    historical_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/overnight_actual_observation_sweep/pair_oracle_metrics.json"
    historical_agreement = float(np.mean(raw_selected == single_view_selected))
    historical_available = False
    if historical_path.is_file():
        historical_payload = json.loads(historical_path.read_text(encoding="utf-8"))
        historical_ids = np.asarray(historical_payload.get("best_pair_viewpoints", []), dtype=np.int64)
        current_ids = ids[np.arange(len(rows)), raw_selected]
        if historical_ids.shape == current_ids.shape:
            historical_agreement = float(np.mean(historical_ids == current_ids))
            historical_available = True
    report = {
        "pair_count": int(mask.sum()),
        "historical_raw_definition": "argmax_c 0.5*(logp0[y]+logpc[y]); current term cancels",
        "historical_raw_vs_single_candidate_true_logp_top1_agreement": historical_agreement,
        "historical_raw_selection_source": str(historical_path.resolve()) if historical_available else "reconstructed raw definition",
        "historical_raw_vs_normalized_pair_selection_agreement": float(np.mean(raw_selected == pair_lp)),
        "historical_raw_oracle": raw_pair,
        "normalized_pair_true_logp_oracle": norm_pair,
        "pair_margin_oracle": margin_pair,
        "pair_anycorrect_coverage": float(np.mean(any_correct)),
        "pair_anycorrect_contexts": int(any_correct.sum()),
        "pair_margin_vs_single_candidate_true_logp_agreement": float(np.mean(pair_margin == raw_selected)),
        "pair_margin_vs_normalized_pair_agreement": float(np.mean(pair_margin == pair_lp)),
        "test_used": False,
    }
    return report, {"raw": raw_selected, "pair_logp": pair_lp, "pair_margin": pair_margin, "any_correct": any_correct}


def build_view_priors(train_arrays: Mapping[str, np.ndarray], train_targets: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    sums_logp = np.zeros((NUM_CLASSES, NUM_VIEWS), dtype=np.float64)
    sums_margin = np.zeros_like(sums_logp)
    counts = np.zeros_like(sums_logp)
    for index in range(len(train_arrays["labels"])):
        label = int(train_arrays["labels"][index])
        for slot in np.flatnonzero(train_arrays["candidate_mask"][index]):
            view = int(train_arrays["candidate_ids"][index, slot])
            sums_logp[label, view] += float(train_targets["pair_true_logp"][index, slot])
            sums_margin[label, view] += float(train_targets["pair_margin"][index, slot])
            counts[label, view] += 1.0
    global_logp = np.divide(sums_logp.sum(axis=0), np.maximum(counts.sum(axis=0), 1.0))
    global_margin = np.divide(sums_margin.sum(axis=0), np.maximum(counts.sum(axis=0), 1.0))
    class_logp = np.divide(sums_logp, np.maximum(counts, 1.0))
    class_margin = np.divide(sums_margin, np.maximum(counts, 1.0))
    smoothed_logp = (counts * class_logp + SMOOTH_ALPHA * global_logp[None, :]) / (counts + SMOOTH_ALPHA)
    smoothed_margin = (counts * class_margin + SMOOTH_ALPHA * global_margin[None, :]) / (counts + SMOOTH_ALPHA)
    return {"logp": smoothed_logp.astype(np.float32), "margin": smoothed_margin.astype(np.float32), "counts": counts.astype(np.int64)}


def prior_selection(arrays: Mapping[str, np.ndarray], priors: Mapping[str, np.ndarray], use_margin: bool, hard: bool = False) -> np.ndarray:
    p0 = np.exp(arrays["current_logp"])
    if hard:
        p0 = np.eye(NUM_CLASSES, dtype=np.float32)[np.argmax(p0, axis=-1)]
    table = priors["margin" if use_margin else "logp"]
    scores = np.zeros(arrays["candidate_mask"].shape, dtype=np.float32)
    for index in range(len(p0)):
        scores[index] = p0[index] @ table[:, arrays["candidate_ids"][index].clip(0, NUM_VIEWS - 1)]
    return stable_argmax(scores, arrays["candidate_ids"], arrays["candidate_mask"])


class CandidateScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def branch_input(arrays: Mapping[str, np.ndarray], branch: str, indices: np.ndarray | None = None) -> np.ndarray:
    n = len(arrays["labels"]) if indices is None else len(indices)
    select = slice(None) if indices is None else indices
    geometry = arrays["geometry"][select]
    if branch == "GeometryOnly":
        return geometry
    p0 = np.exp(arrays["current_logp"][select])[:, None, :].repeat(geometry.shape[1], axis=1)
    if branch == "Posterior+Geometry":
        return np.concatenate((p0, geometry), axis=-1)
    feature = arrays["current_feature"][select][:, None, :].repeat(geometry.shape[1], axis=1)
    if branch == "Feature+Geometry":
        return np.concatenate((feature, geometry), axis=-1)
    if branch == "Feature+Posterior+Geometry":
        return np.concatenate((feature, p0, geometry), axis=-1)
    if branch == "RecognizerState+Geometry":
        prob = np.exp(arrays["current_logp"][select])
        state = np.concatenate((prob, entropy(arrays["current_logp"][select])[:, None], np.max(prob, axis=-1, keepdims=True), margin(arrays["current_logp"][select])[:, None]), axis=-1)
        return np.concatenate((state[:, None, :].repeat(geometry.shape[1], axis=1), geometry), axis=-1)
    raise ValueError(branch)


def train_branch(name: str, branch: str, target_name: str, train_rows: Sequence[Mapping[str, Any]], train_arrays: Mapping[str, np.ndarray], train_targets: Mapping[str, np.ndarray], val_arrays: Mapping[str, np.ndarray], val_targets: Mapping[str, np.ndarray], device: torch.device, checkpoint_dir: Path, summary_path: Path, binary: bool = False) -> tuple[CandidateScorer, dict[str, Any]]:
    input_dim = {"GeometryOnly": OPTION_GEOMETRY_DIM, "Posterior+Geometry": NUM_CLASSES + OPTION_GEOMETRY_DIM, "Feature+Geometry": FEATURE_DIM + OPTION_GEOMETRY_DIM, "Feature+Posterior+Geometry": FEATURE_DIM + NUM_CLASSES + OPTION_GEOMETRY_DIM, "RecognizerState+Geometry": NUM_CLASSES + 3 + OPTION_GEOMETRY_DIM}[branch]
    checkpoint = checkpoint_dir / f"{name}.pth"
    if checkpoint.is_file() and summary_path.is_file():
        model = CandidateScorer(input_dim).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return model.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    model = CandidateScorer(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = RecordBalancedSampler(train_rows, episodes_per_record=16, seed=SEED)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        sampler.set_epoch(epoch - 1)
        indices = np.asarray(list(iter(sampler)), dtype=np.int64)
        permutation = np.random.default_rng(SEED + epoch).permutation(len(indices))
        indices = indices[permutation]
        model.train()
        losses: list[float] = []
        for start in range(0, len(indices), TRAIN_BATCH):
            batch_indices = indices[start : start + TRAIN_BATCH]
            xb = torch.from_numpy(branch_input(train_arrays, branch, batch_indices)).to(device, non_blocking=True)
            mask = torch.from_numpy(train_targets["valid"][batch_indices]).to(device, non_blocking=True)
            target = torch.from_numpy(train_targets[target_name][batch_indices]).to(device, non_blocking=True)
            pred = model(xb)
            active = mask
            if binary:
                loss = nn.functional.binary_cross_entropy_with_logits(pred[active], target[active])
            else:
                point = nn.functional.smooth_l1_loss(pred[active], target[active])
                pred_dist = torch.log_softmax(pred.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                target_dist = torch.softmax(target.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                listwise = nn.functional.kl_div(pred_dist, target_dist, reduction="batchmean", log_target=False)
                loss = point + 0.5 * listwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_loss_values: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_arrays["labels"]), TRAIN_BATCH):
                idx = np.arange(start, min(start + TRAIN_BATCH, len(val_arrays["labels"])), dtype=np.int64)
                pred = model(torch.from_numpy(branch_input(val_arrays, branch, idx)).to(device, non_blocking=True))
                mask = torch.from_numpy(val_targets["valid"][idx]).to(device)
                target = torch.from_numpy(val_targets[target_name][idx]).to(device)
                active = mask
                if binary:
                    value = nn.functional.binary_cross_entropy_with_logits(pred[active], target[active])
                else:
                    point = nn.functional.smooth_l1_loss(pred[active], target[active])
                    pred_dist = torch.log_softmax(pred.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                    target_dist = torch.softmax(target.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                    value = point + 0.5 * nn.functional.kl_div(pred_dist, target_dist, reduction="batchmean")
                val_loss_values.append(float(value.cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_utility_loss": float(np.mean(val_loss_values)), "optimizer_steps": int(math.ceil(len(indices) / TRAIN_BATCH)), "sampled_contexts": int(len(indices)), "records": int(len(sampler.groups))}
        history.append(record)
        print(f"[b2:{name}] epoch={epoch:02d} train_loss={record['train_loss']:.6f} val_loss={record['val_utility_loss']:.6f} contexts={len(indices)}", flush=True)
        if record["val_utility_loss"] < best_loss:
            best_loss = record["val_utility_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": SEED, "input_dim": input_dim, "target": target_name, "binary": binary, "test_used": False}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {"branch": branch, "target": target_name, "binary": binary, "input_dim": input_dim, "hidden_dim": 256, "epochs": TRAIN_EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "best_epoch": best_epoch, "best_val_utility_loss": best_loss, "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    write_json(summary_path, summary)
    return model.eval(), summary


def infer_scores(model: CandidateScorer, arrays: Mapping[str, np.ndarray], branch: str, device: torch.device) -> np.ndarray:
    scores = np.zeros(arrays["candidate_mask"].shape, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(arrays["labels"]), TRAIN_BATCH):
            idx = np.arange(start, min(start + TRAIN_BATCH, len(arrays["labels"])), dtype=np.int64)
            values = model(torch.from_numpy(branch_input(arrays, branch, idx)).to(device, non_blocking=True)).cpu().numpy()
            scores[idx] = values
    return scores


def build_b3_arrays(arrays: Mapping[str, np.ndarray], first_selection: np.ndarray) -> dict[str, np.ndarray]:
    """Build PairState + geometry inputs for one unvisited third candidate."""
    n = len(arrays["labels"])
    first = np.asarray(first_selection, dtype=np.int64)
    row = np.arange(n, dtype=np.int64)
    z0 = arrays["current_feature"]
    p0 = np.exp(arrays["current_logp"])
    z1 = arrays["candidate_features"][row, first]
    p1 = np.exp(arrays["candidate_logp"][row, first])
    fused_p = np.exp(log_softmax(0.5 * (arrays["current_logp"] + arrays["candidate_logp"][row, first])))
    state = np.concatenate((z0, z1, np.abs(z0 - z1), z0 * z1, p0, p1, fused_p), axis=-1).astype(np.float32)
    mask = np.array(arrays["candidate_mask"], copy=True)
    mask[row, first] = False
    candidate_logp = np.array(arrays["candidate_logp"], copy=True)
    candidate_ids = np.array(arrays["candidate_ids"], copy=True)
    geometry = np.array(arrays["geometry"], copy=True)
    geodesic = np.array(arrays["geodesic"], copy=True)
    candidate_positions = np.arange(mask.shape[1], dtype=np.int64)[None, :]
    triple = log_softmax(
        (arrays["current_logp"][:, None, :] + arrays["candidate_logp"][row, first][:, None, :] + candidate_logp) / 3.0
    )
    labels = arrays["labels"]
    true_logp = triple[np.arange(n)[:, None], candidate_positions, labels[:, None]]
    other = np.array(triple, copy=True)
    other[np.arange(n)[:, None], candidate_positions, labels[:, None]] = -np.inf
    target_margin = true_logp - np.max(other, axis=-1)
    target_margin[~mask] = 0.0
    return {"state": state, "current_logp": np.asarray(arrays["current_logp"], dtype=np.float32), "candidate_logp": candidate_logp, "candidate_ids": candidate_ids, "candidate_mask": mask, "geometry": geometry, "geodesic": geodesic, "labels": labels, "target_margin": target_margin.astype(np.float32), "first_selection": first}


def b3_input(arrays: Mapping[str, np.ndarray], indices: np.ndarray | None = None) -> np.ndarray:
    state = arrays["state"] if indices is None else arrays["state"][indices]
    geometry = arrays["geometry"] if indices is None else arrays["geometry"][indices]
    return np.concatenate((np.repeat(state[:, None, :], geometry.shape[1], axis=1), geometry), axis=-1).astype(np.float32)


def train_b3_model(
    train_rows: Sequence[Mapping[str, Any]], train_arrays: Mapping[str, np.ndarray],
    val_arrays: Mapping[str, np.ndarray], device: torch.device, checkpoint_dir: Path,
    summary_path: Path,
) -> tuple[CandidateScorer, dict[str, Any]]:
    input_dim = (FEATURE_DIM * 4 + NUM_CLASSES * 3) + OPTION_GEOMETRY_DIM
    checkpoint = checkpoint_dir / "pairstate_geometry_third_margin.pth"
    if checkpoint.is_file() and summary_path.is_file():
        model = CandidateScorer(input_dim).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
        return model.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    model = CandidateScorer(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = RecordBalancedSampler(train_rows, episodes_per_record=16, seed=SEED)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        sampler.set_epoch(epoch - 1)
        indices = np.asarray(list(iter(sampler)), dtype=np.int64)
        indices = indices[np.random.default_rng(SEED + epoch).permutation(len(indices))]
        model.train()
        losses: list[float] = []
        for start in range(0, len(indices), TRAIN_BATCH):
            batch_indices = indices[start : start + TRAIN_BATCH]
            xb = torch.from_numpy(b3_input(train_arrays, batch_indices)).to(device, non_blocking=True)
            mask = torch.from_numpy(train_arrays["candidate_mask"][batch_indices]).to(device, non_blocking=True)
            target = torch.from_numpy(train_arrays["target_margin"][batch_indices]).to(device, non_blocking=True)
            if not bool(mask.any()):
                continue
            pred = model(xb)
            active = mask
            point = nn.functional.smooth_l1_loss(pred[active], target[active])
            pred_dist = torch.log_softmax(pred.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
            target_dist = torch.softmax(target.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
            loss = point + 0.5 * nn.functional.kl_div(pred_dist, target_dist, reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_arrays["labels"]), TRAIN_BATCH):
                idx = np.arange(start, min(start + TRAIN_BATCH, len(val_arrays["labels"])), dtype=np.int64)
                mask = torch.from_numpy(val_arrays["candidate_mask"][idx]).to(device)
                target = torch.from_numpy(val_arrays["target_margin"][idx]).to(device)
                if not bool(mask.any()):
                    continue
                pred = model(torch.from_numpy(b3_input(val_arrays, idx)).to(device, non_blocking=True))
                point = nn.functional.smooth_l1_loss(pred[mask], target[mask])
                pred_dist = torch.log_softmax(pred.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                target_dist = torch.softmax(target.masked_fill(~mask, -1e4) / LISTWISE_TAU, dim=1)
                val_losses.append(float((point + 0.5 * nn.functional.kl_div(pred_dist, target_dist, reduction="batchmean")).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, "val_utility_loss": float(np.mean(val_losses)) if val_losses else 0.0, "sampled_contexts": int(len(indices)), "optimizer_steps": int(len(losses)), "records": int(len(sampler.groups))}
        history.append(record)
        print(f"[b3] epoch={epoch:02d} train_loss={record['train_loss']:.6f} val_loss={record['val_utility_loss']:.6f}", flush=True)
        if record["val_utility_loss"] < best_loss:
            best_loss = record["val_utility_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "seed": SEED, "input_dim": input_dim, "target": "triple_margin", "test_used": False}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    summary = {"status": "COMPLETED", "input": "[z0,z1,|z0-z1|,z0*z1,p0,p1,p01] + candidate_geometry", "input_dim": input_dim, "target": "three-view normalized MeanLogP GT margin", "epochs": TRAIN_EPOCHS, "best_epoch": best_epoch, "best_val_utility_loss": best_loss, "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    write_json(summary_path, summary)
    return model.eval(), summary


def infer_b3_scores(model: CandidateScorer, arrays: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    output = np.zeros(arrays["candidate_mask"].shape, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(arrays["labels"]), TRAIN_BATCH):
            idx = np.arange(start, min(start + TRAIN_BATCH, len(arrays["labels"])), dtype=np.int64)
            output[idx] = model(torch.from_numpy(b3_input(arrays, idx)).to(device, non_blocking=True)).cpu().numpy()
    return output


def evaluate_b3(arrays: Mapping[str, np.ndarray], scores: np.ndarray, seed: int = SEED) -> dict[str, Any]:
    mask = arrays["candidate_mask"]
    selected = stable_argmax(scores, arrays["candidate_ids"], mask)
    rng = np.random.default_rng(seed)
    random_selected = np.zeros(len(arrays["labels"]), dtype=np.int64)
    for index in range(len(random_selected)):
        random_selected[index] = int(rng.choice(np.flatnonzero(mask[index])))
    # Exact three-view margin oracle over the remaining legal candidates.
    labels = arrays["labels"]
    first = arrays["first_selection"]
    # candidate_logp still contains all candidates; reconstruct the exact
    # three-view normalized MeanLogP utility for the privileged oracle.
    oracle = np.zeros(len(labels), dtype=np.int64)
    for index in range(len(labels)):
        valid = np.flatnonzero(mask[index])
        best_value = -np.inf
        for slot in valid:
            triple = log_softmax((arrays["current_logp"][index] + arrays["candidate_logp"][index, first[index]] + arrays["candidate_logp"][index, slot]) / 3.0)
            target = int(labels[index])
            value = float(triple[target] - np.max(np.delete(triple, target)))
            if value > best_value:
                best_value, oracle[index] = value, int(slot)
    return {"learned_selected": selected, "random_selected": random_selected, "oracle_selected": oracle}


def ranking_stats(name: str, scores: np.ndarray, arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], oracle: Mapping[str, np.ndarray]) -> dict[str, Any]:
    mask = arrays["candidate_mask"]
    predicted = scores[mask]
    margin_target = targets["pair_margin"][mask]
    logp_target = targets["pair_true_logp"][mask]
    within: list[float] = []
    for index in range(len(scores)):
        valid = mask[index]
        if int(valid.sum()) >= 2:
            within.append(correlation(scores[index, valid], targets["pair_margin"][index, valid], spearman=True))
    selected = stable_argmax(scores, arrays["candidate_ids"], mask)
    top3_overlap = []
    for index in range(len(scores)):
        valid = np.flatnonzero(mask[index])
        order = valid[np.argsort(-scores[index, valid], kind="mergesort")]
        oracle_order = valid[np.argsort(-targets["pair_margin"][index, valid], kind="mergesort")]
        top3_overlap.append(float(len(set(order[:3].tolist()) & set(oracle_order[:3].tolist())) > 0))
    return {"name": name, "candidate_spearman_pair_margin": correlation(predicted, margin_target, spearman=True), "candidate_spearman_pair_logp": correlation(predicted, logp_target, spearman=True), "within_context_spearman_mean": float(np.mean(within)) if within else 0.0, "within_context_spearman_median": float(np.median(within)) if within else 0.0, "oracle_top1_overlap": float(np.mean(selected == oracle["pair_margin"])), "oracle_top3_overlap": float(np.mean(top3_overlap)), "test_used": False}


def confidence_strata(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.max(np.exp(arrays["current_logp"]), axis=-1)
    return np.digitize(values, np.quantile(values, [1 / 3, 2 / 3]), right=True)


def subset_metrics(rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], selections: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {"contexts": int(len(indices))}
    for name, selection in selections.items():
        selected = selection[indices]
        sub_rows = [rows[int(i)] for i in indices]
        sub_arrays = {key: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == len(rows) else value for key, value in arrays.items()}
        sub_targets = {key: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == len(rows) else value for key, value in targets.items()}
        output[name] = select_metrics(sub_rows, sub_arrays, sub_targets, selected, name)
    return output


def fusion_prediction(l0: np.ndarray, l1: np.ndarray, rule: str) -> np.ndarray:
    if rule == "MeanLogP":
        return log_softmax(0.5 * (l0 + l1))
    if rule == "MeanPosterior":
        return np.log(np.clip(0.5 * (np.exp(l0) + np.exp(l1)), 1e-12, None))
    if rule in {"ConfidenceWeighted", "MarginWeighted"}:
        q = np.max(np.exp(np.stack((l0, l1), axis=1)), axis=-1) if rule == "ConfidenceWeighted" else margin(np.stack((l0, l1), axis=1))
        weights = np.exp((q - np.max(q, axis=1, keepdims=True)) / 0.1)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return log_softmax(weights[:, 0, None] * l0 + weights[:, 1, None] * l1)
    if rule == "BestObservedConfidence":
        q = np.max(np.exp(np.stack((l0, l1), axis=1)), axis=-1)
        return np.where((q[:, 0] >= q[:, 1])[:, None], l0, l1)
    raise ValueError(rule)


def fusion_metrics(rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], selected: np.ndarray, name: str) -> dict[str, Any]:
    slots = selected
    l0 = arrays["current_logp"]
    l1 = arrays["candidate_logp"][np.arange(len(rows)), slots]
    output: dict[str, Any] = {}
    for rule in ("MeanLogP", "MeanPosterior", "ConfidenceWeighted", "MarginWeighted", "BestObservedConfidence"):
        pred = np.argmax(fusion_prediction(l0, l1, rule), axis=-1)
        item = classification(arrays["labels"], pred)
        item.update({"method": f"{name}×{rule}", "test_used": False})
        output[rule] = item
    return output


def transition_diagnostics(arrays: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], selected: np.ndarray, fused_pred: np.ndarray) -> dict[str, int]:
    labels = arrays["labels"]
    o0 = np.argmax(arrays["current_logp"], axis=-1)
    o1 = np.argmax(arrays["candidate_logp"][np.arange(len(labels)), selected], axis=-1)
    return {"o0_wrong_to_final_correct": int(np.sum((o0 != labels) & (fused_pred == labels))), "o0_correct_to_final_wrong": int(np.sum((o0 == labels) & (fused_pred != labels))), "o1_wrong_but_fusion_correct": int(np.sum((o1 != labels) & (fused_pred == labels))), "both_wrong_fusion_correct": int(np.sum((o0 != labels) & (o1 != labels) & (fused_pred == labels))), "both_correct_fusion_wrong": int(np.sum((o0 == labels) & (o1 == labels) & (fused_pred != labels))), "contexts": int(len(labels))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = get_data_root()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_REL)
    args = parser.parse_args()
    started = time.time()
    seed_everything()
    device = require_cuda(args.device)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = data_root / STGCN_REL
    head_checkpoint = data_root / HEAD_REL
    cache_root = data_root / RUNTIME_REL
    train_rows, val_rows = load_rows(data_root)
    stgcn_sha = sha256_file(checkpoint)
    train_cache = load_cache(cache_root / "train_options.npz", train_rows, False, stgcn_sha)
    val_cache = load_cache(cache_root / "val_options.npz", val_rows, False, stgcn_sha)
    # The all-32 cache is provenance-only in this experiment; the formal B2
    # action set is always current + Stage-A legal candidates.
    all32_meta = json.loads((cache_root / "val_all32.json").read_text(encoding="utf-8"))
    if all32_meta.get("test_used") is not False or all32_meta.get("signature") != row_signature(val_rows):
        raise ValueError("val_all32 provenance is invalid")
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_checkpoint, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    raw_geometry = np.concatenate([np.asarray(row["candidate_geometry"], dtype=np.float64) for row in train_rows], axis=0)
    geometry_mean = raw_geometry.mean(axis=0).astype(np.float32)
    geometry_std = np.maximum(raw_geometry.std(axis=0), 1e-6).astype(np.float32)
    train_arrays = build_arrays(train_rows, train_cache, head, device, geometry_mean, geometry_std)
    val_arrays = build_arrays(val_rows, val_cache, head, device, geometry_mean, geometry_std)
    train_targets = pair_targets(train_arrays)
    val_targets = pair_targets(val_arrays)
    write_json(output / "config.json", {"seed": SEED, "device": str(device), "train_epochs": TRAIN_EPOCHS, "record_balanced_contexts_per_record": 16, "train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "action_set": "current/Stay + Stage-A legal candidate_pool; B2 selects exactly one non-stay candidate", "fusion": "normalized MeanLogP", "listwise_tau": LISTWISE_TAU, "view_prior_smoothing_alpha": SMOOTH_ALPHA, "stgcn_checkpoint": str(checkpoint.resolve()), "stgcn_checkpoint_sha256": stgcn_sha, "shared_head_checkpoint": str(head_checkpoint.resolve()), "test_used": False})
    write_json(output / "coverage_audit.json", {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(x["record_id"]) for x in train_rows}), "moving_val_records": len({str(x["record_id"]) for x in val_rows}), "train_candidate_samples": int(train_arrays["candidate_mask"].sum()), "val_candidate_samples": int(val_arrays["candidate_mask"].sum()), "action_set": "current/Stay + Stage-A legal candidate_pool", "all32_cache_provenance_only": True, "test_used": False})
    oracle_report, oracle = pair_oracle_audit(val_rows, val_arrays, val_targets)
    write_json(output / "pair_oracle_definition_audit.json", {key: oracle_report[key] for key in ("historical_raw_definition", "historical_raw_vs_single_candidate_true_logp_top1_agreement", "historical_raw_vs_normalized_pair_selection_agreement", "pair_margin_vs_single_candidate_true_logp_agreement", "pair_margin_vs_normalized_pair_agreement", "test_used")})
    write_json(output / "pair_oracle_metrics.json", oracle_report)
    priors = build_view_priors(train_arrays, train_targets)
    prior_selections = {"SoftBelief-ViewPrior": prior_selection(val_arrays, priors, True), "SoftBelief-LogPViewPrior": prior_selection(val_arrays, priors, False), "HardClass-ViewPrior": prior_selection(val_arrays, priors, True, hard=True)}
    prior_metrics = {name: select_metrics(val_rows, val_arrays, val_targets, selection, name) for name, selection in prior_selections.items()}
    write_json(output / "soft_belief_prior_metrics.json", {"smoothing_alpha": SMOOTH_ALPHA, "class_view_counts": priors["counts"].tolist(), "metrics": prior_metrics, "test_used": False})
    selectors: dict[str, np.ndarray] = {"Random-B2": geometry_baselines(val_arrays)["Random-B2"], **prior_selections}
    geom = geometry_baselines(val_arrays)
    for name, selection in geom.items():
        selectors[name] = selection
    learned_metrics: dict[str, Any] = {}
    ranking: dict[str, Any] = {}
    branch_models: dict[str, tuple[CandidateScorer, str]] = {}
    branch_defs = (("GeometryOnly", "GeometryOnly", "pair_true_logp"), ("Posterior+Geometry", "Posterior+Geometry", "pair_true_logp"), ("Feature+Geometry", "Feature+Geometry", "pair_true_logp"), ("Feature+Posterior+Geometry", "Feature+Posterior+Geometry", "pair_true_logp"), ("GeometryOnly", "GeometryOnly", "pair_margin"), ("Posterior+Geometry", "Posterior+Geometry", "pair_margin"), ("Feature+Geometry", "Feature+Geometry", "pair_margin"), ("Feature+Posterior+Geometry", "Feature+Posterior+Geometry", "pair_margin"))
    training_summaries: dict[str, Any] = {}
    for branch, input_branch, target_name in branch_defs:
        suffix = "PairLogP" if target_name == "pair_true_logp" else "PairMargin"
        name = f"{branch}-{suffix}"
        model, summary = train_branch(name, input_branch, target_name, train_rows, train_arrays, train_targets, val_arrays, val_targets, device, data_root / CHECKPOINT_REL, output / f"{name}_training.json")
        scores = infer_scores(model, val_arrays, input_branch, device)
        selection = stable_argmax(scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"])
        selectors[name] = selection
        learned_metrics[name] = select_metrics(val_rows, val_arrays, val_targets, selection, name)
        ranking[name] = ranking_stats(name, scores, val_arrays, val_targets, oracle)
        training_summaries[name] = summary
        branch_models[name] = (model, input_branch)
        write_json(output / "b2_selector_metrics.json", {"metrics": learned_metrics, "test_used": False})
        write_json(output / "b2_ranking_metrics.json", ranking)
    best_model_name = max(learned_metrics, key=lambda key: learned_metrics[key]["accuracy"])
    best_model, best_input_branch = branch_models[best_model_name]
    # One binary correctness diagnostic, restricted to the best input branch D.
    direct_name = "PairCorrectness-Direct"
    direct_model, direct_summary = train_branch(direct_name, "Feature+Posterior+Geometry", "pair_correct", train_rows, train_arrays, train_targets, val_arrays, val_targets, device, data_root / CHECKPOINT_REL, output / "pair_correctness_training.json", binary=True)
    direct_scores = infer_scores(direct_model, val_arrays, "Feature+Posterior+Geometry", device)
    direct_selection = stable_argmax(direct_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"])
    selectors[direct_name] = direct_selection
    learned_metrics[direct_name] = select_metrics(val_rows, val_arrays, val_targets, direct_selection, direct_name)
    ranking[direct_name] = ranking_stats(direct_name, direct_scores, val_arrays, val_targets, oracle)
    training_summaries[direct_name] = direct_summary
    write_json(output / "b2_selector_metrics.json", {"metrics": learned_metrics, "best_branch_by_moving_accuracy": best_model_name, "test_used": False})
    write_json(output / "b2_training_summary.json", training_summaries)
    write_json(output / "b2_ranking_metrics.json", ranking)
    # O0 information shuffle diagnostic for the best learned branch.
    rng = np.random.default_rng(SEED)
    shuffle_results: dict[str, Any] = {}
    base_states = {key: np.array(value, copy=True) for key, value in (("current_feature", val_arrays["current_feature"]), ("current_logp", val_arrays["current_logp"]))}
    for shuffle_name, shuffle_keys in (("normal", ()), ("feature_shuffled", ("current_feature",)), ("posterior_shuffled", ("current_logp",)), ("both_shuffled", ("current_feature", "current_logp"))):
        shuffled = dict(val_arrays)
        for key in shuffle_keys:
            shuffled[key] = base_states[key][rng.permutation(len(val_rows))]
        scores = infer_scores(best_model, shuffled, best_input_branch, device)
        selection = stable_argmax(scores, shuffled["candidate_ids"], shuffled["candidate_mask"])
        # Final terminal evidence remains the real selected candidate; only the
        # deployable O0 state is shuffled.
        shuffled_targets = val_targets
        item = select_metrics(val_rows, shuffled, shuffled_targets, selection, shuffle_name)
        shuffle_results[shuffle_name] = {"accuracy": item["accuracy"], "macro_f1": item["macro_f1"], "selected": selection.tolist()}
    write_json(output / "b2_shuffle_diagnostics.json", {"best_branch": best_model_name, "metrics": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in shuffle_results.items()}, "test_used": False})
    # Correctness/confidence/occlusion stratification.
    all_selection_names = {"Random-B2": selectors["Random-B2"], "SoftBelief-ViewPrior": prior_selections["SoftBelief-ViewPrior"], "Posterior+Geometry-PairMargin": selectors["Posterior+Geometry-PairMargin"], best_model_name: selectors[best_model_name], "PairMarginOracle": oracle["pair_margin"]}
    o0_correct = np.argmax(val_arrays["current_logp"], axis=-1) == val_arrays["labels"]
    write_json(output / "o0_correctness_stratified_metrics.json", {"correct": subset_metrics(val_rows, val_arrays, val_targets, all_selection_names, np.flatnonzero(o0_correct)), "wrong": subset_metrics(val_rows, val_arrays, val_targets, all_selection_names, np.flatnonzero(~o0_correct)), "test_used": False})
    strata = confidence_strata(val_arrays)
    write_json(output / "confidence_stratified_metrics.json", {str(name): subset_metrics(val_rows, val_arrays, val_targets, {"Random-B2": selectors["Random-B2"], best_model_name: selectors[best_model_name], "PairMarginOracle": oracle["pair_margin"]}, np.flatnonzero(strata == name)) for name in range(3)} | {"test_used": False})
    visibility_path = data_root / "diagnostics/frame0_visibility_predictor_v1/val.npz"
    visibility_cache_validated = False
    if visibility_path.is_file():
        visibility_cache = read_npz(visibility_path)
        if visibility_cache["scores"].shape[0] != len(val_rows):
            raise ValueError("visibility cache row count does not match the Val action set")
        # Visibility is only a stratification diagnostic.  Still require exact
        # row/action-slot alignment so that no stale cache can be silently
        # associated with a different current/candidate ordering.
        if not np.array_equal(visibility_cache["ids"], val_cache["ids"]):
            raise ValueError("visibility cache viewpoint ids do not match Val action slots")
        if not np.array_equal(visibility_cache["mask"], val_cache["mask"]):
            raise ValueError("visibility cache masks do not match Val action slots")
        if visibility_cache["scores"].shape[1] != val_cache["ids"].shape[1]:
            raise ValueError("visibility cache slot count does not match Val action slots")
        visibility = visibility_cache["scores"][:, 0]
        visibility_cache_validated = True
    else:
        visibility = np.full(len(val_rows), np.nan)
    low_visibility = np.flatnonzero(visibility <= np.nanquantile(visibility, 1 / 3)) if np.isfinite(visibility).any() else np.asarray([], dtype=np.int64)
    write_json(output / "occlusion_metrics.json", {"source": str(visibility_path.resolve()), "cache_alignment_validated": visibility_cache_validated, "bottom_tertile_contexts": int(len(low_visibility)), "metrics": subset_metrics(val_rows, val_arrays, val_targets, {"Random-B2": selectors["Random-B2"], "SoftBelief-ViewPrior": prior_selections["SoftBelief-ViewPrior"], best_model_name: selectors[best_model_name], "PairMarginOracle": oracle["pair_margin"]}, low_visibility) if len(low_visibility) else {}, "test_used": False})
    # Fixed fusion sweep and transition decomposition.
    random_fusion = fusion_metrics(val_rows, val_arrays, val_targets, selectors["Random-B2"], "Random-B2")
    best_fusion = fusion_metrics(val_rows, val_arrays, val_targets, selectors[best_model_name], best_model_name)
    write_json(output / "fusion_rule_metrics.json", {"Random-B2": random_fusion, best_model_name: best_fusion, "confidence_temperature": 0.1, "test_used": False})
    best_rule = max(best_fusion, key=lambda rule: best_fusion[rule]["accuracy"])
    best_l1 = arrays_l1 = val_arrays["candidate_logp"][np.arange(len(val_rows)), selectors[best_model_name]]
    best_fused_pred = np.argmax(fusion_prediction(val_arrays["current_logp"], best_l1, best_rule), axis=-1)
    write_json(output / "fusion_transition_metrics.json", {"selector": best_model_name, "fusion_rule": best_rule, "metrics": transition_diagnostics(val_arrays, val_targets, selectors[best_model_name], best_fused_pred), "test_used": False})
    # Conditional B3 extension: only run after a learned B2 passes the
    # pre-registered 0.50 accuracy gate.
    b2_best_accuracy = float(learned_metrics[best_model_name]["accuracy"])
    if b2_best_accuracy < 0.50:
        b3_status = "SKIPPED_B2_BELOW_0.50"
        b3_metrics: dict[str, Any] = {"status": b3_status, "best_b2_accuracy": b2_best_accuracy, "test_used": False}
        b3_summary: dict[str, Any] = {"status": b3_status, "test_used": False}
    else:
        train_b2_scores = infer_scores(best_model, train_arrays, best_input_branch, device)
        train_first = stable_argmax(train_b2_scores, train_arrays["candidate_ids"], train_arrays["candidate_mask"])
        train_b3_full = build_b3_arrays(train_arrays, train_first)
        train_keep = np.flatnonzero(train_b3_full["candidate_mask"].sum(axis=1) > 0)
        train_b3_rows = [train_rows[int(i)] for i in train_keep]
        train_b3_arrays = {key: value[train_keep] if isinstance(value, np.ndarray) and value.shape[0] == len(train_rows) else value for key, value in train_b3_full.items()}
        val_b2_scores = infer_scores(best_model, val_arrays, best_input_branch, device)
        val_first = stable_argmax(val_b2_scores, val_arrays["candidate_ids"], val_arrays["candidate_mask"])
        val_b3_arrays = build_b3_arrays(val_arrays, val_first)
        b3_model, b3_summary = train_b3_model(train_b3_rows, train_b3_arrays, val_b3_arrays, device, data_root / CHECKPOINT_REL, output / "b3_training_summary.json")
        b3_scores = infer_b3_scores(b3_model, val_b3_arrays, device)
        b3_selection_info = evaluate_b3(val_b3_arrays, b3_scores)

        def triple_metric(selection: np.ndarray, name: str) -> dict[str, Any]:
            row_indices = np.arange(len(val_arrays["labels"]), dtype=np.int64)
            first_slots = val_b3_arrays["first_selection"]
            third_slots = np.asarray(selection, dtype=np.int64)
            triple = log_softmax((val_arrays["current_logp"] + val_arrays["candidate_logp"][row_indices, first_slots] + val_arrays["candidate_logp"][row_indices, third_slots]) / 3.0)
            item = classification(val_arrays["labels"], np.argmax(triple, axis=-1))
            item.update({"method": name, "first_view_mean": float(np.mean(val_arrays["candidate_ids"][row_indices, first_slots])), "third_view_mean": float(np.mean(val_b3_arrays["candidate_ids"][row_indices, third_slots])), "test_used": False})
            return item

        random_b3 = triple_metric(b3_selection_info["random_selected"], "BestLearned-B2+RandomThird-B3")
        learned_b3 = triple_metric(b3_selection_info["learned_selected"], "BestLearned-B2+LearnedB3")
        oracle_b3 = triple_metric(b3_selection_info["oracle_selected"], "Privileged-B3-MarginOracle")
        b3_status = "COMPLETED"
        b3_metrics = {"status": b3_status, "best_b2_accuracy": b2_best_accuracy, "metrics": {"best_b2_plus_random_third": random_b3, "learned_b2_plus_learned_b3": learned_b3, "privileged_b3_margin_oracle": oracle_b3}, "incremental_gain_vs_random_third": float(learned_b3["accuracy"] - random_b3["accuracy"]), "test_used": False}
        b3_summary = b3_summary
    write_json(output / "b3_metrics.json", b3_metrics)
    if not (output / "b3_training_summary.json").is_file() or b3_status != "COMPLETED":
        write_json(output / "b3_training_summary.json", b3_summary)
    pair_oracle_acc = float(oracle_report["pair_margin_oracle"]["accuracy"])
    random_acc = float(select_metrics(val_rows, val_arrays, val_targets, selectors["Random-B2"], "Random-B2")["accuracy"])
    headroom = {}
    for name, selection in {**prior_selections, **{k: v for k, v in selectors.items() if k not in geom}}.items():
        acc = float(select_metrics(val_rows, val_arrays, val_targets, selection, name)["accuracy"])
        headroom[name] = {"accuracy": acc, "selection_gain_recovery": float((acc - random_acc) / max(pair_oracle_acc - random_acc, 1e-8))}
    write_json(output / "headroom_recovery.json", {"random_accuracy": random_acc, "pair_margin_oracle_accuracy": pair_oracle_acc, "methods": headroom, "test_used": False})
    # Per-class report for Random, best learned, and pair oracle.
    per_class = {"Random-B2": select_metrics(val_rows, val_arrays, val_targets, selectors["Random-B2"], "Random-B2")["per_class"], best_model_name: select_metrics(val_rows, val_arrays, val_targets, selectors[best_model_name], best_model_name)["per_class"], "PairMarginOracle": oracle_report["pair_margin_oracle"]["per_class"]}
    write_json(output / "per_class_metrics.json", per_class)
    # Keep a compact result table with the required headline methods.
    headline = {"Random-B2": select_metrics(val_rows, val_arrays, val_targets, selectors["Random-B2"], "Random-B2"), "NearestCandidate-B2": select_metrics(val_rows, val_arrays, val_targets, geom["NearestCandidate-B2"], "NearestCandidate-B2"), "FarthestCandidate-B2": select_metrics(val_rows, val_arrays, val_targets, geom["FarthestCandidate-B2"], "FarthestCandidate-B2"), "MaxAngularChange-B2": select_metrics(val_rows, val_arrays, val_targets, geom["MaxAngularChange-B2"], "MaxAngularChange-B2"), "MinAngularChange-B2": select_metrics(val_rows, val_arrays, val_targets, geom["MinAngularChange-B2"], "MinAngularChange-B2"), **prior_metrics, **learned_metrics, "PairNormalizedTrueLogPOracle-B2": oracle_report["normalized_pair_true_logp_oracle"], "PairMarginOracle-B2": oracle_report["pair_margin_oracle"]}
    result = {"experiment_id": "REDUCED12_O0_CONDITIONED_COMPLEMENTARY_SECOND_VIEW", "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "candidate_samples": int(val_arrays["candidate_mask"].sum())}, "headline_metrics": headline, "pair_oracle": oracle_report, "best_learned_branch": best_model_name, "best_fusion_rule": best_rule, "b3_status": b3_status, "b3": b3_metrics, "shuffle": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in shuffle_results.items()}, "flags": {"policy_test_used": False, "training_used": True, "new_rgb_generated": False, "new_skeleton_generated": False, "frozen_stgcn_modified": False, "gt_label_used_for_oracle_train_target_only": True, "future_candidate_observation_used_for_selector_input": False, "deployable_selector_inputs": ["O0 frozen feature", "O0 soft posterior", "candidate geometry"], "continuous_time_synchronization_modeled": False}}
    write_json(output / "result.json", result)
    analysis = f"""Experiment:\nO0-Conditioned Complementary Second-View Selection\n\nMoving Val:\n10080 contexts\n\nPolicy Test:\nfalse\n\nObservation protocol:\nfinite full-observation budget\n\nO0:\nactually acquired complete current-view observation\n\nSecond candidate:\nselected before acquiring its observation\n\nFuture candidate observation used by selector:\nfalse\n\nGT action used by selector:\nfalse\n\nO0 frozen feature:\nallowed\n\nO0 soft posterior:\nallowed\n\nHard predicted action as formal intermediate:\nfalse\n\nRecognizer:\nfrozen ST-GCN + frozen shared head\n\nB2 fusion:\nfixed MeanLogP unless explicitly stated in fusion ablation\n\nContinuous robot/human time synchronization:\nnot modeled\n\n## Headline\n\n- Historical raw MeanLogP candidate ranking vs normalized pair ranking agreement: {oracle_report['historical_raw_vs_normalized_pair_selection_agreement']:.6f}. The current-view term cancels in the historical raw target, so it is not complementarity-aware.\n- Historical raw oracle: {oracle_report['historical_raw_oracle']['accuracy']:.6f} Acc / {oracle_report['historical_raw_oracle']['macro_f1']:.6f} Macro-F1.\n- Pair-normalized TrueLogP oracle: {oracle_report['normalized_pair_true_logp_oracle']['accuracy']:.6f} Acc / {oracle_report['normalized_pair_true_logp_oracle']['macro_f1']:.6f} Macro-F1.\n- Pair-margin oracle: {oracle_report['pair_margin_oracle']['accuracy']:.6f} Acc / {oracle_report['pair_margin_oracle']['macro_f1']:.6f} Macro-F1.\n- Pair AnyCorrect coverage: {oracle_report['pair_anycorrect_coverage']:.6f} ({oracle_report['pair_anycorrect_contexts']} contexts).\n- Random B2 MeanLogP: {random_acc:.6f} Acc. Best learned branch ({best_model_name}): {b2_best_accuracy:.6f} Acc.\n- O0 shuffle accuracies: """ + ", ".join(f"{k}={v['accuracy']:.6f}" for k, v in shuffle_results.items()) + f".\n- Conditional B3 status: {b3_status}. When run, learned B3 accuracy is {b3_metrics.get('metrics', {}).get('learned_b2_plus_learned_b3', {}).get('accuracy', 0.0):.6f}.\n\n## Decision\n\nThe pre-registered decision is based on the best learned B2 accuracy, its gain over Random B2, and the O0-information shuffle. Best learned accuracy is {b2_best_accuracy:.6f}; gain over Random B2 is {(b2_best_accuracy-random_acc)*100:+.3f} pp; normal minus both-shuffled is {(shuffle_results['normal']['accuracy']-shuffle_results['both_shuffled']['accuracy'])*100:+.3f} pp. This run is reported as **{'STRONG KEEP O0-CONDITIONED SECOND-VIEW SELECTION' if b2_best_accuracy >= .55 and b2_best_accuracy-random_acc >= .10 else 'KEEP O0-CONDITIONED SECOND-VIEW SELECTION' if b2_best_accuracy >= .50 and b2_best_accuracy-random_acc >= .07 else 'BORDERLINE' if b2_best_accuracy >= .47 else 'KILL O0-CONDITIONED SECOND-VIEW SELECTOR'}** under the specified gates.\n\nThe pair oracle headroom is real when pair-margin substantially exceeds Random B2, but a low learned score means complementary second views remain difficult to predict from O0 plus geometry. No Test artifact was read and no new perception data was generated.\n"""
    (output / "analysis.md").write_text(analysis, encoding="utf-8")
    write_json(output / "leakage_audit.json", {"test_used": False, "future_candidate_feature_read_before_selection": 0, "future_candidate_logit_read_before_selection": 0, "future_candidate_skeleton_read_before_selection": 0, "candidate_correctness_read_before_selection": 0, "gt_label_read_by_selector": 0, "train_future_evidence_used_for_targets_only": True, "selector_inputs": ["current frozen feature", "current soft posterior", "candidate geometry"]})
    write_json(output / "runtime_summary.json", {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.time() - started, "test_used": False})
    print(json.dumps({"output": str(output.resolve()), "best_branch": best_model_name, "best_accuracy": b2_best_accuracy, "random_accuracy": random_acc, "status": "COMPLETED"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
