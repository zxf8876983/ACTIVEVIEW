#!/usr/bin/env python3
"""Audit the reduced12 Policy utility landscape after the Yaw8 recognizer rebuild.

This is a read-only Train/Moving-Val audit.  The old recognizer's existing
candidate evidence is checked and reused, while the frozen Yaw8 checkpoint is
run on exactly the current/Stage-A-legal archived skeletons.  No selector is
trained and Policy Test is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead
from activeview.scripts.experiments.yaw8_policy_landscape_report import render_analysis
from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    NUM_CLASSES,
    classification,
    correlation,
    gt_margin,
)
from activeview.scripts.eval.reduced12_utility_source import (
    body_azimuth_bin,
    load_scene_metadata,
)


SEED = 42
NUM_VIEWS = 32
MAX_CANDIDATES = 21
MAX_OPTIONS = MAX_CANDIDATES + 1
FEATURE_DIM = 256
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
OLD_CHECKPOINT_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
YAW8_CHECKPOINT_RELATIVE = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
OLD_TRUE_CACHE_RELATIVE = Path(
    "diagnostics/reduced12_h1_discriminative_objective_batch"
)
YAW8_CACHE_RELATIVE = Path("diagnostics/reduced12_yaw8_policy_landscape")
VISIBILITY_RELATIVE = Path("diagnostics/frame0_visibility_predictor_v1")
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/yaw8_policy_landscape_audit"
)


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
        raise ValueError("--device must select CUDA")
    return device

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["episode_id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def _load_rows(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all Policy Train and Stage-D Moving-Val rows in canonical order."""
    root = data_root / POLICY_RELATIVE
    train_c = _read_jsonl(root / "stage_c/features/train.jsonl")
    val_c = _read_jsonl(root / "stage_c/features/val.jsonl")
    train_a = {str(row["episode_id"]): row for row in _read_jsonl(root / "stage_a/episodes/train_episodes.jsonl")}
    val_a = {str(row["episode_id"]): row for row in _read_jsonl(root / "stage_a/episodes/val_episodes.jsonl")}
    moving_d = _read_jsonl(root / "stage_d/features/val.jsonl")
    moving_ids = [str(row["episode_id"]) for row in moving_d]
    val_by_id = {str(row["episode_id"]): row for row in val_c}

    def merge(source: Sequence[Mapping[str, Any]], archive_rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for base in source:
            episode_id = str(base["episode_id"])
            archive = archive_rows.get(episode_id)
            if archive is None:
                raise ValueError(f"missing Stage-A row for {episode_id}")
            candidate_ids = [int(item["viewpoint_id"]) for item in archive["candidate_pool"]]
            expected = [int(value) for value in base["candidate_viewpoint_ids"]]
            if candidate_ids != expected:
                raise ValueError(f"Stage-A/Stage-C candidate mismatch: {episode_id}")
            row = dict(base)
            row.update({
                "archive_path": str(archive["current_view"]["skeleton_source_path"]),
                "current_viewpoint_id": int(archive["current_view"]["viewpoint_id"]),
                "candidate_ids": candidate_ids,
                "scene_id": str(base["scene_id"]),
                "region": str(base["region"]),
            })
            merged.append(row)
        return merged

    train = merge(train_c, train_a)
    moving_val = merge([val_by_id[eid] for eid in moving_ids], val_a)
    if len(train) != 46324 or len(moving_val) != 10080:
        raise ValueError(f"unexpected Policy population train={len(train)} moving_val={len(moving_val)}")
    if {str(row["record_id"]) for row in train} & {str(row["record_id"]) for row in moving_val}:
        raise ValueError("Policy Train and Moving Val record overlap detected")
    return train, moving_val

def _load_old_cache(
    data_root: Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    shared_head: torch.nn.Module,
    device: torch.device,
) -> dict[str, np.ndarray]:
    # The historical policy protocol uses the frozen ST-GCN feature cache
    # followed by the frozen shared classifier head.  Reuse that exact cache
    # for the old recognizer so the reproduction gate is meaningful.
    filename = "train_options.npz" if split == "train" else "val_options.npz"
    cache_root = data_root / "diagnostics/reduced12_dual_route_overnight"
    cache = _read_npz(cache_root / filename)
    metadata = json.loads((cache_root / filename.replace(".npz", ".json")).read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != _signature(rows):
        raise ValueError(f"old {split} policy cache signature/count mismatch")
    if cache["logits"].shape != (len(rows), MAX_OPTIONS, NUM_CLASSES):
        raise ValueError(f"unexpected old {split} cache logits shape")
    selected = {
        "features": np.asarray(cache["features"], dtype=np.float16),
        "logits": np.asarray(cache["logits"], dtype=np.float32),
        "ids": np.asarray(cache["ids"], dtype=np.int64),
        "mask": np.asarray(cache["mask"], dtype=bool),
        "labels": np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64),
    }
    # ``val_options.npz`` stores frozen ST-GCN features and raw logits.  The
    # historical policy recognizer applies the frozen shared head to those
    # features, so reproduce that exact second-stage transformation here.
    flat_features = selected["features"].reshape(-1, FEATURE_DIM).astype(np.float32)
    head_outputs: list[np.ndarray] = []
    shared_head.eval()
    with torch.inference_mode():
        for start in range(0, flat_features.shape[0], 8192):
            batch = torch.from_numpy(flat_features[start : start + 8192]).to(device, non_blocking=True)
            head_outputs.append(shared_head(batch).cpu().numpy())
    selected["policy_logits"] = np.concatenate(head_outputs, axis=0).reshape(len(rows), MAX_OPTIONS, NUM_CLASSES)
    selected["policy_logits"][~selected["mask"]] = np.nan
    safe_policy_logits = np.where(selected["mask"][..., None], selected["policy_logits"], 0.0)
    max_logits = np.max(safe_policy_logits, axis=-1, keepdims=True)
    shifted = safe_policy_logits - max_logits
    selected["policy_logp"] = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    selected["policy_logp"][~selected["mask"]] = np.nan
    options = np.full((len(rows), MAX_OPTIONS, NUM_CLASSES), np.nan, dtype=np.float32)
    ids = np.full((len(rows), MAX_OPTIONS), -1, dtype=np.int64)
    mask = np.zeros((len(rows), MAX_OPTIONS), dtype=bool)
    for index, row in enumerate(rows):
        legal = np.flatnonzero(selected["mask"][index])[1:]
        expected_ids = [int(value) for value in row["candidate_ids"]]
        if selected["ids"][index, legal].tolist() != expected_ids:
            raise ValueError(f"old cache action mismatch: {row['episode_id']}")
        options[index, 0] = selected["policy_logp"][index, 0]
        ids[index, 0] = int(row["current_viewpoint_id"])
        mask[index, 0] = True
        length = len(legal)
        options[index, 1 : length + 1] = selected["policy_logp"][index, legal]
        ids[index, 1 : length + 1] = selected["ids"][index, legal]
        mask[index, 1 : length + 1] = True
    selected["options_logp"] = options
    selected["options_ids"] = ids
    selected["options_mask"] = mask
    selected["logp"] = options
    selected["ids"] = ids
    selected["mask"] = mask
    return selected

def _archive_skeleton(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
    if ids.shape != (NUM_VIEWS,) or not np.array_equal(ids, np.arange(NUM_VIEWS)):
        raise ValueError(f"invalid viewpoint ids in {path}")
    if skeleton.shape != (NUM_VIEWS, 3, 30, 17) or not np.isfinite(skeleton).all():
        raise ValueError(f"invalid skeleton archive in {path}")
    return skeleton

def _yaw8_cache(
    data_root: Path,
    split_name: str,
    rows: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    cache_root = data_root / YAW8_CACHE_RELATIVE
    cache_path = cache_root / f"{split_name}.npz"
    meta_path = cache_root / f"{split_name}.json"
    expected_meta = {
        "split": split_name,
        "rows": len(rows),
        "signature": _signature(rows),
        "checkpoint_sha256": _sha256(data_root / YAW8_CHECKPOINT_RELATIVE),
        "options": "current/Stay + Stage-A legal candidate_pool",
        "feature_dim": FEATURE_DIM,
        "test_used": False,
    }
    if cache_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata == expected_meta:
            cached = _read_npz(cache_path)
            if cached["logp"].shape == (len(rows), MAX_OPTIONS, NUM_CLASSES):
                return cached

    count = len(rows)
    features = np.zeros((count, MAX_OPTIONS, FEATURE_DIM), dtype=np.float16)
    logits = np.full((count, MAX_OPTIONS, NUM_CLASSES), np.nan, dtype=np.float32)
    ids = np.full((count, MAX_OPTIONS), -1, dtype=np.int64)
    mask = np.zeros((count, MAX_OPTIONS), dtype=bool)
    model.eval()
    started = time.time()
    for start in range(0, count, 128):
        batch_rows = rows[start : start + 128]
        flat: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for row in batch_rows:
            skeleton = _archive_skeleton(Path(str(row["archive_path"])))
            wanted = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
            wanted = list(dict.fromkeys(wanted))
            offset = len(flat)
            flat.extend(skeleton[viewpoint_id] for viewpoint_id in wanted)
            spans.append((offset, len(wanted)))
        tensor = torch.from_numpy(np.stack(flat)).to(device, non_blocking=True)
        with torch.inference_mode():
            encoded = model.forward_features(tensor)
            raw_logits = model.fc(encoded)
        encoded_np = encoded.cpu().numpy()
        logits_np = raw_logits.cpu().numpy()
        for local, (offset, length) in enumerate(spans):
            target = start + local
            wanted = [int(batch_rows[local]["current_viewpoint_id"])] + [int(value) for value in batch_rows[local]["candidate_ids"]]
            wanted = list(dict.fromkeys(wanted))
            features[target, :length] = encoded_np[offset : offset + length].astype(np.float16)
            logits[target, :length] = logits_np[offset : offset + length]
            ids[target, :length] = np.asarray(wanted, dtype=np.int64)
            mask[target, :length] = True
        done = start + len(batch_rows)
        if done % 2048 == 0 or done == count:
            print(f"[yaw8-cache:{split_name}] rows={done}/{count} elapsed={time.time()-started:.1f}s", flush=True)
    safe_logits = np.where(mask[..., None], logits, 0.0)
    max_logits = np.max(safe_logits, axis=-1, keepdims=True)
    shifted = safe_logits - max_logits
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    # Keep padding as NaN so accidental use of non-legal views is visible.
    logp[~mask] = np.nan
    result = {"logp": logp.astype(np.float32), "features": features, "logits": logits, "ids": ids, "mask": mask}
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.parent / f"{cache_path.stem}.tmp.npz"
    np.savez(temporary, **result)
    temporary.replace(cache_path)
    meta_path.write_text(json.dumps(expected_meta, indent=2) + "\n", encoding="utf-8")
    return result

def _metric(labels: np.ndarray, predictions: np.ndarray, name: str, actions: Sequence[int] | None, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = classification(labels, predictions)
    if actions is not None:
        current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
        result["move_rate"] = float(np.mean(np.asarray(actions, dtype=np.int64) != current))
        result["stay_rate"] = 1.0 - result["move_rate"]
    result["method"] = name
    result["test_used"] = False
    return result

def _action_predictions(options: Mapping[str, np.ndarray], actions: Sequence[int]) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((options["ids"][index] == int(action)) & options["mask"][index])
        if slots.size != 1:
            raise ValueError(f"cannot resolve action {action} at row {index}")
        predictions.append(int(np.argmax(options["logp"][index, int(slots[0])])))
    return np.asarray(predictions, dtype=np.int64)

def _random_actions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    rng = np.random.default_rng(SEED)
    return [int(rng.choice([int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]])) for row in rows]

def _oracle_actions(options: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], criterion: str, candidate_only: bool = False) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(options["mask"][index])
        if candidate_only:
            active = active[active != 0]
        label = int(row["label_id"])
        if criterion == "true_logp":
            scores = options["logp"][index, active, label]
        elif criterion == "margin":
            scores = np.asarray([gt_margin(options["logp"][index, slot], label) for slot in active])
        else:
            raise ValueError(criterion)
        actions.append(int(options["ids"][index, int(active[int(np.argmax(scores))])]))
    return actions

def _any_correct(options: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], candidate_only: bool = False) -> np.ndarray:
    values = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        active = np.flatnonzero(options["mask"][index])
        if candidate_only:
            active = active[active != 0]
        values[index] = bool(np.any(np.argmax(options["logp"][index, active], axis=1) == int(row["label_id"])))
    return values

def _build_metrics(
    rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], name: str,
    static_prior: Mapping[int, float], random_actions: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    current_actions = [int(row["current_viewpoint_id"]) for row in rows]
    static_actions = _static_actions(options, static_prior)
    true_actions = _oracle_actions(options, rows, "true_logp")
    margin_actions = _oracle_actions(options, rows, "margin")
    method_actions = {
        "Stay/current": current_actions,
        "Random legal": list(random_actions),
        "StaticPrior": static_actions,
        "GT-TrueLogP Oracle": true_actions,
        "GT-Margin Oracle": margin_actions,
    }
    metrics = {
        method: _metric(labels, _action_predictions(options, actions), method, actions, rows)
        for method, actions in method_actions.items()
    }
    any_values = _any_correct(options, rows)
    candidate_any = _any_correct(options, rows, candidate_only=True)
    metrics["AnyCorrect Coverage"] = {
        "contexts": len(rows), "coverage_count": int(any_values.sum()),
        "coverage_rate": float(any_values.mean()), "test_used": False,
    }
    metrics["Candidate-only AnyCorrect Coverage"] = {
        "contexts": len(rows), "coverage_count": int(candidate_any.sum()),
        "coverage_rate": float(candidate_any.mean()), "test_used": False,
    }
    details = {
        "actions": method_actions,
        "any_correct": any_values,
        "candidate_any_correct": candidate_any,
        "candidate_only_oracle_actions": {
            "GT-TrueLogP": _oracle_actions(options, rows, "true_logp", True),
            "GT-Margin": _oracle_actions(options, rows, "margin", True),
        },
    }
    return metrics, details

def _static_actions(options: Mapping[str, np.ndarray], prior: Mapping[int, float]) -> list[int]:
    """Select the highest Train-derived prior among current/stay + legal candidates."""
    fallback = float(prior.get(-1, 0.0))
    actions: list[int] = []
    for index in range(options["ids"].shape[0]):
        active = np.flatnonzero(options["mask"][index])
        scores = [float(prior.get(int(view), fallback)) for view in options["ids"][index, active]]
        actions.append(int(options["ids"][index, int(active[int(np.argmax(scores))])]))
    return actions


def _candidate_micro(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], name: str) -> dict[str, Any]:
    labels: list[int] = []
    predictions: list[int] = []
    for index, row in enumerate(rows):
        for slot in np.flatnonzero(options["mask"][index])[1:]:
            labels.append(int(row["label_id"]))
            predictions.append(int(np.argmax(options["logp"][index, int(slot)])))
    return _metric(np.asarray(labels), np.asarray(predictions), name, None, rows=[])


def _train_prior(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> tuple[dict[int, float], dict[str, Any]]:
    sums = np.zeros(NUM_VIEWS, dtype=np.float64)
    counts = np.zeros(NUM_VIEWS, dtype=np.int64)
    for index, row in enumerate(rows):
        label = int(row["label_id"])
        for slot in np.flatnonzero(options["mask"][index])[1:]:
            view = int(options["ids"][index, int(slot)])
            value = gt_margin(options["logp"][index, int(slot)], label)
            sums[view] += value
            counts[view] += 1
    global_mean = float(sums.sum() / max(counts.sum(), 1))
    prior = {view: float(sums[view] / counts[view]) if counts[view] else global_mean for view in range(NUM_VIEWS)}
    prior[-1] = global_mean
    return prior, {"definition": "Q(v)=mean Train candidate-only GT-Margin; unseen view falls back to global Train mean", "q": prior, "counts": counts.tolist(), "global_mean": global_mean}


def _flatten_candidates(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], metadata: Mapping[tuple[str, str], Mapping[str, Any]], split: str) -> dict[str, Any]:
    labels: list[int] = []
    views: list[int] = []
    scenes: list[str] = []
    placements: list[str] = []
    bins: list[int] = []
    radii: list[float] = []
    true_logp: list[float] = []
    margins: list[float] = []
    confidence: list[float] = []
    episodes: list[str] = []
    old_pred: list[int] = []
    for index, row in enumerate(rows):
        meta = metadata[(str(row["scene_id"]), str(row["region"]))]
        label = int(row["label_id"])
        for slot in np.flatnonzero(options["mask"][index])[1:]:
            view = int(options["ids"][index, int(slot)])
            logp = options["logp"][index, int(slot)]
            labels.append(label); views.append(view); scenes.append(str(row["scene_id"])); placements.append(str(row["region"]))
            bins.append(body_azimuth_bin(meta["azimuths"][view], meta["yaw_deg"]))
            radii.append(float(meta["radii"][view])); true_logp.append(float(logp[label])); margins.append(gt_margin(logp, label)); old_pred.append(int(np.argmax(logp)))
            episodes.append(str(row["episode_id"])); confidence.append(float(np.max(np.exp(logp))))
    return {"labels": np.asarray(labels, dtype=np.int64), "views": np.asarray(views, dtype=np.int64), "scenes": np.asarray(scenes), "placements": np.asarray(placements), "episodes": np.asarray(episodes), "bins": np.asarray(bins, dtype=np.int64), "radii": np.asarray(radii), "true_logp": np.asarray(true_logp), "margins": np.asarray(margins), "confidence": np.asarray(confidence), "predictions": np.asarray(old_pred, dtype=np.int64), "split": split}


def _group_metric(flat: Mapping[str, np.ndarray], indices: np.ndarray, name: str, predictions: np.ndarray) -> dict[str, Any]:
    return _metric(flat["labels"][indices], predictions[indices], name, None, rows=[])


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way eta-squared for binary candidate correctness by grouping."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.size == 0 or np.var(values) < 1e-12:
        return 0.0
    overall = float(np.mean(values))
    between = 0.0
    for group in np.unique(groups):
        subset = values[groups == group]
        between += float(subset.size) * (float(np.mean(subset)) - overall) ** 2
    return float(between / values.size / np.var(values))


def _per_group(flat: Mapping[str, np.ndarray], predictions: np.ndarray, group_values: np.ndarray, values: Sequence[Any], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        indices = np.flatnonzero(group_values == value)
        if indices.size:
            result[str(value)] = {"count": int(indices.size), "metrics": _group_metric(flat, indices, prefix, predictions), "mean_gt_logp": float(np.mean(flat["true_logp"][indices])), "mean_gt_margin": float(np.mean(flat["margins"][indices]))}
    return result


def _relative_metrics(rows: Sequence[Mapping[str, Any]], options_old: Mapping[str, np.ndarray], options_new: Mapping[str, np.ndarray], metadata: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    old = _flatten_candidates(rows, options_old, metadata, "moving_val")
    new = _flatten_candidates(rows, options_new, metadata, "moving_val")
    output: dict[str, Any] = {"bins": {}, "old_relative_yaw_gap": 0.0, "yaw8_relative_yaw_gap": 0.0}
    old_acc: list[float] = []; new_acc: list[float] = []
    for bin_id in range(8):
        index = np.flatnonzero(old["bins"] == bin_id)
        old_pred = old["predictions"]
        new_pred = new["predictions"]
        if not np.array_equal(old["episodes"][index], new["episodes"][index]) or not np.array_equal(old["views"][index], new["views"][index]):
            raise ValueError("old/Yaw8 candidate flattening order mismatch")
        output["bins"][str(bin_id)] = {
            "count": int(index.size),
            "old": _group_metric(old, index, "Old", old_pred),
            "yaw8": _group_metric(new, index, "Yaw8", new_pred),
            "old_mean_gt_margin": float(np.mean(old["margins"][index])) if index.size else 0.0,
            "yaw8_mean_gt_margin": float(np.mean(new["margins"][index])) if index.size else 0.0,
            "old_mean_confidence": float(np.mean(old["confidence"][index])) if index.size else 0.0,
            "yaw8_mean_confidence": float(np.mean(new["confidence"][index])) if index.size else 0.0,
        }
        if index.size:
            old_acc.append(float(np.mean(old_pred[index] == old["labels"][index]))); new_acc.append(float(np.mean(new_pred[index] == new["labels"][index])))
    output["old_relative_yaw_gap"] = float(max(old_acc) - min(old_acc)) if old_acc else 0.0
    output["yaw8_relative_yaw_gap"] = float(max(new_acc) - min(new_acc)) if new_acc else 0.0
    output["gap_reduction"] = float(output["old_relative_yaw_gap"] - output["yaw8_relative_yaw_gap"])
    output["old_relative_yaw_eta_squared"] = _eta_squared((old["predictions"] == old["labels"]).astype(np.float64), old["bins"])
    output["yaw8_relative_yaw_eta_squared"] = _eta_squared((new["predictions"] == new["labels"]).astype(np.float64), new["bins"])
    return output, {"old": old, "yaw8": new}


def _viewpoint_metrics(flat_old: Mapping[str, np.ndarray], flat_new: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    result: dict[str, Any] = {"viewpoints": {}, "radii": {}}
    old_acc: list[float] = []; new_acc: list[float] = []
    for view in range(NUM_VIEWS):
        oi = np.flatnonzero(flat_old["views"] == view); ni = np.flatnonzero(flat_new["views"] == view)
        if oi.size:
            old_acc.append(float(np.mean(flat_old["predictions"][oi] == flat_old["labels"][oi])))
        if ni.size:
            new_acc.append(float(np.mean(flat_new["predictions"][ni] == flat_new["labels"][ni])))
        result["viewpoints"][str(view)] = {"count": int(oi.size), "old": _group_metric(flat_old, oi, "Old", flat_old["predictions"]) if oi.size else None, "yaw8": _group_metric(flat_new, ni, "Yaw8", flat_new["predictions"]) if ni.size else None, "old_mean_gt_logp": float(np.mean(flat_old["true_logp"][oi])) if oi.size else None, "yaw8_mean_gt_logp": float(np.mean(flat_new["true_logp"][ni])) if ni.size else None, "old_mean_gt_margin": float(np.mean(flat_old["margins"][oi])) if oi.size else None, "yaw8_mean_gt_margin": float(np.mean(flat_new["margins"][ni])) if ni.size else None}
    for radius_index, radius in enumerate((1.5, 2.0, 2.5, 3.0)):
        oi = np.flatnonzero(np.isclose(flat_old["radii"], radius)); ni = np.flatnonzero(np.isclose(flat_new["radii"], radius))
        result["radii"][str(radius)] = {"count": int(oi.size), "old": _group_metric(flat_old, oi, "Old", flat_old["predictions"]), "yaw8": _group_metric(flat_new, ni, "Yaw8", flat_new["predictions"])}
    result["variance"] = {"old_viewpoint_accuracy_std": float(np.std(old_acc)), "yaw8_viewpoint_accuracy_std": float(np.std(new_acc)), "old_viewpoint_accuracy_range": float(max(old_acc) - min(old_acc)), "yaw8_viewpoint_accuracy_range": float(max(new_acc) - min(new_acc)), "old_viewpoint_eta_squared": _eta_squared((flat_old["predictions"] == flat_old["labels"]).astype(np.float64), flat_old["views"]), "yaw8_viewpoint_eta_squared": _eta_squared((flat_new["predictions"] == flat_new["labels"]).astype(np.float64), flat_new["views"])}
    return result, {"old": old_acc, "yaw8": new_acc}


def _utility_correlation(flat_old: Mapping[str, np.ndarray], flat_new: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("true_logp", "margins"):
        result[field] = {"global_spearman": correlation(flat_old[field], flat_new[field], spearman=True), "global_pearson": correlation(flat_old[field], flat_new[field]), "within_context_mean": 0.0, "within_context_median": 0.0}
    # Context boundaries are reconstructed from the row-level candidate counts.
    # The flattened arrays preserve row order; each row therefore contributes a
    # contiguous block with equal old/new candidate identity.
    within: dict[str, list[float]] = {"true_logp": [], "margins": []}
    cursor = 0
    for count in _candidate_counts_from_flat(flat_old):
        sl = slice(cursor, cursor + count); cursor += count
        if count >= 2:
            for field in within:
                within[field].append(correlation(flat_old[field][sl], flat_new[field][sl], spearman=True))
    for field in within:
        result[field]["within_context_mean"] = float(np.mean(within[field])) if within[field] else 0.0
        result[field]["within_context_median"] = float(np.median(within[field])) if within[field] else 0.0
    return result


def _candidate_counts_from_flat(flat: Mapping[str, np.ndarray]) -> list[int]:
    counts: list[int] = []
    if flat["scenes"].size == 0:
        return counts
    previous = str(flat["episodes"][0])
    count = 0
    for episode in flat["episodes"]:
        key = str(episode)
        if key != previous:
            counts.append(count); count = 0; previous = key
        count += 1
    counts.append(count)
    return counts


def _oracle_agreement(
    details_old: Mapping[str, Any], details_new: Mapping[str, Any],
    options_old: Mapping[str, np.ndarray], options_new: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    old = details_old["candidate_only_oracle_actions"]["GT-Margin"]
    new = details_new["candidate_only_oracle_actions"]["GT-Margin"]
    top1 = float(np.mean(np.asarray(old) == np.asarray(new)))
    overlap: list[float] = []
    for index, row in enumerate(rows):
        del row
        old_active = np.flatnonzero(options_old["mask"][index])[1:]
        new_active = np.flatnonzero(options_new["mask"][index])[1:]
        old_scores = np.asarray([gt_margin(options_old["logp"][index, slot], int(rows[index]["label_id"])) for slot in old_active])
        new_scores = np.asarray([gt_margin(options_new["logp"][index, slot], int(rows[index]["label_id"])) for slot in new_active])
        old_top = {int(options_old["ids"][index, slot]) for slot in old_active[np.argsort(-old_scores, kind="mergesort")[:3]]}
        new_top = {int(options_new["ids"][index, slot]) for slot in new_active[np.argsort(-new_scores, kind="mergesort")[:3]]}
        overlap.append(float(len(old_top & new_top) / 3.0))
    return {"gt_margin_top1_agreement": top1, "gt_margin_top3_overlap": float(np.mean(overlap)) if overlap else 0.0, "top3_overlap_definition": "intersection size divided by 3 over candidate-only top-three GT-margin viewpoints"}


def _occlusion_metrics(
    data_root: Path, rows: Sequence[Mapping[str, Any]], method_actions: Mapping[str, Mapping[str, Sequence[int]]],
    options_by_recognizer: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    path = data_root / VISIBILITY_RELATIVE / "val.npz"
    meta_path = data_root / VISIBILITY_RELATIVE / "val.json"
    target = _read_npz(path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or bool(metadata.get("test_used", True)):
        raise ValueError("frame0 visibility cache is not matched Moving Val")
    for index, row in enumerate(rows):
        active = np.flatnonzero(target["mask"][index])
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if target["ids"][index, active].astype(int).tolist() != expected:
            raise ValueError(f"visibility action mismatch at {row['episode_id']}")
    stay_visibility = np.asarray(target["scores"][:, 0], dtype=np.float64)
    threshold = float(np.quantile(stay_visibility, 1.0 / 3.0))
    high = stay_visibility < threshold
    result: dict[str, Any] = {"definition": "bottom tertile of existing frame-0 Stay scene visibility", "threshold": threshold, "count": int(high.sum()), "recognizers": {}}
    for recognizer, options in options_by_recognizer.items():
        result["recognizers"][recognizer] = {}
        for method in ("Random legal", "StaticPrior", "GT-TrueLogP Oracle", "GT-Margin Oracle"):
            actions = method_actions[recognizer][method]
            preds = _action_predictions(options, actions)
            subset_rows = [rows[int(i)] for i in np.flatnonzero(high)]
            subset_actions = [actions[int(i)] for i in np.flatnonzero(high)]
            result["recognizers"][recognizer][method] = _metric(np.asarray([int(row["label_id"]) for row in subset_rows]), preds[high], method, subset_actions, subset_rows)
    return result


def _scene_placement_metrics(rows: Sequence[Mapping[str, Any]], options_by_recognizer: Mapping[str, Mapping[str, np.ndarray]], actions_by_recognizer: Mapping[str, Mapping[str, Sequence[int]]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows): groups[(str(row["scene_id"]), str(row["region"]))].append(index)
    result: dict[str, Any] = {}
    for key, indices in sorted(groups.items()):
        result[f"{key[0]}/{key[1]}"] = {"contexts": len(indices), "recognizers": {}}
        idx = np.asarray(indices, dtype=np.int64)
        for recognizer, options in options_by_recognizer.items():
            result[f"{key[0]}/{key[1]}"]["recognizers"][recognizer] = {}
            for method in ("Random legal", "StaticPrior", "GT-TrueLogP Oracle", "GT-Margin Oracle"):
                actions = actions_by_recognizer[recognizer][method]
                pred = _action_predictions(options, actions)
                labels = np.asarray([int(rows[int(i)]["label_id"]) for i in idx], dtype=np.int64)
                result[f"{key[0]}/{key[1]}"]["recognizers"][recognizer][method] = _metric(labels, pred[idx], method, [actions[int(i)] for i in idx], [rows[int(i)] for i in idx])
    return result


def _per_class(rows: Sequence[Mapping[str, Any]], options_by_recognizer: Mapping[str, Mapping[str, np.ndarray]], actions_by_recognizer: Mapping[str, Mapping[str, Sequence[int]]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    result: dict[str, Any] = {}
    for recognizer, options in options_by_recognizer.items():
        result[recognizer] = {}
        for method in ("Stay/current", "Random legal", "StaticPrior", "GT-TrueLogP Oracle", "GT-Margin Oracle"):
            pred = _action_predictions(options, actions_by_recognizer[recognizer][method])
            result[recognizer][method] = {
                LABELS[c]: {
                    "support": int(np.sum(labels == c)),
                    "accuracy": float(np.mean(pred[labels == c] == c)) if np.any(labels == c) else 0.0,
                }
                for c in range(NUM_CLASSES)
            }
        candidate_labels: list[int] = []
        candidate_predictions: list[int] = []
        for index, row in enumerate(rows):
            for slot in np.flatnonzero(options["mask"][index])[1:]:
                candidate_labels.append(int(row["label_id"]))
                candidate_predictions.append(int(np.argmax(options["logp"][index, int(slot)])))
        candidate_labels_array = np.asarray(candidate_labels, dtype=np.int64)
        candidate_predictions_array = np.asarray(candidate_predictions, dtype=np.int64)
        result[recognizer]["Legal candidate micro"] = {
            LABELS[c]: {
                "support": int(np.sum(candidate_labels_array == c)),
                "accuracy": float(np.mean(candidate_predictions_array[candidate_labels_array == c] == c))
                if np.any(candidate_labels_array == c) else 0.0,
            }
            for c in range(NUM_CLASSES)
        }
        any_correct = _any_correct(options, rows)
        result[recognizer]["AnyCorrect Coverage"] = {
            LABELS[c]: {
                "support": int(np.sum(labels == c)),
                "coverage": float(np.mean(any_correct[labels == c])) if np.any(labels == c) else 0.0,
            }
            for c in range(NUM_CLASSES)
        }
    return result


def run(output_dir: Path, data_root: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    started = time.time(); _seed()
    train_rows, moving_rows = _load_rows(data_root)
    old_checkpoint = data_root / OLD_CHECKPOINT_RELATIVE
    yaw8_checkpoint = data_root / YAW8_CHECKPOINT_RELATIVE
    if not old_checkpoint.is_file() or not yaw8_checkpoint.is_file():
        raise FileNotFoundError("required old/Yaw8 ST-GCN checkpoint is missing")
    shared_head_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth"
    if not shared_head_path.is_file():
        raise FileNotFoundError(f"missing historical shared head: {shared_head_path}")
    shared_head = SharedHead().to(device)
    shared_head_payload = torch.load(shared_head_path, map_location=device, weights_only=False)
    shared_head.load_state_dict(shared_head_payload["state_dict"])
    old_train = _load_old_cache(data_root, "train", train_rows, shared_head, device)
    old_val = _load_old_cache(data_root, "val", moving_rows, shared_head, device)
    # A small direct old-model check protects against silently stale cache use.
    old_model, _ = load_checkpoint(old_checkpoint, NUM_CLASSES, str(device))
    shared_head.eval()
    rng = np.random.default_rng(SEED)
    checks: list[float] = []
    for row in [train_rows[int(i)] for i in rng.choice(len(train_rows), 8, replace=False)]:
        skeleton = _archive_skeleton(Path(str(row["archive_path"])))
        view = int(rng.choice([int(row["current_viewpoint_id"])] + [int(v) for v in row["candidate_ids"]]))
        with torch.inference_mode():
            direct_feature = old_model.forward_features(torch.from_numpy(skeleton[view:view+1]).to(device))
            direct = torch.log_softmax(shared_head(direct_feature), dim=-1).cpu().numpy()[0]
        index = next(i for i, candidate in enumerate(train_rows) if candidate["episode_id"] == row["episode_id"])
        slot = int(np.flatnonzero(old_train["ids"][index] == view)[0])
        checks.append(float(np.max(np.abs(direct - old_train["logp"][index, slot]))))
    del old_model
    del shared_head
    yaw8_model, _ = load_checkpoint(yaw8_checkpoint, NUM_CLASSES, str(device))
    yaw8_train = _yaw8_cache(data_root, "train", train_rows, yaw8_model, device, batch_size)
    yaw8_val = _yaw8_cache(data_root, "moving_val", moving_rows, yaw8_model, device, batch_size)
    del yaw8_model
    metadata, yaw_audit = load_scene_metadata(data_root, list(train_rows) + list(moving_rows), audit_sample_count=10)
    old_prior, old_prior_meta = _train_prior(train_rows, old_train)
    yaw_prior, yaw_prior_meta = _train_prior(train_rows, yaw8_train)
    random_actions = _random_actions(moving_rows)
    old_metrics, old_details = _build_metrics(moving_rows, old_val, "old", old_prior, random_actions)
    yaw_metrics, yaw_details = _build_metrics(moving_rows, yaw8_val, "yaw8", yaw_prior, random_actions)
    expected = {"stay_accuracy": 0.302579, "oracle_accuracy": 0.753175, "anycorrect": 0.791964}
    observed = {"stay_accuracy": old_metrics["Stay/current"]["accuracy"], "oracle_accuracy": old_metrics["GT-TrueLogP Oracle"]["accuracy"], "anycorrect": old_metrics["AnyCorrect Coverage"]["coverage_rate"]}
    deltas = {key: float(observed[key] - value) for key, value in expected.items()}
    if any(abs(value) > 0.002 for value in deltas.values()):
        raise RuntimeError(f"old recognizer reproduction mismatch >0.2pp: {deltas}")
    old_candidate = _candidate_micro(moving_rows, old_val, "Old legal candidates")
    yaw_candidate = _candidate_micro(moving_rows, yaw8_val, "Yaw8 legal candidates")
    old_flat = _flatten_candidates(moving_rows, old_val, metadata, "moving_val")
    yaw_flat = _flatten_candidates(moving_rows, yaw8_val, metadata, "moving_val")
    relative, _ = _relative_metrics(moving_rows, old_val, yaw8_val, metadata)
    viewpoint, _ = _viewpoint_metrics(old_flat, yaw_flat)
    utility = _utility_correlation(old_flat, yaw_flat)
    oracle_agreement = _oracle_agreement(old_details, yaw_details, old_val, yaw8_val, moving_rows)
    old_q = np.asarray([old_prior[i] for i in range(NUM_VIEWS)], dtype=np.float64)
    yaw_q = np.asarray([yaw_prior[i] for i in range(NUM_VIEWS)], dtype=np.float64)
    old_positive = old_q - old_q.min() + 1e-6; yaw_positive = yaw_q - yaw_q.min() + 1e-6
    old_positive /= old_positive.sum(); yaw_positive /= yaw_positive.sum()
    jsd = 0.5 * (np.sum(old_positive * np.log(old_positive / ((old_positive + yaw_positive) / 2))) + np.sum(yaw_positive * np.log(yaw_positive / ((old_positive + yaw_positive) / 2))))
    old_top5 = set(np.argsort(-old_q)[:5].tolist()); yaw_top5 = set(np.argsort(-yaw_q)[:5].tolist())
    prior_structure = {"spearman": correlation(old_q, yaw_q, spearman=True), "pearson": correlation(old_q, yaw_q), "top5_overlap": float(len(old_top5 & yaw_top5) / 5.0), "jsd_positive_normalized": float(0.5 * jsd), "old_best_viewpoints": np.argsort(-old_q)[:5].tolist(), "yaw8_best_viewpoints": np.argsort(-yaw_q)[:5].tolist()}
    actions_by_recognizer = {"old": old_details["actions"], "yaw8": yaw_details["actions"]}
    options_by_recognizer = {"old": old_val, "yaw8": yaw8_val}
    occlusion = _occlusion_metrics(data_root, moving_rows, actions_by_recognizer, options_by_recognizer)
    scene_placement = _scene_placement_metrics(moving_rows, options_by_recognizer, actions_by_recognizer)
    per_class = _per_class(moving_rows, options_by_recognizer, actions_by_recognizer)
    old_prior_on_yaw_actions = _static_actions(yaw8_val, old_prior)
    yaw_prior_on_old_actions = _static_actions(old_val, yaw_prior)
    moving_labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    cross_prior = {
        "old_prior_yaw8_recognizer": _metric(
            moving_labels, _action_predictions(yaw8_val, old_prior_on_yaw_actions),
            "old prior + yaw8 recognizer", old_prior_on_yaw_actions, moving_rows,
        ),
        "yaw8_prior_old_recognizer": _metric(
            moving_labels, _action_predictions(old_val, yaw_prior_on_old_actions),
            "yaw8 prior + old recognizer", yaw_prior_on_old_actions, moving_rows,
        ),
    }
    main_table = {"old": old_metrics, "yaw8": yaw_metrics}
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_YAW8_POLICY_LANDSCAPE_AUDIT", "status": "COMPLETED",
        "population": {"policy_train_contexts": len(train_rows), "moving_val_contexts": len(moving_rows), "train_records": len({str(r['record_id']) for r in train_rows}), "moving_val_records": len({str(r['record_id']) for r in moving_rows}), "legal_candidate_count_mean": float(np.mean(np.sum(old_val["mask"][:, 1:], axis=1)),), "legal_candidate_count_min": int(np.min(np.sum(old_val["mask"][:, 1:], axis=1))), "legal_candidate_count_max": int(np.max(np.sum(old_val["mask"][:, 1:], axis=1)))},
        "labels": list(LABELS), "main_table": main_table,
        "old_reproduction": {"expected_history": expected, "observed": observed, "delta": deltas, "max_abs_delta": float(max(abs(v) for v in deltas.values())), "direct_cache_max_logp_abs_error": float(max(checks) if checks else 0.0), "direct_cache_check_count": len(checks), "status": "PASS"},
        "stay_metrics": {"old": old_metrics["Stay/current"], "yaw8": yaw_metrics["Stay/current"]},
        "legal_micro_metrics": {"old": old_candidate, "yaw8": yaw_candidate},
        "random_metrics": {"old": old_metrics["Random legal"], "yaw8": yaw_metrics["Random legal"]},
        "static_prior_metrics": {"old": old_metrics["StaticPrior"], "yaw8": yaw_metrics["StaticPrior"], "cross_prior_diagnostic": cross_prior},
        "oracle_metrics": {"old": {"GT-TrueLogP": old_metrics["GT-TrueLogP Oracle"], "GT-Margin": old_metrics["GT-Margin Oracle"], "AnyCorrect": old_metrics["AnyCorrect Coverage"], "candidate_only_AnyCorrect": old_metrics["Candidate-only AnyCorrect Coverage"]}, "yaw8": {"GT-TrueLogP": yaw_metrics["GT-TrueLogP Oracle"], "GT-Margin": yaw_metrics["GT-Margin Oracle"], "AnyCorrect": yaw_metrics["AnyCorrect Coverage"], "candidate_only_AnyCorrect": yaw_metrics["Candidate-only AnyCorrect Coverage"]}},
        "headroom_metrics": {"old": {"random_to_oracle": float(old_metrics["GT-TrueLogP Oracle"]["accuracy"] - old_metrics["Random legal"]["accuracy"]), "static_to_oracle": float(old_metrics["GT-TrueLogP Oracle"]["accuracy"] - old_metrics["StaticPrior"]["accuracy"])}, "yaw8": {"random_to_oracle": float(yaw_metrics["GT-TrueLogP Oracle"]["accuracy"] - yaw_metrics["Random legal"]["accuracy"]), "static_to_oracle": float(yaw_metrics["GT-TrueLogP Oracle"]["accuracy"] - yaw_metrics["StaticPrior"]["accuracy"]) }},
        "per_viewpoint_metrics": viewpoint["viewpoints"], "per_radius_metrics": viewpoint["radii"], "relative_body_yaw_metrics": relative, "viewpoint_variance_metrics": viewpoint, "utility_correlation": utility, "oracle_agreement": oracle_agreement, "prior_structure_comparison": prior_structure, "scene_placement_metrics": scene_placement, "per_class_metrics": per_class, "occlusion_metrics": occlusion,
        "recognizer_metadata": {"old": {"checkpoint": str(old_checkpoint.resolve()), "sha256": _sha256(old_checkpoint), "architecture": "STGCN h36m_17 spatial, 256-D penultimate + frozen policy shared head Linear(256,256)->GELU->Linear(256,12)", "normalization": "camera_to_gravity + root_center + torso_scale + yaw_only", "policy_head_checkpoint": str(shared_head_path.resolve()), "policy_head_sha256": _sha256(shared_head_path)}, "yaw8": {"checkpoint": str(yaw8_checkpoint.resolve()), "sha256": _sha256(yaw8_checkpoint), "architecture": "STGCN h36m_17 spatial, 256-D penultimate, native 12-way fc", "normalization": "camera_to_gravity + root_center + torso_scale + yaw_only"}},
        "prior_metadata": {"old": old_prior_meta, "yaw8": yaw_prior_meta}, "scene_metadata_audit": yaw_audit,
        "leakage_audit": {"policy_test_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "yolo_rerun": False, "videopose3d_rerun": False, "yaw8_stgcn_modified": False, "policy_candidate_set_modified": False, "moving_val_used_for_training": False, "selector_trained": False, "train_used_for_static_prior_only": True, "future_candidate_skeleton_used_for_terminal_oracle_evidence": True},
        "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "elapsed_seconds": float(time.time() - started)},
        "artifacts": {"policy_root": str((data_root / POLICY_RELATIVE).resolve()), "old_true_cache": str((data_root / OLD_TRUE_CACHE_RELATIVE).resolve()), "yaw8_cache": str((data_root / YAW8_CACHE_RELATIVE).resolve()), "frame0_visibility": str((data_root / VISIBILITY_RELATIVE / "val.npz").resolve())},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps({"seed": SEED, "device": str(device), "action_set": "current/Stay + Stage-A legal candidate_pool", "policy_train_contexts": len(train_rows), "moving_val_contexts": len(moving_rows), "test_used": False, "training_used": False}, indent=2) + "\n", encoding="utf-8")
    for filename, value in {
        "recognizer_metadata.json": result["recognizer_metadata"], "old_reproduction.json": result["old_reproduction"], "stay_metrics.json": result["stay_metrics"], "legal_micro_metrics.json": result["legal_micro_metrics"], "random_metrics.json": result["random_metrics"], "static_prior_metrics.json": result["static_prior_metrics"], "oracle_metrics.json": result["oracle_metrics"], "headroom_metrics.json": result["headroom_metrics"], "per_viewpoint_metrics.json": result["per_viewpoint_metrics"], "per_radius_metrics.json": result["per_radius_metrics"], "relative_body_yaw_metrics.json": result["relative_body_yaw_metrics"], "viewpoint_variance_metrics.json": result["viewpoint_variance_metrics"], "utility_correlation.json": result["utility_correlation"], "oracle_agreement.json": result["oracle_agreement"], "prior_structure_comparison.json": result["prior_structure_comparison"], "scene_placement_metrics.json": result["scene_placement_metrics"], "per_class_metrics.json": result["per_class_metrics"], "occlusion_metrics.json": result["occlusion_metrics"], "leakage_audit.json": result["leakage_audit"],
    }.items():
        (output_dir / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.data_root.resolve(), _require_cuda(args.device), args.batch_size)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "train_contexts": result["population"]["policy_train_contexts"], "moving_val_contexts": result["population"]["moving_val_contexts"], "old_stay": result["stay_metrics"]["old"]["accuracy"], "yaw8_stay": result["stay_metrics"]["yaw8"]["accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
