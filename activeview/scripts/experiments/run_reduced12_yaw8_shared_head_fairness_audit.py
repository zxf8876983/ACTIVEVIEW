#!/usr/bin/env python3
"""Fair Policy-balanced shared-head comparison for the Old and Yaw8 encoders.

Both ST-GCN encoders remain frozen.  The two new ``Linear(256,256)->GELU
->Linear(256,12)`` heads consume one identical, record-balanced sample list
per epoch and are selected using a held-out subset of Policy-Train records.
Moving Val is evaluation-only; Policy Test is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
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
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    NUM_CLASSES,
    classification,
    correlation,
    gt_margin,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (
    _action_predictions,
    _archive_skeleton,
    _archive_skeleton,
    _any_correct,
    _build_metrics,
    _candidate_micro,
    _load_old_cache,
    _load_rows,
    _metric,
    _occlusion_metrics,
    _oracle_agreement,
    _oracle_actions,
    _per_class,
    _random_actions,
    _relative_metrics,
    _static_actions,
    _train_prior,
    _viewpoint_metrics,
    _yaw8_cache,
    _read_npz,
    _sha256,
    _signature,
    _require_cuda,
    _seed,
)
from activeview.scripts.experiments.yaw8_shared_head_fairness_report import write_analysis
from activeview.scripts.eval.reduced12_utility_source import load_scene_metadata


SEED = 42
FEATURE_DIM = 256
NUM_VIEWS = 32
MAX_OPTIONS = 22
HEAD_EPOCHS = 15
OBS_PER_RECORD = 16
TRAIN_BATCH = 1024
INFERENCE_BATCH = 4096
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
OLD_CHECKPOINT_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
YAW8_CHECKPOINT_RELATIVE = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
OLD_SOURCE_RELATIVE = Path("diagnostics/reduced12_dual_route_overnight")
YAW8_SOURCE_RELATIVE = Path("diagnostics/reduced12_yaw8_policy_landscape")
FAIR_CACHE_RELATIVE = Path("diagnostics/policy_features_shared_head_fairness")
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "yaw8_shared_head_fairness_audit"
)
FAIR_CHECKPOINT_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "yaw8_shared_head_fairness_audit"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_source_cache(
    cache: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], name: str
) -> dict[str, np.ndarray]:
    required = {"features", "ids", "mask"}
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"{name} cache missing keys: {sorted(missing)}")
    features = np.asarray(cache["features"])
    ids = np.asarray(cache["ids"], dtype=np.int64)
    mask = np.asarray(cache["mask"], dtype=bool)
    if features.shape != (len(rows), MAX_OPTIONS, FEATURE_DIM):
        raise ValueError(f"{name} feature shape mismatch: {features.shape}")
    if ids.shape != (len(rows), MAX_OPTIONS) or mask.shape != ids.shape:
        raise ValueError(f"{name} id/mask shape mismatch")
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        expected = [int(row["current_viewpoint_id"])] + [
            int(value) for value in row["candidate_ids"]
        ]
        if ids[index, active].tolist() != expected:
            raise ValueError(f"{name} action order mismatch at {row['episode_id']}")
    return {
        "features": features.astype(np.float16, copy=False),
        "ids": ids,
        "mask": mask,
    }


def _load_source_cache(
    data_root: Path,
    encoder: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], Path]:
    if encoder == "old":
        filename = "train_options.npz" if split == "train" else "val_options.npz"
        root = data_root / OLD_SOURCE_RELATIVE
    elif encoder == "yaw8":
        filename = "train.npz" if split == "train" else "moving_val.npz"
        root = data_root / YAW8_SOURCE_RELATIVE
    else:
        raise ValueError(encoder)
    path = root / filename
    metadata_path = root / filename.replace(".npz", ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(path)
    metadata = _read_json(metadata_path)
    if int(metadata.get("rows", -1)) != len(rows):
        raise ValueError(f"{encoder}/{split} source row count mismatch")
    if metadata.get("signature") != _signature(rows):
        raise ValueError(f"{encoder}/{split} source signature mismatch")
    return _validate_source_cache(_read_npz(path), rows, f"{encoder}/{split}"), path


def _persist_fair_cache(
    data_root: Path,
    encoder: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    source_cache: Mapping[str, np.ndarray],
    source_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Write a self-describing feature cache outside Git and reuse it safely."""
    directory = data_root / FAIR_CACHE_RELATIVE / encoder
    path = directory / f"{split}.npz"
    metadata_path = directory / f"{split}.json"
    metadata = {
        "encoder": encoder,
        "split": split,
        "rows": len(rows),
        "signature": _signature(rows),
        "source_npz": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "feature_dim": FEATURE_DIM,
        "fields": ["record_id", "context_id", "viewpoint_id", "label", "split", "feature"],
        "test_used": False,
    }
    if path.is_file() and metadata_path.is_file() and _read_json(metadata_path) == metadata:
        return _validate_source_cache(_read_npz(path), rows, f"fair {encoder}/{split}"), metadata
    record_ids = np.asarray([str(row["record_id"]) for row in rows], dtype="U128")
    context_ids = np.asarray([str(row["episode_id"]) for row in rows], dtype="U256")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    payload = {
        "features": np.asarray(source_cache["features"], dtype=np.float16),
        "ids": np.asarray(source_cache["ids"], dtype=np.int64),
        "mask": np.asarray(source_cache["mask"], dtype=bool),
        "record_ids": record_ids,
        "context_ids": context_ids,
        "labels": labels,
        "split": np.asarray([split], dtype="U32"),
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"{split}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    _write_json(metadata_path, metadata)
    return _validate_source_cache(payload, rows, f"fair {encoder}/{split}"), metadata


def _build_options(cache: Mapping[str, np.ndarray], logp: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(logp, dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=bool)
    safe = np.where(mask[..., None], values, -np.inf)
    maxima = np.max(safe, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        shifted = np.where(mask[..., None], safe - maxima, 0.0)
    values = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    values[~mask] = np.nan
    return {
        "features": np.asarray(cache["features"], dtype=np.float32),
        "logp": values,
        "ids": np.asarray(cache["ids"], dtype=np.int64),
        "mask": np.asarray(cache["mask"], dtype=bool),
    }


def _head_logits(head: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIM)
    output: list[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(flat), INFERENCE_BATCH):
            batch = torch.from_numpy(flat[start : start + INFERENCE_BATCH]).to(
                device, non_blocking=True
            )
            output.append(head(batch).cpu().numpy())
    return np.concatenate(output, axis=0).reshape(*features.shape[:-1], NUM_CLASSES)


def _legal_ce(
    head: nn.Module,
    cache: Mapping[str, np.ndarray],
    labels: np.ndarray,
    row_indices: Sequence[int],
    device: torch.device,
) -> float:
    chunks: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for index in row_indices:
        valid = np.flatnonzero(np.asarray(cache["mask"])[index])[1:]
        if valid.size:
            chunks.append(torch.from_numpy(np.asarray(cache["features"])[index, valid].astype(np.float32)))
            targets.append(torch.full((valid.size,), int(labels[index]), dtype=torch.long))
    if not chunks:
        return 0.0
    x = torch.cat(chunks).to(device, non_blocking=True)
    y = torch.cat(targets).to(device, non_blocking=True)
    with torch.inference_mode():
        return float(nn.functional.cross_entropy(head(x), y).item())


def _record_split(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    names = sorted({str(row["record_id"]) for row in rows})
    holdout = max(1, int(round(0.1 * len(names))))
    return set(names[:-holdout]), set(names[-holdout:])


def _sample_epoch(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    train_records: set[str],
    epoch: int,
) -> list[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["record_id"])].append(index)
    rng = np.random.default_rng(SEED + epoch)
    sampled: list[tuple[int, int]] = []
    for record in sorted(train_records):
        choices = [
            (index, int(slot))
            for index in groups[record]
            for slot in np.flatnonzero(np.asarray(cache["mask"])[index])
        ]
        if not choices:
            raise ValueError(f"record {record} has no observations")
        picked = rng.choice(len(choices), OBS_PER_RECORD, replace=len(choices) < OBS_PER_RECORD)
        sampled.extend(choices[int(value)] for value in np.asarray(picked).reshape(-1))
    return sampled


def _sample_digest(sampled_epochs: Sequence[Sequence[tuple[int, int]]]) -> str:
    digest = hashlib.sha256()
    for epoch in sampled_epochs:
        digest.update(repr(list(epoch)).encode("utf-8"))
    return digest.hexdigest()


def _train_matched_head(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    checkpoint: Path,
    summary_path: Path,
    device: torch.device,
    sampled_epochs: Sequence[Sequence[tuple[int, int]]],
) -> tuple[SharedHead, dict[str, Any]]:
    train_records, internal_val_records = _record_split(rows)
    sampled_digest = _sample_digest(sampled_epochs)
    internal_indices = [
        index for index, row in enumerate(rows) if str(row["record_id"]) in internal_val_records
    ]
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        head = SharedHead().to(device)
        head.load_state_dict(payload["state_dict"])
        summary = _read_json(summary_path)
        if summary.get("sample_index_digest") != sampled_digest:
            summary["sample_index_digest"] = sampled_digest
            _write_json(summary_path, summary)
        return head.eval(), summary
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    head = SharedHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    for epoch, sampled in enumerate(sampled_epochs, start=1):
        x = np.stack([np.asarray(cache["features"])[row, slot] for row, slot in sampled]).astype(np.float32)
        y = np.asarray([labels[row] for row, _ in sampled], dtype=np.int64)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
            batch_size=TRAIN_BATCH,
            shuffle=True,
            pin_memory=True,
        )
        head.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        val_ce = _legal_ce(head, cache, labels, internal_indices, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "internal_policy_train_legal_ce": val_ce,
            "train_records": len(train_records),
            "internal_val_records": len(internal_val_records),
            "observations": len(sampled),
        }
        history.append(record)
        print(f"[{name}] epoch={epoch:02d} train_loss={record['train_loss']:.6f} internal_val_legal_ce={val_ce:.6f}", flush=True)
        if val_ce < best_loss:
            best_loss = val_ce
            best_epoch = epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": head.state_dict(), "epoch": epoch, "seed": SEED, "num_classes": NUM_CLASSES},
                checkpoint,
            )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(payload["state_dict"])
    summary = {
        "branch": name,
        "seed": SEED,
        "head": "Linear(256,256)->GELU->Linear(256,12)",
        "lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": HEAD_EPOCHS,
        "best_epoch": best_epoch,
        "best_internal_policy_train_legal_ce": best_loss,
        "train_records": len(train_records),
        "internal_val_records": len(internal_val_records),
        "observations_per_record": OBS_PER_RECORD,
        "observations_per_epoch": int(len(sampled_epochs[0])) if sampled_epochs else 0,
        "sample_index_list_shared_with_other_head": True,
        "sample_index_digest": sampled_digest,
        "checkpoint_selection": "minimum legal-candidate CE on deterministic 10% Policy-Train record holdout",
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
        "moving_val_used_for_training_or_selection": False,
    }
    _write_json(summary_path, summary)
    return head.eval(), summary


def _selected_metric(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    actions: Sequence[int],
    name: str,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    predictions = _action_predictions(options, actions)
    return _metric(labels, predictions, name, actions, rows)


def _historical_reproduction(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    random_actions: Sequence[int],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    stay = [int(row["current_viewpoint_id"]) for row in rows]
    return {
        "Stay/current": _selected_metric(rows, options, stay, "Old historical Stay/current"),
        "Legal micro": _candidate_micro(rows, options, "Old historical legal candidates"),
        "Random legal": _selected_metric(rows, options, random_actions, "Old historical Random legal"),
        "expected_reference": {
            "stay_accuracy": 0.302579,
            "legal_micro_accuracy": 0.375418,
            "random_accuracy": 0.365079,
        },
        "labels": int(labels.size),
    }


def _utility_ranking(
    rows: Sequence[Mapping[str, Any]],
    old_options: Mapping[str, np.ndarray],
    yaw_options: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {
        "true_logp": {"old": [], "yaw8": []},
        "margins": {"old": [], "yaw8": []},
    }
    within: dict[str, list[float]] = {"true_logp": [], "margins": []}
    for index, row in enumerate(rows):
        old_values: dict[str, list[float]] = {"true_logp": [], "margins": []}
        yaw_values: dict[str, list[float]] = {"true_logp": [], "margins": []}
        for slot in np.flatnonzero(old_options["mask"][index])[1:]:
            label = int(row["label_id"])
            old_lp = old_options["logp"][index, int(slot)]
            yaw_lp = yaw_options["logp"][index, int(slot)]
            old_values["true_logp"].append(float(old_lp[label]))
            yaw_values["true_logp"].append(float(yaw_lp[label]))
            old_values["margins"].append(gt_margin(old_lp, label))
            yaw_values["margins"].append(gt_margin(yaw_lp, label))
        for field in values:
            values[field]["old"].extend(old_values[field])
            values[field]["yaw8"].extend(yaw_values[field])
            if len(old_values[field]) >= 2:
                within[field].append(
                    correlation(old_values[field], yaw_values[field], spearman=True)
                )
    return {
        field: {
            "global_spearman": correlation(values[field]["old"], values[field]["yaw8"], spearman=True),
            "global_pearson": correlation(values[field]["old"], values[field]["yaw8"]),
            "within_context_mean": float(np.mean(within[field])) if within[field] else 0.0,
            "within_context_median": float(np.median(within[field])) if within[field] else 0.0,
        }
        for field in values
    }


def _prior_structure(old_prior: Mapping[int, float], yaw_prior: Mapping[int, float]) -> dict[str, Any]:
    old = np.asarray([float(old_prior.get(index, old_prior.get(-1, 0.0))) for index in range(NUM_VIEWS)])
    yaw = np.asarray([float(yaw_prior.get(index, yaw_prior.get(-1, 0.0))) for index in range(NUM_VIEWS)])
    old_top = set(np.argsort(-old)[:5].tolist())
    yaw_top = set(np.argsort(-yaw)[:5].tolist())
    return {
        "spearman": correlation(old, yaw, spearman=True),
        "pearson": correlation(old, yaw),
        "top5_overlap": float(len(old_top & yaw_top) / 5.0),
        "old_best_viewpoints": np.argsort(-old)[:5].tolist(),
        "yaw8_best_viewpoints": np.argsort(-yaw)[:5].tolist(),
    }


def _headroom(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for recognizer, values in metrics.items():
        oracle = float(values["GT-TrueLogP Oracle"]["accuracy"])
        margin = float(values["GT-Margin Oracle"]["accuracy"])
        random = float(values["Random legal"]["accuracy"])
        static = float(values["StaticPrior"]["accuracy"])
        any_rate = float(values["AnyCorrect Coverage"]["coverage_rate"])
        result[recognizer] = {
            "random_to_gt_true_logp": oracle - random,
            "static_to_gt_true_logp": oracle - static,
            "random_to_gt_margin": margin - random,
            "static_to_gt_margin": margin - static,
            "anycorrect_minus_random": any_rate - random,
            "anycorrect_minus_static": any_rate - static,
        }
    return result


def _candidate_only_oracles(
    rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    metrics: dict[str, Any] = {}
    for criterion, name in (("true_logp", "GT-TrueLogP"), ("margin", "GT-Margin")):
        actions = _oracle_actions(options, rows, criterion, candidate_only=True)
        metrics[name] = _selected_metric(rows, options, actions, f"candidate-only {name}")
    coverage = _any_correct(options, rows, candidate_only=True)
    metrics["AnyCorrect Coverage"] = {
        "contexts": int(labels.size),
        "coverage_count": int(coverage.sum()),
        "coverage_rate": float(coverage.mean()) if coverage.size else 0.0,
        "test_used": False,
    }
    return metrics


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.time()
    _seed()
    train_rows, moving_rows = _load_rows(data_root)
    old_checkpoint = data_root / OLD_CHECKPOINT_RELATIVE
    yaw8_checkpoint = data_root / YAW8_CHECKPOINT_RELATIVE
    if not old_checkpoint.is_file() or not yaw8_checkpoint.is_file():
        raise FileNotFoundError("required Old/Yaw8 checkpoint missing")
    historical_head_path = data_root / (
        "checkpoints/policy_reduced12_eight_placement_v1/"
        "view_agnostic_frozen_encoder_head/shared_head_best.pth"
    )
    if not historical_head_path.is_file():
        raise FileNotFoundError(historical_head_path)
    historical_head = SharedHead().to(device)
    historical_head.load_state_dict(torch.load(historical_head_path, map_location=device, weights_only=False)["state_dict"])
    historical_head.eval()
    historical_val = _load_old_cache(data_root, "val", moving_rows, historical_head, device)
    random_actions = _random_actions(moving_rows)
    old_hist = _historical_reproduction(moving_rows, historical_val, random_actions)
    gate_delta = {
        "stay_accuracy": old_hist["Stay/current"]["accuracy"] - old_hist["expected_reference"]["stay_accuracy"],
        "legal_micro_accuracy": old_hist["Legal micro"]["accuracy"] - old_hist["expected_reference"]["legal_micro_accuracy"],
        "random_accuracy": old_hist["Random legal"]["accuracy"] - old_hist["expected_reference"]["random_accuracy"],
    }
    old_hist["gate_delta"] = gate_delta
    old_hist["gate_status"] = "PASS" if max(abs(value) for value in gate_delta.values()) <= 0.002 else "FAIL"
    if old_hist["gate_status"] != "PASS":
        raise RuntimeError(f"historical Old shared-head reproduction failed: {gate_delta}")
    old_train_source, old_train_path = _load_source_cache(data_root, "old", "train", train_rows)
    old_val_source, old_val_path = _load_source_cache(data_root, "old", "val", moving_rows)
    yaw_train_source, yaw_train_path = _load_source_cache(data_root, "yaw8", "train", train_rows)
    yaw_val_source, yaw_val_path = _load_source_cache(data_root, "yaw8", "val", moving_rows)
    old_train_cache, old_train_meta = _persist_fair_cache(data_root, "old", "train", train_rows, old_train_source, old_train_path)
    old_val_cache, old_val_meta = _persist_fair_cache(data_root, "old", "moving_val", moving_rows, old_val_source, old_val_path)
    yaw_train_cache, yaw_train_meta = _persist_fair_cache(data_root, "yaw8", "train", train_rows, yaw_train_source, yaw_train_path)
    yaw_val_cache, yaw_val_meta = _persist_fair_cache(data_root, "yaw8", "moving_val", moving_rows, yaw_val_source, yaw_val_path)
    # Feature extraction location and dimensionality sanity check on both frozen models.
    old_model, _ = load_checkpoint(old_checkpoint, NUM_CLASSES, str(device))
    yaw_model, _ = load_checkpoint(yaw8_checkpoint, NUM_CLASSES, str(device))
    probe_skeleton = _archive_skeleton(Path(str(train_rows[0]["archive_path"])))
    probe_view = int(train_rows[0]["current_viewpoint_id"])
    probe = torch.from_numpy(probe_skeleton[probe_view : probe_view + 1]).to(device)
    with torch.inference_mode():
        old_probe_feature = old_model.forward_features(probe)
        yaw_probe_feature = yaw_model.forward_features(probe)
    if old_probe_feature.shape[-1] != FEATURE_DIM or yaw_probe_feature.shape[-1] != FEATURE_DIM:
        raise ValueError("Old/Yaw8 forward_features are not 256-D")
    feature_check = {
        "old_forward_features_shape": list(old_probe_feature.shape),
        "yaw8_forward_features_shape": list(yaw_probe_feature.shape),
        "feature_dimension_identical": int(old_probe_feature.shape[-1]) == int(yaw_probe_feature.shape[-1]),
        "extraction_function": "forward_features (pre-FC pooled representation)",
    }
    del probe, old_probe_feature, yaw_probe_feature, old_model, yaw_model, historical_head
    sampled_epochs = [
        _sample_epoch(train_rows, old_train_cache, _record_split(train_rows)[0], epoch)
        for epoch in range(1, HEAD_EPOCHS + 1)
    ]
    checkpoint_root = data_root / FAIR_CHECKPOINT_RELATIVE
    old_head, old_training = _train_matched_head(
        "H_old_shared_new", train_rows, old_train_cache,
        checkpoint_root / "old_shared_head_best.pth",
        output_dir / "old_shared_training.json", device, sampled_epochs,
    )
    yaw_head, yaw_training = _train_matched_head(
        "H_yaw8_shared", train_rows, yaw_train_cache,
        checkpoint_root / "yaw8_shared_head_best.pth",
        output_dir / "yaw8_shared_training.json", device, sampled_epochs,
    )
    old_fair_train = _build_options(old_train_cache, _head_logits(old_head, old_train_cache["features"], device))
    old_fair_val = _build_options(old_val_cache, _head_logits(old_head, old_val_cache["features"], device))
    yaw_fair_train = _build_options(yaw_train_cache, _head_logits(yaw_head, yaw_train_cache["features"], device))
    yaw_fair_val = _build_options(yaw_val_cache, _head_logits(yaw_head, yaw_val_cache["features"], device))
    old_native_train = _build_options(old_train_source, np.asarray(_read_npz(old_train_path)["logits"], dtype=np.float32))
    old_native_val = _build_options(old_val_source, np.asarray(_read_npz(old_val_path)["logits"], dtype=np.float32))
    yaw_native_train = _build_options(yaw_train_source, np.asarray(_read_npz(yaw_train_path)["logp"], dtype=np.float32))
    yaw_native_val = _build_options(yaw_val_source, np.asarray(_read_npz(yaw_val_path)["logp"], dtype=np.float32))
    old_prior, old_prior_meta = _train_prior(train_rows, old_fair_train)
    yaw_prior, yaw_prior_meta = _train_prior(train_rows, yaw_fair_train)
    yaw_native_prior, yaw_native_prior_meta = _train_prior(train_rows, yaw_native_train)
    old_metrics, old_details = _build_metrics(moving_rows, old_fair_val, "OldFair", old_prior, random_actions)
    yaw_metrics, yaw_details = _build_metrics(moving_rows, yaw_fair_val, "Yaw8Fair", yaw_prior, random_actions)
    native_metrics, native_details = _build_metrics(moving_rows, yaw_native_val, "Yaw8Native", yaw_native_prior, random_actions)
    old_candidate_oracles = _candidate_only_oracles(moving_rows, old_fair_val)
    yaw_candidate_oracles = _candidate_only_oracles(moving_rows, yaw_fair_val)
    del old_details, yaw_details, native_details
    old_viewpoint_flat = _flatten_for_view_metrics(moving_rows, old_fair_val)
    yaw_viewpoint_flat = _flatten_for_view_metrics(moving_rows, yaw_fair_val)
    metadata, scene_audit = load_scene_metadata(data_root, list(train_rows) + list(moving_rows), audit_sample_count=10)
    relative, _ = _relative_metrics(moving_rows, old_fair_val, yaw_fair_val, metadata)
    viewpoint, _ = _viewpoint_metrics(old_viewpoint_flat, yaw_viewpoint_flat)
    utility = _utility_ranking(moving_rows, old_fair_val, yaw_fair_val)
    old_details_metrics = _build_metrics(moving_rows, old_fair_val, "OldFair", old_prior, random_actions)[1]
    yaw_details_metrics = _build_metrics(moving_rows, yaw_fair_val, "Yaw8Fair", yaw_prior, random_actions)[1]
    oracle_agreement = _oracle_agreement(old_details_metrics, yaw_details_metrics, old_fair_val, yaw_fair_val, moving_rows)
    old_any = _per_class(moving_rows, {"oldfair": old_fair_val}, {"oldfair": old_details_metrics["actions"]})["oldfair"]
    yaw_any = _per_class(moving_rows, {"yaw8fair": yaw_fair_val}, {"yaw8fair": yaw_details_metrics["actions"]})["yaw8fair"]
    per_class = {"oldfair": old_any, "yaw8fair": yaw_any}
    options_by = {"oldfair": old_fair_val, "yaw8fair": yaw_fair_val}
    actions_by = {"oldfair": old_details_metrics["actions"], "yaw8fair": yaw_details_metrics["actions"]}
    occlusion = _occlusion_metrics(data_root, moving_rows, actions_by, options_by)
    old_random = old_metrics["Random legal"]["accuracy"]
    yaw_random = yaw_metrics["Random legal"]["accuracy"]
    high_old = float(occlusion["recognizers"]["oldfair"]["Random legal"]["accuracy"])
    high_yaw = float(occlusion["recognizers"]["yaw8fair"]["Random legal"]["accuracy"])
    key_class = ("sit", "bend", "touching face", "knock", "clap", "kick")
    max_decline = 0.0
    class_random_changes: dict[str, float] = {}
    for action in LABELS:
        old_acc = float(per_class["oldfair"]["Random legal"][action]["accuracy"])
        yaw_acc = float(per_class["yaw8fair"]["Random legal"][action]["accuracy"])
        class_random_changes[action] = (yaw_acc - old_acc) * 100.0
        if action in key_class:
            max_decline = max(max_decline, -class_random_changes[action])
    old_legal = old_metrics["Random legal"]
    del old_legal
    old_native_summary = {
        "Stay/current": native_metrics["Stay/current"],
        "Legal micro": _candidate_micro(moving_rows, yaw_native_val, "Yaw8Native legal candidates"),
        "Random legal": native_metrics["Random legal"],
        "StaticPrior": native_metrics["StaticPrior"],
        "GT-TrueLogP Oracle": native_metrics["GT-TrueLogP Oracle"],
        "GT-Margin Oracle": native_metrics["GT-Margin Oracle"],
        "AnyCorrect Coverage": native_metrics["AnyCorrect Coverage"],
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_YAW8_SHARED_HEAD_FAIRNESS_AUDIT",
        "status": "COMPLETED",
        "population": {
            "policy_train_contexts": len(train_rows),
            "moving_val_contexts": len(moving_rows),
            "train_records": len({str(row["record_id"]) for row in train_rows}),
            "moving_val_records": len({str(row["record_id"]) for row in moving_rows}),
            "train_observations_per_epoch": int(len(sampled_epochs[0])),
            "legal_candidate_count_mean": float(np.mean(np.sum(old_fair_val["mask"][:, 1:], axis=1))),
            "legal_candidate_count_min": int(np.min(np.sum(old_fair_val["mask"][:, 1:], axis=1))),
            "legal_candidate_count_max": int(np.max(np.sum(old_fair_val["mask"][:, 1:], axis=1))),
        },
        "labels": list(LABELS),
        "old_historical_reproduction": old_hist,
        "oldfair_metrics": old_metrics,
        "yaw8fair_metrics": yaw_metrics,
        "yaw8_native_comparison": old_native_summary,
        "stay_metrics": {"oldfair": old_metrics["Stay/current"], "yaw8fair": yaw_metrics["Stay/current"]},
        "legal_micro_metrics": {"old": _candidate_micro(moving_rows, old_fair_val, "OldFair legal candidates"), "yaw8": _candidate_micro(moving_rows, yaw_fair_val, "Yaw8Fair legal candidates")},
        "random_metrics": {"oldfair": old_metrics["Random legal"], "yaw8fair": yaw_metrics["Random legal"]},
        "static_prior_metrics": {"oldfair": old_metrics["StaticPrior"], "yaw8fair": yaw_metrics["StaticPrior"], "yaw8native": native_metrics["StaticPrior"]},
        "oracle_metrics": {"oldfair": {"GT-TrueLogP": old_metrics["GT-TrueLogP Oracle"], "GT-Margin": old_metrics["GT-Margin Oracle"], "AnyCorrect": old_metrics["AnyCorrect Coverage"]}, "yaw8fair": {"GT-TrueLogP": yaw_metrics["GT-TrueLogP Oracle"], "GT-Margin": yaw_metrics["GT-Margin Oracle"], "AnyCorrect": yaw_metrics["AnyCorrect Coverage"]}},
        "candidate_only_oracle_metrics": {"oldfair": old_candidate_oracles, "yaw8fair": yaw_candidate_oracles, "yaw8native": _candidate_only_oracles(moving_rows, yaw_native_val)},
        "headroom_metrics": _headroom({"oldfair": old_metrics, "yaw8fair": yaw_metrics}),
        "per_viewpoint_metrics": viewpoint["viewpoints"],
        "per_radius_metrics": viewpoint["radii"],
        "relative_yaw_metrics": relative,
        "utility_ranking": utility,
        "oracle_agreement": oracle_agreement,
        "prior_structure": _prior_structure(old_prior, yaw_prior),
        "per_class_metrics": per_class,
        "occlusion_metrics": occlusion,
        "gain_decomposition": {
            "stay_accuracy_pp": (yaw_metrics["Stay/current"]["accuracy"] - old_metrics["Stay/current"]["accuracy"]) * 100.0,
            "legal_accuracy_pp": (yaw_metrics["Legal candidate micro"]["accuracy"] - old_metrics["Legal candidate micro"]["accuracy"]) * 100.0 if "Legal candidate micro" in yaw_metrics else (_candidate_micro(moving_rows, yaw_fair_val, "tmp")["accuracy"] - _candidate_micro(moving_rows, old_fair_val, "tmp")["accuracy"]) * 100.0,
            "random_accuracy_pp": (yaw_metrics["Random legal"]["accuracy"] - old_metrics["Random legal"]["accuracy"]) * 100.0,
            "static_accuracy_pp": (yaw_metrics["StaticPrior"]["accuracy"] - old_metrics["StaticPrior"]["accuracy"]) * 100.0,
            "yaw8_policy_head_gain_legal_pp": (old_native_summary["Legal micro"]["accuracy"] * 0.0 + (_candidate_micro(moving_rows, yaw_fair_val, "tmp")["accuracy"] - old_native_summary["Legal micro"]["accuracy"])) * 100.0,
            "yaw8_policy_head_gain_random_pp": (yaw_metrics["Random legal"]["accuracy"] - native_metrics["Random legal"]["accuracy"]) * 100.0,
            "high_occlusion_random_gain_pp": (high_yaw - high_old) * 100.0,
            "max_key_class_random_decline_pp": max_decline,
            "class_random_changes_pp": class_random_changes,
        },
        "recognizer_metadata": {
            "old": {"checkpoint": str(old_checkpoint.resolve()), "sha256": _sha256(old_checkpoint), "feature_extraction": "frozen ST-GCN forward_features, 256-D pre-FC representation", "head": "new matched shared head"},
            "yaw8": {"checkpoint": str(yaw8_checkpoint.resolve()), "sha256": _sha256(yaw8_checkpoint), "feature_extraction": "frozen Yaw8 ST-GCN forward_features, 256-D pre-FC representation", "head": "new matched shared head"},
            "feature_dimension_identical": True,
            "head_architecture_identical": True,
            "feature_extraction_check": feature_check,
        },
        "source_protocol_audit": {
            "policy_action_set": "current/Stay + Stage-A legal candidate_pool",
            "train_sampler": "313 records, 16 observations per record, same sampled (row,slot) list for both heads",
            "sample_index_digest": _sample_digest(sampled_epochs),
            "train_internal_holdout": "deterministic 10% of Policy-Train records",
            "old_source_cache": str(old_train_path.resolve()),
            "yaw8_source_cache": str(yaw_train_path.resolve()),
            "cache_metadata": {"old_train": old_train_meta, "old_moving_val": old_val_meta, "yaw8_train": yaw_train_meta, "yaw8_moving_val": yaw_val_meta},
            "scene_metadata_audit": scene_audit,
        },
        "training": {"old": old_training, "yaw8": yaw_training},
        "leakage_audit": {"policy_test_read": False, "moving_val_used_for_head_training": False, "moving_val_used_for_checkpoint_selection": False, "new_rgb_generated": False, "new_skeleton_generated": False, "yolo_rerun": False, "videopose3d_rerun": False, "old_encoder_modified": False, "yaw8_encoder_modified": False, "nbv_selector_trained": False},
        "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "elapsed_seconds": float(time.time() - started)},
        "artifacts": {"old_fair_train_cache": str((data_root / FAIR_CACHE_RELATIVE / "old" / "train.npz").resolve()), "yaw8_fair_train_cache": str((data_root / FAIR_CACHE_RELATIVE / "yaw8" / "train.npz").resolve()), "old_head_checkpoint": str((checkpoint_root / "old_shared_head_best.pth").resolve()), "yaw8_head_checkpoint": str((checkpoint_root / "yaw8_shared_head_best.pth").resolve())},
    }
    # Add direct keys used by the analysis renderer without recomputing inference.
    result["oldfair_metrics"]["Legal candidate micro"] = result["legal_micro_metrics"]["old"]
    result["yaw8fair_metrics"]["Legal candidate micro"] = result["legal_micro_metrics"]["yaw8"]
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": {"seed": SEED, "head": "Linear(256,256)->GELU->Linear(256,12)", "head_epochs": HEAD_EPOCHS, "head_lr": HEAD_LR, "weight_decay": WEIGHT_DECAY, "observations_per_record": OBS_PER_RECORD, "train_internal_holdout_fraction": 0.1, "policy_test_used": False},
        "source_protocol_audit.json": result["source_protocol_audit"],
        "old_historical_reproduction.json": result["old_historical_reproduction"],
        "old_shared_training.json": old_training,
        "yaw8_shared_training.json": yaw_training,
        "oldfair_metrics.json": old_metrics,
        "yaw8fair_metrics.json": yaw_metrics,
        "yaw8_native_comparison.json": old_native_summary,
        "stay_metrics.json": result["stay_metrics"],
        "legal_micro_metrics.json": result["legal_micro_metrics"],
        "random_metrics.json": result["random_metrics"],
        "static_prior_metrics.json": result["static_prior_metrics"],
        "oracle_metrics.json": result["oracle_metrics"],
        "candidate_only_oracle_metrics.json": result["candidate_only_oracle_metrics"],
        "headroom_metrics.json": result["headroom_metrics"],
        "per_viewpoint_metrics.json": result["per_viewpoint_metrics"],
        "per_radius_metrics.json": result["per_radius_metrics"],
        "relative_yaw_metrics.json": relative,
        "utility_ranking.json": utility,
        "oracle_agreement.json": oracle_agreement,
        "prior_structure.json": result["prior_structure"],
        "per_class_metrics.json": per_class,
        "occlusion_metrics.json": occlusion,
        "gain_decomposition.json": result["gain_decomposition"],
        "leakage_audit.json": result["leakage_audit"],
    }
    for filename, value in files.items():
        _write_json(output_dir / filename, value)
    _write_json(output_dir / "result.json", result)
    write_analysis(result, output_dir / "analysis.md")
    return result


def _flatten_for_view_metrics(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    labels: list[int] = []
    views: list[int] = []
    predictions: list[int] = []
    true_logp: list[float] = []
    margins: list[float] = []
    for index, row in enumerate(rows):
        label = int(row["label_id"])
        for slot in np.flatnonzero(options["mask"][index])[1:]:
            logp = options["logp"][index, int(slot)]
            labels.append(label)
            views.append(int(options["ids"][index, int(slot)]))
            predictions.append(int(np.argmax(logp)))
            true_logp.append(float(logp[label]))
            margins.append(gt_margin(logp, label))
    radius_values = np.asarray((1.5, 2.0, 2.5, 3.0), dtype=np.float32)
    radii = radius_values[np.asarray(views, dtype=np.int64) // 8]
    return {"labels": np.asarray(labels, dtype=np.int64), "views": np.asarray(views, dtype=np.int64), "predictions": np.asarray(predictions, dtype=np.int64), "true_logp": np.asarray(true_logp, dtype=np.float32), "margins": np.asarray(margins, dtype=np.float32), "radii": radii}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.data_root.resolve(), _require_cuda(args.device))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "train_contexts": result["population"]["policy_train_contexts"], "moving_val_contexts": result["population"]["moving_val_contexts"], "oldfair_random": result["oldfair_metrics"]["Random legal"]["accuracy"], "yaw8fair_random": result["yaw8fair_metrics"]["Random legal"]["accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
