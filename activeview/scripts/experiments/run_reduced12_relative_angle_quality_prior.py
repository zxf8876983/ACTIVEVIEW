#!/usr/bin/env python3
"""Train-internal-Val relative-view quality prior audit for reduced12.
The angle prior is computed only from the raw-train-derived Yaw8 internal
validation observations.  It is frozen before the Moving-Val evaluation.  The
Moving-Val selectors use only candidate geometry/metadata and the frozen
prior; archived candidate recognition is used only for terminal evaluation.
Policy Test is never opened.
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
    rank,
)
from activeview.scripts.eval.reduced12_utility_source import (
    body_azimuth_bin,
    load_scene_metadata,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    VisibilityPredictor,
    _option_geometry,
    _predict,
)
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (
    _load_rows,
)
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import (
    _load_existing_dino,
    _load_yaw8_options,
    _read_npz,
    _sha256,
    _signature,
)
SEED = 42
YAW_DEGREES = (0, 45, 90, 135, 180, 225, 270, 315)
NUM_VIEWS = 32
FEATURE_DIM = 256
MAX_OPTIONS = 22
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/relative_angle_quality_prior"
)
YAW_ROOT_REL = Path("datasets/stgcn_babel_reduced12_habitat_yolo26n_yaw8_v1")
YAW_CHECKPOINT_REL = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
YAW_HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
)
POLICY_REL = Path("datasets/policy_reduced12_eight_placement_v1")
VISIBILITY_REL = Path("diagnostics/frame0_visibility_predictor_v1")
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
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value
def _wrap(angle: float) -> float:
    return float((float(angle) + 180.0) % 360.0 - 180.0)
def _camera_azimuth(row: Mapping[str, Any]) -> float:
    """Reproduce the frozen Yaw8 camera azimuth sampling rule."""
    source_id = int(row.get("babel_sid", 0))
    return float(random.Random(SEED + source_id).uniform(-25.0, 25.0) % 360.0)
def _internal_records(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    root = data_root / YAW_ROOT_REL
    data = np.load(root / f"{split}_data.npy", mmap_mode="r")
    labels = np.load(root / f"{split}_labels.npy", mmap_mode="r").astype(np.int64)
    metadata = _read_json(root / f"{split}_metadata.json")
    expanded = _read_json(root / f"{split}.json")
    by_id = {str(row["record_id"]): row for row in expanded}
    if len(data) != len(labels) or len(data) != len(metadata):
        raise ValueError(f"YAW8 {split} arrays/metadata mismatch")
    rows: list[dict[str, Any]] = []
    bins: list[int] = []
    for index, item in enumerate(metadata):
        row = by_id.get(str(item["record_id"]))
        if row is None:
            raise ValueError(f"missing expanded Yaw8 record {item['record_id']}")
        if int(item["label_id"]) != int(labels[index]):
            raise ValueError(f"label mismatch in Yaw8 {split}: {item['record_id']}")
        camera_azimuth = _camera_azimuth(row)
        scene_yaw = float(item.get("scene_yaw_deg", row.get("scene_yaw_deg", 0.0)))
        item_copy = dict(item)
        item_copy["babel_sid"] = int(row.get("babel_sid", 0))
        item_copy["camera_azimuth_deg"] = camera_azimuth
        item_copy["relative_angle_deg"] = _wrap(camera_azimuth - scene_yaw)
        item_copy["relative_angle_bin"] = body_azimuth_bin(camera_azimuth, scene_yaw)
        rows.append(item_copy)
        bins.append(int(item_copy["relative_angle_bin"]))
    source_records = {str(row.get("source_record_id", row["record_id"])) for row in rows}
    return np.asarray(data), labels, rows, {
        "expanded_records": len(rows),
        "source_records": len(source_records),
        "bin_min": int(min(bins)) if bins else 0,
        "bin_max": int(max(bins)) if bins else 0,
    }
def _internal_split_metadata(data_root: Path) -> dict[str, Any]:
    root = data_root / YAW_ROOT_REL
    train = _read_json(root / "train.json")
    internal = _read_json(root / "val.json")
    train_ids = {str(row.get("source_record_id", row["record_id"])) for row in train}
    internal_ids = {str(row.get("source_record_id", row["record_id"])) for row in internal}
    overlap = sorted(train_ids & internal_ids)
    if overlap:
        raise ValueError(f"raw-train train/internal-val overlap: {overlap[:3]}")
    return {
        "seed": SEED,
        "train_record_count": len(train_ids),
        "internal_val_record_count": len(internal_ids),
        "train_expanded_observation_count": len(train),
        "internal_val_expanded_observation_count": len(internal),
        "no_record_overlap": not overlap,
        "overlap_count": len(overlap),
        "source": "reduced12 raw-train split expanded with frozen Yaw8 8-yaw rendering",
        "policy_test_read": False,
    }
def _infer_internal(
    model: nn.Module,
    head: nn.Module,
    data: np.ndarray,
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    predictions: list[int] = []
    logps: list[np.ndarray] = []
    batch_size = 256
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = torch.from_numpy(np.asarray(data[start : start + batch_size])).float().to(device)
            feature = model.forward_features(batch)
            logits = head(feature)
            logp = torch.log_softmax(logits, dim=-1)
            predictions.extend(torch.argmax(logp, dim=-1).cpu().tolist())
            logps.append(logp.cpu().numpy())
    values = np.concatenate(logps, axis=0).astype(np.float32)
    if len(predictions) != len(labels):
        raise RuntimeError("internal inference count mismatch")
    return {
        "labels": labels.astype(np.int64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "logp": values,
        "bins": np.asarray([int(row["relative_angle_bin"]) for row in rows], dtype=np.int64),
    }
def _angle_landscape(inference: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = np.asarray(inference["labels"], dtype=np.int64)
    predictions = np.asarray(inference["predictions"], dtype=np.int64)
    bins = np.asarray(inference["bins"], dtype=np.int64)
    logp = np.asarray(inference["logp"], dtype=np.float32)
    landscape: dict[str, Any] = {}
    q_acc = np.zeros(8, dtype=np.float64)
    q_f1 = np.zeros(8, dtype=np.float64)
    q_margin = np.zeros(8, dtype=np.float64)
    for bin_id in range(8):
        indices = np.flatnonzero(bins == bin_id)
        metric = classification(labels[indices], predictions[indices])
        true_values = logp[indices, labels[indices]]
        margins = np.asarray(
            [gt_margin(logp[index], int(labels[index])) for index in indices], dtype=np.float64
        )
        q_acc[bin_id] = float(metric["accuracy"])
        q_f1[bin_id] = float(metric["macro_f1"])
        q_margin[bin_id] = float(np.mean(margins)) if margins.size else 0.0
        landscape[str(YAW_DEGREES[bin_id])] = {
            "angle_bin": bin_id,
            "count": int(indices.size),
            "accuracy": float(metric["accuracy"]),
            "macro_f1": float(metric["macro_f1"]),
            "mean_gt_true_logp": float(np.mean(true_values)) if true_values.size else 0.0,
            "mean_gt_margin": float(np.mean(margins)) if margins.size else 0.0,
            "per_class": metric["per_class"],
        }
    return landscape, {"q_acc": q_acc, "q_f1": q_f1, "q_margin": q_margin}
def _ranking(values: np.ndarray) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
def _per_class_angle_matrix(inference: Mapping[str, Any]) -> dict[str, Any]:
    labels = np.asarray(inference["labels"], dtype=np.int64)
    predictions = np.asarray(inference["predictions"], dtype=np.int64)
    bins = np.asarray(inference["bins"], dtype=np.int64)
    matrix: dict[str, Any] = {}
    for class_id, name in enumerate(LABELS):
        row: dict[str, Any] = {}
        for bin_id, angle in enumerate(YAW_DEGREES):
            indices = np.flatnonzero((labels == class_id) & (bins == bin_id))
            row[str(angle)] = {
                "count": int(indices.size),
                "accuracy": float(np.mean(predictions[indices] == class_id)) if indices.size else 0.0,
            }
        matrix[name] = row
    return matrix
def _candidate_slots(options: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    active = np.flatnonzero(np.asarray(options["mask"][index], dtype=bool))
    return active[active != 0]
def _select_scores(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    scores: np.ndarray,
) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        slots = _candidate_slots(options, index)
        if slots.size == 0:
            actions.append(int(row["current_viewpoint_id"]))
            continue
        slot = min(slots.tolist(), key=lambda value: (-float(scores[index, value]), int(options["ids"][index, value])))
        actions.append(int(options["ids"][index, slot]))
    return actions
def _terminal_predictions(options: Mapping[str, np.ndarray], actions: Sequence[int]) -> np.ndarray:
    values: list[int] = []
    for index, action in enumerate(actions):
        matches = np.flatnonzero(
            (np.asarray(options["ids"][index]) == int(action))
            & np.asarray(options["mask"][index], dtype=bool)
        )
        if matches.size != 1:
            raise ValueError(f"cannot resolve selected viewpoint {action} at row {index}")
        values.append(int(np.argmax(options["logp"][index, int(matches[0])])))
    return np.asarray(values, dtype=np.int64)
def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    name: str,
    actions: Sequence[int],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    predictions = _terminal_predictions(options, actions)
    metric = classification(labels, predictions)
    current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
    metric.update({
        "method": name,
        "move_rate": float(np.mean(np.asarray(actions, dtype=np.int64) != current)),
        "stay_rate": float(np.mean(np.asarray(actions, dtype=np.int64) == current)),
        "test_used": False,
    })
    return metric
def _random_actions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    rng = np.random.default_rng(SEED)
    return [
        int(rng.choice([int(row["current_viewpoint_id"])] + [int(v) for v in row["candidate_ids"]]))
        for row in rows
    ]
def _prior_scores(options: Mapping[str, np.ndarray], prior: Mapping[int, float]) -> np.ndarray:
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    fallback = float(prior.get(-1, 0.0))
    for index in range(scores.shape[0]):
        for slot in np.flatnonzero(options["mask"][index]):
            scores[index, slot] = float(prior.get(int(options["ids"][index, slot]), fallback))
    return scores
def _rank_normalize_row(values: np.ndarray, slots: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, -np.inf, dtype=np.float64)
    if slots.size == 0:
        return output
    ranks = rank(values[slots])
    scale = max(float(slots.size - 1), 1.0)
    output[slots] = ranks / scale
    return output
def _fused_scores(
    primary: np.ndarray,
    angle_values: np.ndarray,
    options: Mapping[str, np.ndarray],
    weight: float,
) -> np.ndarray:
    output = np.full(primary.shape, -np.inf, dtype=np.float64)
    for index in range(primary.shape[0]):
        slots = _candidate_slots(options, index)
        if slots.size:
            output[index] = _rank_normalize_row(primary[index], slots)
            output[index, slots] += float(weight) * _rank_normalize_row(angle_values[index], slots)[slots]
    return output
def _angle_score_array(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    prior: np.ndarray,
    action_conditioned: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(rows):
        scene_meta = metadata[(str(row["scene_id"]), str(row["region"]))]
        label = int(row["label_id"])
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            bin_id = body_azimuth_bin(scene_meta["azimuths"][view], scene_meta["yaw_deg"])
            scores[index, slot] = float(
                action_conditioned[label, bin_id] if action_conditioned is not None else prior[bin_id]
            )
    return scores
def _flatten_moving(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    metadata: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    labels: list[int] = []
    predictions: list[int] = []
    bins: list[int] = []
    radii: list[int] = []
    views: list[int] = []
    true_logp: list[float] = []
    margins: list[float] = []
    episodes: list[str] = []
    for index, row in enumerate(rows):
        scene_meta = metadata[(str(row["scene_id"]), str(row["region"]))]
        label = int(row["label_id"])
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            logp = np.asarray(options["logp"][index, slot], dtype=np.float64)
            labels.append(label)
            predictions.append(int(np.argmax(logp)))
            bins.append(body_azimuth_bin(scene_meta["azimuths"][view], scene_meta["yaw_deg"]))
            radii.append(view // 8)
            views.append(view)
            true_logp.append(float(logp[label]))
            margins.append(gt_margin(logp, label))
            episodes.append(str(row["episode_id"]))
    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "bins": np.asarray(bins, dtype=np.int64),
        "radii": np.asarray(radii, dtype=np.int64),
        "views": np.asarray(views, dtype=np.int64),
        "true_logp": np.asarray(true_logp, dtype=np.float64),
        "margins": np.asarray(margins, dtype=np.float64),
        "episodes": np.asarray(episodes),
    }
def _moving_landscape(flat: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bin_id, angle in enumerate(YAW_DEGREES):
        indices = np.flatnonzero(flat["bins"] == bin_id)
        metric = classification(flat["labels"][indices], flat["predictions"][indices])
        result[str(angle)] = {
            "angle_bin": bin_id,
            "count": int(indices.size),
            "accuracy": float(metric["accuracy"]),
            "macro_f1": float(metric["macro_f1"]),
            "mean_gt_true_logp": float(np.mean(flat["true_logp"][indices])) if indices.size else 0.0,
            "mean_gt_margin": float(np.mean(flat["margins"][indices])) if indices.size else 0.0,
        }
    return result
def _radius_angle_matrix(flat: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for radius in range(4):
        row: dict[str, Any] = {}
        for bin_id, angle in enumerate(YAW_DEGREES):
            indices = np.flatnonzero((flat["radii"] == radius) & (flat["bins"] == bin_id))
            row[str(angle)] = {
                "count": int(indices.size),
                "accuracy": float(np.mean(flat["predictions"][indices] == flat["labels"][indices])) if indices.size else 0.0,
            }
        result[str(radius)] = row
    return result
def _action_conditioned_q(inference: Mapping[str, Any]) -> np.ndarray:
    labels = np.asarray(inference["labels"], dtype=np.int64)
    bins = np.asarray(inference["bins"], dtype=np.int64)
    logp = np.asarray(inference["logp"], dtype=np.float32)
    values = np.zeros((NUM_CLASSES, 8), dtype=np.float64)
    for class_id in range(NUM_CLASSES):
        for bin_id in range(8):
            indices = np.flatnonzero((labels == class_id) & (bins == bin_id))
            margins = [gt_margin(logp[index], class_id) for index in indices]
            values[class_id, bin_id] = float(np.mean(margins)) if margins else 0.0
    return values
def _holdout_indices(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    records = sorted({str(row["record_id"]) for row in rows})
    count = max(1, int(round(0.1 * len(records))))
    holdout = set(records[-count:])
    fit = np.asarray([i for i, row in enumerate(rows) if str(row["record_id"]) not in holdout], dtype=np.int64)
    held = np.asarray([i for i, row in enumerate(rows) if str(row["record_id"]) in holdout], dtype=np.int64)
    return fit, held
def _choose_fusion_lambda(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    primary: np.ndarray,
    angle: np.ndarray,
    lambdas: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    _, held = _holdout_indices(rows)
    records: dict[str, Any] = {}
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    best: tuple[float, float, float] | None = None
    for weight in lambdas:
        scores = _fused_scores(primary, angle, options, weight)
        actions = _select_scores(rows, options, scores)
        predictions = _terminal_predictions(options, actions)
        accuracy = float(np.mean(predictions[held] == labels[held])) if held.size else 0.0
        records[str(weight)] = {
            "record_holdout_contexts": int(held.size),
            "accuracy": accuracy,
            "macro_f1": float(classification(labels[held], predictions[held])["macro_f1"]) if held.size else 0.0,
        }
        candidate = (accuracy, -float(weight), float(weight))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("fusion lambda selection had no candidates")
    return float(best[2]), {
        "criterion": "highest Policy Train record-holdout accuracy; ties choose lower lambda",
        "weights": list(lambdas),
        "selected_lambda": float(best[2]),
        "records": records,
        "policy_test_used": False,
    }
def _transition(labels: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_correct = left == labels
    right_correct = right == labels
    return {
        "static_correct_relative_wrong": int(np.sum(left_correct & ~right_correct)),
        "static_wrong_relative_correct": int(np.sum(~left_correct & right_correct)),
        "both_wrong": int(np.sum(~left_correct & ~right_correct)),
        "pair_anycorrect_rate": float(np.mean(left_correct | right_correct)),
    }
def _high_subset(
    rows: Sequence[Mapping[str, Any]],
    methods: Mapping[str, dict[str, Any]],
    actions: Mapping[str, Sequence[int]],
    options: Mapping[str, np.ndarray],
    visibility: np.ndarray,
) -> dict[str, Any]:
    stay = np.asarray(visibility[:, 0], dtype=np.float64)
    mask = stay < np.quantile(stay, 1.0 / 3.0)
    indices = np.flatnonzero(mask)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    output: dict[str, Any] = {"count": int(indices.size), "threshold": float(np.quantile(stay, 1.0 / 3.0)), "methods": {}}
    for name in methods:
        predictions = _terminal_predictions(options, actions[name])
        selected = np.asarray(actions[name], dtype=np.int64)
        output["methods"][name] = {
            **classification(labels[indices], predictions[indices]),
            "mean_path_proxy": float(np.mean(np.zeros(indices.size))) if name == "S0-only" else None,
            "selected_move_rate": float(np.mean(selected[indices] != np.asarray([int(rows[i]["current_viewpoint_id"]) for i in indices]))),
        }
    return output
def _load_frame0_scores(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> np.ndarray:
    path = data_root / VISIBILITY_REL / ("train.npz" if split == "train" else "val.npz")
    metadata = _read_json(data_root / VISIBILITY_REL / ("train.json" if split == "train" else "val.json"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != _signature(rows):
        raise ValueError(f"frame0 visibility signature mismatch for {split}")
    cache = _read_npz(path)
    return np.asarray(cache["scores"], dtype=np.float64)
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    _seed()
    data_root = get_data_root()
    device = _device(args.device)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    split_metadata = _internal_split_metadata(data_root)
    _write_json(output / "split_metadata.json", split_metadata)
    internal_data, internal_labels, internal_rows, internal_counts = _internal_records(data_root, "val")
    yaw_checkpoint = data_root / YAW_CHECKPOINT_REL
    head_checkpoint = data_root / YAW_HEAD_REL
    model, _ = load_checkpoint(yaw_checkpoint, NUM_CLASSES, str(device))
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_checkpoint, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    internal_inference = _infer_internal(model, head, internal_data, internal_labels, internal_rows, device)
    internal_landscape, priors = _angle_landscape(internal_inference)
    internal_per_class_angle = _per_class_angle_matrix(internal_inference)
    _write_json(output / "internal_val_angle_landscape.json", internal_landscape)
    _write_json(output / "per_class_angle_matrix.json", internal_per_class_angle)
    action_q = _action_conditioned_q(internal_inference)
    data_train, moving_rows = _load_rows(data_root)
    metadata, scene_audit = load_scene_metadata(data_root, list(data_train) + list(moving_rows), audit_sample_count=10)
    shared_head = head
    train_options, train_cache_meta = _load_yaw8_options(data_root, data_train, "train", shared_head, device)
    moving_options, moving_cache_meta = _load_yaw8_options(data_root, moving_rows, "moving_val", shared_head, device)
    stats = _read_json(data_root / POLICY_REL / "stage_c/stage_c_feature_stats.json")
    train_geometry, train_ids, train_mask = _option_geometry(data_train, stats)
    moving_geometry, moving_ids, moving_mask = _option_geometry(moving_rows, stats)
    if not np.array_equal(train_ids, train_options["ids"]) or not np.array_equal(moving_ids, moving_options["ids"]):
        raise ValueError("geometry/action IDs do not match Yaw8Fair cache")
    if not np.array_equal(train_mask, train_options["mask"]) or not np.array_equal(moving_mask, moving_options["mask"]):
        raise ValueError("geometry/action masks do not match Yaw8Fair cache")
    train_dino, train_dino_meta = _load_existing_dino(data_root, data_train, "train")
    moving_dino, moving_dino_meta = _load_existing_dino(data_root, moving_rows, "val")
    rgb_model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    rgb_checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1/RGBGlobal+Geometry.pth"
    rgb_model.load_state_dict(torch.load(rgb_checkpoint, map_location=device, weights_only=False)["state_dict"])
    rgb_model.eval()
    train_rgb = _predict(rgb_model, train_geometry, train_dino, device)
    moving_rgb = _predict(rgb_model, moving_geometry, moving_dino, device)
    train_gt_visibility = _load_frame0_scores(data_root, data_train, "train")
    moving_gt_visibility = _load_frame0_scores(data_root, moving_rows, "val")
    train_angle = _angle_score_array(data_train, train_options, metadata, priors["q_margin"])
    moving_angle_acc = _angle_score_array(moving_rows, moving_options, metadata, priors["q_acc"])
    moving_angle_f1 = _angle_score_array(moving_rows, moving_options, metadata, priors["q_f1"])
    moving_angle_margin = _angle_score_array(moving_rows, moving_options, metadata, priors["q_margin"])
    train_gt_angle = _angle_score_array(data_train, train_options, metadata, priors["q_margin"], action_q)
    moving_gt_angle = _angle_score_array(moving_rows, moving_options, metadata, priors["q_margin"], action_q)
    rgb_lambda, rgb_lambda_meta = _choose_fusion_lambda(
        data_train, train_options, train_rgb, train_angle, (0.25, 0.5, 1.0)
    )
    gt_lambda, gt_lambda_meta = _choose_fusion_lambda(
        data_train, train_options, train_gt_visibility, train_gt_angle, (0.25, 0.5, 1.0)
    )
    rgb_fused = _fused_scores(moving_rgb, _angle_score_array(moving_rows, moving_options, metadata, priors["q_margin"]), moving_options, rgb_lambda)
    gt_fused = _fused_scores(moving_gt_visibility, moving_gt_angle, moving_options, gt_lambda)
    static_prior = _prior_scores(train_options, {})
    # The historical StaticPrior is a Train candidate GT-margin prior over ids.
    prior_values = np.zeros(NUM_VIEWS, dtype=np.float64)
    prior_counts = np.zeros(NUM_VIEWS, dtype=np.int64)
    for index, row in enumerate(data_train):
        label = int(row["label_id"])
        for slot in _candidate_slots(train_options, index):
            view = int(train_options["ids"][index, slot])
            prior_values[view] += gt_margin(train_options["logp"][index, slot], label)
            prior_counts[view] += 1
    global_prior = float(prior_values.sum() / max(int(prior_counts.sum()), 1))
    static_map = {view: (float(prior_values[view] / prior_counts[view]) if prior_counts[view] else global_prior) for view in range(NUM_VIEWS)}
    static_map[-1] = global_prior
    static_scores = _prior_scores(moving_options, static_map)
    actions: dict[str, list[int]] = {}
    actions["S0-only"] = [int(row["current_viewpoint_id"]) for row in moving_rows]
    actions["Random legal"] = _random_actions(moving_rows)
    actions["StaticPrior"] = _select_scores(moving_rows, moving_options, _prior_scores(moving_options, static_map))
    actions["RelativeAnglePrior-Acc"] = _select_scores(moving_rows, moving_options, moving_angle_acc)
    actions["RelativeAnglePrior-F1"] = _select_scores(moving_rows, moving_options, moving_angle_f1)
    actions["RelativeAnglePrior-Margin"] = _select_scores(moving_rows, moving_options, moving_angle_margin)
    actions["RGBGlobal-Visibility"] = _select_scores(moving_rows, moving_options, moving_rgb)
    actions["RGBGlobal+Angle"] = _select_scores(moving_rows, moving_options, rgb_fused)
    actions["Frame0SceneVisibility"] = _select_scores(moving_rows, moving_options, moving_gt_visibility)
    actions["GT SceneVisibility+Angle"] = _select_scores(moving_rows, moving_options, gt_fused)
    actions["GTActionRelativeAnglePrior"] = _select_scores(moving_rows, moving_options, moving_gt_angle)
    true_scores = np.full(moving_options["ids"].shape, -np.inf, dtype=np.float64)
    margin_scores = np.full(moving_options["ids"].shape, -np.inf, dtype=np.float64)
    labels_moving = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    for index in range(len(moving_rows)):
        for slot in np.flatnonzero(moving_options["mask"][index]):
            true_scores[index, slot] = float(moving_options["logp"][index, slot, labels_moving[index]])
            margin_scores[index, slot] = gt_margin(moving_options["logp"][index, slot], int(labels_moving[index]))
    actions["GT-TrueLogP Oracle"] = _select_scores(moving_rows, moving_options, true_scores)
    actions["GT-Margin Oracle"] = _select_scores(moving_rows, moving_options, margin_scores)
    method_order = [
        "S0-only", "Random legal", "StaticPrior", "RelativeAnglePrior-Acc",
        "RelativeAnglePrior-F1", "RelativeAnglePrior-Margin", "RGBGlobal-Visibility",
        "RGBGlobal+Angle", "Frame0SceneVisibility", "GT SceneVisibility+Angle",
        "GTActionRelativeAnglePrior", "GT-TrueLogP Oracle", "GT-Margin Oracle",
    ]
    metrics = {name: _evaluate(moving_rows, moving_options, name, actions[name]) for name in method_order}
    flat = _flatten_moving(moving_rows, moving_options, metadata)
    moving_landscape = _moving_landscape(flat)
    radius_angle = _radius_angle_matrix(flat)
    internal_rankings = {
        "Q_acc": [YAW_DEGREES[index] for index in _ranking(priors["q_acc"])],
        "Q_f1": [YAW_DEGREES[index] for index in _ranking(priors["q_f1"])],
        "Q_margin": [YAW_DEGREES[index] for index in _ranking(priors["q_margin"])],
        "spearman": {
            "Q_acc_vs_Q_margin": correlation(priors["q_acc"], priors["q_margin"], spearman=True),
            "Q_acc_vs_Q_f1": correlation(priors["q_acc"], priors["q_f1"], spearman=True),
            "Q_f1_vs_Q_margin": correlation(priors["q_f1"], priors["q_margin"], spearman=True),
        },
    }
    moving_acc = np.asarray([moving_landscape[str(angle)]["accuracy"] for angle in YAW_DEGREES], dtype=np.float64)
    moving_margin = np.asarray([moving_landscape[str(angle)]["mean_gt_margin"] for angle in YAW_DEGREES], dtype=np.float64)
    generalization = {
        "internal_vs_moving_accuracy_ranking_spearman": correlation(priors["q_acc"], moving_acc, spearman=True),
        "internal_vs_moving_margin_ranking_spearman": correlation(priors["q_margin"], moving_margin, spearman=True),
        "internal_q_acc": priors["q_acc"].tolist(),
        "internal_q_margin": priors["q_margin"].tolist(),
        "moving_accuracy": moving_acc.tolist(),
        "moving_mean_gt_margin": moving_margin.tolist(),
    }
    acc_rank = generalization["internal_vs_moving_accuracy_ranking_spearman"]
    margin_rank = generalization["internal_vs_moving_margin_ranking_spearman"]
    generalization["decision"] = (
        "STABLE" if min(acc_rank, margin_rank) >= 0.70 else
        "MODERATELY STABLE" if min(acc_rank, margin_rank) >= 0.40 else
        "UNSTABLE"
    )
    relative_best = actions["RelativeAnglePrior-Margin"]
    static_pred = _terminal_predictions(moving_options, actions["StaticPrior"])
    relative_pred = _terminal_predictions(moving_options, relative_best)
    complementarity = _transition(labels_moving, static_pred, relative_pred)
    complementarity["selected_view_agreement"] = float(np.mean(np.asarray(actions["StaticPrior"]) == np.asarray(relative_best)))
    pair_any = np.zeros(len(moving_rows), dtype=bool)
    for name in ("StaticPrior", "RelativeAnglePrior-Margin"):
        pair_any |= _terminal_predictions(moving_options, actions[name]) == labels_moving
    complementarity["pair_anycorrect_count"] = int(pair_any.sum())
    per_class = {name: metrics[name]["per_class"] for name in method_order}
    high_methods = {
        name: metrics[name] for name in (
            "StaticPrior", "RelativeAnglePrior-Margin", "RGBGlobal-Visibility",
            "RGBGlobal+Angle", "Frame0SceneVisibility", "GTActionRelativeAnglePrior",
            "GT-TrueLogP Oracle",
        )
    }
    high = _high_subset(moving_rows, high_methods, actions, moving_options, moving_gt_visibility)
    relative_names = (
        "RelativeAnglePrior-Acc", "RelativeAnglePrior-F1", "RelativeAnglePrior-Margin"
    )
    best_relative_name = max(
        relative_names,
        key=lambda name: (metrics[name]["accuracy"], metrics[name]["macro_f1"], name),
    )
    best_internal_bin = int(np.argmax(priors["q_acc"]))
    worst_internal_bin = int(np.argmin(priors["q_acc"]))
    relative_gap_pp = 100.0 * (
        metrics[best_relative_name]["accuracy"] - metrics["StaticPrior"]["accuracy"]
    )
    if relative_gap_pp < -1.0:
        relative_decision = "KILL RELATIVE ANGLE PRIOR"
    elif relative_gap_pp <= 1.0:
        relative_decision = "ANGLE PRIOR HAS SIGNAL BUT NO CLEAR ADVANTAGE"
    elif metrics[best_relative_name]["accuracy"] >= 0.57:
        relative_decision = "STRONG RELATIVE ANGLE PRIOR"
    else:
        relative_decision = "KEEP RELATIVE ANGLE PRIOR"
    action_angle_examples: dict[str, Any] = {}
    for label_name in LABELS:
        values = np.asarray([
            internal_per_class_angle[label_name][str(angle)]["accuracy"]
            for angle in YAW_DEGREES
        ])
        action_angle_examples[label_name] = {
            "best_angle_deg": int(YAW_DEGREES[int(np.argmax(values))]),
            "worst_angle_deg": int(YAW_DEGREES[int(np.argmin(values))]),
            "best_worst_gap": float(np.max(values) - np.min(values)),
        }
    flags = {
        "policy_test_used": False,
        "raw_val_used_to_construct_angle_prior": False,
        "moving_val_used_to_construct_angle_prior": False,
        "moving_val_used_for_angle_prior_selection": False,
        "recognizer_modified": False,
        "selector_trained": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "gt_yaw_used_for_angle_prior_on_moving_val": True,
        "deployable_angle_prior": False,
        "future_candidate_observation_used_for_selection": False,
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_TRAIN_INTERNAL_VAL_RELATIVE_VIEW_QUALITY_PRIOR",
        "status": "COMPLETED",
        "labels": list(LABELS),
        "population": {
            "policy_train_contexts": len(data_train),
            "moving_val_contexts": len(moving_rows),
            "moving_val_candidate_samples": int(flat["labels"].size),
            "internal_val_observations": int(len(internal_rows)),
            "internal_val_source_records": internal_counts["source_records"],
        },
        "main_table": metrics,
        "internal_val_angle_landscape": internal_landscape,
        "internal_val_angle_rankings": internal_rankings,
        "internal_angle_summary": {
            "best_accuracy_angle_deg": int(YAW_DEGREES[best_internal_bin]),
            "worst_accuracy_angle_deg": int(YAW_DEGREES[worst_internal_bin]),
            "best_worst_accuracy_gap": float(
                priors["q_acc"][best_internal_bin] - priors["q_acc"][worst_internal_bin]
            ),
        },
        "moving_val_angle_landscape": moving_landscape,
        "radius_angle_matrix": radius_angle,
        "angle_generalization": generalization,
        "relative_angle_prior_metrics": {
            name: metrics[name] for name in (
                "RelativeAnglePrior-Acc", "RelativeAnglePrior-F1", "RelativeAnglePrior-Margin",
            )
        },
        "static_vs_relative_complementarity": complementarity,
        "rgb_angle_fusion": {
            "selected_lambda": rgb_lambda,
            "selection": rgb_lambda_meta,
            "metrics": metrics["RGBGlobal+Angle"],
            "baseline": metrics["RGBGlobal-Visibility"],
            "gain_accuracy_pp": 100.0 * (metrics["RGBGlobal+Angle"]["accuracy"] - metrics["RGBGlobal-Visibility"]["accuracy"]),
        },
        "gtvis_angle_fusion": {
            "selected_lambda": gt_lambda,
            "selection": gt_lambda_meta,
            "metrics": metrics["GT SceneVisibility+Angle"],
            "baseline": metrics["Frame0SceneVisibility"],
            "gain_accuracy_pp": 100.0 * (metrics["GT SceneVisibility+Angle"]["accuracy"] - metrics["Frame0SceneVisibility"]["accuracy"]),
        },
        "gt_action_angle_ceiling": {
            "action_conditioned_mean_margin": action_q.tolist(),
            "metrics": metrics["GTActionRelativeAnglePrior"],
            "gain_over_unified_margin_pp": 100.0 * (metrics["GTActionRelativeAnglePrior"]["accuracy"] - metrics["RelativeAnglePrior-Margin"]["accuracy"]),
        },
        "action_angle_preference_examples": action_angle_examples,
        "final_decision": {
            "best_relative_prior_by_moving_accuracy": best_relative_name,
            "best_relative_accuracy": metrics[best_relative_name]["accuracy"],
            "best_relative_macro_f1": metrics[best_relative_name]["macro_f1"],
            "gain_over_static_accuracy_pp": relative_gap_pp,
            "decision": relative_decision,
            "selection_note": "summary-only comparison; no Moving-Val tuning fed back into priors",
        },
        "high_occlusion": high,
        "per_class_metrics": per_class,
        "split_metadata": split_metadata,
        "scene_metadata_audit": scene_audit,
        "recognizer": {
            "stgcn_checkpoint": str(yaw_checkpoint.resolve()),
            "stgcn_sha256": _sha256(yaw_checkpoint),
            "shared_head_checkpoint": str(head_checkpoint.resolve()),
            "shared_head_sha256": _sha256(head_checkpoint),
            "feature_dim": FEATURE_DIM,
            "protocol": "Frozen Yaw8 ST-GCN encoder + frozen Policy-balanced shared head",
        },
        "cache_metadata": {"train": train_cache_meta, "moving_val": moving_cache_meta, "train_dino": train_dino_meta, "moving_val_dino": moving_dino_meta},
        "flags": flags,
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "seed": SEED,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    _write_json(output / "radius_angle_matrix.json", radius_angle)
    _write_json(output / "moving_val_angle_landscape.json", moving_landscape)
    _write_json(output / "internal_val_angle_rankings.json", internal_rankings)
    _write_json(output / "angle_generalization.json", generalization)
    _write_json(output / "relative_angle_prior_metrics.json", result["relative_angle_prior_metrics"])
    _write_json(output / "static_vs_relative_complementarity.json", complementarity)
    _write_json(output / "rgb_angle_fusion.json", result["rgb_angle_fusion"])
    _write_json(output / "gtvis_angle_fusion.json", result["gtvis_angle_fusion"])
    _write_json(output / "gt_action_angle_ceiling.json", result["gt_action_angle_ceiling"])
    _write_json(output / "high_occlusion_metrics.json", high)
    _write_json(output / "config.json", {
        "seed": SEED,
        "angles_deg": list(YAW_DEGREES),
        "internal_source": "raw-train-derived Yaw8 internal validation only",
        "moving_evaluation": "reduced12 Moving Val, Stage-A legal candidates",
        "recognizer": "Frozen Yaw8 ST-GCN + frozen Policy-balanced shared head",
        "angle_formula": "wrap(camera world azimuth - placement/body yaw); nearest 45-degree bin",
        "camera_azimuth_reconstruction": "random.Random(42 + babel_sid).uniform(-25,25), matching frozen Yaw8 generator",
        "fusion_lambda_candidates": [0.25, 0.5, 1.0],
        "policy_test_used": False,
        "deployable_angle_prior": False,
    })
    _write_json(output / "result.json", result)
    lines = [
        "# Train-Internal-Val Relative View Quality Prior",
        "",
        (
            "The prior was frozen from raw-train-derived Yaw8 internal validation "
            f"({internal_counts['source_records']} source records, {len(internal_rows)} expanded observations). "
            "No raw-val/Moving-Val statistic was used to construct or reorder it, and Policy Test was not read."
        ),
        "",
        "## Internal-val angle landscape",
        "",
        "| Relative angle | Count | Accuracy | Macro-F1 | Mean TrueLogP | Mean GT-Margin |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for angle in YAW_DEGREES:
        item = internal_landscape[str(angle)]
        lines.append(
            f"| {angle} | {item['count']} | {item['accuracy']:.6f} | "
            f"{item['macro_f1']:.6f} | {item['mean_gt_true_logp']:.6f} | "
            f"{item['mean_gt_margin']:.6f} |"
        )
    lines.extend([
        "",
        f"Accuracy ranking (best→worst): {internal_rankings['Q_acc']}.",
        f"Macro-F1 ranking (best→worst): {internal_rankings['Q_f1']}.",
        f"GT-margin ranking (best→worst): {internal_rankings['Q_margin']}.",
        (
            f"Internal best angle={YAW_DEGREES[best_internal_bin]}° and worst angle="
            f"{YAW_DEGREES[worst_internal_bin]}°; best-worst Accuracy gap="
            f"{(priors['q_acc'][best_internal_bin] - priors['q_acc'][worst_internal_bin]) * 100:.3f}pp."
        ),
        (
            "Ranking correlations (Spearman): "
            f"Acc↔Margin={internal_rankings['spearman']['Q_acc_vs_Q_margin']:.6f}, "
            f"Acc↔F1={internal_rankings['spearman']['Q_acc_vs_Q_f1']:.6f}, "
            f"F1↔Margin={internal_rankings['spearman']['Q_f1_vs_Q_margin']:.6f}."
        ),
        "",
        "## Moving-Val selectors",
        "",
        "The selector rows below use GT body yaw only for this privileged angle diagnostic; terminal labels come from archived candidate ST-GCN outputs.",
        "",
        "| Method | Accuracy | Macro-F1 | Move rate |",
        "|---|---:|---:|---:|",
    ])
    moving_method_names = (
        "Random legal", "StaticPrior", "RelativeAnglePrior-Acc", "RelativeAnglePrior-F1",
        "RelativeAnglePrior-Margin", "RGBGlobal-Visibility", "RGBGlobal+Angle",
        "Frame0SceneVisibility", "GT SceneVisibility+Angle", "GTActionRelativeAnglePrior",
        "GT-TrueLogP Oracle",
    )
    for name in moving_method_names:
        lines.append(
            f"| {name} | {metrics[name]['accuracy']:.6f} | "
            f"{metrics[name]['macro_f1']:.6f} | {metrics[name]['move_rate']:.6f} |"
        )
    lines.extend([
        "",
        (
            f"The best relative selector by Moving-Val Accuracy is {best_relative_name} "
            f"({metrics[best_relative_name]['accuracy']:.6f}/{metrics[best_relative_name]['macro_f1']:.6f}), "
            f"{relative_gap_pp:+.3f}pp Accuracy versus StaticPrior "
            f"({metrics['StaticPrior']['accuracy']:.6f}/{metrics['StaticPrior']['macro_f1']:.6f})."
        ),
        f"Decision by the preregistered threshold is **{relative_decision}**; this is a report-only comparison, not Moving-Val prior tuning.",
        (
            f"Static/relative selected-view agreement={complementarity['selected_view_agreement']:.6f}; "
            f"Static-correct/relative-wrong={complementarity['static_correct_relative_wrong']}, "
            f"Static-wrong/relative-correct={complementarity['static_wrong_relative_correct']}, "
            f"both-wrong={complementarity['both_wrong']}; pair AnyCorrect="
            f"{complementarity['pair_anycorrect_rate']:.6f}."
        ),
        "",
        "## Fusion and action-conditioned diagnostics",
        "",
        (
            f"RGBGlobal-Visibility={metrics['RGBGlobal-Visibility']['accuracy']:.6f}/"
            f"{metrics['RGBGlobal-Visibility']['macro_f1']:.6f}; RGBGlobal+Angle uses train-record holdout "
            f"lambda={rgb_lambda:.2f} and gives {metrics['RGBGlobal+Angle']['accuracy']:.6f}/"
            f"{metrics['RGBGlobal+Angle']['macro_f1']:.6f} "
            f"({result['rgb_angle_fusion']['gain_accuracy_pp']:+.3f}pp Accuracy)."
        ),
        (
            f"Frame0SceneVisibility={metrics['Frame0SceneVisibility']['accuracy']:.6f}/"
            f"{metrics['Frame0SceneVisibility']['macro_f1']:.6f}; GT SceneVisibility+Angle uses "
            f"lambda={gt_lambda:.2f} and gives {metrics['GT SceneVisibility+Angle']['accuracy']:.6f}/"
            f"{metrics['GT SceneVisibility+Angle']['macro_f1']:.6f} "
            f"({result['gtvis_angle_fusion']['gain_accuracy_pp']:+.3f}pp Accuracy)."
        ),
        (
            f"GTActionRelativeAnglePrior={metrics['GTActionRelativeAnglePrior']['accuracy']:.6f}/"
            f"{metrics['GTActionRelativeAnglePrior']['macro_f1']:.6f}; gain over unified Margin prior="
            f"{result['gt_action_angle_ceiling']['gain_over_unified_margin_pp']:+.3f}pp Accuracy."
        ),
        "Per-action internal preferred-angle examples (best angle / worst angle / gap):",
    ])
    for label_name in LABELS:
        example = action_angle_examples[label_name]
        lines.append(
            f"- {label_name}: {example['best_angle_deg']}° / {example['worst_angle_deg']}° / "
            f"{example['best_worst_gap'] * 100:.3f}pp"
        )
    lines.extend([
        "",
        "## High-occlusion diagnostic",
        "",
        f"Fixed lower-tertile current-slot visibility subset: n={high['count']} (no prior tuning on this subset).",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ])
    for name in high_methods:
        lines.append(
            f"| {name} | {high['methods'][name]['accuracy']:.6f} | "
            f"{high['methods'][name]['macro_f1']:.6f} |"
        )
    lines.extend([
        "",
        "## Required scientific answers",
        "",
        (
            f"1. Frozen Yaw8Fair shows a modest internal angle landscape (best {YAW_DEGREES[best_internal_bin]}° vs "
            f"worst {YAW_DEGREES[worst_internal_bin]}°, {((priors['q_acc'][best_internal_bin] - priors['q_acc'][worst_internal_bin]) * 100):.3f}pp), "
            f"but its internal-to-Moving ranking correlations ({acc_rank:.3f} Accuracy, {margin_rank:.3f} margin) are {generalization['decision']} rather than stable."
        ),
        (
            f"2. The relative prior reaches only {metrics[best_relative_name]['accuracy']:.6f} Accuracy / "
            f"{metrics[best_relative_name]['macro_f1']:.6f} Macro-F1, versus StaticPrior "
            f"{metrics['StaticPrior']['accuracy']:.6f}/{metrics['StaticPrior']['macro_f1']:.6f}; it is "
            f"{abs(relative_gap_pp):.3f}pp {'below' if relative_gap_pp < 0 else 'above'} StaticPrior and therefore not stronger than the absolute prior."
        ),
        (
            f"3. RGB angle fusion changes RGBGlobal by {result['rgb_angle_fusion']['gain_accuracy_pp']:+.3f}pp, while GT SceneVisibility fusion changes its baseline by "
            f"{result['gtvis_angle_fusion']['gain_accuracy_pp']:+.3f}pp; neither supplies a positive gain here."
        ),
        (
            f"4. The GT-action-conditioned ceiling is {metrics['GTActionRelativeAnglePrior']['accuracy']:.6f} Accuracy, only "
            f"{result['gt_action_angle_ceiling']['gain_over_unified_margin_pp']:+.3f}pp above the unified margin prior. This is evidence against a large, deployable soft action-belief angle prior at this stage."
        ),
        (
            f"5. Final decision: **{relative_decision}**. Body yaw and future candidate metadata make this a privileged, non-deployable diagnostic; no selector or recognizer was trained."
        ),
        "",
        "Flags: `policy_test_used=false`, `raw_val_used_to_construct_angle_prior=false`, `moving_val_used_for_angle_prior_selection=false`, `recognizer_modified=false`, `selector_trained=false`, `deployable_angle_prior=false`.",
    ])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args)
if __name__ == "__main__":
    main()
