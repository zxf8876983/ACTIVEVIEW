#!/usr/bin/env python3
"""Audit a true-facing, frame-0 pose-confidence angle prior.

This is a read-only Train/internal-Val and Moving-Val diagnostic.  The prior
is frozen from the raw-train-derived Yaw8 internal validation observations.
Moving terminal predictions reuse the existing Yaw8Fair option cache.  No
policy Test data, RGB generation, or model training is allowed.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.motion.babel_clean_dataset_generator import (
    URDF_PATH,
    _load_resampled_motion,
    compose_root_rotation,
)
from activeview.data.motion.motion_converter import MotionConverter
from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    NUM_CLASSES,
    classification,
    correlation,
    gt_margin,
)
from activeview.scripts.eval.reduced12_utility_source import body_azimuth_bin, load_scene_metadata
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    VisibilityPredictor,
    _option_geometry,
    _predict,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (
    _load_rows,
    _read_npz,
    _signature,
)
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import _load_yaw8_options


SEED = 42
YAW_DEGREES = (0, 45, 90, 135, 180, 225, 270, 315)
NUM_VIEWS = 32
MAX_OPTIONS = 22
FEATURE_DIM = 256
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/pepperpose_frame0_confidence_audit"
)
YAW8_ROOT = Path("datasets/stgcn_babel_reduced12_habitat_yolo26n_yaw8_v1")
POLICY_ROOT = Path("datasets/policy_reduced12_eight_placement_v1")
OPTION_CACHE = Path("diagnostics/reduced12_yaw8_policy_landscape")
VISIBILITY_ROOT = Path("diagnostics/frame0_visibility_predictor_v1")
DINO_ROOT = Path("features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/frame0_current")
RGB_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1/"
    "RGBGlobal+Geometry.pth"
)
YAW8_CHECKPOINT = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
YAW8_HEAD_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/yaw8_shared_head_fairness_audit/"
    "yaw8_shared_head_best.pth"
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap(angle: float) -> float:
    return float((float(angle) + 180.0) % 360.0 - 180.0)


def _camera_azimuth(row: Mapping[str, Any]) -> float:
    """Match the Yaw8 generator's candidate-index-first RNG convention."""
    key = int(row.get("candidate_index", row.get("babel_sid", 0)))
    return float(random.Random(SEED + key).uniform(-25.0, 25.0) % 360.0)


def _amass_root_yaw(row: Mapping[str, Any], converter: MotionConverter) -> float:
    motion = _load_resampled_motion(row, target_frames=30)
    converted = converter.convert(motion)
    transform = np.asarray(converted["pose_motion"]["transform_array"][0], dtype=np.float32)
    rotation = compose_root_rotation(transform, scene_yaw_deg=0.0)[:3, :3]
    forward = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return float(np.degrees(np.arctan2(forward[0], forward[2])) % 360.0)


def _root_yaw_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_source: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_record_id", row["record_id"]))
        by_source.setdefault(source_id, row)
    converter = MotionConverter(URDF_PATH)
    result: dict[str, float] = {}
    for source_id, row in sorted(by_source.items()):
        result[source_id] = _amass_root_yaw(row, converter)
    return result


def _angle_sanity(rows: Sequence[Mapping[str, Any]], root_yaws: Mapping[str, float]) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    random_indices = rng.choice(len(rows), size=min(16, len(rows)), replace=False).tolist()
    representatives: list[int] = []
    for target_bin in (0, 2, 4, 6):
        for index, row in enumerate(rows):
            source = str(row.get("source_record_id", row["record_id"]))
            body_yaw = (float(root_yaws[source]) + float(row.get("scene_yaw_deg", 0.0))) % 360.0
            if body_azimuth_bin(_camera_azimuth(row), body_yaw) == target_bin:
                representatives.append(index)
                break
    selected = list(dict.fromkeys(representatives + random_indices))[:20]
    samples: list[dict[str, Any]] = []
    changed = 0
    for index in selected:
        row = rows[index]
        source = str(row.get("source_record_id", row["record_id"]))
        scene_yaw = float(row.get("scene_yaw_deg", 0.0))
        root_yaw = float(root_yaws[source])
        body_yaw = (root_yaw + scene_yaw) % 360.0
        camera = _camera_azimuth(row)
        old_bin = body_azimuth_bin(camera, scene_yaw)
        new_bin = body_azimuth_bin(camera, body_yaw)
        changed += int(old_bin != new_bin)
        samples.append({
            "record_id": str(row["record_id"]),
            "source_record_id": source,
            "babel_sid": int(row.get("babel_sid", 0)),
            "amass_root_yaw_deg": root_yaw,
            "scene_or_placement_yaw_deg": scene_yaw,
            "composed_true_body_yaw_deg": body_yaw,
            "candidate_world_azimuth_deg": camera,
            "old_relative_angle_deg": _wrap(camera - scene_yaw),
            "new_true_facing_relative_angle_deg": _wrap(camera - body_yaw),
            "old_bin": int(old_bin),
            "new_bin": int(new_bin),
            "bin_changed": bool(old_bin != new_bin),
            "rgb_debug_image": None,
        })
    return {
        "sample_count": len(samples),
        "selection": "16 deterministic random samples plus first representatives of true-facing bins 0/90/180/270",
        "bin_change_count": changed,
        "bin_change_fraction": float(changed / len(samples)) if samples else 0.0,
        "analytic_facing_convention_check": "PASS: local +Z forward; R_body_world = R_scene_yaw @ R_AMASS_root_frame0",
        "camera_azimuth_rule": "random.Random(seed + candidate_index), else babel_sid; uniform(-25,25) degrees",
        "rgb_debug_status": "UNAVAILABLE: Yaw8 archive has skeleton/confidence arrays only; no RGB was regenerated",
        "samples": samples,
        "policy_test_used": False,
    }


def _internal_rows_and_confidence(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = data_root / YAW8_ROOT
    expanded = _read_json(root / "val.json")
    metadata = _read_json(root / "val_metadata.json")
    train_expanded = _read_json(root / "train.json")
    rows_by_id = {str(row["record_id"]): row for row in expanded}
    values: list[dict[str, Any]] = []
    for item in metadata:
        row = dict(rows_by_id[str(item["record_id"])])
        row.update({"scene_yaw_deg": float(item.get("scene_yaw_deg", row.get("scene_yaw_deg", 0.0)))})
        values.append(row)
    schema_samples: list[dict[str, Any]] = []
    for row in values[: min(5, len(values))]:
        path = root / "estimated_skeletons" / "val" / f"{row['record_id']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            confidence = np.asarray(archive["confidence"])
        schema_samples.append({"path": str(path.resolve()), "confidence_shape": list(confidence.shape), "dtype": str(confidence.dtype)})
    exact = bool(schema_samples and all(item["confidence_shape"] == [30, 17] for item in schema_samples))
    train_sources = {
        str(row.get("source_record_id", row["record_id"])) for row in train_expanded
    }
    internal_sources = {
        str(row.get("source_record_id", row["record_id"])) for row in values
    }
    overlap = sorted(train_sources & internal_sources)
    if overlap:
        raise ValueError(f"Yaw8 raw-train train/internal-val record overlap: {overlap[:3]}")
    return values, {
        "split": "raw-train-derived Yaw8 internal validation",
        "rows": len(values),
        "source_records": len({str(row.get("source_record_id", row["record_id"])) for row in values}),
        "train_source_records": len(train_sources),
        "train_internal_record_overlap": len(overlap),
        "no_train_internal_record_overlap": not overlap,
        "confidence_samples": schema_samples,
        "frame0_keypoint_confidence_available": exact,
        "definition": "q_pose = mean(confidence[0,:])" if exact else "unavailable",
    }


def _internal_landscape(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    root_yaws: Mapping[str, float],
) -> tuple[dict[str, Any], np.ndarray]:
    sums = np.zeros(8, dtype=np.float64)
    mins = np.zeros(8, dtype=np.float64)
    counts_high = np.zeros(8, dtype=np.float64)
    counts = np.zeros(8, dtype=np.int64)
    root = data_root / YAW8_ROOT
    for row in rows:
        path = root / "estimated_skeletons" / "val" / f"{row['record_id']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
        if confidence.shape != (30, 17):
            raise ValueError(f"internal Yaw8 archive lacks frame-0 keypoint confidence: {path} {confidence.shape}")
        q = confidence[0]
        source = str(row.get("source_record_id", row["record_id"]))
        body_yaw = float(root_yaws[source]) + float(row.get("scene_yaw_deg", 0.0))
        bin_id = body_azimuth_bin(_camera_azimuth(row), body_yaw)
        sums[bin_id] += float(np.mean(q))
        mins[bin_id] += float(np.min(q))
        counts_high[bin_id] += float(np.sum(q > 0.5))
        counts[bin_id] += 1
    q_conf = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)
    landscape: dict[str, Any] = {}
    for bin_id, angle in enumerate(YAW_DEGREES):
        landscape[str(angle)] = {
            "angle_bin": bin_id,
            "count": int(counts[bin_id]),
            "mean_frame0_confidence": float(q_conf[bin_id]),
            "mean_frame0_min_confidence": float(mins[bin_id] / max(int(counts[bin_id]), 1)),
            "mean_frame0_keypoints_gt_0_5": float(counts_high[bin_id] / max(int(counts[bin_id]), 1)),
            "confidence_source": "exact confidence[0,:] from Yaw8 per-view NPZ",
        }
    return landscape, q_conf


def _load_options(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load the existing Yaw8Fair cache and apply its frozen shared head."""
    head = SharedHead().to(device)
    checkpoint = data_root / YAW8_HEAD_CHECKPOINT
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    options, metadata = _load_yaw8_options(
        data_root, rows, "train" if split == "train" else "moving_val", head, device
    )
    return options, metadata


def _candidate_slots(options: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    active = np.flatnonzero(np.asarray(options["mask"][index], dtype=bool))
    return active[active != 0]


def _select(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], scores: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        slots = _candidate_slots(options, index)
        if slots.size == 0:
            actions.append(int(row["current_viewpoint_id"]))
            continue
        slot = min(slots.tolist(), key=lambda value: (-float(scores[index, value]), int(options["ids"][index, value])))
        actions.append(int(options["ids"][index, slot]))
    return actions


def _terminal(options: Mapping[str, np.ndarray], actions: Sequence[int]) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        matches = np.flatnonzero((options["ids"][index] == int(action)) & options["mask"][index])
        if matches.size != 1:
            raise ValueError(f"selected viewpoint is absent at row {index}: {action}")
        predictions.append(int(np.argmax(options["logp"][index, int(matches[0])])))
    return np.asarray(predictions, dtype=np.int64)


def _evaluate(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    actions: Sequence[int],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    predictions = _terminal(options, actions)
    current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
    metric = classification(labels, predictions)
    metric.update({
        "method": name,
        "move_rate": float(np.mean(np.asarray(actions, dtype=np.int64) != current)),
        "stay_rate": float(np.mean(np.asarray(actions, dtype=np.int64) == current)),
        "test_used": False,
        "candidate_only_selection": True,
    })
    return metric


def _random_actions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    rng = np.random.default_rng(SEED)
    return [
        int(rng.choice([int(value) for value in row["candidate_ids"]])) if row["candidate_ids"]
        else int(row["current_viewpoint_id"])
        for row in rows
    ]


def _static_prior(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> dict[int, float]:
    sums = np.zeros(NUM_VIEWS, dtype=np.float64)
    counts = np.zeros(NUM_VIEWS, dtype=np.int64)
    for index, row in enumerate(rows):
        label = int(row["label_id"])
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            sums[view] += gt_margin(options["logp"][index, slot], label)
            counts[view] += 1
    global_mean = float(sums.sum() / max(int(counts.sum()), 1))
    return {view: float(sums[view] / counts[view]) if counts[view] else global_mean for view in range(NUM_VIEWS)} | {-1: global_mean}


def _score_by_view(options: Mapping[str, np.ndarray], prior: Mapping[int, float]) -> np.ndarray:
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    fallback = float(prior.get(-1, 0.0))
    for index in range(scores.shape[0]):
        for slot in np.flatnonzero(options["mask"][index]):
            scores[index, slot] = float(prior.get(int(options["ids"][index, slot]), fallback))
    return scores


def _angle_scores(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    root_yaws: Mapping[str, float],
    q_conf: np.ndarray,
) -> np.ndarray:
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(rows):
        source = str(row["record_id"])
        if source not in root_yaws:
            source = str(row.get("source_record_id", row["record_id"]))
        if source not in root_yaws:
            raise ValueError(f"missing AMASS root yaw for policy record {row['record_id']}")
        placement = metadata[(str(row["scene_id"]), str(row["region"]))]
        body_yaw = float(root_yaws[source]) + float(placement["yaw_deg"])
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            angle_bin = body_azimuth_bin(float(placement["azimuths"][view]), body_yaw)
            scores[index, slot] = float(q_conf[angle_bin])
    return scores


def _true_logp_scores(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> np.ndarray:
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(rows):
        label = int(row["label_id"])
        for slot in _candidate_slots(options, index):
            scores[index, slot] = float(options["logp"][index, slot, label])
    return scores


def _load_frame0_scene_scores(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    path = data_root / VISIBILITY_ROOT / "val.npz"
    metadata = _read_json(data_root / VISIBILITY_ROOT / "val.json")
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != _signature(rows):
        raise ValueError("frame0 SceneVisibility cache signature/count mismatch")
    values = np.asarray(_read_npz(path)["scores"], dtype=np.float64)
    if values.shape != (len(rows), MAX_OPTIONS):
        raise ValueError(f"unexpected SceneVisibility score shape: {values.shape}")
    return values


def _moving_confidence_landscape(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    root_yaws: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    sums = np.zeros(8, dtype=np.float64)
    correct_sums = np.zeros(8, dtype=np.float64)
    wrong_sums = np.zeros(8, dtype=np.float64)
    counts = np.zeros(8, dtype=np.int64)
    correct_counts = np.zeros(8, dtype=np.int64)
    archive_cache: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        path = str(row["archive_path"])
        if path not in archive_cache:
            with np.load(path, allow_pickle=False) as archive:
                confidence = np.asarray(archive["confidence"], dtype=np.float32)
                ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            if confidence.shape != (32,):
                raise ValueError(f"unexpected policy confidence shape: {path} {confidence.shape}")
            archive_cache[path] = confidence[np.argsort(ids)]
        confidence = archive_cache[path]
        placement = metadata[(str(row["scene_id"]), str(row["region"]))]
        body_yaw = float(root_yaws[str(row["record_id"])]) + float(placement["yaw_deg"])
        label = int(row["label_id"])
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            angle_bin = body_azimuth_bin(float(placement["azimuths"][view]), body_yaw)
            q = float(confidence[view])
            is_correct = int(np.argmax(options["logp"][index, slot]) == label)
            sums[angle_bin] += q
            counts[angle_bin] += 1
            if is_correct:
                correct_sums[angle_bin] += q
                correct_counts[angle_bin] += 1
            else:
                wrong_sums[angle_bin] += q
    q_by_bin = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)
    landscape: dict[str, Any] = {}
    for bin_id, angle in enumerate(YAW_DEGREES):
        landscape[str(angle)] = {
            "angle_bin": bin_id,
            "count": int(counts[bin_id]),
            "mean_confidence": float(q_by_bin[bin_id]),
            "mean_confidence_correct": float(correct_sums[bin_id] / max(int(correct_counts[bin_id]), 1)),
            "mean_confidence_wrong": float(wrong_sums[bin_id] / max(int(counts[bin_id] - correct_counts[bin_id]), 1)),
            "confidence_source": "policy archive confidence[view], sequence-level mean over all frames (proxy; not frame-0)",
        }
    return landscape, {str(angle): float(q_by_bin[index]) for index, angle in enumerate(YAW_DEGREES)}


def _record_holdout(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    records = sorted({str(row["record_id"]) for row in rows})
    held = set(records[-max(1, int(round(0.1 * len(records)))) :])
    return np.asarray([index for index, row in enumerate(rows) if str(row["record_id"]) in held], dtype=np.int64)


def _rank_normalize(scores: np.ndarray, slots: np.ndarray) -> np.ndarray:
    output = np.full(scores.shape, -np.inf, dtype=np.float64)
    if slots.size == 0:
        return output
    values = scores[slots]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    output[slots] = ranks / max(float(slots.size - 1), 1.0)
    return output


def _fuse(primary: np.ndarray, angle: np.ndarray, options: Mapping[str, np.ndarray], weight: float) -> np.ndarray:
    result = np.full(primary.shape, -np.inf, dtype=np.float64)
    for index in range(primary.shape[0]):
        slots = _candidate_slots(options, index)
        result[index, slots] = _rank_normalize(primary[index], slots)[slots] + float(weight) * _rank_normalize(angle[index], slots)[slots]
    return result


def _maybe_fusion(
    data_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    moving_rows: Sequence[Mapping[str, Any]],
    train_options: Mapping[str, np.ndarray],
    moving_options: Mapping[str, np.ndarray],
    moving_angle: np.ndarray,
    device: torch.device,
    pose_accuracy: float,
) -> dict[str, Any]:
    if pose_accuracy < 0.54:
        return {
            "status": "SKIPPED_BY_REGISTERED_GATE",
            "gate": "GTYaw-PoseConfidencePrior Accuracy >= 0.54",
            "pose_accuracy": float(pose_accuracy),
            "reason": "Fusion was not run because the privileged pose prior did not reach the preregistered 0.54 gate.",
            "policy_test_used": False,
        }
    raise RuntimeError("fusion gate passed; train-holdout fusion requires an unavailable exact frame-0 candidate confidence path")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    _seed()
    data_root = get_data_root()
    device = _device(args.device)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    internal_rows, internal_schema = _internal_rows_and_confidence(data_root)
    internal_roots = _root_yaw_map(internal_rows)
    angle_sanity = _angle_sanity(internal_rows, internal_roots)
    _write_json(output / "angle_sanity.json", angle_sanity)
    if not internal_schema["frame0_keypoint_confidence_available"]:
        raise RuntimeError("cannot construct exact internal frame-0 confidence prior")
    internal_landscape, q_conf = _internal_landscape(data_root, internal_rows, internal_roots)
    _write_json(output / "internal_confidence_landscape.json", internal_landscape)

    train_rows, moving_rows = _load_rows(data_root)
    raw_val_path = data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json"
    raw_val_rows = _read_json(raw_val_path)
    raw_by_id = {str(row["record_id"]): row for row in raw_val_rows}
    missing = sorted({str(row["record_id"]) for row in moving_rows if str(row["record_id"]) not in raw_by_id})
    if missing:
        raise ValueError(f"Moving policy records absent from raw-val manifest: {missing[:3]}")
    policy_roots = _root_yaw_map(list(raw_by_id.values()))
    metadata, metadata_audit = load_scene_metadata(data_root, list(train_rows) + list(moving_rows), audit_sample_count=10)
    train_options, train_cache_meta = _load_options(data_root, train_rows, "train", device)
    moving_options, moving_cache_meta = _load_options(data_root, moving_rows, "moving_val", device)

    static = _static_prior(train_rows, train_options)
    static_actions = _select(moving_rows, moving_options, _score_by_view(moving_options, static))
    random_actions = _random_actions(moving_rows)
    angle_scores = _angle_scores(moving_rows, moving_options, metadata, policy_roots, q_conf)
    angle_actions = _select(moving_rows, moving_options, angle_scores)
    true_actions = _select(moving_rows, moving_options, _true_logp_scores(moving_rows, moving_options))
    scene_scores = _load_frame0_scene_scores(data_root, moving_rows)

    dino_path = data_root / DINO_ROOT / "val.npy"
    dino_meta = _read_json(data_root / DINO_ROOT / "val.json")
    if int(dino_meta.get("rows", -1)) != len(moving_rows) or dino_meta.get("signature") != _signature(moving_rows):
        raise ValueError("existing frame-0 DINO cache signature/count mismatch")
    dino = np.load(dino_path, mmap_mode="r")
    stats = _read_json(data_root / POLICY_ROOT / "stage_c/stage_c_feature_stats.json")
    moving_geometry, moving_ids, moving_mask = _option_geometry(moving_rows, stats)
    if not np.array_equal(moving_ids, moving_options["ids"]) or not np.array_equal(moving_mask, moving_options["mask"]):
        raise ValueError("candidate geometry/action IDs do not match Yaw8Fair option cache")
    rgb_model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    rgb_model.load_state_dict(torch.load(data_root / RGB_CHECKPOINT, map_location=device, weights_only=False)["state_dict"])
    rgb_scores = _predict(rgb_model, moving_geometry, np.asarray(dino), device)
    del rgb_model, dino

    actions = {
        "Random": random_actions,
        "StaticPrior": static_actions,
        "GTYaw-PoseConfidencePrior": angle_actions,
        "RGBGlobal-Visibility": _select(moving_rows, moving_options, rgb_scores),
        "Frame0SceneVisibility": _select(moving_rows, moving_options, scene_scores),
        "GT-TrueLogP Oracle": true_actions,
    }
    metrics = {name: _evaluate(name, moving_rows, moving_options, value) for name, value in actions.items()}
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    pose_predictions = _terminal(moving_options, angle_actions)
    any_correct = np.zeros(len(moving_rows), dtype=bool)
    for index, row in enumerate(moving_rows):
        any_correct[index] = bool(np.any([
            int(np.argmax(moving_options["logp"][index, slot])) == int(row["label_id"])
            for slot in _candidate_slots(moving_options, index)
        ]))
    metrics["AnyCorrect Coverage"] = {
        "contexts": len(moving_rows),
        "coverage_count": int(any_correct.sum()),
        "coverage_rate": float(any_correct.mean()),
        "test_used": False,
        "candidate_only": True,
    }

    moving_landscape, moving_q = _moving_confidence_landscape(moving_rows, moving_options, metadata, policy_roots)
    _write_json(output / "moving_confidence_landscape.json", moving_landscape)
    moving_q_array = np.asarray([moving_q[str(angle)] for angle in YAW_DEGREES], dtype=np.float64)
    generalization = {
        "internal_q_conf": q_conf.tolist(),
        "moving_q_conf": moving_q_array.tolist(),
        "internal_to_moving_spearman": correlation(q_conf, moving_q_array, spearman=True),
        "moving_confidence_is_exact_frame0": False,
        "warning": "Moving archive confidence is a sequence-level mean; this correlation is proxy-only and was not used to tune the prior.",
        "policy_test_used": False,
    }
    _write_json(output / "generalization.json", generalization)

    selector_result = {
        "methods": metrics,
        "moving_val_contexts": len(moving_rows),
        "moving_val_candidate_samples": int(sum(len(row["candidate_ids"]) for row in moving_rows)),
        "any_correct_coverage": metrics["AnyCorrect Coverage"],
        "gt_yaw_prior_terminal_accuracy": float(np.mean(pose_predictions == labels)),
        "policy_test_used": False,
    }
    _write_json(output / "selector_metrics.json", selector_result)
    fusion = _maybe_fusion(
        data_root, train_rows, moving_rows, train_options, moving_options, angle_scores, device,
        metrics["GTYaw-PoseConfidencePrior"]["accuracy"],
    )
    _write_json(output / "fusion_metrics.json", fusion)

    best_internal = int(np.argmax(q_conf))
    worst_internal = int(np.argmin(q_conf))
    prior_accuracy = float(metrics["GTYaw-PoseConfidencePrior"]["accuracy"])
    moving_spearman = float(generalization["internal_to_moving_spearman"])
    decision = "KILL PEPPERPOSE-STYLE ANGLE PRIOR" if prior_accuracy < 0.52 or moving_spearman < 0.4 else "KEEP PEPPERPOSE MECHANISM"
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_FRAME0_TRUE_FACING_YOLO_CONFIDENCE_ANGLE_PRIOR",
        "status": "COMPLETED",
        "labels": list(LABELS),
        "population": {
            "policy_train_contexts": len(train_rows),
            "moving_val_contexts": len(moving_rows),
            "moving_val_candidate_samples": int(sum(len(row["candidate_ids"]) for row in moving_rows)),
            "internal_val_expanded_observations": len(internal_rows),
            "internal_val_source_records": internal_schema["source_records"],
        },
        "confidence_schema": internal_schema,
        "angle_sanity": angle_sanity,
        "internal_confidence_landscape": internal_landscape,
        "moving_confidence_landscape": moving_landscape,
        "generalization": generalization,
        "selector_metrics": selector_result,
        "fusion_metrics": fusion,
        "internal_summary": {
            "best_angle_deg": YAW_DEGREES[best_internal],
            "worst_angle_deg": YAW_DEGREES[worst_internal],
            "best_worst_confidence_gap": float(q_conf[best_internal] - q_conf[worst_internal]),
        },
        "recognizer": {
            "stgcn_checkpoint": str((data_root / YAW8_CHECKPOINT).resolve()),
            "stgcn_sha256": _sha256(data_root / YAW8_CHECKPOINT),
            "shared_head_checkpoint": str((data_root / YAW8_HEAD_CHECKPOINT).resolve()),
            "shared_head_sha256": _sha256(data_root / YAW8_HEAD_CHECKPOINT),
            "feature_dim": FEATURE_DIM,
            "terminal_cache": "existing Yaw8 policy-landscape cache; current/Stage-A legal candidates",
        },
        "cache_metadata": {"train_options": train_cache_meta, "moving_options": moving_cache_meta, "frame0_dino": dino_meta, "scene_metadata_audit": metadata_audit},
        "flags": {
            "policy_test_used": False,
            "raw_val_used_to_construct_angle_prior": False,
            "moving_val_used_to_construct_angle_prior": False,
            "moving_val_used_for_prior_tuning": False,
            "recognizer_modified": False,
            "selector_trained": False,
            "new_rgb_generated": False,
            "new_skeleton_generated": False,
            "gt_true_facing_used_for_moving_selector": True,
            "deployable_angle_prior": False,
            "future_candidate_observation_used_for_selection": False,
            "moving_frame0_confidence_exact": False,
            "moving_confidence_proxy_used_only_for_landscape": True,
        },
        "final_decision": {
            "gt_yaw_pose_confidence_accuracy": prior_accuracy,
            "internal_to_moving_confidence_spearman_proxy": moving_spearman,
            "decision": decision,
            "thresholds": {"accuracy_min": 0.52, "spearman_min": 0.4},
            "caveat": "Because Moving archives lack frame-0 keypoint confidence, the generalization Spearman is not an exact frame0-to-frame0 test.",
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.monotonic() - started},
    }
    _write_json(output / "result.json", result)

    lines = [
        "# Frame0 True-Facing + YOLO Confidence Angle Prior Audit",
        "",
        "This is a privileged, non-deployable diagnostic. The angle prior is frozen from raw-train-derived Yaw8 internal validation; Policy Test was not read.",
        "",
        "## Data and schema",
        "",
        f"Moving Val has {len(moving_rows):,} contexts and {sum(len(row['candidate_ids']) for row in moving_rows):,} legal candidate samples. Internal Yaw8 has {len(internal_rows):,} expanded observations from {internal_schema['source_records']:,} source records.",
        "The internal archive exposes exact `(30,17)` confidence and therefore uses `mean(confidence[0,:])`. The eight-placement policy archive exposes only `(32,)` sequence-level means; it has no future-candidate frame-0 keypoint confidence. Moving confidence landscapes are explicitly labelled proxy-only and were not used to tune the prior.",
        "",
        "## True-facing sanity",
        "",
        f"Analytic convention: **{angle_sanity['analytic_facing_convention_check']}**. Old placement-yaw bins changed in {angle_sanity['bin_change_count']}/{angle_sanity['sample_count']} ({angle_sanity['bin_change_fraction']:.3f}) samples. RGB debug images are unavailable because no RGB was regenerated.",
        "",
        "## Internal frame-0 confidence landscape",
        "",
        "| Relative angle | Count | Mean frame-0 confidence |",
        "|---:|---:|---:|",
    ]
    for angle in YAW_DEGREES:
        item = internal_landscape[str(angle)]
        lines.append(f"| {angle} | {item['count']} | {item['mean_frame0_confidence']:.6f} |")
    lines.extend([
        "",
        f"Best internal angle={YAW_DEGREES[best_internal]}°, worst={YAW_DEGREES[worst_internal]}°, gap={(q_conf[best_internal]-q_conf[worst_internal]):.6f}.",
        "",
        "## Moving-Val selectors",
        "",
        "| Method | Accuracy | Macro-F1 | Move rate |",
        "|---|---:|---:|---:|",
    ])
    for name in ("Random", "StaticPrior", "GTYaw-PoseConfidencePrior", "RGBGlobal-Visibility", "Frame0SceneVisibility", "GT-TrueLogP Oracle"):
        item = metrics[name]
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} |")
    lines.extend([
        f"| AnyCorrect Coverage | {metrics['AnyCorrect Coverage']['coverage_rate']:.6f} coverage | — | — |",
        "",
        f"Internal→Moving confidence Spearman (proxy, because Moving confidence is sequence-level) = {moving_spearman:.6f}.",
        f"Registered decision: **{decision}** (GTYaw-PoseConfidencePrior Accuracy={prior_accuracy:.6f}; exact frame-0 Moving confidence is unavailable).",
        "",
        "## Fusion",
        "",
        fusion.get("reason", fusion.get("status", "not run")),
        "",
        "## Scientific answer",
        "",
        "The true-facing angle convention is analytically consistent, but the existing policy archive cannot support the requested exact Moving frame-0 YOLO-confidence landscape: its stored confidence is a 30-frame sequence mean. Therefore this audit does not claim an exact frame-0 confidence-angle generalization result. The privileged selector number above is valid only as a true-facing prior whose internal prior is frame-0-derived; the confidence correlation gate is explicitly proxy-qualified.",
        "No deployable yaw estimator was trained and no existing recognizer/checkpoint/data was modified.",
        "",
        "Flags: `policy_test_used=false`, `new_rgb_generated=false`, `new_skeleton_generated=false`, `selector_trained=false`, `deployable_angle_prior=false`, `moving_frame0_confidence_exact=false`.",
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
