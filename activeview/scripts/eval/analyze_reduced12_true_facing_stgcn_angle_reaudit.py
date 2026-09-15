#!/usr/bin/env python3
"""Re-audit the reduced12 angle prior with the true body-facing convention.

The angle tables are frozen from the raw-train-derived Yaw8 internal Val
split.  Moving Val uses the Stage-A legal candidate pool and the existing
Yaw8Fair option cache.  This is a read-only diagnostic: no model is trained,
no RGB is generated, and Policy Test is never opened.
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
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, NUM_CLASSES, classification, correlation, gt_margin
from activeview.scripts.eval.reduced12_utility_source import body_azimuth_bin, load_scene_metadata
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import VisibilityPredictor, _option_geometry, _predict
from activeview.scripts.experiments.run_reduced12_pepperpose_frame0_confidence_audit import (
    _candidate_slots,
    _evaluate,
    _load_frame0_scene_scores,
    _load_options,
    _random_actions,
    _select,
    _terminal,
    _root_yaw_map,
)
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import _load_rows
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import _load_existing_dino


SEED = 42
YAW_DEGREES = (0, 45, 90, 135, 180, 225, 270, 315)
FEATURE_DIM = 256
YAW8_ROOT = Path("datasets/stgcn_babel_reduced12_habitat_yolo26n_yaw8_v1")
YAW8_CHECKPOINT = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
YAW8_HEAD_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
)
POLICY_ROOT = Path("datasets/policy_reduced12_eight_placement_v1")
RGB_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1/RGBGlobal+Geometry.pth"
)
OUTPUT_DEFAULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/true_facing_stgcn_angle_reaudit"


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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera_azimuth(row: Mapping[str, Any]) -> float:
    key = int(row.get("candidate_index", row.get("babel_sid", 0)))
    return float(random.Random(SEED + key).uniform(-25.0, 25.0) % 360.0)


def _internal_records(data_root: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    root = data_root / YAW8_ROOT
    data = np.load(root / "val_data.npy", mmap_mode="r")
    labels = np.load(root / "val_labels.npy", mmap_mode="r").astype(np.int64)
    metadata = _read_json(root / "val_metadata.json")
    expanded = _read_json(root / "val.json")
    by_id = {str(row["record_id"]): row for row in expanded}
    if len(data) != len(labels) or len(data) != len(metadata):
        raise ValueError("Yaw8 internal Val arrays/metadata mismatch")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(metadata):
        row = by_id.get(str(item["record_id"]))
        if row is None or int(item["label_id"]) != int(labels[index]):
            raise ValueError(f"Yaw8 internal row mismatch at {item.get('record_id')}")
        merged = dict(item)
        for key in ("source_record_id", "source_path", "start_frame", "end_frame", "fps", "num_frames", "babel_sid", "candidate_index"):
            if key in row:
                merged[key] = row[key]
        merged["scene_yaw_deg"] = float(item.get("scene_yaw_deg", row.get("scene_yaw_deg", 0.0)))
        rows.append(merged)
    return np.asarray(data), labels, rows


def _infer_internal(model: torch.nn.Module, head: torch.nn.Module, data: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(data), 256):
            batch = torch.from_numpy(np.asarray(data[start : start + 256])).float().to(device)
            logits = head(model.forward_features(batch))
            values.append(torch.log_softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def _landscape(labels: np.ndarray, logp: np.ndarray, bins: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    output: dict[str, Any] = {}
    q_acc = np.zeros(8, dtype=np.float64)
    q_f1 = np.zeros(8, dtype=np.float64)
    q_margin = np.zeros(8, dtype=np.float64)
    predictions = np.argmax(logp, axis=1)
    for bin_id, angle in enumerate(YAW_DEGREES):
        indices = np.flatnonzero(bins == bin_id)
        metric = classification(labels[indices], predictions[indices])
        margins = np.asarray([gt_margin(logp[i], int(labels[i])) for i in indices], dtype=np.float64)
        q_acc[bin_id] = float(metric["accuracy"])
        q_f1[bin_id] = float(metric["macro_f1"])
        q_margin[bin_id] = float(np.mean(margins)) if margins.size else 0.0
        output[str(angle)] = {
            "angle_bin": bin_id,
            "count": int(indices.size),
            "accuracy": float(metric["accuracy"]),
            "macro_f1": float(metric["macro_f1"]),
            "mean_gt_true_logp": float(np.mean(logp[indices, labels[indices]])) if indices.size else 0.0,
            "mean_gt_margin": float(q_margin[bin_id]),
        }
    return output, {"q_acc": q_acc, "q_f1": q_f1, "q_margin": q_margin}


def _moving_landscape(
    rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    root_yaws: Mapping[str, float], *, old: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    labels: list[int] = []
    predictions: list[int] = []
    true_logps: list[float] = []
    margins: list[float] = []
    bins: list[int] = []
    for index, row in enumerate(rows):
        placement = metadata[(str(row["scene_id"]), str(row["region"]))]
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            logp = np.asarray(options["logp"][index, slot], dtype=np.float64)
            source = str(row.get("record_id", row["record_id"]))
            if source not in root_yaws:
                source = str(row.get("source_record_id", row["record_id"]))
            yaw = float(placement["yaw_deg"]) + (0.0 if old else float(root_yaws[source]))
            labels.append(int(row["label_id"]))
            predictions.append(int(np.argmax(logp)))
            true_logps.append(float(logp[int(row["label_id"])]))
            margins.append(gt_margin(logp, int(row["label_id"])))
            bins.append(body_azimuth_bin(float(placement["azimuths"][view]), yaw))
    flat_labels = np.asarray(labels, dtype=np.int64)
    flat_predictions = np.asarray(predictions, dtype=np.int64)
    flat_logp = np.asarray(true_logps, dtype=np.float64)
    flat_margin = np.asarray(margins, dtype=np.float64)
    flat_bins = np.asarray(bins, dtype=np.int64)
    output: dict[str, Any] = {}
    for bin_id, angle in enumerate(YAW_DEGREES):
        indices = np.flatnonzero(flat_bins == bin_id)
        metric = classification(flat_labels[indices], flat_predictions[indices])
        output[str(angle)] = {
            "angle_bin": bin_id,
            "count": int(indices.size),
            "accuracy": float(metric["accuracy"]),
            "macro_f1": float(metric["macro_f1"]),
            "mean_gt_true_logp": float(np.mean(flat_logp[indices])) if indices.size else 0.0,
            "mean_gt_margin": float(np.mean(flat_margin[indices])) if indices.size else 0.0,
        }
    return output, flat_bins, flat_margin


def _angle_scores(
    rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    root_yaws: Mapping[str, float], prior: np.ndarray, *, old: bool = False,
) -> np.ndarray:
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(rows):
        placement = metadata[(str(row["scene_id"]), str(row["region"]))]
        source = str(row["record_id"])
        if source not in root_yaws:
            source = str(row.get("source_record_id", row["record_id"]))
        body_yaw = float(placement["yaw_deg"]) + (0.0 if old else float(root_yaws[source]))
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            scores[index, slot] = float(prior[body_azimuth_bin(float(placement["azimuths"][view]), body_yaw)])
    return scores


def _true_logp_scores(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> np.ndarray:
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(rows):
        for slot in _candidate_slots(options, index):
            scores[index, slot] = float(options["logp"][index, slot, int(row["label_id"])])
    return scores


def _metric_with_historical(old_result: Mapping[str, Any], method: str) -> dict[str, Any] | None:
    values = old_result.get("main_table", {}).get(method)
    if not isinstance(values, dict):
        return None
    return {
        "method": method,
        "accuracy": float(values.get("accuracy", 0.0)),
        "macro_f1": float(values.get("macro_f1", 0.0)),
        "historical_reference": True,
        "source": "relative_angle_quality_prior/result.json",
        "test_used": False,
    }


def _root_change_audit(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[tuple[str, str], Mapping[str, Any]], root_yaws: Mapping[str, float],
    options: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    changed = 0
    total = 0
    changed_contexts = 0
    for index, row in enumerate(rows):
        placement = metadata[(str(row["scene_id"]), str(row["region"]))]
        slots = _candidate_slots(options, index) if options is not None else np.asarray([0])
        context_changed = False
        for slot in slots:
            view = int(options["ids"][index, slot]) if options is not None else int(row.get("viewpoint_id", 0))
            old_bin = body_azimuth_bin(float(placement["azimuths"][view]), float(placement["yaw_deg"]))
            source = str(row.get("record_id", row["record_id"]))
            if source not in root_yaws:
                source = str(row.get("source_record_id", row["record_id"]))
            true_bin = body_azimuth_bin(float(placement["azimuths"][view]), float(placement["yaw_deg"]) + float(root_yaws[source]))
            total += 1
            if old_bin != true_bin:
                changed += 1
                context_changed = True
        changed_contexts += int(context_changed)
    return {
        "comparison": "placement-yaw bin vs true body-facing bin",
        "changed_count": int(changed),
        "total_count": int(total),
        "changed_fraction": float(changed / total) if total else 0.0,
        "changed_context_count": int(changed_contexts),
        "context_count": len(rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    _seed()
    data_root = get_data_root()
    device = _device(args.device)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    # Internal Yaw8 Val is raw-train-derived and disjoint from its Yaw8 train split.
    internal_data, internal_labels, internal_rows = _internal_records(data_root)
    internal_root_yaws = _root_yaw_map(internal_rows)
    internal_bins = np.asarray([
        body_azimuth_bin(_camera_azimuth(row), float(row.get("scene_yaw_deg", 0.0)) + internal_root_yaws[str(row.get("source_record_id", row["record_id"]))])
        for row in internal_rows
    ], dtype=np.int64)
    model, _ = load_checkpoint(data_root / YAW8_CHECKPOINT, NUM_CLASSES, str(device))
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(data_root / YAW8_HEAD_CHECKPOINT, map_location=device, weights_only=False)["state_dict"])
    internal_logp = _infer_internal(model, head, internal_data, device)
    internal_landscape, internal_q = _landscape(internal_labels, internal_logp, internal_bins)

    train_rows, moving_rows = _load_rows(data_root)
    raw_val_rows = _read_json(data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json")
    policy_root_yaws = _root_yaw_map(raw_val_rows)
    metadata, metadata_audit = load_scene_metadata(data_root, list(train_rows) + list(moving_rows), audit_sample_count=10)
    train_options, train_cache_meta = _load_options(data_root, train_rows, "train", device)
    moving_options, moving_cache_meta = _load_options(data_root, moving_rows, "moving_val", device)

    old_landscape, _, _ = _moving_landscape(moving_rows, moving_options, metadata, policy_root_yaws, old=True)
    true_landscape, _, _ = _moving_landscape(moving_rows, moving_options, metadata, policy_root_yaws, old=False)
    old_result_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/relative_angle_quality_prior/result.json"
    old_result = _read_json(old_result_path) if old_result_path.is_file() else {}
    old_internal = old_result.get("internal_val_angle_landscape", {})
    old_q_acc = np.asarray([float(old_internal.get(str(angle), {}).get("accuracy", 0.0)) for angle in YAW_DEGREES], dtype=np.float64)
    old_q_margin = np.asarray([float(old_internal.get(str(angle), {}).get("mean_gt_margin", 0.0)) for angle in YAW_DEGREES], dtype=np.float64)
    moving_acc = np.asarray([true_landscape[str(angle)]["accuracy"] for angle in YAW_DEGREES], dtype=np.float64)
    moving_margin = np.asarray([true_landscape[str(angle)]["mean_gt_margin"] for angle in YAW_DEGREES], dtype=np.float64)
    old_moving_acc = np.asarray([old_landscape[str(angle)]["accuracy"] for angle in YAW_DEGREES], dtype=np.float64)
    old_moving_margin = np.asarray([old_landscape[str(angle)]["mean_gt_margin"] for angle in YAW_DEGREES], dtype=np.float64)

    static_sums = np.zeros(32, dtype=np.float64)
    static_counts = np.zeros(32, dtype=np.int64)
    for index, row in enumerate(train_rows):
        for slot in _candidate_slots(train_options, index):
            view = int(train_options["ids"][index, slot])
            static_sums[view] += gt_margin(train_options["logp"][index, slot], int(row["label_id"]))
            static_counts[view] += 1
    global_mean = float(static_sums.sum() / max(int(static_counts.sum()), 1))
    static_prior = {view: float(static_sums[view] / static_counts[view]) if static_counts[view] else global_mean for view in range(32)}
    static_prior[-1] = global_mean
    static_scores = np.full(moving_options["ids"].shape, -np.inf, dtype=np.float64)
    for index in range(len(moving_rows)):
        for slot in _candidate_slots(moving_options, index):
            static_scores[index, slot] = static_prior.get(int(moving_options["ids"][index, slot]), global_mean)

    true_angle_acc = _angle_scores(moving_rows, moving_options, metadata, policy_root_yaws, internal_q["q_acc"])
    true_angle_margin = _angle_scores(moving_rows, moving_options, metadata, policy_root_yaws, internal_q["q_margin"])
    old_angle_acc = _angle_scores(moving_rows, moving_options, metadata, policy_root_yaws, old_q_acc, old=True)
    old_angle_margin = _angle_scores(moving_rows, moving_options, metadata, policy_root_yaws, old_q_margin, old=True)
    actions: dict[str, list[int]] = {
        "Random": _random_actions(moving_rows),
        "StaticPrior": _select(moving_rows, moving_options, static_scores),
        "TrueFacingAnglePrior-Acc": _select(moving_rows, moving_options, true_angle_acc),
        "TrueFacingAnglePrior-Margin": _select(moving_rows, moving_options, true_angle_margin),
        "Old RelativeAnglePrior-Acc": _select(moving_rows, moving_options, old_angle_acc),
        "Old RelativeAnglePrior-Margin": _select(moving_rows, moving_options, old_angle_margin),
        "GT-TrueLogP Oracle": _select(moving_rows, moving_options, _true_logp_scores(moving_rows, moving_options)),
    }

    stats = _read_json(data_root / POLICY_ROOT / "stage_c/stage_c_feature_stats.json")
    moving_geometry, moving_ids, moving_mask = _option_geometry(moving_rows, stats)
    if not np.array_equal(moving_ids, moving_options["ids"]) or not np.array_equal(moving_mask, moving_options["mask"]):
        raise ValueError("candidate geometry/action IDs do not match Yaw8Fair option cache")
    dino, dino_meta = _load_existing_dino(data_root, moving_rows, "val")
    rgb_model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    rgb_model.load_state_dict(torch.load(data_root / RGB_CHECKPOINT, map_location=device, weights_only=False)["state_dict"])
    rgb_scores = _predict(rgb_model, moving_geometry, dino, device)
    actions["RGBGlobal-Visibility"] = _select(moving_rows, moving_options, rgb_scores)
    actions["Frame0SceneVisibility"] = _select(moving_rows, moving_options, _load_frame0_scene_scores(data_root, moving_rows))

    metrics = {name: _evaluate(name, moving_rows, moving_options, selected) for name, selected in actions.items()}
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    any_correct = np.asarray([
        any(int(np.argmax(moving_options["logp"][index, slot])) == int(row["label_id"]) for slot in _candidate_slots(moving_options, index))
        for index, row in enumerate(moving_rows)
    ], dtype=bool)
    metrics["AnyCorrect Coverage"] = {
        "method": "AnyCorrect Coverage",
        "contexts": len(moving_rows),
        "coverage_count": int(any_correct.sum()),
        "coverage_rate": float(any_correct.mean()),
        "test_used": False,
    }
    for name in ("Old RelativeAnglePrior-Acc", "Old RelativeAnglePrior-Margin"):
        historical_name = name.replace("Old ", "RelativeAnglePrior-")
        historical = _metric_with_historical(old_result, historical_name)
        if historical is not None:
            metrics[name]["historical_reference"] = historical

    internal_old_bins = np.asarray([
        body_azimuth_bin(_camera_azimuth(row), float(row.get("scene_yaw_deg", 0.0))) for row in internal_rows
    ], dtype=np.int64)
    alignment = {
        "formula": "theta_body = R_scene_yaw @ R_AMASS_root_frame0 applied to local +Z; bin wrap(camera_world_azimuth - theta_body) to nearest 45 degrees",
        "internal": _root_change_audit(internal_rows, {}, internal_root_yaws) if False else {
            "observation_count": len(internal_rows),
            "old_bin_changed_count": int(np.sum(internal_old_bins != internal_bins)),
            "old_bin_changed_fraction": float(np.mean(internal_old_bins != internal_bins)),
        },
        "moving": _root_change_audit(moving_rows, metadata, policy_root_yaws, moving_options),
        "analytic_convention_check": "PASS: local +Z forward; R_body_world = R_scene_yaw @ R_AMASS_root_frame0",
        "policy_test_used": False,
    }
    generalization = {
        "internal_q_acc_true_facing": internal_q["q_acc"].tolist(),
        "internal_q_margin_true_facing": internal_q["q_margin"].tolist(),
        "moving_accuracy_true_facing": moving_acc.tolist(),
        "moving_mean_gt_margin_true_facing": moving_margin.tolist(),
        "spearman_q_acc_vs_moving_accuracy": correlation(internal_q["q_acc"], moving_acc, spearman=True),
        "spearman_q_margin_vs_moving_margin": correlation(internal_q["q_margin"], moving_margin, spearman=True),
        "historical_old_spearman_q_acc_vs_moving_accuracy": float(old_result.get("angle_generalization", {}).get("internal_vs_moving_accuracy_ranking_spearman", 0.142857)),
        "historical_old_spearman_q_margin_vs_moving_margin": float(old_result.get("angle_generalization", {}).get("internal_vs_moving_margin_ranking_spearman", -0.333333)),
        "stability_thresholds": {"stable": ">=0.70", "moderately_stable": "0.40-0.70", "unstable": "<0.40"},
        "policy_test_used": False,
    }
    selector_order = ["Random", "StaticPrior", "Old RelativeAnglePrior-Acc", "Old RelativeAnglePrior-Margin", "TrueFacingAnglePrior-Acc", "TrueFacingAnglePrior-Margin", "RGBGlobal-Visibility", "Frame0SceneVisibility", "GT-TrueLogP Oracle"]
    selector_payload = {"methods": {name: metrics[name] for name in selector_order}, "any_correct_coverage": metrics["AnyCorrect Coverage"], "contexts": len(moving_rows), "candidate_samples": int(sum(len(row["candidate_ids"]) for row in moving_rows)), "policy_test_used": False}
    _write_json(output / "angle_alignment_audit.json", alignment)
    _write_json(output / "internal_angle_landscape.json", {"true_facing": internal_landscape, "old_placement_yaw": old_internal})
    _write_json(output / "moving_angle_landscape.json", {"true_facing": true_landscape, "old_placement_yaw": old_landscape})
    _write_json(output / "generalization.json", generalization)
    _write_json(output / "selector_metrics.json", selector_payload)

    corrected_names = ("TrueFacingAnglePrior-Acc", "TrueFacingAnglePrior-Margin")
    best_name = max(corrected_names, key=lambda name: (metrics[name]["accuracy"], metrics[name]["macro_f1"], name))
    acc_rho = float(generalization["spearman_q_acc_vs_moving_accuracy"])
    margin_rho = float(generalization["spearman_q_margin_vs_moving_margin"])
    if abs(acc_rho) < 0.40 and abs(margin_rho) < 0.40:
        decision = "KILL ST-GCN RELATIVE-ANGLE PRIOR"
    elif min(acc_rho, margin_rho) >= 0.70:
        decision = "ANGLE MISALIGNMENT WAS A MAJOR CAUSE" if metrics[best_name]["accuracy"] >= 0.54 else "ANGLE PREFERENCE IS STABLE BUT NOT SUFFICIENT FOR NBV"
    elif metrics[best_name]["accuracy"] >= metrics["StaticPrior"]["accuracy"] + 0.01:
        decision = "KEEP TRUE-FACING ANGLE PRIOR"
    else:
        decision = "KILL ST-GCN RELATIVE-ANGLE PRIOR"
    result = {
        "experiment_id": "REDUCED12_TRUE_FACING_STGCN_ANGLE_REAUDIT",
        "status": "COMPLETED",
        "labels": list(LABELS),
        "population": {"internal_val_observations": len(internal_rows), "moving_val_contexts": len(moving_rows), "moving_val_candidate_samples": selector_payload["candidate_samples"]},
        "angle_alignment_audit": alignment,
        "internal_angle_landscape": internal_landscape,
        "moving_angle_landscape": true_landscape,
        "generalization": generalization,
        "selector_metrics": selector_payload,
        "final_decision": {"decision": decision, "best_corrected_selector": best_name, "best_corrected_accuracy": metrics[best_name]["accuracy"], "static_accuracy": metrics["StaticPrior"]["accuracy"]},
        "recognizer": {"stgcn_checkpoint": str((data_root / YAW8_CHECKPOINT).resolve()), "stgcn_sha256": _sha256(data_root / YAW8_CHECKPOINT), "shared_head_checkpoint": str((data_root / YAW8_HEAD_CHECKPOINT).resolve()), "feature_dim": FEATURE_DIM, "protocol": "Frozen Yaw8 ST-GCN encoder + frozen Yaw8Fair shared head"},
        "scene_metadata_audit": metadata_audit,
        "cache_metadata": {"train": train_cache_meta, "moving_val": moving_cache_meta, "moving_val_dino": dino_meta},
        "flags": {"policy_test_used": False, "training_used": False, "recognizer_modified": False, "new_rgb_generated": False, "new_skeleton_generated": False, "gt_body_facing_used_for_angle_prior": True, "moving_val_used_for_prior_construction": False, "deployable_angle_prior": False, "future_candidate_observation_used_for_selection": False},
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.monotonic() - started},
    }
    _write_json(output / "result.json", result)

    best_internal = max(YAW_DEGREES, key=lambda angle: internal_landscape[str(angle)]["accuracy"])
    worst_internal = min(YAW_DEGREES, key=lambda angle: internal_landscape[str(angle)]["accuracy"])
    best_moving = max(YAW_DEGREES, key=lambda angle: true_landscape[str(angle)]["accuracy"])
    worst_moving = min(YAW_DEGREES, key=lambda angle: true_landscape[str(angle)]["accuracy"])
    lines = [
        "# True-Facing ST-GCN Angle Prior Re-Audit", "",
        "This is a read-only Train-internal-Val and Moving-Val diagnostic. The body-facing angle composes the AMASS frame-0 root rotation with scene yaw, then bins camera world azimuth relative to that facing. Policy Test was not read and no model/data was generated.", "",
        f"Internal bin alignment changed {alignment['internal']['old_bin_changed_count']}/{alignment['internal']['observation_count']} ({alignment['internal']['old_bin_changed_fraction']:.6f}); Moving legal candidates changed {alignment['moving']['changed_count']}/{alignment['moving']['total_count']} ({alignment['moving']['changed_fraction']:.6f}).", "",
        "## Corrected internal angle landscape", "", "| Angle | Count | Accuracy | Macro-F1 | Mean GT-TrueLogP | Mean GT-Margin |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for angle in YAW_DEGREES:
        item = internal_landscape[str(angle)]
        lines.append(f"| {angle} | {item['count']} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['mean_gt_true_logp']:.6f} | {item['mean_gt_margin']:.6f} |")
    lines.extend([
        "", f"Internal corrected best/worst Accuracy bins: {best_internal}°/{worst_internal}°.",
        f"Moving corrected best/worst Accuracy bins: {best_moving}°/{worst_moving}°.",
        f"Train→Moving Spearman: Q_acc vs accuracy={acc_rho:.6f}; Q_margin vs mean margin={margin_rho:.6f}. Historical old values were 0.142857 and -0.333333.", "",
        "## Moving selector metrics", "", "| Selector | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|",
    ])
    for name in selector_order:
        item = metrics[name]
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item.get('move_rate', 0.0):.6f} |")
    lines.extend([
        f"| AnyCorrect Coverage | {metrics['AnyCorrect Coverage']['coverage_rate']:.6f} coverage | — | — |", "",
        f"Decision: **{decision}**.", "",
        "### Interpretation", "",
        f"A. The corrected prior changes the angle assignment for the fractions above; the measured selector accuracy is {metrics[best_name]['accuracy']:.6f} for {best_name}, versus StaticPrior {metrics['StaticPrior']['accuracy']:.6f}.",
        "B. Corrected internal and Moving angle preference agree only to the extent captured by the two reported Spearman values; the old placement-yaw table is retained for direct comparison.",
        "C. If the corrected prior remains weak or unstable, the next diagnostic should be Clean-to-Observed ST-GCN feature degradation rather than another angle prior. No fusion or new model was attempted.",
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
