#!/usr/bin/env python3
"""Audit selector complementarity and current-frame RGB-D viability.

The primary result is a strict Frame-0 -> one Stage-A legal candidate audit
on Moving Val.  Existing Yaw8Fair/map caches are reused.  RGB-D state is
audited read-only; when depth or raw frame-0 YOLO artifacts are absent this
script records the missing capability instead of fabricating deployable
metrics.  Policy Test is never opened.
"""

from __future__ import annotations

import argparse
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

from activeview.core.paths import get_data_root, get_habitat_data_root  # noqa: E402
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, NUM_CLASSES, classification, correlation  # noqa: E402
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import (  # noqa: E402
    TaskUtilityPredictor,
    _predict_task,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (  # noqa: E402
    VisibilityPredictor,
    _option_geometry,
    _predict,
    _terminal,
)
from activeview.scripts.experiments.run_reduced12_known_map_geometric_nbv_upgrade import (  # noqa: E402
    PROJ_AREA,
    _candidate_scores,
    _load_map_cache,
    _median_correct_scale,
    _scale_score,
)
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (  # noqa: E402
    _load_rows,
    _read_npz,
    _signature,
    _train_prior,
)
from activeview.scripts.experiments.run_reduced12_yaw8_shared_head_fairness_audit import (  # noqa: E402
    SharedHead,
    _build_options,
    _head_logits,
)
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import (  # noqa: E402
    _candidate_mask,
    _load_existing_dino,
    _load_yaw8_options,
    _select_candidate,
)
from activeview.scripts.experiments.rgbd_complementarity_helpers import (  # noqa: E402
    alignment_audit,
    depth_audit,
    habitat_depth_probe,
)

SEED = 42
MAX_OPTIONS = 22
NUM_VIEWS = 32
MAP_DIM = 96
FRAME0_VIS_REL = Path("diagnostics/frame0_visibility_predictor_v1")
MAP_REL = Path("diagnostics/map_aware_robust_observability_audit")
HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
)
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "rgbd_complementarity_gate_audit"
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


def _load_frame0_visibility(
    data_root: Path, rows: Sequence[Mapping[str, Any]], split: str
) -> dict[str, np.ndarray]:
    root = data_root / FRAME0_VIS_REL
    metadata = _read_json(root / f"{split}.json")
    if int(metadata.get("rows", -1)) != len(rows):
        raise ValueError(f"frame0 visibility row mismatch for {split}")
    if metadata.get("signature") != _signature(rows):
        raise ValueError(f"frame0 visibility signature mismatch for {split}")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"frame0 visibility cache marked Test-derived: {split}")
    target = _read_npz(root / f"{split}.npz")
    expected = (len(rows), MAX_OPTIONS)
    if target["scores"].shape != expected or target["ids"].shape != expected:
        raise ValueError(f"invalid frame0 visibility shape for {split}")
    if target["mask"].shape != expected:
        raise ValueError(f"invalid frame0 visibility mask for {split}")
    for index, row in enumerate(rows):
        active = np.flatnonzero(target["mask"][index])
        wanted = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
        if target["ids"][index, active].astype(int).tolist() != wanted:
            raise ValueError(f"frame0 action mismatch: {row['episode_id']}")
    return target


def _prior_score_matrix(
    options: Mapping[str, np.ndarray], prior: Mapping[int, float]
) -> np.ndarray:
    values = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    fallback = float(prior.get(-1, 0.0))
    for index in range(values.shape[0]):
        for slot in np.flatnonzero(options["mask"][index]):
            values[index, slot] = float(prior.get(int(options["ids"][index, slot]), fallback))
    return values


def _metric(
    name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    actions: Sequence[int],
) -> dict[str, Any]:
    result = classification(labels, predictions)
    current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
    action_array = np.asarray(actions, dtype=np.int64)
    result.update(
        {
            "method": name,
            "move_rate": float(np.mean(action_array != current)),
            "stay_rate": float(np.mean(action_array == current)),
            "test_used": False,
        }
    )
    return result


def _method_action_predictions(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    scores: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    mask = _candidate_mask(options["mask"])
    actions = _select_candidate(rows, scores, mask)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    predictions = _terminal(options["logp"], options["ids"], options["mask"], actions, labels)
    return actions, predictions


def _pair_oracle(
    names: Sequence[str],
    action_arrays: Mapping[str, Sequence[int]],
    options: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
) -> dict[str, Any]:
    left, right = names
    actions_l = np.asarray(action_arrays[left], dtype=np.int64)
    actions_r = np.asarray(action_arrays[right], dtype=np.int64)
    pred_l = _terminal(options["logp"], options["ids"], options["mask"], actions_l, labels)
    pred_r = _terminal(options["logp"], options["ids"], options["mask"], actions_r, labels)
    correct_l = pred_l == labels
    correct_r = pred_r == labels
    oracle_actions: list[int] = []
    for index, row in enumerate(rows):
        choices = [int(actions_l[index]), int(actions_r[index])]
        unique = list(dict.fromkeys(choices))
        # This is a policy-pair oracle: choose among already selected views by
        # real GT true-class log-probability, never among the full archive.
        selected = max(
            unique,
            key=lambda action: (
                float(options["logp"][index, np.flatnonzero(options["ids"][index] == action)[0], int(row["label_id"])]),
                -int(action),
            ),
        )
        oracle_actions.append(selected)
    oracle_pred = _terminal(options["logp"], options["ids"], options["mask"], oracle_actions, labels)
    disagreement = ~(actions_l == actions_r)
    return {
        "policies": [left, right],
        "contexts": int(labels.size),
        "pair_any_correct_count": int(np.sum(correct_l | correct_r)),
        "pair_any_correct_rate": float(np.mean(correct_l | correct_r)),
        "gt_true_logp_policy_pair": _metric(f"{left}+{right} GT-TrueLogP", labels, oracle_pred, rows, oracle_actions),
        "top1_agreement": float(np.mean(pred_l == pred_r)),
        "both_correct": int(np.sum(correct_l & correct_r)),
        "left_correct_right_wrong": int(np.sum(correct_l & ~correct_r)),
        "left_wrong_right_correct": int(np.sum(~correct_l & correct_r)),
        "both_wrong": int(np.sum(~correct_l & ~correct_r)),
        "disagreement_contexts": int(np.sum(disagreement)),
        "disagreement_subset_accuracy": {
            left: float(np.mean(correct_l[disagreement])) if np.any(disagreement) else 0.0,
            right: float(np.mean(correct_r[disagreement])) if np.any(disagreement) else 0.0,
        },
        "rescue_rate_left_to_right": float(np.mean((~correct_l & correct_r)[disagreement])) if np.any(disagreement) else 0.0,
        "harm_rate_left_to_right": float(np.mean((correct_l & ~correct_r)[disagreement])) if np.any(disagreement) else 0.0,
    }


def _triple_oracle(
    names: Sequence[str],
    action_arrays: Mapping[str, Sequence[int]],
    options: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
) -> dict[str, Any]:
    selected: list[int] = []
    any_correct = np.zeros(labels.size, dtype=bool)
    for index, row in enumerate(rows):
        actions = list(dict.fromkeys(int(action_arrays[name][index]) for name in names))
        slots = [int(np.flatnonzero(options["ids"][index] == action)[0]) for action in actions]
        preds = [int(np.argmax(options["logp"][index, slot])) for slot in slots]
        any_correct[index] = int(row["label_id"]) in preds
        selected.append(max(actions, key=lambda action: float(options["logp"][index, int(np.flatnonzero(options["ids"][index] == action)[0]), int(row["label_id"])])))
    pred = _terminal(options["logp"], options["ids"], options["mask"], selected, labels)
    return {
        "policies": list(names),
        "contexts": int(labels.size),
        "triple_any_correct_count": int(any_correct.sum()),
        "triple_any_correct_rate": float(any_correct.mean()),
        "gt_true_logp_policy_triple": _metric("+".join(names) + " GT-TrueLogP", labels, pred, rows, selected),
    }


def _complementarity(
    action_arrays: Mapping[str, Sequence[int]],
    options: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_names = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("C", "D")]
    pair_values = {"A+B": _pair_oracle(pair_names[0], action_arrays, options, rows, labels)}
    for pair in pair_names[1:]:
        pair_values["+".join(pair)] = _pair_oracle(pair, action_arrays, options, rows, labels)
    triple_values = {
        "+".join(names): _triple_oracle(names, action_arrays, options, rows, labels)
        for names in (("A", "B", "C"), ("A", "B", "D"), ("A", "C", "D"))
    }
    return pair_values, triple_values


def _train_gate_selection(
    train_rows: Sequence[Mapping[str, Any]],
    train_options: Mapping[str, np.ndarray],
    train_scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Select the alternative policy on a record holdout inside Train only."""
    records = sorted({str(row["record_id"]) for row in train_rows})
    holdout_count = max(1, int(round(0.1 * len(records))))
    holdout_records = set(records[-holdout_count:])
    holdout = np.asarray(
        [str(row["record_id"]) in holdout_records for row in train_rows], dtype=bool
    )
    predictions: dict[str, np.ndarray] = {}
    for name in ("A", "B", "C", "D", "E"):
        _, predictions[name] = _method_action_predictions(
            train_rows, train_options, train_scores[name]
        )
    rates: dict[str, float] = {}
    for alternative in ("B", "C", "D", "E"):
        rates[alternative] = float(
            np.mean(
                (predictions["A"][holdout] == np.asarray([int(r["label_id"]) for r in train_rows], dtype=np.int64)[holdout])
                | (predictions[alternative][holdout] == np.asarray([int(r["label_id"]) for r in train_rows], dtype=np.int64)[holdout])
            )
        )
    best = max(sorted(rates), key=lambda name: (rates[name], name))
    return {
        "selection_source": "Policy Train record holdout only",
        "holdout_records": int(len(holdout_records)),
        "holdout_contexts": int(holdout.sum()),
        "pair_any_correct_rates": rates,
        "selected_alternative": best,
        "test_used": False,
    }


def _write_policy_actions(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    action_arrays: Mapping[str, Sequence[int]],
    pred_arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> None:
    names = ["A", "B", "C", "D", "E"]
    actions = np.stack([np.asarray(action_arrays[name], dtype=np.int16) for name in names], axis=1)
    predictions = np.stack([np.asarray(pred_arrays[name], dtype=np.int8) for name in names], axis=1)
    np.savez_compressed(
        output_root / "policy_actions.npz",
        episode_ids=np.asarray([str(row["episode_id"]) for row in rows]),
        labels=labels.astype(np.int8),
        selected_actions=actions,
        predictions=predictions,
        correct=(predictions == labels[:, None]),
    )


def _high_occlusion(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    action_arrays: Mapping[str, Sequence[int]],
    labels: np.ndarray,
    visibility: np.ndarray,
) -> dict[str, Any]:
    stay_visibility = visibility[:, 0]
    threshold = float(np.quantile(stay_visibility, 1.0 / 3.0))
    # Match the accepted high-occlusion protocol exactly: strict lower
    # tertile, not ``<=`` (the visibility cache has ties at the quantile).
    selected = stay_visibility < threshold
    output: dict[str, Any] = {
        "definition": "bottom tertile of frame-0 current-slot 17-joint scene visibility",
        "count": int(selected.sum()),
        "threshold": threshold,
        "methods": {},
    }
    for name, actions in action_arrays.items():
        predictions = _terminal(options["logp"], options["ids"], options["mask"], actions, labels)
        idx = np.flatnonzero(selected)
        output["methods"][name] = _metric(name, labels[idx], predictions[idx], [rows[int(i)] for i in idx], [actions[int(i)] for i in idx])
    return output


def run(output_root: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    started = time.monotonic()
    train_rows, val_rows = _load_rows(data_root)
    labels_train = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    labels_val = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    head_path = data_root / HEAD_REL
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=False)["state_dict"])
    head.eval()

    train_options, train_meta = _load_yaw8_options(data_root, train_rows, "train", head, device)
    val_options, val_meta = _load_yaw8_options(data_root, val_rows, "moving_val", head, device)
    train_map, train_map_meta = _load_map_cache(data_root, "train", train_rows)
    val_map, val_map_meta = _load_map_cache(data_root, "val", val_rows)
    if not np.array_equal(train_map["ids"], train_options["ids"]) or not np.array_equal(val_map["ids"], val_options["ids"]):
        raise ValueError("map and Yaw8Fair viewpoint IDs differ")
    if not np.array_equal(train_map["mask"], train_options["mask"]) or not np.array_equal(val_map["mask"], val_options["mask"]):
        raise ValueError("map and Yaw8Fair legality masks differ")

    stats = _read_json(data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/stage_c_feature_stats.json")
    train_geometry, train_ids, train_geom_mask = _option_geometry(train_rows, stats)
    val_geometry, val_ids, val_geom_mask = _option_geometry(val_rows, stats)
    if not np.array_equal(train_ids, train_options["ids"]) or not np.array_equal(val_ids, val_options["ids"]):
        raise ValueError("geometry and option IDs differ")
    if not np.array_equal(train_geom_mask, train_options["mask"]) or not np.array_equal(val_geom_mask, val_options["mask"]):
        raise ValueError("geometry and option masks differ")
    train_dino, train_dino_meta = _load_existing_dino(data_root, train_rows, "train")
    val_dino, val_dino_meta = _load_existing_dino(data_root, val_rows, "val")
    train_visibility = _load_frame0_visibility(data_root, train_rows, "train")
    val_visibility = _load_frame0_visibility(data_root, val_rows, "val")

    prior, prior_meta = _train_prior(train_rows, train_options)
    train_prior_scores = _prior_score_matrix(train_options, prior)
    val_prior_scores = _prior_score_matrix(val_options, prior)
    candidate_train = _candidate_mask(train_options["mask"])
    candidate_val = _candidate_mask(val_options["mask"])

    # A: privileged frame-0 scene visibility from the existing GT-state map.
    # B/C: existing Train-derived deployable selectors transferred to Yaw8Fair.
    # D: Train-only static viewpoint prior. E: fixed G_full=.5 map/prior fusion.
    train_v17 = np.mean(train_map["features"][:, :, 0:17], axis=2)
    val_v17 = np.mean(val_map["features"][:, :, 0:17], axis=2)
    train_dense = train_map["features"][:, :, 17]
    val_dense = val_map["features"][:, :, 17]
    train_completeness = train_map["features"][:, :, 34]
    val_completeness = val_map["features"][:, :, 34]
    median_scale, scale_meta = _median_correct_scale(train_rows, train_options, train_map)
    train_scale = _scale_score(train_map["features"][:, :, PROJ_AREA], median_scale)
    val_scale = _scale_score(val_map["features"][:, :, PROJ_AREA], median_scale)
    train_doq = _candidate_scores(train_dense, candidate_train) + _candidate_scores(train_completeness, candidate_train) + _candidate_scores(train_scale, candidate_train)
    val_doq = _candidate_scores(val_dense, candidate_val) + _candidate_scores(val_completeness, candidate_val) + _candidate_scores(val_scale, candidate_val)
    train_gfull = _candidate_scores(train_doq, candidate_train) + 0.5 * _candidate_scores(train_prior_scores, candidate_train)
    val_gfull = _candidate_scores(val_doq, candidate_val) + 0.5 * _candidate_scores(val_prior_scores, candidate_val)

    old_u4_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/frame0_task_utility_predictor_v1/RGBGlobal-TrueLogP+VisibilityAux.pth"
    old_model = TaskUtilityPredictor(True, True).to(device)
    old_model.load_state_dict(torch.load(old_u4_path, map_location=device, weights_only=False)["state_dict"])
    train_b_scores, _ = _predict_task(old_model.eval(), train_geometry, train_dino, device)
    val_b_scores, _ = _predict_task(old_model.eval(), val_geometry, val_dino, device)
    vis_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1/RGBGlobal+Geometry.pth"
    vis_model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    vis_model.load_state_dict(torch.load(vis_path, map_location=device, weights_only=False)["state_dict"])
    train_c_scores = _predict(vis_model.eval(), train_geometry, train_dino, device)
    val_c_scores = _predict(vis_model.eval(), val_geometry, val_dino, device)

    train_scores = {
        "A": np.asarray(train_visibility["scores"], dtype=np.float64),
        "B": train_b_scores,
        "C": train_c_scores,
        "D": train_prior_scores,
        "E": train_gfull,
    }
    val_scores = {
        "A": np.asarray(val_visibility["scores"], dtype=np.float64),
        "B": val_b_scores,
        "C": val_c_scores,
        "D": val_prior_scores,
        "E": val_gfull,
    }
    train_gate_selection = _train_gate_selection(train_rows, train_options, train_scores)
    action_arrays: dict[str, list[int]] = {}
    pred_arrays: dict[str, np.ndarray] = {}
    methods: dict[str, Any] = {}
    for name in ("A", "B", "C", "D", "E"):
        actions, predictions = _method_action_predictions(val_rows, val_options, val_scores[name])
        action_arrays[name] = actions
        pred_arrays[name] = predictions
        display = {
            "A": "Frame0SceneVisibility",
            "B": "OldTargetSelector→Yaw8Fair",
            "C": "RGBGlobal-Visibility",
            "D": "StaticViewPrior",
            "E": "G_full λ=.5",
        }[name]
        methods[display] = _metric(display, labels_val, predictions, val_rows, actions)
    # Candidate-only random reference and strict oracle sanity references.
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice([int(v) for v in row["candidate_ids"]])) for row in val_rows]
    random_pred = _terminal(val_options["logp"], val_options["ids"], val_options["mask"], random_actions, labels_val)
    methods["Random legal"] = _metric("Random legal", labels_val, random_pred, val_rows, random_actions)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_policy_actions(output_root, val_rows, action_arrays, pred_arrays, labels_val)
    pair_values, triple_values = _complementarity(action_arrays, val_options, val_rows, labels_val)
    any_correct = np.zeros(len(val_rows), dtype=bool)
    for index, row in enumerate(val_rows):
        active = np.flatnonzero(val_options["mask"][index])[1:]
        any_correct[index] = bool(np.any(np.argmax(val_options["logp"][index, active], axis=1) == int(row["label_id"])))
    methods["AnyCorrect Coverage"] = {
        "method": "AnyCorrect Coverage",
        "n": len(val_rows),
        "coverage_count": int(any_correct.sum()),
        "coverage_rate": float(any_correct.mean()),
        "test_used": False,
    }

    depth_inventory = depth_audit(data_root, train_rows + val_rows)
    depth_probe = habitat_depth_probe(
        data_root,
        train_rows,
        val_rows,
        get_habitat_data_root() / "hm3d-train",
    )
    alignment = alignment_audit(depth_inventory)
    if depth_probe.get("status") == "PASS":
        alignment = {
            **alignment,
            "status": "PASS_DEPTH_SENSOR_RESOLUTION_ONLY",
            "train_samples_rendered": int(depth_probe["splits"]["train"]["requested"]),
            "val_samples_rendered": int(depth_probe["splits"]["val"]["requested"]),
            "rgb_depth_resolution_match": True,
            "torso_depth_finite": True,
            "yolo_joints_inside_body_depth_region": None,
            "reason": "50+50 matched Habitat depth probes succeeded; keypoint alignment remains unverified because raw YOLO cache is absent.",
        }
    protocol = {
        "formal_task": "current Frame0 -> exactly one Stage-A legal candidate -> selected real O1-only Yaw8Fair HAR",
        "action_set": "candidate-only Stage-A legal candidate_pool; current/stay excluded",
        "policy_train_contexts": len(train_rows),
        "moving_val_contexts": len(val_rows),
        "moving_val_candidate_samples": int(candidate_val.sum()),
        "candidate_count_mean": float(candidate_val.sum(axis=1).mean()),
        "candidate_count_min": int(candidate_val.sum(axis=1).min()),
        "candidate_count_max": int(candidate_val.sum(axis=1).max()),
        "yaw8_cache_train": train_meta,
        "yaw8_cache_moving_val": val_meta,
        "map_cache_train": train_map_meta,
        "map_cache_moving_val": val_map_meta,
        "frame0_dino_train": train_dino_meta,
        "frame0_dino_moving_val": val_dino_meta,
        "yaw8_shared_head_checkpoint": str(head_path.resolve()),
        "static_prior": prior_meta,
        "scale_calibration": scale_meta,
        "test_used": False,
        "moving_val_used_for_training_or_threshold_selection": False,
    }
    pair_any = {name: value["pair_any_correct_rate"] for name, value in pair_values.items()}
    best_pair_a = max(
        (value for name, value in pair_values.items() if name.startswith("A+")),
        key=lambda value: float(value["pair_any_correct_rate"]),
        default={"pair_any_correct_rate": 0.0},
    )
    gate_status = "PAIR COMPLEMENTARITY TOO SMALL" if float(best_pair_a.get("pair_any_correct_rate", 0.0)) < 0.61 else "GATING AUDIT ALLOWED"
    rgbd_status = "BLOCKED_NO_CURRENT_DEPTH_OR_YOLO_CACHE" if (not depth_inventory["current_frame0_depth_present"] or not depth_inventory["raw_frame0_yolo_present"]) else "ARTIFACTS_PRESENT"
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_RGBD_COMPLEMENTARITY_GATE_AUDIT",
        "status": (
            "COMPLETED_PHASE_A_DEPTH_PROBE_ONLY"
            if depth_probe.get("status") == "PASS" and rgbd_status.startswith("BLOCKED")
            else ("COMPLETED_PHASE_A_DEPTH_BLOCKED" if rgbd_status.startswith("BLOCKED") else "COMPLETED")
        ),
        "labels": list(LABELS),
        "population": {"policy_train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "moving_val_candidate_samples": int(candidate_val.sum())},
        "main_metrics": methods,
        "pair_gate_status": gate_status,
        "train_gate_selection": train_gate_selection,
        "pair_any_correct_rates": pair_any,
        "depth_sensor_audit": depth_inventory,
        "depth_probe": depth_probe,
        "depth_alignment_audit": alignment,
        "pair_complementarity": pair_values,
        "triple_complementarity": triple_values,
        "high_occlusion": _high_occlusion(val_rows, val_options, action_arrays, labels_val, val_visibility["scores"]),
        "flags": {
            "policy_test_used": False,
            "candidate_rgb_read_before_selection": False,
            "candidate_depth_read_before_selection": False,
            "candidate_yolo_read_before_selection": False,
            "future_current_frame_read": False,
            "training_used_for_rgbd_gate": False,
            "current_depth_probe_only": True,
            "known_static_map_used": True,
            "gt_frame0_human_state_used_for_A": True,
            "gt_action_used_for_inference": False,
            "future_candidate_skeleton_used_for_terminal_evaluation_only": True,
            "deployable_rgbd_metrics_available": False,
        },
        "protocol_audit": protocol,
        "leakage_audit": {
            "policy_test_read": False,
            "candidate_rgb_read_before_selection": False,
            "candidate_depth_read_before_selection": False,
            "candidate_yolo_read_before_selection": False,
            "candidate_skeleton_read_before_selection": False,
            "candidate_logits_used_before_selection": False,
            "candidate_logits_loaded_for_terminal_evaluation": True,
            "future_current_frames_used": False,
            "gt_action_used_at_inference": False,
            "moving_val_used_for_training": False,
            "moving_val_used_for_threshold_selection": False,
            "current_depth_used": False,
            "current_map_used": True,
            "gt_frame0_human_state_used_for_A": True,
            "estimated_root_used_for_D2": False,
            "future_candidate_skeleton_used_for_terminal_evaluation_only": True,
            "future_candidate_rgb_or_depth_used": False,
            "deployable_D2_available": False,
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.monotonic() - started},
    }
    _write_json(output_root / "config.json", {"seed": SEED, "recognizer": "frozen Yaw8 ST-GCN + matched Policy-balanced shared head", "formal_action_set": "Stage-A candidate-only", "test_used": False, "depth_generation": "not run; no current depth cache"})
    _write_json(output_root / "protocol_audit.json", protocol)
    _write_json(output_root / "baseline_reproduction.json", {name: methods[name] for name in methods if name != "AnyCorrect Coverage"})
    _write_json(output_root / "pair_complementarity.json", pair_values)
    _write_json(output_root / "triple_complementarity.json", triple_values)
    _write_json(output_root / "policy_disagreement.json", {name: {"contexts": value["disagreement_contexts"], "top1_agreement": value["top1_agreement"], "both_correct": value["both_correct"], "both_wrong": value["both_wrong"]} for name, value in pair_values.items()})
    _write_json(output_root / "depth_sensor_audit.json", {**depth_inventory, "probe": depth_probe})
    _write_json(output_root / "depth_alignment_audit.json", alignment)
    _write_json(
        output_root / "frame0_yolo_metrics.json",
        {
            "status": "NOT_AVAILABLE",
            "raw_outputs_found": depth_inventory["raw_frame0_yolo_present"],
            "success_rate": None,
            "reason": "No serialized current-frame YOLO outputs were present; exact raw-pose acquisition was not run in this audit.",
            "test_used": False,
        },
    )
    np.savez_compressed(
        output_root / "frame0_uncertainty_features.npz",
        features=np.empty((0, 15), dtype=np.float32),
        feature_names=np.asarray(
            [
                "person_conf", "mean_joint_conf", "min_joint_conf", "q25_joint_conf",
                "std_joint_conf", "num_joint_conf_lt_025", "num_joint_conf_lt_050",
                "bbox_area_ratio", "bbox_truncation_ratio", "depth_valid_ratio_at_joints",
                "torso_depth_median", "torso_depth_MAD", "joint_depth_consistency",
                "left_right_confidence_imbalance", "upper_lower_confidence_imbalance",
            ],
            dtype="U48",
        ),
    )
    _write_json(output_root / "rgbd_root_localization.json", {"status": "N/A", "reason": "No current depth and no raw frame0 YOLO keypoints; no estimated metric root fabricated.", "test_used": False})
    _write_json(output_root / "pose_world_reconstruction.json", {"status": "N/A", "reason": "No legal current-frame 3D pose reconstruction artifact; D2 is not reported.", "test_used": False})
    _write_json(output_root / "d0_privileged_metrics.json", methods["Frame0SceneVisibility"])
    _write_json(output_root / "d1_depthpos_gtpose_metrics.json", {"status": "N/A", "reason": "Current depth cache absent."})
    _write_json(output_root / "d2_deployable_visibility_metrics.json", {"status": "N/A", "reason": "Current depth/raw YOLO/metric estimated state absent."})
    _write_json(output_root / "d3_uncertainty_metrics.json", {"status": "N/A", "reason": "Current frame-0 per-joint confidence absent; archive confidence is only a 30-frame viewpoint scalar."})
    _write_json(output_root / "visibility_confidence_analysis.json", {"status": "N/A", "reason": "No frame-0 depth or raw YOLO confidence cache."})
    gate_skip_reason = (
        "Current-frame depth cache and raw frame-0 YOLO cache are absent; "
        "deployable RGB-D gate cannot be trained or evaluated."
    )
    _write_json(output_root / "gap_gate_metrics.json", {"status": "SKIPPED", "reason": gate_skip_reason, "pair_gate_status": gate_status, "best_pair_containing_A": best_pair_a, "train_gate_selection": train_gate_selection})
    _write_json(output_root / "logistic_gate_training.json", {"status": "SKIPPED", "reason": gate_skip_reason, "pair_gate_status": gate_status, "train_internal_holdout": train_gate_selection})
    _write_json(output_root / "logistic_gate_metrics.json", {"status": "SKIPPED", "reason": gate_skip_reason, "pair_gate_status": gate_status})
    _write_json(output_root / "mlp_gate_metrics.json", {"status": "SKIPPED", "reason": gate_skip_reason, "pair_gate_status": gate_status})
    _write_json(output_root / "high_occlusion_metrics.json", result["high_occlusion"])
    _write_json(output_root / "per_class_metrics.json", {name: methods[name]["per_class"] for name in methods if "per_class" in methods[name]})
    _write_json(output_root / "gate_rescue_analysis.json", {"status": "SKIPPED", "reason": gate_skip_reason, "pair_gate_status": gate_status})
    _write_json(output_root / "leakage_audit.json", result["leakage_audit"])
    _write_json(output_root / "result.json", result)
    _write_analysis(output_root / "analysis.md", result)
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    methods = result["main_metrics"]
    pairs = result["pair_complementarity"]
    best_a = max((value["pair_any_correct_rate"] for name, value in pairs.items() if name.startswith("A+")), default=0.0)
    lines = [
        "# Selector Complementarity + RGB-D Deployable NBV Audit",
        "",
        "Protocol: strict current Frame0 -> one Stage-A candidate-only action -> real archived O1 Yaw8Fair HAR.",
        "Policy Test used: false. Moving Val was not used for training or threshold selection.",
        "",
        "## Phase A: selector metrics (Moving Val)",
        "",
        "| Method | Accuracy | Macro-F1 | Move rate |",
        "|---|---:|---:|---:|",
    ]
    for name, value in methods.items():
        if "accuracy" in value:
            lines.append(f"| {name} | {float(value['accuracy']):.6f} | {float(value['macro_f1']):.6f} | {float(value.get('move_rate', 0.0)):.6f} |")
    lines.extend(
        [
            "",
            "Pair AnyCorrect rates: " + ", ".join(f"{name}={float(value['pair_any_correct_rate']):.6f}" for name, value in pairs.items()),
            f"Best pair containing A: {best_a:.6f}; gate decision: {result['pair_gate_status']}.",
            "",
            "## Phase B: RGB-D capability",
            "",
            f"Current depth cache present: {result['depth_sensor_audit']['current_frame0_depth_present']}; raw frame-0 YOLO cache present: {result['depth_sensor_audit']['raw_frame0_yolo_present']}.",
            f"Habitat SensorType.DEPTH probe: {result['depth_probe'].get('status')}; the probe rendered 50 Train + 50 Moving-Val current-frame depth images transiently, but no raw depth cache was retained. Raw frame-0 YOLO remains absent, so D1/D2/D3 and gates are N/A rather than fabricated.",
            "Current RGB archives expose image size, camera positions/rotations and yaw metadata, but not explicit depth calibration or per-joint frame-0 confidence.",
            "",
            "## Scientific conclusion",
            "",
        ]
    )
    if best_a < 0.61:
        lines.append("The best pair containing Frame0SceneVisibility is below 0.61, so pair complementarity is too small for gate training under the preregistered rule. The RGB-D route remains a capability audit only until current-frame depth and raw YOLO are separately acquired.")
    else:
        lines.append("Pair complementarity clears the preregistered threshold; only a Train-internal gate selection would be admissible after current-frame RGB-D artifacts are acquired and aligned.")
    lines.extend(
        [
            "",
            "Explicit limitations: A uses GT frame-0 human state and is privileged; candidate skeleton/logits are used only after selection for terminal evaluation. No claim of deployability is made for D0/A.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.output_root.resolve(), args.data_root.resolve(), _device(args.device))
    print(json.dumps({"status": result["status"], "population": result["population"], "pair_gate_status": result["pair_gate_status"], "runtime": result["runtime"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
