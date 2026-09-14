#!/usr/bin/env python3
"""Val-only known-map geometric NBV upgrade suite.

The script evaluates deterministic map-derived candidate scores under the
Yaw8Fair recognizer.  It deliberately reuses the existing Stage-A candidate
pool and map/raycast cache; no perception data or model is generated here.
All candidate observations are used only after a score has selected an action
for terminal evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
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

from activeview.scripts.eval.reduced12_nbv_utils import (  # noqa: E402
    LABELS,
    NUM_CLASSES,
    classification,
    correlation,
    gt_margin,
)
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (  # noqa: E402
    _load_rows,
    _train_prior,
)
from activeview.scripts.experiments.run_reduced12_yaw8_shared_head_fairness_audit import (  # noqa: E402
    SharedHead,
)
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import (  # noqa: E402
    _candidate_mask,
    _load_yaw8_options,
    _sha256,
)
from activeview.core.paths import get_data_root  # noqa: E402

SEED = 42
MAX_OPTIONS = 22
MAP_DIM = 96
FEATURE_DIM = 256
MAP_REL = Path("diagnostics/map_aware_robust_observability_audit")
HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
)
STGCN_REL = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "known_map_geometric_nbv_upgrade"
)

# Existing map feature layout (see map_aware_robust_observability_audit).
JOINT_VIS = slice(0, 17)
DENSE_VIS = 17
PROJ_AREA = 31
PROJ_HEIGHT = 32
PROJ_FOV = 34

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

def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_map_cache(
    data_root: Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = _read_json(data_root / MAP_REL / f"{split}.json")
    cache = _read_npz(data_root / MAP_REL / f"{split}.npz")
    if int(metadata.get("rows", -1)) != len(rows):
        raise ValueError(f"map cache row mismatch for {split}")
    if metadata.get("signature") != _row_signature(rows):
        raise ValueError(f"map cache signature mismatch for {split}")
    if cache["features"].shape[:2] != (len(rows), MAX_OPTIONS):
        raise ValueError(f"unexpected map feature shape for {split}: {cache['features'].shape}")
    if cache["features"].shape[-1] != MAP_DIM:
        raise ValueError("unexpected map feature dimension")
    if cache["ids"].shape != (len(rows), MAX_OPTIONS) or cache["mask"].shape != (len(rows), MAX_OPTIONS):
        raise ValueError(f"unexpected map action shape for {split}")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"map cache marked Test-derived: {split}")
    return cache, metadata


def _row_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    payload = "\n".join(str(row["episode_id"]) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_scores(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-context min-max normalization over candidate-only slots."""
    scores = np.full(values.shape, -np.inf, dtype=np.float64)
    for index in range(values.shape[0]):
        active = np.flatnonzero(mask[index])
        if active.size == 0:
            continue
        current = np.asarray(values[index, active], dtype=np.float64)
        finite = np.isfinite(current)
        if not finite.any():
            continue
        low, high = float(np.min(current[finite])), float(np.max(current[finite]))
        normalized = np.zeros(active.size, dtype=np.float64)
        if high - low > 1e-12:
            normalized[finite] = (current[finite] - low) / (high - low)
        scores[index, active] = normalized
    return scores


def _candidate_rank_sum(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the sum of within-context candidate rank percentiles."""
    result = np.full(left.shape, -np.inf, dtype=np.float64)
    for index in range(left.shape[0]):
        active = np.flatnonzero(mask[index])
        if active.size == 0:
            continue
        left_values = np.asarray(left[index, active], dtype=np.float64)
        right_values = np.asarray(right[index, active], dtype=np.float64)
        left_order = np.argsort(np.nan_to_num(left_values, nan=-np.inf), kind="mergesort")
        right_order = np.argsort(np.nan_to_num(right_values, nan=-np.inf), kind="mergesort")
        left_rank = np.empty(active.size, dtype=np.float64)
        right_rank = np.empty(active.size, dtype=np.float64)
        left_rank[left_order] = np.arange(active.size, dtype=np.float64)
        right_rank[right_order] = np.arange(active.size, dtype=np.float64)
        denom = max(active.size - 1, 1)
        result[index, active] = left_rank / denom + right_rank / denom
    return result


def _select(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, mask: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        if active.size == 0:
            actions.append(int(row["current_viewpoint_id"]))
            continue
        ordered = sorted(
            active.tolist(),
            key=lambda slot: (-float(scores[index, slot]), int(row["candidate_ids"][slot - 1]), int(slot)),
        )
        actions.append(int(row["candidate_ids"][ordered[0] - 1]))
    return actions


def _random_actions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    rng = np.random.default_rng(SEED)
    return [
        int(rng.choice([int(value) for value in row["candidate_ids"]]))
        if row["candidate_ids"]
        else int(row["current_viewpoint_id"])
        for row in rows
    ]


def _terminal(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    actions: Sequence[int],
) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((options["ids"][index] == int(action)) & options["mask"][index])
        if slots.size != 1:
            raise ValueError(f"cannot resolve selected viewpoint {action} at row {index}")
        predictions.append(int(np.argmax(options["logp"][index, int(slots[0])])))
    return np.asarray(predictions, dtype=np.int64)


def _metric(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    actions: Sequence[int],
) -> dict[str, Any]:
    result = classification(labels, predictions)
    current = np.asarray([int(row["current_viewpoint_id"]) for row in rows], dtype=np.int64)
    result.update(
        {
            "method": name,
            "move_rate": float(np.mean(np.asarray(actions, dtype=np.int64) != current)),
            "stay_rate": float(np.mean(np.asarray(actions, dtype=np.int64) == current)),
            "test_used": False,
        }
    )
    return result


def _oracle_actions(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    criterion: str,
) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(options["mask"][index])[1:]
        if active.size == 0:
            actions.append(int(row["current_viewpoint_id"]))
            continue
        label = int(row["label_id"])
        if criterion == "true_logp":
            scores = options["logp"][index, active, label]
        elif criterion == "margin":
            scores = np.asarray(
                [gt_margin(options["logp"][index, slot], label) for slot in active],
                dtype=np.float64,
            )
        else:
            raise ValueError(criterion)
        actions.append(int(options["ids"][index, int(active[int(np.argmax(scores))])]))
    return actions


def _any_correct(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        active = np.flatnonzero(options["mask"][index])[1:]
        values[index] = bool(
            active.size
            and np.any(np.argmax(options["logp"][index, active], axis=1) == int(row["label_id"]))
        )
    return values


def _valid_viewpoint_match(map_cache: Mapping[str, np.ndarray], options: Mapping[str, np.ndarray]) -> None:
    if not np.array_equal(map_cache["ids"], options["ids"]):
        raise ValueError("map and Yaw8Fair viewpoint ids differ")
    if not np.array_equal(map_cache["mask"], options["mask"]):
        raise ValueError("map and Yaw8Fair legality masks differ")


def _prior_metrics(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    prior: Mapping[int, float],
) -> np.ndarray:
    values = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    for index in range(len(rows)):
        for slot in np.flatnonzero(options["mask"][index])[1:]:
            values[index, slot] = float(prior.get(int(options["ids"][index, slot]), prior.get(-1, 0.0)))
    return values


def _true_values(rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    true_logp = np.full(options["ids"].shape, np.nan, dtype=np.float64)
    margins = np.full(options["ids"].shape, np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        label = int(row["label_id"])
        for slot in np.flatnonzero(options["mask"][index])[1:]:
            logp = options["logp"][index, slot]
            true_logp[index, slot] = float(logp[label])
            margins[index, slot] = gt_margin(logp, label)
    return true_logp, margins


def _median_correct_scale(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    map_cache: Mapping[str, np.ndarray],
) -> tuple[float, dict[str, Any]]:
    values: list[float] = []
    count = 0
    for index, row in enumerate(rows):
        label = int(row["label_id"])
        for slot in np.flatnonzero(options["mask"][index])[1:]:
            if int(np.argmax(options["logp"][index, slot])) == label:
                area = float(map_cache["features"][index, slot, PROJ_AREA])
                if np.isfinite(area) and area > 0:
                    values.append(area)
                    count += 1
    if not values:
        raise ValueError("no positive Train projected scales")
    return float(np.median(values)), {"correct_candidates": count, "distribution_count": len(values), "source": "Policy Train only"}


def _scale_score(area: np.ndarray, target: float) -> np.ndarray:
    return -np.abs(np.log(np.clip(area, 1e-8, None)) - math.log(max(target, 1e-8)))


def _grid_weights() -> list[tuple[float, float, float]]:
    return [
        (a / 4.0, b / 4.0, (4 - a - b) / 4.0)
        for a in range(5)
        for b in range(5 - a)
    ]


def _calibrate_doq(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    records = sorted({str(row["record_id"]) for row in rows})
    holdout = set(records[-max(1, int(round(0.1 * len(records)))) :])
    indices = np.asarray([i for i, row in enumerate(rows) if str(row["record_id"]) in holdout], dtype=np.int64)
    best = (-1.0, (0.0, 0.0, 1.0))
    component = np.stack(
        [
            np.mean(features[:, :, 0:17], axis=2),
            features[:, :, DENSE_VIS],
            features[:, :, PROJ_FOV],
        ],
        axis=-1,
    )
    mask = _candidate_mask(options["mask"])
    normalized = np.stack([_candidate_scores(component[:, :, i], mask) for i in range(3)], axis=-1)
    for weights in _grid_weights():
        score = np.sum(normalized * np.asarray(weights)[None, None, :], axis=-1)
        actions = _select(rows, score, mask)
        pred = _terminal(rows, options, actions)
        accuracy = float(np.mean(pred[indices] == labels[indices])) if indices.size else 0.0
        if accuracy > best[0] + 1e-12:
            best = (accuracy, weights)
    return best[1], {
        "weights": {"dense_visibility": best[1][0], "completeness": best[1][1], "scale": best[1][2]},
        "internal_holdout_records": len(holdout),
        "internal_holdout_accuracy": best[0],
        "grid_step": 0.25,
        "selection_source": "Policy Train record holdout only",
    }


def _correlations(
    rows: Sequence[Mapping[str, Any]],
    values: Mapping[str, np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, array in values.items():
        active = mask & np.isfinite(array) & np.isfinite(target)
        flat_x, flat_y = array[active], target[active]
        within: list[float] = []
        for index in range(len(rows)):
            slots = np.flatnonzero(active[index])
            if slots.size >= 2:
                within.append(correlation(array[index, slots], target[index, slots], spearman=True))
        result[name] = {
            "candidate_count": int(flat_x.size),
            "candidate_level_spearman": correlation(flat_x, flat_y, spearman=True),
            "within_context_spearman_mean": float(np.mean(within)) if within else 0.0,
            "within_context_spearman_median": float(np.median(within)) if within else 0.0,
            "within_context_count": len(within),
        }
    return result


def _high_occlusion(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    methods: Mapping[str, dict[str, Any]],
    predictions: Mapping[str, np.ndarray],
    actions: Mapping[str, Sequence[int]],
    current_visibility: np.ndarray,
) -> dict[str, Any]:
    threshold = float(np.quantile(current_visibility, 1.0 / 3.0))
    selected = current_visibility < threshold
    output: dict[str, Any] = {
        "definition": "bottom tertile of frame-0 current-slot 17-joint map visibility",
        "count": int(selected.sum()),
        "threshold": threshold,
        "methods": {},
    }
    for name in methods:
        if name not in predictions:
            continue
        subset_rows = [rows[i] for i in np.flatnonzero(selected)]
        output["methods"][name] = _metric(
            name,
            subset_rows,
            labels[selected],
            predictions[name][selected],
            [actions[name][int(i)] for i in np.flatnonzero(selected)],
        )
    return output


def _switch(
    labels: np.ndarray,
    old_actions: Sequence[int],
    new_actions: Sequence[int],
    old_pred: np.ndarray,
    new_pred: np.ndarray,
) -> dict[str, Any]:
    old_a, new_a = np.asarray(old_actions), np.asarray(new_actions)
    changed = old_a != new_a
    return {
        "switch_count": int(changed.sum()),
        "switch_rate": float(changed.mean()),
        "old_accuracy_on_switch": float(np.mean(old_pred[changed] == labels[changed])) if changed.any() else 0.0,
        "new_accuracy_on_switch": float(np.mean(new_pred[changed] == labels[changed])) if changed.any() else 0.0,
        "old_wrong_new_correct": int(np.sum((old_pred != labels) & (new_pred == labels) & changed)),
        "old_correct_new_wrong": int(np.sum((old_pred == labels) & (new_pred != labels) & changed)),
        "both_correct": int(np.sum((old_pred == labels) & (new_pred == labels))),
        "both_wrong": int(np.sum((old_pred != labels) & (new_pred != labels))),
    }


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["main_metrics"]
    baseline = methods["Frame0SceneVisibility"]
    candidates = result["geometric_method_names"]
    best_name = result["best_geometric_method"]
    best = methods[best_name]
    gain_frame = 100.0 * (float(best["accuracy"]) - float(baseline["accuracy"]))
    gain_static = 100.0 * (float(best["accuracy"]) - float(methods["StaticViewPrior"]["accuracy"]))
    if gain_frame >= 1.5 and float(best["accuracy"]) >= 0.60:
        decision = "STRONG KEEP ENHANCED GEOMETRIC NBV"
    elif gain_frame >= 1.0 and float(best["accuracy"]) >= 0.595:
        decision = "KEEP ENHANCED GEOMETRIC NBV"
    else:
        decision = "STOP GEOMETRIC SCORE EXPANSION"
    lines = [
        "# Known-Map Geometric NBV Upgrade Suite",
        "",
        "Protocol: Frame0 legal information → exactly one Stage-A legal candidate → selected real O1 alone → frozen Yaw8Fair HAR.",
        "Moving Val only (10,080 contexts); Policy Test was never read. Map quantities are privileged GT-frame-0 geometry diagnostics.",
        "",
        "## Main privileged results",
        "",
        "| Method | Accuracy | Macro-F1 | Gain vs StaticPrior | Gain vs Frame0Visibility |",
        "|---|---:|---:|---:|---:|",
    ]
    static_acc = float(methods["StaticViewPrior"]["accuracy"])
    frame_acc = float(baseline["accuracy"])
    for name, metric in methods.items():
        if not isinstance(metric, dict) or "accuracy" not in metric:
            continue
        lines.append(
            f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | "
            f"{100*(metric['accuracy']-static_acc):+.3f}pp | "
            f"{100*(metric['accuracy']-frame_acc):+.3f}pp |"
        )
    lines.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. Visibility+prior is evaluated at fixed λ=.25/.5/1 and rank-sum; the best gain over Frame0Visibility is {100*(float(result['best_visibility_prior_accuracy'])-frame_acc):+.3f}pp.",
            f"2. Dense visibility versus 17-joint visibility: {100*(float(methods['DenseVisibility']['accuracy'])-frame_acc):+.3f}pp; completeness-only: {100*(float(methods['Completeness']['accuracy'])-frame_acc):+.3f}pp; scale-only: {100*(float(methods['ScaleOnly']['accuracy'])-frame_acc):+.3f}pp.",
            f"3. Best explicit geometric method is **{best_name}**, Accuracy/F1={best['accuracy']:.6f}/{best['macro_f1']:.6f}; gain vs StaticPrior={gain_static:+.3f}pp and vs Frame0Visibility={gain_frame:+.3f}pp.",
            f"4. Pre-registered decision: **{decision}**. The 60% milestone is {'crossed' if float(best['accuracy']) >= 0.60 else 'not crossed'}.",
            "5. The dense completeness value is explicitly an H36M17 in-FOV proxy because the existing cache persisted dense ray visibility but not dense projected-point counts. Uncertainty weighting is SKIPPED because archives contain only a 30-frame viewpoint scalar confidence, not current frame-0 per-joint confidence; no joint confidence was fabricated.",
            "6. E1/E2 estimated-state scores are N/A: the current pipeline has no legal metric global human localization/world-space estimated pose artifact. This is a state-estimation capability gap, not an imputed score.",
            "7. High-occlusion and per-class tables are saved in high_occlusion_metrics.json and per_class_metrics.json. Correlations are saved in correlation_diagnostics.json.",
            "",
            "## Boundary and leakage",
            "",
            "No candidate RGB, candidate skeleton, candidate ST-GCN output, future frame, GT action, or Policy Test artifact entered a selector score. Selected archived skeletons are read only for terminal evaluation.",
            "",
            "```text",
            "policy_test_used=false",
            "new_model_training=false (Train-only prior/scale statistics; optional DOQ calibration is skipped unless its pre-registered gate is met)",
            "known_static_map_used=true",
            "gt_frame0_human_state_used=true (privileged track)",
            "deployable=false",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
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
    _valid_viewpoint_match(train_map, train_options)
    _valid_viewpoint_match(val_map, val_options)
    candidate_train = _candidate_mask(train_options["mask"])
    candidate_val = _candidate_mask(val_options["mask"])
    prior, prior_meta = _train_prior(train_rows, train_options)
    train_true, train_margin = _true_values(train_rows, train_options)
    val_true, val_margin = _true_values(val_rows, val_options)
    median_scale, scale_meta = _median_correct_scale(train_rows, train_options, train_map)

    values_val: dict[str, np.ndarray] = {
        "V17": np.mean(val_map["features"][:, :, JOINT_VIS], axis=2),
        "V_dense": val_map["features"][:, :, DENSE_VIS],
        "Completeness": val_map["features"][:, :, PROJ_FOV],
        "ProjectedArea": val_map["features"][:, :, PROJ_AREA],
        "ProjectedHeight": val_map["features"][:, :, PROJ_HEIGHT],
    }
    values_train: dict[str, np.ndarray] = {
        "V17": np.mean(train_map["features"][:, :, JOINT_VIS], axis=2),
        "V_dense": train_map["features"][:, :, DENSE_VIS],
        "Completeness": train_map["features"][:, :, PROJ_FOV],
        "ProjectedArea": train_map["features"][:, :, PROJ_AREA],
    }
    values_val["ScaleOnly"] = _scale_score(values_val["ProjectedArea"], median_scale)
    values_train["ScaleOnly"] = _scale_score(values_train["ProjectedArea"], median_scale)
    normalized_val = {name: _candidate_scores(value, candidate_val) for name, value in values_val.items()}
    normalized_train = {name: _candidate_scores(value, candidate_train) for name, value in values_train.items()}

    labels = labels_val
    actions: dict[str, Sequence[int]] = {}
    predictions: dict[str, np.ndarray] = {}
    methods: dict[str, dict[str, Any]] = {}

    def add(name: str, action_values: Sequence[int]) -> None:
        actions[name] = list(action_values)
        predictions[name] = _terminal(val_rows, val_options, action_values)
        methods[name] = _metric(name, val_rows, labels, predictions[name], action_values)

    add("Random legal", _random_actions(val_rows))
    prior_raw = _prior_metrics(val_rows, val_options, prior)
    add("StaticViewPrior", _select(val_rows, prior_raw, candidate_val))
    add("Frame0SceneVisibility", _select(val_rows, values_val["V17"], candidate_val))

    prior_norm = _candidate_scores(prior_raw, candidate_val)
    prior_specs: dict[str, np.ndarray] = {}
    for lam in (0.25, 0.5, 1.0):
        prior_specs[f"Visibility+Prior λ={lam}"] = normalized_val["V17"] + lam * prior_norm
    prior_specs["VisibilityPrior-RankSum"] = _candidate_rank_sum(
        values_val["V17"], prior_raw, candidate_val
    )
    for name, score in prior_specs.items():
        add(name, _select(val_rows, score, candidate_val))

    geometric_scores: dict[str, np.ndarray] = {
        "DenseVisibility": values_val["V_dense"],
        "Completeness": values_val["Completeness"],
        "ScaleOnly": values_val["ScaleOnly"],
    }
    doq_components = [normalized_val["V_dense"], normalized_val["Completeness"], normalized_val["ScaleOnly"]]
    doq_equal = np.sum(np.stack(doq_components, axis=-1), axis=-1)
    geometric_scores["DOQ-Equal"] = doq_equal
    geometric_scores["G_geo"] = doq_equal
    for lam in (0.25, 0.5):
        geometric_scores[f"G_full λ={lam}"] = _candidate_scores(doq_equal, candidate_val) + lam * prior_norm
    for name, score in geometric_scores.items():
        add(name, _select(val_rows, score, candidate_val))

    # Fixed train-only calibration is conditional by protocol; it is not run
    # if the uncalibrated DOQ is below the pre-registered Frame0 reference.
    doq_calibration: dict[str, Any]
    if methods["DOQ-Equal"]["accuracy"] + 1e-12 >= methods["Frame0SceneVisibility"]["accuracy"]:
        weights, doq_calibration = _calibrate_doq(train_rows, train_options, train_map["features"], labels_train)
        calibrated = (
            weights[0] * normalized_val["V_dense"]
            + weights[1] * normalized_val["Completeness"]
            + weights[2] * normalized_val["ScaleOnly"]
        )
        add("DOQ-TrainCalibrated", _select(val_rows, calibrated, candidate_val))
    else:
        doq_calibration = {"status": "SKIPPED", "reason": "DOQ-Equal did not reach Frame0SceneVisibility on Val; no Val-driven calibration allowed."}

    uncertainty_metrics = {
        "status": "SKIPPED",
        "reason": "archives expose confidence shape (32,), a 30-frame viewpoint scalar; current frame-0 per-joint c_j is unavailable",
        "methods": {"Uncertainty25": None, "Uncertainty50": None, "Dense uncertainty": None},
        "per_joint_confidence_fabricated": False,
    }
    true_actions = _oracle_actions(val_rows, val_options, "true_logp")
    margin_actions = _oracle_actions(val_rows, val_options, "margin")
    add("GT-TrueLogP Oracle", true_actions)
    add("GT-Margin Oracle", margin_actions)
    any_correct = _any_correct(val_rows, val_options)
    methods["AnyCorrect Coverage"] = {
        "method": "AnyCorrect Coverage",
        "contexts": len(val_rows),
        "coverage_count": int(any_correct.sum()),
        "coverage_rate": float(any_correct.mean()),
        "test_used": False,
    }

    expected = {
        "StaticViewPrior": (0.549901, 0.571990),
        "Frame0SceneVisibility": (0.583929, 0.598955),
        "GT-TrueLogP Oracle": (0.760714, 0.775961),
        "GT-Margin Oracle": (0.776190, 0.796334),
    }
    observed = {name: (methods[name]["accuracy"], methods[name]["macro_f1"]) for name in expected}
    deltas = {name: {"accuracy_pp": 100 * (observed[name][0] - expected[name][0]), "macro_f1_pp": 100 * (observed[name][1] - expected[name][1])} for name in expected}
    if any(abs(value) > 0.002 for delta in deltas.values() for value in (delta["accuracy_pp"] / 100, delta["macro_f1_pp"] / 100)):
        raise RuntimeError(f"baseline reproduction gate failed: {deltas}")

    geometric_names = [name for name in geometric_scores if name in methods]
    if "DOQ-TrainCalibrated" in methods:
        geometric_names.append("DOQ-TrainCalibrated")
    best_name = max(geometric_names, key=lambda name: float(methods[name]["accuracy"]))
    best_prior = max(prior_specs, key=lambda name: float(methods[name]["accuracy"]))
    current_visibility = values_val["V17"][:, 0]
    high = _high_occlusion(
        val_rows,
        labels,
        {name: methods[name] for name in ("StaticViewPrior", "Frame0SceneVisibility", best_name, "GT-TrueLogP Oracle")},
        predictions,
        actions,
        current_visibility,
    )
    per_class = {
        name: methods[name]["per_class"]
        for name in ("StaticViewPrior", "Frame0SceneVisibility", best_name, "GT-TrueLogP Oracle")
    }
    switch = {
        "best_geometric_vs_frame0_visibility": _switch(
            labels,
            actions["Frame0SceneVisibility"],
            actions[best_name],
            predictions["Frame0SceneVisibility"],
            predictions[best_name],
        ),
        "best_geometric_vs_static_prior": _switch(
            labels,
            actions["StaticViewPrior"],
            actions[best_name],
            predictions["StaticViewPrior"],
            predictions[best_name],
        ),
    }
    ablation_scores = {
        "remove_prior": doq_equal,
        "remove_dense_visibility": normalized_val["Completeness"] + normalized_val["ScaleOnly"],
        "remove_completeness": normalized_val["V_dense"] + normalized_val["ScaleOnly"],
        "remove_scale": normalized_val["V_dense"] + normalized_val["Completeness"],
    }
    ablation_metrics: dict[str, Any] = {}
    ablation_reference = methods["G_full λ=0.5"]
    for name, score in ablation_scores.items():
        ablation_action = _select(val_rows, score, candidate_val)
        ablation_prediction = _terminal(val_rows, val_options, ablation_action)
        metric = _metric(name, val_rows, labels, ablation_prediction, ablation_action)
        metric["accuracy_drop_vs_G_full_0.5_pp"] = 100.0 * (ablation_reference["accuracy"] - metric["accuracy"])
        metric["macro_f1_drop_vs_G_full_0.5_pp"] = 100.0 * (ablation_reference["macro_f1"] - metric["macro_f1"])
        ablation_metrics[name] = metric
    ablation_metrics["remove_uncertainty"] = {"status": "SKIPPED", "reason": uncertainty_metrics["reason"]}
    target = val_true
    correlation_values = {
        "V17": values_val["V17"],
        "V_dense": values_val["V_dense"],
        "Completeness": values_val["Completeness"],
        "Scale": values_val["ScaleOnly"],
        "Prior": prior_raw,
        "G_geo": doq_equal,
        "G_full_0.25": geometric_scores["G_full λ=0.25"],
        "G_full_0.5": geometric_scores["G_full λ=0.5"],
    }
    correlations = _correlations(val_rows, correlation_values, target, candidate_val)

    output_root.mkdir(parents=True, exist_ok=True)
    action_count = candidate_val.sum(axis=1)
    protocol = {
        "formal_task": "Frame0 -> exactly one Stage-A legal candidate -> selected real O1 alone",
        "candidate_action_set": "candidate-only Stage-A legal candidate_pool; current/stay excluded",
        "moving_val_contexts": len(val_rows),
        "policy_train_contexts": len(train_rows),
        "mean_legal_candidates": float(np.mean(action_count)),
        "min_legal_candidates": int(np.min(action_count)),
        "max_legal_candidates": int(np.max(action_count)),
        "map_cache_train": train_map_meta,
        "map_cache_val": val_map_meta,
        "yaw8_cache_train": train_meta,
        "yaw8_cache_val": val_meta,
        "yaw8_stgcn_checkpoint_sha256": _sha256(data_root / STGCN_REL),
        "yaw8_shared_head_checkpoint_sha256": _sha256(head_path),
        "train_prior": prior_meta,
        "train_median_correct_scale": scale_meta,
        "test_used": False,
        "moving_val_used_for_training_or_weight_selection": False,
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_KNOWN_MAP_GEOMETRIC_NBV_UPGRADE",
        "status": "COMPLETED",
        "population": {"policy_train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "moving_val_candidate_samples": int(candidate_val.sum())},
        "main_metrics": methods,
        "geometric_method_names": geometric_names,
        "best_geometric_method": best_name,
        "best_visibility_prior_method": best_prior,
        "best_visibility_prior_accuracy": methods[best_prior]["accuracy"],
        "baseline_reproduction": {"expected": expected, "observed": observed, "deltas": deltas, "status": "PASS"},
        "any_correct_coverage": methods["AnyCorrect Coverage"],
        "train_scale": {"median_correct_scale": median_scale, **scale_meta},
        "doq_calibration": doq_calibration,
        "uncertainty": uncertainty_metrics,
        "high_occlusion": high,
        "per_class": per_class,
        "switch_analysis": switch,
        "correlation_diagnostics": correlations,
        "ablation_metrics": ablation_metrics,
        "protocol_audit": protocol,
        "leakage_audit": {
            "policy_test_read": False,
            "candidate_rgb_read_before_selection": False,
            "candidate_skeleton_read_before_selection": False,
            "candidate_logits_used_before_selection": False,
            "candidate_logits_loaded_for_terminal_and_oracle_evaluation": True,
            "future_human_pose_used": False,
            "future_current_frames_used": False,
            "gt_action_used_at_inference": False,
            "known_static_map_used": True,
            "gt_frame0_human_state_used": True,
            "estimated_state_track": "N/A; no legal metric global human localization artifact",
            "moving_val_used_for_training": False,
            "moving_val_used_for_weight_selection": False,
            "future_candidate_skeleton_used_for_terminal_evaluation_only": True,
            "deployable": False,
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.monotonic() - started},
    }
    _write_json(output_root / "config.json", {"seed": SEED, "recognizer": "frozen Yaw8 ST-GCN + frozen matched Policy-balanced head", "candidate_action_set": "Stage-A candidate-only", "test_used": False, "map_cache": str((data_root / MAP_REL).resolve())})
    _write_json(output_root / "protocol_audit.json", protocol)
    _write_json(output_root / "baseline_reproduction.json", result["baseline_reproduction"])
    _write_json(output_root / "visibility_prior_fusion.json", {name: methods[name] for name in prior_specs})
    _write_json(output_root / "dense_proxy_metadata.json", {"source": str((data_root / MAP_REL).resolve()), "dense_points_per_context": train_map_meta.get("dense_points_per_context"), "definition": "existing deterministic dense body proxy from map cache; no new raycast generated", "test_used": False})
    _write_json(output_root / "dense_visibility_metrics.json", {name: methods[name] for name in ("DenseVisibility",)})
    _write_json(output_root / "completeness_metrics.json", {"Completeness": methods["Completeness"], "definition": "available H36M17 in-FOV fraction proxy; dense in-frame point count was not persisted in existing cache"})
    candidate_area = values_val["ProjectedArea"][candidate_val]
    candidate_height = val_map["features"][:, :, PROJ_HEIGHT][candidate_val]
    _write_json(output_root / "projected_scale_metrics.json", {
        "ScaleOnly": methods["ScaleOnly"],
        "median_correct_scale_train": median_scale,
        "candidate_projected_area_mean": float(np.nanmean(candidate_area)),
        "candidate_projected_area_median": float(np.nanmedian(candidate_area)),
        "candidate_projected_height_mean": float(np.nanmean(candidate_height)),
        "candidate_projected_height_median": float(np.nanmedian(candidate_height)),
        "definition": "projected H36M17 bbox area/height proxy persisted by the existing map cache",
    })
    _write_json(output_root / "doq_metrics.json", {name: methods[name] for name in ("DOQ-Equal", "DOQ-TrainCalibrated") if name in methods} | {"calibration": doq_calibration})
    _write_json(output_root / "uncertainty_visibility_metrics.json", uncertainty_metrics)
    _write_json(output_root / "geometric_fusion_metrics.json", {name: methods[name] for name in geometric_names})
    _write_json(output_root / "ablation_metrics.json", ablation_metrics)
    _write_json(output_root / "privileged_vs_deployable.json", {"E0_GT_position_GT_pose": {"Frame0SceneVisibility": methods["Frame0SceneVisibility"], "BestDenseGeo": methods[best_name]}, "E1_GT_position_estimated_pose": None, "E2_estimated_position_estimated_pose": None, "reason": "no legal metric global estimated human localization/world-pose artifact"})
    _write_json(output_root / "pose_state_audit.json", {"E0": "available via existing map cache GT frame-0 state", "E1": "N/A", "E2": "N/A", "reason": "current production pipeline stores root-relative normalized skeleton and viewpoint scalar confidence, not metric world-space estimated human state", "pose_noise_sensitivity": "not fabricated without an estimated-state artifact"})
    _write_json(output_root / "high_occlusion_metrics.json", high)
    _write_json(output_root / "per_class_metrics.json", per_class)
    _write_json(output_root / "switch_analysis.json", switch)
    _write_json(output_root / "correlation_diagnostics.json", correlations)
    _write_json(output_root / "leakage_audit.json", result["leakage_audit"])
    _write_json(output_root / "result.json", result)
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = get_data_root()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output_root.resolve()), "best_geometric_method": result["best_geometric_method"], "best_accuracy": result["main_metrics"][result["best_geometric_method"]]["accuracy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
