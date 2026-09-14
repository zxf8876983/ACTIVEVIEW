#!/usr/bin/env python3
"""Read-only audit of policy/recognizer distribution coupling.

The audit reuses the reduced12 strict Frame0 artifacts.  It never trains a
model: Train is used only to form old-adaptive viewpoint priors, and Moving Val
is used for all selection and terminal measurements.  Policy Test and future
candidate observations are out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_adaptive_head_frame0_nbv_audit import (
    DIAGNOSTIC_REL,
    FEATURE_DIM,
    MAX_OPTIONS,
    OLD_HEAD_REL,
    POLICY_REL,
    SELECTOR_REL,
    SHARED_HEAD_REL,
    _device,
    _head_logits,
    _load_head,
    _load_raw_options,
    _log_softmax,
    _option_geometry,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead, load_rows
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import (
    TaskUtilityPredictor,
    _load_visibility_targets,
    _predict_task,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    _load_frame0_dino,
    _select,
)

NUM_CLASSES = len(LABELS)
SEED = 42
DEFAULT_OUTPUT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/policy_recognizer_coupling_audit"
)
DEFAULT_RUNTIME = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/adaptive_head_frame0_nbv_audit"
)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_margin(logp: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    target = np.take_along_axis(
        logp,
        np.broadcast_to(labels[:, None, None], (logp.shape[0], logp.shape[1], 1)),
        axis=2,
    ).squeeze(axis=2)
    other = logp.copy()
    other[np.arange(logp.shape[0]), :, labels] = -np.inf
    return (target - np.max(other, axis=-1)).astype(np.float32)


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, rows: Sequence[Mapping[str, Any]], actions: Sequence[int], labels: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    predictions: list[int] = []
    moves = 0
    for index, action in enumerate(actions):
        slots = np.flatnonzero(mask[index] & (ids[index] == int(action)))
        if slots.size != 1:
            raise ValueError(f"action outside legal set: {rows[index]['episode_id']}/{action}")
        moves += int(int(action) != int(rows[index]["current_viewpoint_id"]))
        predictions.append(int(np.argmax(logp[index, int(slots[0])])) )
    metric = classification(labels, np.asarray(predictions, dtype=np.int64))
    metric["move_rate"] = float(moves / len(rows)) if rows else 0.0
    metric["stay_rate"] = 1.0 - metric["move_rate"]
    return metric, np.asarray(predictions, dtype=np.int64)


def _histogram(values: Sequence[int], current: Sequence[int] | None = None) -> dict[str, Any]:
    counts = Counter(int(value) for value in values)
    total = max(len(values), 1)
    probabilities = np.asarray([counts.get(index, 0) / total for index in range(32)], dtype=np.float64)
    azimuth_counts = Counter(int(value) % 8 for value in values)
    radius_counts = Counter(int(value) // 8 for value in values)
    azimuth_probabilities = np.asarray([azimuth_counts.get(index, 0) / total for index in range(8)], dtype=np.float64)
    radius_probabilities = np.asarray([radius_counts.get(index, 0) / total for index in range(4)], dtype=np.float64)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log2(positive)).sum())
    order = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    result: dict[str, Any] = {
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "probabilities": probabilities.tolist(),
        "azimuth_counts": {str(key): int(value) for key, value in sorted(azimuth_counts.items())},
        "azimuth_probabilities": azimuth_probabilities.tolist(),
        "radius_counts": {str(key): int(value) for key, value in sorted(radius_counts.items())},
        "radius_probabilities": radius_probabilities.tolist(),
        "top1_viewpoint": int(order[0][0]) if order else None,
        "top5_viewpoints": [int(key) for key, _ in order[:5]],
        "entropy_bits": entropy,
    }
    if current is not None:
        selected = np.asarray(values, dtype=np.int64)
        current_array = np.asarray(current, dtype=np.int64)
        if selected.shape != current_array.shape:
            raise ValueError("current viewpoint array must match selected viewpoint array")
        result["move_rate"] = float(np.mean(selected != current_array)) if selected.size else 0.0
        result["stay_rate"] = 1.0 - result["move_rate"]
    return result


def _summary(values: Mapping[Any, float]) -> dict[str, float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    if array.size == 0:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {"count": float(array.size), "mean": float(array.mean()), "std": float(array.std()), "min": float(array.min()), "max": float(array.max())}


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    def kl(a: np.ndarray, b: np.ndarray) -> float:
        active = a > 0
        return float(np.sum(a[active] * np.log2(a[active] / np.clip(b[active], 1e-12, None))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _select_prior(rows: Sequence[Mapping[str, Any]], unary: Mapping[int, float], pair: Mapping[tuple[int, int], float]) -> list[int]:
    actions: list[int] = []
    for row in rows:
        current = int(row["current_viewpoint_id"])
        options = [current] + [int(value) for value in row["candidate_ids"]]
        scores = [float(unary.get(current, -1e9))]
        scores.extend(float(pair.get((current, option), unary.get(option, -1e9))) for option in options[1:])
        actions.append(options[int(np.argmax(np.asarray(scores, dtype=np.float64)))])
    return actions


def _viewpoint_priors(train_rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], logp: np.ndarray, labels: np.ndarray) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    margins = _task_margin(logp, labels, cache["mask"])
    unary_sum: defaultdict[int, float] = defaultdict(float)
    unary_count: defaultdict[int, int] = defaultdict(int)
    pair_sum: defaultdict[tuple[int, int], float] = defaultdict(float)
    pair_count: defaultdict[tuple[int, int], int] = defaultdict(int)
    for index, row in enumerate(train_rows):
        current = int(row["current_viewpoint_id"])
        for slot in np.flatnonzero(cache["mask"][index]):
            viewpoint = int(cache["ids"][index, slot])
            value = float(margins[index, slot])
            unary_sum[viewpoint] += value
            unary_count[viewpoint] += 1
            if slot != 0:
                key = (current, viewpoint)
                pair_sum[key] += value
                pair_count[key] += 1
    unary = {key: unary_sum[key] / unary_count[key] for key in unary_sum}
    pair = {key: pair_sum[key] / pair_count[key] for key in pair_sum}
    return unary, pair


def _class_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return classification(labels, predictions)["per_class"]


def _selector_agreement(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(left, dtype=np.int64) == np.asarray(right, dtype=np.int64)
    return {"count": int(values.size), "top1_agreement": float(values.mean()) if values.size else 0.0}


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    adaptive = methods["Adaptive-aware selector + old adaptive"]
    shared_adaptive = methods["Adaptive-aware selector + shared"]
    static = methods["StaticViewPrior + old adaptive"]
    pair = methods["ViewPairPrior + old adaptive"]
    shuffle = result["shuffle_metrics"]
    selected = result["recognizer_on_selected_distribution"]["adaptive_aware"]
    adaptive_gain = float((adaptive["accuracy"] - shared_adaptive["accuracy"]) * 100.0)
    best_prior = max(static["accuracy"], pair["accuracy"])
    rgb_drop = float(shuffle["accuracy_drop_pp"])
    prior_gap_pp = float((adaptive["accuracy"] - best_prior) * 100.0)
    if prior_gap_pp >= 1.5 and rgb_drop >= 1.5:
        primary = "MEANINGFUL INSTANCE-CONDITIONED NBV"
        secondary = "adaptive-aware selector also contains recognizer matching"
    elif prior_gap_pp < 1.0:
        primary = "MOSTLY FIXED VIEWPOINT PRIOR"
        secondary = "RGB carries signal, but the adaptive-aware gain over fixed priors is below 1pp"
    elif adaptive_gain >= 2.0:
        primary = "POLICY–RECOGNIZER DISTRIBUTION MATCHING"
        secondary = "adaptive head is beneficial on the selected distribution"
    else:
        primary = "MOSTLY RECOGNIZER–POLICY DISTRIBUTION MATCHING"
        secondary = "with a substantial fixed-prior component; strict 1.5pp instance-conditioned criterion is not met"
    lines = [
        "# Policy–Recognizer Coupling Audit", "",
        "Experiment: Policy–Recognizer Coupling Audit", "Task: strict Frame0 single-step NBV",
        "Final HAR: selected O1 alone", "Selector candidate observation access: false", "Policy Test: false",
        "Purpose: determine whether the 52.6984% result comes primarily from fixed viewpoint prior, recognizer-policy distribution matching, or instance-conditioned Frame0 information.", "",
        f"Moving Val contexts: {result['population']['moving_val_contexts']}; Train contexts used only for static priors: {result['population']['train_contexts']}.", "",
        "## Main results", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ]
    order = ["Random + shared", "Historical selector + shared", "Historical selector + old adaptive", "StaticViewPrior + old adaptive", "ViewPairPrior + old adaptive", "Adaptive-aware selector + shared", "Adaptive-aware selector + old adaptive", "Adaptive-aware selector RGB-shuffled + old adaptive"]
    for name in order:
        if name in methods:
            lines.append(f"| {name} | {methods[name]['accuracy']:.6f} | {methods[name]['macro_f1']:.6f} |")
    lines.extend([
        "",
        "## Distribution and shuffles",
        "",
        f"Adaptive-aware vs historical s1 JSD: {result['selected_view_distribution']['adaptive_vs_s1_jsd']:.6f}; top-5 overlap: {result['selected_view_distribution']['adaptive_vs_s1_top5_overlap']:.3f}; entropy bits adaptive/historical-s1: {result['selected_view_distribution']['adaptive_aware']['entropy_bits']:.4f}/{result['selected_view_distribution']['historical_s1']['entropy_bits']:.4f}.",
        f"RGB shuffle: normal {shuffle['normal']['accuracy']:.6f}, shuffled {shuffle['rgb_shuffled']['accuracy']:.6f}, accuracy drop {rgb_drop:.3f}pp. Geometry shuffle accuracy drop: {shuffle['geometry_accuracy_drop_pp']:.3f}pp.",
        f"Adaptive-aware selected-set shared/old: {selected['shared']['accuracy']:.6f}/{selected['old_adaptive']['accuracy']:.6f}; adaptive head gain on identical selected views: {adaptive_gain:+.3f}pp.",
        "",
        "## Interpretation",
        "",
        f"Primary conclusion: **{primary}**; secondary: {secondary}.",
        f"The old adaptive head's matched s1 gain is {result['recognizer_on_selected_distribution']['historical']['adaptive_gain_pp']:+.3f}pp, while its legal-candidate specialization audit is recorded in the JSON artifacts.",
        f"Static and pair prior best Accuracy is {best_prior:.6f}; adaptive-aware Accuracy is {adaptive['accuracy']:.6f}. This makes the adaptive-aware minus best-prior gap {prior_gap_pp:+.3f}pp.",
        "",
        "Decision on 52.6984%: retain it as an informative strict Frame0 result only when accompanied by the distribution-matching caveat; do not claim a purely instance-conditioned NBV gain.",
        "",
        "```text",
        "policy_test_used=false",
        "training_used=false",
        "new_rgb_generated=false",
        "future_candidate_observation_used_for_selector=false",
        "future_candidate_recognizer_output_used_for_selector=false",
        "gt_action_used_for_selector=false",
        "terminal=selected real O1 alone",
        "```",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    started = time.monotonic()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    train_rows, val_rows = load_rows(data_root)
    train_cache = _load_raw_options(data_root, train_rows, "train")
    val_cache = _load_raw_options(data_root, val_rows, "val")
    labels_train = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    labels_val = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    shared_path, old_path, selector_path = data_root / SHARED_HEAD_REL, data_root / OLD_HEAD_REL, data_root / SELECTOR_REL
    for path in (shared_path, old_path, selector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    shared = SharedHead().to(device)
    shared.load_state_dict(torch.load(shared_path, map_location=device, weights_only=False)["state_dict"])
    old = _load_head(old_path, device)
    train_shared = _log_softmax(_head_logits(train_cache, shared.eval(), device))
    train_old = _log_softmax(_head_logits(train_cache, old, device))
    val_shared = _log_softmax(_head_logits(val_cache, shared.eval(), device))
    val_old = _log_softmax(_head_logits(val_cache, old, device))
    unary, pair = _viewpoint_priors(train_rows, train_cache, train_old, labels_train)
    stats = json.loads((data_root / POLICY_REL / "stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8"))
    train_geom, _, _ = _option_geometry(train_rows, stats)
    val_geom, _, _ = _option_geometry(val_rows, stats)
    train_tokens, train_dino_meta = _load_frame0_dino(data_root, train_rows, "train", device)
    val_tokens, val_dino_meta = _load_frame0_dino(data_root, val_rows, "val", device)
    historical = TaskUtilityPredictor(True, True).to(device)
    historical.load_state_dict(torch.load(selector_path, map_location=device, weights_only=False)["state_dict"])
    historical.eval()
    historical_scores, _ = _predict_task(historical, val_geom, val_tokens, device)
    adaptive_checkpoint = data_root / DEFAULT_RUNTIME / "selector_adapt_old.pth"
    if not adaptive_checkpoint.is_file():
        raise FileNotFoundError(adaptive_checkpoint)
    adaptive = TaskUtilityPredictor(True, True).to(device)
    adaptive.load_state_dict(torch.load(adaptive_checkpoint, map_location=device, weights_only=False)["state_dict"])
    adaptive.eval()
    adaptive_scores, _ = _predict_task(adaptive, val_geom, val_tokens, device)
    # Generate one deterministic random draw per context (not one shared draw).
    rng = np.random.default_rng(SEED)
    action_random = [int(rng.choice([int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]])) for row in val_rows]
    action_historical = _select(val_rows, historical_scores, val_cache["mask"])
    action_adaptive = _select(val_rows, adaptive_scores, val_cache["mask"])
    action_static = _select_prior(val_rows, unary, {})
    action_pair = _select_prior(val_rows, unary, pair)
    current_ids = np.asarray([int(row["current_viewpoint_id"]) for row in val_rows], dtype=np.int64)
    methods: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    def add(name: str, head_name: str, selected: Sequence[int]) -> None:
        values = val_shared if head_name == "shared" else val_old
        metric, prediction = _terminal(values, val_cache["ids"], val_cache["mask"], val_rows, selected, labels_val)
        methods[name] = metric
        predictions[name] = prediction
    add("Random + shared", "shared", action_random)
    add("Historical selector + shared", "shared", action_historical)
    add("Historical selector + old adaptive", "old", action_historical)
    add("StaticViewPrior + old adaptive", "old", action_static)
    add("ViewPairPrior + old adaptive", "old", action_pair)
    add("Adaptive-aware selector + shared", "shared", action_adaptive)
    add("Adaptive-aware selector + old adaptive", "old", action_adaptive)
    permutation = np.random.default_rng(SEED).permutation(len(val_rows))
    shuffled_scores, _ = _predict_task(adaptive, val_geom, val_tokens[permutation], device)
    action_rgb_shuffled = _select(val_rows, shuffled_scores, val_cache["mask"])
    add("Adaptive-aware selector RGB-shuffled + old adaptive", "old", action_rgb_shuffled)
    shuffled_geometry = val_geom.copy()
    geometry_rng = np.random.default_rng(SEED)
    for index in range(len(val_rows)):
        candidate_slots = np.flatnonzero(val_cache["mask"][index])[1:]
        shuffled_geometry[index, candidate_slots] = shuffled_geometry[index, geometry_rng.permutation(candidate_slots)]
    geometry_scores, _ = _predict_task(adaptive, shuffled_geometry, val_tokens, device)
    action_geometry_shuffled = _select(val_rows, geometry_scores, val_cache["mask"])
    add("Adaptive-aware selector Geometry-shuffled + old adaptive", "old", action_geometry_shuffled)
    vis = _load_visibility_targets(data_root, val_rows, "val")
    visibility = np.asarray(vis["scores"][:, 0], dtype=np.float64)
    bottom = visibility < np.quantile(visibility, 1.0 / 3.0)
    selected_distribution: dict[str, Any] = {
        "historical_selector": _histogram(action_historical, current_ids),
        "adaptive_aware": _histogram(action_adaptive, current_ids),
        "historical_s1": _histogram([int(row["s1_viewpoint_id"]) for row in val_rows], current_ids),
    }
    selected_distribution["adaptive_vs_s1_jsd"] = _js_divergence(np.asarray(selected_distribution["adaptive_aware"]["probabilities"]), np.asarray(selected_distribution["historical_s1"]["probabilities"]))
    selected_distribution["adaptive_vs_s1_top5_overlap"] = len(set(selected_distribution["adaptive_aware"]["top5_viewpoints"]) & set(selected_distribution["historical_s1"]["top5_viewpoints"])) / 5.0
    selected_distribution["adaptive_vs_historical_top1_agreement"] = _selector_agreement(action_adaptive, action_historical)["top1_agreement"]
    selected_distribution["adaptive_vs_historical_s1_top1_agreement"] = _selector_agreement(action_adaptive, [int(row["s1_viewpoint_id"]) for row in val_rows])["top1_agreement"]
    prior_agreement = {"adaptive_vs_static": _selector_agreement(action_adaptive, action_static), "adaptive_vs_pair": _selector_agreement(action_adaptive, action_pair), "adaptive_vs_historical": _selector_agreement(action_adaptive, action_historical)}
    selected_shared, _ = _terminal(val_shared, val_cache["ids"], val_cache["mask"], val_rows, action_adaptive, labels_val)
    selected_old, _ = _terminal(val_old, val_cache["ids"], val_cache["mask"], val_rows, action_adaptive, labels_val)
    historical_shared, _ = _terminal(val_shared, val_cache["ids"], val_cache["mask"], val_rows, action_historical, labels_val)
    historical_old, _ = _terminal(val_old, val_cache["ids"], val_cache["mask"], val_rows, action_historical, labels_val)
    switch = np.asarray(action_historical) != np.asarray(action_adaptive)
    switch_indices = np.flatnonzero(switch)
    switch_rows = [val_rows[int(i)] for i in switch_indices]
    switch_ids = val_cache["ids"][switch_indices]
    switch_mask = val_cache["mask"][switch_indices]
    historical_switch_correct = predictions["Historical selector + old adaptive"][switch] == labels_val[switch]
    adaptive_switch_correct = predictions["Adaptive-aware selector + old adaptive"][switch] == labels_val[switch]
    switch_analysis = {
        "n_switch": int(switch.sum()),
        "historical_selected_old_adaptive": _terminal(val_old[switch_indices], switch_ids, switch_mask, switch_rows, np.asarray(action_historical)[switch].tolist(), labels_val[switch])[0] if switch.any() else {},
        "adaptive_selected_old_adaptive": _terminal(val_old[switch_indices], switch_ids, switch_mask, switch_rows, np.asarray(action_adaptive)[switch].tolist(), labels_val[switch])[0] if switch.any() else {},
        "adaptive_correct_historical_wrong": int(np.sum(adaptive_switch_correct & ~historical_switch_correct)),
        "historical_correct_adaptive_wrong": int(np.sum(historical_switch_correct & ~adaptive_switch_correct)),
        "both_correct": int(np.sum(adaptive_switch_correct & historical_switch_correct)),
        "both_wrong": int(np.sum(~adaptive_switch_correct & ~historical_switch_correct)),
        "adaptive_correct_count": int(np.sum(adaptive_switch_correct)),
        "historical_correct_count": int(np.sum(historical_switch_correct)),
    }
    per_class = {name: _class_metrics(labels_val, prediction) for name, prediction in predictions.items() if name in {"Historical selector + shared", "Historical selector + old adaptive", "Adaptive-aware selector + old adaptive", "StaticViewPrior + old adaptive", "ViewPairPrior + old adaptive"}}
    shuffle_metrics = {
        "normal": methods["Adaptive-aware selector + old adaptive"],
        "rgb_shuffled": methods["Adaptive-aware selector RGB-shuffled + old adaptive"],
        "drop_accuracy_pp": (methods["Adaptive-aware selector RGB-shuffled + old adaptive"]["accuracy"] - methods["Adaptive-aware selector + old adaptive"]["accuracy"]) * 100.0,
        "accuracy_drop_pp": (methods["Adaptive-aware selector + old adaptive"]["accuracy"] - methods["Adaptive-aware selector RGB-shuffled + old adaptive"]["accuracy"]) * 100.0,
        "geometry_shuffled": methods["Adaptive-aware selector Geometry-shuffled + old adaptive"],
        "geometry_drop_accuracy_pp": (methods["Adaptive-aware selector Geometry-shuffled + old adaptive"]["accuracy"] - methods["Adaptive-aware selector + old adaptive"]["accuracy"]) * 100.0,
        "geometry_accuracy_drop_pp": (methods["Adaptive-aware selector + old adaptive"]["accuracy"] - methods["Adaptive-aware selector Geometry-shuffled + old adaptive"]["accuracy"]) * 100.0,
    }
    prior_metrics = {
        "unary_viewpoint_count": len(unary),
        "pair_prior_count": len(pair),
        "unary_margin_summary": _summary(unary),
        "pair_margin_summary": _summary(pair),
        "selection_note": "full action selections are summarized in selected_view_distribution.json",
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_POLICY_RECOGNIZER_COUPLING_AUDIT", "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(r['record_id']) for r in train_rows}), "val_records": len({str(r['record_id']) for r in val_rows})}, "labels": list(LABELS), "methods": methods, "selected_view_distribution": selected_distribution, "view_prior_metrics": {"unary_viewpoint_count": len(unary), "pair_prior_count": len(pair), "static_actions": action_static, "pair_actions": action_pair}, "shuffle_metrics": shuffle_metrics, "recognizer_on_selected_distribution": {"adaptive_aware": {"shared": selected_shared, "old_adaptive": selected_old, "adaptive_gain_pp": (selected_old["accuracy"] - selected_shared["accuracy"]) * 100.0}, "historical": {"shared": historical_shared, "old_adaptive": historical_old, "adaptive_gain_pp": (historical_old["accuracy"] - historical_shared["accuracy"]) * 100.0}}, "selector_agreement": prior_agreement, "switch_analysis": switch_analysis, "per_class_metrics": per_class, "high_occlusion": {"definition": "bottom tertile of frame-0 SceneVisibility", "count": int(bottom.sum()), "methods": {name: classification(labels_val[bottom], prediction[bottom]) for name, prediction in predictions.items() if name in {"Random + shared", "Historical selector + shared", "Adaptive-aware selector + old adaptive", "StaticViewPrior + old adaptive"}}}, "leakage_audit": {"policy_test_used": False, "training_used": False, "new_rgb_generated": False, "future_candidate_observation_used_for_selector": False, "future_candidate_recognizer_output_used_for_selector": False, "gt_action_used_for_selector": False, "terminal_selected_real_o1_alone": True}, "protocol": {"action_set": "current/Stay + Stage-A legal candidate_pool", "selector_input": "current frame-0 DINO + geometry + fixed historical visibility-aux architecture", "terminal": "selected real O1 alone", "train_usage": "old adaptive head only to calculate static viewpoint priors", "test_used": False}, "artifacts": {"train_options": str((data_root / DIAGNOSTIC_REL / "train_options.npz").resolve()), "val_options": str((data_root / DIAGNOSTIC_REL / "val_options.npz").resolve()), "historical_selector": str(selector_path.resolve()), "adaptive_selector": str(adaptive_checkpoint.resolve()), "frame0_dino_train": train_dino_meta, "frame0_dino_val": val_dino_meta}, "checkpoint_sha256": {"shared_head": _sha256(shared_path), "old_head": _sha256(old_path), "historical_selector": _sha256(selector_path), "adaptive_selector": _sha256(adaptive_checkpoint)}, "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.monotonic() - started}}
    result["view_prior_metrics"] = prior_metrics
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"seed": SEED, "test_used": False, "action_set": result["protocol"]["action_set"]}, indent=2) + "\n", encoding="utf-8")
    (output_root / "selected_view_distribution.json").write_text(json.dumps(selected_distribution, indent=2) + "\n", encoding="utf-8")
    (output_root / "view_prior_metrics.json").write_text(json.dumps(result["view_prior_metrics"], indent=2) + "\n", encoding="utf-8")
    (output_root / "shuffle_metrics.json").write_text(json.dumps(shuffle_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "recognizer_on_selected_distribution.json").write_text(json.dumps(result["recognizer_on_selected_distribution"], indent=2) + "\n", encoding="utf-8")
    (output_root / "selector_agreement.json").write_text(json.dumps(prior_agreement, indent=2) + "\n", encoding="utf-8")
    (output_root / "switch_analysis.json").write_text(json.dumps(switch_analysis, indent=2) + "\n", encoding="utf-8")
    (output_root / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2) + "\n", encoding="utf-8")
    (output_root / "leakage_audit.json").write_text(json.dumps(result["leakage_audit"], indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output_root.resolve()), "adaptive_accuracy": result["methods"]["Adaptive-aware selector + old adaptive"]["accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
