#!/usr/bin/env python3
"""Val-only audit of action-agnostic real candidate observation quality.

All selectors use the matched reduced12 Moving-Val action set.  Quality values
are read from archived/previously generated diagnostics; no model is trained
and future quality is used only for this privileged diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation, gt_margin
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import load_rows, row_signature

NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
SEED = 42
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/real_candidate_quality_privileged_audit"
QUALITY_NAMES = ("RealPoseConfidence", "SceneVisibility", "HumanVisibility", "TotalVisibility", "ProjectedHumanArea", "VisibleJointRatio", "TemporalMotionRetention")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _metrics(labels: np.ndarray, predictions: np.ndarray, name: str, actions: Sequence[int], rows: Sequence[Mapping[str, Any]], logp: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    result = classification(labels, predictions)
    moves = np.asarray([int(int(action) != int(row["current_viewpoint_id"])) for action, row in zip(actions, rows)])
    selected_scores, selected_margins = [], []
    for index, action in enumerate(actions):
        slot = _slot(ids[index], mask[index], int(action))
        selected_scores.append(float(logp[index, slot, int(labels[index])]))
        selected_margins.append(gt_margin(logp[index, slot], int(labels[index])))
    result.update({"selector": name, "move_rate": float(moves.mean()) if moves.size else 0.0, "stay_rate": float(1.0 - moves.mean()) if moves.size else 1.0, "selected_gt_true_logp": float(np.mean(selected_scores)) if selected_scores else 0.0, "selected_gt_margin": float(np.mean(selected_margins)) if selected_margins else 0.0, "test_used": False})
    return result


def _slot(ids: np.ndarray, mask: np.ndarray, action: int) -> int:
    found = np.flatnonzero((np.asarray(ids) == int(action)) & np.asarray(mask, dtype=bool))
    if found.size != 1:
        raise ValueError(f"viewpoint {action} is not uniquely present")
    return int(found[0])


def _validate_options(options: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> None:
    if len(rows) != 10080 or row_signature(rows) != str(meta.get("signature")):
        raise ValueError("Moving-Val rows/cache signature mismatch")
    if options["features"].shape != (len(rows), 22, 256) or options["logits"].shape != (len(rows), 22, NUM_CLASSES):
        raise ValueError("unexpected shared recognizer cache shape")
    for index, row in enumerate(rows):
        valid = np.flatnonzero(options["mask"][index])
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if options["ids"][index, valid].astype(int).tolist() != expected:
            raise ValueError(f"action-set mismatch at context {index}")


def _load_shared_logp(data_root: Path, rows: Sequence[Mapping[str, Any]], device_name: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead

    cache_root = data_root / "diagnostics/reduced12_dual_route_overnight"
    options_path = cache_root / "val_options.npz"
    options = _load_npz(options_path)
    meta = json.loads((cache_root / "val_options.json").read_text(encoding="utf-8"))
    _validate_options(options, rows, meta)
    checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    model = SharedHead().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    features = np.asarray(options["features"], dtype=np.float32)
    flat = features.reshape(-1, features.shape[-1])
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            outputs.append(model(torch.from_numpy(flat[start : start + batch_size]).to(device, non_blocking=True)).cpu().numpy())
    logits = np.concatenate(outputs).reshape(len(rows), 22, NUM_CLASSES)
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return options["ids"].astype(np.int64), options["mask"].astype(bool), logp


def _archive_confidence(data_root: Path, rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    root = data_root / "datasets/offline/habitat-train/00006-00087"
    output = np.full(ids.shape, np.nan, dtype=np.float64)
    seen: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        path = root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"
        key = str(path)
        if key not in seen:
            with np.load(path, allow_pickle=False) as archive:
                view_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                confidence = np.asarray(archive["confidence"], dtype=np.float64)
            if view_ids.shape != (NUM_VIEWS,) or confidence.shape != (NUM_VIEWS,) or not np.isfinite(confidence).all():
                raise ValueError(f"invalid confidence archive: {path}")
            seen[key] = confidence
        confidence = seen[key]
        for slot in np.flatnonzero(mask[index]):
            match = np.flatnonzero(view_ids == int(ids[index, slot]))
            if match.size != 1:
                raise ValueError(f"confidence viewpoint mismatch: {path}")
            output[index, slot] = confidence[int(match[0])]
    return output, {"archives_read": len(seen), "stay_coverage": float(np.isfinite(output[:, 0]).mean()), "candidate_coverage": float(np.isfinite(output[:, 1:]).sum() / max(mask[:, 1:].sum(), 1)), "definition": "archive confidence scalar (mean YOLO keypoint confidence over 30 frames)"}


def _candidate_quality(data_root: Path, rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    quality: dict[str, np.ndarray] = {}
    coverage: dict[str, Any] = {}
    pose, pose_meta = _archive_confidence(data_root, rows, ids, mask)
    quality["RealPoseConfidence"] = pose
    coverage["RealPoseConfidence"] = {**pose_meta, "source": "archived skeleton NPZ confidence", "granularity": "per-view scalar", "uses_gt_action": False, "uses_future_candidate_observation": True, "deployable": False}
    metrics_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_view_observability_oracle/candidate_metrics.npz"
    metrics = _load_npz(metrics_path)
    scene_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz"
    scene = _load_npz(scene_path)
    temporal_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/skeleton_motion_recoverability_audit/candidate_metrics.npz"
    temporal = _load_npz(temporal_path)
    for archive, source in ((metrics, "human_view_observability_oracle/candidate_metrics.npz"), (scene, "scene_joint_visibility_oracle/visibility.npz")):
        if not np.array_equal(archive["episode_ids"], np.asarray([row["episode_id"] for row in rows])):
            raise ValueError(f"quality artifact episode order mismatch: {source}")
    def fill(name: str, values: np.ndarray, source: str, definition: str) -> None:
        output = np.full(ids.shape, np.nan, dtype=np.float64)
        for index, row in enumerate(rows):
            valid = np.flatnonzero(mask[index, 1:])
            source_ids = np.asarray((metrics if source == "metrics" else scene)["candidate_ids"][index], dtype=np.int64)
            source_mask = np.asarray((metrics if source == "metrics" else scene)["candidate_mask"][index], dtype=bool)
            source_map = {int(view): int(slot) for slot, view in enumerate(source_ids[source_mask])}
            for local in valid:
                view = int(ids[index, local + 1]); source_slot = source_map.get(view)
                if source_slot is None:
                    raise ValueError(f"quality candidate missing: {row['episode_id']}/{view}")
                output[index, local + 1] = float(values[index, source_slot])
        quality[name] = output
        coverage[name] = {"candidate_coverage": float(np.isfinite(output[:, 1:]).sum() / max(mask[:, 1:].sum(), 1)), "stay_coverage": 0.0, "source": source_path(source), "granularity": "per-view scalar", "definition": definition, "uses_gt_action": False, "uses_future_candidate_observation": True, "deployable": False}
    fill("SceneVisibility", scene["scene_visibility_score"], "scene", "existing Habitat environmental joint visibility score")
    fill("VisibleJointRatio", np.asarray(scene["visibility"]).mean(axis=2), "scene", "mean of existing 17-joint scene visibility vector")
    self_visibility, total_visibility = _load_human_self_visibility(rows, ids, mask)
    quality["HumanVisibility"] = self_visibility
    coverage["HumanVisibility"] = {"candidate_coverage": float(np.isfinite(self_visibility[:, 1:]).sum() / max(mask[:, 1:].sum(), 1)), "stay_coverage": 0.0, "source": "experiments/reduced12_eight_placement_v1/human_self_visibility_oracle/diagnostics.json", "granularity": "per-view scalar", "definition": "existing human-only body-part visibility diagnostic", "uses_gt_action": False, "uses_future_candidate_observation": True, "deployable": False}
    quality["TotalVisibility"] = total_visibility
    coverage["TotalVisibility"] = {"candidate_coverage": float(np.isfinite(total_visibility[:, 1:]).sum() / max(mask[:, 1:].sum(), 1)), "stay_coverage": 0.0, "source": "experiments/reduced12_eight_placement_v1/human_self_visibility_oracle/diagnostics.json", "granularity": "per-view scalar", "definition": "existing scene+human observability score", "uses_gt_action": False, "uses_future_candidate_observation": True, "deployable": False}
    fill("ProjectedHumanArea", metrics["projected_area"], "metrics", "existing normalized projected human area")
    temporal_map: dict[tuple[str, int], float] = {}
    for episode_id, candidate_id, value in zip(temporal["episode_id"], temporal["candidate_id"], temporal["motion_retention"]):
        key = (str(episode_id), int(candidate_id))
        if key in temporal_map:
            raise ValueError(f"duplicate temporal quality key: {key}")
        temporal_map[key] = float(value)
    temporal_quality = np.full(ids.shape, np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        for slot in np.flatnonzero(mask[index, 1:]):
            key = (str(row["episode_id"]), int(ids[index, slot + 1]))
            if key not in temporal_map:
                raise ValueError(f"temporal quality missing: {key}")
            temporal_quality[index, slot + 1] = temporal_map[key]
    quality["TemporalMotionRetention"] = temporal_quality
    coverage["TemporalMotionRetention"] = {"candidate_coverage": float(np.isfinite(temporal_quality[:, 1:]).sum() / max(mask[:, 1:].sum(), 1)), "stay_coverage": 0.0, "source": "experiments/reduced12_eight_placement_v1/skeleton_motion_recoverability_audit/candidate_metrics.npz", "granularity": "per-view sequence-level scalar", "definition": "existing action-agnostic estimated-skeleton motion retention proxy; not GT motion fidelity", "uses_gt_action": False, "uses_future_candidate_observation": True, "deployable": False}
    return quality, coverage


def _load_human_self_visibility(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_self_visibility_oracle/diagnostics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[str, dict[int, tuple[float, float]]] = {}
    for item in payload.get("candidate_records", []):
        records.setdefault(str(item["episode_id"]), {})[int(item["candidate_id"])] = (float(item["human_self_visibility"]), float(item["total_visibility"]))
    human = np.full(ids.shape, np.nan, dtype=np.float64)
    total = np.full(ids.shape, np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        mapping = records.get(str(row["episode_id"]), {})
        for slot in np.flatnonzero(mask[index, 1:]):
            values = mapping.get(int(ids[index, slot + 1]))
            if values is None:
                raise ValueError(f"human visibility missing: {row['episode_id']}/{ids[index, slot + 1]}")
            human[index, slot + 1], total[index, slot + 1] = values
    return human, total


def source_path(kind: str) -> str:
    return "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz" if kind == "scene" else "experiments/reduced12_eight_placement_v1/human_view_observability_oracle/candidate_metrics.npz"


def _select_quality(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, values: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        finite = np.isfinite(values[index, valid])
        if not finite.any():
            actions.append(int(row["current_viewpoint_id"])); continue
        candidate_slots = valid[finite]
        # Stable first occurrence resolves exact ties; stay participates whenever its quality exists.
        best_slot = int(candidate_slots[np.argmax(values[index, candidate_slots])])
        actions.append(int(ids[index, best_slot]))
    return actions


def _oracle_actions(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, logp: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index]); label = int(row["label_id"])
        slot = int(valid[np.argmax(logp[index, valid, label])])
        actions.append(int(ids[index, slot]))
    return actions


def _quality_diagnostics(name: str, values: np.ndarray, rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, logp: np.ndarray, labels: np.ndarray, gt_actions: Sequence[int]) -> dict[str, Any]:
    q, truth, margins, correct = [], [], [], []
    within, top1, top3 = [], [], []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index, 1:]) + 1
        finite = valid[np.isfinite(values[index, valid])]
        if finite.size == 0: continue
        q.extend(values[index, finite].tolist()); truth.extend(logp[index, finite, labels[index]].tolist()); margins.extend([gt_margin(logp[index, slot], int(labels[index])) for slot in finite]); correct.extend((np.argmax(logp[index, finite], axis=1) == labels[index]).astype(float).tolist())
        if finite.size >= 2:
            within.append(correlation(values[index, finite], logp[index, finite, labels[index]], spearman=True))
        ranked = finite[np.argsort(values[index, finite])[::-1]]
        oracle = int(gt_actions[index]); top1.append(int(ranked.size > 0 and int(ids[index, ranked[0]]) == oracle)); top3.append(int(oracle in ids[index, ranked[:3]].astype(int)))
    return {"name": name, "candidate_count": len(q), "candidate_spearman_true_logp": correlation(q, truth, spearman=True), "candidate_spearman_gt_margin": correlation(q, margins, spearman=True), "point_biserial_correctness": correlation(q, correct), "within_context_spearman_mean": float(np.mean(within)) if within else 0.0, "within_context_spearman_median": float(np.median(within)) if within else 0.0, "within_context_count": len(within), "top1_overlap_gt_true_logp": float(np.mean(top1)) if top1 else 0.0, "top3_overlap_gt_true_logp": float(np.mean(top3)) if top3 else 0.0, "mean_quality_correct": float(np.mean(np.asarray(q)[np.asarray(correct, dtype=bool)])) if any(correct) else 0.0, "mean_quality_wrong": float(np.mean(np.asarray(q)[~np.asarray(correct, dtype=bool)])) if any(np.asarray(correct) == 0) else 0.0}


def _transitions(labels: np.ndarray, old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    a = int(np.sum((old != labels) & (new == labels))); b = int(np.sum((old == labels) & (new != labels)))
    return {"old_wrong_new_correct": a, "old_correct_new_wrong": b, "both_correct": int(np.sum((old == labels) & (new == labels))), "both_wrong": int(np.sum((old != labels) & (new != labels))), "correction_rate": float(a / max(np.sum(old != labels), 1)), "regression_rate": float(b / max(np.sum(old == labels), 1)), "net_gain": float((a - b) / max(len(labels), 1))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root(); _, rows = load_rows(data_root)
    ids, mask, logp = _load_shared_logp(data_root, rows, args.device, args.inference_batch_size)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    quality, source_meta = _candidate_quality(data_root, rows, ids, mask)
    historical_result = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/historical_route1_shared_head_synergy/result.json").read_text(encoding="utf-8"))
    historical_actions = [int(x) for x in historical_result["selected_actions"]]
    stay_actions = [int(row["current_viewpoint_id"]) for row in rows]
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice([int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]])) for row in rows]
    gt_actions = _oracle_actions(rows, ids, mask, logp)
    original_predictions = np.asarray([int(np.argmax(logp[i, _slot(ids[i], mask[i], stay_actions[i])])) for i in range(len(rows))])
    metrics: dict[str, Any] = {"Stay": _metrics(labels, original_predictions, "Stay", stay_actions, rows, logp, ids, mask)}
    selector_actions: dict[str, list[int]] = {"Random legal": random_actions, "Historical Route-1": historical_actions}
    for name, values in quality.items(): selector_actions[name] = _select_quality(rows, ids, mask, values)
    selector_actions["GT-TrueLogP Oracle"] = gt_actions
    for name, actions in selector_actions.items():
        predictions = np.asarray([int(np.argmax(logp[i, _slot(ids[i], mask[i], action)])) for i, action in enumerate(actions)])
        metrics[name] = _metrics(labels, predictions, name, actions, rows, logp, ids, mask)
    metrics["Legal AnyCorrect"] = {"selector": "Legal AnyCorrect", "coverage": float(np.mean([np.any(np.argmax(logp[i, np.flatnonzero(mask[i, 1:]) + 1], axis=1) == labels[i]) for i in range(len(rows))])), "coverage_definition": "any legal candidate (candidate-only; stay excluded)", "test_used": False}
    gt_actions_shared = gt_actions
    correlations = {name: _quality_diagnostics(name, values, rows, ids, mask, logp, labels, gt_actions_shared) for name, values in quality.items()}
    best_name = max(QUALITY_NAMES, key=lambda name: metrics[name]["accuracy"])
    best_predictions = np.asarray([int(np.argmax(logp[i, _slot(ids[i], mask[i], selector_actions[best_name][i])])) for i in range(len(rows))])
    stay_predictions = original_predictions; random_predictions = np.asarray([int(np.argmax(logp[i, _slot(ids[i], mask[i], random_actions[i])])) for i in range(len(rows))])
    result = {"experiment_id": "REDUCED12_REAL_CANDIDATE_QUALITY_PRIVILEGED_AUDIT", "protocol": {"split": "Moving Val", "contexts": len(rows), "action_set": "current/stay + Stage-A legal candidate_pool", "recognizer": "frozen reduced12 ST-GCN encoder + shared adapted head", "test_used": False, "training_used": False, "deployable": False}, "quality_sources": source_meta, "selectors": metrics, "quality_diagnostics": correlations, "best_action_agnostic_selector": best_name, "deltas": {name: {"vs_stay_accuracy_pp": 100.0 * (metrics[name]["accuracy"] - metrics["Stay"]["accuracy"]), "vs_random_accuracy_pp": 100.0 * (metrics[name]["accuracy"] - metrics["Random legal"]["accuracy"]), "gap_to_historical_shared_pp": 100.0 * (metrics["Historical Route-1"]["accuracy"] - metrics[name]["accuracy"]), "gap_to_gt_true_logp_shared_pp": 100.0 * (metrics["GT-TrueLogP Oracle"]["accuracy"] - metrics[name]["accuracy"])} for name in QUALITY_NAMES}, "error_decomposition": {"best_vs_stay": _transitions(labels, stay_predictions, best_predictions), "best_vs_random": _transitions(labels, random_predictions, best_predictions)}, "occlusion_stratification": {"available": False, "reason": "Existing scene/human quality artifacts cover legal candidates only; no current/stay occlusion scalar is available, so no artificial stay value or quantile split was created."}, "selected_actions": selector_actions, "flags": {"future_candidate_quality_used_for_privileged_diagnostic_only": True, "gt_action_used_for_quality_score": False, "gt_recognizer_correctness_used_for_quality_score": False, "policy_test_used": False}}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "quality_sources.json").write_text(json.dumps(source_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "per_selector_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "correlation_metrics.json").write_text(json.dumps(correlations, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "context_transitions.json").write_text(json.dumps(result["error_decomposition"], indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "occlusion_stratified_metrics.json").write_text(json.dumps(result["occlusion_stratification"], indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _analysis(result: Mapping[str, Any]) -> str:
    selectors = result["selectors"]; best = result["best_action_agnostic_selector"]; d = result["deltas"][best]; corr = result["quality_diagnostics"][best]
    historical_gap = -float(d["gap_to_historical_shared_pp"])
    historical_comparison = (
        f"{historical_gap:+.3f}pp (higher)"
        if historical_gap >= 0.0
        else f"{-historical_gap:.3f}pp lower"
    )
    lines = ["# Real Candidate Quality Privileged Audit", "", "Moving Val only: 10,080 contexts; recognizer is the frozen reduced12 encoder plus shared adapted head. Quality selectors are privileged because they read archived future candidate quality, while GT action/correctness are excluded from quality scores.", "", "## Main results", "", "| Selector | Type | HAR Acc | Macro-F1 | Move rate |", "|---|---|---:|---:|---:|"]
    types = {"Stay": "deployable baseline", "Random legal": "baseline", "Historical Route-1": "learned task-aware", "GT-TrueLogP Oracle": "privileged task-aware"}
    for name in ["Stay", "Random legal", *QUALITY_NAMES, "Historical Route-1", "GT-TrueLogP Oracle"]:
        item = selectors[name]; lines.append(f"| {name} | {types.get(name, 'privileged action-agnostic')} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item.get('move_rate', 0.0):.6f} |")
    lines += ["| Legal AnyCorrect | coverage only | — | — | — |", "", f"Best action-agnostic selector: **{best}**, {selectors[best]['accuracy']:.6f} Acc / {selectors[best]['macro_f1']:.6f} Macro-F1.", f"Against Stay: {d['vs_stay_accuracy_pp']:+.3f}pp; against Random: {d['vs_random_accuracy_pp']:+.3f}pp; versus Historical Route-1 + Shared: {historical_comparison}; gap to GT-TrueLogP Oracle + Shared: {d['gap_to_gt_true_logp_shared_pp']:+.3f}pp.", "", "## Ranking diagnostics", "", f"Best selector within-context Spearman with shared GT true-logp: mean {corr['within_context_spearman_mean']:.6f}, median {corr['within_context_spearman_median']:.6f}; candidate-level Spearman: {corr['candidate_spearman_true_logp']:.6f}; GT-TrueLogP top-1/top-3 overlap: {corr['top1_overlap_gt_true_logp']:.6f}/{corr['top3_overlap_gt_true_logp']:.6f}.", "", "## Occlusion subset", "", "No occlusion-stratified result is reported: existing Scene/Human artifacts have legal-candidate values but no current/stay value, so no synthetic stay score or unsupported quantile split was introduced.", "", "## Decision", ""]
    if d["vs_random_accuracy_pp"] <= 2.0 and all(result["deltas"][name]["vs_random_accuracy_pp"] <= 2.0 for name in QUALITY_NAMES):
        lines.append("All action-agnostic quality selectors are within 2pp of Random. Kill the hypothesis that scalar ‘look-clearer’ quality alone supports the main Route-1 strategy; do not train a future-quality predictor from this audit.")
    elif d["vs_random_accuracy_pp"] >= 5.0:
        if historical_gap >= 0.0:
            lines.append(f"A quality selector exceeds Random by at least 5pp and is {historical_gap:.3f}pp above the historical task-aware route on this matched audit, but it remains privileged rather than deployable. Keep future observability as a possible target, while combining it with task evidence rather than relying on quality alone.")
        else:
            lines.append(f"A quality selector exceeds Random by at least 5pp but remains {-historical_gap:.3f}pp below the historical task-aware route. Keep future observability as a possible target, but combine it with task evidence rather than relying on quality alone.")
    elif selectors[best]["accuracy"] >= 0.48:
        lines.append("The best action-agnostic selector reaches at least 48%; keep scalar/structured observability prediction as a candidate follow-up, while noting the privileged gap to task-aware selection.")
    else:
        lines.append("Quality alone is insufficient for the principal Route-1 headroom. If pursued later, require structured visibility and task evidence; this audit does not authorize predictor training.")
    lines += ["", "Observation quality is not recognition utility, and a privileged real-quality selector is not a deployable future-quality predictor.", "", "No Policy Test was read; no RGB, skeleton, DINO, recognizer, policy or runtime artifact was modified."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    args = parser.parse_args()
    result = run(args)
    (OUTPUT_ROOT / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (OUTPUT_ROOT / "config.json").write_text(json.dumps({"device": args.device, "inference_batch_size": args.inference_batch_size, "seed": SEED, "test_used": False, "training_used": False}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str((OUTPUT_ROOT / "result.json").resolve()), "contexts": result["protocol"]["contexts"], "best": result["best_action_agnostic_selector"], "accuracy": result["selectors"][result["best_action_agnostic_selector"]]["accuracy"], "test_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
