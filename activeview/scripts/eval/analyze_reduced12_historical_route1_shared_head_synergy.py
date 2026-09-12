#!/usr/bin/env python3
"""Audit a historical Route-1 selector with original and shared heads.

The selector is replayed from its frozen checkpoint on the matched reduced12
Moving-Val population.  Only the recognizer head is swapped; selected
viewpoint IDs are therefore identical for both recognizers.  Policy Test and
future observations are never read as selector inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.policy_data import load_feature_statistics
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation, gt_margin
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_h1_stay_aware_batch import (
    StayDataset,
    _load_stay_checkpoint,
    _scores,
)

NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
SEED = 42
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/historical_route1_shared_head_synergy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _validate_options(options: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> None:
    if len(rows) != 10080 or row_signature(rows) != str(meta.get("signature")):
        raise ValueError("Moving-Val rows do not match cached signature")
    features = np.asarray(options["features"])
    logits = np.asarray(options["logits"])
    ids = np.asarray(options["ids"])
    mask = np.asarray(options["mask"], dtype=bool)
    if features.shape != (len(rows), 22, 256) or logits.shape != (len(rows), 22, NUM_CLASSES):
        raise ValueError(f"unexpected option cache shapes: {features.shape}, {logits.shape}")
    if ids.shape != mask.shape or ids.shape[1] != 22:
        raise ValueError("invalid option cache id/mask shape")
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if ids[index, valid].astype(int).tolist() != expected:
            raise ValueError(f"candidate/action mismatch at row {index}")
        if len(expected) > 22 or len(valid) != len(expected):
            raise ValueError(f"invalid legal action count at row {index}")


def _candidate_cache(options: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    count = len(rows)
    max_candidates = 21
    geodesic = np.zeros((count, max_candidates), dtype=np.float32)
    for index, row in enumerate(rows):
        values = np.asarray(row["candidate_geodesic"], dtype=np.float32)
        geodesic[index, : values.size] = values
    return {
        "candidate_logp": np.asarray(options["logits"][:, 1:], dtype=np.float32),
        "current_logp": np.asarray(options["logits"][:, 0], dtype=np.float32),
        "candidate_ids": np.asarray(options["ids"][:, 1:], dtype=np.int64),
        "candidate_mask": np.asarray(options["mask"][:, 1:], dtype=bool),
        "candidate_geodesic": geodesic,
    }


def _load_shared_head(path: Path, device: torch.device) -> tuple[SharedHead, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    head = SharedHead().to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, payload


def _head_logits(head: SharedHead, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float32).reshape(-1, features.shape[-1])
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = torch.from_numpy(flat[start : start + batch_size]).to(device, non_blocking=True)
            output.append(head(batch).cpu().numpy())
    return np.concatenate(output, axis=0).reshape(*features.shape[:-1], NUM_CLASSES).astype(np.float32)


def _slot(ids: np.ndarray, mask: np.ndarray, action: int) -> int:
    matches = np.flatnonzero((ids == int(action)) & mask)
    if matches.size != 1:
        raise ValueError(f"action {action} is not uniquely present in legal action set")
    return int(matches[0])


def _actions_from_scores(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], stays: np.ndarray, candidates: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(cache["candidate_mask"][index])
        values = np.concatenate(([float(stays[index])], np.asarray(candidates[index, valid], dtype=np.float64)))
        options = [int(row["current_viewpoint_id"])] + cache["candidate_ids"][index, valid].astype(int).tolist()
        actions.append(options[int(np.argmax(values))])
    return actions


def _random_actions(rows: Sequence[Mapping[str, Any]], seed: int = SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.choice([int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]])) for row in rows]


def _evaluate_actions(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, logp: np.ndarray, actions: Sequence[int], name: str) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    predictions: list[int] = []
    selected_scores: list[float] = []
    selected_margins: list[float] = []
    moves = 0
    for index, action in enumerate(actions):
        slot = _slot(ids[index], mask[index], int(action))
        values = logp[index, slot]
        predictions.append(int(np.argmax(values)))
        label = int(labels[index])
        selected_scores.append(float(values[label]))
        selected_margins.append(gt_margin(values, label))
        moves += int(int(action) != int(rows[index]["current_viewpoint_id"]))
    result = classification(labels, np.asarray(predictions, dtype=np.int64))
    result.update({"selector": name, "move_rate": float(moves / len(rows)), "stay_rate": float(1.0 - moves / len(rows)), "selected_gt_true_logp": float(np.mean(selected_scores)), "selected_gt_margin": float(np.mean(selected_margins)), "test_used": False})
    return result


def _oracle_actions(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, logp: np.ndarray, mode: str) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        label = int(row["label_id"])
        values = logp[index, valid]
        if mode == "gt_true_logp":
            scores = values[:, label]
        elif mode == "any_correct":
            correct = np.argmax(values, axis=1) == label
            actions.append(int(ids[index, valid[np.flatnonzero(correct)[0]]]) if bool(correct.any()) else int(row["current_viewpoint_id"]))
            continue
        else:
            margins = np.asarray([gt_margin(item, label) for item in values], dtype=np.float64)
            scores = margins
        actions.append(int(ids[index, valid[int(np.argmax(scores))]]))
    return actions


def _coverage(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, logp: np.ndarray, include_stay: bool) -> dict[str, Any]:
    values: list[bool] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index]) if include_stay else np.flatnonzero(mask[index])[1:]
        values.append(bool(np.any(np.argmax(logp[index, valid], axis=1) == int(row["label_id"]))))
    return {"contexts": int(len(values)), "covered": int(np.sum(values)), "coverage": float(np.mean(values)), "include_stay": include_stay}


def _context_transitions(labels: np.ndarray, old_pred: np.ndarray, new_pred: np.ndarray) -> dict[str, Any]:
    old_correct = old_pred == labels
    new_correct = new_pred == labels
    a = int(np.sum(~old_correct & new_correct)); b = int(np.sum(old_correct & ~new_correct))
    c = int(np.sum(old_correct & new_correct)); d = int(np.sum(~old_correct & ~new_correct))
    total = max(len(labels), 1)
    return {"old_wrong_shared_correct": a, "old_correct_shared_wrong": b, "both_correct": c, "both_wrong": d, "correction_rate": float(a / max(np.sum(~old_correct), 1)), "regression_rate": float(b / max(np.sum(old_correct), 1)), "net_gain": float((a - b) / total)}


def _selected_view_metrics(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, old_logp: np.ndarray, shared_logp: np.ndarray, actions: Sequence[int]) -> tuple[dict[str, Any], float]:
    frequency = np.zeros(NUM_VIEWS, dtype=np.int64)
    old_correct = np.zeros(NUM_VIEWS, dtype=np.int64)
    shared_correct = np.zeros(NUM_VIEWS, dtype=np.int64)
    for index, action in enumerate(actions):
        viewpoint = int(action); frequency[viewpoint] += 1
        slot = _slot(ids[index], mask[index], viewpoint)
        label = int(rows[index]["label_id"])
        old_correct[viewpoint] += int(np.argmax(old_logp[index, slot]) == label)
        shared_correct[viewpoint] += int(np.argmax(shared_logp[index, slot]) == label)
    entries: list[dict[str, Any]] = []
    observed = frequency > 0
    gains = np.zeros(NUM_VIEWS, dtype=np.float64)
    for viewpoint in range(NUM_VIEWS):
        count = int(frequency[viewpoint])
        old_acc = float(old_correct[viewpoint] / count) if count else 0.0
        shared_acc = float(shared_correct[viewpoint] / count) if count else 0.0
        gains[viewpoint] = shared_acc - old_acc
        entries.append({"viewpoint_id": viewpoint, "selection_frequency": count, "original_accuracy": old_acc, "shared_accuracy": shared_acc, "delta_accuracy": float(gains[viewpoint])})
    corr = correlation(frequency[observed], gains[observed], spearman=True) if int(np.sum(observed)) >= 2 else 0.0
    return {"viewpoints": entries, "observed_viewpoints": int(np.sum(observed)), "spearman_selection_frequency_vs_shared_gain": corr}, corr


def _per_class(rows: Sequence[Mapping[str, Any]], original: Mapping[str, Any], shared: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in LABELS:
        old = original["per_class"][label]; new = shared["per_class"][label]
        output[label] = {"support": int(old["support"]), "original_recall": float(old["recall"]), "shared_recall": float(new["recall"]), "delta_recall": float(new["recall"] - old["recall"]), "original_f1": float(old["f1"]), "shared_f1": float(new["f1"]), "delta_f1": float(new["f1"] - old["f1"])}
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root()
    device = _require_cuda(args.device)
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    feature_root = policy_root / "stage_c"
    train_rows, val_rows = load_rows(data_root)
    options_path = data_root / "diagnostics/reduced12_dual_route_overnight/val_options.npz"
    options_meta_path = options_path.with_suffix(".json")
    with np.load(options_path, allow_pickle=False) as archive:
        options = {key: np.asarray(archive[key]) for key in archive.files}
    options_meta = json.loads(options_meta_path.read_text(encoding="utf-8"))
    _validate_options(options, val_rows, options_meta)
    original_logp = _log_softmax(options["logits"])
    stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json")
    feature_summary_path = feature_root / "stage_c_feature_summary.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    candidate_cache = _candidate_cache(options, val_rows)
    old_checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_stay_aware_objective_batch/margin_listwise/margin_listwise_best.pth"
    if not old_checkpoint.exists():
        raise FileNotFoundError(f"historical Route-1 checkpoint missing: {old_checkpoint}")
    old_model = _load_stay_checkpoint(old_checkpoint, feature_summary, device)
    stays, candidate_scores = _scores(old_model, StayDataset(val_rows, candidate_cache, stats), device, args.batch_size)
    historical_actions = _actions_from_scores(val_rows, candidate_cache, stays, candidate_scores)
    shared_checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth"
    shared_head, shared_payload = _load_shared_head(shared_checkpoint, device)
    shared_logits = _head_logits(shared_head, options["features"], device, args.inference_batch_size)
    shared_logp = _log_softmax(shared_logits)
    ids = np.asarray(options["ids"], dtype=np.int64); mask = np.asarray(options["mask"], dtype=bool)
    labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    stay_actions = [int(row["current_viewpoint_id"]) for row in val_rows]
    random_actions = _random_actions(val_rows)
    gt_actions_original = _oracle_actions(val_rows, ids, mask, original_logp, "gt_true_logp")
    gt_actions = _oracle_actions(val_rows, ids, mask, shared_logp, "gt_true_logp")
    metrics = {
        "stay_original": _evaluate_actions(val_rows, ids, mask, original_logp, stay_actions, "Stay/current + Original"),
        "stay_shared": _evaluate_actions(val_rows, ids, mask, shared_logp, stay_actions, "Stay/current + Shared"),
        "historical_original": _evaluate_actions(val_rows, ids, mask, original_logp, historical_actions, "Historical strong Route-1 + Original"),
        "historical_shared": _evaluate_actions(val_rows, ids, mask, shared_logp, historical_actions, "Historical strong Route-1 + Shared"),
        "random_shared": _evaluate_actions(val_rows, ids, mask, shared_logp, random_actions, "Random legal + Shared"),
        "gt_true_logp_original": _evaluate_actions(val_rows, ids, mask, original_logp, gt_actions_original, "GT-TrueLogP Oracle + Original"),
        "gt_true_logp_shared": _evaluate_actions(val_rows, ids, mask, shared_logp, gt_actions, "GT-TrueLogP Oracle + Shared"),
    }
    original_coverage = {"candidate_only": _coverage(val_rows, ids, mask, original_logp, False), "stay_plus_candidates": _coverage(val_rows, ids, mask, original_logp, True)}
    shared_coverage = {"candidate_only": _coverage(val_rows, ids, mask, shared_logp, False), "stay_plus_candidates": _coverage(val_rows, ids, mask, shared_logp, True)}
    transitions = {"stay": _context_transitions(labels, np.argmax(original_logp[np.arange(len(labels)), [ _slot(ids[i], mask[i], stay_actions[i]) for i in range(len(labels))]], axis=1), np.argmax(shared_logp[np.arange(len(labels)), [ _slot(ids[i], mask[i], stay_actions[i]) for i in range(len(labels))]], axis=1)), "historical": _context_transitions(labels, np.asarray([int(np.argmax(original_logp[i, _slot(ids[i], mask[i], historical_actions[i])])) for i in range(len(labels))]), np.asarray([int(np.argmax(shared_logp[i, _slot(ids[i], mask[i], historical_actions[i])])) for i in range(len(labels))]))}
    per_view, _ = _selected_view_metrics(val_rows, ids, mask, original_logp, shared_logp, historical_actions)
    per_class = _per_class(val_rows, metrics["historical_original"], metrics["historical_shared"])
    stay_original = float(metrics["stay_original"]["accuracy"]); stay_shared = float(metrics["stay_shared"]["accuracy"])
    historical_original = float(metrics["historical_original"]["accuracy"]); historical_shared = float(metrics["historical_shared"]["accuracy"])
    expected_additive = stay_original + (stay_shared - stay_original) + (historical_original - stay_original)
    result = {
        "protocol": {"population": "reduced12 matched Moving Val", "contexts": len(val_rows), "action_set": "current/stay + Stage-A legal candidate_pool", "route": "Route-1 pre-observation selector", "labels": list(LABELS), "test_used": False},
        "historical_method": {"name": "Stay-aware GTMargin Listwise (OldBestOneShot)", "historical_accuracy": 0.4587301587301587, "historical_macro_f1": 0.446342, "historical_result_path": str((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").resolve()), "historical_checkpoint": str(old_checkpoint.resolve()), "historical_checkpoint_sha256": _sha256(old_checkpoint), "recognizer": "frozen reduced12 ST-GCN original head", "candidate_action_set": "current + Stage-A legal candidates", "stay_included": True, "future_candidate_observation_unavailable": True, "protocol_match": True},
        "historical_search": {"compliant_gt50_method_found": False, "selected_closest_compliant": "OldBestOneShot / margin_listwise", "excluded": [{"name": "RealEvidence-GTMarginListwise", "accuracy": 0.502778, "reason": "future candidate real evidence used"}, {"name": "EXP036-R1 dense/GMRF Route-1", "reason": "16-class historical protocol mismatch"}]},
        "shared_head": {"checkpoint": str(shared_checkpoint.resolve()), "checkpoint_sha256": _sha256(shared_checkpoint), "manifest": str((REPO_ROOT / "experiments/reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_recognizer_manifest.json").resolve()), "payload_epoch": int(shared_payload.get("epoch", -1)), "architecture": "Linear(256,256)->GELU->Linear(256,12)"},
        "metrics": metrics,
        "deltas": {"recognizer_gain_at_stay": stay_shared - stay_original, "policy_gain_original": historical_original - stay_original, "policy_gain_shared": historical_shared - stay_shared, "combined_gain": historical_shared - stay_original, "expected_additive_accuracy": expected_additive, "synergy": historical_shared - expected_additive},
        "original_privileged_references": {"stay": metrics["stay_original"], "gt_true_logp_oracle": metrics["gt_true_logp_original"], "legal_any_correct_coverage": original_coverage["stay_plus_candidates"], "candidate_only_any_correct_coverage": original_coverage["candidate_only"]},
        "shared_privileged_references": {"stay": metrics["stay_shared"], "random_legal": metrics["random_shared"], "historical_route1": metrics["historical_shared"], "gt_true_logp_oracle": metrics["gt_true_logp_shared"], "legal_any_correct_coverage": shared_coverage["stay_plus_candidates"], "candidate_only_any_correct_coverage": shared_coverage["candidate_only"]},
        "any_correct_coverage": shared_coverage,
        "context_transitions": transitions,
        "per_class_metrics": per_class,
        "per_view_metrics": per_view,
        "selected_actions": historical_actions,
        "test_read": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "per_view_metrics.json").write_text(json.dumps(per_view, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "context_transition_counts.json").write_text(json.dumps(transitions, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _analysis(result: Mapping[str, Any]) -> str:
    m = result["metrics"]; d = result["deltas"]; hist = result["historical_method"]
    lines = ["# Historical Route-1 × Shared Adapted Head Synergy Audit", "", "## Historical method", "", f"No deployment-legal reduced12 matched-protocol historical Route-1 method above 50% was found. The closest compliant method is **{hist['name']}**, with the archived historical result {hist['historical_accuracy']:.6f} Acc / {hist['historical_macro_f1']:.6f} Macro-F1. Its checkpoint was replayed on the current 10,080-context matched population.", "", "The 0.502778 RealEvidence-GTMarginListwise result is excluded because it consumes future candidate evidence; old 16-class EXP036 results are protocol mismatches.", "", "## Same selected viewpoints", "", "| Policy / recognizer | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|"]
    for key in ("stay_original", "stay_shared", "historical_original", "historical_shared"):
        item = m[key]; lines.append(f"| {item['selector']} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} |")
    lines += ["", "## Additive decomposition", "", f"- Δ recognizer at Stay: **{d['recognizer_gain_at_stay']:+.6f}**", f"- Δ policy under Original: **{d['policy_gain_original']:+.6f}**", f"- Δ policy under Shared: **{d['policy_gain_shared']:+.6f}**", f"- Δ combined (Historical+Shared − Stay+Original): **{d['combined_gain']:+.6f}**", f"- Expected additive accuracy: **{d['expected_additive_accuracy']:.6f}**", f"- Synergy: **{d['synergy']:+.6f}**", "", "## Context transitions", "", "For both Stay and the historical selected viewpoint, `result.json` gives old-wrong→shared-correct correction, old-correct→shared-wrong regression, both-correct and both-wrong counts.", "", "## Shared recognizer privileged references", "", f"GT-TrueLogP Oracle: {m['gt_true_logp_shared']['accuracy']:.6f} Acc / {m['gt_true_logp_shared']['macro_f1']:.6f} Macro-F1; Legal AnyCorrect Coverage (stay + candidates): {result['any_correct_coverage']['stay_plus_candidates']['coverage']:.6f}; candidate-only coverage: {result['any_correct_coverage']['candidate_only']['coverage']:.6f}.", "", "## Decision", ""]
    if d["combined_gain"] >= 0.02:
        lines.append("Historical policy + Shared head clears the +2pp gate. This audit does not retrain the policy automatically; a separately approved second-stage replay would be required.")
    else:
        lines.append("Historical policy + Shared head does not clear the +2pp gate. Stop here and do not retrain the historical policy.")
    lines += ["", "No Policy Test files were read. No recognizer, policy architecture, data, RGB, skeleton or DINO artifact was modified."]
    return "\n".join(lines) + "\n"


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    args = parser.parse_args()
    result = run(args)
    (OUTPUT_ROOT / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    config = {"device": args.device, "batch_size": args.batch_size, "inference_batch_size": args.inference_batch_size, "seed": SEED, "test_used": False, "policy_test_read": False, "output": str(OUTPUT_ROOT.resolve())}
    (OUTPUT_ROOT / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str((OUTPUT_ROOT / "result.json").resolve()), "contexts": result["protocol"]["contexts"], "historical_shared_accuracy": result["metrics"]["historical_shared"]["accuracy"], "test_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
