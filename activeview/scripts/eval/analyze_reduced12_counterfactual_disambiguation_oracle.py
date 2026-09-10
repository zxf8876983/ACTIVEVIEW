#!/usr/bin/env python3
"""Val-only action-agnostic counterfactual disambiguation audit.

The selector scores a camera by how differently frozen ST-GCN evidence would
look under the actions supported by the current posterior.  This is a
privileged diagnostic: alternate-action observations are never available to a
deployable policy.  The runner intentionally fails before scoring when the
archive cannot provide a strict same-placement/same-camera counterfactual
pairing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl

NUM_CLASSES = 12
SEED = 42
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
EXPERIMENT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/counterfactual_disambiguation_oracle"
DATASET = "policy_reduced12_eight_placement_v1"
ARCHIVE_RELATIVE = "datasets/offline/habitat-train/00006-00087"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _classification(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(LABELS):
        support = int(confusion[class_id].sum())
        predicted = int(confusion[:, class_id].sum())
        true_positive = int(confusion[class_id, class_id])
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"support": support, "recall": recall, "f1": f1}
    return {
        "n": int(labels.size),
        "accuracy": float(np.mean(labels == predictions)) if labels.size else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _utility(logp: np.ndarray, label: int) -> float:
    competitors = np.delete(np.asarray(logp), int(label))
    return float(logp[int(label)] - np.max(competitors))


def _load_inputs(data_root: Path) -> dict[str, Any]:
    policy_root = data_root / "datasets" / DATASET
    stage_c_rows = load_jsonl(policy_root / "stage_c/features/val.jsonl")
    moving_rows = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    moving_ids = {str(row["episode_id"]) for row in moving_rows}
    cache_path = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"
    cache = _read_npz(cache_path)
    if len(stage_c_rows) != int(cache["labels"].shape[0]):
        raise ValueError("Stage-C Val rows and frozen evidence cache have different lengths")
    labels = np.asarray([int(row["label_id"]) for row in stage_c_rows], dtype=np.int64)
    if not np.array_equal(labels, cache["labels"]):
        raise ValueError("Stage-C labels do not align with the frozen evidence cache")
    moving_indices = np.asarray(
        [index for index, row in enumerate(stage_c_rows) if str(row["episode_id"]) in moving_ids],
        dtype=np.int64,
    )
    if moving_indices.size != len(moving_rows):
        raise ValueError("Moving Stage-D IDs do not map one-to-one to Stage-C Val rows")
    return {
        "stage_c_rows": stage_c_rows,
        "moving_rows": moving_rows,
        "moving_indices": moving_indices,
        "cache": cache,
        "cache_path": cache_path,
        "archive_root": data_root / ARCHIVE_RELATIVE,
    }


def _candidate_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, int], list[int]]:
    index: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        for candidate_id in row["candidate_viewpoint_ids"]:
            key = (str(row["scene_id"]), str(row["region"]), int(candidate_id), int(row["label_id"]))
            index[key].append(row_index)
    return index


def _current_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int, int], list[int]]:
    index: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        key = (
            str(row["scene_id"]), str(row["region"]),
            int(row["current_viewpoint_id"]), int(row["label_id"]),
        )
        index[key].append(row_index)
    return index


def _manifest_audit(rows: Sequence[Mapping[str, Any]], archive_root: Path) -> dict[str, Any]:
    """Check canonical placement/camera metadata without opening skeleton arrays."""
    pairs = {(str(row["scene_id"]), str(row["region"])) for row in rows}
    checked = 0
    missing = []
    invalid = []
    for scene_id, placement_id in sorted(pairs):
        path = archive_root / scene_id / "candidate_metadata/manifest.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        placements = [item for item in payload.get("placements_data", []) if str(item.get("placement_id")) == placement_id]
        if len(placements) != 1:
            invalid.append({"scene_id": scene_id, "placement_id": placement_id, "reason": "placement missing or duplicated"})
            continue
        viewpoints = {int(item["viewpoint_id"]): item for item in placements[0].get("viewpoints", [])}
        if len(viewpoints) != 32:
            invalid.append({"scene_id": scene_id, "placement_id": placement_id, "reason": "expected 32 canonical viewpoints"})
            continue
        checked += 1
    return {
        "scene_placement_pairs": len(pairs),
        "canonical_manifests_checked": checked,
        "missing_manifests": missing,
        "invalid_manifests": invalid,
        "placement_position_yaw_camera_source": "single furniture-placement-v2 candidate manifest per scene/placement",
    }


def _pairing_audit(inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str, int, int], list[int]], dict[tuple[str, str, int, int], list[int]]]:
    rows = inputs["stage_c_rows"]
    moving_indices = set(int(value) for value in inputs["moving_indices"])
    candidate_rows = _candidate_index(rows)
    current_rows = _current_index(rows)
    candidate_coverage: list[int] = []
    stay_coverage: list[int] = []
    for row_index in sorted(moving_indices):
        row = rows[row_index]
        stay_key = (str(row["scene_id"]), str(row["region"]), int(row["current_viewpoint_id"]))
        stay_coverage.append(sum(bool(current_rows.get((*stay_key, class_id))) for class_id in range(NUM_CLASSES)))
        for candidate_id in row["candidate_viewpoint_ids"]:
            key = (str(row["scene_id"]), str(row["region"]), int(candidate_id))
            candidate_coverage.append(sum(bool(candidate_rows.get((*key, class_id))) for class_id in range(NUM_CLASSES)))
    legal_sets_by_action: dict[tuple[str, str], dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row_index in sorted(moving_indices):
        row = rows[row_index]
        legal_sets_by_action[(str(row["scene_id"]), str(row["region"]))][int(row["label_id"])] = set(int(value) for value in row["candidate_viewpoint_ids"])
    legal_mismatches = []
    for key, by_action in legal_sets_by_action.items():
        if len(by_action) < NUM_CLASSES:
            continue
        sets = list(by_action.values())
        if any(value != sets[0] for value in sets[1:]):
            legal_mismatches.append({
                "scene_id": key[0], "placement_id": key[1],
                "action_sets": {str(action): sorted(values) for action, values in by_action.items()},
            })
    manifest = _manifest_audit(rows, inputs["archive_root"])
    stay_full = sum(value == NUM_CLASSES for value in stay_coverage)
    candidate_full = sum(value == NUM_CLASSES for value in candidate_coverage)
    # Strict disambiguation requires a counterfactual at the same current
    # viewpoint for stay and an identical legal set across action hypotheses.
    passed = (
        not manifest["missing_manifests"]
        and not manifest["invalid_manifests"]
        and stay_full == len(stay_coverage)
        and candidate_full == len(candidate_coverage)
        and not legal_mismatches
    )
    audit = {
        "status": "PASSED" if passed else "STOPPED",
        "reason": None if passed else "Strict same-camera counterfactual pairing is incomplete; no selector was evaluated.",
        "stage_c_val_contexts": len(rows),
        "moving_val_contexts": len(moving_indices),
        "moving_stay_counterfactual_full_12": stay_full,
        "moving_stay_counterfactual_coverage": float(stay_full / max(len(stay_coverage), 1)),
        "moving_candidate_pairs": len(candidate_coverage),
        "moving_candidate_counterfactual_full_12": candidate_full,
        "moving_candidate_counterfactual_coverage": float(candidate_full / max(len(candidate_coverage), 1)),
        "candidate_hypothesis_coverage_counts": dict(Counter(candidate_coverage)),
        "stay_hypothesis_coverage_counts": dict(Counter(stay_coverage)),
        "legal_set_mismatch_scene_placements": len(legal_mismatches),
        "legal_set_mismatch_examples": legal_mismatches[:10],
        "manifest": manifest,
        "same_scene_placement_camera_protocol": "candidate metadata manifest is canonical; no nearest-neighbor pairing used",
    }
    return audit, candidate_rows, current_rows


def _hypothesis_coverage(inputs: Mapping[str, Any]) -> dict[str, Any]:
    cache = inputs["cache"]
    moving = inputs["moving_indices"]
    labels = np.asarray(cache["labels"][moving], dtype=np.int64)
    logp = np.asarray(cache["current_logp"][moving], dtype=np.float64)
    order = np.argsort(-logp, axis=1)
    return {
        "contexts": int(labels.size),
        "p_gt_in_top1": float(np.mean(order[:, :1] == labels[:, None])),
        "p_gt_in_top2": float(np.mean(np.any(order[:, :2] == labels[:, None], axis=1))),
        "p_gt_in_top3": float(np.mean(np.any(order[:, :3] == labels[:, None], axis=1))),
        "p_gt_in_top5": float(np.mean(np.any(order[:, :5] == labels[:, None], axis=1))),
        "definition": "coverage of the frozen s0 posterior; no predicted class is supplied to a selector",
    }


def _stop_outputs(audit: Mapping[str, Any], coverage: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    flags = {
        "test_used": False,
        "training_used": False,
        "gt_action_used_for_selector": False,
        "predicted_action_used": False,
        "current_action_belief_used": True,
        "multiple_action_hypotheses_retained": False,
        "counterfactual_future_skeleton_used_for_privileged_oracle_only": True,
        "counterfactual_future_stgcn_evidence_used_for_privileged_oracle_only": True,
        "actual_future_candidate_skeleton_used_for_terminal_evaluation_only": True,
        "future_rgb_used": False,
        "future_dino_used": False,
        "deployable": False,
    }
    result = {
        "experiment_id": "REDUCED12_COUNTERFACTUAL_DISAMBIGUATION_ORACLE",
        "status": "STOPPED_PAIRING_INTEGRITY",
        "population": {"stage_c_val": len(inputs["stage_c_rows"]), "moving_val": len(inputs["moving_indices"])},
        "labels": list(LABELS),
        "pairing_audit": audit,
        "hypothesis_coverage": coverage,
        "metrics_moving": {},
        "selector_metrics_available": False,
        "protocol_flags": flags,
        "artifacts_read": {"stage_c_features": str((inputs["cache_path"].parent).resolve()), "counterfactual_source": "existing frozen true_logp cache only"},
    }
    (EXPERIMENT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "pairing_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "hypothesis_coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "diagnostics.json").write_text(json.dumps({"status": "NOT_RUN", "reason": audit["reason"], "test_used": False}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "separability_metrics.json").write_text(json.dumps({"status": "NOT_RUN", "reason": audit["reason"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "regret_metrics.json").write_text(json.dumps({"status": "NOT_RUN", "reason": audit["reason"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "per_class_metrics.json").write_text(json.dumps({"status": "NOT_RUN", "reason": audit["reason"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    analysis = [
        "# Action-Agnostic Counterfactual Disambiguation Oracle",
        "",
        "## Status: STOPPED_PAIRING_INTEGRITY",
        "",
        "The main selector was not evaluated. The protocol requires a same-scene, same-placement, same-camera counterfactual for all 12 action hypotheses, including the stay/current viewpoint. Existing Val archives do not provide that strict pairing.",
        "",
        f"- Moving Val contexts: {len(inputs['moving_indices'])}",
        f"- Stay same-camera 12-action coverage: {audit['moving_stay_counterfactual_full_12']}/{len(inputs['moving_indices'])} ({audit['moving_stay_counterfactual_coverage']:.3%})",
        f"- Candidate same-camera 12-action coverage: {audit['moving_candidate_counterfactual_full_12']}/{audit['moving_candidate_pairs']} ({audit['moving_candidate_counterfactual_coverage']:.3%})",
        f"- Scene/placement groups with action-dependent legal-set mismatch: {audit['legal_set_mismatch_scene_placements']}",
        "",
        "Because a current-viewpoint counterfactual is missing for most contexts, substituting another initial viewpoint or using nearest-neighbor records would violate the requested pairing contract. Therefore no JSD, feature-separability, terminal selector, or regret metric was produced.",
        "",
        "The frozen s0 hypothesis-support audit is still valid: see hypothesis_coverage.json. It is not a selector result.",
        "",
        "test_used=false; training_used=false; gt_action_used_for_selector=false; predicted_action_used=false; deployable=false.",
    ]
    (EXPERIMENT / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    args = parser.parse_args()
    random.seed(SEED)
    inputs = _load_inputs(args.data_root.resolve())
    audit, _, _ = _pairing_audit(inputs)
    coverage = _hypothesis_coverage(inputs)
    if audit["status"] != "PASSED":
        result = _stop_outputs(audit, coverage, inputs)
        print(json.dumps({"status": result["status"], "output": str((EXPERIMENT / "result.json").resolve()), "test_used": False}, ensure_ascii=False))
        return
    raise RuntimeError("Pairing passed but selector implementation is intentionally disabled in this first safety audit")


if __name__ == "__main__":
    main()
