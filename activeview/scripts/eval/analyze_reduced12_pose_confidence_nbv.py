#!/usr/bin/env python3
"""Val-only audit of viewpoint-level pose confidence as a privileged NBV cue.

The diagnostic selects among the already archived legal candidates using the
stored scalar confidence.  Candidate terminal labels come from the existing
frozen reduced12 ST-GCN true-logp cache; no model is trained and no perception
artifact is regenerated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl

SEED = 42
NUM_CLASSES = 12
VIEW_COUNT = 32
SKELETON_SHAPE = (32, 3, 30, 17)
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/pose_confidence_nbv_oracle"
)


def _cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _archive_path(archive_root: Path, row: Mapping[str, Any]) -> Path:
    scene = str(row["scene_id"])
    placement = str(row.get("placement_id") or row.get("region"))
    record = str(row["record_id"])
    path = archive_root / scene / placement / f"{record}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing archived skeleton: {path}")
    return path


def _metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if predictions.shape != labels.shape or predictions.ndim != 1:
        raise ValueError("predictions and labels must be aligned one-dimensional arrays")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels.tolist(), predictions.tolist()):
        if not 0 <= int(target) < NUM_CLASSES or not 0 <= int(prediction) < NUM_CLASSES:
            raise ValueError("label outside reduced12 range")
        matrix[int(target), int(prediction)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(LABELS):
        tp = float(matrix[class_id, class_id])
        support = float(matrix[class_id].sum())
        predicted = float(matrix[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "label_id": class_id,
            "count": int(support),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    count = int(labels.size)
    return {
        "count": count,
        "accuracy": float(np.mean(predictions == labels)) if count else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling for Spearman."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_rank = _rank(left)
    right_rank = _rank(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _load_candidate_confidences(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    archive_root: Path,
) -> tuple[list[np.ndarray], int]:
    """Read one existing archive per context and return legal-view confidence."""
    output: list[np.ndarray] = []
    archive_count = 0
    for index, row in enumerate(rows):
        path = _archive_path(archive_root, row)
        with np.load(path, allow_pickle=False) as archive:
            skeleton = np.asarray(archive["skeleton"])
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
        if skeleton.shape != SKELETON_SHAPE:
            raise ValueError(f"unexpected skeleton shape in {path}: {skeleton.shape}")
        if viewpoint_ids.shape != (VIEW_COUNT,) or confidence.shape != (VIEW_COUNT,):
            raise ValueError(f"invalid confidence schema in {path}")
        if not np.isfinite(confidence).all():
            raise ValueError(f"non-finite confidence in {path}")
        valid = np.flatnonzero(np.asarray(cache["candidate_mask"][index], dtype=bool))
        candidate_ids = np.asarray(cache["candidate_ids"][index], dtype=np.int64)[valid]
        values: list[float] = []
        for viewpoint_id in candidate_ids.tolist():
            matches = np.flatnonzero(viewpoint_ids == int(viewpoint_id))
            if matches.size != 1:
                raise ValueError(f"candidate viewpoint alignment failure in {path}: {viewpoint_id}")
            values.append(float(confidence[int(matches[0])]))
        output.append(np.asarray(values, dtype=np.float32))
        archive_count += 1
    return output, archive_count


def _select_predictions(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    candidate_confidences: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    current_logp = np.asarray(cache["current_logp"], dtype=np.float64)
    candidate_logp = np.asarray(cache["candidate_logp"], dtype=np.float64)
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    random_generator = np.random.default_rng(SEED)
    predictions = {name: np.empty(labels.size, dtype=np.int64) for name in ("S0-only", "Random", "MaxPoseConfidence", "AnyCorrect Oracle")}
    for index, label in enumerate(labels.tolist()):
        valid = np.flatnonzero(mask[index])
        if valid.size != candidate_confidences[index].size:
            raise ValueError("confidence and legal-candidate counts differ")
        predictions["S0-only"][index] = int(np.argmax(current_logp[index]))
        choice = int(random_generator.integers(0, valid.size + 1)) if valid.size else 0
        predictions["Random"][index] = int(np.argmax(current_logp[index])) if choice == 0 else int(np.argmax(candidate_logp[index, valid[choice - 1]]))
        if not valid.size:
            predictions["MaxPoseConfidence"][index] = int(np.argmax(current_logp[index]))
        else:
            best = int(valid[int(np.argmax(candidate_confidences[index]))])
            predictions["MaxPoseConfidence"][index] = int(np.argmax(candidate_logp[index, best]))
        if int(np.argmax(current_logp[index])) == int(label):
            predictions["AnyCorrect Oracle"][index] = int(label)
        else:
            correct = [candidate for candidate in valid if int(np.argmax(candidate_logp[index, candidate])) == int(label)]
            predictions["AnyCorrect Oracle"][index] = int(label) if correct else int(np.argmax(current_logp[index]))
    return predictions


def _confidence_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    confidences: Sequence[np.ndarray],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    candidate_logp = np.asarray(cache["candidate_logp"], dtype=np.float64)
    mask = np.asarray(cache["candidate_mask"], dtype=bool)
    correlations: list[float] = []
    correct_confidence: list[float] = []
    wrong_confidence: list[float] = []
    for index, label in enumerate(labels.tolist()):
        valid = np.flatnonzero(mask[index])
        conf = np.asarray(confidences[index], dtype=np.float64)
        true_scores = candidate_logp[index, valid, int(label)]
        if conf.size >= 2:
            correlations.append(_spearman(conf, true_scores))
        candidate_correct = np.argmax(candidate_logp[index, valid], axis=1) == int(label)
        correct_confidence.extend(conf[candidate_correct].tolist())
        wrong_confidence.extend(conf[~candidate_correct].tolist())
    return {
        "per_context_spearman": {
            "count": len(correlations),
            "mean": float(np.mean(correlations)) if correlations else 0.0,
            "median": float(np.median(correlations)) if correlations else 0.0,
        },
        "candidate_confidence_by_true_stgcn_correctness": {
            "correct_count": len(correct_confidence),
            "wrong_count": len(wrong_confidence),
            "mean_correct": float(np.mean(correct_confidence)) if correct_confidence else 0.0,
            "mean_wrong": float(np.mean(wrong_confidence)) if wrong_confidence else 0.0,
            "median_correct": float(np.median(correct_confidence)) if correct_confidence else 0.0,
            "median_wrong": float(np.median(wrong_confidence)) if wrong_confidence else 0.0,
            "mean_difference": (float(np.mean(correct_confidence) - np.mean(wrong_confidence)) if correct_confidence and wrong_confidence else 0.0),
        },
    }


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]
    frozen = methods["FrozenStageCv0"]
    max_conf = methods["MaxPoseConfidence"]
    gain_acc = 100.0 * (float(max_conf["accuracy"]) - float(frozen["accuracy"]))
    gain_f1 = 100.0 * (float(max_conf["macro_f1"]) - float(frozen["macro_f1"]))
    rho = result["confidence_diagnostics"]["per_context_spearman"]["mean"]
    rows = [
        "# Reduced12 Pose Observation Confidence NBV Oracle",
        "",
        "Val Moving contexts only. Pose Observation Confidence is a viewpoint-level scalar (the archived mean YOLO keypoint confidence over the 30-frame sequence). It is used only as a privileged future-candidate diagnostic; no model was trained and no Test artifact was read.",
        "",
        "## Metrics",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔF1 vs Frozen |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("S0-only", "FrozenStageCv0", "Random", "MaxPoseConfidence", "AnyCorrect Oracle"):
        metric = methods[name]
        rows.append(
            f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | "
            f"{100.0 * (float(metric['accuracy']) - float(frozen['accuracy'])):+.3f} | "
            f"{100.0 * (float(metric['macro_f1']) - float(frozen['macro_f1'])):+.3f} |"
        )
    rows.extend([
        "",
        "## Confidence audit",
        "",
        f"- Per-context Spearman(confidence, GT-class true logp): mean **{rho:.6f}**, median **{result['confidence_diagnostics']['per_context_spearman']['median']:.6f}**.",
        f"- Mean confidence for ST-GCN-correct candidates: **{result['confidence_diagnostics']['candidate_confidence_by_true_stgcn_correctness']['mean_correct']:.6f}**.",
        f"- Mean confidence for ST-GCN-wrong candidates: **{result['confidence_diagnostics']['candidate_confidence_by_true_stgcn_correctness']['mean_wrong']:.6f}**.",
        f"- Correct-minus-wrong mean confidence: **{result['confidence_diagnostics']['candidate_confidence_by_true_stgcn_correctness']['mean_difference']:+.6f}**.",
        "",
        "## Focus classes (MaxPoseConfidence Recall/F1)",
        "",
        "| bend | stumble | knock | touching face |",
        "|---:|---:|---:|---:|",
        f"| {max_conf['per_class']['bend']['recall']:.3f}/{max_conf['per_class']['bend']['f1']:.3f} | {max_conf['per_class']['stumble']['recall']:.3f}/{max_conf['per_class']['stumble']['f1']:.3f} | {max_conf['per_class']['knock']['recall']:.3f}/{max_conf['per_class']['knock']['f1']:.3f} | {max_conf['per_class']['touching face']['recall']:.3f}/{max_conf['per_class']['touching face']['f1']:.3f} |",
        "",
        "## Scientific interpretation",
        "",
        f"MaxPoseConfidence changes Moving Accuracy by **{gain_acc:+.3f} pp** and Macro-F1 by **{gain_f1:+.3f} pp** relative to FrozenStageCv0.",
        ("The confidence cue shows a positive candidate-quality association in this audit; the next authorized direction would be to predict future confidence from current observation, scene model and candidate geometry." if rho > 0.2 and result['confidence_diagnostics']['candidate_confidence_by_true_stgcn_correctness']['mean_difference'] > 0 else "The confidence cue does not show a strong consistent positive association with frozen ST-GCN candidate quality; no confidence predictor is justified by this audit alone."),
        "MaxPoseConfidence is privileged and deployable=false because it reads future candidate confidence. It is not connected to the policy.",
        "",
        "test_used=false; future_candidate_confidence_used_for_target_or_oracle_only=true; no RGB/skeleton/DINO data was regenerated.",
    ])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(data_root: Path, output_dir: Path, archive_root: Path) -> dict[str, Any]:
    rows_path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_d/features/val.jsonl"
    stage_c_path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/features/val.jsonl"
    cache_path = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"
    moving_rows = load_jsonl(rows_path)
    stage_c_rows = load_jsonl(stage_c_path)
    by_id = {str(row["episode_id"]): row for row in stage_c_rows}
    rows = [by_id[str(row["episode_id"])] for row in moving_rows]
    cache_all = _cache(cache_path)
    if len(cache_all["labels"]) != len(stage_c_rows):
        raise ValueError("Stage-C Val rows and frozen true-logp cache lengths differ")
    stage_c_index = {str(row["episode_id"]): index for index, row in enumerate(stage_c_rows)}
    indices = [stage_c_index[str(row["episode_id"])] for row in rows]
    cache = {key: value[np.asarray(indices)] for key, value in cache_all.items()}
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    if not np.array_equal(cache["labels"], labels):
        raise ValueError("Val labels and frozen true-logp cache are not aligned")
    confidences, archive_count = _load_candidate_confidences(rows, cache, archive_root)
    predictions = _select_predictions(rows, cache, confidences)
    previous = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").read_text(encoding="utf-8"))
    methods: dict[str, Any] = {
        "S0-only": _metrics(predictions["S0-only"], labels),
        "FrozenStageCv0": previous["metrics_moving"]["FrozenStageCv0"],
        "Random": _metrics(predictions["Random"], labels),
        "MaxPoseConfidence": _metrics(predictions["MaxPoseConfidence"], labels),
        "AnyCorrect Oracle": _metrics(predictions["AnyCorrect Oracle"], labels),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_POSE_CONFIDENCE_NBV_ORACLE",
        "status": "COMPLETED",
        "seed": SEED,
        "test_used": False,
        "deployable": False,
        "population": {"split": "val", "moving_contexts": len(rows), "archives_read": archive_count, "legal_candidate_samples": int(sum(len(value) for value in confidences))},
        "methods": methods,
        "confidence_diagnostics": _confidence_diagnostics(rows, cache, confidences),
        "archive_schema": {"confidence_shape": [32], "confidence_unit": "viewpoint-level scalar mean YOLO keypoint confidence over 30 frames", "skeleton_shape": list(SKELETON_SHAPE)},
        "protocol": {"candidate_set": "same legal candidate_mask/candidate_ids as existing reduced12 cache", "terminal_prediction": "existing frozen reduced12 ST-GCN true_logp for archived candidate skeleton", "max_pose_confidence": "argmax archived candidate confidence over legal candidates", "random_seed": SEED},
        "leakage_flags": {"test_used": False, "test_paths_read": False, "future_candidate_confidence_used_for_target_or_oracle_only": True, "deployable": False, "training_performed": False, "rgb_regenerated": False, "skeleton_regenerated": False, "dino_regenerated": False},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result["per_class_metrics"] = {label: {name: metric["per_class"][label] for name, metric in methods.items() if "per_class" in metric} for label in LABELS}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "per_class_metrics.json").write_text(json.dumps(result["per_class_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    archive_root = (args.archive_root or (data_root / "datasets/offline/habitat-train/00006-00087")).resolve()
    result = run(data_root, args.output_dir.resolve(), archive_root)
    print(json.dumps({"output": str(args.output_dir.resolve()), "moving_contexts": result["population"]["moving_contexts"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
