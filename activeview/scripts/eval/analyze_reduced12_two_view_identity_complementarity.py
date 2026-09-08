#!/usr/bin/env python3
"""Val-only audit of two-view skeleton identity complementarity.

The archived reduced12 skeletons are already canonicalized by the frozen
perception pipeline.  This diagnostic therefore fuses the stored skeletons
directly and never applies a second camera rotation.  Archive confidence is a
single scalar per viewpoint, so the confidence baseline selects an entire
skeleton; it does not manufacture joint-level confidence values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.recognition.stgcn.model import load_checkpoint


SEED = 42
NUM_CLASSES = 12
SKELETON_SHAPE = (3, 30, 17)
BATCH_SIZE = 256
OUTPUT_DIR = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "two_view_identity_complementarity"
)
DEFAULT_ARCHIVE_ROOT_NAME = "datasets/offline/habitat-train/00006-00087"
LABEL_NAMES = (
    "walk",
    "sit",
    "stand up",
    "bend",
    "crawl",
    "stumble",
    "clap",
    "throw",
    "kick",
    "knock",
    "punch",
    "touching face",
)


def _seed_everything() -> None:
    """Set deterministic seeds for the frozen inference audit."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Return accuracy, macro-F1, confusion and per-class recall/F1."""
    if predictions.shape != labels.shape or predictions.ndim != 1:
        raise ValueError("predictions and labels must be one-dimensional and aligned")
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, prediction in zip(labels.tolist(), predictions.tolist()):
        if not 0 <= int(target) < NUM_CLASSES or not 0 <= int(prediction) < NUM_CLASSES:
            raise ValueError("label outside reduced12 range")
        matrix[int(target), int(prediction)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(LABEL_NAMES):
        true_positive = float(matrix[class_id, class_id])
        support = float(matrix[class_id].sum())
        predicted = float(matrix[:, class_id].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[name] = {
            "label_id": class_id,
            "count": int(support),
            "recall": recall,
            "f1": f1,
            "precision": precision,
        }
    total = int(matrix.sum())
    return {
        "count": total,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _archive_path(archive_root: Path, row: Mapping[str, Any]) -> Path:
    """Resolve an existing archive without scanning undeclared disks."""
    scene_id = str(row["scene_id"])
    placement_id = str(row.get("placement_id") or row.get("region"))
    record_id = str(row["record_id"])
    if not scene_id or not placement_id or not record_id:
        raise ValueError("Val row is missing scene, placement or record identity")
    path = archive_root / scene_id / placement_id / f"{record_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing archived skeleton: {path}")
    return path


def _load_two_view_observations(
    rows: Sequence[Mapping[str, Any]],
    archive_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load s0/s1 skeletons and viewpoint-level confidence from Val archives."""
    s0_values: list[np.ndarray] = []
    s1_values: list[np.ndarray] = []
    c0_values: list[float] = []
    c1_values: list[float] = []
    for row in rows:
        path = _archive_path(archive_root, row)
        with np.load(path, allow_pickle=False) as archive:
            skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            confidence = np.asarray(archive["confidence"], dtype=np.float32)
        if skeletons.shape != (32, *SKELETON_SHAPE):
            raise ValueError(f"unexpected skeleton shape in {path}: {skeletons.shape}")
        if viewpoint_ids.shape != (32,) or confidence.shape != (32,):
            raise ValueError(f"archive schema is not the frozen viewpoint-level format: {path}")
        if not np.isfinite(skeletons).all() or not np.isfinite(confidence).all():
            raise ValueError(f"non-finite skeleton or confidence in {path}")
        indices0 = np.flatnonzero(viewpoint_ids == int(row["s0_viewpoint_id"]))
        indices1 = np.flatnonzero(viewpoint_ids == int(row["s1_viewpoint_id"]))
        if indices0.size != 1 or indices1.size != 1:
            raise ValueError(f"viewpoint id alignment failure in {path}")
        index0, index1 = int(indices0[0]), int(indices1[0])
        s0_values.append(skeletons[index0])
        s1_values.append(skeletons[index1])
        c0_values.append(float(confidence[index0]))
        c1_values.append(float(confidence[index1]))
    if not s0_values:
        raise ValueError("Val moving contexts are empty")
    return (
        np.stack(s0_values).astype(np.float32),
        np.stack(s1_values).astype(np.float32),
        np.asarray(c0_values, dtype=np.float32),
        np.asarray(c1_values, dtype=np.float32),
    )


def _stgcn_predictions(
    model: torch.nn.Module,
    skeletons: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run frozen ST-GCN argmax inference on [N,3,30,17] skeletons."""
    if skeletons.ndim != 4 or skeletons.shape[1:] != SKELETON_SHAPE:
        raise ValueError(f"unexpected inference skeleton shape: {skeletons.shape}")
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, skeletons.shape[0], batch_size):
            batch = torch.from_numpy(skeletons[start : start + batch_size]).to(
                device=device,
                dtype=torch.float32,
            )
            logits = model(batch.unsqueeze(-1))
            predictions.append(logits.argmax(dim=-1).cpu().numpy().astype(np.int64))
    return np.concatenate(predictions, axis=0)


def _per_class_any_correct(
    any_correct: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize Best-of-two AnyCorrect by ground-truth class."""
    output: dict[str, dict[str, float | int]] = {}
    for class_id, name in enumerate(LABEL_NAMES):
        mask = labels == class_id
        count = int(mask.sum())
        output[name] = {
            "label_id": class_id,
            "count": count,
            "any_correct_rate": float(any_correct[mask].mean()) if count else 0.0,
        }
    return output


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    """Write a concise evidence-first Val interpretation."""
    methods = result["methods"]
    rows = [
        "# Reduced12 two-view identity complementarity",
        "",
        "Val moving contexts only. No model was trained, no Test artifact was read, and no perception data was regenerated.",
        "",
        "## Classification comparison",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAccuracy vs S1 | ΔMacro-F1 vs S1 |",
        "|---|---:|---:|---:|---:|",
    ]
    s1_accuracy = float(methods["S1-only"]["accuracy"])
    s1_f1 = float(methods["S1-only"]["macro_f1"])
    for name in ("S0-only", "S1-only", "SimpleMeanSkeleton", "ScalarConfidenceSelect"):
        value = methods[name]
        rows.append(
            f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} | "
            f"{100.0 * (float(value['accuracy']) - s1_accuracy):+.3f} | "
            f"{100.0 * (float(value['macro_f1']) - s1_f1):+.3f} |"
        )
    rows.extend(
        [
            "",
            "Prior identity-estimator references (from the existing Val audit):",
            "",
            "| Reference | Accuracy | Macro-F1 |",
            "|---|---:|---:|",
            f"| ST-GCN feature-only MLP | {result['reference_identity_estimators']['ST-GCN feature-only MLP']['accuracy']:.6f} | {result['reference_identity_estimators']['ST-GCN feature-only MLP']['macro_f1']:.6f} |",
            f"| Current all-feature MLP | {result['reference_identity_estimators']['Current all-feature MLP']['accuracy']:.6f} | {result['reference_identity_estimators']['Current all-feature MLP']['macro_f1']:.6f} |",
        ]
    )
    oracle = result["best_of_two_oracle"]
    rows.extend(
        [
            "",
            "## Complementarity audit",
            "",
            f"- S0 correct rate: {result['complementarity']['s0_correct_rate']:.6f}",
            f"- S1 correct rate: {result['complementarity']['s1_correct_rate']:.6f}",
            f"- Both correct: {result['complementarity']['both_correct_rate']:.6f}",
            f"- Only S0 correct: {result['complementarity']['only_s0_correct_rate']:.6f}",
            f"- Only S1 correct: {result['complementarity']['only_s1_correct_rate']:.6f}",
            f"- Neither correct: {result['complementarity']['neither_correct_rate']:.6f}",
            f"- Best-of-two AnyCorrect: {oracle['any_correct_rate']:.6f}",
            f"- Complementarity gain over max(S0, S1): {result['complementarity']['gain_pp']:+.3f} pp",
            f"- Mean number of correct views: {result['complementarity']['mean_correct_views']:.6f}",
            "",
            "The archive stores one confidence scalar per viewpoint, not per joint. Therefore the requested fraction of high-confidence joints coming from different views is not identifiable and is reported as null; ScalarConfidenceSelect is a whole-skeleton view-level baseline.",
            f"View-level confidence winner differs between s0/s1 in {result['confidence_audit']['winner_diff_rate']:.6f} of contexts.",
            "",
            "## Focus actions (Recall/F1)",
            "",
            "| Method | bend | stumble | knock | touching face |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("S0-only", "S1-only", "SimpleMeanSkeleton", "ScalarConfidenceSelect"):
        focus = methods[name]["per_class"]
        rows.append(
            f"| {name} | {focus['bend']['recall']:.3f}/{focus['bend']['f1']:.3f} | "
            f"{focus['stumble']['recall']:.3f}/{focus['stumble']['f1']:.3f} | "
            f"{focus['knock']['recall']:.3f}/{focus['knock']['f1']:.3f} | "
            f"{focus['touching face']['recall']:.3f}/{focus['touching face']['f1']:.3f} |"
        )
    references = result["reference_identity_estimators"]
    all_feature = references["Current all-feature MLP"]
    if float(oracle["any_correct_rate"]) > float(all_feature["accuracy"]):
        oracle_sentence = "Best-of-two AnyCorrect is above the prior all-feature MLP accuracy, indicating recoverable information may remain in the two views."
    else:
        oracle_sentence = "Best-of-two AnyCorrect does not exceed the prior all-feature MLP accuracy, so the two archived views alone do not establish a large unused identity ceiling."
    if max(float(methods["SimpleMeanSkeleton"]["macro_f1"]), float(methods["ScalarConfidenceSelect"]["macro_f1"])) > s1_f1:
        fusion_sentence = "At least one direct skeleton fusion baseline improves over S1-only Macro-F1; this supports testing a learned multi-view pose/history encoder."
    else:
        fusion_sentence = "Simple direct skeleton fusion does not improve S1-only Macro-F1; a learned multi-view encoder would be needed before attributing gains to joint complementarity."
    rows.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            oracle_sentence,
            fusion_sentence,
            "All comparisons use the same frozen reduced12 ST-GCN and the same Val moving contexts. The fixed all-feature MLP and ST-GCN feature-only references are prior Val results, not newly evaluated with this script.",
            "",
            "## Protocol",
            "",
            "- Skeleton convention: existing `camera_to_gravity + root_center + torso_scale + yaw_only`; no second camera rotation.",
            "- Confidence convention: archive `(32,)` viewpoint-level scalar; no per-joint confidence was fabricated.",
            "- `test_used=false`; no WM/JR/ST-GCN artifact was modified.",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(
    data_root: Path,
    checkpoint: Path,
    archive_root: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    """Execute the frozen Val-only two-view audit."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    _seed_everything()
    rows_path = data_root / (
        "datasets/policy_reduced12_eight_placement_v1/"
        "stage_d/features/val.jsonl"
    )
    rows = load_jsonl(rows_path)
    if not rows:
        raise ValueError(f"empty Val moving feature file: {rows_path}")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in rows):
        raise ValueError("non-Val row found in Val moving feature file")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    s0, s1, confidence0, confidence1 = _load_two_view_observations(rows, archive_root)
    model, device = load_checkpoint(checkpoint, NUM_CLASSES, device_name)
    if device_name.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("frozen ST-GCN did not load on CUDA")
    simple_mean = (0.5 * (s0 + s1)).astype(np.float32)
    choose_s0 = confidence0 >= confidence1
    confidence_select = np.where(choose_s0[:, None, None, None], s0, s1).astype(np.float32)
    stacked = np.concatenate([s0, s1, simple_mean, confidence_select], axis=0)
    stacked_predictions = _stgcn_predictions(model, stacked, device, batch_size)
    count = len(rows)
    prediction_arrays = {
        "S0-only": stacked_predictions[:count],
        "S1-only": stacked_predictions[count : 2 * count],
        "SimpleMeanSkeleton": stacked_predictions[2 * count : 3 * count],
        "ScalarConfidenceSelect": stacked_predictions[3 * count : 4 * count],
    }
    methods = {
        name: _classification_metrics(predictions, labels)
        for name, predictions in prediction_arrays.items()
    }
    correct0 = prediction_arrays["S0-only"] == labels
    correct1 = prediction_arrays["S1-only"] == labels
    any_correct = correct0 | correct1
    both_correct = correct0 & correct1
    only_s0 = correct0 & ~correct1
    only_s1 = ~correct0 & correct1
    none_correct = ~correct0 & ~correct1
    s0_rate = float(correct0.mean())
    s1_rate = float(correct1.mean())
    any_rate = float(any_correct.mean())
    complementarity = {
        "s0_correct_rate": s0_rate,
        "s1_correct_rate": s1_rate,
        "both_correct_rate": float(both_correct.mean()),
        "only_s0_correct_rate": float(only_s0.mean()),
        "only_s1_correct_rate": float(only_s1.mean()),
        "neither_correct_rate": float(none_correct.mean()),
        "mean_correct_views": float((correct0.astype(np.int64) + correct1).mean()),
        "gain_pp": 100.0 * (any_rate - max(s0_rate, s1_rate)),
    }
    best_oracle = {
        "count": count,
        "any_correct_rate": any_rate,
        "per_class_any_correct": _per_class_any_correct(any_correct, labels),
    }
    confidence_audit = {
        "archive_confidence_shape": [32],
        "confidence_unit": "viewpoint-level scalar",
        "mean_s0_confidence": float(confidence0.mean()),
        "median_s0_confidence": float(np.median(confidence0)),
        "mean_s1_confidence": float(confidence1.mean()),
        "median_s1_confidence": float(np.median(confidence1)),
        "winner_diff_rate": float(np.mean(~np.isclose(confidence0, confidence1))),
        "high_confidence_joint_from_different_views": None,
        "joint_level_confidence_available": False,
    }
    relative_to_s1 = {
        name: {
            "accuracy_pp": 100.0 * (float(value["accuracy"]) - s1_rate),
            "macro_f1_pp": 100.0 * (float(value["macro_f1"]) - float(methods["S1-only"]["macro_f1"])),
        }
        for name, value in methods.items()
    }
    reference_estimators = {
        "S1-only (prior identity fusion audit)": {
            "accuracy": 0.454266,
            "macro_f1": 0.444782,
        },
        "ST-GCN feature-only MLP": {
            "accuracy": 0.531052,
            "macro_f1": 0.555232,
        },
        "Current all-feature MLP": {
            "accuracy": 0.547619,
            "macro_f1": 0.566860,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_TWO_VIEW_IDENTITY_COMPLEMENTARITY",
        "status": "COMPLETED",
        "seed": SEED,
        "test_used": False,
        "training_performed": False,
        "perception_regenerated": False,
        "population": {
            "split": "val",
            "moving_contexts": count,
            "unique_archived_observations": count * 2,
            "archive_root": str(archive_root.resolve()),
        },
        "label_names": list(LABEL_NAMES),
        "methods": methods,
        "method_order": list(methods),
        "relative_to_s1_pp": relative_to_s1,
        "best_of_two_oracle": best_oracle,
        "complementarity": complementarity,
        "confidence_audit": confidence_audit,
        "reference_identity_estimators": reference_estimators,
        "protocol": {
            "stgcn_checkpoint": str(checkpoint.resolve()),
            "skeleton_shape": list(SKELETON_SHAPE),
            "skeleton_coordinate_convention": "camera_to_gravity + root_center + torso_scale + yaw_only",
            "additional_camera_rotation": False,
            "confidence_selection": "select whole skeleton with larger viewpoint-level scalar confidence",
        },
        "leakage_flags": {
            "test_used": False,
            "test_paths_read": False,
            "wm_modified": False,
            "jr_modified": False,
            "stgcn_modified": False,
            "perception_regenerated": False,
            "per_joint_confidence_fabricated": False,
        },
        "artifacts": {
            "result": str((output_dir / "result.json").resolve()),
            "analysis": str((output_dir / "analysis.md").resolve()),
            "per_class_metrics": str((output_dir / "per_class_metrics.json").resolve()),
        },
    }
    (output_dir / "per_class_metrics.json").write_text(
        json.dumps(
            {
                "label_names": list(LABEL_NAMES),
                "methods": {
                    name: value["per_class"] for name, value in methods.items()
                },
                "best_of_two_any_correct": best_oracle["per_class_any_correct"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_analysis(result, output_dir / "analysis.md")
    return result


def _default_checkpoint() -> Path:
    result_path = REPO_ROOT / (
        "experiments/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/result.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return Path(str(payload["checkpoint"])).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Existing reduced12 archive root; no recursive scan is performed.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    checkpoint = (args.checkpoint or _default_checkpoint()).expanduser().resolve()
    archive_root = (
        args.archive_root
        if args.archive_root is not None
        else data_root / DEFAULT_ARCHIVE_ROOT_NAME
    ).expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    result = run(
        data_root,
        checkpoint,
        archive_root,
        args.output_dir.expanduser().resolve(),
        args.device,
        args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "moving_contexts": result["population"]["moving_contexts"],
                "best_of_two_any_correct_rate": result["best_of_two_oracle"]["any_correct_rate"],
                "test_used": result["test_used"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
