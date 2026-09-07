#!/usr/bin/env python3
"""Audit reduced14 recognizability and viewpoint recoverability on Val only.

The development-Val metrics are computed with the frozen reduced14 ST-GCN on
the 270-sample ST-GCN development Val tensor.  ActiveView metrics use the
existing reduced14 eight-placement Val utility rows, whose current and legal
candidate predictions were produced by that same frozen recognizer.  No Test
file, policy Test cache, training loop, or data-generation path is opened.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint


LABELS = (
    "walk",
    "sit",
    "stand up",
    "bend",
    "crawl",
    "stumble",
    "kneel",
    "clap",
    "throw",
    "clean something",
    "kick",
    "knock",
    "punch",
    "touching face",
)
SEED = 42
NUM_CLASSES = len(LABELS)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if truth.shape != pred.shape:
        raise ValueError("labels and predictions must have the same shape")
    matrix = np.bincount(
        truth * num_classes + pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for class_id in range(num_classes):
        tp = float(matrix[class_id, class_id])
        support = float(matrix[class_id].sum())
        predicted = float(matrix[:, class_id].sum())
        fp = predicted - tp
        fn = support - tp
        f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) else 0.0
        f1_values.append(f1)
        per_class.append({
            "action": LABELS[class_id],
            "action_id": class_id,
            "support": int(support),
            "accuracy": float(tp / support) if support else float("nan"),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })
    return {
        "count": int(truth.size),
        "accuracy": float(np.mean(pred == truth)) if truth.size else float("nan"),
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _development_val_metrics(
    data_root: Path,
    checkpoint: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run frozen ST-GCN inference on the independent 270-sample Dev Val."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in the habitat environment; refusing CPU fallback "
            "for frozen ST-GCN evaluation"
        )
    raw_root = data_root / "datasets/reduced14_kneel_babel_diversity_v1/raw-train"
    data_path = raw_root / "val_data.npy"
    labels_path = raw_root / "val_labels.npy"
    mapping_path = raw_root / "label_mapping.json"
    data = np.load(data_path, mmap_mode="r")
    labels = np.asarray(np.load(labels_path), dtype=np.int64)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    categories = [
        label for label, _ in sorted(mapping.items(), key=lambda item: int(item[1]))
    ]
    if tuple(categories) != LABELS:
        raise ValueError(f"Development mapping does not match reduced14 taxonomy: {categories}")
    if data.shape != (len(labels), 3, 30, 17, 1):
        raise ValueError(f"Unexpected development Val shape: {data.shape}")
    model, device = load_checkpoint(checkpoint, NUM_CLASSES, device_name)
    if device.type != "cuda":
        raise RuntimeError("Frozen ST-GCN did not load on CUDA")
    predictions: list[int] = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            batch_array = np.array(data[start:start + batch_size], dtype=np.float32, copy=True)
            batch = torch.from_numpy(batch_array).to(device)
            logits = model(batch)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    metrics = _classification_metrics(labels, predictions)
    metrics.update({
        "source_data": str(data_path.resolve()),
        "source_data_sha256": _file_sha256(data_path),
        "source_labels": str(labels_path.resolve()),
        "source_labels_sha256": _file_sha256(labels_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "device": str(device),
        "inference_seconds": time.monotonic() - started,
    })
    return metrics


def _validate_val_cache(data_root: Path, utility_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate only the existing moving-Val cache against Val utility IDs."""
    cache_path = (
        data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
        / "counterfactual_cache/val.npz"
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        ids = [str(value) for value in np.asarray(archive["episode_ids"]).tolist()]
        labels = np.asarray(archive["label_id"], dtype=np.int64)
        if labels.shape != (len(ids),):
            raise ValueError("Val cache label shape is not aligned")
        if np.asarray(archive["true_logp"]).shape != (len(ids), 32, NUM_CLASSES):
            raise ValueError("Val cache true_logp shape is not canonical")
    rows_by_id = {str(row["episode_id"]): row for row in utility_rows}
    missing = [episode_id for episode_id in ids if episode_id not in rows_by_id]
    if missing:
        raise ValueError(f"Val cache IDs missing from utility rows: {missing[:3]}")
    label_mismatch = [
        episode_id for episode_id, label in zip(ids, labels)
        if int(rows_by_id[episode_id]["label_id"]) != int(label)
    ]
    if label_mismatch:
        raise ValueError(f"Val cache label mismatch: {label_mismatch[:3]}")
    return {
        "path": str(cache_path.resolve()),
        "sha256": _file_sha256(cache_path),
        "moving_contexts": len(ids),
        "utility_contexts": len(utility_rows),
        "cache_ids_subset_of_val": True,
    }


def _activeview_metrics(
    data_root: Path,
    utility_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _load_jsonl(utility_path)
    if not rows:
        raise ValueError("ActiveView Val utility rows are empty")
    if any(str(row.get("policy_split")) != "val" for row in rows):
        raise ValueError("ActiveView utility input contains a non-Val row")
    cache_info = _validate_val_cache(data_root, rows)
    labels = [int(row["label_id"]) for row in rows]
    s0_predictions: list[int] = []
    random_predictions: list[int] = []
    candidate_oracle_predictions: list[int] = []
    safe_oracle_predictions: list[int] = []
    legal_labels: list[int] = []
    legal_predictions: list[int] = []
    any_candidate: list[bool] = []
    any_including_stay: list[bool] = []
    correct_counts: list[int] = []
    legal_counts: list[int] = []
    rng = np.random.default_rng(SEED)

    for row in rows:
        label = int(row["label_id"])
        current = dict(row["current"])
        candidates = [dict(candidate) for candidate in row.get("candidates", [])]
        if not candidates:
            raise ValueError(f"Val context has no legal candidates: {row['episode_id']}")
        current_prediction = int(current["predicted_label_id"])
        s0_predictions.append(current_prediction)
        candidate_correct = [
            int(int(candidate["predicted_label_id"]) == label) for candidate in candidates
        ]
        any_candidate.append(bool(any(candidate_correct)))
        any_including_stay.append(bool(current_prediction == label or any(candidate_correct)))
        correct_counts.append(int(sum(candidate_correct)))
        legal_counts.append(len(candidates))
        legal_labels.extend([label] * len(candidates))
        legal_predictions.extend(int(candidate["predicted_label_id"]) for candidate in candidates)

        random_index = int(rng.integers(0, len(candidates)))
        random_predictions.append(int(candidates[random_index]["predicted_label_id"]))
        best_candidate = max(
            enumerate(candidates),
            key=lambda item: (float(item[1]["logp_true"]), -item[0]),
        )[1]
        candidate_oracle_predictions.append(int(best_candidate["predicted_label_id"]))
        if float(best_candidate["logp_true"]) > float(current["logp_true"]):
            safe_oracle_predictions.append(int(best_candidate["predicted_label_id"]))
        else:
            safe_oracle_predictions.append(current_prediction)

    methods = {
        "s0": _classification_metrics(labels, s0_predictions),
        "legal_view_mean": _classification_metrics(legal_labels, legal_predictions),
        "random_legal_view": _classification_metrics(labels, random_predictions),
        "h0_candidate_oracle": _classification_metrics(labels, candidate_oracle_predictions),
        "h0_safe_oracle": _classification_metrics(labels, safe_oracle_predictions),
    }
    per_class_aux: list[dict[str, Any]] = []
    for class_id, action in enumerate(LABELS):
        indices = [index for index, label in enumerate(labels) if label == class_id]
        legal_indices = [index for index, label in enumerate(legal_labels) if label == class_id]
        per_class_aux.append({
            "action": action,
            "action_id": class_id,
            "support": len(indices),
            "h0_any_correct_candidate_only_rate": _mean(any_candidate[i] for i in indices),
            "h0_any_correct_rate_including_stay": _mean(any_including_stay[i] for i in indices),
            "mean_correct_view_ratio": _mean(correct_counts[i] / legal_counts[i] for i in indices),
            "mean_correct_legal_views": _mean(correct_counts[i] for i in indices),
            "mean_legal_candidate_count": _mean(legal_counts[i] for i in indices),
            "legal_view_sample_count": len(legal_indices),
        })
    methods["auxiliary_per_class"] = per_class_aux
    methods["episode_count"] = len(rows)
    methods["legal_candidate_count"] = len(legal_labels)
    methods["cache_alignment"] = cache_info
    return methods, {
        "labels": labels,
        "s0_predictions": s0_predictions,
        "random_predictions": random_predictions,
        "candidate_oracle_predictions": candidate_oracle_predictions,
        "safe_oracle_predictions": safe_oracle_predictions,
    }


def _merge_per_class(active: Mapping[str, Any], development: Mapping[str, Any]) -> list[dict[str, Any]]:
    active_by_action = {row["action_id"]: row for row in active["auxiliary_per_class"]}
    result: list[dict[str, Any]] = []
    for dev_row in development["per_class"]:
        class_id = int(dev_row["action_id"])
        row = {
            "action": LABELS[class_id],
            "action_id": class_id,
            "support": int(dev_row["support"]),
            "stgcn_development_val_accuracy": float(dev_row["accuracy"]),
            "stgcn_development_val_f1": float(dev_row["f1"]),
        }
        row.update({
            "activeview_val_s0_accuracy": float(active["s0"]["per_class"][class_id]["accuracy"]),
            "activeview_val_s0_f1": float(active["s0"]["per_class"][class_id]["f1"]),
            "legal_view_mean_accuracy": float(active["legal_view_mean"]["per_class"][class_id]["accuracy"]),
            "random_legal_view_accuracy": float(active["random_legal_view"]["per_class"][class_id]["accuracy"]),
            "random_legal_view_f1": float(active["random_legal_view"]["per_class"][class_id]["f1"]),
            "h0_candidate_oracle_accuracy": float(active["h0_candidate_oracle"]["per_class"][class_id]["accuracy"]),
            "h0_candidate_oracle_f1": float(active["h0_candidate_oracle"]["per_class"][class_id]["f1"]),
            "h0_safe_oracle_accuracy": float(active["h0_safe_oracle"]["per_class"][class_id]["accuracy"]),
            "h0_safe_oracle_f1": float(active["h0_safe_oracle"]["per_class"][class_id]["f1"]),
        })
        row.update({
            key: active_by_action[class_id][key]
            for key in (
                "h0_any_correct_candidate_only_rate",
                "h0_any_correct_rate_including_stay",
                "mean_correct_view_ratio",
                "mean_correct_legal_views",
                "mean_legal_candidate_count",
                "legal_view_sample_count",
            )
        })
        row["clean_recognizability"] = row["stgcn_development_val_f1"]
        row["viewpoint_recoverability"] = (
            row["h0_safe_oracle_accuracy"] - row["random_legal_view_accuracy"]
        )
        result.append(row)
    return result


def _group_classes(per_class: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clean = np.asarray([float(row["clean_recognizability"]) for row in per_class])
    recovery = np.asarray([float(row["viewpoint_recoverability"]) for row in per_class])
    any_correct = np.asarray([float(row["h0_any_correct_rate_including_stay"]) for row in per_class])
    clean_high = float(np.median(clean))
    recovery_high = float(np.median(recovery))
    clean_low = float(np.percentile(clean, 25.0))
    any_low = float(np.median(any_correct))
    groups: dict[str, list[str]] = {"A_recommended_to_retain": [], "B_manual_review": [], "C_potential_removal_candidates": []}
    annotated: list[dict[str, Any]] = []
    for row in per_class:
        clean_value = float(row["clean_recognizability"])
        recovery_value = float(row["viewpoint_recoverability"])
        any_value = float(row["h0_any_correct_rate_including_stay"])
        if clean_value >= clean_high and recovery_value >= recovery_high:
            group = "A_recommended_to_retain"
            rationale = "clean F1 and recoverability are both at/above their class medians"
        elif clean_value <= clean_low and any_value <= any_low:
            group = "C_potential_removal_candidates"
            rationale = "clean F1 is in the bottom quartile and even s0-or-candidate AnyCorrect is at/below median"
        else:
            group = "B_manual_review"
            rationale = "mixed clean recognizability and viewpoint-recovery evidence"
        groups[group].append(str(row["action"]))
        annotated.append({"action": row["action"], "group": group, "group_rationale": rationale})
    candidates = [
        row for row in per_class
        if row["action"] in groups["C_potential_removal_candidates"]
    ]
    candidates = sorted(
        candidates,
        key=lambda row: (float(row["clean_recognizability"]), float(row["h0_any_correct_rate_including_stay"])),
    )[:4]
    return {
        "thresholds": {
            "clean_high_median": clean_high,
            "recoverability_high_median": recovery_high,
            "clean_low_25th_percentile": clean_low,
            "any_correct_low_median": any_low,
        },
        "groups": groups,
        "annotations": annotated,
        "potential_removal_candidates_max4": [str(row["action"]) for row in candidates],
        "selection_basis": "diagnostic clean ST-GCN F1, H0 AnyCorrect and viewpoint recoverability only; no final-policy/Test score",
    }


def _special_action_audit(per_class: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = {"clean something", "knock", "touching face", "throw", "clap", "kneel", "crawl"}
    rows: list[dict[str, Any]] = []
    for row in per_class:
        if row["action"] not in selected:
            continue
        action = str(row["action"])
        if action in {"clean something", "knock", "touching face", "throw", "clap"}:
            hypothesis = "upper-body/hand or object interaction may merit manual inspection"
        else:
            hypothesis = "ground-level posture/leg visibility may merit manual inspection"
        rows.append({
            "action": action,
            "hypothesis_to_check": hypothesis,
            "evidence": {
                "clean_f1": row["clean_recognizability"],
                "activeview_s0_accuracy": row["activeview_val_s0_accuracy"],
                "h0_any_correct": row["h0_any_correct_rate_including_stay"],
                "viewpoint_recoverability": row["viewpoint_recoverability"],
            },
            "interpretation_limit": "metrics do not by themselves prove H36M17 insufficiency or an object/occlusion cause",
        })
    return rows


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    per_class = result["per_class"]
    groups = result["groups"]
    lines = [
        "# Reduced14 per-class recognizability and viewpoint-recoverability audit",
        "",
        "Val-only audit using the frozen reduced14 ST-GCN and existing reduced14 eight-placement ActiveView Val utility/cache. No model was trained, no data was regenerated, and no policy Test artifact was opened.",
        "",
        "## Complete per-class table",
        "",
        "| Action | Dev Val Acc/F1 | ActiveView s0 Acc/F1 | Legal-view mean Acc | Random legal Acc | H0 CandidateOracle Acc | H0 SafeOracle Acc | AnyCorrect | Correct-view ratio | # correct legal views | Recoverability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_class:
        lines.append(
            f"| {row['action']} | {row['stgcn_development_val_accuracy']:.4f}/{row['stgcn_development_val_f1']:.4f} | "
            f"{row['activeview_val_s0_accuracy']:.4f}/{row['activeview_val_s0_f1']:.4f} | "
            f"{row['legal_view_mean_accuracy']:.4f} | {row['random_legal_view_accuracy']:.4f} | "
            f"{row['h0_candidate_oracle_accuracy']:.4f} | {row['h0_safe_oracle_accuracy']:.4f} | "
            f"{row['h0_any_correct_rate_including_stay']:.4f} | {row['mean_correct_view_ratio']:.4f} | "
            f"{row['mean_correct_legal_views']:.3f} | {row['viewpoint_recoverability']:.4f} |"
        )
    lines.extend([
        "",
        "`Dev Val Acc/F1` is the frozen ST-GCN development-Val class recall/F1 on 270 samples. `ActiveView s0` and all H0/legal-view quantities use the 19,440 records-only ActiveView Val contexts and their embedded frozen predictions. `AnyCorrect` is reported including the current s0 view; candidate-only values remain in `result.json`.",
        "",
        "## Diagnostic groups",
        "",
        f"Thresholds are transparent distributional triage thresholds: clean-high = class-median F1 ({groups['thresholds']['clean_high_median']:.4f}), recoverability-high = class-median ({groups['thresholds']['recoverability_high_median']:.4f}), clean-low = 25th-percentile F1 ({groups['thresholds']['clean_low_25th_percentile']:.4f}), and AnyCorrect-low = class median ({groups['thresholds']['any_correct_low_median']:.4f}). They are not deletion rules.",
        "",
        f"- **A — recommended retain:** {', '.join(groups['groups']['A_recommended_to_retain']) or 'none'}",
        f"- **B — manual review:** {', '.join(groups['groups']['B_manual_review']) or 'none'}",
        f"- **C — potential removal candidates:** {', '.join(groups['groups']['C_potential_removal_candidates']) or 'none'}",
        "",
        f"At most four lowest clean-F1/AnyCorrect classes from group C are listed as potential candidates (not automatically removed): {', '.join(groups['potential_removal_candidates_max4']) or 'none'}.",
        "",
        "## Requested action checks",
        "",
        "The following actions receive a focused metric readout because they involve hand/object interaction or ground-level posture. These are hypotheses for human inspection only; the audit does not add an artificial rule or establish H36M17 insufficiency from metrics alone.",
        "",
        "| Action | Clean F1 | s0 Acc | H0 AnyCorrect | Recoverability |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in result["special_action_audit"]:
        lines.append(
            f"| {row['action']} | {row['evidence']['clean_f1']:.4f} | {row['evidence']['activeview_s0_accuracy']:.4f} | "
            f"{row['evidence']['h0_any_correct']:.4f} | {row['evidence']['viewpoint_recoverability']:.4f} |"
        )
    lines.extend([
        "",
        "## Scientific interpretation",
        "",
        "- A high clean F1 with a large positive viewpoint-recoverability gap supports retaining the action while treating viewpoint/occlusion as a plausible limiting factor.",
        "- Mixed cases belong in manual review; a low ActiveView score alone is not evidence for deletion.",
        "- Potential deletion candidates require both low clean recognizability and low H0 AnyCorrect, and should be checked independently of final-policy performance.",
        "- This audit does not provide evidence by itself that reduced14 should be shortened to 10–12 classes; any taxonomy change requires a separate protocol decision.",
        "- No Test score, Ours/final-policy score, or policy outcome was used to select a class group.",
        "",
        "## Leakage/runtime flags",
        "",
        "- `test_used = false`",
        "- `training_performed = false`",
        "- `taxonomy_modified = false`",
        "- `data_regenerated = false`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    data_root: Path,
    checkpoint: Path,
    utility_path: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    started = time.monotonic()
    development = _development_val_metrics(data_root, checkpoint, device_name, batch_size)
    active, _ = _activeview_metrics(data_root, utility_path)
    per_class = _merge_per_class(active, development)
    result: dict[str, Any] = {
        "experiment": "reduced14_per_class_recognizability_audit",
        "version": "reduced14-eight-placement-val-v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "val",
        "taxonomy": list(LABELS),
        "scene_split": False,
        "test_used": False,
        "training_performed": False,
        "taxonomy_modified": False,
        "data_regenerated": False,
        "source": {
            "data_root": str(data_root.resolve()),
            "stgcn_development_val": development,
            "activeview_utility": str(utility_path.resolve()),
            "activeview_utility_sha256": _file_sha256(utility_path),
            "activeview": active,
        },
        "per_class": per_class,
        "groups": _group_classes(per_class),
        "special_action_audit": _special_action_audit(per_class),
        "definitions": {
            "clean_recognizability": "frozen ST-GCN development Val per-class F1",
            "viewpoint_recoverability": "H0 SafeOracle per-class accuracy minus Random legal-view per-class accuracy",
            "h0_any_correct_main": "at least one correct legal candidate or correct current s0 prediction",
            "legal_view_mean": "candidate-pair correctness averaged over all legal candidates of that class",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_root = get_data_root()
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth",
    )
    parser.add_argument(
        "--utility-path",
        type=Path,
        default=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1/stage_b/utility_labels/val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/reduced14_eight_placement_v1/per_class_recognizability_audit"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = run_audit(
        data_root=args.data_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        utility_path=args.utility_path.resolve(),
        output_dir=args.output_dir.resolve(),
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "status": "completed",
        "split": result["split"],
        "test_used": result["test_used"],
        "development_val_count": result["source"]["stgcn_development_val"]["count"],
        "activeview_val_contexts": result["source"]["activeview"]["episode_count"],
        "legal_candidate_count": result["source"]["activeview"]["legal_candidate_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
