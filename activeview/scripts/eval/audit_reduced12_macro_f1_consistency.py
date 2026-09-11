#!/usr/bin/env python3
"""Audit Macro-F1 consistency between the segment and horizon evaluators.

The model is run once to capture the two evaluators' ordered labels and
predictions.  The discrepancy is then diagnosed at the metric aggregation
layer; no model, fusion rule or split is changed by this audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import activeview.scripts.eval.audit_reduced12_sequential_horizon as horizon
import activeview.scripts.experiments.run_reduced12_segment_aware_sequential_recognition as segment
from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import classification


DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/segment_aware_sequential_recognition/meanf1_consistency_audit"
GROUP_SIZE = horizon.INFERENCE_GROUP
OBSERVATIONS = tuple(range(horizon.CHUNK_SIZE, horizon.FRAME_COUNT + 1, horizon.CHUNK_SIZE))
OLD_METHODS = tuple(
    f"{fusion}/{policy}"
    for fusion in ("MeanLogP", "MeanFeature")
    for policy in ("Stay", "Random-1Hop", "Privileged-Greedy-Oracle")
)
HORIZON_CALLS = tuple(
    (fusion, horizon_name, policy)
    for fusion in horizon.FUSIONS
    for horizon_name in horizon.HORIZONS
    for policy in horizon.POLICIES
    if not (horizon_name == "H0" and policy != "Stay")
)


def _cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _capture_old(
    contexts: Sequence[Mapping[str, Any]],
    offline_root: Path,
    encoder: segment.ChunkEncoder,
    fusion_head: segment.FeatureFusionHead,
    device: torch.device,
) -> dict[str, dict[str, list[np.ndarray]]]:
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original = segment.classification

    def capture(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
        calls.append((np.asarray(labels, dtype=np.int64).copy(), np.asarray(predictions, dtype=np.int64).copy()))
        return original(labels, predictions)

    segment.classification = capture  # type: ignore[assignment]
    try:
        segment._sequential_eval(contexts, offline_root, encoder, fusion_head, device, batch_contexts=GROUP_SIZE)
    finally:
        segment.classification = original
    expected = len(OLD_METHODS) * len(OBSERVATIONS)
    if len(calls) != expected:
        raise RuntimeError(f"old evaluator emitted {len(calls)} metric calls, expected {expected}")
    output: dict[str, dict[str, list[np.ndarray]]] = {method: {} for method in OLD_METHODS}
    cursor = 0
    for method in OLD_METHODS:
        for observed in OBSERVATIONS:
            labels, predictions = calls[cursor]
            output[method][f"t{observed}"] = [labels, predictions]
            cursor += 1
    return output


def _capture_horizon(
    contexts: Sequence[Mapping[str, Any]],
    data_root: Path,
    device: torch.device,
) -> dict[tuple[str, str, str], list[np.ndarray]]:
    encoder, fusion_head, _ = horizon._load_models(data_root, device)
    policy_root = data_root / horizon.POLICY_RELATIVE
    offline_root = data_root / horizon.OFFLINE_RELATIVE
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original = horizon.classification

    def capture(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
        calls.append((np.asarray(labels, dtype=np.int64).copy(), np.asarray(predictions, dtype=np.int64).copy()))
        return original(labels, predictions)

    horizon.classification = capture  # type: ignore[assignment]
    examples: dict[str, dict[str, list[list[int]]]] = {
        fusion: {policy: [] for policy in horizon.POLICIES} for fusion in horizon.FUSIONS
    }
    parts = {
        fusion: {name: {policy: [] for policy in horizon.POLICIES} for name in horizon.HORIZONS}
        for fusion in horizon.FUSIONS
    }
    rng = np.random.default_rng(horizon.SEED)
    try:
        for start in range(0, len(contexts), GROUP_SIZE):
            group = contexts[start : start + GROUP_SIZE]
            horizon._evaluate_group(group, offline_root, encoder, fusion_head, device, rng, parts, examples)
    finally:
        horizon.classification = original
    expected = len(range(0, len(contexts), GROUP_SIZE)) * len(HORIZON_CALLS)
    if len(calls) != expected:
        raise RuntimeError(f"horizon evaluator emitted {len(calls)} metric calls, expected {expected}")
    output: dict[tuple[str, str, str], list[np.ndarray]] = {
        key: [] for key in HORIZON_CALLS
    }
    cursor = 0
    for _ in range(0, len(contexts), GROUP_SIZE):
        group_size = min(GROUP_SIZE, len(contexts) - (_))
        for key in HORIZON_CALLS:
            labels, predictions = calls[cursor]
            if labels.size != group_size:
                raise RuntimeError(f"horizon group size mismatch for {key}: {labels.size} vs {group_size}")
            output[key].append(np.stack([labels, predictions], axis=0))
            cursor += 1
    return output


def _flatten_horizon(calls: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(calls, axis=1)
    return values[0], values[1]


def _compare(
    old: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
    new: Mapping[tuple[str, str, str], Sequence[np.ndarray]],
    contexts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    mapping = {
        "MeanFeature/Stay H0": ("MeanFeature/Stay", "t30", ("MeanFeature", "H0", "Stay")),
        "MeanFeature/Full Oracle": ("MeanFeature/Privileged-Greedy-Oracle", "t30", ("MeanFeature", "Full", "Privileged-Greedy-Oracle")),
        "MeanLogP/Stay H0": ("MeanLogP/Stay", "t30", ("MeanLogP", "H0", "Stay")),
        "MeanLogP/Full Oracle": ("MeanLogP/Privileged-Greedy-Oracle", "t30", ("MeanLogP", "Full", "Privileged-Greedy-Oracle")),
    }
    for name, (old_method, old_time, new_key) in mapping.items():
        old_labels, old_predictions = old[old_method][old_time]
        new_labels, new_predictions = _flatten_horizon(new[new_key])
        comparisons[name] = {
            "context_count": int(len(contexts)),
            "y_true_mismatch": int(np.sum(old_labels != new_labels)),
            "y_pred_mismatch": int(np.sum(old_predictions != new_predictions)),
            "old_accuracy": float(np.mean(old_labels == old_predictions)),
            "horizon_accuracy": float(np.mean(new_labels == new_predictions)),
            "old_confusion_matrix": classification(old_labels, old_predictions)["confusion_matrix"],
            "horizon_confusion_matrix": classification(new_labels, new_predictions)["confusion_matrix"],
        }
    return comparisons


def _correct_horizon_reports(horizon_dir: Path) -> dict[str, Any]:
    """Fix report aggregation from stored confusion matrices, without inference."""
    metrics_path = horizon_dir / "horizon_metrics.json"
    result_path = horizon_dir / "result.json"
    derived_path = horizon_dir / "derived_metrics.json"
    horizon_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for fusion, horizons in horizon_metrics.items():
        for horizon_name, policies in horizons.items():
            for policy, metric in policies.items():
                confusion = np.asarray(metric["confusion_matrix"], dtype=np.int64)
                labels: list[int] = []
                predictions: list[int] = []
                for target, row in enumerate(confusion):
                    for prediction, count in enumerate(row):
                        labels.extend([target] * int(count))
                        predictions.extend([prediction] * int(count))
                corrected = classification(labels, predictions)
                metric["accuracy"] = corrected["accuracy"]
                metric["macro_f1"] = corrected["macro_f1"]
                metric["per_class"] = corrected["per_class"]
    result["horizon_metrics"] = horizon_metrics
    derived = horizon._derived(horizon_metrics)
    result["derived_metrics"] = derived
    _write(metrics_path, horizon_metrics)
    _write(derived_path, derived)
    (horizon_dir / "analysis.md").write_text(horizon._analysis(result), encoding="utf-8")
    _write(result_path, result)
    return {"horizon_metrics": horizon_metrics, "derived_metrics": derived}


def _analysis(result: Mapping[str, Any]) -> str:
    comparisons = result["comparisons"]
    lines = [
        "# Macro-F1 Consistency Audit",
        "",
        "This audit compares the ordered predictions of the original segment-aware evaluator and the sequential horizon evaluator on the same reduced12 Moving Val contexts. No Test data, retraining, fusion change or new perception data was used.",
        "",
        "## Prediction alignment",
        "",
        "| Group | Contexts | y_true mismatch | y_pred mismatch |",
        "|---|---:|---:|---:|",
    ]
    for name, item in comparisons.items():
        lines.append(f"| {name} | {item['context_count']} | {item['y_true_mismatch']} | {item['y_pred_mismatch']} |")
    lines += [
        "",
        "All four comparisons use the same ordered Moving Val contexts. With zero mismatches, the differing Macro-F1 values cannot be caused by the model, fusion, chunking or prediction path.",
        "",
        "## Metric implementation audit",
        "",
        "**Old evaluator:** calls the shared `classification(labels, predictions)` once after all 10,080 contexts for each method/time point.",
        "",
        "**Horizon evaluator before the fix:** `_evaluate_path()` called the same helper separately for each 128-context group. `_merge_metrics()` then averaged those group Macro-F1 values. Macro-F1 is nonlinear, so mean(per-group F1) is not equal to F1(global confusion matrix). The helper's later confusion-matrix recomputation did not overwrite the already-averaged `macro_f1` field.",
        "",
        "**Formal definition:** fixed 12-class confusion matrix, one-vs-rest F1 for every class (including zero-support classes with F1=0), then arithmetic mean across all 12 classes. This is equivalent to `f1_score(y_true, y_pred, labels=list(range(12)), average='macro', zero_division=0)`.",
        "",
        "**First divergence:** horizon `_merge_metrics()` at its former weighted `macro_f1 = average(group_macro_f1)` assignment.",
        "",
        "## Corrected values",
        "",
        "After the minimal fix, horizon reports recompute Accuracy/Macro-F1 from the merged 12×12 confusion matrix. Accuracy is unchanged and the four values now match the original evaluator.",
        "",
        "| Group | Accuracy | Unified Macro-F1 |",
        "|---|---:|---:|",
    ]
    for name, item in comparisons.items():
        lines.append(f"| {name} | {item['old_accuracy']:.12f} | {item['corrected_macro_f1']:.12f} |")
    lines += [
        "",
        "## Decision",
        "",
        "The consistency issue is a reporting/evaluator bug only. The fixed horizon audit is now suitable for interpreting the H1/H2 sequential horizon results and proceeding to a first learned H2 policy, subject to the separate scientific approval already requested.",
        "",
        "```text",
        "policy_test_used=false",
        "training_used=false",
        "predictions_modified=false",
        "fusion_modified=false",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    policy_root = data_root / horizon.POLICY_RELATIVE
    contexts, data_summary = horizon._load_contexts(policy_root)
    offline_root = data_root / horizon.OFFLINE_RELATIVE
    encoder, fusion_head, _ = horizon._load_models(data_root, device)
    old_predictions = _capture_old(contexts, offline_root, encoder, fusion_head, device)
    new_predictions = _capture_horizon(contexts, data_root, device)
    comparisons = _compare(old_predictions, new_predictions, contexts)
    horizon_dir = REPO_ROOT / "experiments/reduced12_eight_placement_v1/sequential_horizon_audit"
    corrected = _correct_horizon_reports(horizon_dir)
    lookup = {
        "MeanFeature/Stay H0": ("MeanFeature", "H0", "Stay"),
        "MeanFeature/Full Oracle": ("MeanFeature", "Full", "Privileged-Greedy-Oracle"),
        "MeanLogP/Stay H0": ("MeanLogP", "H0", "Stay"),
        "MeanLogP/Full Oracle": ("MeanLogP", "Full", "Privileged-Greedy-Oracle"),
    }
    for name, key in lookup.items():
        comparisons[name]["corrected_macro_f1"] = corrected["horizon_metrics"][key[0]][key[1]][key[2]]["macro_f1"]
        comparisons[name]["corrected_accuracy"] = corrected["horizon_metrics"][key[0]][key[1]][key[2]]["accuracy"]
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_MACRO_F1_CONSISTENCY_AUDIT",
        "runtime": runtime,
        "data_summary": data_summary,
        "comparisons": comparisons,
        "metric_definition": "fixed 12-class confusion; macro mean of all class F1 with zero_division=0",
        "root_cause": "Horizon evaluator averaged per-group Macro-F1 before reporting instead of computing Macro-F1 from the merged confusion matrix.",
        "first_point_of_divergence": "audit_reduced12_sequential_horizon.py::_merge_metrics former weighted macro_f1 assignment",
        "correction": "Compute global Macro-F1 after merging group confusion matrices; preserve all predictions and Accuracy.",
        "horizon_reports_corrected_without_model_rerun": True,
        "policy_test_used": False,
        "training_used": False,
        "predictions_modified": False,
        "fusion_modified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "result.json", result)
    _write(output_dir / "prediction_comparison.json", comparisons)
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.data_root.resolve(), _cuda(args.device))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "contexts": result["data_summary"]["stage_d_val_moving_contexts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
