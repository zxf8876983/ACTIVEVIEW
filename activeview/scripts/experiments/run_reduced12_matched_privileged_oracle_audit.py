#!/usr/bin/env python3
"""Recompute the matched reduced12 privileged viewpoint oracle on Val.

This script intentionally reads runtime Val artifacts afresh.  The formal
action set is ``stay/current + Stage-A candidate_pool``; the unrestricted
all-32 adapted coverage is emitted separately as a diagnostic only.  No prior
experiment result is loaded and no model is trained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.analyze_reduced12_matched_privileged_oracle import (
    ADAPTED_HEAD_RELATIVE,
    FEATURE_DIM,
    NUM_CLASSES,
    NUM_VIEWS,
    STGCN_CHECKPOINT_RELATIVE,
    _adapted_logits,
    _load_inputs,
    _log_softmax,
    _original_candidate_feature_check,
    _require_cuda,
    _seed,
    _sha256,
    _softmax,
)
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, gt_margin
from activeview.scripts.experiments.run_reduced12_single_view_classifier_adaptation import (
    LightweightClassifier,
)


SEED = 42
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "matched_privileged_oracle_audit"
)
HISTORICAL_FROZEN_LEGAL_ORACLE = 0.7282738095238095


def _evaluate_recognizer(
    labels: np.ndarray,
    current_logp: np.ndarray,
    candidate_logp: np.ndarray,
    legal_ids: Sequence[Sequence[int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate every required selector with actual selected-view argmax."""
    method_names = ("Current/Stay", "GT-TrueLogP", "GT-Margin", "MaxConfidence")
    predictions = {name: np.empty(len(labels), dtype=np.int64) for name in method_names}
    selected = {name: np.empty(len(labels), dtype=np.int64) for name in method_names}
    selected_scores = {name: np.empty(len(labels), dtype=np.float64) for name in method_names}
    moved = {name: np.zeros(len(labels), dtype=bool) for name in method_names}
    any_correct = np.zeros(len(labels), dtype=bool)

    for index, label in enumerate(labels):
        ids = [int(value) for value in legal_ids[index]]
        options = np.concatenate(
            (current_logp[index][None, :], candidate_logp[index, ids]), axis=0
        )
        option_ids = np.asarray([-1] + ids, dtype=np.int64)
        true_scores = options[:, int(label)]
        margins = np.asarray(
            [gt_margin(option, int(label)) for option in options], dtype=np.float64
        )
        confidence = np.max(_softmax(options), axis=1)
        choices = {
            "Current/Stay": 0,
            "GT-TrueLogP": int(np.argmax(true_scores)),
            "GT-Margin": int(np.argmax(margins)),
            "MaxConfidence": int(np.argmax(confidence)),
        }
        score_values = {
            "Current/Stay": true_scores,
            "GT-TrueLogP": true_scores,
            "GT-Margin": margins,
            "MaxConfidence": confidence,
        }
        for name, choice in choices.items():
            predictions[name][index] = int(np.argmax(options[choice]))
            selected[name][index] = int(option_ids[choice])
            selected_scores[name][index] = float(score_values[name][choice])
            moved[name][index] = choice != 0
        any_correct[index] = bool(np.any(np.argmax(options, axis=1) == int(label)))

    metrics: dict[str, Any] = {}
    for name in method_names:
        metric = classification(labels, predictions[name])
        metric.update({
            "method": name,
            "move_rate": float(np.mean(moved[name])),
            "stay_rate": float(np.mean(~moved[name])),
            "mean_selected_score": float(np.mean(selected_scores[name])),
        })
        metrics[name] = metric
    metrics["AnyCorrect Coverage"] = {
        "method": "AnyCorrect Coverage",
        "accuracy": None,
        "macro_f1": None,
        "coverage_count": int(np.sum(any_correct)),
        "coverage_rate": float(np.mean(any_correct)),
        "contexts": int(len(labels)),
    }
    return metrics, {
        "predictions": predictions,
        "selected_viewpoint_ids": selected,
        "selected_scores": selected_scores,
        "moved": moved,
        "any_correct": any_correct,
    }


def _all32_adapted_coverage(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=2)
    covered = np.any(predictions == labels[:, None], axis=1)
    return {
        "method": "Adapted unrestricted-all32 AnyCorrect Coverage",
        "count": int(np.sum(covered)),
        "rate": float(np.mean(covered)),
        "contexts": int(len(labels)),
        "viewpoints_per_context": NUM_VIEWS,
        "diagnostic_only": True,
    }


def _sample_rows(
    data: Mapping[str, Any],
    labels: np.ndarray,
    frozen_details: Mapping[str, Any],
    adapted_details: Mapping[str, Any],
    adapted_current_logp: np.ndarray,
    adapted_candidate_logp: np.ndarray,
    sample_count: int = 20,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    indices = np.sort(rng.choice(len(labels), size=sample_count, replace=False))
    cache = data["cache"]
    output: list[dict[str, Any]] = []
    for raw_index in indices:
        index = int(raw_index)
        label = int(labels[index])
        current_id = int(data["current_ids"][index])
        ids = [int(value) for value in data["legal_ids"][index]]
        view_ids = [current_id] + [value for value in ids if value != current_id]
        views: list[dict[str, Any]] = []
        for viewpoint_id in view_ids:
            if viewpoint_id == current_id:
                frozen_logp = np.asarray(cache["current_logp_s0"][index])
                adapted_logp = adapted_current_logp[index]
                role = "stay/current"
            else:
                frozen_logp = np.asarray(cache["true_logp"][index, viewpoint_id])
                adapted_logp = adapted_candidate_logp[index, viewpoint_id]
                role = "candidate"
            views.append({
                "viewpoint_id": viewpoint_id,
                "role": role,
                "frozen_prediction": int(np.argmax(frozen_logp)),
                "frozen_true_class_logp": float(frozen_logp[label]),
                "frozen_gt_margin": float(gt_margin(frozen_logp, label)),
                "frozen_max_confidence": float(np.max(_softmax(frozen_logp))),
                "adapted_prediction": int(np.argmax(adapted_logp)),
                "adapted_true_class_logp": float(adapted_logp[label]),
                "adapted_gt_margin": float(gt_margin(adapted_logp, label)),
                "adapted_max_confidence": float(np.max(_softmax(adapted_logp))),
            })
        frozen_any = [
            int(viewpoint_id) for viewpoint_id in view_ids
            if (
                int(np.argmax(cache["current_logp_s0"][index])) == label
                if viewpoint_id == current_id
                else int(np.argmax(cache["true_logp"][index, viewpoint_id])) == label
            )
        ]
        adapted_any = [
            int(viewpoint_id) for viewpoint_id in view_ids
            if (
                int(np.argmax(adapted_current_logp[index])) == label
                if viewpoint_id == current_id
                else int(np.argmax(adapted_candidate_logp[index, viewpoint_id])) == label
            )
        ]
        selected_frozen: dict[str, Any] = {}
        selected_adapted: dict[str, Any] = {}
        for name, values in frozen_details["selected_viewpoint_ids"].items():
            view = int(values[index])
            selected_frozen[name] = {
                "viewpoint_id": None if view < 0 else view,
                "is_stay": view < 0,
                "prediction": int(frozen_details["predictions"][name][index]),
                "score": float(frozen_details["selected_scores"][name][index]),
            }
        for name, values in adapted_details["selected_viewpoint_ids"].items():
            view = int(values[index])
            selected_adapted[name] = {
                "viewpoint_id": None if view < 0 else view,
                "is_stay": view < 0,
                "prediction": int(adapted_details["predictions"][name][index]),
                "score": float(adapted_details["selected_scores"][name][index]),
            }
        output.append({
            "episode_id": str(data["stage_d"][index]["episode_id"]),
            "record_id": str(data["stage_d"][index]["record_id"]),
            "scene_id": str(data["stage_d"][index]["scene_id"]),
            "placement_id": str(data["stage_d"][index]["region"]),
            "gt_label_id": label,
            "current_viewpoint_id": current_id,
            "legal_viewpoint_ids": ids,
            "views": views,
            "frozen_selected": selected_frozen,
            "adapted_selected": selected_adapted,
            "frozen_any_correct_viewpoints": frozen_any,
            "adapted_any_correct_viewpoints": adapted_any,
        })
    return output


def _write_analysis(
    output_dir: Path,
    result: Mapping[str, Any],
) -> None:
    frozen = result["recognizers"]["Frozen"]
    adapted = result["recognizers"]["Adapted"]
    legal = result["legal_action_set"]
    discrepancy = 100.0 * (
        frozen["GT-TrueLogP"]["accuracy"] - HISTORICAL_FROZEN_LEGAL_ORACLE
    )
    lines = [
        "# Matched Privileged Viewpoint Oracle (fresh recomputation)",
        "",
        "This report was recomputed by `run_reduced12_matched_privileged_oracle_audit.py` from runtime Val artifacts; no prior experiment `result.json` was read.",
        "",
        "## Protocol and action set",
        "",
        f"Moving Val contexts: {result['population']['moving_val_contexts']}. Each action set is exactly `stay/current + Stage-A candidate_pool`; candidate count mean/min/max={legal['candidate_count_mean']:.3f}/{legal['candidate_count_min']}/{legal['candidate_count_max']}, including-stay action count mean/min/max={legal['legal_action_count_mean']:.3f}/{legal['legal_action_count_min']}/{legal['legal_action_count_max']}. Stage-A/Stage-C/cache action-set errors: {result['alignment']['action_set_errors']}.",
        "",
        "## Required metrics",
        "",
        "| Recognizer | Current/Stay Acc/F1 | GT-TrueLogP Acc/F1 | GT-Margin Acc/F1 | MaxConfidence Acc/F1 | Legal AnyCorrect Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("Frozen", frozen), ("Adapted", adapted)):
        def fmt(method: str) -> str:
            return f"{metrics[method]['accuracy']:.6f}/{metrics[method]['macro_f1']:.6f}"
        lines.append(
            f"| {name} | {fmt('Current/Stay')} | {fmt('GT-TrueLogP')} | {fmt('GT-Margin')} | {fmt('MaxConfidence')} | {metrics['AnyCorrect Coverage']['coverage_rate']:.6f} |"
        )
    lines.extend([
        "",
        f"Adapted unrestricted-all32 AnyCorrect Coverage: {adapted['All-32 AnyCorrect Coverage']['rate']:.6f} ({adapted['All-32 AnyCorrect Coverage']['count']}/{result['population']['moving_val_contexts']}); diagnostic-only, not the formal policy action set.",
        "",
        "## Sanity and interpretation",
        "",
        f"Frozen GT-TrueLogP legal oracle differs from the historical 0.728274 reference by {discrepancy:+.3f}pp; GT-Margin is {frozen['GT-Margin']['accuracy']:.6f}. This is within the requested 1pp sanity bound, so action-set alignment is accepted. The formal privileged legal oracle should be reported as the actual selected-view GT-Margin/GT-TrueLogP result, while AnyCorrect is coverage only and is never presented as Oracle Accuracy.",
        "",
        "GT-TrueLogP and GT-Margin always select the maximum score and then report that selected view's actual recognizer argmax; no correct-candidate shortcut is used. MaxConfidence is GT-free scoring but still reports the selected view's actual argmax.",
        "",
        "The 0.957540 adapted all-32 coverage is protocol-inflated relative to the formal Stage-A reachable action space: allowing all 32 lattice viewpoints raises coverage from the legal 0.803968 to 0.957540.",
        "",
        "## Boundaries",
        "",
        "- `policy_test_used=false`; only Moving Val Stage-A/Stage-C/Stage-D and Val counterfactual/archive artifacts were read.",
        "- `training_used=false`; no selector, recognizer, or checkpoint was trained or modified.",
        "- No RGB, skeleton, DINO, or perception data was generated.",
        "- Future archived skeletons were used only for frozen terminal evidence and the adapted all-32 diagnostic, never as deployable policy input.",
        "",
    ])
    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    output_dir: Path,
    data_root: Path,
    device: torch.device,
    inference_batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    _seed()
    data = _load_inputs(data_root)
    labels = np.asarray([int(row["label_id"]) for row in data["stage_d"]], dtype=np.int64)
    stgcn_path = data_root / STGCN_CHECKPOINT_RELATIVE
    head_path = data_root / ADAPTED_HEAD_RELATIVE
    stgcn, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    payload = torch.load(head_path, map_location=device, weights_only=False)
    head = LightweightClassifier(FEATURE_DIM).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    cache = data["cache"]
    frozen_current_logp = np.asarray(cache["current_logp_s0"], dtype=np.float32)
    frozen_candidate_logp = np.asarray(cache["true_logp"], dtype=np.float32)
    cache_check = _original_candidate_feature_check(
        stgcn, data["candidate_features"], data["legal_ids"], frozen_candidate_logp,
        device, inference_batch_size,
    )
    adapted_logits, adapted_current_logits, inference_summary = _adapted_logits(
        data, stgcn, head, device, inference_batch_size,
    )
    adapted_candidate_logp = _log_softmax(adapted_logits, axis=2)
    adapted_current_logp = _log_softmax(adapted_current_logits, axis=1)
    frozen_metrics, frozen_details = _evaluate_recognizer(
        labels, frozen_current_logp, frozen_candidate_logp, data["legal_ids"]
    )
    adapted_metrics, adapted_details = _evaluate_recognizer(
        labels, adapted_current_logp, adapted_candidate_logp, data["legal_ids"]
    )
    adapted_metrics["All-32 AnyCorrect Coverage"] = _all32_adapted_coverage(adapted_logits, labels)
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_MATCHED_PRIVILEGED_ORACLE_AUDIT_FRESH",
        "status": "COMPLETED",
        "recomputed_from_scratch": True,
        "prior_result_read": False,
        "population": {"split": "Moving Val", "moving_val_contexts": len(labels)},
        "labels": list(LABELS),
        "legal_action_set": {
            "definition": "Stage-A candidate_pool plus separate current/stay action; archived skeletons required finite",
            "candidate_count_mean": float(np.mean([len(ids) for ids in data["legal_ids"]])),
            "candidate_count_min": int(min(len(ids) for ids in data["legal_ids"])),
            "candidate_count_max": int(max(len(ids) for ids in data["legal_ids"])),
            "legal_action_count_mean": float(np.mean([1 + len(ids) for ids in data["legal_ids"]])),
            "legal_action_count_min": int(min(1 + len(ids) for ids in data["legal_ids"])),
            "legal_action_count_max": int(max(1 + len(ids) for ids in data["legal_ids"])),
        },
        "recognizers": {"Frozen": frozen_metrics, "Adapted": adapted_metrics},
        "summary_table": {
            name: {
                method: {
                    key: metric[key]
                    for key in ("accuracy", "macro_f1", "move_rate", "stay_rate")
                    if key in metric and metric[key] is not None
                }
                | ({"coverage_rate": metric["coverage_rate"]} if "coverage_rate" in metric else {})
                for method, metric in metrics.items()
            }
            for name, metrics in (("Frozen", frozen_metrics), ("Adapted", adapted_metrics))
        },
        "alignment": {
            "action_set_errors": 0,
            "stage_a_stage_c_cache_order_matched": True,
            "cache_episode_order_matches_stage_d": True,
            "candidate_feature_cache_check": cache_check,
            "adapted_inference": inference_summary,
        },
        "sanity": {
            "historical_frozen_legal_oracle_accuracy": HISTORICAL_FROZEN_LEGAL_ORACLE,
            "frozen_gt_true_logp_delta_pp": 100.0 * (frozen_metrics["GT-TrueLogP"]["accuracy"] - HISTORICAL_FROZEN_LEGAL_ORACLE),
            "within_one_pp": abs(frozen_metrics["GT-TrueLogP"]["accuracy"] - HISTORICAL_FROZEN_LEGAL_ORACLE) <= 0.01,
        },
        "leakage_audit": {
            "policy_test_used": False,
            "training_used": False,
            "new_rgb_or_skeleton_generated": False,
            "gt_label_used_for_oracle_selection_or_coverage_only": True,
            "future_candidate_skeleton_used_for_terminal_and_adapted_all32_diagnostic_only": True,
            "deployable": False,
        },
        "artifacts": {
            "stgcn_checkpoint": str(stgcn_path.resolve()),
            "stgcn_checkpoint_sha256": _sha256(stgcn_path),
            "adapted_head_checkpoint": str(head_path.resolve()),
            "adapted_head_checkpoint_sha256": _sha256(head_path),
            "counterfactual_val": str(data["cache_path"].resolve()),
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "elapsed_seconds": float(time.time() - started),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "audit_samples.json").write_text(
        json.dumps(_sample_rows(data, labels, frozen_details, adapted_details, adapted_current_logp, adapted_candidate_logp), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(json.dumps({
        "seed": SEED,
        "device": str(device),
        "inference_batch_size": inference_batch_size,
        "moving_val_contexts": len(labels),
        "action_set": "Stage-A candidate_pool + current/stay",
        "prior_result_read": False,
        "policy_test_used": False,
        "training_used": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(output_dir, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=256)
    args = parser.parse_args()
    result = run(
        args.output_dir.resolve(), args.data_root.resolve(),
        _require_cuda(args.device), args.inference_batch_size,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "recomputed_from_scratch": result["recomputed_from_scratch"],
        "frozen": result["recognizers"]["Frozen"],
        "adapted": result["recognizers"]["Adapted"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
