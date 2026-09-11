#!/usr/bin/env python3
"""Measure all-view headroom for the trained reduced12 single-view head.

This is a Moving-Val-only, single-view audit.  Every archived 32-view
skeleton is passed through the frozen ST-GCN encoder and the already trained
lightweight classifier head.  No candidate selector, sequential fusion,
policy Test, or perception generation is used.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.experiments.run_reduced12_single_view_classifier_adaptation import (
    BATCH_SIZE,
    CHECKPOINT_RELATIVE,
    FEATURE_DIM,
    HEAD_RUNTIME_RELATIVE,
    LABELS,
    NUM_CLASSES,
    LightweightClassifier,
    _classification,
    _evaluate_head,
    _load_split,
)


SEED = 42
NUM_VIEWPOINTS = 32
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "adapted_single_view_headroom_audit"
)
OFFLINE_RELATIVE = Path("datasets/offline/habitat-train/00006-00087")
POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
REFERENCE_S1_ACCURACY = 0.5310515873015873
REFERENCE_S1_MACRO_F1 = 0.5560586763323812
HISTORICAL_OLD_FROZEN_ACCURACY = 0.454266


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _require_cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.clip(values.sum(axis=-1, keepdims=True), 1e-12, None)


def _viewpoint_name(viewpoint_id: int) -> str:
    return f"r{int(viewpoint_id) // 8 + 1}_a{int(viewpoint_id) % 8}"


def _load_all_view_predictions(
    val: Mapping[str, Any],
    data_root: Path,
    stgcn_checkpoint: Path,
    head_checkpoint: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the frozen encoder/head over every archived view in Val."""
    stgcn, _ = load_checkpoint(stgcn_checkpoint, NUM_CLASSES, str(device))
    payload = torch.load(head_checkpoint, map_location=device, weights_only=False)
    head = LightweightClassifier(FEATURE_DIM).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    labels = np.asarray(val["labels"], dtype=np.int64)
    predictions = np.empty((len(labels), NUM_VIEWPOINTS), dtype=np.int16)
    logits = np.empty((len(labels), NUM_VIEWPOINTS, NUM_CLASSES), dtype=np.float32)
    navigation_counts: list[int] = []
    skeleton_count = 0
    s1_feature_max_error = 0.0
    s1_logp_max_error = 0.0
    for row_index, row in enumerate(val["rows"]):
        archive = (
            data_root / OFFLINE_RELATIVE / str(row["scene_id"]) /
            str(row["region"]) / f"{row['record_id']}.npz"
        )
        if not archive.exists():
            raise FileNotFoundError(f"missing 32-view archive: {archive}")
        with np.load(archive, allow_pickle=False) as loaded:
            skeletons = np.asarray(loaded["skeleton"], dtype=np.float32)
            viewpoint_ids = np.asarray(loaded["viewpoint_ids"], dtype=np.int64)
            navigable = np.asarray(loaded["viewpoint_is_navigable"], dtype=bool)
        if skeletons.shape != (NUM_VIEWPOINTS, 3, 30, 17):
            raise ValueError(f"unexpected archive skeleton shape {skeletons.shape}: {archive}")
        if viewpoint_ids.shape != (NUM_VIEWPOINTS,) or not np.array_equal(
            viewpoint_ids, np.arange(NUM_VIEWPOINTS, dtype=np.int64)
        ):
            raise ValueError(f"archive viewpoint ids are not the canonical 0..31 order: {archive}")
        if navigable.shape != (NUM_VIEWPOINTS,):
            raise ValueError(f"unexpected navigation mask shape {navigable.shape}: {archive}")
        if not np.isfinite(skeletons).all():
            raise ValueError(f"non-finite skeleton in {archive}")
        navigation_counts.append(int(navigable.sum()))
        skeleton_count += NUM_VIEWPOINTS
        view_logits: list[np.ndarray] = []
        for start in range(0, NUM_VIEWPOINTS, BATCH_SIZE):
            batch = torch.from_numpy(skeletons[start : start + BATCH_SIZE]).to(
                device, non_blocking=True
            )
            with torch.inference_mode():
                features = stgcn.forward_features(batch)
                head_logits = head(features)
                view_logits.append(head_logits.cpu().numpy())
                s1_id = int(row["s1_viewpoint_id"])
                if start <= s1_id < start + len(batch):
                    local = s1_id - start
                    fresh_feature = features[local].cpu().numpy()
                    fresh_logp = torch.log_softmax(
                        stgcn.fc(features[local]), dim=0
                    ).cpu().numpy()
                    s1_feature_max_error = max(
                        s1_feature_max_error,
                        float(np.max(np.abs(fresh_feature - val["features"][row_index]))),
                    )
                    s1_logp_max_error = max(
                        s1_logp_max_error,
                        float(np.max(np.abs(fresh_logp - val["cached_logp"][row_index]))),
                    )
        row_logits = np.concatenate(view_logits, axis=0).astype(np.float32)
        logits[row_index] = row_logits
        predictions[row_index] = np.argmax(row_logits, axis=1).astype(np.int16)
        if (row_index + 1) % 500 == 0:
            print(
                f"[headroom-data] evaluated {row_index + 1}/{len(labels)} archives",
                flush=True,
            )
    return predictions, logits, labels, {
        "archives": len(val["rows"]),
        "skeletons_evaluated": skeleton_count,
        "viewpoints_per_archive": NUM_VIEWPOINTS,
        "navigable_viewpoints_mean": float(np.mean(navigation_counts)),
        "navigable_viewpoints_min": int(np.min(navigation_counts)),
        "navigable_viewpoints_max": int(np.max(navigation_counts)),
        "s1_feature_max_abs_error_vs_cache": s1_feature_max_error,
        "s1_logp_max_abs_error_vs_cache": s1_logp_max_error,
    }


def _selected_metrics(
    name: str,
    selected: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    selected_predictions = predictions[np.arange(len(labels)), selected]
    result = _classification(labels, selected_predictions)
    result.update({"method": name, "move_rate": 1.0})
    return result


def _fixed_view_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return {
        str(viewpoint_id): {
            "viewpoint_id": viewpoint_id,
            "viewpoint": _viewpoint_name(viewpoint_id),
            **_classification(labels, predictions[:, viewpoint_id]),
        }
        for viewpoint_id in range(NUM_VIEWPOINTS)
    }


def _best_single_oracle(
    predictions: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    correct = predictions == labels[:, None]
    selected = np.empty(len(labels), dtype=np.int64)
    any_correct = correct.any(axis=1)
    selected[any_correct] = np.argmax(correct[any_correct], axis=1)
    if np.any(~any_correct):
        missing = np.flatnonzero(~any_correct)
        probabilities = _softmax(logits[missing])
        true_prob = probabilities[np.arange(len(missing)), :, labels[missing]]
        # The specified fallback is the highest adapted-head true-class
        # probability; argmax supplies a deterministic viewpoint-id tie break.
        selected[missing] = np.argmax(true_prob, axis=1)
    return selected, any_correct


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    s1 = methods["Stage-D s1"]
    random_view = methods["Random single view"]
    best_single = methods["BestSingle Oracle"]
    any_correct = result["any_correct_oracle"]
    mean_fixed = result["fixed_view_summary"]["mean_accuracy"]
    s1_gap = s1["accuracy"] - mean_fixed
    oracle_gap = best_single["accuracy"] - s1["accuracy"]
    any_gap = any_correct["rate"] - s1["accuracy"]
    if s1_gap >= 0.05:
        specialization = "明显：s1 比 32-view 固定视角均值高至少 5pp，存在 viewpoint-distribution specialization。"
    elif s1_gap >= 0.02:
        specialization = "中等：s1 比 32-view 固定视角均值高 2–5pp，可能存在一定 specialization。"
    else:
        specialization = "不明显：s1 与 32-view 固定视角均值差异小于 2pp。"
    if any_correct["rate"] >= 0.70:
        decision = "Oracle 仍≥70%，recognizer 提升后仍保留较大 viewpoint headroom，主动选视角值得继续。"
    elif any_correct["rate"] >= 0.60:
        decision = "Oracle 约60–70%，viewpoint headroom 中等，应谨慎评估主动选视角的主课题价值。"
    else:
        decision = "Oracle<60%，若 32-view 泛化正常，则 recognizer 已吃掉大部分旧 headroom。"
    lines = [
        "# Adapted Recognizer Single-View Headroom Audit",
        "",
        "Moving Val only. The frozen reduced12 ST-GCN encoder and the already trained single-view classifier head were applied independently to all 32 archived viewpoints per context. All 32 archive entries were retained exactly as requested; navigation metadata was audited but not used as a filter. No active selector, sequential fusion, policy Test, or new perception was used.",
        "",
        "## Matched Stage-D s1",
        "",
        f"Stage-D s1: Accuracy={s1['accuracy']:.6f}, Macro-F1={s1['macro_f1']:.6f}; expected 0.531052/0.556059. The baseline gate passed before interpreting all-view headroom.",
        f"Archive-recomputed s1: Accuracy={result['methods']['Archive-recomputed s1']['accuracy']:.6f}, Macro-F1={result['methods']['Archive-recomputed s1']['macro_f1']:.6f}; cache-vs-archive prediction mismatches={result['archive_summary']['s1_prediction_mismatch_vs_cache']}, maximum feature/logp errors={result['archive_summary']['s1_feature_max_abs_error_vs_cache']:.3e}/{result['archive_summary']['s1_logp_max_abs_error_vs_cache']:.3e}.",
        f"Random single view: Accuracy={random_view['accuracy']:.6f}, Macro-F1={random_view['macro_f1']:.6f}.",
        "",
        "## Headroom summary",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| Stage-D s1 | {s1['accuracy']:.6f} | {s1['macro_f1']:.6f} |",
        f"| Random single view | {random_view['accuracy']:.6f} | {random_view['macro_f1']:.6f} |",
        f"| BestSingle Oracle | {best_single['accuracy']:.6f} | {best_single['macro_f1']:.6f} |",
        f"| AnyCorrect Oracle (rate) | {any_correct['rate']:.6f} | — |",
        "",
        f"BestSingle − Stage-D s1: {oracle_gap * 100:+.3f}pp Accuracy. AnyCorrect − Stage-D s1: {any_gap * 100:+.3f}pp.",
        f"Historical frozen-recognizer references (FrozenStageCv0≈45.43%, AnyCorrect≈72.83%) are shown only for context and are not mixed with this adapted-head protocol.",
        "",
        "## Fixed-view generalization",
        "",
        f"Across fixed lattice viewpoints, mean Accuracy/Macro-F1={mean_fixed:.6f}/{result['fixed_view_summary']['mean_macro_f1']:.6f}, best={result['fixed_view_summary']['best_viewpoint']} ({result['fixed_view_summary']['best_accuracy']:.6f}), worst={result['fixed_view_summary']['worst_viewpoint']} ({result['fixed_view_summary']['worst_accuracy']:.6f}), Accuracy range={result['fixed_view_summary']['range_accuracy']:.6f}.",
        f"{specialization}",
        "",
        "## Decision",
        "",
        decision,
        "The oracle is computed from the adapted head's own 32-view predictions; a high value indicates remaining single-view selection headroom, while a low value should only be interpreted as recognizer saturation when viewpoint generalization is normal.",
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "training_used=false",
        "new_rgb_generated=false",
        "new_skeleton_generated=false",
        "single_view_only=true",
        "frozen_stgcn_modified=false",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.time()
    _seed()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_root = data_root / POLICY_RELATIVE
    val = _load_split(policy_root / "stage_d/features/val.jsonl")
    if len(val["labels"]) != 10080:
        raise ValueError(f"expected 10080 Moving Val contexts, found {len(val['labels'])}")
    stgcn_checkpoint = data_root / CHECKPOINT_RELATIVE
    head_checkpoint = data_root / HEAD_RUNTIME_RELATIVE
    if not stgcn_checkpoint.exists():
        raise FileNotFoundError(f"missing frozen ST-GCN checkpoint: {stgcn_checkpoint}")
    if not head_checkpoint.exists():
        raise FileNotFoundError(f"missing adapted head checkpoint: {head_checkpoint}")

    predictions, logits, labels, archive_summary = _load_all_view_predictions(
        val, data_root, stgcn_checkpoint, head_checkpoint, device
    )
    random_generator = np.random.default_rng(SEED)
    random_selected = random_generator.integers(0, NUM_VIEWPOINTS, size=len(labels))
    s1_indices = np.asarray([int(row["s1_viewpoint_id"]) for row in val["rows"]], dtype=np.int64)
    cached_head = LightweightClassifier(FEATURE_DIM).to(device)
    cached_payload = torch.load(head_checkpoint, map_location=device, weights_only=False)
    cached_head.load_state_dict(cached_payload["state_dict"])
    cached_s1_metrics, cached_head_logits = _evaluate_head(
        cached_head, val["features"], labels, device
    )
    archive_s1_metrics = _selected_metrics(
        "Archive-recomputed s1", s1_indices, predictions, labels
    )
    # The protocol baseline is the learned head on cached Stage-D s1 features.
    # The archive recomputation is reported separately because tiny numerical
    # differences can flip a classifier decision near a logit boundary.
    s1_metrics = dict(cached_s1_metrics)
    s1_metrics["method"] = "Stage-D s1"
    s1_metrics["move_rate"] = 1.0
    archive_summary["s1_prediction_mismatch_vs_cache"] = int(np.sum(
        predictions[np.arange(len(labels)), s1_indices]
        != np.argmax(cached_head_logits, axis=1)
    ))
    random_metrics = _selected_metrics("Random single view", random_selected, predictions, labels)
    fixed = _fixed_view_metrics(predictions, labels)
    best_single_selected, any_correct = _best_single_oracle(predictions, logits, labels)
    best_single_metrics = _selected_metrics(
        "BestSingle Oracle", best_single_selected, predictions, labels
    )
    any_correct_rate = float(np.mean(any_correct))
    if abs(s1_metrics["accuracy"] - REFERENCE_S1_ACCURACY) > 1e-9:
        raise RuntimeError(f"adapted s1 Accuracy gate failed: {s1_metrics['accuracy']}")
    if abs(s1_metrics["macro_f1"] - REFERENCE_S1_MACRO_F1) > 1e-9:
        raise RuntimeError(f"adapted s1 Macro-F1 gate failed: {s1_metrics['macro_f1']}")

    fixed_items = list(fixed.values())
    best_fixed = max(fixed_items, key=lambda item: (item["accuracy"], -item["viewpoint_id"]))
    worst_fixed = min(fixed_items, key=lambda item: (item["accuracy"], item["viewpoint_id"]))
    mean_fixed_accuracy = float(np.mean([item["accuracy"] for item in fixed_items]))
    mean_fixed_macro_f1 = float(np.mean([item["macro_f1"] for item in fixed_items]))
    fixed_summary = {
        "mean_accuracy": mean_fixed_accuracy,
        "mean_macro_f1": mean_fixed_macro_f1,
        "best_viewpoint": best_fixed["viewpoint"],
        "best_viewpoint_id": best_fixed["viewpoint_id"],
        "best_accuracy": best_fixed["accuracy"],
        "best_macro_f1": best_fixed["macro_f1"],
        "worst_viewpoint": worst_fixed["viewpoint"],
        "worst_viewpoint_id": worst_fixed["viewpoint_id"],
        "worst_accuracy": worst_fixed["accuracy"],
        "worst_macro_f1": worst_fixed["macro_f1"],
        "range_accuracy": float(best_fixed["accuracy"] - worst_fixed["accuracy"]),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_ADAPTED_SINGLE_VIEW_HEADROOM_AUDIT",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.time() - started,
        },
        "protocol": {
            "split": "Stage-D Moving Val",
            "contexts": len(labels),
            "viewpoints_per_context": NUM_VIEWPOINTS,
            "viewpoint_lattice": "4 radii x 8 azimuths; archive viewpoint_id 0..31",
            "navigation_filter": "none; all 32 archived viewpoints evaluated",
            "matched_viewpoint": "Stage-D s1_viewpoint_id",
            "stgcn_checkpoint": str(stgcn_checkpoint.resolve()),
            "classifier_checkpoint": str(head_checkpoint.resolve()),
            "feature_dim": FEATURE_DIM,
            "terminal_evaluation": "adapted head prediction on selected real archived skeleton",
            "historical_old_frozen_stagecv0_accuracy": HISTORICAL_OLD_FROZEN_ACCURACY,
        },
        "archive_summary": archive_summary,
        "methods": {
            "Stage-D s1": s1_metrics,
            "Random single view": random_metrics,
            "BestSingle Oracle": best_single_metrics,
            "Archive-recomputed s1": archive_s1_metrics,
        },
        "fixed_view_metrics": fixed,
        "fixed_view_summary": fixed_summary,
        "any_correct_oracle": {
            "rate": any_correct_rate,
            "count": int(np.sum(any_correct)),
            "contexts": len(labels),
            "historical_old_frozen_reference_rate": 0.728274,
        },
        "gaps": {
            "best_single_minus_stage_d_s1_accuracy": float(
                best_single_metrics["accuracy"] - s1_metrics["accuracy"]
            ),
            "any_correct_minus_stage_d_s1_accuracy": float(
                any_correct_rate - s1_metrics["accuracy"]
            ),
            "best_single_minus_stage_d_s1_macro_f1": float(
                best_single_metrics["macro_f1"] - s1_metrics["macro_f1"]
            ),
        },
        "leakage_audit": {
            "policy_test_used": False,
            "training_used": False,
            "future_observation_used_as_model_input": False,
            "future_skeleton_used_for_terminal_evaluation_only": True,
            "new_rgb_or_skeleton_generated": False,
            "candidate_selector_trained": False,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps({
            "seed": SEED,
            "contexts": len(labels),
            "viewpoints_per_context": NUM_VIEWPOINTS,
            "split": "Stage-D Moving Val",
            "classifier_checkpoint": str(head_checkpoint.resolve()),
            "stgcn_checkpoint": str(stgcn_checkpoint.resolve()),
            "policy_test_used": False,
            "training_used": False,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_view_metrics.json").write_text(
        json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(
        args.output_dir.resolve(), args.data_root.resolve(), _require_cuda(args.device)
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "contexts": result["protocol"]["contexts"],
        "stage_d_s1": result["methods"]["Stage-D s1"],
        "random": result["methods"]["Random single view"],
        "best_single": result["methods"]["BestSingle Oracle"],
        "any_correct_rate": result["any_correct_oracle"]["rate"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
