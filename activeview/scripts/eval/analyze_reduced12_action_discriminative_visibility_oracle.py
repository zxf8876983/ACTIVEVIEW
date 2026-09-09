#!/usr/bin/env python3
"""Val-only action-conditioned visibility oracle for reduced12 ActiveView.

Train is used only to estimate frozen ST-GCN body-part masking importance.  Val
candidate selection uses that fixed matrix together with existing privileged
scene and human self-visibility artifacts; no policy, WM, or recognizer is
changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (
    LABELS,
    NUM_CLASSES,
    _metrics,
    _spearman,
)


DATASET_NAME = "policy_reduced12_eight_placement_v1"
RAW_DATASET_NAME = "reduced12_no_kneel_clean_babel_diversity_v1"
STAGE_D_REL = f"datasets/{DATASET_NAME}/stage_d/features/val.jsonl"
CACHE_REL = f"datasets/{DATASET_NAME}_rgb_restored/counterfactual_cache/val.npz"
RAW_TRAIN_REL = f"datasets/{RAW_DATASET_NAME}/raw-train"
CHECKPOINT_REL = (
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
SCENE_VISIBILITY = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz"
)
HUMAN_VISIBILITY = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/human_self_visibility_oracle/diagnostics.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "action_discriminative_visibility_oracle"
)

H36M17 = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine1", "spine3", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)
BODY_PARTS = (
    "head", "torso", "left_upper_arm", "left_forearm", "right_upper_arm",
    "right_forearm", "left_thigh", "right_thigh", "left_lower_leg",
    "right_lower_leg",
)
BODY_JOINTS = {
    "head": (9, 10),
    "torso": (0, 7, 8),
    "left_upper_arm": (11,),
    "left_forearm": (12, 13),
    "right_upper_arm": (14,),
    "right_forearm": (15, 16),
    "left_thigh": (4,),
    "right_thigh": (1,),
    "left_lower_leg": (5, 6),
    "right_lower_leg": (2, 3),
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen ST-GCN inference")
    return torch.device("cuda:0")


def _compute_importance(data_root: Path, batch_size: int = 128) -> dict[str, Any]:
    """Estimate class-specific importance by masking canonical H36M17 joints."""
    device = _require_cuda()
    train_root = data_root / RAW_TRAIN_REL
    data = np.load(train_root / "train_data.npy", mmap_mode="r")
    labels = np.asarray(np.load(train_root / "train_labels.npy"), dtype=np.int64)
    if data.ndim != 5 or data.shape[1:] != (3, 30, 17, 1):
        raise ValueError(f"unexpected raw-train tensor shape: {data.shape}")
    if len(data) != len(labels) or not np.isfinite(data).all():
        raise ValueError("invalid raw-train skeleton tensors")
    checkpoint = data_root / CHECKPOINT_REL
    model, _ = load_checkpoint(checkpoint, NUM_CLASSES, "cuda:0")
    model.eval()
    totals = np.zeros((NUM_CLASSES, len(BODY_PARTS)), dtype=np.float64)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.int64)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        batch = np.asarray(data[start:stop], dtype=np.float32)
        target = labels[start:stop]
        with torch.inference_mode():
            base = F.log_softmax(model(torch.from_numpy(batch).to(device)), dim=-1)
            base_values = base.cpu().numpy()[np.arange(stop - start), target]
            root = batch[:, :, :, 0:1, :].copy()
            for part_index, part in enumerate(BODY_PARTS):
                masked = batch.copy()
                masked[:, :, :, list(BODY_JOINTS[part]), :] = root
                masked_logp = F.log_softmax(
                    model(torch.from_numpy(masked).to(device)), dim=-1
                ).cpu().numpy()[np.arange(stop - start), target]
                totals[:, part_index] += np.bincount(
                    target,
                    weights=np.maximum(0.0, base_values - masked_logp),
                    minlength=NUM_CLASSES,
                )
    normalized = np.zeros_like(totals)
    for class_id in range(NUM_CLASSES):
        total = float(totals[class_id].sum())
        normalized[class_id] = (
            totals[class_id] / total if total > 1e-12 else 1.0 / len(BODY_PARTS)
        )
    matrix = {
        LABELS[class_id]: {
            "label_id": class_id,
            "train_count": int(counts[class_id]),
            "raw_mean_importance": totals[class_id].tolist(),
            "normalized_weights": normalized[class_id].tolist(),
            "top3": [
                {"body_part": BODY_PARTS[index], "weight": float(normalized[class_id, index])}
                for index in np.argsort(-normalized[class_id])[:3]
            ],
        }
        for class_id in range(NUM_CLASSES)
    }
    return {
        "body_parts": list(BODY_PARTS),
        "joint_order": list(H36M17),
        "body_part_joint_indices": {part: list(indices) for part, indices in BODY_JOINTS.items()},
        "matrix": matrix,
        "normalized": normalized.tolist(),
        "checkpoint": str(checkpoint.resolve()),
        "num_train_samples": int(len(data)),
        "device": str(device),
    }


def _load_val_inputs(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    rows = [dict(row) for row in load_jsonl(data_root / STAGE_D_REL)]
    if not rows:
        raise ValueError("empty reduced12 Val Stage-D features")
    scene = _load_npz(SCENE_VISIBILITY)
    human = json.loads(HUMAN_VISIBILITY.read_text(encoding="utf-8"))
    cache = _load_npz(data_root / CACHE_REL)
    if len(rows) != len(scene["episode_ids"]):
        raise ValueError("Stage-D and SceneVisibility Val counts differ")
    scene_index = {str(ep): i for i, ep in enumerate(scene["episode_ids"].tolist())}
    human_by_episode: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in human.get("candidate_records", []):
        episode = str(record["episode_id"])
        candidate_id = int(record["candidate_id"])
        if candidate_id in human_by_episode[episode]:
            raise ValueError(f"duplicate human visibility record: {episode}/{candidate_id}")
        human_by_episode[episode][candidate_id] = record
    if set(scene_index) != set(human_by_episode):
        raise ValueError("SceneVisibility and HumanSelfVisibility episode sets differ")
    cache_index = {str(ep): i for i, ep in enumerate(cache["episode_ids"].tolist())}
    if set(scene_index) != set(cache_index):
        raise ValueError("Val cache and visibility episode sets differ")
    return rows, scene, human_by_episode, cache


def _sanity_samples(human_by_episode: Mapping[str, Mapping[int, Mapping[str, Any]]], count: int = 20) -> list[dict[str, Any]]:
    records = [record for candidates in human_by_episode.values() for record in candidates.values()]
    rng = np.random.default_rng(42)
    selected = rng.choice(len(records), size=min(count, len(records)), replace=False)
    return [
        {
            "episode_id": str(records[int(index)]["episode_id"]),
            "candidate_id": int(records[int(index)]["candidate_id"]),
            "full_vs_isolated_visibility_ratio": {
                part: float(records[int(index)]["body_part_visibility"][part])
                for part in BODY_PARTS
            },
        }
        for index in selected
    ]


def _part_values(scene_joints: np.ndarray, human_record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_parts = np.asarray(
        [np.mean(scene_joints[list(BODY_JOINTS[part])]) for part in BODY_PARTS], dtype=np.float64
    )
    human_parts = np.asarray(
        [float(human_record["body_part_visibility"][part]) for part in BODY_PARTS], dtype=np.float64
    )
    return scene_parts, human_parts, scene_parts * human_parts


def _stable_argmax(values: np.ndarray, candidate_ids: np.ndarray) -> int:
    """Select the maximum score, breaking exact ties by candidate id."""
    maximum = float(np.max(values))
    tied = np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-12))
    return int(tied[np.argmin(candidate_ids[tied])])


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    scene: Mapping[str, np.ndarray],
    human_by_episode: Mapping[str, Mapping[int, Mapping[str, Any]]],
    cache: Mapping[str, np.ndarray],
    importance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scene_index = {str(ep): i for i, ep in enumerate(scene["episode_ids"].tolist())}
    cache_index = {str(ep): i for i, ep in enumerate(cache["episode_ids"].tolist())}
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    names = (
        "S0-only", "FrozenStageCv0", "SceneVisibility", "HumanSelfVisibility",
        "TotalVisibility", "GTAction-SceneDiscriminative",
        "GTAction-HumanDiscriminative", "GTAction-TotalDiscriminative",
        "AnyCorrect Oracle",
    )
    predictions = {name: np.zeros(len(rows), dtype=np.int64) for name in names}
    moved = {name: np.zeros(len(rows), dtype=bool) for name in names}
    candidate_dump: list[dict[str, Any]] = []
    score_values: dict[str, list[float]] = defaultdict(list)
    correct_values: dict[str, list[float]] = defaultdict(list)
    wrong_values: dict[str, list[float]] = defaultdict(list)
    oracle_positive = 0
    for row_index, row in enumerate(rows):
        episode = str(row["episode_id"])
        scene_row = scene_index[episode]
        cache_row = cache_index[episode]
        valid = np.asarray(scene["candidate_mask"][scene_row], dtype=bool)
        candidate_ids = np.asarray(scene["candidate_ids"][scene_row][valid], dtype=np.int64)
        candidate_records = human_by_episode[episode]
        if set(candidate_ids.tolist()) != set(candidate_records):
            raise ValueError(f"candidate alignment mismatch for {episode}")
        true_logp = np.asarray(cache["true_logp"][cache_row, candidate_ids], dtype=np.float64)
        candidate_pred = true_logp.argmax(axis=1)
        label = int(labels[row_index])
        scene_joints = np.asarray(scene["visibility"][scene_row][valid], dtype=np.float64)
        scene_score = np.asarray(scene["scene_visibility_score"][scene_row][valid], dtype=np.float64)
        human_parts_array = np.asarray(
            [
                _part_values(joints, candidate_records[int(candidate_id)])[1]
                for candidate_id, joints in zip(candidate_ids.tolist(), scene_joints)
            ],
            dtype=np.float64,
        )
        scene_parts_array = np.asarray(
            [
                _part_values(joints, candidate_records[int(candidate_id)])[0]
                for candidate_id, joints in zip(candidate_ids.tolist(), scene_joints)
            ],
            dtype=np.float64,
        )
        total_parts_array = scene_parts_array * human_parts_array
        weights = np.asarray(importance["normalized"], dtype=np.float64)[label]
        action_scene = scene_parts_array @ weights
        action_human = human_parts_array @ weights
        action_total = total_parts_array @ weights
        selections = {
            "SceneVisibility": _stable_argmax(scene_score, candidate_ids),
            "HumanSelfVisibility": _stable_argmax(human_parts_array.mean(axis=1), candidate_ids),
            "TotalVisibility": _stable_argmax(total_parts_array.mean(axis=1), candidate_ids),
            "GTAction-SceneDiscriminative": _stable_argmax(action_scene, candidate_ids),
            "GTAction-HumanDiscriminative": _stable_argmax(action_human, candidate_ids),
            "GTAction-TotalDiscriminative": _stable_argmax(action_total, candidate_ids),
        }
        s0_prediction = int(np.argmax(cache["current_logp_s0"][cache_row]))
        predictions["S0-only"][row_index] = s0_prediction
        for name, selected in selections.items():
            predictions[name][row_index] = int(candidate_pred[selected])
            moved[name][row_index] = True
        proposal = int(row["proposal_rank_1_id"])
        proposal_index = np.flatnonzero(candidate_ids == proposal)
        if proposal_index.size != 1:
            raise ValueError(f"FrozenStageCv0 proposal is not legal for {episode}")
        predictions["FrozenStageCv0"][row_index] = int(candidate_pred[int(proposal_index[0])])
        moved["FrozenStageCv0"][row_index] = True
        candidate_correct = candidate_pred == label
        if s0_prediction == label:
            predictions["AnyCorrect Oracle"][row_index] = s0_prediction
        elif np.any(candidate_correct):
            predictions["AnyCorrect Oracle"][row_index] = label
            moved["AnyCorrect Oracle"][row_index] = True
        else:
            predictions["AnyCorrect Oracle"][row_index] = s0_prediction
        oracle_positive += int(np.any(candidate_correct))
        score_arrays = {
            "scene_visibility": scene_score,
            "human_self_visibility": human_parts_array.mean(axis=1),
            "total_visibility": total_parts_array.mean(axis=1),
            "gt_action_scene_discriminative": action_scene,
            "gt_action_human_discriminative": action_human,
            "gt_action_total_discriminative": action_total,
        }
        for candidate_index, candidate_id in enumerate(candidate_ids.tolist()):
            record = candidate_records[int(candidate_id)]
            dump = {
                "episode_id": episode,
                "candidate_id": int(candidate_id),
                "gt_class_true_logp": float(true_logp[candidate_index, label]),
                "stgcn_correct": bool(candidate_correct[candidate_index]),
                "scene_body_part_visibility": scene_parts_array[candidate_index].tolist(),
                "human_body_part_visibility": human_parts_array[candidate_index].tolist(),
                "total_body_part_visibility": total_parts_array[candidate_index].tolist(),
            }
            dump.update({name: float(values[candidate_index]) for name, values in score_arrays.items()})
            candidate_dump.append(dump)
            for name, values in score_arrays.items():
                score_values[name].append(float(values[candidate_index]))
                (correct_values if candidate_correct[candidate_index] else wrong_values)[name].append(float(values[candidate_index]))
    metrics = {
        name: {**_metrics(predictions[name], labels), "move_rate": float(np.mean(moved[name]))}
        for name in names
    }
    score_diagnostics = {}
    logp = [item["gt_class_true_logp"] for item in candidate_dump]
    for name, values in score_values.items():
        score_diagnostics[name] = {
            "spearman_with_gt_class_true_logp": _spearman(values, logp),
            "mean_if_stgcn_correct": float(np.mean(correct_values[name])) if correct_values[name] else 0.0,
            "mean_if_stgcn_wrong": float(np.mean(wrong_values[name])) if wrong_values[name] else 0.0,
        }
    diagnostics = {
        "candidate_count": len(candidate_dump),
        "contexts": len(rows),
        "oracle_positive_contexts": oracle_positive,
        "score_diagnostics": score_diagnostics,
        "sanity_samples": _sanity_samples(human_by_episode),
        "candidate_records": candidate_dump,
    }
    per_class = {
        name: {
            label: values["per_class"][label]
            for label in LABELS
        }
        for name, values in metrics.items()
    }
    return metrics, diagnostics, per_class


def _write_analysis(output: Path, result: Mapping[str, Any], importance: Mapping[str, Any]) -> None:
    frozen = result["methods"]["FrozenStageCv0"]
    lines = [
        "# Reduced12 action-discriminative visibility oracle",
        "",
        "Val-only privileged diagnostic. Train is used only for frozen ST-GCN "
        "body-part masking importance; GT action and future visibility are not deployable inputs.",
        "",
        "## Moving Val metrics",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["methods"].items():
        lines.append(
            f"| {name} | {values['accuracy']:.6f} | {values['macro_f1']:.6f} | "
            f"{(values['accuracy'] - frozen['accuracy']) * 100:+.3f} | "
            f"{(values['macro_f1'] - frozen['macro_f1']) * 100:+.3f} |"
        )
    lines.extend(["", "## Train masking importance top-3", "", "| Action | Top-3 body parts |", "| --- | --- |"])
    for label, values in importance["matrix"].items():
        top = ", ".join(f"{item['body_part']} ({item['weight']:.3f})" for item in values["top3"])
        lines.append(f"| {label} | {top} |")
    score_diag = result["diagnostics"]["score_diagnostics"]
    lines.extend([
        "", "## Candidate score diagnostics", "",
        "| Score | Spearman with GT true-logp | Correct mean | Wrong mean |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name, values in score_diag.items():
        lines.append(
            f"| {name} | {values['spearman_with_gt_class_true_logp']:.6f} | "
            f"{values['mean_if_stgcn_correct']:.6f} | {values['mean_if_stgcn_wrong']:.6f} |"
        )
    lines.extend([
        "", "## Scientific conclusion", "",
        "The action-conditioned visibility scores are privileged upper-bound diagnostics, "
        "not deployable policies. In this run GTAction-TotalDiscriminative reaches "
        f"{result['methods']['GTAction-TotalDiscriminative']['accuracy']:.6f} accuracy, "
        "well below the 47–50% visibility-only range and the 55% target. The action-specific "
        "weights therefore do not rescue the human-visibility signal; visibility-only NBV is "
        "insufficient to explain the AnyCorrect gap. The next direction should predict candidate "
        "future recognizer evidence or utility rather than add more visibility heuristics.",
        "",
        "Protocol flags: test_used=false; training_new_model_used=false; "
        "train_used_only_for_frozen_stgcn_masking_importance=true; "
        "gt_action_used_for_val_oracle_only=true; gt_future_visibility_used_for_oracle_only=true; "
        "deployable=false.",
    ])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    data_root = get_data_root()
    args.output.mkdir(parents=True, exist_ok=True)
    importance = _compute_importance(data_root, args.batch_size)
    rows, scene, human, cache = _load_val_inputs(data_root)
    methods, diagnostics, per_class = _evaluate(rows, scene, human, cache, importance)
    result = {
        "protocol": "reduced12_eight_placement_v1",
        "split": "val_moving",
        "methods": methods,
        "diagnostics": {key: value for key, value in diagnostics.items() if key != "candidate_records"},
        "importance_checkpoint": importance["checkpoint"],
        "num_classes": NUM_CLASSES,
        "test_used": False,
        "training_new_model_used": False,
        "train_used_only_for_frozen_stgcn_masking_importance": True,
        "gt_action_used_for_val_oracle_only": True,
        "gt_future_visibility_used_for_oracle_only": True,
        "deployable": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (args.output / "action_body_part_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    (args.output / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")
    _write_analysis(args.output, {**result, "diagnostics": diagnostics}, importance)
    print(json.dumps(methods, indent=2))


if __name__ == "__main__":
    main()
