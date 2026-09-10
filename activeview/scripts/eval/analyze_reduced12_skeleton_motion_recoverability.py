#!/usr/bin/env python3
"""Val-only audit of estimated-skeleton temporal recoverability."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.scripts.eval.analyze_reduced12_same_azimuth_radius import (
    LABELS,
    _artifact_map,
    _candidate_records,
    _load_npz,
    _metadata_views,
    _metrics,
)

OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/skeleton_motion_recoverability_audit"
ARCHIVE_REL = "datasets/offline/habitat-train/00006-00087"
TRUE_CACHE_REL = "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"
SCENE_VIS = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz"
HUMAN_VIS = REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_view_observability_oracle/candidate_metrics.npz"
QUAL_MANIFEST = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization/case_manifest.json"
REGRET_RESULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_utility_regret_audit/result.json"
JOINT_COUNT = 17
FRAME_COUNT = 30
EPS = 1.0e-8
EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)
LR_PAIRS = ((13, 16), (12, 15), (5, 2), (6, 3))
QUALITY_DIRECTION = {
    "mean_joint_motion": 1.0,
    "max_joint_motion": 1.0,
    "top4_joint_motion_mean": 1.0,
    "total_motion_energy": 1.0,
    "mean_velocity_norm": 1.0,
    "mean_acceleration_norm": -1.0,
    "mean_jerk_norm": -1.0,
    "p95_acceleration": -1.0,
    "p95_jerk": -1.0,
    "velocity_smoothness": -1.0,
    "acceleration_smoothness": -1.0,
    "mean_bone_cv": -1.0,
    "max_bone_cv": -1.0,
    "bone_length_temporal_error": -1.0,
    "lr_distance_instability": -1.0,
    "mean_body_extent": 1.0,
    "body_extent_cv": -1.0,
    "min_body_extent": 1.0,
    "motion_retention": 1.0,
    "motion_deviation": -1.0,
    "SceneVisibility": 1.0,
    "HumanObservability": 1.0,
    "ProjectedArea": 1.0,
    "PoseConfidence": 1.0,
    "Distance": -1.0,
}
TEMPORAL_CUES = tuple(QUALITY_DIRECTION.keys())[:20]


def _rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _corr(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.size != y.size or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0, 0.0
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = _rank(x), _rank(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > EPS and np.std(ry) > EPS else 0.0
    return pearson, spearman


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {"count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)), "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90))}


def _skeleton_xyz(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, FRAME_COUNT, JOINT_COUNT):
        array = np.transpose(array, (1, 2, 0))
    elif array.shape != (FRAME_COUNT, JOINT_COUNT, 3):
        raise ValueError(f"unexpected archived skeleton shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("archived skeleton contains non-finite values")
    return array


def _temporal_metrics(value: np.ndarray) -> dict[str, Any]:
    skeleton = _skeleton_xyz(value)
    velocity = np.diff(skeleton, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    velocity_norm = np.linalg.norm(velocity, axis=2)
    acceleration_norm = np.linalg.norm(acceleration, axis=2)
    jerk_norm = np.linalg.norm(jerk, axis=2)
    motion_amp = velocity_norm.mean(axis=0)
    bone_length = np.stack([np.linalg.norm(skeleton[:, left] - skeleton[:, right], axis=1) for left, right in EDGES], axis=1)
    bone_mean = bone_length.mean(axis=0)
    bone_cv = bone_length.std(axis=0) / (bone_mean + EPS)
    bone_median = np.median(bone_length, axis=0)
    left_right = np.stack([np.linalg.norm(skeleton[:, left] - skeleton[:, right], axis=1) for left, right in LR_PAIRS], axis=1)
    pelvis = skeleton[:, 0:1]
    body_extent = np.linalg.norm(skeleton - pelvis, axis=2).max(axis=1)
    bbox_extent = np.linalg.norm(np.ptp(skeleton, axis=1), axis=1)
    return {
        "motion_amp": motion_amp,
        "mean_joint_motion": float(np.mean(motion_amp)),
        "max_joint_motion": float(np.max(motion_amp)),
        "top4_joint_motion_mean": float(np.mean(np.sort(motion_amp)[-4:])),
        "total_motion_energy": float(np.mean(velocity_norm ** 2)),
        "mean_velocity_norm": float(np.mean(velocity_norm)),
        "mean_acceleration_norm": float(np.mean(acceleration_norm)),
        "mean_jerk_norm": float(np.mean(jerk_norm)),
        "p95_acceleration": float(np.percentile(acceleration_norm, 95)),
        "p95_jerk": float(np.percentile(jerk_norm, 95)),
        "velocity_smoothness": float(np.mean(np.linalg.norm(np.diff(velocity, axis=0), axis=2))),
        "acceleration_smoothness": float(np.mean(np.linalg.norm(np.diff(acceleration, axis=0), axis=2))),
        "mean_bone_cv": float(np.mean(bone_cv)),
        "max_bone_cv": float(np.max(bone_cv)),
        "bone_length_temporal_error": float(np.mean(np.abs(bone_length - bone_median[None, :]))),
        "lr_distance_instability": float(np.mean(np.std(left_right, axis=0))),
        "mean_body_extent": float(np.mean(body_extent)),
        "body_extent_cv": float(np.std(body_extent) / (np.mean(body_extent) + EPS)),
        "min_body_extent": float(np.min(body_extent)),
        "mean_bbox3d_extent": float(np.mean(bbox_extent)),
        "_skeleton": skeleton,
    }


def _archive_path(archive_root: Path, record: Mapping[str, Any]) -> Path:
    path = archive_root / str(record["scene_id"]) / str(record["placement_id"]) / f"{record['record_id']}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _attach_temporal(records: list[dict[str, Any]], archive_root: Path, human_map: Mapping[str, Mapping[int, float]]) -> None:
    for record in records:
        archive_path = _archive_path(archive_root, record)
        with np.load(archive_path, allow_pickle=False) as archive:
            skeletons = np.asarray(archive["skeleton"])
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        if skeletons.shape != (32, 3, FRAME_COUNT, JOINT_COUNT):
            raise ValueError(f"unexpected archive skeleton schema: {archive_path} {skeletons.shape}")
        by_id = {int(viewpoint): index for index, viewpoint in enumerate(viewpoint_ids.tolist())}
        visibility = human_map.get(str(record["episode_id"]), {})
        for candidate in record["candidates"]:
            candidate_id = int(candidate["candidate_id"])
            if candidate_id not in by_id:
                raise ValueError(f"archive viewpoint missing: {archive_path} {candidate_id}")
            metrics = _temporal_metrics(skeletons[by_id[candidate_id]])
            metrics["HumanObservability"] = float(visibility.get(candidate_id, float("nan")))
            candidate.update(metrics)
        amplitudes = np.stack([candidate["motion_amp"] for candidate in record["candidates"]], axis=0)
        reference = np.median(amplitudes, axis=0)
        for candidate in record["candidates"]:
            candidate["motion_retention"] = float(np.mean(np.minimum(candidate["motion_amp"] / (reference + EPS), 2.0)))
            candidate["motion_deviation"] = float(np.mean(np.abs(candidate["motion_amp"] - reference) / (reference + EPS)))


def _add_predictions(records: Sequence[Mapping[str, Any]], moving_rows: Sequence[Mapping[str, Any]]) -> None:
    for record, row in zip(records, moving_rows):
        s0, s1 = np.asarray(row["s0_feature"], dtype=np.float64), np.asarray(row["s1_feature"], dtype=np.float64)
        record["s0_prediction"] = int(np.argmax(s0[256:268]))
        record["frozen_prediction"] = int(np.argmax(s1[256:268]))


def _candidate_value(candidate: Mapping[str, Any], name: str) -> float:
    if name == "SceneVisibility":
        return float(candidate["scene_visibility"])
    if name == "ProjectedArea":
        return float(candidate["projected_area"])
    if name == "PoseConfidence":
        return float(candidate["pose_confidence"])
    if name == "Distance":
        return float(candidate["distance_m"])
    return float(candidate[name])


def _flatten(records: Sequence[Mapping[str, Any]], name: str) -> tuple[np.ndarray, np.ndarray]:
    values, utilities = [], []
    for record in records:
        for candidate in record["candidates"]:
            value = _candidate_value(candidate, name)
            if np.isfinite(value):
                values.append(value)
                utilities.append(float(candidate["gt_margin"]))
    return np.asarray(values, dtype=np.float64), np.asarray(utilities, dtype=np.float64)


def _correlation_table(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, direction in QUALITY_DIRECTION.items():
        values, utility = _flatten(records, name)
        quality = values * direction
        pearson, spearman = _corr(quality, utility)
        raw_pearson, raw_spearman = _corr(values, utility)
        output[name] = {"count": int(values.size), "direction": "higher_is_better" if direction > 0 else "lower_is_better", "pearson_quality_vs_gt_margin": pearson, "spearman_quality_vs_gt_margin": spearman, "pearson_raw_vs_gt_margin": raw_pearson, "spearman_raw_vs_gt_margin": raw_spearman}
    return output


def _context_ranking(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, direction in QUALITY_DIRECTION.items():
        correlations = []
        for record in records:
            candidates = [candidate for candidate in record["candidates"] if np.isfinite(_candidate_value(candidate, name))]
            if len(candidates) < 2:
                continue
            values = np.asarray([_candidate_value(candidate, name) * direction for candidate in candidates])
            utility = np.asarray([float(candidate["gt_margin"]) for candidate in candidates])
            correlations.append(_corr(values, utility)[1])
        array = np.asarray(correlations, dtype=np.float64)
        output[name] = {"contexts": int(array.size), "mean": float(np.mean(array)) if array.size else 0.0, "median": float(np.median(array)) if array.size else 0.0, "fraction_positive": float(np.mean(array > 0.0)) if array.size else 0.0, "fraction_above_0.3": float(np.mean(array > 0.3)) if array.size else 0.0}
    return output


def _selector_metrics(records: Sequence[Mapping[str, Any]], selector: str, seed: int = 42) -> dict[str, Any]:
    labels, predictions, selected_values = [], [], []
    rng = np.random.default_rng(seed)
    fallback_count = 0
    for record in records:
        label = int(record["label"])
        labels.append(label)
        candidates = list(record["candidates"])
        if selector == "S0-only":
            predictions.append(int(record["s0_prediction"]))
            continue
        if selector == "FrozenStageCv0":
            predictions.append(int(record["frozen_prediction"]))
            continue
        if selector == "RandomMove":
            candidate = candidates[int(rng.integers(0, len(candidates)))] if candidates else None
        else:
            name = {"MaxMotionRetention": "motion_retention", "MinBoneInstability": "mean_bone_cv", "MinJerk": "mean_jerk_norm", "MaxMotionEnergy": "total_motion_energy", "SceneVisibility": "SceneVisibility", "ProjectedArea": "ProjectedArea", "PoseConfidence": "PoseConfidence"}[selector]
            direction = QUALITY_DIRECTION[name]
            candidate = max(candidates, key=lambda item: (_candidate_value(item, name) * direction, -int(item["candidate_id"]))) if candidates else None
        if candidate is None:
            fallback_count += 1
            predictions.append(int(record["s0_prediction"]))
        else:
            predictions.append(int(candidate["prediction"]))
            selected_values.append(float(candidate["gt_margin"]))
    metric = _metrics(predictions, labels)
    metric.update({"selector": selector, "move_rate": float((len(records) - fallback_count) / max(len(records), 1)), "fallback_to_s0": fallback_count, "mean_selected_gt_margin": float(np.mean(selected_values)) if selected_values else 0.0})
    return metric


def _any_correct(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    predictions = []
    for record in records:
        if int(record["s0_prediction"]) == int(record["label"]) or any(candidate["correct"] for candidate in record["candidates"]):
            predictions.append(int(record["label"]))
        else:
            predictions.append(int(record["s0_prediction"]))
    return _metrics(predictions, labels)


def _selected_ids() -> dict[str, int | None]:
    if not REGRET_RESULT.is_file():
        return {}
    payload = json.loads(REGRET_RESULT.read_text(encoding="utf-8"))
    rows = load_jsonl(get_data_root().resolve() / "datasets/policy_reduced12_eight_placement_v1/stage_c/features/val.jsonl")
    moving_ids = {row["episode_id"] for row in load_jsonl(get_data_root().resolve() / "datasets/policy_reduced12_eight_placement_v1/stage_d/features/val.jsonl")}
    ordered = [row for row in rows if row["episode_id"] in moving_ids]
    selected = payload["metrics"]["Candidate-Conditioned Spatial"]["selected_action_ids"]
    return {str(row["episode_id"]): (None if value is None else int(value)) for row, value in zip(ordered, selected)}


def _pair_summary(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    output = {"pair_count": len(rows), "cues": {}}
    for cue in ("mean_joint_motion", "total_motion_energy", "mean_bone_cv", "mean_jerk_norm", "body_extent_cv", "motion_retention"):
        differences = np.asarray([float(row[right][cue]) - float(row[left][cue]) for row in rows], dtype=np.float64)
        lower_is_better = cue in {"mean_bone_cv", "mean_jerk_norm", "body_extent_cv"}
        output["cues"][cue] = {"mean_difference_right_minus_left": float(np.mean(differences)) if differences.size else 0.0, "median_difference_right_minus_left": float(np.median(differences)) if differences.size else 0.0, "right_better_fraction": float(np.mean(differences < 0.0 if lower_is_better else differences > 0.0)) if differences.size else 0.0}
    return output


def _selected_vs_oracle(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_map = _selected_ids()
    all_rows, failure_rows, severe_rows = [], [], []
    excluded_stay = 0
    for record in records:
        oracle = max(record["candidates"], key=lambda item: (float(item["gt_margin"]), -int(item["candidate_id"])))
        selected_id = selected_map.get(str(record["episode_id"]))
        selected = next((item for item in record["candidates"] if item["candidate_id"] == selected_id), None)
        if selected is None:
            excluded_stay += 1
            continue
        row = {"episode_id": record["episode_id"], "selected_id": selected_id, "oracle_id": oracle["candidate_id"], "selected": selected, "oracle": oracle}
        all_rows.append(row)
        if oracle["correct"] and not selected["correct"]:
            failure_rows.append(row)
            if float(selected["gt_margin"]) < -1.0:
                severe_rows.append(row)
    def pack(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {"pairs": len(rows), "summary": _pair_summary(rows, "selected", "oracle"), "oracle_correct_selector_wrong": sum(bool(row["oracle"]["correct"] and not row["selected"]["correct"]) for row in rows)}
    return {"selected_mapping_available": bool(selected_map), "all_move_pairs": pack(all_rows), "oracle_correct_selector_wrong": pack(failure_rows), "severe_miss": pack(severe_rows), "excluded_stay_or_missing": excluded_stay}


def _same_azimuth_pairs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for record in records:
        grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
        for candidate in record["candidates"]:
            grouped[float(candidate["azimuth"])].append(candidate)
        for azimuth, candidates in grouped.items():
            for first, second in combinations(candidates, 2):
                if bool(first["correct"]) == bool(second["correct"]):
                    continue
                wrong, right = (first, second) if not first["correct"] else (second, first)
                pairs.append({"episode_id": record["episode_id"], "azimuth": azimuth, "wrong": wrong, "correct": right})
    summary = _pair_summary(pairs, "wrong", "correct")
    summary["wrong_to_correct_pairs"] = len(pairs)
    summary["examples"] = [{"episode_id": row["episode_id"], "azimuth": row["azimuth"], "wrong_id": row["wrong"]["candidate_id"], "correct_id": row["correct"]["candidate_id"], "wrong_radius": row["wrong"]["radius"], "correct_radius": row["correct"]["radius"], "wrong_margin": row["wrong"]["gt_margin"], "correct_margin": row["correct"]["gt_margin"]} for row in pairs[:100]]
    return summary


def _high_low(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    high, low = defaultdict(list), defaultdict(list)
    for record in records:
        candidates = sorted(record["candidates"], key=lambda item: float(item["gt_margin"]), reverse=True)
        count = max(1, int(math.ceil(0.2 * len(candidates))))
        for bucket, selected in ((high, candidates[:count]), (low, candidates[-count:])):
            for candidate in selected:
                for cue in TEMPORAL_CUES:
                    value = _candidate_value(candidate, cue)
                    if np.isfinite(value):
                        bucket[cue].append(value)
    output = {"top20_percent_count": int(sum(len(value) for value in high.values()) / max(len(TEMPORAL_CUES), 1)), "bottom20_percent_count": int(sum(len(value) for value in low.values()) / max(len(TEMPORAL_CUES), 1)), "metrics": {}}
    for cue in TEMPORAL_CUES:
        output["metrics"][cue] = {"high_utility": _summary(high[cue]), "low_utility": _summary(low[cue]), "high_minus_low_mean": float(np.mean(high[cue]) - np.mean(low[cue])) if high[cue] and low[cue] else 0.0}
    return output


def _case_pairs(records: Sequence[Mapping[str, Any]], archive_root: Path) -> list[dict[str, Any]]:
    if not QUAL_MANIFEST.is_file():
        return []
    by_episode = {str(record["episode_id"]): record for record in records}
    cases = json.loads(QUAL_MANIFEST.read_text(encoding="utf-8"))
    result: list[dict[str, Any]] = []
    for case in cases:
        figure = case.get("figure_record", {})
        selected_id = case.get("selected_id") if case.get("selected_id") is not None else figure.get("selected_viewpoint_id")
        oracle_id = case.get("oracle_id") if case.get("oracle_id") is not None else figure.get("oracle_viewpoint_id")
        if selected_id is None or oracle_id is None or abs(float(figure.get("azimuth_difference_deg", 999.0))) > 1.0e-6:
            continue
        if int(case["selected_pred"]) == int(case["label"]) or int(case["oracle_pred"]) != int(case["label"]):
            continue
        episode_id = str(case["episode_id"])
        if episode_id not in by_episode:
            continue
        record = by_episode[episode_id]
        path = _archive_path(archive_root, record)
        with np.load(path, allow_pickle=False) as archive:
            skeletons = np.asarray(archive["skeleton"])
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        slots = {int(viewpoint): index for index, viewpoint in enumerate(viewpoint_ids.tolist())}
        if int(selected_id) not in slots or int(oracle_id) not in slots:
            continue
        views = _metadata_views(archive_root, str(record["scene_id"]), str(record["placement_id"]))
        selected = _temporal_metrics(skeletons[slots[int(selected_id)]])
        oracle = _temporal_metrics(skeletons[slots[int(oracle_id)]])
        reference = np.median(np.stack([candidate["motion_amp"] for candidate in record["candidates"]], axis=0), axis=0)
        for metric in (selected, oracle):
            metric["motion_retention"] = float(np.mean(np.minimum(metric["motion_amp"] / (reference + EPS), 2.0)))
            metric["motion_deviation"] = float(np.mean(np.abs(metric["motion_amp"] - reference) / (reference + EPS)))
        selected.update({"radius": float(views[int(selected_id)]["radius_m"]), "azimuth": float(views[int(selected_id)]["azimuth_deg"]), "gt_margin": float(case["selected_margin"])})
        oracle.update({"radius": float(views[int(oracle_id)]["radius_m"]), "azimuth": float(views[int(oracle_id)]["azimuth_deg"]), "gt_margin": float(case["oracle_margin"])})
        result.append({"case_id": case["case_id"], "episode_id": episode_id, "action": LABELS[int(case["label"])], "selected_id": int(selected_id), "oracle_id": int(oracle_id), "selected": selected, "oracle": oracle, "rgb_path": str((REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization/qualitative_rgb" / f"{case['case_id']}.npz").resolve())})
    return result


def _render_examples(cases: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    selected_cases = list(cases[:6])
    if not selected_cases:
        fig, axis = plt.subplots(figsize=(8, 2))
        axis.text(0.5, 0.5, "No existing qualitative RGB case met same-azimuth wrong→correct criteria.", ha="center", va="center")
        axis.axis("off")
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    fig = plt.figure(figsize=(24, 4.5 * len(selected_cases)))
    for row_index, case in enumerate(selected_cases):
        with np.load(case["rgb_path"], allow_pickle=False) as archive:
            rgb = np.asarray(archive["rgb"])
            viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            frame_ids = np.asarray(archive["frame_ids"], dtype=np.int64)
        frame_slot = int(np.flatnonzero(frame_ids == 15)[0]) if np.any(frame_ids == 15) else len(frame_ids) // 2
        for side, key, title in ((0, "selected", "Selected / wrong"), (1, "oracle", "GT-best / correct")):
            base = row_index * 10 + side * 5
            ax_rgb = fig.add_subplot(len(selected_cases), 10, base + 1)
            viewpoint = int(case[f"{key}_id"])
            rgb_slot = int(np.flatnonzero(viewpoint_ids == viewpoint)[0])
            ax_rgb.imshow(np.transpose(rgb[rgb_slot, frame_slot], (1, 2, 0)) if rgb[rgb_slot, frame_slot].shape[0] == 3 else rgb[rgb_slot, frame_slot])
            metric = case[key]
            ax_rgb.set_title(f"{case['case_id']} {title}\n{case['action']}  r={metric['radius']:.1f}m  margin={metric['gt_margin']:.2f}")
            ax_rgb.axis("off")
            skeleton = metric["_skeleton"]
            for offset, frame in enumerate((0, 15, 29), start=2):
                axis = fig.add_subplot(len(selected_cases), 10, base + offset, projection="3d")
                points = skeleton[frame]
                for left, right in EDGES:
                    axis.plot(points[[left, right], 0], points[[left, right], 1], points[[left, right], 2], color="tab:blue", linewidth=1.0)
                axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, color="tab:red")
                axis.set_title(f"t={frame}", fontsize=8)
                axis.set_xticks([]); axis.set_yticks([]); axis.set_zticks([])
            info = fig.add_subplot(len(selected_cases), 10, base + 5)
            info.axis("off")
            info.text(0.0, 0.85, f"motion={metric['mean_joint_motion']:.3f}\nenergy={metric['total_motion_energy']:.3f}\nbone_cv={metric['mean_bone_cv']:.3f}\njerk={metric['mean_jerk_norm']:.3f}\nretention={metric.get('motion_retention', float('nan')):.3f}", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "_skeleton" and key != "motion_amp"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run(data_root: Path, output: Path) -> dict[str, Any]:
    policy = data_root / "datasets/policy_reduced12_eight_placement_v1"
    stage_c_rows = load_jsonl(policy / "stage_c/features/val.jsonl")
    moving_rows = load_jsonl(policy / "stage_d/features/val.jsonl")
    if any("test" in str(path).lower() for path in (policy / "stage_c/features/val.jsonl", policy / "stage_d/features/val.jsonl")):
        raise ValueError("Val-only audit received a Test path")
    true_cache_path = data_root / TRUE_CACHE_REL
    if "test" in str(true_cache_path).lower():
        raise ValueError("Test cache is forbidden")
    true_cache = _load_npz(true_cache_path)
    scene_map = _artifact_map(SCENE_VIS, "scene_visibility_score")
    human_map = _artifact_map(HUMAN_VIS, "human_observability")
    projected_map = _artifact_map(HUMAN_VIS, "projected_area")
    archive_root = data_root / ARCHIVE_REL
    records = _candidate_records(data_root, moving_rows, stage_c_rows, true_cache, scene_map, projected_map, archive_root)
    _add_predictions(records, moving_rows)
    _attach_temporal(records, archive_root, human_map)
    output.mkdir(parents=True, exist_ok=True)
    candidate_arrays: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        for candidate in record["candidates"]:
            candidate_arrays["episode_id"].append(str(record["episode_id"]))
            candidate_arrays["candidate_id"].append(int(candidate["candidate_id"]))
            for name in ("gt_margin", "gt_logp", "scene_visibility", "HumanObservability", "projected_area", "pose_confidence", "distance_m", *TEMPORAL_CUES):
                candidate_arrays[name].append(_candidate_value(candidate, name))
            candidate_arrays["motion_amp"].append(candidate["motion_amp"])
    np.savez_compressed(output / "candidate_metrics.npz", episode_id=np.asarray(candidate_arrays["episode_id"]), candidate_id=np.asarray(candidate_arrays["candidate_id"], dtype=np.int16), **{name: np.asarray(values) for name, values in candidate_arrays.items() if name not in {"episode_id", "candidate_id"}})
    correlations = _correlation_table(records)
    context_ranking = _context_ranking(records)
    selectors = {name: _selector_metrics(records, name) for name in ("S0-only", "FrozenStageCv0", "RandomMove", "SceneVisibility", "MaxMotionRetention", "MinBoneInstability", "MinJerk", "MaxMotionEnergy", "ProjectedArea", "PoseConfidence")}
    selectors["AnyCorrect Oracle"] = _any_correct(records)
    selectors["Candidate-Conditioned Spatial"] = {"count": 10080, "accuracy": 0.471528, "macro_f1": 0.463723, "source": "existing frozen Val audit; not rerun"}
    selected_pairs = _selected_vs_oracle(records)
    same_azimuth = _same_azimuth_pairs(records)
    high_low = _high_low(records)
    qualitative = _case_pairs(records, archive_root)
    _render_examples(qualitative, output / "paired_examples.png")
    result = {"experiment_id": "REDUCED12_SKELETON_MOTION_RECOVERABILITY_AUDIT", "status": "COMPLETED", "population": {"split": "val_moving", "moving_contexts": len(records), "candidate_samples": int(sum(len(record["candidates"]) for record in records)), "scene_count": len({record["scene_id"] for record in records})}, "labels": list(LABELS), "temporal_cue_metrics": correlations, "ranking_correlations": context_ranking, "selected_vs_oracle_pairs": selected_pairs, "same_azimuth_pairs": same_azimuth, "high_low_utility_comparison": high_low, "single_cue_selector_metrics": selectors, "qualitative_same_azimuth_cases": [{key: _json_safe(value) for key, value in case.items() if key not in {"selected", "oracle"}} for case in qualitative], "protocol": {"test_used": False, "training_used": False, "new_rgb_rendered": False, "new_skeleton_generated": False, "gt_action_used_for_posthoc_diagnostic_only": True, "estimated_skeleton_temporal_cues_used": True, "gt_motion_fidelity_available": False, "gt_motion_fidelity_reason": "No verified exact GT/estimated canonical alignment; no MPJPE or fidelity claim was computed.", "motion_retention_proxy_is_not_gt_motion_fidelity": True, "deployable": False, "candidate_set": "legal move candidates only"}, "artifacts": {"archive_root": str(archive_root.resolve()), "true_logp_cache": str(true_cache_path.resolve()), "scene_visibility": str(SCENE_VIS.resolve()), "human_observability": str(HUMAN_VIS.resolve())}}
    for filename, payload in (("result.json", result), ("temporal_cue_metrics.json", correlations), ("ranking_correlations.json", context_ranking), ("selected_vs_oracle_pairs.json", selected_pairs), ("same_azimuth_pairs.json", same_azimuth), ("high_low_utility_comparison.json", high_low), ("single_cue_selector_metrics.json", selectors)):
        (output / filename).write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output / "analysis.md")
    return result


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    cues = result["temporal_cue_metrics"]
    ranking = result["ranking_correlations"]
    selectors = result["single_cue_selector_metrics"]
    pairs = result["selected_vs_oracle_pairs"]
    same = result["same_azimuth_pairs"]
    reference_names = ("SceneVisibility", "ProjectedArea", "Distance", "PoseConfidence")
    candidate_cues = [name for name in cues if name in TEMPORAL_CUES]
    best_cue = max(candidate_cues, key=lambda name: float(ranking[name]["mean"])) if candidate_cues else "none"
    best_ref = max(reference_names, key=lambda name: float(ranking[name]["mean"]))
    temporal_vs_ref = float(ranking[best_cue]["mean"] - ranking[best_ref]["mean"]) if best_cue != "none" else 0.0
    instability_keys = {"mean_bone_cv", "mean_jerk_norm", "body_extent_cv"}
    fail = pairs["oracle_correct_selector_wrong"]
    fail_summary = fail["summary"]["cues"]
    stability_improved = all(float(fail_summary[name]["right_better_fraction"]) >= 0.5 for name in ("mean_bone_cv", "mean_jerk_norm", "body_extent_cv")) if fail["pairs"] else False
    retention_improved = float(fail_summary["motion_retention"]["right_better_fraction"]) >= 0.5 if fail["pairs"] else False
    high = result["high_low_utility_comparison"]["metrics"]
    artifact_warning = any(float(high[name]["high_minus_low_mean"]) > 0.0 for name in ("mean_bone_cv", "mean_jerk_norm", "body_extent_cv"))
    if artifact_warning:
        explanation = "D. ST-GCN/perception artifact exploitation remains a concern: some high-utility candidates are less temporally stable, so utility cannot be equated with physical quality."
    elif temporal_vs_ref >= 0.05 and max(float(ranking[best_cue]["mean"]), 0.0) >= 0.2 and any(float(selectors[name]["accuracy"]) > float(selectors["SceneVisibility"]["accuracy"]) for name in ("MaxMotionRetention", "MinBoneInstability", "MinJerk", "MaxMotionEnergy")):
        explanation = "C. pose reconstruction / temporal motion recoverability is the strongest supported low-dimensional explanation."
    elif float(ranking["Distance"]["mean"]) > float(ranking[best_cue]["mean"]):
        explanation = "B. image-scale / distance remains stronger than the tested temporal cues."
    elif float(ranking["SceneVisibility"]["mean"]) > float(ranking[best_cue]["mean"]):
        explanation = "A. visibility / geometry remains stronger than the tested temporal cues."
    else:
        explanation = "E. no single low-dimensional cue explains candidate utility robustly."
    lines = ["# Skeleton Motion Recoverability / Perception-Quality Audit", "", f"Val Moving only: {result['population']['moving_contexts']} contexts, {result['population']['candidate_samples']} legal move candidates, {result['population']['scene_count']} scenes. Test was not read; no training or perception regeneration was performed.", "", "## Candidate-level quality correlations", "", "| Cue | Quality Spearman vs GT-margin | Context mean Spearman | Context median | P(context rho>0) | P(rho>0.3) |", "|---|---:|---:|---:|---:|---:|"]
    for name in (*TEMPORAL_CUES, "SceneVisibility", "HumanObservability", "ProjectedArea", "PoseConfidence", "Distance"):
        lines.append(f"| {name} | {cues[name]['spearman_quality_vs_gt_margin']:.6f} | {ranking[name]['mean']:.6f} | {ranking[name]['median']:.6f} | {ranking[name]['fraction_positive']:.6f} | {ranking[name]['fraction_above_0.3']:.6f} |")
    lines.extend(["", "## Single-cue selector diagnostic", "", "| Selector | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|"])
    for name in selectors:
        lines.append(f"| {name} | {selectors[name]['accuracy']:.6f} | {selectors[name]['macro_f1']:.6f} | {selectors[name].get('move_rate', 'existing') } |")
    same_bone = float(same["cues"]["mean_bone_cv"]["right_better_fraction"])
    same_jerk = float(same["cues"]["mean_jerk_norm"]["right_better_fraction"])
    same_extent = float(same["cues"]["body_extent_cv"]["right_better_fraction"])
    same_pattern = same_bone >= 0.6 and same_jerk >= 0.6 and same_extent >= 0.6
    lines.extend(["", "## Selected vs GT-best paired audit", "", f"Move pairs available: {pairs['all_move_pairs']['pairs']}; oracle-correct/selector-wrong pairs: {pairs['oracle_correct_selector_wrong']['pairs']}; selected stay/missing excluded: {pairs['excluded_stay_or_missing']}.", f"On oracle-correct/selector-wrong pairs, GT-best has lower bone instability/jerk/body extent CV on all three cues: {stability_improved}; higher motion retention: {retention_improved}.", "", "## Same-azimuth wrong→correct pairs", "", f"Pairs: {same['wrong_to_correct_pairs']}. Correct-minus-wrong cue summaries are in `same_azimuth_pairs.json`; this is a paired diagnostic, not a deployable selector.", "", "## High/low utility artifact check", "", f"High-utility means top 20% within each context; low-utility means bottom 20%. Any high-utility increase in instability cues: {artifact_warning}.", "", "## Scientific answers", "", f"Q1. GT-best skeleton stability: {'usually improved' if stability_improved else 'not consistently improved'} on the oracle-correct/selector-wrong subset.", f"Q2. GT-best temporal motion retention: {'usually higher' if retention_improved else 'not consistently higher'}; this is a cross-view proxy only.", f"Q3. Same-azimuth wrong→correct flips show the requested skeleton-quality improvement pattern: {'yes' if same_pattern else 'not clearly'} (correct-better fractions: bone_cv={same_bone:.3f}, jerk={same_jerk:.3f}, body_extent_cv={same_extent:.3f}).", f"Q4. Best temporal context-ranking cue: {best_cue} (mean Spearman {ranking[best_cue]['mean']:.6f}); strongest reference cue: {best_ref} (mean Spearman {ranking[best_ref]['mean']:.6f}); temporal-minus-reference={temporal_vs_ref:.6f}.", f"Q5. Most supported explanation: {explanation}", "", "Motion-retention proxy is not GT motion fidelity. No MPJPE, velocity error, or acceleration error was computed because exact GT/estimated canonical alignment was not verified.", "", "Flags: `test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_skeleton_generated=false`; `gt_action_used_for_posthoc_diagnostic_only=true`; `estimated_skeleton_temporal_cues_used=true`; `gt_motion_fidelity_used=false`; `deployable=false."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.data_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "population": result["population"], "test_used": False}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
