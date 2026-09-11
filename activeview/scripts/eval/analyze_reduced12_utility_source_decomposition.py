#!/usr/bin/env python3
"""Decompose reduced12 candidate HAR utility on Moving Val.

This is a read-only audit of archived frozen-ST-GCN evidence.  It estimates
body-relative maps, crossed motion/scene consistency, and transparent
leave-one-group additive effects; it does not train or alter any policy.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.eval.reduced12_nbv_utils import LABELS, correlation, load_train_val
from activeview.scripts.eval.reduced12_utility_source import (
    AZIMUTH_COUNT,
    RADIUS_COUNT,
    build_utility_samples,
    flatten_cell_values,
    grouped_by,
    load_scene_metadata,
)


OUT = Path("experiments/reduced12_eight_placement_v1/utility_source_decomposition")
SEED = 42


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None}
    return {"mean": float(np.mean(array)), "median": float(np.median(array)), "p25": float(np.quantile(array, 0.25)), "p75": float(np.quantile(array, 0.75))}


def _pair_metrics(left: Mapping[str, Any], right: Mapping[str, Any], cells: Sequence[tuple[int, int]]) -> dict[str, float] | None:
    if len(cells) < 2:
        return None
    x = np.asarray([left["utility"][cell] for cell in cells], dtype=np.float64)
    y = np.asarray([right["utility"][cell] for cell in cells], dtype=np.float64)
    signs = np.sign(x) == np.sign(y)
    left_correct = {cell for cell in cells if left["correct"][cell]}
    right_correct = {cell for cell in cells if right["correct"][cell]}
    union = left_correct | right_correct
    jaccard = 1.0 if not union else float(len(left_correct & right_correct) / len(union))
    return {"pearson": correlation(x, y), "spearman": correlation(x, y, spearman=True), "sign_agreement": float(np.mean(signs)), "correct_jaccard": jaccard, "common_cells": float(len(cells))}


def _summarize_pair_entries(entries: Sequence[Mapping[str, Any]], attempted: int, *, pair_count_name: str = "usable_pair_count") -> dict[str, Any]:
    names = ("pearson", "spearman", "sign_agreement", "correct_jaccard", "common_cells")
    result: dict[str, Any] = {"attempted_pair_count": int(attempted), pair_count_name: int(len(entries))}
    for name in names:
        result[name] = _quantiles([float(item[name]) for item in entries])
    return result


def _pair_collection(
    samples: Sequence[Mapping[str, Any]], key_fn: Callable[[Mapping[str, Any]], Any], *, same_label: bool = False,
) -> tuple[dict[str, Any], dict[int, tuple[list[float], list[float]]], dict[int, tuple[list[float], list[float]]], dict[int, list[dict[str, Any]]]]:
    """Collect map-level pairs plus cheap flattened radius/azimuth conditionals."""
    groups = grouped_by(samples, key_fn)
    entries: list[dict[str, Any]] = []
    per_radius: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    per_azimuth: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    per_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    attempted = 0
    for values in groups.values():
        for left, right in combinations(values, 2):
            attempted += 1
            if same_label and int(left["label"]) != int(right["label"]):
                continue
            cells = sorted(set(left["utility"]) & set(right["utility"]))
            if len(cells) < 4:
                continue
            metric = _pair_metrics(left, right, cells)
            if metric is None:
                continue
            metric = dict(metric)
            metric["label"] = int(left["label"])
            entries.append(metric)
            for cell in cells:
                radius_x, radius_y = per_radius[cell[0]]
                azimuth_x, azimuth_y = per_azimuth[cell[1]]
                radius_x.append(float(left["utility"][cell])); radius_y.append(float(right["utility"][cell]))
                azimuth_x.append(float(left["utility"][cell])); azimuth_y.append(float(right["utility"][cell]))
            per_class[int(left["label"])].append(metric)
    return _summarize_pair_entries(entries, attempted), per_radius, per_azimuth, per_class


def _conditional_pair_summary(
    groups: Mapping[int, tuple[Sequence[float], Sequence[float]]], attempted: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, (left, right) in sorted(groups.items()):
        x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        result[str(key)] = {
            "attempted_pair_count": int(attempted),
            "candidate_cell_pairs": int(x.size),
            "pearson": correlation(x, y),
            "spearman": correlation(x, y, spearman=True),
            "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))) if x.size else None,
        }
    return result


def _matched_different_motion(
    samples: Sequence[Mapping[str, Any]], target_attempted: int,
) -> tuple[dict[str, Any], dict[int, tuple[list[float], list[float]]], dict[int, tuple[list[float], list[float]]], dict[int, list[dict[str, Any]]]]:
    """Use same scene/placement groups but different record IDs, seed 42."""
    groups = grouped_by(samples, lambda item: (item["scene_id"], item["placement_id"]))
    pair_specs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for values in groups.values():
        pair_specs.extend((left, right) for left, right in combinations(values, 2) if left["record_id"] != right["record_id"])
    rng = np.random.default_rng(SEED)
    if len(pair_specs) > target_attempted:
        selected = rng.choice(len(pair_specs), size=target_attempted, replace=False)
        pair_specs = [pair_specs[int(index)] for index in selected]
    entries: list[dict[str, Any]] = []
    per_radius: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    per_azimuth: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    per_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for left, right in pair_specs:
        cells = sorted(set(left["utility"]) & set(right["utility"]))
        if len(cells) < 4:
            continue
        metric = _pair_metrics(left, right, cells)
        if metric is None:
            continue
        metric = dict(metric)
        metric["label"] = int(left["label"])
        entries.append(metric)
        for cell in cells:
            radius_x, radius_y = per_radius[cell[0]]
            azimuth_x, azimuth_y = per_azimuth[cell[1]]
            radius_x.append(float(left["utility"][cell])); radius_y.append(float(right["utility"][cell]))
            azimuth_x.append(float(left["utility"][cell])); azimuth_y.append(float(right["utility"][cell]))
        per_class[int(left["label"])].append(metric)
    return _summarize_pair_entries(entries, len(pair_specs)), per_radius, per_azimuth, per_class


def _model_summary(targets: Sequence[float], predictions: Sequence[float]) -> dict[str, Any]:
    if not targets:
        return {"samples": 0, "explained_variance": None, "pearson": None, "spearman": None, "residual_variance_fraction": None}
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    centered = y - np.mean(y)
    total = float(np.sum(centered * centered))
    residual = float(np.sum((y - p) ** 2))
    return {"samples": int(y.size), "explained_variance": float(1.0 - residual / total) if total > 1e-12 else 0.0, "pearson": correlation(p, y), "spearman": correlation(p, y, spearman=True), "residual_variance_fraction": float(np.var(y - p) / np.var(y)) if np.var(y) > 1e-12 else 0.0}


def _additive_decomposition(samples: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Estimate M/S/MS predictions with leave-one-sample-out group means."""
    values = flatten_cell_values(samples)
    by_cell: dict[tuple[int, int], list[tuple[Mapping[str, Any], tuple[int, int], float, bool]]] = defaultdict(list)
    for item in values:
        by_cell[item[1]].append(item)
    rows: list[dict[str, Any]] = []
    collected: dict[str, tuple[list[float], list[float]]] = {name: ([], []) for name in ("motion_only", "scene_only", "motion_scene")}
    for cell, cell_values in by_cell.items():
        total_sum = sum(item[2] for item in cell_values)
        motion_groups: dict[str, list[float]] = defaultdict(list)
        scene_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for sample, _, value, _ in cell_values:
            motion_groups[str(sample["record_id"])].append(value)
            scene_groups[(str(sample["scene_id"]), str(sample["placement_id"]))].append(value)
        for sample, _, value, correct in cell_values:
            n = len(cell_values)
            global_mean = (total_sum - value) / (n - 1) if n > 1 else None
            motion_values = motion_groups[str(sample["record_id"])]
            scene_values = scene_groups[(str(sample["scene_id"]), str(sample["placement_id"]))]
            motion_mean = (sum(motion_values) - value) / (len(motion_values) - 1) if len(motion_values) > 1 else None
            scene_mean = (sum(scene_values) - value) / (len(scene_values) - 1) if len(scene_values) > 1 else None
            prediction_values = {
                "motion_only": motion_mean,
                "scene_only": scene_mean,
                "motion_scene": (motion_mean + scene_mean - global_mean) if motion_mean is not None and scene_mean is not None and global_mean is not None else None,
            }
            for name, prediction in prediction_values.items():
                if prediction is not None:
                    collected[name][0].append(value)
                    collected[name][1].append(float(prediction))
            rows.append({"cell": list(cell), "record_id": str(sample["record_id"]), "scene_id": str(sample["scene_id"]), "placement_id": str(sample["placement_id"]), "label": int(sample["label"]), "utility": float(value), "correct": bool(correct), "motion_only": motion_mean, "scene_only": scene_mean, "motion_scene": prediction_values["motion_scene"]})
    result = {name: _model_summary(targets, predictions) for name, (targets, predictions) in collected.items()}
    result["cell_count"] = len(by_cell)
    result["candidate_sample_count"] = len(values)
    return result, rows


def _conditional_decomposition(rows: Sequence[Mapping[str, Any]], field: str, count: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index in range(count):
        selected = [row for row in rows if int(row["cell"][0 if field == "radius" else 1]) == index]
        result[str(index)] = {name: _model_summary([float(row["utility"]) for row in selected if row[name] is not None], [float(row[name]) for row in selected if row[name] is not None]) for name in ("motion_only", "scene_only", "motion_scene")}
    return result


def _class_decomposition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, name in enumerate(LABELS):
        selected = [row for row in rows if int(row["label"]) == label]
        result[str(label)] = {"action": name, **{model: _model_summary([float(row["utility"]) for row in selected if row[model] is not None], [float(row[model]) for row in selected if row[model] is not None]) for model in ("motion_only", "scene_only", "motion_scene")}}
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    data = load_train_val()
    rows = data["stage_d_rows"]["val"]
    metadata, alignment_audit = load_scene_metadata(data["data_root"], rows)
    if not alignment_audit["convention_confirmed"]:
        raise RuntimeError("body-relative yaw convention was not confirmed")
    samples = build_utility_samples(data, metadata)
    if len(samples) != len(rows):
        raise ValueError(f"utility sample count mismatch: {len(samples)} != {len(rows)}")

    same_motion, motion_radius, motion_azimuth, motion_class = _pair_collection(samples, lambda item: item["record_id"])
    same_scene, scene_radius, scene_azimuth, scene_class = _pair_collection(samples, lambda item: (item["scene_id"], item["placement_id"]), same_label=False)
    same_scene_class, _, _, scene_class_same_label = _pair_collection(samples, lambda item: (item["scene_id"], item["placement_id"]), same_label=True)
    different_motion, diff_radius, diff_azimuth, diff_class = _matched_different_motion(samples, same_motion["attempted_pair_count"])
    additive, additive_rows = _additive_decomposition(samples)

    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "body_relative_alignment_audit.json", alignment_audit)
    _write_json(OUT / "same_motion_consistency.json", {"same_motion": same_motion, "different_motion_matched": different_motion, "per_radius": _conditional_pair_summary(motion_radius, same_motion["attempted_pair_count"]), "different_motion_per_radius": _conditional_pair_summary(diff_radius, different_motion["attempted_pair_count"]), "per_azimuth": _conditional_pair_summary(motion_azimuth, same_motion["attempted_pair_count"]), "different_motion_per_azimuth": _conditional_pair_summary(diff_azimuth, different_motion["attempted_pair_count"])})
    _write_json(OUT / "same_scene_consistency.json", {"same_scene_placement": same_scene, "same_scene_same_action": same_scene_class, "per_radius": _conditional_pair_summary(scene_radius, same_scene["attempted_pair_count"]), "per_azimuth": _conditional_pair_summary(scene_azimuth, same_scene["attempted_pair_count"])})
    _write_json(OUT / "additive_decomposition.json", additive)
    _write_json(OUT / "per_radius.json", {"same_motion": _conditional_pair_summary(motion_radius, same_motion["attempted_pair_count"]), "same_scene": _conditional_pair_summary(scene_radius, same_scene["attempted_pair_count"]), "motion_scene_decomposition": _conditional_decomposition(additive_rows, "radius", RADIUS_COUNT)})
    _write_json(OUT / "per_azimuth.json", {"same_motion": _conditional_pair_summary(motion_azimuth, same_motion["attempted_pair_count"]), "same_scene": _conditional_pair_summary(scene_azimuth, same_scene["attempted_pair_count"]), "motion_scene_decomposition": _conditional_decomposition(additive_rows, "azimuth", AZIMUTH_COUNT)})
    per_class: dict[str, Any] = {}
    for label, name in enumerate(LABELS):
        per_class[str(label)] = {"action": name, "same_motion": _summarize_pair_entries(motion_class.get(label, []), same_motion["attempted_pair_count"]), "same_scene_same_action": _summarize_pair_entries(scene_class_same_label.get(label, []), same_scene_class["attempted_pair_count"]), "different_motion": _summarize_pair_entries(diff_class.get(label, []), different_motion["attempted_pair_count"])}
        per_class[str(label)]["additive"] = _class_decomposition(additive_rows)[str(label)]
    _write_json(OUT / "per_class.json", per_class)

    result = {
        "experiment_id": "REDUCED12_UTILITY_SOURCE_DECOMPOSITION",
        "population": {"split": "val_moving", "contexts": len(samples), "candidate_samples": int(sum(len(item["candidates"]) for item in samples)), "scene_placement_groups": len(metadata), "records": len({item["record_id"] for item in samples})},
        "alignment": alignment_audit,
        "same_motion": same_motion,
        "different_motion_matched": different_motion,
        "same_scene_placement": same_scene,
        "same_scene_same_action": same_scene_class,
        "additive_decomposition": additive,
        "labels": list(LABELS),
        "protocol": {"map_shape": [RADIUS_COUNT, AZIMUTH_COUNT], "missing_cells": "NaN/omitted; no interpolation", "utility": "frozen ST-GCN GT-class margin", "pair_common_cells_minimum": 4, "different_motion_matching": "same scene/placement, seed 42"},
        "policy_test_used": False,
        "policy_val_used_for_diagnostic_only": True,
        "training_used": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "new_dino_generated": False,
        "gt_action_used_for_posthoc_grouping_only": True,
        "gt_margin_used_for_privileged_diagnostic_only": True,
        "future_candidate_observation_used_at_inference": False,
        "deployable": False,
    }
    _write_json(OUT / "result.json", result)
    lines = ["# Reduced12 candidate utility source decomposition", "", f"Moving Val contexts: {len(samples)}; candidate map samples: {additive['candidate_sample_count']}", "", "## Alignment", "", f"Body-relative convention confirmed: `{alignment_audit['convention_confirmed']}`; max position/azimuth error {alignment_audit['max_position_azimuth_error_deg']:.6f} deg.", "", "## Consistency", "", "| Comparison | Usable pairs | Spearman mean | Spearman median | Sign agreement mean | Correct-map Jaccard mean |", "|---|---:|---:|---:|---:|---:|"]
    comparisons = [("Same motion across scene/placement", same_motion), ("Different motion matched", different_motion), ("Same scene/placement", same_scene), ("Same scene/placement, same action", same_scene_class)]
    for name, item in comparisons:
        lines.append(f"| {name} | {item['usable_pair_count']} | {item['spearman']['mean'] if item['spearman']['mean'] is not None else float('nan'):.6f} | {item['spearman']['median'] if item['spearman']['median'] is not None else float('nan'):.6f} | {item['sign_agreement']['mean'] if item['sign_agreement']['mean'] is not None else float('nan'):.6f} | {item['correct_jaccard']['mean'] if item['correct_jaccard']['mean'] is not None else float('nan'):.6f} |")
    lines.extend(["", "## Leave-one-group additive decomposition", "", "| Model | Explained variance | Pearson | Spearman | Residual variance / total | Samples |", "|---|---:|---:|---:|---:|---:|"])
    for name, item in additive.items():
        if not isinstance(item, dict) or "explained_variance" not in item:
            continue
        lines.append(f"| {name} | {item['explained_variance'] if item['explained_variance'] is not None else float('nan'):.6f} | {item['pearson'] if item['pearson'] is not None else float('nan'):.6f} | {item['spearman'] if item['spearman'] is not None else float('nan'):.6f} | {item['residual_variance_fraction'] if item['residual_variance_fraction'] is not None else float('nan'):.6f} | {item['samples']} |")
    lines.extend(["", "## Interpretation", "", "The report separates motion consistency, scene/placement consistency, and additive motion+scene effects. Low additive explained variance with a large residual indicates motion×scene×viewpoint interaction; this audit does not launch a new policy.", "", "policy_test_used=false; training_used=false; future_candidate_observation_used_at_inference=false."])
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
