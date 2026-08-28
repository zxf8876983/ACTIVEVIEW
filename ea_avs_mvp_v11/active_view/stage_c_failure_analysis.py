"""Read-only failure analysis for frozen Stage C-v0 predictions.

The module intentionally consumes only serialized Stage A/B/C artifacts.  It
never writes to an upstream artifact and never invokes Habitat or a model.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


NEAR_ZERO = 1e-6
SMALL_REGRET = 1e-2
REQUIRED_SPLITS = ("val", "test")
GEOMETRY_NAMES = (
    "ego_x", "ego_y", "ego_z", "euclidean", "geodesic", "sin_azimuth",
    "cos_azimuth", "path_ratio", "current_radius", "candidate_radius",
    "delta_radius",
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    import json
    from pathlib import Path

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _distribution(values: Sequence[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": int(array.size), "mean": float(array.mean()), "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
        "min": float(array.min()), "max": float(array.max()),
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    for value in np.unique(array):
        indexes = np.flatnonzero(array == value)
        if len(indexes) > 1:
            ranks[indexes] = ranks[indexes].mean()
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_rank(x), _rank(y))


def _headroom(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float | int]:
    positive = [row for row in rows if float(row["safe_oracle_utility"]) > NEAR_ZERO]
    safe_sum = sum(float(row["safe_oracle_utility"]) for row in positive)
    selected_sum = sum(max(0.0, float(row["selected_true_utility"])) for row in positive)
    ratios = [float(row["selected_true_utility"]) / float(row["safe_oracle_utility"]) for row in positive]
    clipped = [float(np.clip(value, 0.0, 1.0)) for value in ratios]
    return {
        "positive_episode_count": len(positive),
        "positive_episode_ratio": _rate(len(positive), len(rows)),
        "clipped_mean": _mean(clipped),
        "aggregate_capture": float(selected_sum / safe_sum) if safe_sum else 0.0,
    }


def _candidate_geometry(row: Mapping[str, Any], viewpoint_id: int) -> Dict[str, float] | None:
    ids = [int(value) for value in row["candidate_viewpoint_ids"]]
    if viewpoint_id not in ids:
        return None
    geometry = np.asarray(row["candidate_geometry"], dtype=np.float64)
    values = geometry[ids.index(viewpoint_id)]
    if values.shape != (11,) or not np.isfinite(values).all():
        return None
    result = {name: float(value) for name, value in zip(GEOMETRY_NAMES, values)}
    result["abs_azimuth_deg"] = abs(math.degrees(math.atan2(result["sin_azimuth"], result["cos_azimuth"])))
    result["signed_azimuth_deg"] = math.degrees(math.atan2(result["sin_azimuth"], result["cos_azimuth"]))
    return result


def _class_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    no_correct = sum(int(row["current_predicted_label_id"]) == int(row["label_id"]) for row in rows)
    set_correct = sum(int(row["selected_predicted_label_id"]) == int(row["label_id"]) for row in rows)
    safe_correct = sum(int(row["safe_oracle_predicted_label_id"]) == int(row["label_id"]) for row in rows)
    regrets = [float(row["regret"]) for row in rows]
    return {
        "n": n,
        "NoMove_accuracy": _rate(no_correct, n),
        "Set_accuracy": _rate(set_correct, n),
        "SafeOracle_accuracy": _rate(safe_correct, n),
        "Set_gain_vs_NoMove": _rate(set_correct - no_correct, n),
        "regret": _distribution(regrets),
        "positive_headroom": _headroom(rows),
        "current_correct_ratio": _rate(no_correct, n),
    }


def _record_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = _class_metrics(rows)
    catastrophic = sum(bool(row["catastrophic_top5pct"]) for row in rows)
    metrics["episode_count"] = len(rows)
    metrics["catastrophic_ratio"] = _rate(catastrophic, len(rows))
    metrics["mean_regret"] = metrics["regret"]["mean"]
    return metrics


def _taxonomy(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        safe_move = not bool(row["safe_oracle_stays"])
        pred_move = not bool(row["predicted_stays"])
        same_action = str(row["predicted_action"]) == str(row["safe_oracle_action"])
        if same_action:
            name = "D_correct_safe_action"
        elif safe_move and not pred_move:
            name = "A_missed_move"
        elif not safe_move and pred_move:
            name = "B_unnecessary_move"
        elif safe_move and pred_move:
            name = "C1_wrong_near_optimal" if float(row["regret"]) <= SMALL_REGRET else "C2_wrong_high_utility_loss"
        else:
            name = "unclassified"
        groups[name].append(row)
    output: Dict[str, Any] = {}
    for name in ("A_missed_move", "B_unnecessary_move", "C1_wrong_near_optimal", "C2_wrong_high_utility_loss", "D_correct_safe_action", "unclassified"):
        members = groups.get(name, [])
        output[name] = {
            "count": len(members), "ratio": _rate(len(members), len(rows)),
            "regret": _distribution([float(row["regret"]) for row in members]),
            "safe_oracle_utility": _distribution([float(row["safe_oracle_utility"]) for row in members]),
            "selected_true_utility": _distribution([float(row["selected_true_utility"]) for row in members]),
            "degradation_count": sum(
                int(int(row["current_predicted_label_id"]) == int(row["label_id"]) and int(row["selected_predicted_label_id"]) != int(row["label_id"]))
                for row in members
            ),
        }
    return output


def _regret_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    regrets = [float(row["regret"]) for row in rows]
    thresholds = {"near_optimal_1e-3": 1e-3, "near_optimal_1e-2": 1e-2, "median": _percentile(regrets, 50), "p75": _percentile(regrets, 75), "p90": _percentile(regrets, 90), "p95": _percentile(regrets, 95), "p99": _percentile(regrets, 99)}
    conditions = {
        "G0_near_optimal": lambda value: value <= 1e-3,
        "G1_low_regret": lambda value: 1e-3 < value <= thresholds["p75"],
        "G2_moderate_regret": lambda value: thresholds["p75"] < value <= thresholds["p90"],
        "G3_high_regret": lambda value: value > thresholds["p90"],
        "G4_catastrophic_top5pct": lambda value: value >= thresholds["p95"],
        "G4_extreme_top1pct": lambda value: value >= thresholds["p99"],
    }
    output: Dict[str, Any] = {"thresholds": thresholds, "distribution": _distribution(regrets)}
    for name, condition in conditions.items():
        members = [row for row in rows if condition(float(row["regret"]))]
        output[name] = {"count": len(members), "ratio": _rate(len(members), len(rows)), "regret": _distribution([float(row["regret"]) for row in members])}
    for row in rows:
        value = float(row["regret"])
        if value <= 1e-3:
            row["regret_group"] = "G0_near_optimal"
        elif value <= thresholds["p75"]:
            row["regret_group"] = "G1_low_regret"
        elif value <= thresholds["p90"]:
            row["regret_group"] = "G2_moderate_regret"
        else:
            row["regret_group"] = "G3_high_regret"
        row["catastrophic_top5pct"] = value >= thresholds["p95"]
        row["extreme_top1pct"] = value >= thresholds["p99"]
    return output


def _miss_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    misses = [row for row in rows if int(row["predicted_candidate_viewpoint_id"]) != int(row["candidate_oracle_viewpoint_id"])]
    gaps = [float(row["candidate_oracle_utility"]) - float(row["selected_true_utility"]) for row in misses]
    ratios: Dict[str, float] = {}
    valid_ratios: List[float] = []
    for row in misses:
        oracle = float(row["candidate_oracle_utility"])
        if oracle > NEAR_ZERO:
            valid_ratios.append(float(row["selected_true_utility"]) / oracle)
    for threshold in (0.9, 0.75, 0.5):
        ratios[f"selected_at_least_{int(threshold * 100)}pct_oracle"] = _rate(sum(value >= threshold for value in valid_ratios), len(valid_ratios))
    return {"miss_count": len(misses), "miss_ratio": _rate(len(misses), len(rows)), "absolute_utility_gap": _distribution(gaps), "ratio_denominator_count": len(valid_ratios), "ratio_thresholds": ratios}


def _state_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for group in ("G0_near_optimal", "G1_low_regret", "G2_moderate_regret", "G3_high_regret"):
        members = [row for row in rows if row["regret_group"] == group]
        output[group] = {
            "count": len(members),
            "current_entropy": _distribution([float(row["current_entropy"]) for row in members]),
            "current_margin": _distribution([float(row["current_margin"]) for row in members]),
            "current_pose_confidence": _distribution([float(row["current_pose_confidence"]) for row in members]),
            "current_logp_true": _distribution([float(row["current_logp_true"]) for row in members]),
            "current_correct_ratio": _rate(sum(int(row["current_predicted_label_id"]) == int(row["label_id"]) for row in members), len(members)),
            "move_rate": _rate(sum(not bool(row["predicted_stays"]) for row in members), len(members)),
            "safe_oracle_move_rate": _rate(sum(not bool(row["safe_oracle_stays"]) for row in members), len(members)),
            "safe_oracle_utility": _distribution([float(row["safe_oracle_utility"]) for row in members]),
        }
    output["correlations_with_regret"] = {
        name: {"pearson": _pearson([float(row[name]) for row in rows], [float(row["regret"]) for row in rows]), "spearman": _spearman([float(row[name]) for row in rows], [float(row["regret"]) for row in rows])}
        for name in ("current_entropy", "current_margin", "current_pose_confidence", "current_logp_true")
    }
    for correct, name in ((True, "current_correct"), (False, "current_incorrect")):
        members = [row for row in rows if (int(row["current_predicted_label_id"]) == int(row["label_id"])) == correct]
        output[name] = {
            "count": len(members), "move_rate": _rate(sum(not bool(row["predicted_stays"]) for row in members), len(members)),
            "mean_regret": _mean([float(row["regret"]) for row in members]), "safe_headroom": _headroom(members),
        }
    return output


def _geometry_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    oracle_rows: List[Dict[str, float]] = []
    selected_rows: List[Dict[str, float]] = []
    for row in rows:
        oracle = _candidate_geometry(row, int(row["candidate_oracle_viewpoint_id"]))
        if oracle is not None:
            oracle_rows.append(oracle)
        if not bool(row["predicted_stays"]):
            selected = _candidate_geometry(row, int(row["predicted_candidate_viewpoint_id"]))
            if selected is not None:
                selected_rows.append(selected)

    def summarize(values: Sequence[Dict[str, float]]) -> Dict[str, Any]:
        return {key: _distribution([item[key] for item in values]) for key in ("abs_azimuth_deg", "geodesic", "euclidean", "current_radius", "candidate_radius", "delta_radius", "path_ratio")}

    azimuth_bins = [0, 30, 60, 90, 120, 150, 180.0001]
    def bins(values: Sequence[Dict[str, float]]) -> Dict[str, int]:
        counts = Counter()
        for item in values:
            index = min(len(azimuth_bins) - 2, int(np.searchsorted(azimuth_bins, item["abs_azimuth_deg"], side="right") - 1))
            counts[f"{azimuth_bins[index]:g}-{azimuth_bins[index + 1]:g}"] += 1
        return dict(counts)

    oracle_geodesics = [item["geodesic"] for item in oracle_rows]
    geo_edges = [
        min(oracle_geodesics) if oracle_geodesics else 0.0,
        _percentile(oracle_geodesics, 25),
        _percentile(oracle_geodesics, 50),
        _percentile(oracle_geodesics, 75),
        max(oracle_geodesics) if oracle_geodesics else 0.0,
    ]
    geo_bins: Dict[str, Dict[str, Any]] = {}
    for index, label in enumerate(("q0_q25", "q25_q50", "q50_q75", "q75_q100")):
        low, high = geo_edges[index], geo_edges[index + 1]
        members = [row for row in rows if (low <= float(_candidate_geometry(row, int(row["candidate_oracle_viewpoint_id"]))["geodesic"]) <= high if index == 3 else low <= float(_candidate_geometry(row, int(row["candidate_oracle_viewpoint_id"]))["geodesic"]) < high)]
        geo_bins[label] = {"lower": low, "upper": high, "count": len(members), "regret": _distribution([float(row["regret"]) for row in members]), "headroom": _headroom(members)}

    def radius_direction(values: Sequence[Dict[str, float]]) -> Dict[str, int]:
        return {"closer": sum(item["delta_radius"] < -0.25 for item in values), "same": sum(abs(item["delta_radius"]) <= 0.25 for item in values), "farther": sum(item["delta_radius"] > 0.25 for item in values)}

    high = [row for row in rows if row["regret_group"] == "G3_high_regret"]
    high_oracle = []
    for row in high:
        item = _candidate_geometry(row, int(row["candidate_oracle_viewpoint_id"]))
        if item is not None:
            high_oracle.append(item)
    geodesic_values = [item["geodesic"] for item in oracle_rows]
    return {"oracle_selected_geometry": {"oracle": summarize(oracle_rows), "model_selected_move": summarize(selected_rows)}, "azimuth_bins": {"oracle_all": bins(oracle_rows), "model_selected": bins(selected_rows), "high_regret_oracle": bins(high_oracle)}, "geodesic_bins_by_oracle": geo_bins, "radius_direction": {"oracle": radius_direction(oracle_rows), "model_selected_move": radius_direction(selected_rows)}, "oracle_geometry_count": len(oracle_rows), "model_selected_move_count": len(selected_rows), "high_regret_oracle_count": len(high_oracle), "oracle_geodesic_quantiles": {"q25": _percentile(geodesic_values, 25), "q50": _percentile(geodesic_values, 50), "q75": _percentile(geodesic_values, 75)} }


def _difficulty_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    gaps: List[float] = []
    for row in rows:
        utilities = np.asarray(row["utility_targets"], dtype=np.float64)
        ordered = np.sort(utilities)[::-1]
        gap = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
        gaps.append(gap)
        records.append({"max_utility": float(utilities.max()), "min_utility": float(utilities.min()), "utility_range": float(utilities.max() - utilities.min()), "utility_std": float(utilities.std()), "top1_top2_gap": gap, "positive_candidate_count": int(np.sum(utilities > NEAR_ZERO)), "negative_candidate_count": int(np.sum(utilities < -NEAR_ZERO)), "near_zero_candidate_count": int(np.sum(np.abs(utilities) <= NEAR_ZERO)), "row": row})
    gap_thresholds = {"q25": _percentile(gaps, 25), "q50": _percentile(gaps, 50), "q75": _percentile(gaps, 75)}
    output: Dict[str, Any] = {"thresholds": gap_thresholds, "overall": {key: _distribution([item[key] for item in records]) for key in ("max_utility", "min_utility", "utility_range", "utility_std", "top1_top2_gap")}}
    labels = (("very_small", lambda value: value <= gap_thresholds["q25"]), ("small", lambda value: value <= gap_thresholds["q50"]), ("medium", lambda value: value <= gap_thresholds["q75"]), ("large", lambda value: value > gap_thresholds["q75"]))
    for name, condition in labels:
        members = [item["row"] for item in records if condition(float(item["top1_top2_gap"]))]
        output[name] = {"count": len(members), "candidate_hit_rate": _rate(sum(int(row["predicted_candidate_viewpoint_id"]) == int(row["candidate_oracle_viewpoint_id"]) for row in members), len(members)), "regret": _distribution([float(row["regret"]) for row in members]), "headroom": _headroom(members)}
    return output


def _symmetric_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pair_diffs: List[float] = []
    candidates: List[Tuple[Mapping[str, Any], float, float]] = []
    for row in rows:
        ids = [int(value) for value in row["candidate_viewpoint_ids"]]
        utils = [float(value) for value in row["utility_targets"]]
        for left in range(len(ids)):
            a = _candidate_geometry(row, ids[left])
            if a is None:
                continue
            for right in range(left + 1, len(ids)):
                b = _candidate_geometry(row, ids[right])
                if b is None:
                    continue
                if abs(a["current_radius"] - b["current_radius"]) <= 0.25 and abs(a["geodesic"] - b["geodesic"]) <= 0.5 and abs(a["abs_azimuth_deg"] - b["abs_azimuth_deg"]) <= 10.0 and a["signed_azimuth_deg"] * b["signed_azimuth_deg"] <= 0:
                    difference = abs(utils[left] - utils[right])
                    pair_diffs.append(difference)
                    candidates.append((row, difference, max(utils[left], utils[right])))
    threshold = _percentile(pair_diffs, 90)
    affected = [item for item in candidates if item[1] >= threshold and pair_diffs]
    high_rows = {str(item[0]["episode_id"]) for item in affected}
    high_regret_rows = {str(row["episode_id"]) for row in rows if row["regret_group"] == "G3_high_regret"}
    return {"definition": {"radius_tolerance_m": 0.25, "geodesic_tolerance_m": 0.5, "absolute_azimuth_tolerance_deg": 10.0, "large_utility_difference_threshold": threshold}, "candidate_pair_count": len(candidates), "large_difference_pair_count": len(affected), "episode_count": len(high_rows), "episode_ratio": _rate(len(high_rows), len(rows)), "high_regret_episode_enrichment": _rate(len(high_rows & high_regret_rows), len(high_rows)) if high_rows else 0.0, "high_regret_overlap_count": len(high_rows & high_regret_rows), "overall_pair_utility_difference": _distribution(pair_diffs)}


def _record_level(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["record_id"])].append(row)
    output = []
    for record_id, members in grouped.items():
        output.append({"record_id": record_id, "action_label": str(members[0]["action_label"]), "episode_count": len(members), "mean_regret": _mean([float(row["regret"]) for row in members]), "p90_regret": _percentile([float(row["regret"]) for row in members], 90), "Set_accuracy": _rate(sum(int(row["selected_predicted_label_id"]) == int(row["label_id"]) for row in members), len(members)), "NoMove_accuracy": _rate(sum(int(row["current_predicted_label_id"]) == int(row["label_id"]) for row in members), len(members)), "SafeOracle_accuracy": _rate(sum(int(row["safe_oracle_predicted_label_id"]) == int(row["label_id"]) for row in members), len(members)), "headroom_capture": _headroom(members)["aggregate_capture"], "catastrophic_ratio": _rate(sum(bool(row["catastrophic_top5pct"]) for row in members), len(members))})
    output.sort(key=lambda item: (-float(item["mean_regret"]), str(item["record_id"])))
    catastrophic = [row for row in rows if bool(row["catastrophic_top5pct"])]
    top10_count = max(1, int(math.ceil(len(output) * 0.10)))
    top10 = output[:top10_count]
    top10_ids = {item["record_id"] for item in top10}
    return {"record_count": len(output), "top10pct_record_count": top10_count, "top10pct_mean_regret_records": top10, "top20_worst": output[:20], "top20_best": list(reversed(output[-20:])), "catastrophic_episode_count": len(catastrophic), "catastrophic_episode_share_in_top10pct_records": _rate(sum(str(row["record_id"]) in top10_ids for row in catastrophic), len(catastrophic)), "all_records": output}


def _representative(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def compact(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: row.get(key) for key in ("episode_id", "record_id", "action_label", "scene_id", "region", "current_viewpoint_id", "current_predicted_label_id", "current_entropy", "current_margin", "current_pose_confidence", "predicted_stays", "predicted_candidate_viewpoint_id", "predicted_action", "safe_oracle_stays", "safe_oracle_viewpoint_id", "safe_oracle_action", "candidate_oracle_viewpoint_id", "selected_true_utility", "safe_oracle_utility", "regret", "candidate_viewpoint_ids", "utility_targets", "predicted_utilities")}
    cases: Dict[str, Mapping[str, Any]] = {}
    for name, condition in {
        "correct_stay": lambda row: bool(row["predicted_stays"]) and bool(row["safe_oracle_stays"]),
        "correct_move": lambda row: not bool(row["predicted_stays"]) and not bool(row["safe_oracle_stays"]) and str(row["predicted_action"]) == str(row["safe_oracle_action"]),
        "candidate_miss_near_optimal": lambda row: int(row["predicted_candidate_viewpoint_id"]) != int(row["candidate_oracle_viewpoint_id"]) and float(row["regret"]) <= SMALL_REGRET,
        "missed_move": lambda row: row["failure_type"] == "A_missed_move",
        "unnecessary_move": lambda row: row["failure_type"] == "B_unnecessary_move",
        "wrong_candidate_high_regret": lambda row: row["failure_type"] == "C2_wrong_high_utility_loss",
        "catastrophic_top1pct": lambda row: bool(row["extreme_top1pct"]),
    }.items():
        candidates = [row for row in rows if condition(row)]
        if candidates:
            selected = min(candidates, key=lambda row: float(row["regret"])) if name in {"correct_stay", "correct_move"} else max(candidates, key=lambda row: float(row["regret"]))
            cases[name] = compact(selected)
    return cases


def analyze_rows(rows: List[Dict[str, Any]], categories: Sequence[str]) -> Dict[str, Any]:
    """Annotate and summarize aligned Set Ranker Test rows."""
    if not rows:
        raise ValueError("No prediction rows available")
    for row in rows:
        row["current_correct"] = int(row["current_predicted_label_id"]) == int(row["label_id"])
        row["selected_correct"] = int(row["selected_predicted_label_id"]) == int(row["label_id"])
        row["action_label"] = str(row.get("action_label", row.get("label_id")))
        row["candidate_oracle_utility"] = float(row.get("candidate_oracle_utility", row["safe_oracle_utility"]))
        row["current_margin"] = float(row.get("current_margin", row.get("current_feature", [0.0] * 275)[-2]))
        row["current_pose_confidence"] = float(row.get("current_pose_confidence", row.get("current_feature", [0.0] * 275)[-1]))
        row["current_entropy"] = float(row["current_entropy"])
        row["failure_type"] = "unclassified"
    regret_info = _regret_groups(rows)
    taxonomy = _taxonomy(rows)
    for row in rows:
        safe_move = not bool(row["safe_oracle_stays"])
        pred_move = not bool(row["predicted_stays"])
        same_action = str(row["predicted_action"]) == str(row["safe_oracle_action"])
        if same_action:
            row["failure_type"] = "D_correct_safe_action"
        elif safe_move and not pred_move:
            row["failure_type"] = "A_missed_move"
        elif not safe_move and pred_move:
            row["failure_type"] = "B_unnecessary_move"
        elif safe_move and pred_move:
            row["failure_type"] = "C1_wrong_near_optimal" if float(row["regret"]) <= SMALL_REGRET else "C2_wrong_high_utility_loss"
    action_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    region_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        action_groups[str(row["action_label"])].append(row)
        region_groups[str(row["region"])].append(row)
    action_stats = {key: _class_metrics(value) for key, value in sorted(action_groups.items())}
    for key, value in action_stats.items():
        value["oracle_gap_accuracy"] = value["SafeOracle_accuracy"] - value["Set_accuracy"]
    region_stats: Dict[str, Any] = {}
    for key, value in sorted(region_groups.items()):
        region_stats[key] = _class_metrics(value)
        region_stats[key]["Set_stay_rate"] = _rate(sum(bool(row["predicted_stays"]) for row in value), len(value))
        region_stats[key]["SafeOracle_stay_rate"] = _rate(sum(bool(row["safe_oracle_stays"]) for row in value), len(value))
    selected_dist = [float(row["selected_true_utility"]) for row in rows]
    true_dist = [float(row["safe_oracle_utility"]) for row in rows]
    high_rows = [row for row in rows if row["regret_group"] == "G3_high_regret"]
    high_action_counts = Counter(str(row["action_label"]) for row in high_rows)
    high_region_counts = Counter(str(row["region"]) for row in high_rows)
    for row in rows:
        row["selected_geometry"] = _candidate_geometry(row, int(row["predicted_candidate_viewpoint_id"])) if not bool(row["predicted_stays"]) else None
        row["oracle_geometry"] = _candidate_geometry(row, int(row["candidate_oracle_viewpoint_id"]))
    summary = {
        "analysis_protocol": {"model": "set_ranker", "split": "test", "near_zero_tolerance": NEAR_ZERO, "small_regret_threshold": SMALL_REGRET, "episode_iid_warning": "13,774 episodes are repeated scene/region views; independent motion records are 194."},
        "episode_count": len(rows), "record_count": len({str(row["record_id"]) for row in rows}), "categories": list(categories),
        "regret": regret_info, "failure_taxonomy": taxonomy, "candidate_miss": _miss_analysis(rows),
        "action_class": action_stats, "region": region_stats, "current_state": _state_analysis(rows),
        "geometry": _geometry_analysis(rows), "candidate_set_difficulty": _difficulty_analysis(rows),
        "symmetric_geometry_ambiguity": _symmetric_analysis(rows), "record_level": _record_level(rows),
        "high_regret_action_counts": dict(high_action_counts), "high_regret_region_counts": dict(high_region_counts),
        "representative_cases": _representative(rows),
        "overall": {"selected_true_utility": _distribution(selected_dist), "safe_oracle_utility": _distribution(true_dist), "headroom": _headroom(rows), "set_accuracy": _rate(sum(bool(row["selected_correct"]) for row in rows), len(rows)), "nomove_accuracy": _rate(sum(bool(row["current_correct"]) for row in rows), len(rows)), "safe_oracle_accuracy": _rate(sum(int(row["safe_oracle_predicted_label_id"]) == int(row["label_id"]) for row in rows), len(rows))},
    }
    return summary


def prepare_aligned_rows(stage_a: Sequence[Mapping[str, Any]], stage_b: Sequence[Mapping[str, Any]], features: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lengths = {len(stage_a), len(stage_b), len(features), len(predictions)}
    if len(lengths) != 1:
        raise ValueError(f"Stage A/B/feature/prediction lengths disagree: {sorted(lengths)}")
    for name, source in (("Stage A", stage_a), ("Stage B", stage_b), ("feature", features), ("prediction", predictions)):
        ids = [str(item.get("episode_id", "")) for item in source]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {name} episode_id detected")
    rows: List[Dict[str, Any]] = []
    for index, (episode, utility, feature, prediction) in enumerate(zip(stage_a, stage_b, features, predictions)):
        ids = [str(item.get("episode_id")) for item in (episode, utility, feature, prediction)]
        if len(set(ids)) != 1:
            raise ValueError(f"Episode alignment mismatch at line {index + 1}: {ids}")
        if str(prediction.get("policy_split")) != "test":
            raise ValueError(f"Unexpected prediction split at {prediction['episode_id']}")
        row = dict(prediction)
        row["action_label"] = str(episode.get("action_label", episode.get("label_id")))
        row["stage_b"] = utility
        row["feature_row"] = feature
        row["current_margin"] = float(feature["current_feature"][-2])
        row["current_pose_confidence"] = float(feature["current_feature"][-1])
        row["current_logp_true"] = float(utility["current"]["logp_true"])
        row["candidate_geometry"] = feature["candidate_geometry"]
        row["candidate_oracle_utility"] = float(utility["oracle"].get("candidate_oracle_utility", utility["oracle"]["safe_oracle_utility"]))
        rows.append(row)
    return rows
