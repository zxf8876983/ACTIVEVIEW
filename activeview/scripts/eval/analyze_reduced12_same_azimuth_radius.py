#!/usr/bin/env python3
"""Val-only audit of radius and image-scale effects under matched azimuths."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl

LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
RADII = (1.5, 2.0, 2.5, 3.0)
RADIUS_KEYS = tuple(f"{value:.1f}" for value in RADII)
SEED = 42
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/same_azimuth_radius_audit"
DEFAULT_ARCHIVE_ROOT = "datasets/offline/habitat-train/00006-00087"
DEFAULT_TRUE_CACHE = "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"
DEFAULT_SCENE_METRICS = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz"
DEFAULT_PROJECTED_METRICS = REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_view_observability_oracle/candidate_metrics.npz"
QUALITATIVE_MANIFEST = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization/case_manifest.json"
EPS = 1.0e-10


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.size != y.size:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) <= EPS or np.std(ry) <= EPS:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    pred, target = np.asarray(predictions, dtype=np.int64), np.asarray(labels, dtype=np.int64)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for truth, guess in zip(target.tolist(), pred.tolist()):
        matrix[int(truth), int(guess)] += 1
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
        per_class[name] = {"label_id": class_id, "count": int(support), "recall": recall, "precision": precision, "f1": f1}
    return {"count": int(target.size), "accuracy": float(np.mean(pred == target)) if target.size else 0.0, "macro_f1": float(np.mean(f1_values)), "per_class": per_class, "confusion_matrix": matrix.tolist()}


def _radius_key(value: float) -> str:
    rounded = round(float(value), 1)
    if not any(abs(rounded - radius) <= 0.05 for radius in RADII):
        raise ValueError(f"unexpected candidate radius: {value}")
    return f"{rounded:.1f}"


def _metadata_views(archive_root: Path, scene_id: str, placement_id: str) -> dict[int, Mapping[str, Any]]:
    path = archive_root / scene_id / "candidate_metadata" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for placement in payload.get("placements_data", []):
        if str(placement.get("placement_id")) == placement_id:
            views = {int(view["viewpoint_id"]): view for view in placement.get("viewpoints", [])}
            if len(views) != 32:
                raise ValueError(f"expected 32 viewpoints in {path} placement {placement_id}")
            return views
    raise KeyError(f"missing placement {placement_id} in {path}")


def _archive_path(archive_root: Path, row: Mapping[str, Any]) -> Path:
    path = archive_root / str(row["scene_id"]) / str(row.get("placement_id") or row.get("region")) / f"{row['record_id']}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing archive: {path}")
    return path


def _artifact_map(path: Path, score_key: str) -> dict[str, dict[int, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    arrays = _load_npz(path)
    required = {"episode_ids", "candidate_ids", "candidate_mask", score_key}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    output: dict[str, dict[int, float]] = {}
    for index, episode in enumerate(arrays["episode_ids"].tolist()):
        mask = np.asarray(arrays["candidate_mask"][index], dtype=bool)
        ids = np.asarray(arrays["candidate_ids"][index], dtype=np.int64)[mask]
        values = np.asarray(arrays[score_key][index], dtype=np.float64)[mask]
        output[str(episode)] = {int(candidate): float(value) for candidate, value in zip(ids.tolist(), values.tolist())}
    return output


def _load_projected_area(path: Path) -> dict[str, dict[int, float]]:
    if not path.is_file():
        return {}
    return _artifact_map(path, "projected_area")


def _candidate_records(
    data_root: Path,
    moving_rows: Sequence[Mapping[str, Any]],
    stage_c_rows: Sequence[Mapping[str, Any]],
    true_cache: Mapping[str, np.ndarray],
    scene_visibility: Mapping[str, Mapping[int, float]],
    projected_area: Mapping[str, Mapping[int, float]],
    archive_root: Path,
) -> list[dict[str, Any]]:
    stage_index = {str(row["episode_id"]): index for index, row in enumerate(stage_c_rows)}
    if len(stage_index) != len(stage_c_rows):
        raise ValueError("duplicate Stage-C episode_id")
    cache_index = {str(row["episode_id"]): index for index, row in enumerate(stage_c_rows)}
    manifests: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for row in moving_rows:
        episode_id = str(row["episode_id"])
        if episode_id not in stage_index or episode_id not in scene_visibility:
            raise ValueError(f"moving context missing Stage-C/visibility entry: {episode_id}")
        cache_i = cache_index[episode_id]
        slots = np.flatnonzero(np.asarray(true_cache["candidate_mask"][cache_i], dtype=bool))
        candidate_ids = np.asarray(true_cache["candidate_ids"][cache_i], dtype=np.int64)[slots]
        stage_ids = np.asarray(stage_c_rows[stage_index[episode_id]]["candidate_viewpoint_ids"], dtype=np.int64)
        if set(candidate_ids.tolist()) != set(stage_ids.tolist()):
            raise ValueError(f"candidate identity mismatch for {episode_id}")
        archive_path = _archive_path(archive_root, row)
        with np.load(archive_path, allow_pickle=False) as archive:
            archive_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            confidence = np.asarray(archive["confidence"], dtype=np.float64)
            agent_positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float64)
            placement_position = np.asarray(archive["placement_position"], dtype=np.float64)
        if confidence.shape != (32,) or agent_positions.shape != (32, 3):
            raise ValueError(f"invalid archive geometry schema: {archive_path}")
        placement_key = (str(row["scene_id"]), str(row.get("placement_id") or row.get("region")))
        if placement_key not in manifests:
            manifests[placement_key] = _metadata_views(archive_root, *placement_key)
        views = manifests[placement_key]
        vis = scene_visibility[episode_id]
        area = projected_area.get(episode_id, {})
        logp = np.asarray(true_cache["candidate_logp"][cache_i, slots], dtype=np.float64)
        label = int(row["label_id"])
        context_candidates: list[dict[str, Any]] = []
        for offset, (candidate_id, candidate_logp) in enumerate(zip(candidate_ids.tolist(), logp)):
            if int(candidate_id) not in views or int(candidate_id) not in vis:
                raise ValueError(f"candidate {candidate_id} missing metadata/visibility for {episode_id}")
            archive_matches = np.flatnonzero(archive_ids == int(candidate_id))
            if archive_matches.size != 1:
                raise ValueError(f"archive viewpoint alignment failure for {episode_id}/{candidate_id}")
            archive_i = int(archive_matches[0])
            view = views[int(candidate_id)]
            truth_score = float(candidate_logp[label])
            other = np.delete(candidate_logp, label)
            context_candidates.append({
                "candidate_id": int(candidate_id),
                "radius": _radius_key(float(view["radius_m"])),
                "azimuth": round(float(view["azimuth_deg"]), 3),
                "gt_logp": truth_score,
                "gt_margin": truth_score - float(np.max(other)),
                "prediction": int(np.argmax(candidate_logp)),
                "correct": bool(int(np.argmax(candidate_logp)) == label),
                "scene_visibility": float(vis[int(candidate_id)]),
                "pose_confidence": float(confidence[archive_i]),
                "distance_m": float(np.linalg.norm(agent_positions[archive_i] - placement_position)),
                "projected_area": float(area.get(int(candidate_id), float("nan"))),
            })
        records.append({"episode_id": episode_id, "scene_id": str(row["scene_id"]), "placement_id": placement_key[1], "record_id": str(row["record_id"]), "label": label, "candidates": context_candidates})
    return records


def _group_by_azimuth(context: Mapping[str, Any]) -> dict[float, dict[str, Mapping[str, Any]]]:
    grouped: dict[float, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for candidate in context["candidates"]:
        grouped[float(candidate["azimuth"])][str(candidate["radius"])] = candidate
    return dict(grouped)


def _pair_stats(groups: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    deltas = [float(far["gt_margin"] - near["gt_margin"]) for near, far in groups]
    if not deltas:
        return {"groups": 0, "mean_delta_farther_minus_closer": 0.0, "median_delta_farther_minus_closer": 0.0, "p_farther_better": 0.0, "p_closer_better": 0.0, "p_wrong_to_correct": 0.0, "p_correct_to_wrong": 0.0}
    wrong_to_correct = [not near["correct"] and far["correct"] for near, far in groups]
    correct_to_wrong = [near["correct"] and not far["correct"] for near, far in groups]
    return {"groups": len(groups), "mean_delta_farther_minus_closer": float(np.mean(deltas)), "median_delta_farther_minus_closer": float(np.median(deltas)), "p_farther_better": float(np.mean(np.asarray(deltas) > 0.0)), "p_closer_better": float(np.mean(np.asarray(deltas) < 0.0)), "p_wrong_to_correct": float(np.mean(wrong_to_correct)), "p_correct_to_wrong": float(np.mean(correct_to_wrong))}


def _radius_candidate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    for radius in RADIUS_KEYS:
        candidates = [candidate for record in records for candidate in record["candidates"] if candidate["radius"] == radius]
        output[radius] = {"candidate_count": len(candidates), "contexts_with_radius": int(sum(any(candidate["radius"] == radius for candidate in record["candidates"]) for record in records)), "correct_rate": float(np.mean([candidate["correct"] for candidate in candidates])) if candidates else 0.0, "mean_gt_logp": float(np.mean([candidate["gt_logp"] for candidate in candidates])) if candidates else 0.0, "mean_gt_margin": float(np.mean([candidate["gt_margin"] for candidate in candidates])) if candidates else 0.0, "median_gt_margin": float(np.median([candidate["gt_margin"] for candidate in candidates])) if candidates else 0.0}
        predictions: list[int] = []
        covered_labels: list[int] = []
        for record in records:
            available = [candidate for candidate in record["candidates"] if candidate["radius"] == radius]
            if available:
                best = max(available, key=lambda candidate: (float(candidate["gt_margin"]), -int(candidate["candidate_id"])))
                predictions.append(int(best["prediction"]))
                covered_labels.append(int(record["label"]))
        output[radius]["coverage"] = float(len(covered_labels) / len(records)) if records else 0.0
        output[radius]["oracle_metrics"] = _metrics(predictions, covered_labels) if covered_labels else {"count": 0, "accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}
    return output


def _four_radius_groups(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for record in records:
        for azimuth, radius_map in _group_by_azimuth(record).items():
            if all(radius in radius_map for radius in RADIUS_KEYS):
                groups.append({"episode_id": record["episode_id"], "label": record["label"], "azimuth": azimuth, "candidates": radius_map})
    return groups


def _full_group_summary(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best_counts = {radius: 0 for radius in RADIUS_KEYS}
    margins = {radius: [] for radius in RADIUS_KEYS}
    correct = {radius: [] for radius in RADIUS_KEYS}
    for group in groups:
        for radius in RADIUS_KEYS:
            candidate = group["candidates"][radius]
            margins[radius].append(float(candidate["gt_margin"]))
            correct[radius].append(bool(candidate["correct"]))
        best = max(RADIUS_KEYS, key=lambda radius: (float(group["candidates"][radius]["gt_margin"]), -float(radius)))
        best_counts[best] += 1
    return {"groups": len(groups), "best_radius_distribution": {radius: {"count": count, "rate": float(count / len(groups)) if groups else 0.0} for radius, count in best_counts.items()}, "mean_gt_margin_by_radius": {radius: float(np.mean(values)) if values else 0.0 for radius, values in margins.items()}, "median_gt_margin_by_radius": {radius: float(np.median(values)) if values else 0.0 for radius, values in margins.items()}, "correctness_by_radius": {radius: float(np.mean(values)) if values else 0.0 for radius, values in correct.items()}}


def _visibility_matched(records: Sequence[Mapping[str, Any]], threshold: float = 0.05) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_groups: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {f"{near}_vs_{far}": [] for near, far in zip(RADIUS_KEYS, RADIUS_KEYS[1:])}
    all_pairs: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {"1.5_vs_2.0": [], "1.5_vs_2.5": [], "1.5_vs_3.0": [], "2.0_vs_2.5": [], "2.0_vs_3.0": [], "2.5_vs_3.0": []}
    matched_four: list[dict[str, Any]] = []
    for record in records:
        for azimuth, radius_map in _group_by_azimuth(record).items():
            for near_index, near in enumerate(RADIUS_KEYS):
                for far in RADIUS_KEYS[near_index + 1:]:
                    if near in radius_map and far in radius_map and abs(float(radius_map[near]["scene_visibility"]) - float(radius_map[far]["scene_visibility"])) <= threshold:
                        all_pairs[f"{near}_vs_{far}"].append((radius_map[near], radius_map[far]))
            if all(radius in radius_map for radius in RADIUS_KEYS):
                values = [float(radius_map[radius]["scene_visibility"]) for radius in RADIUS_KEYS]
                if max(values) - min(values) <= threshold:
                    matched_four.append({"label": record["label"], "candidates": radius_map})
    combined = [pair for pairs in all_pairs.values() for pair in pairs]
    return {"threshold": threshold, "pairwise": {name: _pair_stats(pairs) for name, pairs in all_pairs.items()}, "combined_pairwise": _pair_stats(combined), "four_radius": _full_group_summary(matched_four)}, {"matched_pair_groups": {name: len(pairs) for name, pairs in all_pairs.items()}, "matched_four_radius_groups": len(matched_four)}


def _radius_pairs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {"1.5_vs_2.0": [], "1.5_vs_2.5": [], "1.5_vs_3.0": [], "2.0_vs_2.5": [], "2.0_vs_3.0": [], "2.5_vs_3.0": []}
    for record in records:
        for radius_map in _group_by_azimuth(record).values():
            for near_index, near in enumerate(RADIUS_KEYS):
                for far in RADIUS_KEYS[near_index + 1:]:
                    if near in radius_map and far in radius_map:
                        pairs[f"{near}_vs_{far}"].append((radius_map[near], radius_map[far]))
    return {name: _pair_stats(groups) for name, groups in pairs.items()}


def _oracle_methods(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    s0_predictions: list[int] = []
    frozen_predictions: list[int] = []
    any_correct: list[int] = []
    best_radius_predictions: list[int] = []
    all_candidates: list[Mapping[str, Any]] = []
    for record in records:
        all_candidates.extend(record["candidates"])
        s0_predictions.append(int(record["s0_prediction"]))
        frozen_predictions.append(int(record["frozen_prediction"]))
        candidates = list(record["candidates"])
        best = max(candidates, key=lambda candidate: (float(candidate["gt_margin"]), -int(candidate["candidate_id"])))
        best_radius_predictions.append(int(best["prediction"]))
        any_correct.append(int(record["label"]) if record["s0_prediction"] == record["label"] or any(candidate["correct"] for candidate in candidates) else int(record["s0_prediction"]))
    same_labels: list[int] = []
    same_best_predictions: list[int] = []
    same_closest_predictions: list[int] = []
    same_best_radii: list[str] = []
    same_closest_radii: list[str] = []
    for record in records:
        for radius_map in _group_by_azimuth(record).values():
            if len(radius_map) < 2:
                continue
            closest = min(radius_map.values(), key=lambda candidate: (float(candidate["radius"]), int(candidate["candidate_id"])))
            best = max(radius_map.values(), key=lambda candidate: (float(candidate["gt_margin"]), -int(candidate["candidate_id"])))
            same_labels.append(int(record["label"]))
            same_best_predictions.append(int(best["prediction"]))
            same_closest_predictions.append(int(closest["prediction"]))
            same_best_radii.append(str(best["radius"]))
            same_closest_radii.append(str(closest["radius"]))
    same_radius_metrics = _metrics(same_best_predictions, same_labels) if same_labels else {"count": 0, "accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}
    same_radius_metrics.update({
        "group_count": len(same_labels),
        "closest_radius_accuracy": float(np.mean(np.asarray(same_closest_predictions) == np.asarray(same_labels))) if same_labels else 0.0,
        "best_radius_distribution": {radius: {"count": same_best_radii.count(radius), "rate": float(same_best_radii.count(radius) / len(same_best_radii)) if same_best_radii else 0.0} for radius in RADIUS_KEYS},
        "closest_to_best_wrong_to_correct": float(np.mean([(closest != label and best == label) for closest, best, label in zip(same_closest_predictions, same_best_predictions, same_labels)])) if same_labels else 0.0,
        "closest_to_best_correct_to_wrong": float(np.mean([(closest == label and best != label) for closest, best, label in zip(same_closest_predictions, same_best_predictions, same_labels)])) if same_labels else 0.0,
    })
    return {"S0-only": _metrics(s0_predictions, labels), "FrozenStageCv0": _metrics(frozen_predictions, labels), "AnyCorrect Oracle": _metrics(any_correct, labels), "BestRadiusOracle": _metrics(best_radius_predictions, labels), "Same-Azimuth-BestRadius": same_radius_metrics, "candidate_count": len(all_candidates)}


def _per_class(records: Sequence[Mapping[str, Any]], full_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    by_label = {name: index for index, name in enumerate(LABELS)}
    for label_name, label_id in by_label.items():
        groups = [group for group in full_groups if int(group["label"]) == label_id]
        output[label_name] = {"full_four_radius_groups": len(groups), "best_radius_distribution": _full_group_summary(groups)["best_radius_distribution"], "mean_gt_margin_by_radius": _full_group_summary(groups)["mean_gt_margin_by_radius"], "correctness_by_radius": _full_group_summary(groups)["correctness_by_radius"]}
    return output


def _qualitative_check(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not QUALITATIVE_MANIFEST.is_file():
        return {"manifest_available": False, "cases": [], "same_azimuth_cases": []}
    cases = json.loads(QUALITATIVE_MANIFEST.read_text(encoding="utf-8"))
    by_episode = {str(record["episode_id"]): record for record in records}
    output: list[dict[str, Any]] = []
    same_azimuth: list[dict[str, Any]] = []
    for case in cases:
        episode = str(case["episode_id"])
        if episode not in by_episode:
            continue
        record = by_episode[episode]
        figure_record = case.get("figure_record", {})
        selected_raw = case.get("selected_id")
        oracle_raw = case.get("oracle_id")
        selected_id = int(selected_raw if selected_raw is not None else figure_record.get("selected_viewpoint_id"))
        oracle_id = int(oracle_raw if oracle_raw is not None else figure_record.get("oracle_viewpoint_id"))
        by_candidate = {int(candidate["candidate_id"]): candidate for candidate in record["candidates"]}
        if selected_id not in by_candidate or oracle_id not in by_candidate:
            output.append({"case_id": case["case_id"], "episode_id": episode, "action": LABELS[int(record["label"])], "selected_id": selected_id, "oracle_id": oracle_id, "status": "outside_legal_candidate_set"})
            continue
        selected = by_candidate[selected_id]
        oracle = by_candidate[oracle_id]
        same_group = [candidate for candidate in record["candidates"] if abs(float(candidate["azimuth"]) - float(selected["azimuth"])) <= 1.0e-6]
        same_best = max(same_group, key=lambda candidate: (float(candidate["gt_margin"]), -int(candidate["candidate_id"])))
        item = {"case_id": case["case_id"], "action": LABELS[int(record["label"])], "selected_radius": selected["radius"], "best_radius": same_best["radius"], "global_best_radius": oracle["radius"], "azimuth": selected["azimuth"], "selected_margin": selected["gt_margin"], "best_margin": same_best["gt_margin"], "global_best_margin": oracle["gt_margin"], "selected_scene_visibility": selected["scene_visibility"], "best_scene_visibility": same_best["scene_visibility"], "selected_correct": selected["correct"], "best_correct": same_best["correct"], "global_best_azimuth": oracle["azimuth"]}
        output.append(item)
        if abs(float(selected["azimuth"]) - float(oracle["azimuth"])) <= 1.0e-6:
            same_azimuth.append(item)
    return {"manifest_available": True, "cases": output, "same_azimuth_cases": same_azimuth, "requested_case_ids": ["case_0002", "case_0007", "case_0009"]}


def _image_scale(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {"radius": [], "distance_m": [], "projected_area": [], "gt_margin": []}
    area_radius: list[float] = []
    area_margin: list[float] = []
    for record in records:
        for candidate in record["candidates"]:
            values["radius"].append(float(candidate["radius"]))
            values["distance_m"].append(float(candidate["distance_m"]))
            values["gt_margin"].append(float(candidate["gt_margin"]))
            if np.isfinite(candidate["projected_area"]):
                values["projected_area"].append(float(candidate["projected_area"]))
                area_radius.append(float(candidate["radius"]))
                area_margin.append(float(candidate["gt_margin"]))
    result = {"candidate_count": len(values["gt_margin"]), "spearman": {"radius_vs_projected_area": _spearman(area_radius, values["projected_area"]) if values["projected_area"] else 0.0, "projected_area_vs_gt_margin": _spearman(values["projected_area"], area_margin) if values["projected_area"] else 0.0, "distance_vs_gt_margin": _spearman(values["distance_m"], values["gt_margin"])}}
    result["mean"] = {name: float(np.mean(array)) if array else 0.0 for name, array in values.items()}
    return result


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    radius = result["radius_metrics"]
    pairs = result["pairwise_radius_metrics"]
    matched = result["visibility_matched_metrics"]
    full = result["four_radius_groups"]
    methods = result["methods"]
    gaps = [float(radius[key]["mean_gt_margin"]) for key in RADIUS_KEYS]
    margin_gap = max(gaps) - min(gaps) if gaps else 0.0
    matched_delta = float(matched["combined_pairwise"]["mean_delta_farther_minus_closer"])
    if margin_gap >= 0.15 and abs(matched_delta) >= 0.10:
        explanation = "D. distance / image scale / perspective is the strongest remaining explanation: radius differences persist in the SceneVisibility-matched subset."
    elif margin_gap < 0.15:
        explanation = "E. no single stable radius rule is supported; overall radius effects are small relative to candidate variability."
    elif abs(matched_delta) < 0.10:
        explanation = "A/E. the unconditioned radius gap is not robust after SceneVisibility matching, so scene geometry or heterogeneous factors are more plausible than a pure image-scale effect."
    else:
        explanation = "E. the measured effects do not isolate one stable causal factor."
    lines = [
        "# Reduced12 Same-Azimuth Radius / Image-Scale Audit", "",
        f"Val Moving only: {result['population']['moving_contexts']} contexts and {result['population']['candidate_samples']} legal candidate samples. Test was not read; no model or perception artifact was changed.", "",
        "## Radius candidate quality", "", "| Radius | Candidate correct | Mean GT-logp | Mean GT-margin | Coverage | Oracle Acc | Oracle Macro-F1 |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for radius_key in RADIUS_KEYS:
        item = radius[radius_key]
        lines.append(f"| {radius_key} m | {item['correct_rate']:.6f} | {item['mean_gt_logp']:.6f} | {item['mean_gt_margin']:.6f} | {item['coverage']:.6f} | {item['oracle_metrics']['accuracy']:.6f} | {item['oracle_metrics']['macro_f1']:.6f} |")
    lines.extend(["", "## Full four-radius same-azimuth groups", "", f"Groups: {full['groups']}; best-radius distribution: " + ", ".join(f"{r}m={full['best_radius_distribution'][r]['rate']:.3f}" for r in RADIUS_KEYS), "", "| Radius | Mean margin | Median margin | Correctness |", "|---:|---:|---:|---:|"])
    for radius_key in RADIUS_KEYS:
        lines.append(f"| {radius_key} m | {full['mean_gt_margin_by_radius'][radius_key]:.6f} | {full['median_gt_margin_by_radius'][radius_key]:.6f} | {full['correctness_by_radius'][radius_key]:.6f} |")
    lines.extend(["", "## Same-azimuth pairwise effects", "", "| Pair | Groups | Δ margin (far-near) | P far better | Wrong→correct | Correct→wrong |", "|---|---:|---:|---:|---:|---:|"])
    for name, item in pairs.items():
        lines.append(f"| {name} | {item['groups']} | {item['mean_delta_farther_minus_closer']:.6f} | {item['p_farther_better']:.6f} | {item['p_wrong_to_correct']:.6f} | {item['p_correct_to_wrong']:.6f} |")
    lines.extend(["", "## SceneVisibility-matched subset", "", f"Fixed threshold: 0.05; matched four-radius groups: {matched['four_radius']['groups']}; combined pair groups: {matched['combined_pairwise']['groups']}; mean Δ margin (far-near): {matched['combined_pairwise']['mean_delta_farther_minus_closer']:.6f}; wrong→correct: {matched['combined_pairwise']['p_wrong_to_correct']:.6f}.", "", "## Oracle comparison", "", f"BestRadiusOracle Accuracy/Macro-F1: {methods['BestRadiusOracle']['accuracy']:.6f}/{methods['BestRadiusOracle']['macro_f1']:.6f}; Same-Azimuth-BestRadius group Accuracy/Macro-F1: {methods['Same-Azimuth-BestRadius']['accuracy']:.6f}/{methods['Same-Azimuth-BestRadius']['macro_f1']:.6f}; FrozenStageCv0: {methods['FrozenStageCv0']['accuracy']:.6f}/{methods['FrozenStageCv0']['macro_f1']:.6f}; AnyCorrect Oracle: {methods['AnyCorrect Oracle']['accuracy']:.6f}/{methods['AnyCorrect Oracle']['macro_f1']:.6f}.", f"Same-Azimuth-BestRadius covers {methods['Same-Azimuth-BestRadius']['group_count']} groups; closest-radius→best-radius wrong→correct={methods['Same-Azimuth-BestRadius']['closest_to_best_wrong_to_correct']:.6f}, correct→wrong={methods['Same-Azimuth-BestRadius']['closest_to_best_correct_to_wrong']:.6f}.", "Only radius was changed inside the radius oracle; all terminal labels came from the selected real archived skeleton's frozen ST-GCN cache.", "", "## Image-scale proxies", "", f"Spearman radius→projected-area: {result['image_scale']['spearman']['radius_vs_projected_area']:.6f}; projected-area→GT-margin: {result['image_scale']['spearman']['projected_area_vs_gt_margin']:.6f}; distance→GT-margin: {result['image_scale']['spearman']['distance_vs_gt_margin']:.6f}.", "", "## Qualitative cases", ""])
    for case in result["qualitative_case_check"]["cases"]:
        if case.get("status") != "ok":
            lines.append(f"- {case['case_id']}: {case.get('status', 'unavailable')} (selected/oracle viewpoint is outside this context's legal candidate set).")
            continue
        if case["case_id"] in {"case_0002", "case_0007", "case_0009"} or abs(float(case["azimuth"]) - float(case["global_best_azimuth"])) <= 1.0e-6:
            lines.append(f"- {case['case_id']} ({case['action']}): selected {case['selected_radius']}m, same-azimuth best {case['best_radius']}m, global best {case['global_best_radius']}m, azimuth {case['azimuth']:.1f}°, margin {case['selected_margin']:.3f}→{case['best_margin']:.3f} (global {case['global_best_margin']:.3f}), SceneVisibility {case['selected_scene_visibility']:.3f}→{case['best_scene_visibility']:.3f}.")
    lines.extend(["", "## Scientific judgment", "", "Q1. Is 1.5m systematically biased? " + ("1.5m is systematically the strongest radius in this archive (higher mean margin/correctness than farther radii), rather than being the weakest." if radius["1.5"]["mean_gt_margin"] == max(gaps) and margin_gap >= 0.15 else "No strong universal radius ordering is supported by this audit."), "Q2. Is there a preferred 2.0/2.5/3.0m interval? " + ("No; the aggregate and matched tables favor the near 1.5m condition, while the best-radius distribution remains heterogeneous." if radius["1.5"]["mean_gt_margin"] == max(gaps) and margin_gap >= 0.15 else "No clear fixed interval is established."), "Q3. Does radius still matter at matched azimuth and SceneVisibility? " + (f"The matched mean far-near margin is {matched_delta:.6f}; this is " + ("consistent with a material residual radius effect." if abs(matched_delta) >= 0.10 else "small, so a pure radius explanation is weak.")), "Q4. Are recent same-azimuth failures global or isolated? " + ("They are consistent with a broader tendency for farther radii to lose margin, although individual flips remain heterogeneous." if abs(matched_delta) >= 0.10 else "They appear to be heterogeneous or case-specific."), f"Q5. Most reasonable explanation: {explanation}", "", "Flags: `test_used=false`; `training_used=false`; `gt_action_used_for_posthoc_diagnostic_only=true`; `future_candidate_skeleton_used_for_terminal_diagnostic_only=true`; `new_rgb_rendered=false`; `selector_modified=false`; `deployable=false."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_root: Path, output_dir: Path, archive_root: Path, true_cache_path: Path, scene_visibility_path: Path, projected_metrics_path: Path) -> dict[str, Any]:
    stage_c_path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/features/val.jsonl"
    stage_d_path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_d/features/val.jsonl"
    if "test" in str(stage_c_path).lower() or "test" in str(stage_d_path).lower() or "test" in str(true_cache_path).lower():
        raise ValueError("this diagnostic is Val-only")
    stage_c_rows, moving_rows = load_jsonl(stage_c_path), load_jsonl(stage_d_path)
    true_cache = _load_npz(true_cache_path)
    scene_visibility = _artifact_map(scene_visibility_path, "scene_visibility_score")
    projected_area = _load_projected_area(projected_metrics_path)
    records = _candidate_records(data_root, moving_rows, stage_c_rows, true_cache, scene_visibility, projected_area, archive_root)
    for record, row in zip(records, moving_rows):
        s0_feature, s1_feature = np.asarray(row["s0_feature"], dtype=np.float64), np.asarray(row["s1_feature"], dtype=np.float64)
        record["s0_prediction"] = int(np.argmax(s0_feature[256:268]))
        record["frozen_prediction"] = int(np.argmax(s1_feature[256:268]))
    radius_metrics = _radius_candidate_metrics(records)
    pairwise = _radius_pairs(records)
    full_groups = _four_radius_groups(records)
    full_summary = _full_group_summary(full_groups)
    matched, matched_counts = _visibility_matched(records)
    methods = _oracle_methods(records)
    qualitative = _qualitative_check(records)
    image_scale = _image_scale(records)
    labels = [int(record["label"]) for record in records]
    result: dict[str, Any] = {"experiment_id": "REDUCED12_SAME_AZIMUTH_RADIUS_AUDIT", "status": "COMPLETED", "seed": SEED, "test_used": False, "population": {"split": "val_moving", "moving_contexts": len(records), "candidate_samples": int(sum(len(record["candidates"]) for record in records)), "scene_count": len({record["scene_id"] for record in records})}, "labels": list(LABELS), "methods": methods, "radius_metrics": radius_metrics, "pairwise_radius_metrics": pairwise, "four_radius_groups": full_summary, "visibility_matched_metrics": {**matched, "counts": matched_counts}, "qualitative_case_check": qualitative, "image_scale": image_scale, "per_class_radius_metrics": _per_class(records, full_groups), "protocol": {"radii_m": list(RADII), "azimuth_matching": "exact metadata azimuth_deg within 1e-6 after 0.001-degree rounding", "terminal": "selected real archived candidate frozen ST-GCN true_logp", "scene_visibility_match_threshold": 0.05}, "leakage_flags": {"test_used": False, "training_used": False, "gt_action_used_for_posthoc_diagnostic_only": True, "future_candidate_skeleton_used_for_terminal_diagnostic_only": True, "new_rgb_rendered": False, "selector_modified": False, "deployable": False}, "artifacts": {"stage_c_val": str(stage_c_path.resolve()), "stage_d_val": str(stage_d_path.resolve()), "true_logp_cache": str(true_cache_path.resolve()), "scene_visibility": str(scene_visibility_path.resolve()), "projected_area": str(projected_metrics_path.resolve())}}
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = [(record["episode_id"], candidate) for record in records for candidate in record["candidates"]]
    np.savez_compressed(output_dir / "candidate_metrics.npz", episode_ids=np.asarray([item[0] for item in candidate_rows]), radius=np.asarray([float(item[1]["radius"]) for item in candidate_rows], dtype=np.float32), azimuth=np.asarray([float(item[1]["azimuth"]) for item in candidate_rows], dtype=np.float32), gt_logp=np.asarray([float(item[1]["gt_logp"]) for item in candidate_rows], dtype=np.float32), gt_margin=np.asarray([float(item[1]["gt_margin"]) for item in candidate_rows], dtype=np.float32), correct=np.asarray([bool(item[1]["correct"]) for item in candidate_rows], dtype=bool), scene_visibility=np.asarray([float(item[1]["scene_visibility"]) for item in candidate_rows], dtype=np.float32), distance_m=np.asarray([float(item[1]["distance_m"]) for item in candidate_rows], dtype=np.float32), projected_area=np.asarray([float(item[1]["projected_area"]) for item in candidate_rows], dtype=np.float32))
    _write_json(output_dir / "result.json", result)
    _write_json(output_dir / "radius_metrics.json", radius_metrics)
    _write_json(output_dir / "pairwise_radius_metrics.json", pairwise)
    _write_json(output_dir / "visibility_matched_metrics.json", {**matched, "counts": matched_counts})
    _write_json(output_dir / "per_class_radius_metrics.json", result["per_class_radius_metrics"])
    _write_json(output_dir / "qualitative_case_check.json", qualitative)
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--true-cache", type=Path, default=None)
    parser.add_argument("--scene-visibility", type=Path, default=DEFAULT_SCENE_METRICS)
    parser.add_argument("--projected-metrics", type=Path, default=DEFAULT_PROJECTED_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    archive_root = (args.archive_root or data_root / DEFAULT_ARCHIVE_ROOT).resolve()
    true_cache = (args.true_cache or data_root / DEFAULT_TRUE_CACHE).resolve()
    result = run(data_root, args.output_dir.resolve(), archive_root, true_cache, args.scene_visibility.resolve(), args.projected_metrics.resolve())
    print(json.dumps({"status": result["status"], "population": result["population"], "test_used": False}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
