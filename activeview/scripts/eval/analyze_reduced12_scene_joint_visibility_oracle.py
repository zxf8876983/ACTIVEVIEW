#!/usr/bin/env python3
"""Val-only Habitat environmental joint-visibility oracle for reduced12.

The script evaluates a privileged, scene-geometry-only NBV signal.  It uses
the existing moving Val contexts and archived terminal recognizer outputs;
the six sampled human poses are reconstructed from the raw BABEL/AMASS
motion, while ray casts run in a scene-only simulator so humanoid geometry is
never counted as an occluder.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.motion.babel_clean_dataset_generator import (
    _load_resampled_motion,
    apply_humanoid_pose,
    precompute_grounding_offsets,
)
from activeview.data.motion.motion_converter import MotionConverter


NUM_CLASSES = 12
VIEW_COUNT = 32
JOINT_COUNT = 17
FRAME_IDS = (0, 6, 12, 18, 24, 29)
SENSOR_HEIGHT_M = 1.10
RAY_EPSILON_M = 0.03
LOS_RESOLUTION_M = 0.20
SEED = 42
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
H36M17_LINKS = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine1", "spine3", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)
DATASET_NAME = "policy_reduced12_eight_placement_v1"
ARCHIVE_REL = "datasets/offline/habitat-train/00006-00087"
RAW_VAL_REL = "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _scene_file(scene_root: Path, scene_id: str, suffix: str) -> Path:
    matches = sorted((scene_root / scene_id).glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {suffix} asset for {scene_id}, found {len(matches)}")
    return matches[0]


def _scene_sim(scene_root: Path, scene_id: str, *, physics: bool) -> Any:
    import habitat_sim

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(_scene_file(scene_root, scene_id, ".basis.glb"))
    config.enable_physics = bool(physics)
    agent = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    navmesh = _scene_file(scene_root, scene_id, ".basis.navmesh")
    if not sim.pathfinder.is_loaded and not sim.pathfinder.load_nav_mesh(str(navmesh)):
        sim.close()
        raise RuntimeError(f"failed to load navmesh for {scene_id}")
    return sim


def _archive_path(data_root: Path, row: Mapping[str, Any]) -> Path:
    path = data_root / ARCHIVE_REL / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(target * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for cls, name in enumerate(LABELS):
        tp = float(matrix[cls, cls])
        support = float(matrix[cls].sum())
        predicted = float(matrix[:, cls].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        value = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1.append(value)
        per_class[name] = {"label_id": cls, "count": int(support), "recall": recall, "precision": precision, "f1": value}
    return {
        "count": int(target.size),
        "accuracy": float(np.mean(pred == target)) if target.size else 0.0,
        "macro_f1": float(np.mean(f1)) if f1 else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _grid_cell(point: np.ndarray, lower: np.ndarray, resolution: float, shape: tuple[int, int]) -> tuple[int, int] | None:
    col = int(math.floor((float(point[0]) - float(lower[0])) / resolution))
    row = int(math.floor((float(point[2]) - float(lower[2])) / resolution))
    if row < 0 or col < 0 or row >= shape[0] or col >= shape[1]:
        return None
    return row, col


def _topdown_grid(sim: Any, floor_y: float) -> tuple[np.ndarray, np.ndarray]:
    lower, _ = sim.pathfinder.get_bounds()
    grid = np.asarray(sim.pathfinder.get_topdown_view(LOS_RESOLUTION_M, float(floor_y), eps=0.5), dtype=bool)
    if grid.ndim != 2 or not grid.any():
        raise RuntimeError(f"empty top-down navmesh slice at y={floor_y:.3f}")
    return grid, np.asarray(lower, dtype=np.float64)


def _line_of_sight(start: np.ndarray, end: np.ndarray, grid: np.ndarray, lower: np.ndarray) -> bool:
    distance = float(np.linalg.norm((end - start)[[0, 2]]))
    steps = max(1, int(np.ceil(distance / (LOS_RESOLUTION_M * 0.5))))
    for point in np.linspace(start, end, steps + 1):
        cell = _grid_cell(point, lower, LOS_RESOLUTION_M, grid.shape)
        if cell is None or not bool(grid[cell]):
            return False
    return True


def _ray_visible(sim: Any, origin: np.ndarray, endpoint: np.ndarray) -> bool:
    import habitat_sim

    delta = endpoint - origin
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance <= 1e-6:
        return False
    ray = habitat_sim.geo.Ray(origin.astype(np.float32), (delta / distance).astype(np.float32))
    hits = sim.cast_ray(ray).hits
    distances = [float(getattr(hit, "ray_distance", np.inf)) for hit in hits]
    distances = [value for value in distances if np.isfinite(value)]
    return not distances or min(distances) >= distance - RAY_EPSILON_M


def _placement_map(data_root: Path, scene_id: str) -> dict[str, dict[str, Any]]:
    path = data_root / ARCHIVE_REL / "placement_sampling_v2" / scene_id / "placements.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    placements = payload.get("placements", [])
    if len(placements) != 8:
        raise ValueError(f"expected 8 placements for {scene_id}")
    return {str(item["placement_id"]): item for item in placements}


def _world_h36m17(
    human: Any,
    converted: Mapping[str, Any],
    placement: Mapping[str, Any],
    yaw_deg: float,
) -> np.ndarray:
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=yaw_deg)
    names = {str(human.get_link_name(index)): index for index in range(int(human.num_links))}
    required = [name for name in H36M17_LINKS if name != "pelvis"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"humanoid is missing H36M17 links: {missing}")
    output = np.empty((len(FRAME_IDS), JOINT_COUNT, 3), dtype=np.float32)
    base = np.asarray(placement["position"], dtype=np.float32)
    for out_index, frame_id in enumerate(FRAME_IDS):
        apply_humanoid_pose(
            human, joints[frame_id], roots[frame_id], base_position=base,
            scene_yaw_deg=yaw_deg, floor_y=float(base[1]), grounding_offset=float(offsets[frame_id]),
        )
        values: list[np.ndarray] = [np.asarray(human.translation, dtype=np.float32)]
        for name in H36M17_LINKS[1:]:
            node = human.get_link_scene_node(names[name])
            values.append(np.asarray(node.absolute_translation, dtype=np.float32))
        output[out_index] = np.asarray(values, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("non-finite world-space H36M17 joints")
    return output


def _load_raw_records(data_root: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads((data_root / RAW_VAL_REL).read_text(encoding="utf-8"))
    records = {str(item["record_id"]): item for item in payload}
    if len(records) != len(payload):
        raise ValueError("duplicate record_id in raw-val manifest")
    return records


def _build_visibility(
    data_root: Path,
    scene_root: Path,
    stage_c_rows: Sequence[Mapping[str, Any]],
    stage_c_predictions: Sequence[Mapping[str, Any]],
    moving_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    max_contexts: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import habitat_sim  # noqa: F401 - ensure the runtime is Habitat-enabled

    stage_c_by_key = {
        (str(row["record_id"]), str(row["scene_id"]), str(row["region"]), int(row["current_viewpoint_id"])): row
        for row in stage_c_rows
    }
    prediction_by_key = {
        (str(row["record_id"]), str(row["scene_id"]), str(row["region"]), int(row["current_viewpoint_id"])): row
        for row in stage_c_predictions
    }
    cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    records = _load_raw_records(data_root)
    selected = list(moving_rows[:max_contexts]) if max_contexts is not None else list(moving_rows)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(selected):
        grouped[str(row["scene_id"])].append((index, row))
    bundles: list[dict[str, Any] | None] = [None] * len(selected)
    started = time.perf_counter()
    archive_root = data_root / ARCHIVE_REL
    for scene_id, scene_rows in sorted(grouped.items()):
        ray_sim = _scene_sim(scene_root, scene_id, physics=True)
        human_sim = _scene_sim(scene_root, scene_id, physics=True)
        urdf = get_humanoid_urdf_path("male_0")
        human = human_sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(urdf))
        converter = MotionConverter(urdf)
        placements = _placement_map(data_root, scene_id)
        grid_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        converted_cache: dict[str, Mapping[str, Any]] = {}
        joints_cache: dict[tuple[str, str], np.ndarray] = {}
        try:
            for local_index, row in scene_rows:
                key = (str(row["record_id"]), scene_id, str(row["region"]), int(row["s0_viewpoint_id"]))
                if key not in stage_c_by_key or key not in prediction_by_key:
                    raise ValueError(f"missing Stage-C H1 context for {key}")
                episode_id = str(row["episode_id"])
                if episode_id not in cache_index:
                    raise ValueError(f"moving episode missing from counterfactual cache: {episode_id}")
                stage_row = stage_c_by_key[key]
                prediction_row = prediction_by_key[key]
                candidate_ids = np.asarray(stage_row["candidate_viewpoint_ids"], dtype=np.int64)
                prediction_ids = np.asarray(prediction_row["candidate_viewpoint_ids"], dtype=np.int64)
                predicted_utilities = np.asarray(prediction_row["predicted_utilities"], dtype=np.float64)
                if candidate_ids.ndim != 1 or prediction_ids.shape != candidate_ids.shape or set(prediction_ids.tolist()) != set(candidate_ids.tolist()):
                    raise ValueError(f"Stage-C candidate identity mismatch: {key}")
                if predicted_utilities.shape != candidate_ids.shape:
                    raise ValueError(f"Stage-C utility shape mismatch: {key}")
                cache_i = int(cache_index[episode_id])
                label = int(row["label_id"])
                if int(cache["label_id"][cache_i]) != label:
                    raise ValueError(f"label/cache mismatch: {episode_id}")
                archive = _load_npz(_archive_path(data_root, row))
                viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                camera_positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
                if viewpoint_ids.shape != (VIEW_COUNT,) or camera_positions.shape != (VIEW_COUNT, 3):
                    raise ValueError(f"invalid viewpoint archive schema: {episode_id}")
                placement_id = str(row["region"])
                if placement_id not in placements:
                    raise ValueError(f"unknown placement {placement_id} for {scene_id}")
                placement = placements[placement_id]
                record_id = str(row["record_id"])
                if record_id not in records:
                    raise ValueError(f"record not present in raw-val manifest: {record_id}")
                if record_id not in converted_cache:
                    converted_cache[record_id] = converter.convert(_load_resampled_motion(records[record_id], 30))
                joint_key = (record_id, placement_id)
                if joint_key not in joints_cache:
                    joints_cache[joint_key] = _world_h36m17(human, converted_cache[record_id], placement, float(placement["yaw_deg"]))
                world_joints = joints_cache[joint_key]
                placement_position = np.asarray(placement["position"], dtype=np.float32)
                floor_key = round(float(placement_position[1]), 3)
                if floor_key not in grid_cache:
                    grid_cache[floor_key] = _topdown_grid(ray_sim, floor_key)
                grid, lower = grid_cache[floor_key]
                viewpoint_index = {int(value): index for index, value in enumerate(viewpoint_ids.tolist())}
                visibility = np.empty((candidate_ids.size, JOINT_COUNT), dtype=np.float32)
                topdown = np.empty(candidate_ids.size, dtype=np.int8)
                for candidate_index, viewpoint_id in enumerate(candidate_ids.tolist()):
                    if int(viewpoint_id) not in viewpoint_index:
                        raise ValueError(f"viewpoint missing from archive: {episode_id}/{viewpoint_id}")
                    camera = camera_positions[viewpoint_index[int(viewpoint_id)]].astype(np.float32)
                    camera_sensor = camera + np.asarray([0.0, SENSOR_HEIGHT_M, 0.0], dtype=np.float32)
                    topdown[candidate_index] = int(_line_of_sight(camera_sensor, placement_position, grid, lower))
                    for joint_index in range(JOINT_COUNT):
                        visibility[candidate_index, joint_index] = float(np.mean([
                            _ray_visible(ray_sim, camera_sensor, world_joints[frame_index, joint_index])
                            for frame_index in range(len(FRAME_IDS))
                        ]))
                bundles[local_index] = {
                    "episode_id": episode_id,
                    "cache_index": cache_i,
                    "candidate_ids": candidate_ids,
                    "predicted_utilities": predicted_utilities,
                    "candidate_geodesic": np.asarray(stage_row["candidate_geodesic"], dtype=np.float64),
                    "visibility": visibility,
                    "scene_visibility": visibility.mean(axis=1),
                    "topdown_los": topdown,
                    "label": label,
                    "s0_viewpoint_id": int(row["s0_viewpoint_id"]),
                    "original_viewpoint_id": int(row["proposal_rank_1_id"]),
                }
                if (local_index + 1) % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"scene-joint-visibility: {local_index + 1}/{len(selected)} contexts ({elapsed:.1f}s)", flush=True)
        finally:
            human_sim.close()
            ray_sim.close()
    if any(bundle is None for bundle in bundles):
        raise RuntimeError("failed to build visibility for one or more contexts")
    return [bundle for bundle in bundles if bundle is not None], {
        "scene_count": len(grouped),
        "scene_ids": sorted(grouped),
        "frame_ids": list(FRAME_IDS),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate(
    moving_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    bundles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    names = ("S0-only", "FrozenStageCv0", "TopDownLOS-Filter+Frozen", "SceneVisibility", "SceneVisibility-Filter+Frozen", "AnyCorrect Oracle")
    predictions = {name: np.empty(labels.size, dtype=np.int64) for name in names}
    moved = {name: np.zeros(labels.size, dtype=bool) for name in names}
    candidate_visibility: list[np.ndarray] = []
    candidate_ids: list[np.ndarray] = []
    topdown_filter_diffs: list[float] = []
    correct_visibility: list[float] = []
    wrong_visibility: list[float] = []
    positive_visibility_count = 0
    positive_visibility_correct = 0
    zero_visibility_count = 0
    zero_visibility_correct = 0
    rank_scores: list[float] = []
    rank_truth: list[float] = []
    max_min: list[float] = []
    no_discrimination = 0
    los_equal = 0
    topdown_pred = []
    for index, (row, bundle) in enumerate(zip(moving_rows, bundles)):
        cache_i = int(bundle["cache_index"])
        label = int(labels[index])
        ids = np.asarray(bundle["candidate_ids"], dtype=np.int64)
        visibility = np.asarray(bundle["visibility"], dtype=np.float64)
        scene_scores = np.asarray(bundle["scene_visibility"], dtype=np.float64)
        los = np.asarray(bundle["topdown_los"], dtype=np.int8)
        utilities = np.asarray(bundle["predicted_utilities"], dtype=np.float64)
        geodesic = np.asarray(bundle["candidate_geodesic"], dtype=np.float64)
        true_logp = np.asarray(cache["true_logp"][cache_i, ids], dtype=np.float64)
        candidate_pred = np.argmax(true_logp, axis=1)
        s0_pred = int(np.argmax(cache["current_logp_s0"][cache_i]))
        s1_pred = int(np.argmax(cache["current_logp_s1"][cache_i]))
        predictions["S0-only"][index] = s0_pred
        predictions["FrozenStageCv0"][index] = s1_pred
        moved["FrozenStageCv0"][index] = True
        frozen_index = np.flatnonzero(ids == int(bundle["original_viewpoint_id"]))
        if frozen_index.size != 1:
            raise ValueError(f"Frozen proposal is not legal: {row['episode_id']}")
        frozen_index_value = int(frozen_index[0])
        visible = np.flatnonzero(los == 1)
        if visible.size:
            topdown_index = min(visible.tolist(), key=lambda j: (-utilities[j], geodesic[j], ids[j]))
            topdown_filter_diffs.append(float(topdown_index != frozen_index_value))
            predictions["TopDownLOS-Filter+Frozen"][index] = int(candidate_pred[topdown_index])
            moved["TopDownLOS-Filter+Frozen"][index] = True
        else:
            topdown_filter_diffs.append(0.0)
            predictions["TopDownLOS-Filter+Frozen"][index] = s1_pred
            moved["TopDownLOS-Filter+Frozen"][index] = True
        scene_index = int(np.argmax(scene_scores))
        predictions["SceneVisibility"][index] = int(candidate_pred[scene_index])
        moved["SceneVisibility"][index] = True
        median = float(np.median(scene_scores))
        retained = np.flatnonzero(scene_scores >= median)
        if retained.size:
            filtered_index = min(retained.tolist(), key=lambda j: (-utilities[j], geodesic[j], ids[j]))
            predictions["SceneVisibility-Filter+Frozen"][index] = int(candidate_pred[filtered_index])
            moved["SceneVisibility-Filter+Frozen"][index] = True
        else:
            predictions["SceneVisibility-Filter+Frozen"][index] = s1_pred
            moved["SceneVisibility-Filter+Frozen"][index] = True
        if s0_pred == label:
            predictions["AnyCorrect Oracle"][index] = s0_pred
        elif np.any(candidate_pred == label):
            predictions["AnyCorrect Oracle"][index] = label
            moved["AnyCorrect Oracle"][index] = True
        else:
            predictions["AnyCorrect Oracle"][index] = s0_pred
        correct_mask = candidate_pred == label
        if np.any(correct_mask):
            correct_visibility.extend(scene_scores[correct_mask].tolist())
        if np.any(~correct_mask):
            wrong_visibility.extend(scene_scores[~correct_mask].tolist())
        positive_mask = scene_scores > 0.0
        positive_visibility_count += int(np.sum(positive_mask))
        positive_visibility_correct += int(np.sum(positive_mask & correct_mask))
        zero_mask = ~positive_mask
        zero_visibility_count += int(np.sum(zero_mask))
        zero_visibility_correct += int(np.sum(zero_mask & correct_mask))
        rank_scores.extend(scene_scores.tolist())
        rank_truth.extend(true_logp[:, label].tolist())
        max_min.append(float(np.max(scene_scores) - np.min(scene_scores)))
        no_discrimination += int(np.ptp(scene_scores) <= 1e-8)
        los_equal += int(np.all(los == los[0]))
        topdown_pred.extend(los.tolist())
        candidate_visibility.append(visibility.astype(np.float32))
        candidate_ids.append(ids)
    methods = {name: _metrics(predictions[name], labels) for name in names}
    frozen = methods["FrozenStageCv0"]
    diagnostics = {
        "candidate_count": int(sum(len(item) for item in candidate_ids)),
        "scene_visibility_correct_mean": float(np.mean(correct_visibility)) if correct_visibility else 0.0,
        "scene_visibility_wrong_mean": float(np.mean(wrong_visibility)) if wrong_visibility else 0.0,
        "candidate_spearman_scene_visibility_vs_gt_logp": _spearman(rank_scores, rank_truth),
        "context_visibility_range_mean": float(np.mean(max_min)) if max_min else 0.0,
        "contexts_visibility_indistinguishable": {"count": no_discrimination, "rate": float(no_discrimination / len(bundles)) if bundles else 0.0},
        "contexts_all_topdown_los_equal": {"count": los_equal, "rate": float(los_equal / len(bundles)) if bundles else 0.0},
        "topdown_los1_candidate_fraction": float(np.mean(topdown_pred)) if topdown_pred else 0.0,
        "frozen_selection_removed_by_topdown_filter": {"count": int(np.sum(topdown_filter_diffs)), "rate": float(np.mean(topdown_filter_diffs)) if topdown_filter_diffs else 0.0},
        "candidate_correctness_by_visibility": {
            "positive_count": positive_visibility_count,
            "positive_correct_count": positive_visibility_correct,
            "zero_count": zero_visibility_count,
            "zero_correct_count": zero_visibility_correct,
            "p_correct_given_visibility_positive": float(positive_visibility_correct / positive_visibility_count) if positive_visibility_count else 0.0,
            "p_correct_given_visibility_zero": float(zero_visibility_correct / zero_visibility_count) if zero_visibility_count else 0.0,
        },
    }
    runtime = {
        "scene_visibility": np.asarray([np.asarray(bundle["scene_visibility"], dtype=np.float32) for bundle in bundles], dtype=object),
        "visibility": candidate_visibility,
        "candidate_ids": candidate_ids,
    }
    return methods, {"move_rates": {name: float(np.mean(moved[name])) for name in names}, "diagnostics": diagnostics}, runtime


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]
    frozen = methods["FrozenStageCv0"]
    scene = methods["SceneVisibility"]
    filtered = methods["SceneVisibility-Filter+Frozen"]
    gain_scene = 100.0 * (scene["accuracy"] - frozen["accuracy"])
    gain_filter = 100.0 * (filtered["accuracy"] - frozen["accuracy"])
    diag = result["diagnostics"]
    lines = [
        "# Reduced12 3D Scene Joint Visibility Oracle",
        "",
        "Val Moving contexts only. Static HM3D geometry was ray-cast from each candidate camera to six reconstructed GT world-space H36M17 poses. The ray-cast simulator contains no humanoid, so this is environmental occlusion only; the future archived skeleton is used only for terminal recognizer evaluation.",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("S0-only", "FrozenStageCv0", "TopDownLOS-Filter+Frozen", "SceneVisibility", "SceneVisibility-Filter+Frozen", "AnyCorrect Oracle"):
        metric = methods[name]
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {100.0 * (metric['accuracy'] - frozen['accuracy']):+.3f} | {100.0 * (metric['macro_f1'] - frozen['macro_f1']):+.3f} | {result['move_rates'][name]:.6f} |")
    lines.extend([
        "",
        "## Visibility diagnostics",
        "",
        f"Correct candidate mean visibility: {diag['scene_visibility_correct_mean']:.6f}; wrong candidate mean: {diag['scene_visibility_wrong_mean']:.6f}.",
        f"Candidate Spearman(scene visibility, GT-class true log-probability): {diag['candidate_spearman_scene_visibility_vs_gt_logp']:.6f}.",
        f"Mean within-context visibility range: {diag['context_visibility_range_mean']:.6f}; indistinguishable contexts: {diag['contexts_visibility_indistinguishable']['rate']:.6f}.",
        f"Top-down LOS=1 candidate fraction: {diag['topdown_los1_candidate_fraction']:.6f}; all-LOS-equal contexts: {diag['contexts_all_topdown_los_equal']['rate']:.6f}.",
        "",
        "## Scientific judgment",
        "",
    ])
    if gain_scene >= 1.0 or 100.0 * (scene["macro_f1"] - frozen["macro_f1"]) >= 1.0:
        lines.append(f"SceneVisibility gives a material gain over FrozenStageCv0 ({gain_scene:+.3f} pp Accuracy), indicating that true 3D environmental joint visibility has NBV value in this diagnostic.")
    else:
        lines.append(f"SceneVisibility does not materially exceed FrozenStageCv0 ({gain_scene:+.3f} pp Accuracy, {100.0 * (scene['macro_f1'] - frozen['macro_f1']):+.3f} pp Macro-F1); visibility-only NBV is not validated as a deployable selector.")
    if gain_filter >= 1.0 or 100.0 * (filtered["macro_f1"] - frozen["macro_f1"]) >= 1.0:
        lines.append(f"Median visibility filtering plus Frozen ranking improves the baseline ({gain_filter:+.3f} pp Accuracy), suggesting complementary geometry and HAR ranking signals.")
    else:
        lines.append("Median visibility filtering does not provide a material additional gain; no threshold or score tuning was performed.")
    lines.extend([
        "If gains remain small, the next diagnostic should examine view-dependent human observability and self-occlusion rather than tune this visibility score.",
        "",
        "Flags: `test_used=false`, `training_used=false`, `habitat_gt_scene_geometry_used=true`, `gt_future_human_world_joints_used_for_oracle_only=true`, `future_candidate_rgb_used=false`, `future_candidate_skeleton_used_only_for_terminal_evaluation=true`, `deployable=false`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(data_root: Path, scene_root: Path, output_dir: Path, max_contexts: int | None = None) -> dict[str, Any]:
    policy_root = data_root / "datasets" / DATASET_NAME
    stage_c_path = policy_root / "stage_c/features/val.jsonl"
    stage_c_predictions_path = policy_root / "stage_c/predictions/val_predictions.jsonl"
    stage_d_path = policy_root / "stage_d/features/val.jsonl"
    if not all(path.name == "val.jsonl" or path.name == "val_predictions.jsonl" for path in (stage_c_path, stage_d_path, stage_c_predictions_path)):
        raise ValueError("this diagnostic is Val-only")
    stage_c_rows = load_jsonl(stage_c_path)
    stage_c_predictions = load_jsonl(stage_c_predictions_path)
    moving_rows = load_jsonl(stage_d_path)
    if len(stage_c_rows) != len(stage_c_predictions):
        raise ValueError("Stage-C feature/prediction lengths differ")
    cache_path = policy_root / "counterfactual_cache/val.npz"
    cache = _load_npz(cache_path)
    if set(cache["episode_ids"].tolist()) != {str(row["episode_id"]) for row in moving_rows}:
        raise ValueError("Val Stage-D and counterfactual cache IDs are not aligned")
    selected_rows = moving_rows[:max_contexts] if max_contexts is not None else moving_rows
    bundles, build_summary = _build_visibility(data_root, scene_root, stage_c_rows, stage_c_predictions, selected_rows, cache, max_contexts=None)
    methods, extra, runtime = _evaluate(selected_rows, cache, bundles)
    labels = np.asarray([int(row["label_id"]) for row in selected_rows], dtype=np.int64)
    frozen = methods["FrozenStageCv0"]
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_SCENE_JOINT_VISIBILITY_ORACLE",
        "status": "COMPLETED",
        "population": {"val_moving_contexts": int(labels.size), "scene_count": int(build_summary["scene_count"]), "frame_samples": list(FRAME_IDS), "joint_count": JOINT_COUNT},
        "labels": list(LABELS),
        "methods": methods,
        "move_rates": extra["move_rates"],
        "deltas_vs_frozen": {name: {"accuracy_pp": 100.0 * (methods[name]["accuracy"] - frozen["accuracy"]), "macro_f1_pp": 100.0 * (methods[name]["macro_f1"] - frozen["macro_f1"])} for name in methods},
        "diagnostics": extra["diagnostics"],
        "protocol": {"visibility": "17 H36M joints, six frames, static HM3D ray cast; humanoid excluded from ray simulator", "frame_ids": list(FRAME_IDS), "sensor_height_m": SENSOR_HEIGHT_M, "ray_epsilon_m": RAY_EPSILON_M, "filter": "scene visibility >= context median, then FrozenStageCv0 utility", "terminal": "selected real archived skeleton through frozen reduced12 ST-GCN"},
        "leakage_flags": {"test_used": False, "training_used": False, "habitat_gt_scene_geometry_used": True, "gt_future_human_world_joints_used_for_oracle_only": True, "future_candidate_rgb_used": False, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "deployable": False},
        "artifacts": {"counterfactual_cache": str(cache_path.resolve()), "stage_c_val": str(stage_c_path.resolve()), "stage_d_val": str(stage_d_path.resolve()), "visibility_npz": str((output_dir / "visibility.npz").resolve())},
        "runtime": {"build": build_summary, "max_contexts": max_contexts},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    max_candidates = max(len(ids) for ids in runtime["candidate_ids"]) if runtime["candidate_ids"] else 0
    visibility_array = np.full((len(runtime["visibility"]), max_candidates, JOINT_COUNT), -1.0, dtype=np.float32)
    id_array = np.full((len(runtime["candidate_ids"]), max_candidates), -1, dtype=np.int64)
    mask_array = np.zeros((len(runtime["candidate_ids"]), max_candidates), dtype=bool)
    score_array = np.full((len(runtime["candidate_ids"]), max_candidates), -1.0, dtype=np.float32)
    los_array = np.full((len(runtime["candidate_ids"]), max_candidates), -1, dtype=np.int8)
    for index, (ids, vis, bundle) in enumerate(zip(runtime["candidate_ids"], runtime["visibility"], bundles)):
        count = len(ids)
        visibility_array[index, :count] = vis
        id_array[index, :count] = ids
        mask_array[index, :count] = True
        score_array[index, :count] = np.asarray(bundle["scene_visibility"], dtype=np.float32)
        los_array[index, :count] = np.asarray(bundle["topdown_los"], dtype=np.int8)
    np.savez_compressed(output_dir / "visibility.npz", episode_ids=np.asarray([str(row["episode_id"]) for row in selected_rows]), candidate_ids=id_array, candidate_mask=mask_array, visibility=visibility_array, scene_visibility_score=score_array, topdown_los=los_array)
    flat = visibility_array[mask_array]
    per_joint = {name: {"joint_index": index, "mean": float(np.mean(flat[:, index])), "median": float(np.median(flat[:, index])), "count": int(flat.shape[0])} for index, name in enumerate(H36M17_LINKS)}
    (output_dir / "per_joint_visibility_stats.json").write_text(json.dumps({"joints": per_joint, "visibility_npz": str((output_dir / "visibility.npz").resolve())}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(result["diagnostics"], indent=2) + "\n", encoding="utf-8")
    (output_dir / "per_class_metrics.json").write_text(json.dumps({name: methods[name]["per_class"] for name in methods}, indent=2) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-contexts", type=int, default=None, help="small Val smoke subset; omit for full Moving Val")
    args = parser.parse_args()
    if args.max_contexts is not None and args.max_contexts <= 0:
        raise ValueError("--max-contexts must be positive")
    result = evaluate(args.data_root.resolve(), args.scene_root.resolve(), args.output_dir.resolve(), args.max_contexts)
    print(json.dumps({"status": result["status"], "population": result["population"], "methods": {key: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]} for key, value in result["methods"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
