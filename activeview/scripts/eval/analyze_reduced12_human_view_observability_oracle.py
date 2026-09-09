#!/usr/bin/env python3
"""Val-only privileged human-view observability oracle for reduced12.

The diagnostic projects reconstructed GT world-space H36M17 joints into each
legal candidate camera.  It never uses future RGB/DINO or estimated future
skeletons for selection.  The selected archived candidate skeleton is used
only for the final frozen recognizer evaluation.
"""

from __future__ import annotations

import argparse
import json
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
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (
    ARCHIVE_REL,
    DATASET_NAME,
    FRAME_IDS,
    H36M17_LINKS,
    LABELS,
    JOINT_COUNT,
    NUM_CLASSES,
    _archive_path,
    _load_npz,
    _load_raw_records,
    _metrics,
    _placement_map,
    _scene_file,
    _scene_sim,
    _world_h36m17,
    _spearman,
)
from activeview.data.motion.babel_clean_dataset_generator import _load_resampled_motion
from activeview.data.motion.motion_converter import MotionConverter


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
HFOV_DEG = 75.0
SENSOR_HEIGHT_M = 1.10
EPS = 1.0e-8
JOINT_SEPARATION_PAIRS = ((13, 16), (12, 15), (5, 2), (6, 3))
H36M17_EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_view_observability_oracle"
SCENE_VISIBILITY_NPZ = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz"


def _rotation_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Return the Habitat agent local-to-world rotation for WXYZ."""
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"invalid WXYZ quaternion shape/value: {values}")
    norm = float(np.linalg.norm(values))
    if norm <= EPS:
        raise ValueError("zero-norm camera quaternion")
    w, x, y, z = values / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _project_joints(
    world_joints: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project frames x 17 world joints into the Habitat camera."""
    rotation = _rotation_wxyz_to_matrix(camera_rotation_wxyz)
    sensor = np.asarray(camera_position, dtype=np.float64) + np.asarray([0.0, SENSOR_HEIGHT_M, 0.0])
    camera = np.einsum("ij,ftj->fti", rotation.T, np.asarray(world_joints, dtype=np.float64) - sensor)
    depth = -camera[..., 2]
    front = np.isfinite(depth) & (depth > EPS)
    focal = 0.5 * IMAGE_WIDTH / np.tan(np.deg2rad(HFOV_DEG) * 0.5)
    x = focal * camera[..., 0] / np.maximum(depth, EPS) + 0.5 * IMAGE_WIDTH
    y = focal * (-camera[..., 1]) / np.maximum(depth, EPS) + 0.5 * IMAGE_HEIGHT
    finite = np.isfinite(x) & np.isfinite(y)
    in_fov = front & finite & (x >= 0.0) & (x < IMAGE_WIDTH) & (y >= 0.0) & (y < IMAGE_HEIGHT)
    return np.stack([x, y], axis=-1), front & finite, in_fov, depth


def _projected_area(points: np.ndarray, front: np.ndarray, in_fov: np.ndarray) -> float:
    valid = in_fov & np.isfinite(points).all(axis=-1)
    if int(np.sum(valid)) < 2:
        return 0.0
    xy = points[valid]
    area = float((np.max(xy[:, 0]) - np.min(xy[:, 0])) * (np.max(xy[:, 1]) - np.min(xy[:, 1])))
    area = float(np.clip(area / (IMAGE_WIDTH * IMAGE_HEIGHT), 0.0, 1.0))
    return area * float(np.mean(in_fov)) * float(np.mean(front))


def _limb_projection(world: np.ndarray, points: np.ndarray, front: np.ndarray) -> float:
    values: list[float] = []
    for left, right in H36M17_EDGES:
        valid = front[:, left] & front[:, right]
        if not np.any(valid):
            continue
        length_3d = np.linalg.norm(world[:, left] - world[:, right], axis=-1)
        length_2d = np.linalg.norm(points[:, left] - points[:, right], axis=-1)
        good = valid & np.isfinite(length_3d) & np.isfinite(length_2d) & (length_3d > EPS)
        if np.any(good):
            values.extend((length_2d[good] / length_3d[good]).tolist())
    return float(np.mean(values)) if values else 0.0


def _joint_separation(points: np.ndarray, front: np.ndarray) -> float:
    values: list[float] = []
    clipped = np.clip(points, [0.0, 0.0], [IMAGE_WIDTH - 1.0, IMAGE_HEIGHT - 1.0])
    for frame in range(points.shape[0]):
        valid_points = front[frame] & np.isfinite(clipped[frame]).all(axis=-1)
        if int(np.sum(valid_points)) < 2:
            continue
        body = clipped[frame, valid_points]
        diagonal = float(np.linalg.norm(np.max(body, axis=0) - np.min(body, axis=0)))
        if diagonal <= EPS:
            continue
        for left, right in JOINT_SEPARATION_PAIRS:
            if valid_points[left] and valid_points[right]:
                values.append(float(np.linalg.norm(clipped[frame, left] - clipped[frame, right]) / diagonal))
    return float(np.mean(values)) if values else 0.0


def _candidate_geometry_metrics(
    world_joints: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation_wxyz: np.ndarray,
) -> tuple[float, float, float]:
    points, front, in_fov, _ = _project_joints(world_joints, camera_position, camera_rotation_wxyz)
    return (
        _projected_area(points, front, in_fov),
        _limb_projection(world_joints, points, front),
        _joint_separation(points, front),
    )


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    low, high = float(np.min(values)), float(np.max(values))
    if high - low <= EPS:
        return np.zeros_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def _load_scene_visibility(path: Path, moving_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"required SceneVisibility artifact is missing: {path}")
    arrays = _load_npz(path)
    required = {"episode_ids", "candidate_ids", "candidate_mask", "scene_visibility_score"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"SceneVisibility artifact missing fields: {sorted(missing)}")
    expected = {str(row["episode_id"]) for row in moving_rows}
    observed = {str(value) for value in arrays["episode_ids"].tolist()}
    if not expected.issubset(observed):
        missing_expected = sorted(expected.difference(observed))[:3]
        raise ValueError(f"SceneVisibility episode IDs missing selected Moving Val contexts: {missing_expected}")
    output: dict[str, dict[int, float]] = {}
    for index, episode in enumerate(arrays["episode_ids"].tolist()):
        episode_id = str(episode)
        mask = np.asarray(arrays["candidate_mask"][index], dtype=bool)
        ids = np.asarray(arrays["candidate_ids"][index], dtype=np.int64)[mask]
        scores = np.asarray(arrays["scene_visibility_score"][index], dtype=np.float64)[mask]
        if len(ids) != len(set(ids.tolist())):
            raise ValueError(f"duplicate SceneVisibility candidate IDs for {episode_id}")
        output[episode_id] = {int(candidate): float(score) for candidate, score in zip(ids.tolist(), scores.tolist())}
    return output


def _build_metrics(
    data_root: Path,
    scene_root: Path,
    stage_c_rows: Sequence[Mapping[str, Any]],
    moving_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    scene_visibility: Mapping[str, Mapping[int, float]],
    max_contexts: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import habitat_sim  # noqa: F401 - require the Conda Habitat runtime

    stage_c_by_key = {
        (str(row["record_id"]), str(row["scene_id"]), str(row["region"]), int(row["current_viewpoint_id"])): row
        for row in stage_c_rows
    }
    cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    records = _load_raw_records(data_root)
    selected = list(moving_rows[:max_contexts]) if max_contexts is not None else list(moving_rows)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(selected):
        grouped[str(row["scene_id"])].append((index, row))
    bundles: list[dict[str, Any] | None] = [None] * len(selected)
    started = time.perf_counter()
    for scene_id, scene_rows in sorted(grouped.items()):
        human_sim = _scene_sim(scene_root, scene_id, physics=True)
        human = human_sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
        converter = MotionConverter(get_humanoid_urdf_path("male_0"))
        placements = _placement_map(data_root, scene_id)
        converted_cache: dict[str, Mapping[str, Any]] = {}
        joints_cache: dict[tuple[str, str], np.ndarray] = {}
        try:
            for local_index, row in scene_rows:
                key = (str(row["record_id"]), scene_id, str(row["region"]), int(row["s0_viewpoint_id"]))
                if key not in stage_c_by_key:
                    raise ValueError(f"missing Stage-C context: {key}")
                episode_id = str(row["episode_id"])
                if episode_id not in cache_index or episode_id not in scene_visibility:
                    raise ValueError(f"missing cache/visibility episode: {episode_id}")
                stage_row = stage_c_by_key[key]
                candidate_ids = np.asarray(stage_row["candidate_viewpoint_ids"], dtype=np.int64)
                cache_i = int(cache_index[episode_id])
                label = int(row["label_id"])
                if int(cache["label_id"][cache_i]) != label:
                    raise ValueError(f"label/cache mismatch: {episode_id}")
                archive = _load_npz(_archive_path(data_root, row))
                viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
                rotations = np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32)
                if positions.shape != (32, 3) or rotations.shape != (32, 4):
                    raise ValueError(f"invalid camera metadata shape for {episode_id}")
                index_by_viewpoint = {int(value): index for index, value in enumerate(viewpoint_ids.tolist())}
                if not set(candidate_ids.tolist()).issubset(index_by_viewpoint):
                    raise ValueError(f"candidate/archive viewpoint mismatch: {episode_id}")
                scene_scores_by_id = scene_visibility[episode_id]
                if set(candidate_ids.tolist()) != set(scene_scores_by_id):
                    raise ValueError(f"SceneVisibility candidate mismatch: {episode_id}")
                placement_id = str(row["region"])
                if placement_id not in placements:
                    raise ValueError(f"unknown placement {placement_id} for {scene_id}")
                record_id = str(row["record_id"])
                if record_id not in records:
                    raise ValueError(f"record missing from raw-val: {record_id}")
                if record_id not in converted_cache:
                    converted_cache[record_id] = converter.convert(_load_resampled_motion(records[record_id], 30))
                joint_key = (record_id, placement_id)
                if joint_key not in joints_cache:
                    placement = placements[placement_id]
                    joints_cache[joint_key] = _world_h36m17(
                        human, converted_cache[record_id], placement, float(placement["yaw_deg"])
                    )
                world_joints = joints_cache[joint_key]
                projected_area: list[float] = []
                limb_projection: list[float] = []
                joint_separation: list[float] = []
                scene_scores: list[float] = []
                for candidate_id in candidate_ids.tolist():
                    camera_i = index_by_viewpoint[int(candidate_id)]
                    area, limb, separation = _candidate_geometry_metrics(
                        world_joints, positions[camera_i], rotations[camera_i]
                    )
                    projected_area.append(area)
                    limb_projection.append(limb)
                    joint_separation.append(separation)
                    scene_scores.append(float(scene_scores_by_id[int(candidate_id)]))
                bundles[local_index] = {
                    "episode_id": episode_id,
                    "cache_index": cache_i,
                    "candidate_ids": candidate_ids,
                    "projected_area": np.asarray(projected_area, dtype=np.float64),
                    "limb_projection": np.asarray(limb_projection, dtype=np.float64),
                    "joint_separation": np.asarray(joint_separation, dtype=np.float64),
                    "scene_visibility": np.asarray(scene_scores, dtype=np.float64),
                    "label": label,
                    "s0_viewpoint_id": int(row["s0_viewpoint_id"]),
                }
                if (local_index + 1) % 100 == 0:
                    print(f"human-view-observability: {local_index + 1}/{len(selected)} contexts ({time.perf_counter() - started:.1f}s)", flush=True)
        finally:
            human_sim.close()
    if any(bundle is None for bundle in bundles):
        raise RuntimeError("failed to build human-view metrics for one or more contexts")
    return [bundle for bundle in bundles if bundle is not None], {
        "scene_count": len(grouped),
        "scene_ids": sorted(grouped),
        "frame_ids": list(FRAME_IDS),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _select_index(scores: np.ndarray) -> int:
    if scores.size == 0:
        raise ValueError("empty candidate score array")
    return int(np.argmax(scores))


def _evaluate(
    moving_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    bundles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any], dict[str, np.ndarray]]:
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    names = (
        "S0-only", "FrozenStageCv0", "SceneVisibility", "MaxProjectedArea",
        "MaxLimbProjection", "MaxJointSeparation", "HumanObservability",
        "SceneHumanObservability", "AnyCorrect Oracle",
    )
    predictions = {name: np.empty(labels.size, dtype=np.int64) for name in names}
    moved = {name: np.zeros(labels.size, dtype=bool) for name in names}
    metric_names = ("projected_area", "limb_projection", "joint_separation", "human_observability", "scene_visibility", "scene_human_observability")
    flat_metrics: dict[str, list[float]] = {name: [] for name in metric_names}
    flat_truth: list[float] = []
    correct_values: dict[str, list[float]] = {name: [] for name in metric_names}
    wrong_values: dict[str, list[float]] = {name: [] for name in metric_names}
    context_ranges: dict[str, list[float]] = {name: [] for name in ("projected_area", "limb_projection", "joint_separation", "human_observability", "scene_visibility", "scene_human_observability")}
    for index, (row, bundle) in enumerate(zip(moving_rows, bundles)):
        cache_i = int(bundle["cache_index"])
        label = int(labels[index])
        candidate_ids = np.asarray(bundle["candidate_ids"], dtype=np.int64)
        true_logp = np.asarray(cache["true_logp"][cache_i, candidate_ids], dtype=np.float64)
        candidate_predictions = np.argmax(true_logp, axis=1)
        current_s0 = int(np.argmax(cache["current_logp_s0"][cache_i]))
        current_s1 = int(np.argmax(cache["current_logp_s1"][cache_i]))
        predictions["S0-only"][index] = current_s0
        predictions["FrozenStageCv0"][index] = current_s1
        moved["FrozenStageCv0"][index] = True
        raw = {name: np.asarray(bundle[name], dtype=np.float64) for name in ("projected_area", "limb_projection", "joint_separation", "scene_visibility")}
        normalized = {name: _minmax(values) for name, values in raw.items()}
        normalized["human_observability"] = normalized["projected_area"] + normalized["limb_projection"] + normalized["joint_separation"]
        normalized["scene_human_observability"] = 0.5 * _minmax(raw["scene_visibility"]) + 0.5 * _minmax(normalized["human_observability"])
        all_scores = {**raw, "human_observability": normalized["human_observability"], "scene_human_observability": normalized["scene_human_observability"]}
        for metric_name, values in all_scores.items():
            flat_metrics[metric_name].extend(values.tolist())
            context_ranges[metric_name].append(float(np.max(values) - np.min(values)))
            correct_mask = candidate_predictions == label
            correct_values[metric_name].extend(values[correct_mask].tolist())
            wrong_values[metric_name].extend(values[~correct_mask].tolist())
        flat_truth.extend(true_logp[:, label].tolist())
        selected = {
            "SceneVisibility": _select_index(normalized["scene_visibility"]),
            "MaxProjectedArea": _select_index(raw["projected_area"]),
            "MaxLimbProjection": _select_index(raw["limb_projection"]),
            "MaxJointSeparation": _select_index(raw["joint_separation"]),
            "HumanObservability": _select_index(normalized["human_observability"]),
            "SceneHumanObservability": _select_index(normalized["scene_human_observability"]),
        }
        for method, candidate_index in selected.items():
            predictions[method][index] = int(candidate_predictions[candidate_index])
            moved[method][index] = True
        if current_s0 == label:
            predictions["AnyCorrect Oracle"][index] = current_s0
        elif np.any(candidate_predictions == label):
            predictions["AnyCorrect Oracle"][index] = label
            moved["AnyCorrect Oracle"][index] = True
        else:
            predictions["AnyCorrect Oracle"][index] = current_s0
    methods = {name: _metrics(predictions[name], labels) for name in names}
    diagnostics = {
        "candidate_count": int(sum(len(bundle["candidate_ids"]) for bundle in bundles)),
        "candidate_spearman_vs_gt_true_logp": {name: _spearman(flat_metrics[name], flat_truth) for name in metric_names},
        "mean_metric_if_candidate_stgcn_correct": {name: float(np.mean(correct_values[name])) if correct_values[name] else 0.0 for name in metric_names},
        "mean_metric_if_candidate_stgcn_wrong": {name: float(np.mean(wrong_values[name])) if wrong_values[name] else 0.0 for name in metric_names},
        "mean_context_metric_range": {name: float(np.mean(context_ranges[name])) if context_ranges[name] else 0.0 for name in metric_names},
        "contexts_metric_indistinguishable": {name: {"count": int(np.sum(np.asarray(context_ranges[name]) <= 1.0e-8)), "rate": float(np.mean(np.asarray(context_ranges[name]) <= 1.0e-8))} for name in metric_names},
    }
    per_metric = {
        name: {
            "mean": float(np.mean(flat_metrics[name])) if flat_metrics[name] else 0.0,
            "median": float(np.median(flat_metrics[name])) if flat_metrics[name] else 0.0,
            "std": float(np.std(flat_metrics[name])) if flat_metrics[name] else 0.0,
            "spearman_vs_gt_true_logp": diagnostics["candidate_spearman_vs_gt_true_logp"][name],
            "correct_candidate_mean": diagnostics["mean_metric_if_candidate_stgcn_correct"][name],
            "wrong_candidate_mean": diagnostics["mean_metric_if_candidate_stgcn_wrong"][name],
            "context_range_mean": diagnostics["mean_context_metric_range"][name],
            "indistinguishable_context_rate": diagnostics["contexts_metric_indistinguishable"][name]["rate"],
        }
        for name in metric_names
    }
    runtime = {name: np.asarray(flat_metrics[name], dtype=np.float32) for name in metric_names}
    return methods, {name: float(np.mean(moved[name])) for name in names}, diagnostics, {"per_metric": per_metric, "flat": runtime}


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]
    frozen = methods["FrozenStageCv0"]
    scene = methods["SceneVisibility"]
    human = methods["HumanObservability"]
    scene_human = methods["SceneHumanObservability"]
    lines = [
        "# Reduced12 Human View Observability Oracle",
        "",
        "Val Moving contexts only. GT world-space H36M17 joints were projected into legal candidate cameras using the fixed 256x256/HFOV 75 degree protocol. This is a privileged geometry diagnostic; future RGB and estimated future skeletons were not used for selection.",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("S0-only", "FrozenStageCv0", "SceneVisibility", "MaxProjectedArea", "MaxLimbProjection", "MaxJointSeparation", "HumanObservability", "SceneHumanObservability", "AnyCorrect Oracle"):
        metric = methods[name]
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {100.0 * (metric['accuracy'] - frozen['accuracy']):+.3f} | {100.0 * (metric['macro_f1'] - frozen['macro_f1']):+.3f} | {result['move_rates'][name]:.6f} |")
    lines.extend(["", "## Candidate diagnostics", ""])
    for name, stats in result["per_metric_stats"].items():
        lines.append(f"- **{name}**: Spearman with GT-class true logp `{stats['spearman_vs_gt_true_logp']:.6f}`; correct-candidate mean `{stats['correct_candidate_mean']:.6f}` vs wrong `{stats['wrong_candidate_mean']:.6f}`; indistinguishable-context rate `{stats['indistinguishable_context_rate']:.6f}`.")
    lines.extend(["", "## Scientific judgment", ""])
    human_gain = 100.0 * (human["accuracy"] - frozen["accuracy"])
    scene_gain = 100.0 * (scene["accuracy"] - frozen["accuracy"])
    combined_gain = 100.0 * (scene_human["accuracy"] - frozen["accuracy"])
    if human_gain >= 1.0 or 100.0 * (human["macro_f1"] - frozen["macro_f1"]) >= 1.0:
        lines.append(f"HumanObservability materially exceeds FrozenStageCv0 ({human_gain:+.3f} pp Accuracy), so view-dependent human geometry has diagnostic NBV value.")
    else:
        lines.append(f"HumanObservability does not materially exceed FrozenStageCv0 ({human_gain:+.3f} pp Accuracy); this privileged geometry-only score is not sufficient as a selector.")
    if combined_gain >= 1.0 or 100.0 * (scene_human["macro_f1"] - frozen["macro_f1"]) >= 1.0:
        lines.append(f"The fixed Scene+Human combination gives a material gain ({combined_gain:+.3f} pp Accuracy), suggesting complementary scene and human geometry signals.")
    else:
        lines.append(f"The fixed Scene+Human combination gives no material additional gain over Frozen ({combined_gain:+.3f} pp Accuracy); no weight tuning was performed.")
    lines.append(f"For reference, SceneVisibility alone changes Accuracy by {scene_gain:+.3f} pp versus Frozen; if all geometry-only gains remain small, the next diagnostic should examine realistic human self-occlusion/view-dependent observability rather than tune weights.")
    lines.extend(["", "Flags: `test_used=false`, `training_used=false`, `gt_future_human_geometry_used_for_oracle_only=true`, `future_candidate_rgb_used=false`, `future_candidate_skeleton_used_only_for_terminal_evaluation=true`, `deployable=false`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(data_root: Path, scene_root: Path, output_dir: Path, scene_visibility_path: Path, max_contexts: int | None = None) -> dict[str, Any]:
    policy_root = data_root / "datasets" / DATASET_NAME
    stage_c_path = policy_root / "stage_c/features/val.jsonl"
    stage_d_path = policy_root / "stage_d/features/val.jsonl"
    cache_path = policy_root / "counterfactual_cache/val.npz"
    if "test" in str(stage_c_path).lower() or "test" in str(stage_d_path).lower() or "test" in str(cache_path).lower():
        raise ValueError("this diagnostic is Val-only")
    stage_c_rows = load_jsonl(stage_c_path)
    moving_rows = load_jsonl(stage_d_path)
    cache = _load_npz(cache_path)
    if set(cache["episode_ids"].tolist()) != {str(row["episode_id"]) for row in moving_rows}:
        raise ValueError("Val Stage-D and counterfactual cache IDs are not aligned")
    selected = list(moving_rows[:max_contexts]) if max_contexts is not None else list(moving_rows)
    scene_visibility = _load_scene_visibility(scene_visibility_path, selected if max_contexts is not None else moving_rows)
    bundles, build_summary = _build_metrics(data_root, scene_root, stage_c_rows, selected, cache, scene_visibility, None)
    methods, move_rates, diagnostics, metric_payload = _evaluate(selected, cache, bundles)
    frozen = methods["FrozenStageCv0"]
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_HUMAN_VIEW_OBSERVABILITY_ORACLE",
        "status": "COMPLETED",
        "population": {"val_moving_contexts": len(selected), "scene_count": build_summary["scene_count"], "frame_samples": list(FRAME_IDS), "joint_count": JOINT_COUNT, "candidate_count": diagnostics["candidate_count"]},
        "labels": list(LABELS),
        "methods": methods,
        "move_rates": move_rates,
        "deltas_vs_frozen": {name: {"accuracy_pp": 100.0 * (metric["accuracy"] - frozen["accuracy"]), "macro_f1_pp": 100.0 * (metric["macro_f1"] - frozen["macro_f1"])} for name, metric in methods.items()},
        "diagnostics": diagnostics,
        "per_metric_stats": metric_payload["per_metric"],
        "protocol": {"projection": "Habitat camera 256x256, HFOV 75 degrees, sensor height 1.10m; WXYZ rotation; camera forward is -Z", "projected_area": "in-FOV 17-joint bbox area/image area multiplied by front and in-FOV fractions", "limb_projection": "mean projected 2D limb length / GT 3D limb length over H36M17 edges and six frames", "joint_separation": "mean left/right wrist/elbow/knee/ankle separation normalized by projected body bbox diagonal", "normalization": "within-context legal-candidate min-max; equal weights only", "terminal": "selected real archived skeleton through frozen reduced12 ST-GCN"},
        "leakage_flags": {"test_used": False, "training_used": False, "gt_future_human_geometry_used_for_oracle_only": True, "future_candidate_rgb_used": False, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "deployable": False},
        "artifacts": {"counterfactual_cache": str(cache_path.resolve()), "stage_c_val": str(stage_c_path.resolve()), "stage_d_val": str(stage_d_path.resolve()), "scene_visibility_npz": str(scene_visibility_path.resolve()), "candidate_metrics_npz": str((output_dir / "candidate_metrics.npz").resolve())},
        "runtime": {"build": build_summary, "max_contexts": max_contexts},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    max_candidates = max(len(bundle["candidate_ids"]) for bundle in bundles)
    episode_ids = np.asarray([str(bundle["episode_id"]) for bundle in bundles])
    candidate_ids = np.full((len(bundles), max_candidates), -1, dtype=np.int64)
    candidate_mask = np.zeros((len(bundles), max_candidates), dtype=bool)
    arrays = {name: np.full((len(bundles), max_candidates), np.nan, dtype=np.float32) for name in ("projected_area", "limb_projection", "joint_separation", "human_observability", "scene_visibility", "scene_human_observability")}
    for index, bundle in enumerate(bundles):
        count = len(bundle["candidate_ids"])
        candidate_ids[index, :count] = bundle["candidate_ids"]
        candidate_mask[index, :count] = True
        raw = {name: np.asarray(bundle[name], dtype=np.float64) for name in ("projected_area", "limb_projection", "joint_separation", "scene_visibility")}
        normalized = {name: _minmax(values) for name, values in raw.items()}
        normalized["human_observability"] = normalized["projected_area"] + normalized["limb_projection"] + normalized["joint_separation"]
        normalized["scene_human_observability"] = 0.5 * normalized["scene_visibility"] + 0.5 * _minmax(normalized["human_observability"])
        all_scores = {**raw, "human_observability": normalized["human_observability"], "scene_human_observability": normalized["scene_human_observability"]}
        for name, values in all_scores.items():
            arrays[name][index, :count] = values.astype(np.float32)
    np.savez_compressed(output_dir / "candidate_metrics.npz", episode_ids=episode_ids, candidate_ids=candidate_ids, candidate_mask=candidate_mask, **arrays)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "per_metric_stats.json").write_text(json.dumps(metric_payload["per_metric"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "per_class_metrics.json").write_text(json.dumps({name: methods[name]["per_class"] for name in methods}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--scene-visibility", type=Path, default=SCENE_VISIBILITY_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-contexts", type=int, default=None, help="small Val smoke subset; omit for full Moving Val")
    args = parser.parse_args()
    if args.max_contexts is not None and args.max_contexts <= 0:
        raise ValueError("--max-contexts must be positive")
    result = evaluate(args.data_root.resolve(), args.scene_root.resolve(), args.output_dir.resolve(), args.scene_visibility.resolve(), args.max_contexts)
    print(json.dumps({"status": result["status"], "population": result["population"], "methods": {key: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]} for key, value in result["methods"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
