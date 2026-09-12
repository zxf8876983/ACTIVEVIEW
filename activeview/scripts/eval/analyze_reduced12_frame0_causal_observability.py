#!/usr/bin/env python3
"""Val-only audit of causal frame-0 scene visibility for viewpoint selection.

The selector sees only the frame-0 reconstructed human pose, static Habitat
scene geometry and the matched Stay + Stage-A legal candidate set.  Existing
full-temporal candidate visibility is used only as a matched reference; all
terminal HAR predictions come from the frozen shared recognizer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.scripts.eval.analyze_reduced12_real_candidate_quality_privileged import (
    _archive_confidence,
    _load_shared_logp,
)
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (
    FRAME_IDS,
    H36M17_LINKS,
    SENSOR_HEIGHT_M,
    _archive_path as scene_archive_path,
    _load_npz,
    _load_raw_records,
    _load_resampled_motion,
    _placement_map,
    _ray_visible,
    _scene_sim,
    _world_h36m17,
)
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import load_rows

NUM_CLASSES = len(LABELS)
JOINT_COUNT = len(H36M17_LINKS)
NUM_VIEWS = 32
SEED = 42
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/frame0_causal_observability_audit"
ARCHIVE_REL = "datasets/offline/habitat-train/00006-00087"
SCENE_VISIBILITY_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle"
HISTORICAL_RESULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/historical_route1_shared_head_synergy/result.json"


def _action_set(row: Mapping[str, Any]) -> list[int]:
    current = int(row["current_viewpoint_id"])
    return list(dict.fromkeys([current] + [int(value) for value in row["candidate_ids"]]))


def _slot(ids: np.ndarray, mask: np.ndarray, action: int) -> int:
    found = np.flatnonzero((np.asarray(ids) == int(action)) & np.asarray(mask, dtype=bool))
    if found.size != 1:
        raise ValueError(f"viewpoint {action} is not uniquely present")
    return int(found[0])


def _select_max_quality(actions: Sequence[int], current: int, scores: Mapping[int, float]) -> int:
    if any(not np.isfinite(float(scores[action])) for action in actions):
        raise ValueError("non-finite frame-0/full-temporal quality in legal action set")
    # Stay wins exact ties; moving ties use the smallest viewpoint id.
    return min(actions, key=lambda action: (-float(scores[action]), 0 if int(action) == int(current) else 1, int(action)))


def _load_full_temporal_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[int, float]], dict[str, Any]]:
    path = SCENE_VISIBILITY_ROOT / "visibility.npz"
    artifact = _load_npz(path)
    episode_ids = np.asarray(artifact["episode_ids"]).astype(str)
    expected_ids = np.asarray([str(row["episode_id"]) for row in rows])
    if not np.array_equal(episode_ids, expected_ids):
        raise ValueError("full-temporal SceneVisibility artifact is not aligned to Moving Val rows")
    scores: list[dict[int, float]] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(np.asarray(artifact["candidate_mask"][index], dtype=bool))
        candidate_ids = np.asarray(artifact["candidate_ids"][index, valid], dtype=np.int64)
        expected = np.asarray(row["candidate_ids"], dtype=np.int64)
        if not np.array_equal(candidate_ids, expected):
            raise ValueError(f"full-temporal candidate mismatch at {row['episode_id']}")
        values = np.asarray(artifact["scene_visibility_score"][index, valid], dtype=np.float64)
        if values.shape != candidate_ids.shape or not np.isfinite(values).all():
            raise ValueError(f"invalid full-temporal visibility at {row['episode_id']}")
        scores.append({int(view): float(value) for view, value in zip(candidate_ids, values)})
    metadata = {
        "source": str(path.resolve()),
        "contexts": len(scores),
        "candidate_coverage": 1.0,
        "frame_ids": list(FRAME_IDS),
        "definition": "existing six-frame environmental H36M17 scene raycast; candidate-only source",
    }
    return scores, metadata


def _load_historical_actions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    payload = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
    actions = [int(value) for value in payload.get("selected_actions", [])]
    if len(actions) != len(rows):
        raise ValueError("historical Route-1 result does not match Moving Val context count")
    for row, action in zip(rows, actions):
        if action not in _action_set(row):
            raise ValueError(f"historical action {action} is outside legal set for {row['episode_id']}")
    return actions


def _build_frame0_scores(
    data_root: Path,
    scene_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[int, float]], list[float], dict[str, Any]]:
    """Ray-cast frame 0 for Stay + candidates and six-frame Stay reference."""
    raw_records = _load_raw_records(data_root)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["scene_id"])].append((index, row))
    frame0_scores: list[dict[int, float] | None] = [None] * len(rows)
    full_stay_scores = np.full(len(rows), np.nan, dtype=np.float64)
    archive_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    converted_cache: dict[str, Mapping[str, Any]] = {}
    world_cache: dict[tuple[str, str, str], np.ndarray] = {}
    started = time.perf_counter()
    processed = 0
    for scene_id, scene_rows in sorted(grouped.items()):
        ray_sim = _scene_sim(scene_root, scene_id, physics=True)
        human_sim = _scene_sim(scene_root, scene_id, physics=True)
        human = None
        try:
            from activeview.core.paths import get_humanoid_urdf_path
            from activeview.data.motion.motion_converter import MotionConverter

            urdf = get_humanoid_urdf_path("male_0")
            human = human_sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(urdf))
            converter = MotionConverter(urdf)
            placements = _placement_map(data_root, scene_id)
            for index, row in scene_rows:
                episode_id = str(row["episode_id"])
                archive_path = scene_archive_path(data_root, row)
                archive_key = str(archive_path)
                if archive_key not in archive_cache:
                    archive = _load_npz(archive_path)
                    view_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                    camera_positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
                    if view_ids.shape != (NUM_VIEWS,) or camera_positions.shape != (NUM_VIEWS, 3):
                        raise ValueError(f"invalid viewpoint archive schema: {archive_path}")
                    archive_cache[archive_key] = view_ids, camera_positions
                view_ids, camera_positions = archive_cache[archive_key]
                view_index = {int(view): offset for offset, view in enumerate(view_ids.tolist())}
                placement_id = str(row["region"])
                if placement_id not in placements:
                    raise ValueError(f"unknown placement {placement_id} for {scene_id}")
                record_id = str(row["record_id"])
                if record_id not in raw_records:
                    raise ValueError(f"record not present in raw-val manifest: {record_id}")
                if record_id not in converted_cache:
                    converted_cache[record_id] = converter.convert(_load_resampled_motion(raw_records[record_id], 30))
                world_key = (scene_id, placement_id, record_id)
                if world_key not in world_cache:
                    placement = placements[placement_id]
                    world_cache[world_key] = _world_h36m17(
                        human,
                        converted_cache[record_id],
                        placement,
                        float(placement["yaw_deg"]),
                    )
                world_joints = world_cache[world_key]
                placement = placements[placement_id]
                actions = _action_set(row)
                values: dict[int, float] = {}
                for action in actions:
                    if action not in view_index:
                        raise ValueError(f"viewpoint missing from archive: {episode_id}/{action}")
                    camera = camera_positions[view_index[action]] + np.asarray([0.0, SENSOR_HEIGHT_M, 0.0], dtype=np.float32)
                    visible = [_ray_visible(ray_sim, camera, world_joints[0, joint]) for joint in range(JOINT_COUNT)]
                    values[action] = float(np.mean(visible))
                frame0_scores[index] = values
                stay = int(row["current_viewpoint_id"])
                if stay not in view_index:
                    raise ValueError(f"current viewpoint missing from archive: {episode_id}/{stay}")
                stay_camera = camera_positions[view_index[stay]] + np.asarray([0.0, SENSOR_HEIGHT_M, 0.0], dtype=np.float32)
                all_visible = [
                    _ray_visible(ray_sim, stay_camera, world_joints[frame, joint])
                    for frame in range(len(FRAME_IDS))
                    for joint in range(JOINT_COUNT)
                ]
                full_stay_scores[index] = float(np.mean(all_visible))
                processed += 1
                if processed % 100 == 0:
                    print(f"frame0-causal: {processed}/{len(rows)} contexts ({time.perf_counter() - started:.1f}s)", flush=True)
        finally:
            human_sim.close()
            ray_sim.close()
    if any(item is None for item in frame0_scores) or not np.isfinite(full_stay_scores).all():
        raise RuntimeError("failed to compute frame-0 or Stay full-temporal visibility for a context")
    return [item for item in frame0_scores if item is not None], full_stay_scores.tolist(), {
        "scene_count": len(grouped),
        "scene_ids": sorted(grouped),
        "contexts": len(rows),
        "frame_ids": [0],
        "full_stay_frame_ids": list(FRAME_IDS),
        "archive_files_read": len(archive_cache),
        "elapsed_seconds": time.perf_counter() - started,
        "frame0_context_coverage": 1.0,
        "full_stay_context_coverage": 1.0,
    }


def _terminal_predictions(
    rows: Sequence[Mapping[str, Any]],
    ids: np.ndarray,
    mask: np.ndarray,
    logp: np.ndarray,
    actions: Sequence[int],
) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        slot = _slot(ids[index], mask[index], int(action))
        predictions.append(int(np.argmax(logp[index, slot])))
    return np.asarray(predictions, dtype=np.int64)


def _metrics(labels: np.ndarray, predictions: np.ndarray, actions: Sequence[int], rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    result = classification(labels, predictions)
    move_rate = float(np.mean([int(action) != int(row["current_viewpoint_id"]) for action, row in zip(actions, rows)]))
    result.update({"selector": name, "move_rate": move_rate, "stay_rate": 1.0 - move_rate, "test_used": False})
    return result


def _transition(labels: np.ndarray, left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> dict[str, Any]:
    left_correct = left == labels
    right_correct = right == labels
    return {
        "left": left_name,
        "right": right_name,
        "left_wrong_right_correct": int(np.sum(~left_correct & right_correct)),
        "left_correct_right_wrong": int(np.sum(left_correct & ~right_correct)),
        "both_correct": int(np.sum(left_correct & right_correct)),
        "both_wrong": int(np.sum(~left_correct & ~right_correct)),
        "right_minus_left_accuracy_pp": float(100.0 * np.mean(right_correct.astype(np.float64) - left_correct.astype(np.float64))),
    }


def _sorted_actions(actions: Sequence[int], current: int, scores: Mapping[int, float]) -> list[int]:
    return sorted(actions, key=lambda action: (-float(scores[action]), 0 if int(action) == int(current) else 1, int(action)))


def _ranking_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    ids: np.ndarray,
    mask: np.ndarray,
    logp: np.ndarray,
    frame0: Sequence[Mapping[int, float]],
    full: Sequence[Mapping[int, float]],
    frame0_actions: Sequence[int],
    full_actions: Sequence[int],
    gt_actions: Sequence[int],
) -> dict[str, Any]:
    frame_full_x: list[float] = []
    frame_full_y: list[float] = []
    frame_gt_x: list[float] = []
    frame_gt_y: list[float] = []
    within_full: list[float] = []
    within_gt: list[float] = []
    frame_full_top1: list[int] = []
    frame_full_top3: list[int] = []
    frame_gt_top1: list[int] = []
    frame_gt_top3: list[int] = []
    selected_ranks: list[int] = []
    for index, row in enumerate(rows):
        actions = _action_set(row)
        label = int(row["label_id"])
        gt_values = {
            action: float(logp[index, _slot(ids[index], mask[index], action), label])
            for action in actions
        }
        x = [float(frame0[index][action]) for action in actions]
        y = [float(full[index][action]) for action in actions]
        z = [gt_values[action] for action in actions]
        frame_full_x.extend(x); frame_full_y.extend(y); frame_gt_x.extend(x); frame_gt_y.extend(z)
        if len(actions) >= 2:
            within_full.append(correlation(x, y, spearman=True))
            within_gt.append(correlation(x, z, spearman=True))
        full_ranked = _sorted_actions(actions, int(row["current_viewpoint_id"]), full[index])
        frame_ranked = _sorted_actions(actions, int(row["current_viewpoint_id"]), frame0[index])
        frame_full_top1.append(int(frame_ranked[0] == full_ranked[0]))
        frame_full_top3.append(int(full_ranked[0] in frame_ranked[:3]))
        frame_gt_top1.append(int(frame_ranked[0] == int(gt_actions[index])))
        frame_gt_top3.append(int(int(gt_actions[index]) in frame_ranked[:3]))
        selected_ranks.append(full_ranked.index(int(frame0_actions[index])) + 1)
    return {
        "candidate_action_count": len(frame_gt_x),
        "frame0_vs_full_temporal": {
            "candidate_spearman": correlation(frame_full_x, frame_full_y, spearman=True),
            "within_context_spearman_mean": float(np.mean(within_full)),
            "within_context_spearman_median": float(np.median(within_full)),
            "within_context_count": len(within_full),
            "selected_view_agreement": float(np.mean(frame_full_top1)),
            "top3_overlap": float(np.mean(frame_full_top3)),
            "frame0_rank_within_full_mean": float(np.mean(selected_ranks)),
            "frame0_rank_within_full_median": float(np.median(selected_ranks)),
        },
        "frame0_vs_shared_gt_true_logp": {
            "candidate_spearman": correlation(frame_gt_x, frame_gt_y, spearman=True),
            "within_context_spearman_mean": float(np.mean(within_gt)),
            "within_context_spearman_median": float(np.median(within_gt)),
            "within_context_count": len(within_gt),
            "gt_true_logp_top1_overlap": float(np.mean(frame_gt_top1)),
            "gt_true_logp_top3_overlap": float(np.mean(frame_gt_top3)),
        },
    }


def _occlusion_stratification(
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, np.ndarray],
    stay_visibility: np.ndarray,
) -> dict[str, Any]:
    q33, q67 = np.quantile(stay_visibility, [1.0 / 3.0, 2.0 / 3.0])
    groups = {
        "low_occlusion": stay_visibility >= q67,
        "medium_occlusion": (stay_visibility >= q33) & (stay_visibility < q67),
        "high_occlusion": stay_visibility < q33,
    }
    output: dict[str, Any] = {"quantiles": {"q33_stay_frame0_visibility": float(q33), "q67_stay_frame0_visibility": float(q67)}, "groups": {}}
    for name, selected in groups.items():
        indices = np.flatnonzero(selected)
        output["groups"][name] = {
            "count": int(indices.size),
            "stay": classification(labels[indices], predictions["Stay"][indices]),
            "random_legal": classification(labels[indices], predictions["Random legal"][indices]),
            "frame0_scene_visibility": classification(labels[indices], predictions["Frame0SceneVisibility"][indices]),
            "full_temporal_scene_visibility": classification(labels[indices], predictions["FullTemporalSceneVisibility"][indices]),
            "historical_route1": classification(labels[indices], predictions["Historical Route-1"][indices]),
        }
    return output


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    frame = methods["Frame0SceneVisibility"]
    full = methods["FullTemporalSceneVisibility"]
    random = methods["Random legal"]
    historical = methods["Historical Route-1"]
    stay = methods["Stay"]
    frame_gain_random = 100.0 * (frame["accuracy"] - random["accuracy"])
    frame_gain_historical = 100.0 * (frame["accuracy"] - historical["accuracy"])
    full_gap = 100.0 * (full["accuracy"] - frame["accuracy"])
    ranking = result["ranking_diagnostics"]
    protocol_lines = [
        "# Frame-0 Causal Observability Audit",
        "",
        "## Protocol",
        "",
        "- Moving Val contexts = 10,080",
        "- Policy Test used = false",
        "- training = none",
        "- recognizer = frozen reduced12 ST-GCN feature cache + shared adapted head",
        "- decision-time human pose = frame 0 only for Frame0SceneVisibility",
        "- future motion frames used by Frame0 selector = false",
        "- GT action used by selector = false",
        "- candidate recognizer evidence used by selector = false",
        "- action set = current/Stay + Stage-A legal candidates",
        "- raycast = scene-only Habitat HM3D geometry to reconstructed world-space H36M17 joints",
        "- tie rule = exact Stay tie wins; moving ties use smallest candidate viewpoint id",
        "",
        "## Matched results",
        "",
        "| Selector | Type | Accuracy | Macro-F1 | Move rate |",
        "|---|---|---:|---:|---:|",
    ]
    types = {
        "Stay": "baseline",
        "Random legal": "baseline",
        "RealPoseConfidence": "non-causal reference",
        "Frame0SceneVisibility": "causal privileged",
        "FullTemporalSceneVisibility": "future-aware reference",
        "Historical Route-1": "learned task-aware",
        "GT-TrueLogP Oracle": "privileged task-aware",
    }
    for name in ("Stay", "Random legal", "RealPoseConfidence", "Frame0SceneVisibility", "FullTemporalSceneVisibility", "Historical Route-1", "GT-TrueLogP Oracle"):
        item = methods[name]
        protocol_lines.append(f"| {name} | {types[name]} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} |")
    protocol_lines += [
        "| CandidateOnly AnyCorrect | coverage only | — | — | — |",
        "| StayPlusCandidate AnyCorrect | coverage only | — | — | — |",
        "",
        f"Frame0SceneVisibility vs Random: {frame_gain_random:+.3f}pp Accuracy / {100.0 * (frame['macro_f1'] - random['macro_f1']):+.3f}pp Macro-F1.",
        f"Frame0SceneVisibility vs Historical Route-1 + Shared: {frame_gain_historical:+.3f}pp Accuracy.",
        f"FullTemporalSceneVisibility - Frame0SceneVisibility: {full_gap:+.3f}pp Accuracy / {100.0 * (full['macro_f1'] - frame['macro_f1']):+.3f}pp Macro-F1.",
        "",
        "## Frame-0 pose confidence",
        "",
        "Frame0PoseConfidence unavailable: archives store only a 30-frame aggregated viewpoint scalar, not per-frame keypoint confidence. No frame-0 value was fabricated.",
        "",
        "## Occlusion stratification",
        "",
        "Initial occlusion is defined by Stay frame-0 visibility (low occlusion = top tertile visibility; high occlusion = bottom tertile). See `occlusion_stratified_metrics.json` for Stay, Random, Frame0, FullTemporal and Historical results.",
        "",
        "## Ranking and transition diagnostics",
        "",
        f"Frame0 vs FullTemporal selected-view agreement: {ranking['frame0_vs_full_temporal']['selected_view_agreement']:.6f}; top-3 overlap: {ranking['frame0_vs_full_temporal']['top3_overlap']:.6f}; Frame0 selected rank within FullTemporal mean/median: {ranking['frame0_vs_full_temporal']['frame0_rank_within_full_mean']:.3f}/{ranking['frame0_vs_full_temporal']['frame0_rank_within_full_median']:.3f}.",
        f"Frame0 vs FullTemporal candidate Spearman: {ranking['frame0_vs_full_temporal']['candidate_spearman']:.6f}; within-context mean/median: {ranking['frame0_vs_full_temporal']['within_context_spearman_mean']:.6f}/{ranking['frame0_vs_full_temporal']['within_context_spearman_median']:.6f}.",
        f"Frame0 vs shared GT-TrueLogP candidate Spearman: {ranking['frame0_vs_shared_gt_true_logp']['candidate_spearman']:.6f}; within-context mean/median: {ranking['frame0_vs_shared_gt_true_logp']['within_context_spearman_mean']:.6f}/{ranking['frame0_vs_shared_gt_true_logp']['within_context_spearman_median']:.6f}; top1/top3 overlap: {ranking['frame0_vs_shared_gt_true_logp']['gt_true_logp_top1_overlap']:.6f}/{ranking['frame0_vs_shared_gt_true_logp']['gt_true_logp_top3_overlap']:.6f}.",
        "",
        "## Judgment",
        "",
    ]
    if frame["accuracy"] >= 0.48 and frame_gain_random >= 8.0:
        protocol_lines.append("Frame-0 scene visibility reaches the strong causal threshold (at least 48% and +8pp over Random); keep a strict causal observability route and prioritize predicting scalar frame-0 quality before adding structured visibility.")
        decision = "KEEP strict causal observability"
    elif frame["accuracy"] >= 0.43 and frame_gain_random >= 5.0:
        protocol_lines.append("Frame-0 scene visibility is materially above Random but below the strong threshold; keep observability as an auxiliary cue rather than a standalone policy target.")
        decision = "KEEP only as auxiliary cue"
    elif frame["accuracy"] < 0.43 and full_gap >= 5.0:
        protocol_lines.append("Frame-0 scene visibility is weak while full-temporal visibility is substantially stronger; the earlier temporal quality result depends on future pose information. Kill standalone pre-action visibility as the main Route-1 target.")
        decision = "KILL standalone pre-action visibility"
    else:
        protocol_lines.append("Frame-0 scene visibility does not establish a strong standalone causal selector. Treat it as auxiliary evidence and do not train a predictor from this audit alone.")
        decision = "KEEP only as auxiliary cue"
    protocol_lines += [
        f"Final decision: **{decision}**.",
        "Observation quality is not recognition utility; the FullTemporal and GT-TrueLogP rows are privileged references, not deployable selectors.",
        "No Policy Test was read, no model was trained, and no RGB/skeleton/DINO/runtime data were regenerated or modified.",
    ]
    return "\n".join(protocol_lines) + "\n"


def evaluate(data_root: Path, scene_root: Path, output_root: Path, device: str, inference_batch_size: int, max_contexts: int | None = None) -> dict[str, Any]:
    _, all_rows = load_rows(data_root)
    rows = list(all_rows[:max_contexts]) if max_contexts is not None else list(all_rows)
    if len(rows) != 10080:
        raise ValueError("full audit requires exactly 10,080 Moving Val contexts; use no --max-contexts")
    ids, mask, logp = _load_shared_logp(data_root, rows, device, inference_batch_size)
    full_temporal_candidates, full_meta = _load_full_temporal_scores(rows)
    frame0, full_stay, build_meta = _build_frame0_scores(data_root, scene_root, rows)
    pose_confidence, pose_meta = _archive_confidence(data_root, rows, ids, mask)
    historical_actions = _load_historical_actions(rows)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    current_actions = [int(row["current_viewpoint_id"]) for row in rows]
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice(_action_set(row))) for row in rows]
    frame0_actions = [_select_max_quality(_action_set(row), int(row["current_viewpoint_id"]), frame0[index]) for index, row in enumerate(rows)]
    full_actions = []
    pose_actions = []
    gt_actions = []
    for index, row in enumerate(rows):
        actions = _action_set(row)
        current = int(row["current_viewpoint_id"])
        full_values = dict(full_temporal_candidates[index]); full_values[current] = float(full_stay[index])
        full_actions.append(_select_max_quality(actions, current, full_values))
        pose_values = {action: float(pose_confidence[index, _slot(ids[index], mask[index], action)]) for action in actions}
        pose_actions.append(_select_max_quality(actions, current, pose_values))
        true_values = {action: float(logp[index, _slot(ids[index], mask[index], action), int(labels[index])]) for action in actions}
        gt_actions.append(_select_max_quality(actions, current, true_values))
    action_map = {
        "Stay": current_actions,
        "Random legal": random_actions,
        "RealPoseConfidence": pose_actions,
        "Frame0SceneVisibility": frame0_actions,
        "FullTemporalSceneVisibility": full_actions,
        "Historical Route-1": historical_actions,
        "GT-TrueLogP Oracle": gt_actions,
    }
    predictions = {name: _terminal_predictions(rows, ids, mask, logp, actions) for name, actions in action_map.items()}
    methods = {name: _metrics(labels, predictions[name], actions, rows, name) for name, actions in action_map.items()}
    candidate_correct = []
    legal_correct = []
    for index, row in enumerate(rows):
        actions = _action_set(row)
        candidate_slots = [_slot(ids[index], mask[index], action) for action in row["candidate_ids"]]
        legal_slots = [_slot(ids[index], mask[index], action) for action in actions]
        candidate_correct.append(bool(np.any(np.argmax(logp[index, candidate_slots], axis=1) == labels[index])))
        legal_correct.append(bool(np.any(np.argmax(logp[index, legal_slots], axis=1) == labels[index])))
    stay_visibility = np.asarray([float(frame0[index][int(row["current_viewpoint_id"])]) for index, row in enumerate(rows)], dtype=np.float64)
    ranking = _ranking_diagnostics(rows, ids, mask, logp, frame0, [{**full_temporal_candidates[i], current_actions[i]: full_stay[i]} for i in range(len(rows))], frame0_actions, full_actions, gt_actions)
    transitions = {
        "frame0_vs_stay": _transition(labels, predictions["Stay"], predictions["Frame0SceneVisibility"], "Stay", "Frame0SceneVisibility"),
        "frame0_vs_historical": _transition(labels, predictions["Historical Route-1"], predictions["Frame0SceneVisibility"], "Historical Route-1", "Frame0SceneVisibility"),
    }
    coverage = {
        "moving_val_contexts": len(rows),
        "action_set": "current/stay + Stage-A legal candidate_pool",
        "action_count_mean": float(np.mean([len(_action_set(row)) for row in rows])),
        "action_count_min": int(min(len(_action_set(row)) for row in rows)),
        "action_count_max": int(max(len(_action_set(row)) for row in rows)),
        "frame0_scene_visibility_context_coverage": 1.0,
        "full_temporal_scene_visibility_candidate_coverage": full_meta["candidate_coverage"],
        "full_temporal_scene_visibility_stay_coverage": build_meta["full_stay_context_coverage"],
        "real_pose_confidence_stay_coverage": pose_meta["stay_coverage"],
        "real_pose_confidence_candidate_coverage": pose_meta["candidate_coverage"],
        "frame0_pose_confidence_available": False,
        "frame0_pose_confidence_reason": "archives contain only 30-frame aggregated viewpoint confidence",
        "candidate_only_any_correct_coverage": float(np.mean(candidate_correct)),
        "stay_plus_candidate_any_correct_coverage": float(np.mean(legal_correct)),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_FRAME0_CAUSAL_OBSERVABILITY_AUDIT",
        "status": "COMPLETED",
        "population": {"split": "Moving Val", "contexts": len(rows), "scene_count": build_meta["scene_count"]},
        "labels": list(LABELS),
        "methods": methods,
        "deltas": {name: {"vs_random_accuracy_pp": 100.0 * (methods[name]["accuracy"] - methods["Random legal"]["accuracy"]), "vs_stay_accuracy_pp": 100.0 * (methods[name]["accuracy"] - methods["Stay"]["accuracy"]), "vs_historical_accuracy_pp": 100.0 * (methods[name]["accuracy"] - methods["Historical Route-1"]["accuracy"])} for name in methods},
        "coverage_audit": coverage,
        "ranking_diagnostics": ranking,
        "context_transitions": transitions,
        "occlusion_stratified_metrics": _occlusion_stratification(labels, rows, predictions, stay_visibility),
        "protocol": {"action_set": coverage["action_set"], "frame0_selector": "mean environmental visibility over 17 H36M joints at frame 0", "full_temporal_reference": "mean environmental visibility over frame ids [0,6,12,18,24,29]", "terminal": "selected real archived 30-frame skeleton through frozen shared recognizer", "tie_break": "Stay on exact ties, then smallest moving viewpoint id"},
        "flags": {"policy_test_used": False, "training_used": False, "recognizer_modified": False, "future_motion_frames_used_by_frame0_selector": False, "future_candidate_rgb_used": False, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "gt_action_used_for_selector": False, "gt_action_used_for_posthoc_reference_only": True, "habitat_gt_scene_geometry_used_for_oracle_diagnostic": True, "deployable": False},
        "artifacts": {"full_temporal_visibility": full_meta["source"], "shared_recognizer": str((data_root / "checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth").resolve()), "historical_route1": str(HISTORICAL_RESULT.resolve())},
        "runtime": {"frame0_build": build_meta, "full_temporal_source": full_meta, "device": device, "inference_batch_size": inference_batch_size},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "per_selector_metrics.json").write_text(json.dumps(methods, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "ranking_diagnostics.json").write_text(json.dumps(ranking, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "context_transitions.json").write_text(json.dumps(transitions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "occlusion_stratified_metrics.json").write_text(json.dumps(result["occlusion_stratified_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "coverage_audit.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"device": device, "inference_batch_size": inference_batch_size, "seed": SEED, "frame_ids": [0], "full_temporal_frame_ids": list(FRAME_IDS), "test_used": False, "training_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--max-contexts", type=int, default=None, help="reserved for debugging; full audit refuses partial output")
    args = parser.parse_args()
    if args.max_contexts is not None:
        raise ValueError("Frame-0 causal audit requires all 10,080 Moving Val contexts; omit --max-contexts")
    result = evaluate(args.data_root.resolve(), args.scene_root.resolve(), args.output_root.resolve(), args.device, args.inference_batch_size)
    print(json.dumps({"status": result["status"], "contexts": result["population"]["contexts"], "frame0_accuracy": result["methods"]["Frame0SceneVisibility"]["accuracy"], "full_temporal_accuracy": result["methods"]["FullTemporalSceneVisibility"]["accuracy"], "test_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
