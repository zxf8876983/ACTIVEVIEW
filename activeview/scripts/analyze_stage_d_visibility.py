#!/usr/bin/env python3
"""EXP030 observed-only candidate visibility and utility pilot."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_predictability import oracle_action_index, oracle_margin
from activeview.active_view.stage_d_visibility import (
    FEATURE_NAMES,
    analytic_candidate_order,
    candidate_visibility_features,
    extract_human_anchors,
    full_scene_visibility,
)
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.dataset.babel_clean_dataset_generator import MotionConverter, URDF_PATH, _load_resampled_motion, apply_humanoid_pose, precompute_grounding_offsets
from activeview.scripts.analyze_stage_d_semantic_bev import _records_by_identity
from activeview.scripts.generate_hm3d_train_rgb_observations import FRAME_INDEX, TARGET_FRAMES, _load_skeleton_metadata, _set_agent_state


PIXEL_STRIDE = 8
WORKERS = 4
RAY_RADIUS = 0.12
SENSOR_HEIGHT_M = 1.1
EXP_ID = "EXP030"
RUNTIME_DEFAULT = get_data_root() / "datasets/policy_v11_5/experiments/stage_d/EXP030_candidate_visibility_pilot"


def _depth_simulator(scene_root: Path, scene_id: str) -> tuple[Any, Any]:
    """Create a depth-only simulator; semantic assets are out of scope here."""
    import habitat_sim
    import magnum as mn

    scene_dir = scene_root / scene_id
    glb = next(scene_dir.glob("*.basis.glb")); navmesh = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration(); backend.scene_id = str(glb); backend.enable_physics = True
    agents = []
    for index in range(2):
        sensor = habitat_sim.CameraSensorSpec(); sensor.uuid = f"depth_{index}"; sensor.sensor_type = habitat_sim.SensorType.DEPTH; sensor.resolution = [256, 256]; sensor.position = mn.Vector3(0.0, SENSOR_HEIGHT_M, 0.0); sensor.hfov = 75.0
        agent = habitat_sim.AgentConfiguration(); agent.sensor_specifications = [sensor]; agents.append(agent)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, agents)); sim.pathfinder.load_nav_mesh(str(navmesh))
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    return sim, human


def _pose_and_observations(sim: Any, human: Any, record: Any, metadata: Mapping[str, Any], ids: Sequence[int]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    motion = _load_resampled_motion(record.motion, TARGET_FRAMES)
    converted = MotionConverter(URDF_PATH).convert(motion)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
    roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=0.0)
    placement = np.asarray(metadata["placement_position"], dtype=np.float32)
    apply_humanoid_pose(human, joints[FRAME_INDEX], roots[FRAME_INDEX], base_position=placement, scene_yaw_deg=0.0, floor_y=float(placement[1]), grounding_offset=float(offsets[FRAME_INDEX]))
    anchors = extract_human_anchors(human)
    positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
    rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
    output: dict[int, np.ndarray] = {}
    observed: list[np.ndarray] = []
    for index, viewpoint_id in enumerate(ids):
        agent_id = index % 2
        _set_agent_state(sim.get_agent(agent_id), positions[int(viewpoint_id)], rotations[int(viewpoint_id)])
        observations = sim.get_sensor_observations([agent_id])
        observation = observations[agent_id] if isinstance(observations, Mapping) and agent_id in observations else observations[0]
        depth = np.asarray(observation[f"depth_{agent_id}"], dtype=np.float32)
        ys = np.arange(0, depth.shape[0], PIXEL_STRIDE, dtype=np.float64); xs = np.arange(0, depth.shape[1], PIXEL_STRIDE, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys); focal = depth.shape[1] / (2.0 * np.tan(np.deg2rad(75.0) / 2.0)); rotation = _rotation_matrix_local(rotations[int(viewpoint_id)])
        local = np.stack(((grid_x - depth.shape[1] / 2.0) / focal, (depth.shape[0] / 2.0 - grid_y) / focal, -np.ones_like(grid_x)), axis=-1)
        origin = np.asarray(positions[int(viewpoint_id)], dtype=np.float64) + rotation @ np.array([0.0, SENSOR_HEIGHT_M, 0.0])
        sampled = depth[::PIXEL_STRIDE, ::PIXEL_STRIDE]; points = origin + np.einsum("ij,hwj->hwi", rotation, local * sampled[..., None]); valid = np.isfinite(sampled) & (sampled > 0.0); points[~valid] = np.nan
        output[int(viewpoint_id)] = points[valid]
        observed.append(output[int(viewpoint_id)])
    all_points = np.concatenate(observed, axis=0) if observed else np.empty((0, 3), dtype=np.float32)
    return {"points": all_points, "anchors": anchors}, np.asarray(anchors, dtype=np.float32)


def _rotation_matrix_local(rotation_wxyz: Sequence[float]) -> np.ndarray:
    from activeview.active_view.stage_d_visibility import rotation_matrix

    return rotation_matrix(rotation_wxyz)


def _worker(worker_id: int, tasks: Sequence[tuple[Mapping[str, Any], Any]], source_root: Path, scene_root: Path, config_path: Path, output: Path) -> None:
    if not tasks:
        return
    os.environ.setdefault("PYBULLET_EGL", "1")
    if os.environ.get("EXP030_DEBUG") != "1":
        log_path = output.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(log, 1); os.dup2(log, 2); os.close(log)
    current_scene: str | None = None
    sim = human = None
    converter = MotionConverter(URDF_PATH)
    record_cache: dict[tuple[str, str, str], tuple[Any, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    try:
        for feature, record in tasks:
            key = (str(record.scene_id), str(record.region), str(record.record_id))
            if key not in record_cache:
                record_cache[key] = (record, _load_skeleton_metadata(record.source_path))
            if current_scene != record.scene_id:
                if sim is not None:
                    sim.close()
                sim, human = _depth_simulator(scene_root, record.scene_id)
                current_scene = record.scene_id
            metadata = record_cache[key][1]
            ids = [int(feature["s0_viewpoint_id"]), int(feature["s1_viewpoint_id"])]
            observations, anchors = _pose_and_observations(sim, human, record, metadata, ids)
            candidate_ids = [int(value) for value in feature["remaining_candidate_ids"]]
            # Stage-D feature rows retain candidate geometry but not camera pose;
            # the immutable skeleton archive has all 32 exact candidate states.
            positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32)
            rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
            candidate_features: dict[str, list[float]] = {}
            full_scene: dict[str, list[float] | None] = {}
            for candidate_id in candidate_ids:
                values = candidate_visibility_features(observations["points"], anchors, candidate_position=positions[candidate_id], candidate_rotation_wxyz=rotations[candidate_id], radius=RAY_RADIUS)
                candidate_features[str(candidate_id)] = values.tolist()
                try:
                    visible = full_scene_visibility(sim, positions[candidate_id], rotations[candidate_id], anchors)
                    full_scene[str(candidate_id)] = visible.tolist()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    full_scene[str(candidate_id)] = None
            rows.append({"episode_id": str(feature["episode_id"]), "record_id": str(feature["record_id"]), "scene_id": str(feature["scene_id"]), "region": str(feature["region"]), "policy_split": str(feature["policy_split"]), "label_id": int(feature["label_id"]), "remaining_candidate_ids": candidate_ids, "second_step_utility_targets": [float(v) for v in feature["second_step_utility_targets"]], "candidate_features": candidate_features, "full_scene_visible": full_scene, "s0_viewpoint_id": ids[0], "s1_viewpoint_id": ids[1], "human_anchor_count": int(len(anchors))})
    except Exception:
        output.with_suffix(".error").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if sim is not None:
            sim.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _generate(rows: Sequence[Mapping[str, Any]], split: str, runtime: Path, source_root: Path, motion_root: Path, scene_root: Path, config_path: Path, workers: int) -> dict[str, Any]:
    path = runtime / f"{split}_visibility.jsonl"
    if path.is_file():
        cached = load_jsonl(path)
        if {str(item.get("episode_id")) for item in cached} == {str(row["episode_id"]) for row in rows} and len(cached) == len(rows):
            return {"count": len(cached), "path": str(path.resolve()), "sha256": file_sha256(path), "generation_time_sec": 0.0, "resumed": True}
    records = _records_by_identity(source_root, motion_root, sorted({str(row["scene_id"]) for row in rows}))
    grouped: dict[str, list[tuple[Mapping[str, Any], Any]]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
        if key not in records:
            raise ValueError(f"Missing canonical source record: {key}")
    collected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        grouped.setdefault(str(row["scene_id"]), []).append((row, records[(str(row["scene_id"]), str(row["region"]), str(row["record_id"]))]))
    started = time.monotonic()
    # Habitat scene state is released only when its simulator is destroyed;
    # process one scene at a time to bound GPU/physics memory across 21 scenes.
    for scene_id in sorted(grouped):
        tasks = grouped[scene_id]
        temp = runtime / f".{split}_workers_{scene_id}"; temp.mkdir(parents=True, exist_ok=True)
        processes: list[mp.Process] = []
        for worker_id in range(workers):
            process = mp.get_context("fork").Process(target=_worker, args=(worker_id, tasks[worker_id::workers], source_root, scene_root, config_path, temp / f"worker_{worker_id}.jsonl"))
            process.start(); processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"EXP030 Habitat worker failure in {scene_id}: {[p.exitcode for p in processes]}")
        for worker_id in range(workers):
            path = temp / f"worker_{worker_id}.jsonl"
            if path.is_file():
                for item in load_jsonl(path):
                    collected[str(item["episode_id"])] = item
                path.unlink()
            for suffix in (".log", ".error"):
                diagnostic = path.with_suffix(suffix)
                if diagnostic.is_file():
                    diagnostic.unlink()
        temp.rmdir()
    if set(collected) != {str(row["episode_id"]) for row in rows}:
        raise ValueError(f"Visibility cache alignment mismatch for {split}")
    path.write_text("".join(json.dumps(collected[str(row["episode_id"])], separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    return {"count": len(rows), "path": str(path.resolve()), "sha256": file_sha256(path), "generation_time_sec": time.monotonic() - started}


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train.ndim != 2 or train.shape[0] == 0:
        return train, values, np.empty((0, 2), dtype=np.float32)
    mean = train.mean(axis=0); std = train.std(axis=0); std[std < 1e-6] = 1.0
    return (train - mean) / std, (values - mean) / std, np.stack([mean, std])


def _corr(pred: np.ndarray, true: np.ndarray) -> dict[str, float | None]:
    if len(pred) < 2:
        return {"pearson": None, "spearman": None}
    pearson = float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) > 0 and np.std(true) > 0 else 0.0
    rank_pred = np.argsort(np.argsort(pred)); rank_true = np.argsort(np.argsort(true))
    spearman = float(np.corrcoef(rank_pred, rank_true)[0, 1]) if np.std(rank_pred) > 0 and np.std(rank_true) > 0 else 0.0
    return {"pearson": pearson, "spearman": spearman}


def _fit_regressor(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn
    torch.manual_seed(42)
    base_dim = train_x.shape[1] - len(FEATURE_NAMES)

    class CandidateRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = nn.Linear(base_dim, 128)
            self.candidate = nn.Linear(len(FEATURE_NAMES), 64)
            self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            base = torch.nn.functional.gelu(self.base(values[:, :base_dim]))
            candidate = torch.nn.functional.gelu(self.candidate(values[:, base_dim:]))
            return self.head(torch.cat([base, candidate], dim=1))

    model = CandidateRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.SmoothL1Loss()
    tx = torch.from_numpy(train_x.astype(np.float32)); ty = torch.from_numpy(train_y.astype(np.float32)).reshape(-1, 1)
    for _ in range(20):
        permutation = torch.randperm(len(tx))
        for start in range(0, len(tx), 512):
            idx = permutation[start : start + 512]; loss = loss_fn(model(tx[idx]), ty[idx]); optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        pred = model(torch.from_numpy(val_x.astype(np.float32))).reshape(-1).numpy()
    with torch.inference_mode():
        train_pred = model(tx).reshape(-1).numpy()
    return pred, {"architecture": f"base Linear({base_dim},128) + candidate Linear({len(FEATURE_NAMES)},64) -> concat(192) -> Linear(192,128)->GELU->Linear(128,64)->GELU->Linear(64,1)", "epochs": 20, "batch_size": 512, "learning_rate": 1e-3, "loss": "SmoothL1Loss", "train_final_mae": float(np.mean(np.abs(train_pred - train_y)))}


def _fit_ranker(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(train_x.shape[1], 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.BCEWithLogitsLoss(); tx = torch.from_numpy(train_x.astype(np.float32)); ty = torch.from_numpy(train_y.astype(np.float32)).reshape(-1, 1)
    for _ in range(20):
        permutation = torch.randperm(len(tx))
        for start in range(0, len(tx), 256):
            idx = permutation[start : start + 256]; loss = loss_fn(model(tx[idx]), ty[idx]); optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        pred = torch.sigmoid(model(torch.from_numpy(val_x.astype(np.float32)))).reshape(-1).numpy()
    return pred, {"architecture": f"Linear({train_x.shape[1]},128)->GELU->Linear(128,64)->GELU->Linear(64,1)", "epochs": 20, "learning_rate": 1e-3, "loss": "BCEWithLogitsLoss"}


def _method_metrics(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, Sequence[float] | int], mode: str) -> dict[str, Any]:
    oracle: list[int] = []; guess: list[int] = []; utility: list[float] = []; margins: list[float] = []
    for row in rows:
        ids = [int(v) for v in row["remaining_candidate_ids"]]; targets = [float(v) for v in row["second_step_utility_targets"]]; scores = [0.0] + [float(v) for v in predictions[str(row["episode_id"])]]
        action = int(np.argmax(np.asarray(scores, dtype=np.float64))); oracle.append(oracle_action_index(targets)); guess.append(action); utility.append(0.0 if action == 0 else targets[action - 1]); margins.append(float(oracle_margin(targets)["margin_1"]))
    oracle_a = np.asarray(oracle); guess_a = np.asarray(guess)
    both_move = (oracle_a > 0) & (guess_a > 0)
    candidate_hit = float(np.mean(oracle_a[both_move] == guess_a[both_move])) if both_move.any() else None
    harmful = (guess_a > 0) & (np.asarray(utility) <= 0.0)
    missed = (guess_a == 0) & (oracle_a > 0)
    result = {"episode_count": len(rows), "three_way_accuracy": float(np.mean(oracle_a == guess_a)) if rows else 0.0, "binary_move_stay_accuracy": float(np.mean((oracle_a > 0) == (guess_a > 0))) if rows else 0.0, "selected_action_mean_true_utility": float(np.mean(utility)) if utility else 0.0, "harmful_move_count": int(harmful.sum()), "missed_beneficial_move_count": int(missed.sum()), "both_oracle_and_model_move_count": int(both_move.sum()), "both_move_candidate_hit": candidate_hit, "action_counts": {"stay": int(np.sum(guess_a == 0)), "p2": int(np.sum(guess_a == 1)), "p3": int(np.sum(guess_a == 2))}}
    for threshold in (0.25, 0.5, 1.0, 2.0):
        mask = np.asarray(margins) >= threshold
        move_mask = mask & both_move
        result[f"margin_ge_{threshold}"] = {"count": int(mask.sum()), "three_way_accuracy": float(np.mean(oracle_a[mask] == guess_a[mask])) if mask.any() else None, "candidate_hit": float(np.mean(oracle_a[move_mask] == guess_a[move_mask])) if move_mask.any() else None}
    return result


def analyze(*, cache_root: Path, v0_val_predictions: Path, output: Path, runtime: Path, source_root: Path, motion_root: Path, scene_root: Path, config_path: Path, workers: int = WORKERS, train_limit: int | None = None, val_limit: int | None = None) -> dict[str, Any]:
    summary = json.loads((cache_root / "stage_d_feature_summary.json").read_text()); train_rows = load_jsonl(Path(summary["feature_files"]["train"])); val_rows = load_jsonl(Path(summary["feature_files"]["val"]))
    if train_limit is not None: train_rows = train_rows[:train_limit]
    if val_limit is not None: val_rows = val_rows[:val_limit]
    if train_limit is None and val_limit is None and (len(train_rows), len(val_rows)) != (29133, 9742):
        raise ValueError(f"Unexpected EXP030 eligible counts: {len(train_rows)}, {len(val_rows)}")
    v0_rows = load_jsonl(v0_val_predictions)
    if any(str(row.get("policy_split", "")).lower() != "val" for row in v0_rows):
        raise ValueError("EXP030 accepts only explicit Val frozen-v0 predictions")
    expected_ids = {str(row["episode_id"]) for row in val_rows}
    actual_ids = {str(row["episode_id"]) for row in v0_rows if not bool(row.get("predicted_stays"))}
    if val_limit is None and expected_ids != actual_ids:
        raise ValueError("EXP030 Val rows and frozen-v0 Move IDs are not exactly aligned")
    runtime.mkdir(parents=True, exist_ok=True)
    train_generation = _generate(train_rows, "train", runtime, source_root, motion_root, scene_root, config_path, workers)
    val_generation = _generate(val_rows, "val", runtime, source_root, motion_root, scene_root, config_path, workers)
    train_features = load_jsonl(Path(train_generation["path"])); val_features = load_jsonl(Path(val_generation["path"]))
    if {str(r["policy_split"]) for r in train_features} != {"train"} or {str(r["policy_split"]) for r in val_features} != {"val"}:
        raise ValueError("EXP030 cache split mismatch")
    train_base = np.asarray([np.asarray([*row["s0_feature"], *row["s1_feature"], *row["delta_semantic"]], dtype=np.float32) for row in train_rows])
    val_base = np.asarray([np.asarray([*row["s0_feature"], *row["s1_feature"], *row["delta_semantic"]], dtype=np.float32) for row in val_rows])
    train_base, val_base, _ = _standardize(train_base, val_base)
    train_candidate_x: list[np.ndarray] = []; val_candidate_x: list[np.ndarray] = []; train_y: list[float] = []; val_y: list[float] = []
    train_by_id = {str(r["episode_id"]): r for r in train_features}; val_by_id = {str(r["episode_id"]): r for r in val_features}
    for split_rows, cache, base, xs, ys in ((train_rows, train_by_id, train_base, train_candidate_x, train_y), (val_rows, val_by_id, val_base, val_candidate_x, val_y)):
        for index, row in enumerate(split_rows):
            cached = cache[str(row["episode_id"])];
            for candidate_id, target in zip(cached["remaining_candidate_ids"], cached["second_step_utility_targets"]):
                xs.append(np.concatenate([base[index], np.asarray(cached["candidate_features"][str(candidate_id)], dtype=np.float32)])); ys.append(float(target))
    train_x = np.asarray(train_candidate_x, dtype=np.float32); val_x = np.asarray(val_candidate_x, dtype=np.float32); train_x, val_x, _ = _standardize(train_x, val_x)
    val_pred_flat, reg_model = _fit_regressor(train_x, np.asarray(train_y, dtype=np.float32), val_x)
    reg_true = np.asarray(val_y, dtype=np.float64); val_pred_flat = np.asarray(val_pred_flat, dtype=np.float64)
    reg_metrics = {"n": len(reg_true), "mae": float(np.mean(np.abs(val_pred_flat - reg_true))), "rmse": float(np.sqrt(np.mean((val_pred_flat - reg_true) ** 2))), **_corr(val_pred_flat, reg_true)}
    predictions: dict[str, list[float]] = {}; offset = 0
    for row in val_rows:
        count = len(val_by_id[str(row["episode_id"])] ["remaining_candidate_ids"]); predictions[str(row["episode_id"])] = val_pred_flat[offset : offset + count].tolist(); offset += count
    method_b = _method_metrics(val_rows, predictions, "regression")
    rank_train_x: list[np.ndarray] = []; rank_val_x: list[np.ndarray] = []; rank_train_y: list[int] = []; rank_val_y: list[int] = []; rank_val_margins: list[float] = []
    for split_rows, cache, base, xs, ys in ((train_rows, train_by_id, train_base, rank_train_x, rank_train_y), (val_rows, val_by_id, val_base, rank_val_x, rank_val_y)):
        for index, row in enumerate(split_rows):
            cached = cache[str(row["episode_id"])]
            if len(cached["remaining_candidate_ids"]) < 2 or oracle_action_index(cached["second_step_utility_targets"]) == 0: continue
            left = np.asarray(cached["candidate_features"][str(cached["remaining_candidate_ids"][0])], dtype=np.float32); right = np.asarray(cached["candidate_features"][str(cached["remaining_candidate_ids"][1])], dtype=np.float32)
            xs.append(np.concatenate([base[index], left, right, left - right])); ys.append(int(float(cached["second_step_utility_targets"][0]) > float(cached["second_step_utility_targets"][1])))
            if split_rows is val_rows:
                rank_val_margins.append(float(oracle_margin(cached["second_step_utility_targets"])["margin_1"]))
    rank_train = np.asarray(rank_train_x, dtype=np.float32); rank_val = np.asarray(rank_val_x, dtype=np.float32)
    if rank_train.ndim == 2 and rank_train.shape[0] > 0 and rank_val.ndim == 2 and rank_val.shape[0] > 0:
        rank_train, rank_val, _ = _standardize(rank_train, rank_val)
        rank_pred, rank_model = _fit_ranker(rank_train, np.asarray(rank_train_y, dtype=np.float32), rank_val)
        rank_labels = np.asarray(rank_val_y, dtype=bool); rank_guess = rank_pred > 0.5
        rank_accuracy = float(np.mean(rank_guess == rank_labels)) if rank_val_y else None
        tpr = float(np.mean(rank_guess[rank_labels])) if np.any(rank_labels) else 0.0
        tnr = float(np.mean(~rank_guess[~rank_labels])) if np.any(~rank_labels) else 0.0
        rank_balanced = (tpr + tnr) / 2.0
        rank_margin = {str(threshold): {"count": int(np.sum(np.asarray(rank_val_margins) >= threshold)), "winner_accuracy": float(np.mean(rank_guess[np.asarray(rank_val_margins) >= threshold] == rank_labels[np.asarray(rank_val_margins) >= threshold])) if np.any(np.asarray(rank_val_margins) >= threshold) else None} for threshold in (0.25, 0.5, 1.0, 2.0)}
    else:
        rank_model = {"status": "SKIPPED_EMPTY_ELIGIBLE_ROWS"}
        rank_accuracy = None; rank_balanced = None; rank_margin = {}
    a0_rows = [row for row in val_features if len(row["remaining_candidate_ids"]) >= 2 and oracle_action_index(row["second_step_utility_targets"]) > 0]; a0_correct = 0; visibility_values: list[float] = []; utility_values: list[float] = []; high_margin: dict[str, Any] = {}
    for row in a0_rows:
        feats = {int(k): v for k, v in row["candidate_features"].items()}; winner = analytic_candidate_order(feats); oracle = int(row["remaining_candidate_ids"][int(np.argmax(np.asarray(row["second_step_utility_targets"], dtype=np.float64)))])
        a0_correct += int(winner == oracle)
        visibility_values.extend(float(values[0]) for values in feats.values())
        utility_values.extend(float(value) for value in row["second_step_utility_targets"])
    for threshold in (0.25, 0.5, 1.0, 2.0):
        subset = [row for row in a0_rows if float(oracle_margin(row["second_step_utility_targets"])["margin_1"]) >= threshold]; high_margin[str(threshold)] = {"count": len(subset), "winner_accuracy": float(np.mean([analytic_candidate_order({int(k): v for k, v in row["candidate_features"].items()}) == int(row["remaining_candidate_ids"][int(np.argmax(np.asarray(row["second_step_utility_targets"], dtype=np.float64)))]) for row in subset])) if subset else None}
    full_values = [value for row in val_features for value in row["full_scene_visible"].values() if value is not None]
    full_winner_hits: list[bool] = []; full_visibility_scores: list[float] = []; full_utility_targets: list[float] = []
    for row in val_features:
        visible = row["full_scene_visible"]
        if any(value is None for value in visible.values()) or len(row["remaining_candidate_ids"]) < 2:
            continue
        ids = [int(value) for value in row["remaining_candidate_ids"]]; scores = [float(np.mean(visible[str(value)])) for value in ids]
        target = [float(value) for value in row["second_step_utility_targets"]]
        if oracle_action_index(target) > 0:
            full_winner_hits.append(int(ids[int(np.argmax(scores))]) == int(ids[int(np.argmax(np.asarray(target, dtype=np.float64)))]))
        full_visibility_scores.extend(scores); full_utility_targets.extend(target)
    method_d = {"available": bool(full_values), "candidate_count": len(full_values), "mean_visible_joint_fraction": float(np.mean([np.mean(v) for v in full_values])) if full_values else None, "winner_accuracy_on_oracle_move": float(np.mean(full_winner_hits)) if full_winner_hits else None, "visibility_utility_correlation": _corr(np.asarray(full_visibility_scores, dtype=np.float64), np.asarray(full_utility_targets, dtype=np.float64)) if full_visibility_scores else {"pearson": None, "spearman": None}, "full_unobserved_scene_geometry_used": True, "oracle_visibility_upper_bound": True}
    visibility_utility_corr = _corr(np.asarray(visibility_values, dtype=np.float64), np.asarray(utility_values, dtype=np.float64))
    a0_accuracy = float(a0_correct / len(a0_rows)) if a0_rows else 0.0
    if method_d["winner_accuracy_on_oracle_move"] is not None and method_d["winner_accuracy_on_oracle_move"] > a0_accuracy + 0.1:
        decision = "CASE_B"
    elif method_b["three_way_accuracy"] > 0.444570 + 0.02 or a0_accuracy > 0.55:
        decision = "CASE_A"
    else:
        decision = "CASE_C"
    result: dict[str, Any] = {"experiment_id": EXP_ID, "experiment_name": "fast_candidate_conditioned_visibility_utility_pilot", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": True, "stgcn_retrained": False, "eligible_episode_counts": {"train": len(train_rows), "val": len(val_rows)}, "method_a0_observed_visibility": {"episode_count": len(a0_rows), "winner_accuracy": a0_accuracy, "visibility_utility_correlation": visibility_utility_corr, "high_margin": high_margin}, "method_b_utility_regression": {"model": reg_model, "candidate_level": reg_metrics, "episode_level": method_b}, "method_c_candidate_ranking": {"model": rank_model, "val_episode_count": len(rank_val_y), "winner_accuracy": rank_accuracy, "balanced_accuracy": rank_balanced, "margin_conditioned": rank_margin, "positive_train_count": len(rank_train_y)}, "method_d_full_scene_visibility": method_d, "comparison_references": {"exp029_r1_knn_k25_three_way": 0.430404, "exp028_knn_k25_three_way": 0.444570, "exp027_both_move_candidate_hit": 0.5996}, "decision": decision, "leakage_flags": {"future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_semantic_used": False, "future_candidate_skeleton_used": False, "full_unobserved_scene_geometry_used_method_a_b_c": False, "full_unobserved_scene_geometry_used_method_d": True, "oracle_visibility_upper_bound": True, "oracle_human_geometry_used": True, "test_used": False}, "provenance": {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])), "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])), "runtime_train_visibility": train_generation, "runtime_val_visibility": val_generation}}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_name("feature_summary.json").write_text(json.dumps({"experiment_id": EXP_ID, "feature_names": FEATURE_NAMES, "feature_dim": len(FEATURE_NAMES), "pixel_stride": PIXEL_STRIDE, "ray_block_radius_m": RAY_RADIUS, "eligible_episode_counts": result["eligible_episode_counts"], "runtime_visibility_paths": {"train": train_generation["path"], "val": val_generation["path"]}, "future_candidate_observation_used": False, "oracle_human_geometry_used": True}, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_name("analysis.md").write_text("# EXP030 — Fast Candidate-Conditioned Visibility / Utility Pilot\n\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    data_root = get_data_root(); parser = argparse.ArgumentParser(description=__doc__); cache = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"; parser.add_argument("--cache-root", type=Path, default=cache); parser.add_argument("--v0-val-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"); parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP030_candidate_visibility_pilot/result.json"); parser.add_argument("--runtime", type=Path, default=RUNTIME_DEFAULT); parser.add_argument("--source-root", type=Path, default=data_root / "datasets/offline/hm3d-train"); parser.add_argument("--motion-root", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"); parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train"); parser.add_argument("--config-path", type=Path, default=get_habitat_data_root() / "hm3d-train-semantic-configs/hm3d_annotated_train_basis.scene_dataset_config.json"); parser.add_argument("--workers", type=int, default=WORKERS); parser.add_argument("--train-limit", type=int, default=None); parser.add_argument("--val-limit", type=int, default=None); return parser


def main() -> None:
    result = analyze(**vars(build_parser().parse_args())); print(json.dumps({"experiment_id": EXP_ID, "status": result["status"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
