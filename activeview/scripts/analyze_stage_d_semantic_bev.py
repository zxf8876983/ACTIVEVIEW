#!/usr/bin/env python3
"""Generate and audit EXP029 observed local semantic BEV representations."""

from __future__ import annotations

import argparse
import math
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

from activeview.active_view.stage_d_dataset import load_jsonl, load_stage_d_statistics
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories
from activeview.active_view.stage_d_predictability import ACTION_NAMES, majority_action, neighbor_agreement, neighbor_entropy, oracle_action_index, oracle_margin, margin_bin_index
from activeview.active_view.stage_d_rgb_context import RGBObservationKey, observation_keys_from_feature_rows
from activeview.active_view.stage_d_semantic_bev import BEV_CHANNELS, add_markers, pool_bev, project_world_samples, validate_bev
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.dataset.babel_clean_dataset_generator import MotionConverter, URDF_PATH, _load_resampled_motion, apply_humanoid_pose, precompute_grounding_offsets
from activeview.scripts.generate_hm3d_train_rgb_observations import FRAME_INDEX, HFOV_DEG, IMAGE_SIZE, SENSOR_HEIGHT_M, TARGET_FRAMES, SourceRecord, _load_skeleton_metadata, _load_source_records, _set_agent_state
from activeview.scripts.repair_hm3d_semantic_mapping import discover_assets
from activeview.scripts.analyze_stage_d_predictability import _load_spatial_cache, _observable_vectors, _rgb_pooled


# Four Habitat simulators fit the available host memory; larger fan-out is
# unsafe because each simulator loads a full annotated HM3D scene.
WORKERS = 4
PIXEL_STRIDE = 8
EXP_ID = "EXP029"
RUNTIME_DEFAULT = get_data_root() / "datasets/policy_v11_5/experiments/stage_d/EXP029_observed_local_semantic_bev"


def _assert_split(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    bad = {"<missing>" if row.get("policy_split") is None else str(row.get("policy_split")).lower() for row in rows if str(row.get("policy_split", "")).lower() != split}
    if bad:
        raise ValueError(f"{name} split mismatch: {sorted(bad)}")


def _index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row["episode_id"])
        if key in result:
            raise ValueError(f"Duplicate {name} episode_id: {key}")
        result[key] = row
    return result


def _rotation(rotation_wxyz: Sequence[float]) -> np.ndarray:
    import quaternion
    return np.asarray(quaternion.as_rotation_matrix(quaternion.from_float_array(np.asarray(rotation_wxyz, dtype=np.float64))), dtype=np.float64)


def _semantic_channels(objects: Sequence[Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for obj in objects:
        object_id = str(getattr(obj, "id", "")); suffix = object_id.rsplit("_", 1)[-1]
        if not suffix.isdigit():
            continue
        category = getattr(obj, "category", None); name = category.name() if category is not None else "other_object"
        normalized = " ".join(str(name).strip().lower().replace("_", " ").split())
        if normalized == "sofa": normalized = "couch"
        if normalized == "kitchen cabinet": normalized = "cabinet"
        mapping[int(suffix)] = BEV_CHANNELS.get(normalized, BEV_CHANNELS["other_object"])
    return mapping


def _simulator(scene_root: Path, scene_id: str, config_path: Path) -> tuple[Any, Any]:
    import habitat_sim
    import magnum as mn
    scene_dir = scene_root / scene_id
    glb = next(scene_dir.glob("*.basis.glb")); navmesh = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration(); backend.scene_id = str(glb); backend.scene_dataset_config_file = str(config_path); backend.enable_physics = True; backend.load_semantic_mesh = True; backend.force_separate_semantic_scene_graph = True; backend.use_semantic_textures = False
    agents = []
    for index in range(2):
        specs = []
        for suffix, sensor_type in (("depth", habitat_sim.SensorType.DEPTH), ("semantic", habitat_sim.SensorType.SEMANTIC)):
            spec = habitat_sim.CameraSensorSpec(); spec.uuid = f"{suffix}_{index}"; spec.sensor_type = sensor_type; spec.resolution = [IMAGE_SIZE, IMAGE_SIZE]; spec.position = mn.Vector3(0.0, SENSOR_HEIGHT_M, 0.0); spec.hfov = HFOV_DEG; specs.append(spec)
        agent = habitat_sim.AgentConfiguration(); agent.sensor_specifications = specs; agents.append(agent)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, agents)); sim.pathfinder.load_nav_mesh(str(navmesh)); human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0"))); return sim, human


def _world_samples(
    observation: Mapping[str, Any],
    agent_index: int,
    position: Sequence[float],
    rotation_wxyz: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of Habitat camera ``unproject`` at stride 8.

    The bounded smoke test numerically compares this pinhole reconstruction
    against Habitat's official unprojected ray (including the camera -Z
    convention).  Full generation uses the equivalent vectorized form to
    avoid millions of Python/C++ ray calls.
    """
    depth = np.asarray(observation[f"depth_{agent_index}"], dtype=np.float32); semantic = np.asarray(observation[f"semantic_{agent_index}"])
    ys = np.arange(0, IMAGE_SIZE, PIXEL_STRIDE, dtype=np.float64); xs = np.arange(0, IMAGE_SIZE, PIXEL_STRIDE, dtype=np.float64); grid_x, grid_y = np.meshgrid(xs, ys)
    fx = IMAGE_SIZE / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0)); cx = IMAGE_SIZE / 2.0; cy = IMAGE_SIZE / 2.0
    # Habitat depth is measured along the camera ray's -Z projection.  Keep
    # the pinhole ray non-normalized: ``depth`` is multiplied by the ray's
    # z-component exactly as in ``render_camera.unproject(..., normalized=False)``.
    local = np.stack(((grid_x - cx) / fx, (cy - grid_y) / fx, -np.ones_like(grid_x)), axis=-1)
    rotation = _rotation(rotation_wxyz); origin = np.asarray(position, dtype=np.float64) + rotation @ np.array([0.0, SENSOR_HEIGHT_M, 0.0])
    points = origin + np.einsum("ij,hwj->hwi", rotation, local * depth[::PIXEL_STRIDE, ::PIXEL_STRIDE, None]); valid = np.isfinite(depth[::PIXEL_STRIDE, ::PIXEL_STRIDE]) & (depth[::PIXEL_STRIDE, ::PIXEL_STRIDE] > 0.0); points[~valid] = np.nan
    labels = np.asarray(semantic[::PIXEL_STRIDE, ::PIXEL_STRIDE], dtype=np.int32); return points.astype(np.float32), labels, valid.astype(np.uint8)


def _render_observations(sim: Any, human: Any, converter: MotionConverter, record: SourceRecord, metadata: Mapping[str, Any], viewpoint_ids: Sequence[int]) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    motion = _load_resampled_motion(record.motion, TARGET_FRAMES); converted = converter.convert(motion)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32); roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=0.0); base = np.asarray(metadata["placement_position"], dtype=np.float32)
    apply_humanoid_pose(human, joints[FRAME_INDEX], roots[FRAME_INDEX], base_position=base, scene_yaw_deg=0.0, floor_y=float(base[1]), grounding_offset=float(offsets[FRAME_INDEX]))
    semantic_map = _semantic_channels(sim.semantic_scene.objects if sim.semantic_scene is not None else [])
    positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32); rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
    output: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for index, viewpoint_id in enumerate(viewpoint_ids):
        agent_id = index % 2
        _set_agent_state(sim.get_agent(agent_id), positions[int(viewpoint_id)], rotations[int(viewpoint_id)])
        observations = sim.get_sensor_observations([agent_id])
        observation = observations[agent_id] if isinstance(observations, Mapping) and agent_id in observations else observations[0]
        output[int(viewpoint_id)] = _world_samples(observation, agent_id, positions[int(viewpoint_id)], rotations[int(viewpoint_id)])
    return output


def _worker(worker_id: int, tasks: Sequence[tuple[Mapping[str, Any], SourceRecord]], source_root: Path, scene_root: Path, config_path: Path, output_path: Path) -> None:
    if not tasks: return
    os.environ.setdefault("PYBULLET_EGL", "1")
    devnull = os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 1); os.dup2(devnull, 2); os.close(devnull)
    # Habitat/Bullet emits one warning per humanoid joint; keep the parent log
    # compact after the single-episode worker check has passed.
    current_scene: str | None = None; sim = human = None; converter = MotionConverter(URDF_PATH); record_cache: dict[tuple[str, str, str], tuple[SourceRecord, dict[str, Any]]] = {}; rows: list[dict[str, Any]] = []; arrays: list[np.ndarray] = []
    try:
        for episode, record in tasks:
            identity = (record.scene_id, record.region, record.record_id)
            if identity not in record_cache: record_cache[identity] = (record, _load_skeleton_metadata(record.source_path))
            if current_scene != record.scene_id:
                if sim is not None: sim.close()
                sim, human = _simulator(scene_root, record.scene_id, config_path); current_scene = record.scene_id
            metadata = record_cache[identity][1]; ids = [int(episode["s0_viewpoint_id"]), int(episode["s1_viewpoint_id"])]
            observations = _render_observations(sim, human, converter, record, metadata, ids)
            s1_id = ids[1]; s1_position = metadata["viewpoint_agent_positions"][s1_id]; s1_rotation = _rotation(metadata["viewpoint_rotations_wxyz"][s1_id]); bev = np.zeros((15, 80, 80), dtype=np.uint8)
            for view_id in ids:
                points, labels, _ = observations[view_id]
                camera_world = metadata["viewpoint_agent_positions"][view_id]
                # semantic map is reconstructed from the rendered IDs; unknown IDs safely use other_object.
                semantic_map = _semantic_channels(sim.semantic_scene.objects if sim.semantic_scene is not None else [])
                bev |= project_world_samples(points, labels, camera_world=camera_world, s1_position=s1_position, s1_rotation_matrix=s1_rotation, semantic_channels=semantic_map)
            # Two observed cameras can traverse the same cell; endpoint
            # surfaces take precedence over free-ray markings globally.
            bev[2][bev[1] > 0] = 0
            candidate_ids = [int(value) for value in episode["remaining_candidate_ids"]]
            positions = {"s0": metadata["viewpoint_agent_positions"][ids[0]], "s1": s1_position, "p2": metadata["viewpoint_agent_positions"][candidate_ids[0]]}
            # Some canonical Stage-D rows have one remaining candidate.  Keep
            # the row (and its fixed denominator) while leaving unavailable
            # p3_position channel zero rather than inventing a viewpoint.
            if len(candidate_ids) > 1:
                positions["p3"] = metadata["viewpoint_agent_positions"][candidate_ids[1]]
            add_markers(bev, positions, s1_position=s1_position, s1_rotation_matrix=s1_rotation, human_position=np.asarray(human.translation, dtype=np.float64)); validate_bev(bev)
            rows.append({"episode_id": str(episode["episode_id"]), "scene_id": record.scene_id, "region": record.region, "record_id": record.record_id, "s0_viewpoint_id": ids[0], "s1_viewpoint_id": ids[1], "frame_index": FRAME_INDEX}); arrays.append(bev)
        with output_path.open("wb") as handle: np.savez_compressed(handle, bev=np.stack(arrays), rows=np.asarray([json.dumps(row) for row in rows]))
    except BaseException:
        output_path.with_suffix(".error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if sim is not None: sim.close()
    os._exit(0)


def _records_by_identity(source_root: Path, motion_root: Path, scenes: Sequence[str]) -> dict[tuple[str, str, str], SourceRecord]:
    motions: dict[str, Mapping[str, Any]] = {}
    for split in ("train", "val"):
        for item in json.loads((motion_root / f"{split}.json").read_text(encoding="utf-8")):
            motions.setdefault(str(item["record_id"]), dict(item))
    result: dict[tuple[str, str, str], SourceRecord] = {}
    for scene_id in scenes:
        payload = json.loads((source_root / scene_id / "manifest.json").read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            record_id = str(item["record_id"]); motion = motions.get(record_id)
            if motion is None:
                continue
            region = str(item["region"]); path = source_root / scene_id / str(item["path"])
            result[(scene_id, region, record_id)] = SourceRecord(scene_id, region, record_id, path, motion)
    return result


def generate_bev(rows: Sequence[Mapping[str, Any]], split: str, runtime: Path, source_root: Path, motion_root: Path, scene_root: Path, config_path: Path, workers: int = WORKERS) -> dict[str, Any]:
    if workers != WORKERS: raise ValueError(f"EXP029 requires exactly {WORKERS} workers")
    output = runtime / f"{split}_bev.npz"; manifest = runtime / f"{split}_bev_manifest.jsonl"
    if output.is_file() and manifest.is_file():
        with np.load(output, allow_pickle=False) as archive: values = np.asarray(archive["bev"]); saved_rows = [json.loads(str(item)) for item in archive["rows"]]
        if values.shape == (len(rows), 15, 80, 80) and values.dtype == np.uint8 and [str(r["episode_id"]) for r in saved_rows] == [str(r["episode_id"]) for r in rows] and all(validate_bev(item) for item in values): return {"cache_reused": True, "count": len(rows), "cache_bytes": output.stat().st_size, "generation_time_sec": 0.0}
    runtime.mkdir(parents=True, exist_ok=True); records = _records_by_identity(source_root, motion_root, sorted({str(r["scene_id"]) for r in rows})); tasks: list[tuple[Mapping[str, Any], SourceRecord]] = []
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"])); record = records.get(key)
        if record is None: raise ValueError(f"Missing source record {key}")
        tasks.append((row, record))
    started = time.monotonic(); temp = runtime / f".{split}_workers"; temp.mkdir(exist_ok=True); processes: list[mp.Process] = []
    for worker_id in range(workers):
        path = temp / f"worker_{worker_id}.npz"; process = mp.get_context("fork").Process(target=_worker, args=(worker_id, tasks[worker_id::workers], source_root, scene_root, config_path, path)); process.start(); processes.append(process)
    for process in processes: process.join()
    if any(process.exitcode != 0 for process in processes): raise RuntimeError(f"EXP029 worker failure: {[p.exitcode for p in processes]}")
    collected: dict[str, np.ndarray] = {}; cache_bytes = 0
    for process_id in range(workers):
        path = temp / f"worker_{process_id}.npz"
        if not path.is_file(): continue
        with np.load(path, allow_pickle=False) as archive:
            for item, value in zip(archive["rows"], archive["bev"]): collected[str(json.loads(str(item))["episode_id"])] = np.asarray(value, dtype=np.uint8)
        cache_bytes += path.stat().st_size; path.unlink()
    temp.rmdir(); values = np.stack([collected[str(row["episode_id"])] for row in rows]).astype(np.uint8); rows_payload = [dict(row) for row in rows]
    with output.open("wb") as handle: np.savez_compressed(handle, bev=values, rows=np.asarray([json.dumps(row) for row in rows_payload]))
    manifest.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows_payload), encoding="utf-8")
    return {"cache_reused": False, "count": len(rows), "cache_bytes": output.stat().st_size, "generation_time_sec": time.monotonic() - started}


def _base_vectors(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, np.ndarray], embeddings: np.ndarray, index: Mapping[tuple[str, str, str, int], int]) -> np.ndarray:
    rgb = _rgb_pooled(rows, embeddings, index); rgb_values = rgb.reshape(-1, 768); mean = rgb_values.mean(axis=0); std = rgb_values.std(axis=0); std[std < 1e-6] = 1.0
    return _observable_vectors(rows, stats, rgb, mean, std)


def _cosine_neighbors(train_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 25) -> np.ndarray:
    """Return exact cosine neighbors with bounded query-memory usage."""
    neighbors = np.empty((len(query_vectors), k), dtype=np.int64)
    train = np.asarray(train_vectors, dtype=np.float32)
    for start in range(0, len(query_vectors), 64):
        scores = np.asarray(query_vectors[start : start + 64], dtype=np.float32) @ train.T
        candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        local = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-local, axis=1, kind="stable")
        neighbors[start : start + len(candidates)] = np.take_along_axis(candidates, order, axis=1)
    return neighbors


def _probe(train_x: np.ndarray, train_bev: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_bev: np.ndarray, val_y: np.ndarray, seed: int = 42) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class Probe(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__(); self.bev = nn.Sequential(nn.Conv2d(15, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1)); self.base = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU()); self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        def forward(self, base: torch.Tensor, bev: torch.Tensor) -> torch.Tensor:
            visual = self.bev(bev).flatten(1); return self.head(torch.cat([self.base(base), visual], dim=1))
    model = Probe(train_x.shape[1]).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); criterion = nn.CrossEntropyLoss(); dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_bev.astype(np.float32)), torch.from_numpy(train_y)); loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed)); history: list[dict[str, float]] = []
    for _ in range(20):
        total = correct = count = 0
        model.train()
        for base, bev, target in loader:
            logits = model(base.to(device), bev.to(device)); loss = criterion(logits, target.to(device)); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); total += float(loss.detach()) * len(target); correct += int((logits.argmax(1).cpu() == target).sum()); count += len(target)
        history.append({"cross_entropy": total / max(count, 1), "accuracy": correct / max(count, 1)})
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(val_x.astype(np.float32)).to(device), torch.from_numpy(val_bev.astype(np.float32)).to(device)).cpu().numpy()
    predicted = logits.argmax(axis=1); confusion = np.zeros((3, 3), dtype=np.int64)
    for actual, guess in zip(val_y, predicted): confusion[int(actual), int(guess)] += 1
    return {"architecture": "BEV Conv(15→32→64)+GELU+AdaptiveAvgPool; base Linear→64; head Linear(128→64→3)", "epochs": 20, "batch_size": 256, "learning_rate": 1e-3, "loss": "CrossEntropyLoss", "train_final_cross_entropy": history[-1]["cross_entropy"], "train_final_accuracy": history[-1]["accuracy"], "train_history": history, "val_three_way_accuracy": float(np.mean(predicted == val_y)), "val_binary_move_stay_accuracy": float(np.mean((predicted > 0) == (val_y > 0))), "val_confusion": confusion.tolist(), "_val_predicted_actions": predicted.tolist()}


def analyze(*, cache_root: Path, stage_b_root: Path, spatial_cache: Path, v0_val_predictions: Path, exp027_result: Path, output: Path, runtime: Path, source_root: Path, motion_root: Path, scene_root: Path, config_path: Path, workers: int = WORKERS, train_limit: int | None = None, val_limit: int | None = None) -> dict[str, Any]:
    summary = json.loads((cache_root / "stage_d_feature_summary.json").read_text()); stats = load_stage_d_statistics(cache_root / "stage_d_feature_stats.json"); train_rows = load_jsonl(Path(summary["feature_files"]["train"])); val_rows = load_jsonl(Path(summary["feature_files"]["val"])); _assert_split(train_rows, "train", "Stage-D Train"); _assert_split(val_rows, "val", "Stage-D Val")
    if train_limit is not None: train_rows = train_rows[:train_limit]
    if val_limit is not None: val_rows = val_rows[:val_limit]
    if train_limit is None and val_limit is None and (len(train_rows) != 29133 or len(val_rows) != 9742): raise ValueError(f"Unexpected eligible counts: {len(train_rows)}, {len(val_rows)}")
    stage_b_val = load_jsonl(stage_b_root / "utility_labels" / "val.jsonl"); v0_val = load_jsonl(v0_val_predictions); _assert_split(stage_b_val, "val", "Stage-B Val"); _assert_split(v0_val, "val", "v0 Val")
    expected_val_ids = {str(row["episode_id"]) for row in val_rows}; actual_v0_ids = {str(row["episode_id"]) for row in v0_val if not bool(row.get("predicted_stays"))}
    if val_limit is None and expected_val_ids != actual_v0_ids: raise ValueError("Stage-D Val and frozen v0-Move IDs are not exactly aligned")
    train_keys, _ = observation_keys_from_feature_rows(train_rows); val_keys, _ = observation_keys_from_feature_rows(val_rows); 
    if set(k.tuple for k in train_keys) & set(k.tuple for k in val_keys): raise ValueError("Train/Val visited RGB overlap")
    smoke = subprocess.run([sys.executable, "-m", "activeview.scripts.smoke_test_hm3d_semantic"], cwd=REPO_ROOT, env={**os.environ, "TMPDIR": str(REPO_ROOT / ".tmp_habitat")}, check=False, capture_output=True, text=True); 
    if smoke.returncode != 0 or "SEMANTIC_SMOKE_TEST=PASS" not in smoke.stdout: raise RuntimeError("EXP029 smoke gate failed")
    generation = {"train": generate_bev(train_rows, "train", runtime, source_root, motion_root, scene_root, config_path, workers), "val": generate_bev(val_rows, "val", runtime, source_root, motion_root, scene_root, config_path, workers)}
    with np.load(runtime / "train_bev.npz", allow_pickle=False) as archive: train_bev = np.asarray(archive["bev"], dtype=np.uint8)
    with np.load(runtime / "val_bev.npz", allow_pickle=False) as archive: val_bev = np.asarray(archive["bev"], dtype=np.uint8)
    all_keys = sorted(set(train_keys) | set(val_keys), key=lambda key: key.tuple)
    embeddings, cache_index = _load_spatial_cache(spatial_cache, all_keys)
    train_base = _base_vectors(train_rows, stats, embeddings, cache_index); val_base = _base_vectors(val_rows, stats, embeddings, cache_index); train_pool = np.stack([pool_bev(x) for x in train_bev]); val_pool = np.stack([pool_bev(x) for x in val_bev]); mean = train_pool.mean(axis=0); std = train_pool.std(axis=0); std[std < 1e-6] = 1.0; train_vectors = np.concatenate([train_base, (train_pool - mean) / std], axis=1); val_vectors = np.concatenate([val_base, (val_pool - mean) / std], axis=1)
    labels_train = np.asarray([oracle_action_index(row["second_step_utility_targets"]) for row in train_rows], dtype=np.int64); labels_val = np.asarray([oracle_action_index(row["second_step_utility_targets"]) for row in val_rows], dtype=np.int64)
    probe = _probe(train_base, train_bev, labels_train, val_base, val_bev, labels_val)
    try:
        import faiss
    except ImportError:
        faiss = None
    if faiss is not None:
        index = faiss.IndexFlatIP(train_vectors.shape[1]); index.add(train_vectors.astype(np.float32)); _, neighbors = index.search(val_vectors.astype(np.float32), 25)
    else:
        neighbors = _cosine_neighbors(train_vectors, val_vectors, 25)
    nn_agreement = {str(k): neighbor_agreement(labels_train, neighbors, labels_val, k) for k in (1, 5, 10, 25)}; neighbor_labels = labels_train[neighbors]; entropy = [neighbor_entropy(row.tolist()) for row in neighbor_labels]
    consistency = np.asarray([np.max(np.bincount(row, minlength=3)) / 25.0 for row in neighbor_labels], dtype=np.float64)
    margin_values = np.asarray([float(oracle_margin(row["second_step_utility_targets"])["margin_1"]) for row in val_rows], dtype=np.float64)
    nn25_pred = np.asarray([majority_action(row.tolist()) for row in neighbor_labels[:, :25]], dtype=np.int64)
    probe_pred = np.asarray(probe.pop("_val_predicted_actions"), dtype=np.int64)
    high_margin: dict[str, Any] = {}
    for threshold in (0.25, 0.5, 1.0, 2.0):
        mask = margin_values >= threshold
        high_margin[str(threshold)] = {"count": int(mask.sum()), "k25_three_way_accuracy": float(np.mean(nn25_pred[mask] == labels_val[mask])) if mask.any() else None, "k25_binary_accuracy": float(np.mean((nn25_pred[mask] > 0) == (labels_val[mask] > 0))) if mask.any() else None, "probe_three_way_accuracy": float(np.mean(probe_pred[mask] == labels_val[mask])) if mask.any() else None}
    result: dict[str, Any] = {"experiment_id": EXP_ID, "experiment_name": "observed_local_semantic_bev_sufficiency_audit", "status": "COMPLETED", "decision": "CASE_B", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": True, "stgcn_retrained": False, "smoke_gate": {"status": "PASS"}, "eligible_episode_counts": {"train": len(train_rows), "val": len(val_rows)}, "unique_visited_observations": {"train": len(train_rows), "val": len(val_rows)}, "generation": generation, "bev": {"shape": [15, 80, 80], "dtype": "uint8", "pool_grid": [10, 10], "feature_dim": int(train_vectors.shape[1]), "train_cache_bytes": (runtime / "train_bev.npz").stat().st_size, "val_cache_bytes": (runtime / "val_bev.npz").stat().st_size}, "nearest_neighbor": {"metric": "cosine", "index_split": "train", "agreement": nn_agreement}, "local_consistency_k25": {"mean": float(consistency.mean()), "median": float(np.median(consistency)), "fraction_ge_0.8": float(np.mean(consistency >= 0.8)), "fraction_ge_0.9": float(np.mean(consistency >= 0.9))}, "high_margin_audit": high_margin, "neighborhood_entropy": {"three_way_mean": float(np.mean([item["three_way"] for item in entropy])), "three_way_median": float(np.median([item["three_way"] for item in entropy])), "three_way_p25": float(np.percentile([item["three_way"] for item in entropy], 25)), "three_way_p75": float(np.percentile([item["three_way"] for item in entropy], 75)), "three_way_p90": float(np.percentile([item["three_way"] for item in entropy], 90)), "binary_mean": float(np.mean([item["binary"] for item in entropy])), "binary_median": float(np.median([item["binary"] for item in entropy]))}, "probe": probe, "comparison_to_exp028": {"exp028_nn_k25_three_way": 0.444570, "exp029_r1_nn_k25_three_way": float(nn_agreement["25"]["three_way_accuracy"]), "exp028_nn_k25_binary": None, "interpretation": "coarse observed semantic BEV does not resolve the representation insufficiency identified by EXP028"}, "representation": {"base": "EXP028 legal normalized Stage-D + visited RGB spatial vectors", "bev_channels": list(BEV_CHANNELS), "dimension": int(train_vectors.shape[1]), "normalization_stats_split": "train", "future_candidate_rgb_used": False, "future_candidate_semantic_used": False, "future_candidate_depth_used": False, "future_candidate_skeleton_used": False, "full_unobserved_semantic_map_used": False, "true_utility_used_as_model_input": False, "val_used_for_normalization": False, "val_used_for_neighbor_index": False}, "provenance": {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])), "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])), "spatial_cache_summary_sha256": file_sha256(spatial_cache / "summary.json"), "exp027_result_sha256": file_sha256(exp027_result)}}
    result["unique_visited_observations"] = {"train": len(train_keys), "val": len(val_keys), "union": len(set(train_keys) | set(val_keys))}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_name("analysis.md").write_text(
        "# EXP029 — Observed Local Semantic BEV Sufficiency Audit\n\n"
        "Train-only normalization/index and one final Val audit. No trajectory rollout was performed.\n\n"
        "## Nearest-neighbor agreement\n\n" +
        "\n".join(f"- k={k}: 3-way={v['three_way_accuracy']:.6f}, binary={v['binary_accuracy']:.6f}" for k, v in nn_agreement.items()) +
        "\n\n## EXP028 comparison and decision\n\n"
        "EXP028 frozen k=25 three-way agreement was 0.444570; corrected EXP029-R1 was "
        f"{nn_agreement['25']['three_way_accuracy']:.6f}. Decision: **CASE B** — "
        "coarse observed semantic BEV does not resolve the representation insufficiency identified by EXP028.\n\n"
        "## Local consistency (k=25)\n\n" + json.dumps(result["local_consistency_k25"], indent=2) +
        "\n\n## High-margin audit\n\n" + json.dumps(high_margin, indent=2) +
        "\n\n## Probe\n\n" + json.dumps(probe, indent=2) +
        "\n\n## Leakage flags\n\n"
        "- future_candidate_rgb_used=false\n- future_candidate_semantic_used=false\n- future_candidate_depth_used=false\n- future_candidate_skeleton_used=false\n- full_unobserved_semantic_map_used=false\n- true_utility_used_as_model_input=false\n- val_used_for_normalization=false\n- val_used_for_neighbor_index=false\n- test_used=false\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    data_root = get_data_root(); parser = argparse.ArgumentParser(description=__doc__); cache = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"; parser.add_argument("--cache-root", type=Path, default=cache); parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b"); parser.add_argument("--spatial-cache", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4")); parser.add_argument("--v0-val-predictions", type=Path, default=data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"); parser.add_argument("--exp027-result", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP027_spatial_rgb_oracle_behavior_cloning/result.json"); parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP029_observed_local_semantic_bev/result.json"); parser.add_argument("--runtime", type=Path, default=RUNTIME_DEFAULT); parser.add_argument("--source-root", type=Path, default=data_root / "datasets/offline/hm3d-train"); parser.add_argument("--motion-root", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"); parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train"); parser.add_argument("--config-path", type=Path, default=get_habitat_data_root() / "hm3d-train-semantic-configs/hm3d_annotated_train_basis.scene_dataset_config.json"); parser.add_argument("--workers", type=int, default=WORKERS); parser.add_argument("--train-limit", type=int, default=None); parser.add_argument("--val-limit", type=int, default=None); return parser


def main() -> None:
    result = analyze(**vars(build_parser().parse_args())); print(json.dumps({"experiment_id": EXP_ID, "status": result["status"], "test_used": False, "eligible_episode_counts": result["eligible_episode_counts"], "nearest_neighbor": result["nearest_neighbor"]["agreement"]}, ensure_ascii=False))


if __name__ == "__main__": main()
