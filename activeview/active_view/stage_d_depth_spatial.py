"""Habitat metric-depth cache for EXP026 visited s0/s1 observations."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from activeview.active_view.stage_d_rgb_context import RGBObservationKey, RGB_DATASET_VERSION
from activeview.scripts.generate_hm3d_train_rgb_observations import (
    FRAME_INDEX,
    HFOV_DEG,
    IMAGE_SIZE,
    SENSOR_HEIGHT_M,
    TARGET_FRAMES,
    SourceRecord,
    _load_skeleton_metadata,
    _load_source_records,
    _set_agent_state,
)
from activeview.core.paths import get_humanoid_urdf_path
from activeview.dataset.babel_clean_dataset_generator import MotionConverter, URDF_PATH, _load_resampled_motion, apply_humanoid_pose, precompute_grounding_offsets


DEPTH_GRID_SIZE = 4
DEPTH_FEATURES_PER_CELL = 4
DEPTH_FEATURE_DIM = DEPTH_GRID_SIZE * DEPTH_GRID_SIZE * DEPTH_FEATURES_PER_CELL


class SpatialRGBDUtilityRegressor(nn.Module):
    """EXP026 RGB spatial/depth branch and raw utility regression head."""

    input_dim = 128 + 1 + 128 + 128 + 128 + 32 + 32 + 32

    def __init__(self) -> None:
        super().__init__()
        self.rgb_projector = nn.Sequential(nn.Linear(768, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu")
        self.rgb_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.depth_encoder = nn.Sequential(nn.Linear(4, 32), nn.GELU())
        self.regression_head = nn.Sequential(nn.Linear(self.input_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def _rgb(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (16, 768):
            raise ValueError("RGB spatial input must have shape (batch,16,768)")
        return self.rgb_encoder(self.rgb_projector(values)).mean(dim=1)

    def _depth(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (16, 4):
            raise ValueError("Depth input must have shape (batch,16,4)")
        return self.depth_encoder(values).mean(dim=1)

    def forward(self, contextual: torch.Tensor, predicted: torch.Tensor, rgb_s0: torch.Tensor, rgb_s1: torch.Tensor, depth_s0: torch.Tensor, depth_s1: torch.Tensor) -> torch.Tensor:
        if contextual.ndim != 2 or contextual.shape[1] != 128:
            raise ValueError("contextual input must have shape (batch,128)")
        if predicted.ndim == 1:
            predicted = predicted.unsqueeze(1)
        if predicted.shape != (contextual.shape[0], 1):
            raise ValueError("predicted utility must have shape (batch,1)")
        z0, z1 = self._rgb(rgb_s0), self._rgb(rgb_s1)
        d0, d1 = self._depth(depth_s0), self._depth(depth_s1)
        features = torch.cat([contextual, predicted, z0, z1, z1 - z0, d0, d1, d1 - d0], dim=1)
        return self.regression_head(features).squeeze(-1)


def depth_spatial_features(depth: np.ndarray) -> np.ndarray:
    """Pool metric depth into [16,4] mean/min/std/valid-ratio cells."""
    values = np.asarray(depth, dtype=np.float32)
    if values.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Unexpected depth shape: {values.shape}")
    valid = np.isfinite(values) & (values >= 0.0)
    clipped = np.clip(np.where(valid, values, 0.0), 0.0, 10.0)
    output = np.zeros((DEPTH_GRID_SIZE, DEPTH_GRID_SIZE, DEPTH_FEATURES_PER_CELL), dtype=np.float32)
    cell_h = IMAGE_SIZE // DEPTH_GRID_SIZE
    for row in range(DEPTH_GRID_SIZE):
        for col in range(DEPTH_GRID_SIZE):
            cell = clipped[row * cell_h:(row + 1) * cell_h, col * cell_h:(col + 1) * cell_h]
            mask = valid[row * cell_h:(row + 1) * cell_h, col * cell_h:(col + 1) * cell_h]
            finite = cell[mask]
            if finite.size:
                output[row, col] = [float(finite.mean()), float(finite.min()), float(finite.std()), float(finite.size / mask.size)]
    return output.reshape(16, 4).astype(np.float16)


def _make_depth_sim(scene_root: Path, scene_id: str) -> tuple[Any, Any]:
    import habitat_sim
    import magnum as mn
    scene_dir = scene_root / scene_id
    glb = next(scene_dir.glob("*.basis.glb"))
    navmesh = next(scene_dir.glob("*.basis.navmesh"))
    backend = habitat_sim.SimulatorConfiguration(); backend.scene_id = str(glb); backend.enable_physics = True
    agents = []
    for index in range(4):
        sensor = habitat_sim.CameraSensorSpec(); sensor.uuid = f"depth_{index}"; sensor.sensor_type = habitat_sim.SensorType.DEPTH; sensor.resolution = [IMAGE_SIZE, IMAGE_SIZE]; sensor.position = mn.Vector3(0.0, SENSOR_HEIGHT_M, 0.0); sensor.hfov = HFOV_DEG
        agent = habitat_sim.AgentConfiguration(); agent.sensor_specifications = [sensor]; agents.append(agent)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, agents)); sim.pathfinder.load_nav_mesh(str(navmesh))
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    return sim, human


def _render_depths(sim: Any, human: Any, record: SourceRecord, source_meta: Mapping[str, Any], viewpoint_ids: Sequence[int]) -> dict[int, np.ndarray]:
    motion = _load_resampled_motion(record.motion, TARGET_FRAMES); converted = MotionConverter(URDF_PATH).convert(motion)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32); roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
    offsets, _ = precompute_grounding_offsets(human, joints, roots, scene_yaw_deg=0.0); base = np.asarray(source_meta["placement_position"], dtype=np.float32)
    apply_humanoid_pose(human, joints[FRAME_INDEX], roots[FRAME_INDEX], base_position=base, scene_yaw_deg=0.0, floor_y=float(base[1]), grounding_offset=float(offsets[FRAME_INDEX]))
    positions = np.asarray(source_meta["viewpoint_agent_positions"], dtype=np.float32); rotations = np.asarray(source_meta["viewpoint_rotations_wxyz"], dtype=np.float32)
    output: dict[int, np.ndarray] = {}
    for start in range(0, len(viewpoint_ids), 4):
        batch = list(viewpoint_ids[start:start + 4])
        for agent_index, view_index in enumerate(batch): _set_agent_state(sim.get_agent(agent_index), positions[view_index], rotations[view_index])
        observations = sim.get_sensor_observations(list(range(len(batch))))
        for agent_index, view_index in enumerate(batch):
            depth = np.asarray(observations[agent_index][f"depth_{agent_index}"], dtype=np.float32)
            output[int(view_index)] = depth
    return output


def _worker(worker_id: int, tasks: Sequence[tuple[SourceRecord, tuple[RGBObservationKey, ...]]], source_root: Path, scene_root: Path, output_path: Path) -> None:
    if not tasks: return
    # Habitat/Bullet emits one extremely verbose URDF warning per joint while
    # restoring the frozen humanoid pose.  Keep the experiment log usable; the
    # parent still validates worker exit codes and output files.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    sim, human = _make_depth_sim(scene_root, tasks[0][0].scene_id); rows: list[dict[str, Any]] = []; features: list[np.ndarray] = []
    try:
        for record, keys in tasks:
            meta = _load_skeleton_metadata(record.source_path); raw = _render_depths(sim, human, record, meta, [key.viewpoint_id for key in keys]); source_hash = _sha256(record.source_path)
            for key in keys:
                rows.append({"scene_id": key.scene_id, "region": key.region, "record_id": key.record_id, "viewpoint_id": key.viewpoint_id, "source_skeleton_sha256": source_hash, "frame_index": FRAME_INDEX, "resolution": [IMAGE_SIZE, IMAGE_SIZE], "hfov_deg": HFOV_DEG, "depth_unit": "meter", "feature_shape": [16, 4], "dtype": "float16"})
                features.append(depth_spatial_features(raw[key.viewpoint_id]))
        with output_path.open("wb") as handle: np.savez_compressed(handle, features=np.stack(features), rows=np.asarray([json.dumps(row) for row in rows]))
    finally:
        os._exit(0)


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256();
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def build_or_load_depth_cache(*, source_root: Path, motion_manifest: Path, scene_root: Path, cache_dir: Path, keys: Sequence[RGBObservationKey], workers: int = 16) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(keys, key=lambda key: key.tuple); cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path, manifest_path, summary_path = cache_dir / "features.npy", cache_dir / "manifest.jsonl", cache_dir / "summary.json"
    if emb_path.is_file() and manifest_path.is_file() and summary_path.is_file():
        try:
            values = np.load(emb_path, mmap_mode="r"); rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
            if values.shape == (len(ordered), 16, 4) and values.dtype == np.float16 and rows == _manifest_rows(ordered): return values, rows, {"cache_reused": True, "cache_hit_count": len(ordered), "cache_miss_count": 0, "extraction_time_sec": 0.0}
        except (OSError, ValueError, json.JSONDecodeError): pass
    started = time.monotonic(); records, _ = _load_source_records(source_root, motion_manifest)
    record_by_key = {(row.scene_id, row.region, row.record_id): row for row in records}; grouped: dict[str, list[tuple[SourceRecord, tuple[RGBObservationKey, ...]]]] = {}
    key_groups: dict[tuple[str, str, str], list[RGBObservationKey]] = defaultdict(list)
    for key in ordered: key_groups[(key.scene_id, key.region, key.record_id)].append(key)
    for identity, group in key_groups.items():
        record = record_by_key.get(identity)
        if record is None: raise ValueError(f"Missing canonical skeleton record for {identity}")
        grouped.setdefault(record.scene_id, []).append((record, tuple(sorted(group, key=lambda key: key.viewpoint_id))))
    tmp_root = cache_dir / ".workers"; tmp_root.mkdir(exist_ok=True); collected: list[dict[str, Any]] = []; chunks: list[np.ndarray] = []
    for scene_id, scene_tasks in grouped.items():
        processes: list[mp.Process] = []
        for worker_id in range(workers):
            path = tmp_root / f"{scene_id}_{worker_id}.npz"; process = mp.get_context("spawn").Process(target=_worker, args=(worker_id, scene_tasks[worker_id::workers], source_root, scene_root, path)); process.start(); processes.append(process)
        for process in processes: process.join()
        if any(process.exitcode != 0 for process in processes): raise RuntimeError(f"Depth worker failed for {scene_id}")
        for worker_id in range(workers):
            path = tmp_root / f"{scene_id}_{worker_id}.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as archive:
                    chunks.append(np.asarray(archive["features"], dtype=np.float16)); collected.extend(json.loads(str(item)) for item in archive["rows"])
                path.unlink()
    tmp_root.rmdir(); by_key = {(row["scene_id"], row["region"], row["record_id"], int(row["viewpoint_id"])): (row, feat) for row, feat in zip(collected, np.concatenate(chunks, axis=0))}
    rows = _manifest_rows(ordered); values = np.stack([by_key[(key.scene_id, key.region, key.record_id, key.viewpoint_id)][1] for key in ordered]).astype(np.float16)
    np.save(emb_path.with_suffix(".tmp.npy"), values); emb_path.with_suffix(".tmp.npy").replace(emb_path); manifest_path.with_suffix(".tmp.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)); manifest_path.with_suffix(".tmp.jsonl").replace(manifest_path)
    summary = {"encoder": "Habitat metric depth", "rgb_dataset_version": RGB_DATASET_VERSION, "frame_index": FRAME_INDEX, "sensor_resolution": [IMAGE_SIZE, IMAGE_SIZE], "hfov_deg": HFOV_DEG, "feature_shape": [16, 4], "dtype": "float16", "observation_count": len(ordered), "future_candidate_depth_used": False}
    summary_path.write_text(json.dumps(summary, indent=2)); return values, rows, {"cache_reused": False, "cache_hit_count": 0, "cache_miss_count": len(ordered), "extraction_time_sec": time.monotonic() - started}


def _manifest_rows(keys: Sequence[RGBObservationKey]) -> list[dict[str, Any]]:
    return [{"scene_id": key.scene_id, "region": key.region, "record_id": key.record_id, "viewpoint_id": key.viewpoint_id, "frame_index": FRAME_INDEX, "feature_shape": [16, 4], "dtype": "float16"} for key in keys]
