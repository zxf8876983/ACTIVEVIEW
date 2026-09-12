#!/usr/bin/env python3
"""Final Train/Val-only structured observability oracle audit for reduced12.

This diagnostic keeps the action set and frozen recognizer fixed while testing
whether a frame-0 17-joint scene-visibility vector and the current frame-0
pose explain more candidate utility than a scalar visibility target.  Policy
Test, future candidate observations, and production recognizer changes are
never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (
    H36M17_LINKS,
    SENSOR_HEIGHT_M,
    _archive_path,
    _load_npz,
    _load_raw_records,
    _load_resampled_motion,
    _placement_map,
    _ray_visible,
    _scene_sim,
    _world_h36m17,
)
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.privileged_information_ladder_support import ranking_metrics
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    _archive_views,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import _load_options
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import _option_geometry

NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
MAX_OPTIONS = 22
JOINT_COUNT = len(H36M17_LINKS)
POSE_DIM = 3 * JOINT_COUNT
SEED = 42
EPOCHS = 12
OBS_PER_RECORD = 16
TRAIN_BATCH = 512
EVAL_BATCH = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4
LISTWISE_WEIGHT = 0.5
LISTWISE_TAU = 0.5
STRUCTURED_REL = Path("diagnostics/structured_observability_final_audit")
SCALAR_REL = Path("diagnostics/frame0_visibility_predictor_v1")
SHARED_HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "view_agnostic_frozen_encoder_head/shared_head_best.pth"
)
CHECKPOINT_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/structured_observability_final_audit"
)
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/structured_observability_final_audit"
HISTORICAL_LADDER = REPO_ROOT / "experiments/reduced12_eight_placement_v1/privileged_information_ladder/result.json"
HISTORICAL_GEOMETRY_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/privileged_information_ladder/geometryonly_utility.pth"
)

BRANCHES = (
    "ScalarVisibility+Geometry",
    "StructuredVisibility17+Geometry",
    "CurrentPose+Geometry",
    "CurrentPose+StructuredVisibility17+Geometry",
    "GTAction+CurrentPose+StructuredVisibility17+Geometry",
)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actions(row: Mapping[str, Any]) -> list[int]:
    actions = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
    if len(actions) > MAX_OPTIONS or len(set(actions)) != len(actions):
        raise ValueError(f"invalid legal action set for {row['episode_id']}")
    return actions


def _validate_action_arrays(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, name: str) -> None:
    if ids.shape != (len(rows), MAX_OPTIONS) or mask.shape != ids.shape:
        raise ValueError(f"{name} action shape mismatch: {ids.shape}/{mask.shape}")
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        if ids[index, active].astype(int).tolist() != _actions(row):
            raise ValueError(f"{name} action-set mismatch for {row['episode_id']}")


def _load_cache(path: Path, metadata_path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray] | None:
    if not path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        return None
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"cache is marked Test-derived: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _build_structured_cache(
    data_root: Path,
    scene_root: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_root = data_root / STRUCTURED_REL
    cache = _load_cache(cache_root / f"{split}.npz", cache_root / f"{split}.json", rows)
    if cache is not None:
        _validate_action_arrays(rows, cache["ids"], cache["mask"], f"{split} structured visibility")
        if cache["visibility"].shape != (len(rows), MAX_OPTIONS, JOINT_COUNT):
            raise ValueError(f"{split} structured visibility shape mismatch")
        return cache, json.loads((cache_root / f"{split}.json").read_text(encoding="utf-8"))
    raw_records = _load_raw_records(data_root)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["scene_id"])].append((index, row))
    visibility = np.full((len(rows), MAX_OPTIONS, JOINT_COUNT), -1.0, dtype=np.float32)
    ids = np.full((len(rows), MAX_OPTIONS), -1, dtype=np.int64)
    mask = np.zeros((len(rows), MAX_OPTIONS), dtype=bool)
    started = time.monotonic()
    for scene_id, scene_rows in sorted(grouped.items()):
        ray_sim = _scene_sim(scene_root, scene_id, physics=True)
        human_sim = _scene_sim(scene_root, scene_id, physics=True)
        human = human_sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            str(get_humanoid_urdf_path("male_0"))
        )
        from activeview.data.motion.motion_converter import MotionConverter

        converter = MotionConverter(get_humanoid_urdf_path("male_0"))
        placements = _placement_map(data_root, scene_id)
        converted_cache: dict[str, Mapping[str, Any]] = {}
        world_cache: dict[tuple[str, str], np.ndarray] = {}
        archive_cache: dict[str, dict[str, np.ndarray]] = {}
        visibility_action_cache: dict[tuple[str, tuple[str, str], int], np.ndarray] = {}
        try:
            for index, row in scene_rows:
                archive_key = str(row["archive_path"])
                if archive_key not in archive_cache:
                    archive_cache[archive_key] = _load_npz(Path(archive_key))
                archive = archive_cache[archive_key]
                viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                camera_positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
                if viewpoint_ids.shape != (NUM_VIEWS,) or camera_positions.shape != (NUM_VIEWS, 3):
                    raise ValueError(f"invalid viewpoint archive schema: {row['episode_id']}")
                record_id, placement_id = str(row["record_id"]), str(row["region"])
                if record_id not in raw_records or placement_id not in placements:
                    raise ValueError(f"missing motion/placement for {row['episode_id']}")
                if record_id not in converted_cache:
                    converted_cache[record_id] = converter.convert(_load_resampled_motion(raw_records[record_id], 30))
                world_key = (record_id, placement_id)
                if world_key not in world_cache:
                    placement = placements[placement_id]
                    world_cache[world_key] = _world_h36m17(
                        human,
                        converted_cache[record_id],
                        placement,
                        float(placement["yaw_deg"]),
                    )
                viewpoint_index = {int(value): offset for offset, value in enumerate(viewpoint_ids.tolist())}
                for slot, action in enumerate(_actions(row)):
                    if action not in viewpoint_index:
                        raise ValueError(f"legal viewpoint absent from archive: {row['episode_id']}/{action}")
                    cache_key = (archive_key, world_key, int(action))
                    if cache_key not in visibility_action_cache:
                        camera = camera_positions[viewpoint_index[action]] + np.asarray(
                            [0.0, SENSOR_HEIGHT_M, 0.0], dtype=np.float32
                        )
                        visibility_action_cache[cache_key] = np.asarray(
                            [_ray_visible(ray_sim, camera, world_cache[world_key][0, joint]) for joint in range(JOINT_COUNT)],
                            dtype=np.float32,
                        )
                    visibility[index, slot] = visibility_action_cache[cache_key]
                    ids[index, slot] = action
                    mask[index, slot] = True
                if (index + 1) % 500 == 0:
                    print(f"[structured-{split}] {index + 1}/{len(rows)} ({time.monotonic() - started:.1f}s)", flush=True)
        finally:
            human_sim.close()
            ray_sim.close()
    if not np.isfinite(visibility[mask]).all():
        raise ValueError(f"non-finite {split} structured visibility")
    metadata: dict[str, Any] = {
        "split": split,
        "rows": len(rows),
        "signature": row_signature(rows),
        "shape": list(visibility.shape),
        "dtype": str(visibility.dtype),
        "definition": "frame-0 H36M17 binary scene-only Habitat raycast",
        "frame_ids": [0],
        "stay_coverage": float(mask[:, 0].mean()),
        "candidate_coverage": float(mask[:, 1:].mean()),
        "candidate_count": int(mask[:, 1:].sum()),
        "per_joint_mean": np.mean(visibility[mask], axis=0).astype(float).tolist(),
        "future_motion_used": False,
        "future_candidate_observation_used": False,
        "test_used": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_root / f"{split}.npz", visibility=visibility, ids=ids, mask=mask)
    (cache_root / f"{split}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"visibility": visibility, "ids": ids, "mask": mask}, metadata


def _load_scalar(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> np.ndarray:
    root = data_root / SCALAR_REL
    metadata = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        raise ValueError(f"{split} scalar visibility signature mismatch")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"{split} scalar visibility is Test-derived")
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        scores, old_ids, old_mask = np.asarray(archive["scores"]), np.asarray(archive["ids"]), np.asarray(archive["mask"])
    scalar = np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32)
    legal_mask = np.zeros_like(scalar, dtype=bool)
    for index, row in enumerate(rows):
        old = {int(value): slot for slot, value in enumerate(old_ids[index, old_mask[index]].tolist())}
        for slot, action in enumerate(_actions(row)):
            if action not in old:
                raise ValueError(f"{split} scalar visibility missing {row['episode_id']}/{action}")
            scalar[index, slot] = float(scores[index, old[action]])
            legal_mask[index, slot] = True
    if not np.isfinite(scalar[legal_mask]).all():
        raise ValueError(f"non-finite {split} scalar visibility")
    return scalar


def _load_pose(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> np.ndarray:
    root = data_root / STRUCTURED_REL
    path, meta_path = root / f"{split}_current_pose.npz", root / f"{split}_current_pose.json"
    if path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(metadata.get("rows", -1)) == len(rows) and metadata.get("signature") == row_signature(rows):
            with np.load(path, allow_pickle=False) as archive:
                pose = np.asarray(archive["pose"], dtype=np.float32)
            if pose.shape == (len(rows), POSE_DIM) and np.isfinite(pose).all():
                return pose
    pose = np.empty((len(rows), POSE_DIM), dtype=np.float32)
    for index, row in enumerate(rows):
        skeleton = _archive_views(Path(str(row["archive_path"])))
        pose[index] = skeleton[int(row["current_viewpoint_id"]), :, 0, :].reshape(-1)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pose=pose)
    meta_path.write_text(
        json.dumps({"split": split, "rows": len(rows), "signature": row_signature(rows), "definition": "current frame-0 archived skeleton only", "test_used": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return pose


def _build_inputs(
    branch: str,
    geometry: np.ndarray,
    scalar: np.ndarray,
    structured: np.ndarray,
    pose: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    n_rows, n_slots = geometry.shape[:2]
    # Replace padded scalar-cache NaNs with a finite sentinel; loss masking remains unchanged.
    scalar_feature = np.nan_to_num(scalar[..., None].astype(np.float32), nan=0.0)
    structured_feature = np.nan_to_num(structured.astype(np.float32), nan=0.0)
    pose_feature = np.repeat(pose[:, None, :], n_slots, axis=1).astype(np.float32)
    if branch == "ScalarVisibility+Geometry":
        return np.concatenate((geometry, scalar_feature), axis=-1)
    if branch == "StructuredVisibility17+Geometry":
        return np.concatenate((geometry, structured_feature), axis=-1)
    if branch == "CurrentPose+Geometry":
        return np.concatenate((pose_feature, geometry), axis=-1)
    if branch == "CurrentPose+StructuredVisibility17+Geometry":
        return np.concatenate((pose_feature, structured_feature, geometry), axis=-1)
    if branch == "GTAction+CurrentPose+StructuredVisibility17+Geometry":
        one_hot = np.broadcast_to(np.eye(NUM_CLASSES, dtype=np.float32)[labels][:, None, :], (n_rows, n_slots, NUM_CLASSES))
        return np.concatenate((one_hot, pose_feature, structured_feature, geometry), axis=-1)
    raise ValueError(f"unknown branch: {branch}")


class UtilityMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape[:-1]
        return self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(shape)


def _utility_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(predicted[mask], target[mask])
    terms: list[torch.Tensor] = []
    for index in range(predicted.shape[0]):
        active = mask[index]
        if int(active.sum()) >= 2:
            target_prob = torch.softmax(target[index, active] / LISTWISE_TAU, dim=0)
            terms.append(-(target_prob * torch.log_softmax(predicted[index, active] / LISTWISE_TAU, dim=0)).sum())
    ranking = torch.stack(terms).mean() if terms else regression.new_zeros(())
    return regression + LISTWISE_WEIGHT * ranking, regression, ranking


def _sample_rows(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["record_id"])].append(index)
    selected: list[int] = []
    for record in sorted(groups):
        candidates = np.asarray(groups[record], dtype=np.int64)
        selected.extend(rng.choice(candidates, OBS_PER_RECORD, replace=len(candidates) < OBS_PER_RECORD).tolist())
    return np.asarray(selected, dtype=np.int64)


def _train_branch(
    branch: str,
    train_inputs: np.ndarray,
    val_inputs: np.ndarray,
    train_target: np.ndarray,
    val_target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    output_root: Path,
    device: torch.device,
) -> tuple[UtilityMLP, dict[str, Any]]:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in branch).strip("_")
    checkpoint = data_root / CHECKPOINT_REL / f"{slug}.pth"
    summary_path = output_root / f"{slug}_training.json"
    model = UtilityMLP(int(train_inputs.shape[-1])).to(device)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if int(payload.get("input_dim", -1)) != train_inputs.shape[-1] or bool(payload.get("test_used", True)):
            raise ValueError(f"checkpoint metadata mismatch for {branch}")
        model.load_state_dict(payload["state_dict"])
        return model.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    history: list[dict[str, Any]] = []
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        sampled = _sample_rows(train_rows, rng)
        order = rng.permutation(len(sampled))
        model.train()
        train_total: list[float] = []
        train_reg: list[float] = []
        train_rank: list[float] = []
        for start in range(0, len(order), TRAIN_BATCH):
            index = sampled[order[start : start + TRAIN_BATCH]]
            inputs = torch.from_numpy(train_inputs[index]).to(device, non_blocking=True)
            target = torch.from_numpy(train_target[index]).to(device, non_blocking=True)
            mask = torch.from_numpy(train_mask[index]).to(device, non_blocking=True)
            total, regression, ranking = _utility_loss(model(inputs), target, mask)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total.append(float(total.detach().cpu()))
            train_reg.append(float(regression.detach().cpu()))
            train_rank.append(float(ranking.detach().cpu()))
        model.eval()
        val_total: list[float] = []
        val_reg: list[float] = []
        val_rank: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_inputs), EVAL_BATCH):
                sl = slice(start, start + EVAL_BATCH)
                pred = model(torch.from_numpy(val_inputs[sl]).to(device, non_blocking=True))
                total, regression, ranking = _utility_loss(
                    pred,
                    torch.from_numpy(val_target[sl]).to(device, non_blocking=True),
                    torch.from_numpy(val_mask[sl]).to(device, non_blocking=True),
                )
                val_total.append(float(total.cpu()))
                val_reg.append(float(regression.cpu()))
                val_rank.append(float(ranking.cpu()))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_total)),
            "train_regression": float(np.mean(train_reg)),
            "train_listwise": float(np.mean(train_rank)),
            "val_total_loss": float(np.mean(val_total)),
            "val_regression": float(np.mean(val_reg)),
            "val_listwise": float(np.mean(val_rank)),
            "sampled_contexts": int(len(sampled)),
            "unique_records": int(len({str(row['record_id']) for row in train_rows})),
        }
        history.append(record)
        print(f"[{branch}] epoch={epoch:02d} train={record['train_loss']:.6f} val={record['val_total_loss']:.6f}", flush=True)
        if record["val_total_loss"] < best_loss:
            best_loss, best_epoch = float(record["val_total_loss"]), epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "branch": branch, "input_dim": int(train_inputs.shape[-1]), "epoch": epoch, "seed": SEED, "test_used": False},
                checkpoint,
            )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {
        "branch": branch,
        "seed": SEED,
        "epochs": EPOCHS,
        "input_dim": int(train_inputs.shape[-1]),
        "architecture": "Linear(input_dim,256)-GELU-Linear(256,256)-GELU-Linear(256,1)",
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "observations_per_record": OBS_PER_RECORD,
        "checkpoint_selection": "minimum Moving-Val utility prediction loss",
        "best_epoch": best_epoch,
        "best_val_total_loss": best_loss,
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model.eval(), summary


def _predict(model: UtilityMLP, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), EVAL_BATCH):
            batch = torch.from_numpy(inputs[start : start + EVAL_BATCH]).to(device, non_blocking=True)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _select(scores: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index in range(len(scores)):
        active = np.flatnonzero(mask[index])
        slot = min(active.tolist(), key=lambda item: (-float(scores[index, item]), 0 if item == 0 else 1, int(ids[index, item])))
        actions.append(int(ids[index, slot]))
    return actions


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((ids[index] == int(action)) & mask[index])
        if slots.size != 1:
            raise ValueError(f"selected action is not uniquely legal at row {index}: {action}")
        predictions.append(int(np.argmax(logp[index, int(slots[0])])))
    return np.asarray(predictions, dtype=np.int64)


def _metrics(name: str, rows: Sequence[Mapping[str, Any]], labels: np.ndarray, predictions: np.ndarray, actions: Sequence[int], deployable: bool) -> dict[str, Any]:
    result = classification(labels, predictions)
    move_rate = float(np.mean([int(action) != int(row["current_viewpoint_id"]) for action, row in zip(actions, rows)]))
    result.update({"selector": name, "move_rate": move_rate, "stay_rate": 1.0 - move_rate, "deployable": deployable, "test_used": False})
    return result


def _target_utility(logp: np.ndarray, labels: np.ndarray) -> np.ndarray:
    target = np.full(logp.shape[:2], np.nan, dtype=np.float32)
    for index, label in enumerate(labels):
        target[index] = logp[index, :, int(label)]
    return target


def _smooth_l1_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(pred[mask] - target[mask])
    return float(np.mean(np.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)))


def _load_historical_geometry() -> tuple[dict[str, Any], dict[str, Any]]:
    if not HISTORICAL_LADDER.is_file():
        raise FileNotFoundError(HISTORICAL_LADDER)
    payload = json.loads(HISTORICAL_LADDER.read_text(encoding="utf-8"))
    method = payload["methods"]["GeometryOnly Utility"]
    ranking = payload["ranking_metrics"]["GeometryOnly Utility"]
    return method, ranking


def _joint_importance(
    model: UtilityMLP,
    val_inputs: np.ndarray,
    val_geometry: np.ndarray,
    val_structured: np.ndarray,
    val_pose: np.ndarray,
    val_labels: np.ndarray,
    val_target: np.ndarray,
    val_logp: np.ndarray,
    val_ids: np.ndarray,
    val_mask: np.ndarray,
    val_rows: Sequence[Mapping[str, Any]],
    normal_scores: np.ndarray,
    normal_metrics: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    base_loss = _smooth_l1_np(normal_scores, val_target, val_mask)
    result: dict[str, Any] = {"branch": "StructuredVisibility17+Geometry", "base_accuracy": normal_metrics["accuracy"], "base_utility_loss": base_loss, "joints": {}}
    for joint_index, joint_name in enumerate(H36M17_LINKS):
        shuffled = val_structured.copy()
        shuffled[:, :, joint_index] = shuffled[rng.permutation(len(shuffled)), :, joint_index]
        inputs = _build_inputs("StructuredVisibility17+Geometry", val_geometry, val_structured.mean(axis=-1), shuffled, val_pose, val_labels)
        scores = _predict(model, inputs, device)
        actions = _select(scores, val_ids, val_mask)
        preds = _terminal(val_logp, val_ids, val_mask, actions)
        metrics = _metrics("permutation", val_rows, val_labels, preds, actions, False)
        result["joints"][joint_name] = {
            "joint_index": joint_index,
            "accuracy": metrics["accuracy"],
            "accuracy_drop": float(normal_metrics["accuracy"] - metrics["accuracy"]),
            "utility_loss": _smooth_l1_np(scores, val_target, val_mask),
            "utility_loss_increase": float(_smooth_l1_np(scores, val_target, val_mask) - base_loss),
        }
    result["ranking"] = {name: values for name, values in sorted(result["joints"].items(), key=lambda item: -float(item[1]["accuracy_drop"]))}
    return result


def _high_occlusion(
    scalar: np.ndarray,
    methods: Mapping[str, Mapping[str, Any]],
    labels: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    subset = scalar[:, 0] < np.quantile(scalar[:, 0], 1.0 / 3.0)
    output: dict[str, Any] = {"definition": "strict bottom tertile of current frame-0 scalar visibility", "count": int(subset.sum()), "methods": {}}
    for name, metric in methods.items():
        target = labels[subset]
        pred = predictions[name][subset]
        output["methods"][name] = classification(target, pred)
    return output


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    scalar = methods["ScalarVisibility+Geometry"]
    structured = methods["StructuredVisibility17+Geometry"]
    pose_structured = methods["CurrentPose+StructuredVisibility17+Geometry"]
    ultimate = methods["GTAction+CurrentPose+StructuredVisibility17+Geometry"]
    oracle = methods["GT-TrueLogP Oracle"]
    structured_gain = 100.0 * (structured["accuracy"] - scalar["accuracy"])
    pose_gain = 100.0 * (pose_structured["accuracy"] - structured["accuracy"])
    action_gain = 100.0 * (ultimate["accuracy"] - pose_structured["accuracy"])
    residual = 100.0 * (oracle["accuracy"] - ultimate["accuracy"])
    if ultimate["accuracy"] < 0.60 or residual >= 15.0:
        conclusion = "D. KILL PRE-ACTION OBSERVABILITY SELECTOR FAMILY"
        decision = "KILL entire pre-action observability selector family"
    elif structured_gain >= 3.0 and pose_gain >= 3.0:
        conclusion = "A. STRUCTURED VISIBILITY IS A STRONG MISSING FACTOR" if structured_gain >= pose_gain else "B. CURRENT POSE + STRUCTURED VISIBILITY IS A STRONG MISSING FACTOR"
        decision = "KEEP structured observability"
    elif structured_gain >= 3.0:
        conclusion, decision = "A. STRUCTURED VISIBILITY IS A STRONG MISSING FACTOR", "KEEP structured observability"
    elif pose_gain >= 3.0:
        conclusion, decision = "B. CURRENT POSE + STRUCTURED VISIBILITY IS A STRONG MISSING FACTOR", "KEEP structured observability"
    else:
        conclusion, decision = "C. STRUCTURED OBSERVABILITY HELPS BUT REMAINS TOO WEAK FOR A MAIN ROUTE", "KILL entire pre-action observability selector family"
    lines = [
        "# Structured Observability Privileged Audit — Final Chance", "",
        "Experiment:", "final privileged structured observability audit", "",
        "Protocol:", "pre-action/frame-0 full-view terminal recognition", "",
        "Train:", "Policy Train", "", "Val:", "Moving Val", "", "Policy Test:", "false", "",
        "Action set:", "Stay + Stage-A legal candidates", "",
        "Recognizer:", "frozen ST-GCN encoder + frozen shared head", "",
        "Terminal HAR:", "selected viewpoint real 30-frame skeleton", "",
        "Structured visibility:", "exact privileged frame-0 H36M17 per-joint scene visibility", "",
        "Current pose:", "current viewpoint frame-0 H36M17 only", "",
        "Future motion input:", "false", "", "Candidate skeleton input:", "false", "",
        "Candidate RGB input:", "false", "", "Candidate recognizer output:", "training target / oracle only", "",
        f"Moving Val contexts: {result['population']['moving_val_contexts']}; Train contexts: {result['population']['train_contexts']}.",
        "", "## Moving-Val results", "",
        "| Method | Information | Accuracy | Macro-F1 | Move rate |", "|---|---|---:|---:|---:|",
    ]
    info = result["method_information"]
    for name in result["method_order"]:
        metric = methods[name]
        lines.append(f"| {name} | {info[name]} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend([
        "", "## Core gains and residual", "",
        f"- StructuredGain (V17+G - Vscalar+G): {structured_gain:+.3f} pp Accuracy",
        f"- PoseGain (P0+V17+G - V17+G): {pose_gain:+.3f} pp Accuracy",
        f"- ActionResidualGain (Y+P0+V17+G - P0+V17+G): {action_gain:+.3f} pp Accuracy",
        f"- Final ResidualGap (GT-TrueLogP Oracle - Y+P0+V17+G): {residual:+.3f} pp Accuracy",
        "", "## Scalar-equivalence sanity check", "",
        f"max_abs_difference={result['visibility_consistency_audit']['max_abs_difference']:.9g}; mean_abs_difference={result['visibility_consistency_audit']['mean_abs_difference']:.9g}.",
        "", "## Ranking diagnostics", "",
        "| Branch | Candidate Spearman | Within-context Spearman | Top-1 overlap | Top-3 overlap |", "|---|---:|---:|---:|---:|",
    ])
    for name, metric in result["ranking_metrics"].items():
        lines.append(f"| {name} | {metric.get('candidate_level_spearman_candidate_only', metric.get('candidate_level_spearman', 0.0)):.6f} | {metric['within_context_spearman_mean']:.6f} | {metric.get('gt_true_logp_oracle_top1_overlap', metric.get('gt_utility_oracle_top1_overlap', 0.0)):.6f} | {metric.get('gt_true_logp_oracle_top3_overlap', metric.get('gt_utility_oracle_top3_overlap', 0.0)):.6f} |")
    lines.extend(["", "## High-occlusion subset", "", f"Contexts: {result['occlusion_stratified_metrics']['count']}.", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name, metric in result["occlusion_stratified_metrics"]["methods"].items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    top_joints = list(result["joint_importance"].get("ranking", {}).items())[:5]
    top_joint_text = ", ".join(f"{name} ({values['accuracy_drop']:+.6f})" for name, values in top_joints) or "not available"
    gain_rows = result["per_class_decomposition"]
    top_classes = sorted(gain_rows.items(), key=lambda item: -float(item[1]["structured_gain_f1"]))[:3]
    top_class_text = ", ".join(f"{name} ({values['structured_gain_f1']:+.6f})" for name, values in top_classes) or "not available"
    lines.extend([
        "", "## Per-joint / per-class diagnostics", "", f"Largest structured-visibility permutation effects: {top_joint_text}.", f"Largest per-class structured F1 gains: {top_class_text}.",
        "", "## Main conclusion", "", f"**{conclusion}**", "", f"Decision: **{decision}**.",
        "The structured branches are privileged diagnostics, not deployable policies. The gap quantities are descriptive and not additive causal effects.",
        "If the kill rule is met, do not continue structured visibility predictors, per-limb visibility, RGB visibility networks, body-part heatmap predictors, larger vision encoders, or task-aware visibility fusion.",
        "", "```text", "test_used=false", "training_new_model_used=true", "train_used_only_for_frozen_stgcn_masking_importance=false", "gt_action_used_for_val_oracle_only=true", "gt_future_visibility_used_for_oracle_only=true", "deployable=false", "```",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root, scene_root, output_root = args.data_root.resolve(), args.scene_root.resolve(), args.output_root.resolve()
    train_rows, val_rows = load_rows(data_root)
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    shared_head_path = data_root / SHARED_HEAD_REL
    shared_head = SharedHead().to(device)
    shared_head.load_state_dict(torch.load(shared_head_path, map_location=device, weights_only=False)["state_dict"])
    shared_head.eval()
    train_logp, train_ids, train_mask = _load_options(data_root, train_rows, "train", shared_head, device)
    val_logp, val_ids, val_mask = _load_options(data_root, val_rows, "val", shared_head, device)
    train_vis, train_vis_meta = _build_structured_cache(data_root, scene_root, train_rows, "train")
    val_vis, val_vis_meta = _build_structured_cache(data_root, scene_root, val_rows, "val")
    _validate_action_arrays(train_rows, train_vis["ids"], train_vis["mask"], "train structured visibility")
    _validate_action_arrays(val_rows, val_vis["ids"], val_vis["mask"], "val structured visibility")
    if not np.array_equal(train_ids, train_vis["ids"]) or not np.array_equal(val_ids, val_vis["ids"]):
        raise ValueError("recognizer and structured visibility action IDs differ")
    train_scalar, val_scalar = _load_scalar(data_root, train_rows, "train"), _load_scalar(data_root, val_rows, "val")
    train_structured, val_structured = train_vis["visibility"], val_vis["visibility"]
    equivalence_values = np.concatenate(
        [(train_structured[train_vis["mask"]].mean(axis=-1) - train_scalar[train_vis["mask"]]), (val_structured[val_vis["mask"]].mean(axis=-1) - val_scalar[val_vis["mask"]])]
    )
    consistency = {"max_abs_difference": float(np.max(np.abs(equivalence_values))), "mean_abs_difference": float(np.mean(np.abs(equivalence_values))), "tolerance": 1e-6, "pass": bool(np.max(np.abs(equivalence_values)) <= 1e-6)}
    if not consistency["pass"]:
        raise RuntimeError(f"structured/scalar visibility definition drift: {consistency}")
    train_pose, val_pose = _load_pose(data_root, train_rows, "train"), _load_pose(data_root, val_rows, "val")
    stats = json.loads((data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8"))
    train_geometry, _, _ = _option_geometry(train_rows, stats)
    val_geometry, _, _ = _option_geometry(val_rows, stats)
    train_target, val_target = _target_utility(train_logp, train_labels), _target_utility(val_logp, val_labels)
    output_root.mkdir(parents=True, exist_ok=True)
    methods: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    actions_by_method: dict[str, list[int]] = {}
    stay_actions = [int(row["current_viewpoint_id"]) for row in val_rows]
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice(_actions(row))) for row in val_rows]
    for name, actions in (("Stay", stay_actions), ("Random", random_actions)):
        actions_by_method[name] = actions
        predictions[name] = _terminal(val_logp, val_ids, val_mask, actions)
        methods[name] = _metrics(name, val_rows, val_labels, predictions[name], actions, True)
    oracle_actions = _select(val_target, val_ids, val_mask)
    ranking: dict[str, Any] = {}
    historical_geometry, historical_geometry_ranking = _load_historical_geometry()
    geometry_checkpoint = data_root / HISTORICAL_GEOMETRY_CHECKPOINT
    if geometry_checkpoint.is_file():
        geometry_model = UtilityMLP(train_geometry.shape[-1]).to(device)
        geometry_payload = torch.load(geometry_checkpoint, map_location=device, weights_only=False)
        geometry_model.load_state_dict(geometry_payload["state_dict"])
        geometry_scores = _predict(geometry_model.eval(), val_geometry, device)
        geometry_actions = _select(geometry_scores, val_ids, val_mask)
        predictions["GeometryOnly Utility"] = _terminal(val_logp, val_ids, val_mask, geometry_actions)
        methods["GeometryOnly Utility"] = _metrics("GeometryOnly Utility", val_rows, val_labels, predictions["GeometryOnly Utility"], geometry_actions, True)
        if abs(methods["GeometryOnly Utility"]["accuracy"] - historical_geometry["accuracy"]) > 0.005:
            raise RuntimeError("historical GeometryOnly checkpoint does not reproduce its committed result")
        actions_by_method["GeometryOnly Utility"] = geometry_actions
        ranking["GeometryOnly Utility"] = ranking_metrics("GeometryOnly Utility", val_rows, geometry_scores, val_target, val_ids, val_mask, oracle_actions)
    else:
        methods["GeometryOnly Utility"] = historical_geometry
        actions_by_method["GeometryOnly Utility"] = []
        predictions["GeometryOnly Utility"] = predictions["Stay"].copy()
    scalar_inputs_train = _build_inputs(BRANCHES[0], train_geometry, train_scalar, train_structured, train_pose, train_labels)
    scalar_inputs_val = _build_inputs(BRANCHES[0], val_geometry, val_scalar, val_structured, val_pose, val_labels)
    models: dict[str, UtilityMLP] = {}
    training: dict[str, Any] = {}
    scores_by_branch: dict[str, np.ndarray] = {}
    branch_input_pairs = {
        BRANCHES[0]: (scalar_inputs_train, scalar_inputs_val),
        BRANCHES[1]: (_build_inputs(BRANCHES[1], train_geometry, train_scalar, train_structured, train_pose, train_labels), _build_inputs(BRANCHES[1], val_geometry, val_scalar, val_structured, val_pose, val_labels)),
        BRANCHES[2]: (_build_inputs(BRANCHES[2], train_geometry, train_scalar, train_structured, train_pose, train_labels), _build_inputs(BRANCHES[2], val_geometry, val_scalar, val_structured, val_pose, val_labels)),
        BRANCHES[3]: (_build_inputs(BRANCHES[3], train_geometry, train_scalar, train_structured, train_pose, train_labels), _build_inputs(BRANCHES[3], val_geometry, val_scalar, val_structured, val_pose, val_labels)),
        BRANCHES[4]: (_build_inputs(BRANCHES[4], train_geometry, train_scalar, train_structured, train_pose, train_labels), _build_inputs(BRANCHES[4], val_geometry, val_scalar, val_structured, val_pose, val_labels)),
    }
    if "GeometryOnly Utility" not in ranking:
        ranking["GeometryOnly Utility"] = historical_geometry_ranking
    for branch in BRANCHES:
        train_inputs, val_inputs = branch_input_pairs[branch]
        model, summary = _train_branch(branch, train_inputs, val_inputs, train_target, val_target, train_mask, val_mask, train_rows, data_root, output_root, device)
        models[branch], training[branch] = model, summary
        scores = _predict(model, val_inputs, device)
        scores_by_branch[branch] = scores
        actions = _select(scores, val_ids, val_mask)
        actions_by_method[branch] = actions
        predictions[branch] = _terminal(val_logp, val_ids, val_mask, actions)
        methods[branch] = _metrics(branch, val_rows, val_labels, predictions[branch], actions, False)
        ranking[branch] = ranking_metrics(branch, val_rows, scores, val_target, val_ids, val_mask, oracle_actions)
    if abs(methods[BRANCHES[0]]["accuracy"] - 0.5201388889) > 0.005:
        raise RuntimeError(f"ScalarVisibility+Geometry failed baseline reproduction: {methods[BRANCHES[0]]}")
    actions_by_method["GT-TrueLogP Oracle"] = oracle_actions
    predictions["GT-TrueLogP Oracle"] = _terminal(val_logp, val_ids, val_mask, oracle_actions)
    methods["GT-TrueLogP Oracle"] = _metrics("GT-TrueLogP Oracle", val_rows, val_labels, predictions["GT-TrueLogP Oracle"], oracle_actions, False)
    ranking["GT-TrueLogP Oracle"] = ranking_metrics("GT-TrueLogP Oracle", val_rows, val_target, val_target, val_ids, val_mask, oracle_actions)
    method_order = ("Stay", "Random", "GeometryOnly Utility", *BRANCHES, "GT-TrueLogP Oracle")
    per_class = {name: metric["per_class"] for name, metric in methods.items()}
    per_class_decomposition = {
        action: {
            "structured_gain_f1": float(methods[BRANCHES[1]]["per_class"][action]["f1"] - methods[BRANCHES[0]]["per_class"][action]["f1"]),
            "pose_gain_f1": float(methods[BRANCHES[3]]["per_class"][action]["f1"] - methods[BRANCHES[1]]["per_class"][action]["f1"]),
        }
        for action in LABELS
    }
    joint_importance = _joint_importance(models[BRANCHES[1]], branch_input_pairs[BRANCHES[1]][1], val_geometry, val_structured, val_pose, val_labels, val_target, val_logp, val_ids, val_mask, val_rows, scores_by_branch[BRANCHES[1]], methods[BRANCHES[1]], device)
    occlusion_names = ("Stay", "Random", "GeometryOnly Utility", *BRANCHES, "GT-TrueLogP Oracle")
    occlusion_predictions = {name: predictions[name] for name in occlusion_names}
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_STRUCTURED_OBSERVABILITY_FINAL_AUDIT",
        "status": "COMPLETED",
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(row['record_id']) for row in train_rows}), "moving_val_records": len({str(row['record_id']) for row in val_rows})},
        "labels": list(LABELS),
        "method_order": list(method_order),
        "methods": methods,
        "method_information": {"Stay": "—", "Random": "—", "GeometryOnly Utility": "G", "ScalarVisibility+Geometry": "Vscalar+G", "StructuredVisibility17+Geometry": "V17+G", "CurrentPose+Geometry": "P0+G", "CurrentPose+StructuredVisibility17+Geometry": "P0+V17+G", "GTAction+CurrentPose+StructuredVisibility17+Geometry": "Y+P0+V17+G", "GT-TrueLogP Oracle": "exact candidate utility"},
        "ranking_metrics": ranking,
        "training": training,
        "target_audit": {"train_structured": train_vis_meta, "val_structured": val_vis_meta, "shared_head_sha256": _sha256(shared_head_path), "action_set": "Stay/current + Stage-A legal candidate_pool", "target": "GT-TrueLogP", "recognizer": "frozen ST-GCN encoder + frozen shared head", "candidate_skeleton_input": False, "candidate_recognizer_output_as_input": False, "test_used": False},
        "visibility_consistency_audit": consistency,
        "per_class_decomposition": per_class_decomposition,
        "occlusion_stratified_metrics": _high_occlusion(val_scalar, methods, val_labels, occlusion_predictions),
        "joint_importance": joint_importance,
        "flags": {"test_used": False, "training_new_model_used": True, "frozen_stgcn_modified": False, "gt_action_used_for_val_oracle_only": True, "gt_future_visibility_used_for_oracle_only": True, "future_candidate_rgb_used": False, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "deployable": False},
        "leakage_audit": {"ScalarVisibility+Geometry": {"uses_gt_action_input": False, "uses_current_pose": False, "uses_real_candidate_visibility": True, "uses_candidate_skeleton_input": False, "uses_candidate_rgb_input": False, "uses_candidate_recognizer_input": False, "deployable": False, "test_used": False}, "StructuredVisibility17+Geometry": {"uses_gt_action_input": False, "uses_current_pose": False, "uses_real_candidate_visibility": True, "uses_candidate_skeleton_input": False, "uses_candidate_rgb_input": False, "uses_candidate_recognizer_input": False, "deployable": False, "test_used": False}, "CurrentPose+Geometry": {"uses_gt_action_input": False, "uses_current_pose": True, "uses_real_candidate_visibility": False, "uses_candidate_skeleton_input": False, "uses_candidate_rgb_input": False, "uses_candidate_recognizer_input": False, "deployable": False, "test_used": False}, "CurrentPose+StructuredVisibility17+Geometry": {"uses_gt_action_input": False, "uses_current_pose": True, "uses_real_candidate_visibility": True, "uses_candidate_skeleton_input": False, "uses_candidate_rgb_input": False, "uses_candidate_recognizer_input": False, "deployable": False, "test_used": False}, "GTAction+CurrentPose+StructuredVisibility17+Geometry": {"uses_gt_action_input": True, "uses_current_pose": True, "uses_real_candidate_visibility": True, "uses_candidate_skeleton_input": False, "uses_candidate_rgb_input": False, "uses_candidate_recognizer_input": False, "deployable": False, "test_used": False}},
        "runtime": {"device": str(device), "seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "structured_cache_root": str((data_root / STRUCTURED_REL).resolve()), "output_root": str(output_root)},
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"branches": list(BRANCHES), "architecture": "Linear(input_dim,256)-GELU-Linear(256,256)-GELU-Linear(256,1)", "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "seed": SEED, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "listwise_weight": LISTWISE_WEIGHT, "listwise_tau": LISTWISE_TAU, "protocol": "pre-action/frame-0 full-view; Stay + Stage-A legal candidates", "test_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "per_branch_metrics.json").write_text(json.dumps({name: methods[name] for name in method_order}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "ranking_metrics.json").write_text(json.dumps(ranking, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "joint_importance.json").write_text(json.dumps(joint_importance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "occlusion_stratified_metrics.json").write_text(json.dumps(result["occlusion_stratified_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "visibility_consistency_audit.json").write_text(json.dumps(consistency, indent=2) + "\n", encoding="utf-8")
    (output_root / "target_audit.json").write_text(json.dumps(result["target_audit"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "leakage_audit.json").write_text(json.dumps(result["leakage_audit"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "training_summary.json").write_text(json.dumps(training, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "population": result["population"], "methods": {name: {"accuracy": result["methods"][name]["accuracy"], "macro_f1": result["methods"][name]["macro_f1"]} for name in result["method_order"]}}, indent=2), flush=True)
if __name__ == "__main__":
    main()
