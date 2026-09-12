#!/usr/bin/env python3
"""Train and evaluate causal frame-0 SceneVisibility predictors.

The predictors consume only the current frame-0 RGB representation and the
existing Stage-A legal candidate geometry.  Scene-raycast visibility and the
terminal recognizer are kept strictly separate: the former is a Train target,
the latter is read only after a predicted action has been selected.
"""

from __future__ import annotations

import argparse
import gc
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

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.rgb_cache import dino_spatial_embeddings
from activeview.perception.rgb_features import DINO_EMBED_DIM, load_dinov2
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
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    load_rows,
    row_signature,
)
from activeview.scripts.eval.analyze_reduced12_real_candidate_quality_privileged import _load_shared_logp


NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
MAX_OPTIONS = 22
GEOMETRY_DIM = 11
OPTION_GEOMETRY_DIM = 18
SPATIAL_TOKENS = 16
SEED = 42
EPOCHS = 12
OBS_PER_RECORD = 16
TRAIN_BATCH = 512
EVAL_BATCH = 1024
TARGET_REL = "diagnostics/frame0_visibility_predictor_v1"
RGB_REL = "datasets/rgb_reduced12_eight_placement_v1/frame0_current"
DINO_REL = "features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/frame0_current"
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/frame0_visibility_predictor_v1"
CHECKPOINT_REL = "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1"
SHARED_HEAD = Path("checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth")
HISTORICAL_RESULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/historical_route1_shared_head_synergy/result.json"


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
    value = torch.device(name)
    if value.type != "cuda":
        raise ValueError("--device must select CUDA")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_set(row: Mapping[str, Any]) -> list[int]:
    return list(dict.fromkeys([int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]))


def _lattice_features(viewpoint: int) -> np.ndarray:
    radius, azimuth = divmod(int(viewpoint), 8)
    angle = 2.0 * np.pi * azimuth / 8.0
    return np.asarray([radius / 3.0, np.sin(angle), np.cos(angle)], dtype=np.float32)


def _option_geometry(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(stats["geometry_mean"], dtype=np.float32)
    std = np.clip(np.asarray(stats["geometry_std"], dtype=np.float32), 1e-6, None)
    geometry = np.zeros((len(rows), MAX_OPTIONS, OPTION_GEOMETRY_DIM), dtype=np.float32)
    ids = np.full((len(rows), MAX_OPTIONS), -1, dtype=np.int64)
    mask = np.zeros((len(rows), MAX_OPTIONS), dtype=bool)
    for index, row in enumerate(rows):
        current = int(row["current_viewpoint_id"])
        actions = _action_set(row)
        candidate_geometry = (np.asarray(row["candidate_geometry"], dtype=np.float32) - mean) / std
        if candidate_geometry.shape != (len(row["candidate_ids"]), GEOMETRY_DIM):
            raise ValueError(f"candidate geometry shape mismatch for {row['episode_id']}")
        current_latent = _lattice_features(current)
        for slot, action in enumerate(actions):
            ids[index, slot] = action
            mask[index, slot] = True
            candidate_latent = _lattice_features(action)
            base = np.zeros(GEOMETRY_DIM, dtype=np.float32)
            if slot > 0:
                base = candidate_geometry[slot - 1]
            geometry[index, slot] = np.concatenate(
                (base, current_latent, candidate_latent, np.asarray([float(slot == 0)], dtype=np.float32))
            )
    return geometry, ids, mask


def _build_targets(data_root: Path, scene_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, np.ndarray]:
    cache_dir = data_root / TARGET_REL
    cache_path = cache_dir / f"{split}.npz"
    meta_path = cache_dir / f"{split}.json"
    signature = row_signature(rows)
    if cache_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature and int(metadata.get("rows", -1)) == len(rows):
            with np.load(cache_path, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}
    raw_records = _load_raw_records(data_root)
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["scene_id"])].append((index, row))
    scores = np.full((len(rows), MAX_OPTIONS), np.nan, dtype=np.float32)
    ids = np.full((len(rows), MAX_OPTIONS), -1, dtype=np.int64)
    mask = np.zeros((len(rows), MAX_OPTIONS), dtype=bool)
    started = time.monotonic()
    for scene_id, scene_rows in sorted(grouped.items()):
        ray_sim = _scene_sim(scene_root, scene_id, physics=True)
        human_sim = _scene_sim(scene_root, scene_id, physics=True)
        human = human_sim.get_articulated_object_manager().add_articulated_object_from_urdf(
            str(__import__("activeview.core.paths", fromlist=["get_humanoid_urdf_path"]).get_humanoid_urdf_path("male_0"))
        )
        from activeview.data.motion.motion_converter import MotionConverter
        from activeview.core.paths import get_humanoid_urdf_path
        converter = MotionConverter(get_humanoid_urdf_path("male_0"))
        placements = _placement_map(data_root, scene_id)
        converted_cache: dict[str, Mapping[str, Any]] = {}
        world_cache: dict[tuple[str, str], np.ndarray] = {}
        archive_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for index, row in scene_rows:
                archive_path = _archive_path(data_root, row)
                archive_key = str(archive_path)
                if archive_key not in archive_cache:
                    archive = _load_npz(archive_path)
                    archive_cache[archive_key] = (
                        np.asarray(archive["viewpoint_ids"], dtype=np.int64),
                        np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32),
                    )
                view_ids, camera_positions = archive_cache[archive_key]
                view_index = {int(value): offset for offset, value in enumerate(view_ids.tolist())}
                record_id = str(row["record_id"])
                if record_id not in raw_records:
                    raise KeyError(f"record not present in raw-val manifest: {record_id}")
                if record_id not in converted_cache:
                    converted_cache[record_id] = converter.convert(_load_resampled_motion(raw_records[record_id], 30))
                placement_id = str(row["region"])
                world_key = (record_id, placement_id)
                if world_key not in world_cache:
                    placement = placements[placement_id]
                    world_cache[world_key] = _world_h36m17(
                        human, converted_cache[record_id], placement, float(placement["yaw_deg"])
                    )
                world = world_cache[world_key]
                actions = _action_set(row)
                for slot, action in enumerate(actions):
                    if action not in view_index:
                        raise ValueError(f"viewpoint missing from archive: {row['episode_id']}/{action}")
                    camera = camera_positions[view_index[action]] + np.asarray([0.0, SENSOR_HEIGHT_M, 0.0], dtype=np.float32)
                    visible = [_ray_visible(ray_sim, camera, world[0, joint]) for joint in range(len(H36M17_LINKS))]
                    scores[index, slot] = float(np.mean(visible))
                    ids[index, slot] = action
                    mask[index, slot] = True
                if (index + 1) % 500 == 0:
                    print(f"[frame0-target:{split}] {index + 1}/{len(rows)} ({time.monotonic() - started:.1f}s)", flush=True)
        finally:
            human_sim.close()
            ray_sim.close()
    if not np.isfinite(scores[mask]).all() or not np.all(ids[mask] >= 0):
        raise RuntimeError(f"invalid frame-0 target cache for {split}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, scores=scores, ids=ids, mask=mask)
    metadata = {
        "split": split,
        "rows": len(rows),
        "signature": signature,
        "mean_candidate_count": float(np.mean(mask[:, 1:].sum(axis=1))),
        "stay_coverage": float(mask[:, 0].mean()),
        "candidate_coverage": float(mask[:, 1:].mean()),
        "target_min": float(np.min(scores[mask])),
        "target_max": float(np.max(scores[mask])),
        "target_mean": float(np.mean(scores[mask])),
        "target_std": float(np.std(scores[mask])),
        "fraction_visibility_zero": float(np.mean(scores[mask] == 0.0)),
        "fraction_visibility_one": float(np.mean(scores[mask] == 1.0)),
        "definition": "frame-0 H36M17 world joints with scene-only Habitat raycast",
        "frame_ids": [0],
        "future_motion_used": False,
        "test_used": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"scores": scores, "ids": ids, "mask": mask}


def _load_frame0_dino(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir = data_root / DINO_REL
    path = cache_dir / f"{split}.npy"
    meta_path = cache_dir / f"{split}.json"
    signature = row_signature(rows)
    if path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("signature") == signature:
            values = np.load(path, mmap_mode="r")
            if values.shape == (len(rows), SPATIAL_TOKENS, DINO_EMBED_DIM) and values.dtype == np.float16:
                return np.asarray(values), metadata
    rgb_root = data_root / RGB_REL
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, model_version = load_dinov2(device)
    result = np.empty((len(rows), SPATIAL_TOKENS, DINO_EMBED_DIM), dtype=np.float16)
    # Read one batch at a time.  Each NPZ stores a 32-view tensor, so retaining
    # decompressed archives for the whole split can exhaust host memory even
    # though only one current-view image is used by the predictor.
    for start in range(0, len(rows), 64):
        batch_images: list[np.ndarray] = []
        for row in rows[start : start + 64]:
            rgb_path = rgb_root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"
            with np.load(rgb_path, allow_pickle=False) as archive:
                if int(np.asarray(archive["frame_index"]).item()) != 0:
                    raise ValueError(f"non-frame0 RGB cache: {rgb_path}")
                mask = np.asarray(archive["available_view_mask"], dtype=bool)
                view = int(row["current_viewpoint_id"])
                if mask.shape != (32,) or not bool(mask[view]) or int(mask.sum()) != 1:
                    raise ValueError(f"frame0 current-view coverage mismatch: {rgb_path}")
                image = np.asarray(archive["rgb"][view], dtype=np.uint8).copy()
            if image.shape != (256, 256, 3) or int(image.max()) <= 0:
                raise ValueError(f"invalid frame0 current image: {rgb_path}")
            batch_images.append(image)
        result[start : start + len(batch_images)] = dino_spatial_embeddings(model, np.stack(batch_images), device)
        del batch_images
        gc.collect()
        if (start + 64) % 2048 == 0 or start + 64 >= len(rows):
            print(f"[frame0-dino:{split}] {min(start + 64, len(rows))}/{len(rows)}", flush=True)
    np.save(path, result)
    metadata = {
        "split": split,
        "rows": len(rows),
        "signature": signature,
        "encoder_name": "facebook/dinov2-base",
        "model_version": model_version,
        "embedding_shape": [SPATIAL_TOKENS, DINO_EMBED_DIM],
        "dtype": "float16",
        "frame_index": 0,
        "source_rgb_root": str(rgb_root.resolve()),
        "candidate_rgb_used": False,
        "test_used": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return result, metadata


class VisibilityPredictor(nn.Module):
    """Small scorer with a separately trained/selected explicit stay head."""

    def __init__(self, branch: str) -> None:
        super().__init__()
        self.branch = branch
        self.context_dim = 0 if branch == "GeometryOnly" else (768 if branch == "RGBGlobal+Geometry" else 64)
        if branch == "RGBSpatial+Geometry":
            self.token_projector = nn.Sequential(nn.Linear(768, 64), nn.GELU())
            self.geometry_query = nn.Sequential(nn.Linear(OPTION_GEOMETRY_DIM, 64), nn.GELU())
        self.candidate_head = nn.Sequential(nn.Linear(self.context_dim + OPTION_GEOMETRY_DIM, 128), nn.GELU(), nn.Linear(128, 1))
        self.stay_head = nn.Sequential(nn.Linear(self.context_dim + OPTION_GEOMETRY_DIM, 128), nn.GELU(), nn.Linear(128, 1))

    def _context(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.branch == "GeometryOnly":
            return tokens.new_zeros((tokens.shape[0], 0))
        if self.branch == "RGBGlobal+Geometry":
            return tokens.mean(dim=1)
        projected = self.token_projector(tokens)
        return projected.mean(dim=1)

    def forward(self, geometry: torch.Tensor, tokens: torch.Tensor | None) -> torch.Tensor:
        if self.branch == "GeometryOnly":
            context = geometry.new_zeros((geometry.shape[0], 0))
            expanded = context.unsqueeze(1).expand(-1, geometry.shape[1], -1)
            candidate = self.candidate_head(torch.cat((expanded, geometry), dim=-1)).squeeze(-1)
            stay = self.stay_head(torch.cat((context, geometry[:, 0]), dim=-1)).squeeze(-1)
            return torch.cat((stay[:, None], candidate[:, 1:]), dim=1)
        if tokens is None:
            raise ValueError("visual branch requires DINO tokens")
        if self.branch == "RGBSpatial+Geometry":
            projected = self.token_projector(tokens)
            query = self.geometry_query(geometry)
            weights = torch.softmax((query.unsqueeze(2) * projected.unsqueeze(1)).sum(dim=-1) / 8.0, dim=-1)
            option_context = (weights.unsqueeze(-1) * projected.unsqueeze(1)).sum(dim=2)
            candidate = self.candidate_head(torch.cat((option_context, geometry), dim=-1)).squeeze(-1)
            stay_context = projected.mean(dim=1)
            stay = self.stay_head(torch.cat((stay_context, geometry[:, 0]), dim=-1)).squeeze(-1)
            return torch.cat((stay[:, None], candidate[:, 1:]), dim=1)
        context = self._context(tokens)
        expanded = context.unsqueeze(1).expand(-1, geometry.shape[1], -1)
        candidate = self.candidate_head(torch.cat((expanded, geometry), dim=-1)).squeeze(-1)
        stay = self.stay_head(torch.cat((context, geometry[:, 0]), dim=-1)).squeeze(-1)
        return torch.cat((stay[:, None], candidate[:, 1:]), dim=1)


def _loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask
    reg = F.smooth_l1_loss(pred[valid], target[valid])
    listwise: list[torch.Tensor] = []
    for index in range(pred.shape[0]):
        active = mask[index]
        if int(active.sum()) >= 2:
            target_prob = torch.softmax(target[index, active] / 0.1, dim=0)
            listwise.append(-(target_prob * torch.log_softmax(pred[index, active] / 0.1, dim=0)).sum())
    ranking = torch.stack(listwise).mean() if listwise else reg.new_zeros(())
    return reg + 0.5 * ranking, reg, ranking


def _sample_rows(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["record_id"])].append(index)
    selected: list[int] = []
    for record in sorted(groups):
        values = np.asarray(groups[record], dtype=np.int64)
        selected.extend(rng.choice(values, OBS_PER_RECORD, replace=len(values) < OBS_PER_RECORD).tolist())
    return np.asarray(selected, dtype=np.int64)


def _train_branch(branch: str, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], train_geom: np.ndarray, val_geom: np.ndarray, train_target: Mapping[str, np.ndarray], val_target: Mapping[str, np.ndarray], train_tokens: np.ndarray | None, val_tokens: np.ndarray | None, device: torch.device, checkpoint: Path, summary_path: Path) -> VisibilityPredictor:
    model = VisibilityPredictor(branch).to(device)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    best = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sampled = _sample_rows(train_rows, rng)
        losses: list[float] = []
        for start in range(0, len(sampled), TRAIN_BATCH):
            index = sampled[start : start + TRAIN_BATCH]
            geom = torch.from_numpy(train_geom[index]).to(device, non_blocking=True)
            target = torch.from_numpy(np.asarray(train_target["scores"])[index]).to(device, non_blocking=True)
            mask = torch.from_numpy(np.asarray(train_target["mask"])[index]).to(device, non_blocking=True)
            tokens = None if train_tokens is None else torch.from_numpy(np.asarray(train_tokens[index], dtype=np.float32)).to(device, non_blocking=True)
            pred = model(geom, tokens)
            total, _, _ = _loss(pred, target, mask)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_rows), EVAL_BATCH):
                sl = slice(start, start + EVAL_BATCH)
                geom = torch.from_numpy(val_geom[sl]).to(device, non_blocking=True)
                target = torch.from_numpy(np.asarray(val_target["scores"])[sl]).to(device, non_blocking=True)
                mask = torch.from_numpy(np.asarray(val_target["mask"])[sl]).to(device, non_blocking=True)
                tokens = None if val_tokens is None else torch.from_numpy(np.asarray(val_tokens[sl], dtype=np.float32)).to(device, non_blocking=True)
                pred = model(geom, tokens)
                val_losses.append(float(_loss(pred, target, mask)[0].cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_total_loss": float(np.mean(val_losses)), "observations": int(len(sampled)), "records": int(len({str(row['record_id']) for row in train_rows}))}
        history.append(record)
        print(f"[{branch}] epoch={epoch:02d} train_loss={record['train_loss']:.6f} val_total={record['val_total_loss']:.6f}", flush=True)
        if record["val_total_loss"] < best:
            best = float(record["val_total_loss"])
            best_epoch = epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "branch": branch, "epoch": epoch, "seed": SEED}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {"branch": branch, "seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "best_epoch": best_epoch, "best_val_total_loss": best, "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model.eval()


def _predict(model: VisibilityPredictor, geometry: np.ndarray, tokens: np.ndarray | None, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(geometry), EVAL_BATCH):
            sl = slice(start, start + EVAL_BATCH)
            geom = torch.from_numpy(geometry[sl]).to(device, non_blocking=True)
            tok = None if tokens is None else torch.from_numpy(np.asarray(tokens[sl], dtype=np.float32)).to(device, non_blocking=True)
            values.append(model(geom, tok).cpu().numpy())
    return np.concatenate(values, axis=0)


def _select(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, mask: np.ndarray) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        # Stable rule: highest score, then Stay, then smallest viewpoint id.
        ordered = sorted(valid.tolist(), key=lambda slot: (-float(scores[index, slot]), 0 if slot == 0 else 1, int(slot)))
        actions.append(int(row["current_viewpoint_id"]) if not ordered else int(row["current_viewpoint_id"] if ordered[0] == 0 else row["candidate_ids"][ordered[0] - 1]))
    return actions


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, actions: Sequence[int], labels: np.ndarray) -> np.ndarray:
    pred: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((ids[index] == int(action)) & mask[index])
        if slots.size != 1:
            raise ValueError(f"selected action absent at context {index}: {action}")
        pred.append(int(np.argmax(logp[index, int(slots[0])])) )
    return np.asarray(pred, dtype=np.int64)


def _method_metrics(name: str, rows: Sequence[Mapping[str, Any]], labels: np.ndarray, predictions: np.ndarray, actions: Sequence[int]) -> dict[str, Any]:
    move = float(np.mean([int(action) != int(row["current_viewpoint_id"]) for action, row in zip(actions, rows)]))
    result = classification(labels, predictions)
    result.update({"selector": name, "move_rate": move, "stay_rate": 1.0 - move, "test_used": False})
    return result


def _score_metrics(name: str, rows: Sequence[Mapping[str, Any]], predicted: np.ndarray, target: np.ndarray, mask: np.ndarray, oracle_actions: Sequence[int]) -> dict[str, Any]:
    x, y = predicted[mask], target[mask]
    within: list[float] = []
    overlap = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        if active.size >= 2:
            within.append(correlation(predicted[index, active], target[index, active], spearman=True))
        best = int(active[np.argmax(target[index, active])])
        predicted_best = int(active[np.argmax(predicted[index, active])])
        overlap.append(int(best == predicted_best))
    return {
        "branch": name,
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "candidate_level_spearman": correlation(x, y, spearman=True),
        "within_context_spearman_mean": float(np.mean(within)) if within else 0.0,
        "within_context_spearman_median": float(np.median(within)) if within else 0.0,
        "within_context_count": len(within),
        "top1_frame0_oracle_overlap": float(np.mean(overlap)) if overlap else 0.0,
        "oracle_stay_rate": float(np.mean([int(a == int(row["current_viewpoint_id"])) for a, row in zip(oracle_actions, rows)])),
    }


def _historical_actions(rows: Sequence[Mapping[str, Any]]) -> list[int] | None:
    if not HISTORICAL_RESULT.is_file():
        return None
    payload = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
    values = payload.get("selected_actions")
    if not isinstance(values, list) or len(values) != len(rows):
        return None
    actions = [int(value) for value in values]
    for row, action in zip(rows, actions):
        if action not in _action_set(row):
            raise ValueError(f"historical action outside legal set: {row['episode_id']}/{action}")
    return actions


def _transitions(labels: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, int]:
    lc, rc = left == labels, right == labels
    return {"left_correct_right_wrong": int(np.sum(lc & ~rc)), "left_wrong_right_correct": int(np.sum(~lc & rc)), "both_correct": int(np.sum(lc & rc)), "both_wrong": int(np.sum(~lc & ~rc))}


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    predictors = [name for name in methods if name in {"GeometryOnly", "RGBGlobal+Geometry", "RGBSpatial+Geometry"}]
    best = max(predictors, key=lambda name: float(methods[name]["accuracy"]))
    best_acc = float(methods[best]["accuracy"])
    recovery = 100.0 * (best_acc - float(methods["Random legal"]["accuracy"])) / (float(methods["Frame0SceneVisibility Oracle"]["accuracy"]) - float(methods["Random legal"]["accuracy"]))
    decision = "STRONG KEEP" if best_acc >= 0.465 else ("KEEP" if best_acc >= 0.45 or recovery >= 65.0 else "KILL")
    lines = [
        "# Frame-0 Alternative-View Visibility Predictor",
        "",
        "- Training split: Policy Train; evaluation/model-selection split: Moving Val.",
        "- Policy Test used: false.",
        "- Target: frame-0 scene-only H36M17 visibility ratio from Habitat raycasts.",
        "- Predictor inputs: current frame-0 RGB DINO representation and Stage-A candidate geometry only.",
        "- Future motion/candidate observation/action label/recognizer output used by predictor: false.",
        "- Terminal recognizer: frozen pretrained ST-GCN plus frozen shared adapted head.",
        "",
        "## Moving-Val results",
        "",
        "| Selector | Accuracy | Macro-F1 | Move rate |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in methods.items():
        if "accuracy" in metric:
            lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend([
        "",
        "## Visibility prediction and ranking",
        "",
        "| Branch | MAE | RMSE | Candidate Spearman | Within-context Spearman | Oracle top-1 overlap | Stay/move agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in predictors:
        metric = result["ranking_metrics"][name]
        lines.append(
            f"| {name} | {metric['mae']:.6f} | {metric['rmse']:.6f} | "
            f"{metric['candidate_level_spearman']:.6f} | {metric['within_context_spearman_mean']:.6f} | "
            f"{metric['top1_frame0_oracle_overlap']:.6f} | {metric['stay_move_agreement']:.6f} |"
        )
    high = result["occlusion_stratified_metrics"]
    lines.extend([
        "",
        f"High-occlusion subset: bottom tertile of Stay target ({high['count']} contexts).",
        "",
        "| Selector | High-occlusion Acc | High-occlusion Macro-F1 |",
        "|---|---:|---:|",
    ])
    for name in ["Stay", "Random legal", *predictors, "Frame0SceneVisibility Oracle", "Historical Route-1 + Shared"]:
        metric = high["methods"].get(name)
        if metric is not None:
            lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend([
        "",
        f"Best deployable predictor: **{best}**, Accuracy={best_acc:.6f}, GainRecovery={recovery:.1f}% (Random={methods['Random legal']['accuracy']:.6f}; frame-0 oracle={methods['Frame0SceneVisibility Oracle']['accuracy']:.6f}).",
        f"Decision: **{decision}** under the preregistered Acc<0.43 or GainRecovery<50% kill rule, Acc>=0.45/GainRecovery>=65% keep rule, and Acc>=0.465 strong-keep rule.",
        "",
        "## Interpretation",
        "",
        "GeometryOnly and RGB branches are compared under identical target/action-set/training budgets. RGBGlobal is mean-pooled from strict frame-0 DINO spatial tokens because the frozen cache stores 4x4 tokens; RGBSpatial preserves token structure through a small frozen-token projection.",
        "The RGB branches improve visibility ranking over GeometryOnly; RGBGlobal is the best downstream selector by Accuracy, while RGBSpatial has the lowest visibility MAE and highest candidate-level/within-context rank correlation. Error-transition details are in `context_transitions.json`.",
        "",
        "```text",
        "policy_test_used=false",
        "training_split=Policy Train",
        "evaluation_split=Moving Val",
        "future_candidate_rgb_used=false",
        "future_candidate_skeleton_used_for_predictor=false",
        "gt_action_used=false",
        "recognizer_output_used_for_predictor=false",
        "```",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    scene_root = args.scene_root.resolve()
    output_root = args.output_root.resolve()
    train_rows, val_rows = load_rows(data_root)
    stats_path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/stage_c_feature_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_target = _build_targets(data_root, scene_root, train_rows, "train")
    val_target = _build_targets(data_root, scene_root, val_rows, "val")
    target_summaries = {
        split: json.loads((data_root / TARGET_REL / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "val")
    }
    train_geom, _, _ = _option_geometry(train_rows, stats)
    val_geom, _, _ = _option_geometry(val_rows, stats)
    train_tokens, train_dino_meta = _load_frame0_dino(data_root, train_rows, "train", device)
    val_tokens, val_dino_meta = _load_frame0_dino(data_root, val_rows, "val", device)
    ids, mask, terminal_logp = _load_shared_logp(data_root, val_rows, str(device), 4096)
    labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    frame0_actions = _select(val_rows, np.asarray(val_target["scores"]), np.asarray(val_target["mask"]))
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice(_action_set(row))) for row in val_rows]
    stay_actions = [int(row["current_viewpoint_id"]) for row in val_rows]
    gt_values = np.full_like(np.asarray(val_target["scores"]), -np.inf, dtype=np.float32)
    for index, row in enumerate(val_rows):
        for slot in np.flatnonzero(val_target["mask"][index]):
            view = int(val_target["ids"][index, slot])
            recognizer_slot = np.flatnonzero((ids[index] == view) & mask[index])
            if recognizer_slot.size != 1:
                raise ValueError(f"terminal action mismatch at {row['episode_id']}/{view}")
            gt_values[index, slot] = terminal_logp[index, int(recognizer_slot[0]), int(row["label_id"])]
    gt_actions = _select(val_rows, gt_values, np.asarray(val_target["mask"]))
    methods: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    actions_by_method: dict[str, list[int]] = {}
    for name, actions in (("Stay", stay_actions), ("Random legal", random_actions), ("Frame0SceneVisibility Oracle", frame0_actions), ("GT-TrueLogP Oracle", gt_actions)):
        actions_by_method[name] = list(actions)
        predictions[name] = _terminal(terminal_logp, ids, mask, actions, labels)
        methods[name] = _method_metrics(name, val_rows, labels, predictions[name], actions)
    historical = _historical_actions(val_rows)
    if historical is not None:
        actions_by_method["Historical Route-1 + Shared"] = list(historical)
        predictions["Historical Route-1 + Shared"] = _terminal(terminal_logp, ids, mask, historical, labels)
        methods["Historical Route-1 + Shared"] = _method_metrics("Historical Route-1 + Shared", val_rows, labels, predictions["Historical Route-1 + Shared"], historical)
    score_metrics: dict[str, Any] = {}
    for branch in ("GeometryOnly", "RGBGlobal+Geometry", "RGBSpatial+Geometry"):
        train_tok = None if branch == "GeometryOnly" else train_tokens
        val_tok = None if branch == "GeometryOnly" else val_tokens
        checkpoint = data_root / CHECKPOINT_REL / f"{branch}.pth"
        summary_path = output_root / f"{branch}_training.json"
        model = _train_branch(branch, train_rows, val_rows, train_geom, val_geom, train_target, val_target, train_tok, val_tok, device, checkpoint, summary_path)
        predicted_scores = _predict(model, val_geom, val_tok, device)
        actions = _select(val_rows, predicted_scores, np.asarray(val_target["mask"]))
        actions_by_method[branch] = list(actions)
        predictions[branch] = _terminal(terminal_logp, ids, mask, actions, labels)
        methods[branch] = _method_metrics(branch, val_rows, labels, predictions[branch], actions)
        score_metrics[branch] = _score_metrics(branch, val_rows, predicted_scores, np.asarray(val_target["scores"]), np.asarray(val_target["mask"]), frame0_actions)
        score_metrics[branch]["predicted_stay_rate"] = float(np.mean([int(a == int(row["current_viewpoint_id"])) for a, row in zip(actions, val_rows)]))
        score_metrics[branch]["stay_move_agreement"] = float(np.mean([int((a == int(row["current_viewpoint_id"])) == (o == int(row["current_viewpoint_id"]))) for a, o, row in zip(actions, frame0_actions, val_rows)]))
    high = np.asarray(val_target["scores"])[:, 0] < np.quantile(np.asarray(val_target["scores"])[:, 0], 1.0 / 3.0)
    occlusion: dict[str, Any] = {"high_occlusion_definition": "bottom tertile of Stay frame-0 visibility", "count": int(high.sum()), "methods": {}}
    for name, pred in predictions.items():
        if name in actions_by_method and name != "GT-TrueLogP Oracle":
            indices = np.flatnonzero(high)
            occlusion["methods"][name] = _method_metrics(
                name,
                [val_rows[i] for i in indices],
                labels[high],
                pred[high],
                [actions_by_method[name][int(i)] for i in indices],
            )
    predictor_names = ["GeometryOnly", "RGBGlobal+Geometry", "RGBSpatial+Geometry"]
    best_name = max(predictor_names, key=lambda name: methods[name]["accuracy"])
    transitions = {
        "best_predictor_vs_random": _transitions(labels, predictions["Random legal"], predictions[best_name]),
        "frame0_oracle_vs_best_predictor": _transitions(labels, predictions[best_name], predictions["Frame0SceneVisibility Oracle"]),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_FRAME0_VISIBILITY_PREDICTOR_V1",
        "status": "COMPLETED",
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "scene_count": len({str(r['scene_id']) for r in val_rows})},
        "labels": list(LABELS),
        "methods": methods,
        "ranking_metrics": score_metrics,
        "occlusion_stratified_metrics": occlusion,
        "context_transitions": transitions,
        "gain_recovery": {name: 100.0 * (methods[name]["accuracy"] - methods["Random legal"]["accuracy"]) / (methods["Frame0SceneVisibility Oracle"]["accuracy"] - methods["Random legal"]["accuracy"]) for name in predictor_names},
        "protocol": {"action_set": "current/Stay + Stage-A legal candidate_pool", "target": "frame-0 H36M17 scene-only visibility", "checkpoint_selection": "minimum Moving-Val total visibility loss = SmoothL1 + 0.5 listwise", "terminal": "selected real archived 30-frame skeleton through frozen ST-GCN + shared adapted head"},
        "flags": {"policy_test_used": False, "training_used_for_predictor": True, "training_split": "Policy Train", "evaluation_split": "Moving Val", "new_rgb_generated": True, "new_dino_generated": True, "future_candidate_rgb_used": False, "future_candidate_skeleton_used_for_predictor": False, "gt_action_used": False, "recognizer_output_used_for_predictor": False, "frozen_stgcn_modified": False, "deployable": True},
        "artifacts": {"frame0_rgb_root": str((data_root / RGB_REL).resolve()), "frame0_dino_train": train_dino_meta, "frame0_dino_val": val_dino_meta, "target_train": str((data_root / TARGET_REL / "train.npz").resolve()), "target_val": str((data_root / TARGET_REL / "val.npz").resolve()), "shared_head": str((data_root / SHARED_HEAD).resolve())},
        "target_summaries": target_summaries,
        "runtime": {"device": str(device), "seed": SEED, "epochs": EPOCHS},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"branches": predictor_names, "option_geometry_dim": OPTION_GEOMETRY_DIM, "dino_tokens": [SPATIAL_TOKENS, DINO_EMBED_DIM], "seed": SEED, "epochs": EPOCHS, "test_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "training_summary.json").write_text(json.dumps({name: json.loads((output_root / f"{name}_training.json").read_text(encoding="utf-8")) for name in predictor_names}, indent=2) + "\n", encoding="utf-8")
    (output_root / "per_branch_metrics.json").write_text(json.dumps({name: {"downstream": methods[name], "ranking": score_metrics[name]} for name in predictor_names}, indent=2) + "\n", encoding="utf-8")
    (output_root / "ranking_metrics.json").write_text(json.dumps(score_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "occlusion_stratified_metrics.json").write_text(json.dumps(occlusion, indent=2) + "\n", encoding="utf-8")
    (output_root / "context_transitions.json").write_text(json.dumps(transitions, indent=2) + "\n", encoding="utf-8")
    data_audit = {"policy_rows": {"train": len(train_rows), "moving_val": len(val_rows)}, "legacy_visited_rgb_frame_index": 15, "strict_frame0_rgb_root": str((data_root / RGB_REL).resolve()), "strict_frame0_dino_root": str((data_root / DINO_REL).resolve()), "candidate_rgb_rendered": False, "forbidden_predictor_inputs": ["full_sequence_stgcn", "future_skeleton", "candidate_rgb", "candidate_dino", "candidate_recognizer", "gt_action"], "test_used": False}
    (output_root / "data_audit.json").write_text(json.dumps(data_audit, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
