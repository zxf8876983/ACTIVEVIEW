"""Stage D second-step cache, dataset and navigation-only geometry helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from activeview.active_view.stage_c_features import candidate_geometry_features, current_state_features, frozen_current_features
from activeview.active_view.stage_d_policy import (
    CURRENT_DIM,
    DELTA_SEMANTIC_DIM,
    GEOMETRY_DIM,
    TOP_K,
    order_candidates,
    semantic_delta,
    second_step_utility,
)
from activeview.active_view.utility_label_builder import file_sha256


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _lookup(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row["episode_id"])
        if key in result:
            raise ValueError(f"Duplicate {name} episode_id: {key}")
        result[key] = row
    return result


def load_pairwise_geodesic(path: Path) -> dict[int, dict[int, float]]:
    """Load one navigation-only scene/region matrix written by the builder."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [int(value) for value in payload["viewpoint_ids"]]
    matrix = np.asarray(payload["geodesic_distance_m"], dtype=np.float64)
    if matrix.shape != (len(ids), len(ids)):
        raise ValueError(f"Invalid pairwise geodesic shape in {path}")
    if np.isnan(matrix).any() or np.any(matrix < 0.0):
        raise ValueError(f"Invalid pairwise geodesic values in {path}")
    return {left: {right: float(matrix[i, j]) for j, right in enumerate(ids) if np.isfinite(matrix[i, j])} for i, left in enumerate(ids)}


def _view_from_archive(path: Path, viewpoint_id: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
        positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        rotations = np.asarray(archive["viewpoint_rotations_wxyz"], dtype=np.float32)
        placement = np.asarray(archive["placement_position"], dtype=np.float32)
    matches = np.flatnonzero(ids == int(viewpoint_id))
    if matches.size != 1 or skeletons.shape != (32, 3, 30, 17) or confidence.shape != (32,):
        raise ValueError(f"Invalid archive schema for viewpoint {viewpoint_id}: {path}")
    index = int(matches[0])
    skeleton = skeletons[index]
    pose_confidence = float(confidence[index])
    if positions.shape != (32, 3) or rotations.shape != (32, 4) or placement.shape != (3,):
        raise ValueError(f"Invalid navigation schema: {path}")
    if not np.isfinite(skeleton).all() or not np.isfinite(pose_confidence):
        raise ValueError(f"Non-finite s1 observation: {path} viewpoint {viewpoint_id}")
    return skeleton, pose_confidence, positions[index], rotations[index]


def _relative_azimuth(delta: np.ndarray, rotation_wxyz: np.ndarray) -> float:
    quat = np.asarray(rotation_wxyz, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat.tolist()
    yaw = float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))
    ego_x = float(np.cos(yaw) * delta[0] - np.sin(yaw) * delta[2])
    ego_z = float(np.sin(yaw) * delta[0] + np.cos(yaw) * delta[2])
    return float(np.degrees(np.arctan2(ego_x, ego_z)))


def second_step_geometry(
    *,
    s1_position: Sequence[float],
    s1_rotation_wxyz: Sequence[float],
    target_position: Sequence[float],
    target_snapped_position: Sequence[float],
    target_geodesic: float,
    placement_position: Sequence[float],
) -> np.ndarray:
    """Build the existing 11-D geometry schema from s1 to an unvisited view."""
    delta = np.asarray(target_position, dtype=np.float32) - np.asarray(s1_position, dtype=np.float32)
    candidate = {
        "relative_position": delta.tolist(),
        "snapped_position": np.asarray(target_snapped_position, dtype=np.float32).tolist(),
        "euclidean_distance_m": float(np.linalg.norm(delta)),
        "geodesic_distance_m": float(target_geodesic),
        "relative_azimuth_deg": _relative_azimuth(delta, np.asarray(s1_rotation_wxyz, dtype=np.float32)),
    }
    return candidate_geometry_features(
        candidate,
        current_position=np.asarray(s1_position, dtype=np.float32),
        current_rotation_wxyz=np.asarray(s1_rotation_wxyz, dtype=np.float32),
        placement_position=np.asarray(placement_position, dtype=np.float32),
    )


def compute_stage_d_statistics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot compute Stage D statistics from empty rows")
    current = np.asarray([row["s0_feature"] for row in rows] + [row["s1_feature"] for row in rows], dtype=np.float64)
    delta = np.asarray([row["delta_semantic"] for row in rows], dtype=np.float64)
    geometry = np.concatenate([np.asarray(row["second_step_candidate_geometry"], dtype=np.float64) for row in rows], axis=0)
    result: Dict[str, np.ndarray] = {}
    for name, values in (("current", current), ("delta", delta), ("geometry", geometry)):
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        result[f"{name}_mean"] = mean.astype(np.float32)
        result[f"{name}_std"] = std.astype(np.float32)
    return result


def save_stage_d_statistics(path: Path, statistics: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: np.asarray(value).tolist() for key, value in statistics.items()}, indent=2), encoding="utf-8")


def load_stage_d_statistics(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}


class StageDDataset(Dataset[Dict[str, Any]]):
    """Train/Val second-step samples with Train-only normalization stats."""

    def __init__(self, path: Path, statistics: Mapping[str, np.ndarray]) -> None:
        self.rows = load_jsonl(path)
        self.statistics = {key: np.asarray(value, dtype=np.float32) for key, value in statistics.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        s0 = (np.asarray(row["s0_feature"], dtype=np.float32) - self.statistics["current_mean"]) / self.statistics["current_std"]
        s1 = (np.asarray(row["s1_feature"], dtype=np.float32) - self.statistics["current_mean"]) / self.statistics["current_std"]
        delta = (np.asarray(row["delta_semantic"], dtype=np.float32) - self.statistics["delta_mean"]) / self.statistics["delta_std"]
        geometry = (np.asarray(row["second_step_candidate_geometry"], dtype=np.float32) - self.statistics["geometry_mean"]) / self.statistics["geometry_std"]
        return {
            "s0_feature": torch.from_numpy(s0), "s1_feature": torch.from_numpy(s1), "delta_semantic": torch.from_numpy(delta),
            "candidate_geometry": torch.from_numpy(geometry), "utility_targets": torch.tensor(row["second_step_utility_targets"], dtype=torch.float32),
            "candidate_geodesic": torch.tensor(row["second_step_candidate_geodesic"], dtype=torch.float32),
            "candidate_ids": [int(value) for value in row["remaining_candidate_ids"]], "episode_id": str(row["episode_id"]),
            "record_id": str(row["record_id"]), "policy_split": str(row["policy_split"]), "scene_id": str(row["scene_id"]),
            "region": str(row["region"]), "label_id": int(row["label_id"]),
        }


def collate_stage_d(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    size = len(batch)
    max_candidates = max(len(row["candidate_ids"]) for row in batch)
    geometry = torch.zeros((size, max_candidates, GEOMETRY_DIM), dtype=torch.float32)
    targets = torch.zeros((size, max_candidates), dtype=torch.float32)
    geodesic = torch.zeros((size, max_candidates), dtype=torch.float32)
    mask = torch.zeros((size, max_candidates), dtype=torch.bool)
    ids: list[list[int]] = []
    for index, row in enumerate(batch):
        count = len(row["candidate_ids"])
        geometry[index, :count] = row["candidate_geometry"]
        targets[index, :count] = row["utility_targets"]
        geodesic[index, :count] = row["candidate_geodesic"]
        mask[index, :count] = True
        ids.append(list(row["candidate_ids"]))
    return {
        "s0_feature": torch.stack([row["s0_feature"] for row in batch]), "s1_feature": torch.stack([row["s1_feature"] for row in batch]),
        "delta_semantic": torch.stack([row["delta_semantic"] for row in batch]), "candidate_geometry": geometry,
        "utility_targets": targets, "candidate_geodesic": geodesic, "candidate_mask": mask, "candidate_ids": ids,
        "episode_id": [str(row["episode_id"]) for row in batch], "record_id": [str(row["record_id"]) for row in batch],
        "policy_split": [str(row["policy_split"]) for row in batch], "scene_id": [str(row["scene_id"]) for row in batch],
        "region": [str(row["region"]) for row in batch], "label_id": torch.tensor([int(row["label_id"]) for row in batch], dtype=torch.long),
    }


class StageDRecordBalancedSampler(Sampler[int]):
    """Sample a fixed number of second-step Episodes per eligible record."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], episodes_per_record: int = 16, seed: int = 42) -> None:
        if episodes_per_record <= 0:
            raise ValueError("episodes_per_record must be positive")
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.groups[str(row["record_id"])].append(index)
        self.episodes_per_record = int(episodes_per_record)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        selected: list[int] = []
        for record_id in sorted(self.groups):
            group = np.asarray(self.groups[record_id], dtype=np.int64)
            selected.extend(int(value) for value in rng.choice(group, size=self.episodes_per_record, replace=len(group) < self.episodes_per_record))
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.groups) * self.episodes_per_record


def build_second_step_rows(
    *,
    stage_a_rows: Sequence[Mapping[str, Any]],
    stage_b_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    v0_prediction_rows: Sequence[Mapping[str, Any]],
    pairwise_by_region: Mapping[tuple[str, str], Mapping[int, Mapping[int, float]]],
    stgcn_model: torch.nn.Module,
    device: torch.device,
) -> tuple[list[Dict[str, Any]], int]:
    """Build second-step rows from one split; return rows and v0-move count."""
    stage_a = _lookup(stage_a_rows, "Stage A")
    stage_b = _lookup(stage_b_rows, "Stage B")
    predictions = _lookup(v0_prediction_rows, "v0 prediction")
    archive_feature_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]] = {}
    output: list[Dict[str, Any]] = []
    move_count = 0
    for feature in feature_rows:
        episode_id = str(feature["episode_id"])
        episode = stage_a.get(episode_id)
        utility = stage_b.get(episode_id)
        prediction = predictions.get(episode_id)
        if episode is None or utility is None or prediction is None:
            raise ValueError(f"Missing aligned Stage A/B/prediction row for {episode_id}")
        if str(feature["policy_split"]) != str(utility["policy_split"]):
            raise ValueError(f"Split mismatch for {episode_id}")
        ids = [int(value) for value in prediction["candidate_viewpoint_ids"]]
        predicted = [float(value) for value in prediction["predicted_utilities"]]
        by_id = {int(item["viewpoint_id"]): item for item in utility["candidates"]}
        geodesic = [float(by_id[item]["geodesic_distance_m"]) for item in ids]
        if set(ids) != set(by_id):
            raise ValueError(f"v0 candidate IDs disagree for {episode_id}")
        ordered = order_candidates(predicted, ids, geodesic, top_k=TOP_K)
        if not ordered or float(predicted[ids.index(ordered[0])]) <= 0.0:
            continue
        move_count += 1
        p1_id = int(ordered[0])
        remaining = [candidate_id for candidate_id in ordered[1:] if candidate_id in by_id]
        if not remaining:
            continue
        p1_source = next(item for item in episode["candidate_pool"] if int(item["viewpoint_id"]) == p1_id)
        archive_path = Path(p1_source["skeleton_source_path"])
        cache_key = (str(archive_path), p1_id)
        if cache_key not in archive_feature_cache:
            skeleton, confidence, position, rotation = _view_from_archive(archive_path, p1_id)
            feature_vector, log_probs = frozen_current_features(stgcn_model, skeleton, device)
            current_vector = current_state_features(feature_vector, log_probs, confidence)
            with np.load(archive_path, allow_pickle=False) as archive:
                placement = np.asarray(archive["placement_position"], dtype=np.float32)
            archive_feature_cache[cache_key] = (current_vector, log_probs, confidence, position, rotation, placement)
        s1_feature, s1_log_probs, _confidence, s1_position, s1_rotation, placement = archive_feature_cache[cache_key]
        stage_a_candidates = {int(item["viewpoint_id"]): item for item in episode["candidate_pool"]}
        pairwise = pairwise_by_region.get((str(episode["scene_id"]), str(episode["region"])), {})
        valid_remaining: list[int] = []
        geometries: list[np.ndarray] = []
        targets: list[float] = []
        distances: list[float] = []
        for candidate_id in remaining:
            if p1_id not in pairwise or candidate_id not in pairwise[p1_id]:
                continue
            source = stage_a_candidates.get(candidate_id)
            if source is None:
                continue
            distance = float(pairwise[p1_id][candidate_id])
            if not np.isfinite(distance):
                continue
            geometry = second_step_geometry(
                s1_position=s1_position, s1_rotation_wxyz=s1_rotation,
                target_position=source["snapped_position"], target_snapped_position=source["snapped_position"],
                target_geodesic=distance, placement_position=placement,
            )
            valid_remaining.append(candidate_id)
            geometries.append(geometry)
            targets.append(second_step_utility(by_id[candidate_id]["logp_true"], by_id[p1_id]["logp_true"]))
            distances.append(distance)
        if not valid_remaining:
            continue
        s0_feature = np.asarray(feature["current_feature"], dtype=np.float32)
        if s0_feature.shape != (CURRENT_DIM,) or not np.isfinite(s0_feature).all():
            raise ValueError(f"Invalid Stage C current feature for {episode_id}: {s0_feature.shape}")
        output.append({
            "episode_id": episode_id, "record_id": str(episode["record_id"]), "policy_split": str(episode["policy_split"]),
            "scene_id": str(episode["scene_id"]), "region": str(episode["region"]), "label_id": int(episode["label_id"]),
            "s0_viewpoint_id": int(episode["current_view"]["viewpoint_id"]), "s1_viewpoint_id": p1_id,
            "proposal_rank_1_id": p1_id,
            "s0_feature": s0_feature.tolist(), "s1_feature": s1_feature.tolist(),
            "delta_semantic": semantic_delta(s0_feature, s1_feature).tolist(),
            "remaining_candidate_ids": valid_remaining, "second_step_candidate_geometry": np.stack(geometries).tolist(),
            "second_step_utility_targets": targets, "second_step_candidate_geodesic": distances,
            "first_step_geodesic": float(by_id[p1_id]["geodesic_distance_m"]),
            "first_step_predicted_utility": float(predicted[ids.index(p1_id)]),
            "future_unvisited_candidate_perception_used_as_input": False,
            "visited_s1_perception_used_as_input": True, "gt_label_used_as_input": False,
            "logp_true_used_as_input": False, "safe_oracle_used_as_input": False,
        })
    return output, move_count


def build_cache_summary(
    *, output_dir: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]],
    source_paths: Mapping[str, Path], checkpoint: Path, pairwise_root: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {"train": list(train_rows), "val": list(val_rows)}
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for split, rows in all_rows.items():
        path = feature_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        counts[split] = len(rows)
        hashes[split] = file_sha256(path)
    stats_path = output_dir / "stage_d_feature_stats.json"
    save_stage_d_statistics(stats_path, compute_stage_d_statistics(all_rows["train"]))
    summary = {
        "protocol": "ACTIVEVIEW Stage D two-step sequential cache",
        "status": "generated", "built_splits": ["train", "val"], "test_built": False,
        "schema": {"s0_feature_dim": CURRENT_DIM, "s1_feature_dim": CURRENT_DIM, "delta_semantic_dim": DELTA_SEMANTIC_DIM, "candidate_geometry_dim": GEOMETRY_DIM, "proposal_top_k": TOP_K},
        "feature_files": {split: str((feature_dir / f"{split}.jsonl").resolve()) for split in all_rows},
        "feature_file_sha256": hashes, "feature_file_counts": counts,
        "feature_stats": str(stats_path.resolve()), "feature_stats_sha256": file_sha256(stats_path),
        "source_stage_c_v0_predictions": {split: str(source_paths[split].resolve()) for split in ("train", "val")},
        "source_stage_c_v0_predictions_sha256": {split: file_sha256(source_paths[split]) for split in ("train", "val")},
        "stgcn_checkpoint": str(checkpoint.resolve()), "stgcn_checkpoint_sha256": file_sha256(checkpoint),
        "pairwise_geodesic_root": str(pairwise_root.resolve()),
        "future_unvisited_candidate_perception_used_as_input": False, "visited_s1_perception_used_as_input": True,
        "gt_label_used_as_input": False, "logp_true_used_as_input": False, "safe_oracle_used_as_input": False,
    }
    summary_path = output_dir / "stage_d_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
