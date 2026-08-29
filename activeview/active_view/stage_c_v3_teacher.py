"""Future-perception teacher cache and dataset helpers for Stage C-v3.

The teacher is intentionally a diagnostic upper bound.  It consumes only
serialized Stage C current features and Stage B candidate diagnostics; it never
invokes Habitat or any perception model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from activeview.active_view.stage_c_evaluation import _candidate_choice


CURRENT_DIM = 275
GEOMETRY_DIM = 11
FUTURE_DIM = 17
NUM_CLASSES = 16
FUTURE_FEATURE_NAMES = tuple(
    [f"future_candidate_predicted_label_one_hot_{index}" for index in range(NUM_CLASSES)]
    + ["future_candidate_entropy"]
)


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    """Load non-empty JSONL records in file order."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def future_perception_vector(candidate: Mapping[str, Any]) -> np.ndarray:
    """Encode the future fields actually persisted by Stage B.

    Stage B does not persist a full 16-D log-probability vector, pooled ST-GCN
    feature or pose confidence.  We therefore use only future predicted class
    (one-hot) and entropy.  The true-label log probability is deliberately
    excluded because it depends on the ground-truth label and is a direct
    ingredient of the Stage B utility target.  The ``correct`` flag is also
    excluded because it is ground-truth-derived.
    """
    predicted = int(candidate["predicted_label_id"])
    if predicted < 0 or predicted >= NUM_CLASSES:
        raise ValueError(f"Invalid future predicted_label_id: {predicted}")
    values = np.zeros(FUTURE_DIM, dtype=np.float32)
    values[predicted] = 1.0
    values[NUM_CLASSES] = float(candidate["entropy"])
    if not np.isfinite(values).all():
        raise ValueError("Future candidate perception contains non-finite values")
    return values


def _utility_by_id(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    candidates = record.get("candidates", [])
    result = {int(item["viewpoint_id"]): item for item in candidates}
    if len(result) != len(candidates):
        raise ValueError(f"Duplicate candidate viewpoint in {record.get('episode_id')}")
    return result


def _build_episode_rows(feature_rows: Sequence[Mapping[str, Any]], utility_rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    utilities = {str(row["episode_id"]): row for row in utility_rows}
    if len(utilities) != len(utility_rows):
        raise ValueError("Duplicate Stage B episode_id in teacher input")
    output: list[Dict[str, Any]] = []
    for feature in feature_rows:
        episode_id = str(feature["episode_id"])
        utility = utilities.get(episode_id)
        if utility is None:
            raise ValueError(f"Missing Stage B utility for {episode_id}")
        if str(utility["policy_split"]) != str(feature["policy_split"]):
            raise ValueError(f"Split mismatch for {episode_id}")
        feature_ids = [int(value) for value in feature["candidate_viewpoint_ids"]]
        feature_geometry = _finite_vector(
            feature["candidate_geometry"],
            (len(feature_ids), GEOMETRY_DIM),
            "candidate_geometry",
        )
        feature_targets = _finite_vector(feature["utility_targets"], (len(feature_ids),), "utility_targets")
        utility_by_id = _utility_by_id(utility)
        if set(feature_ids) != set(utility_by_id):
            raise ValueError(f"Candidate IDs mismatch for {episode_id}")
        future = np.stack([future_perception_vector(utility_by_id[candidate_id]) for candidate_id in feature_ids])
        targets = np.asarray([float(utility_by_id[candidate_id]["utility"]) for candidate_id in feature_ids], dtype=np.float32)
        if not np.allclose(targets, feature_targets, atol=1e-6, rtol=0.0):
            raise ValueError(f"Utility targets mismatch for {episode_id}")
        output.append({
            "episode_id": episode_id,
            "record_id": str(feature["record_id"]),
            "policy_split": str(feature["policy_split"]),
            "scene_id": str(feature["scene_id"]),
            "region": str(feature["region"]),
            "label_id": int(feature["label_id"]),
            "current_viewpoint_id": int(feature["current_viewpoint_id"]),
            "current_feature": _finite_vector(feature["current_feature"], (CURRENT_DIM,), "current_feature").tolist(),
            "candidate_viewpoint_ids": feature_ids,
            "candidate_geometry": feature_geometry.tolist(),
            "future_candidate_perception": future.tolist(),
            "utility_targets": targets.tolist(),
            "candidate_geodesic": [float(utility_by_id[candidate_id]["geodesic_distance_m"]) for candidate_id in feature_ids],
        })
    return output


def _stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, list[float]]:
    current = np.asarray([row["current_feature"] for row in rows], dtype=np.float64)
    geometry = np.concatenate([np.asarray(row["candidate_geometry"], dtype=np.float64) for row in rows])
    future = np.concatenate([np.asarray(row["future_candidate_perception"], dtype=np.float64) for row in rows])
    result: Dict[str, np.ndarray] = {}
    for name, values in (("current", current), ("geometry", geometry), ("future", future)):
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        result[f"{name}_mean"] = mean.astype(np.float32)
        result[f"{name}_std"] = std.astype(np.float32)
    return {key: value.tolist() for key, value in result.items()}


def build_teacher_cache(
    *, feature_root: Path, stage_b_root: Path, output_dir: Path, splits: Sequence[str] = ("train", "val"),
) -> Dict[str, Any]:
    """Convert frozen v0 rows into a future-perception teacher cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((feature_root / "stage_c_feature_summary.json").read_text(encoding="utf-8"))
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    hashes: Dict[str, str] = {}
    all_rows: Dict[str, list[Dict[str, Any]]] = {}
    for split in splits:
        if split not in ("train", "val"):
            raise ValueError("Teacher cache is restricted to train/val; Test is locked")
        rows = _build_episode_rows(
            load_jsonl(feature_root / "features" / f"{split}.jsonl"),
            load_jsonl(stage_b_root / "utility_labels" / f"{split}.jsonl"),
        )
        path = feature_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        all_rows[split] = rows
        counts[split] = len(rows)
        hashes[split] = _sha256(path)
    stats_path = output_dir / "teacher_feature_stats.json"
    stats_path.write_text(json.dumps(_stats(all_rows["train"]), indent=2), encoding="utf-8")
    summary = {
        "protocol": "ACTIVEVIEW Stage C-v3 future-perception teacher diagnostic",
        "status": "generated",
        "diagnostic_only": True,
        "deployable_policy": False,
        "future_candidate_perception_used": True,
        "schema": {
            "current_feature_dim": CURRENT_DIM,
            "candidate_geometry_dim": GEOMETRY_DIM,
            "future_candidate_perception_dim": FUTURE_DIM,
            "future_candidate_perception_names": list(FUTURE_FEATURE_NAMES),
            "future_fields_source": ["predicted_label_id", "entropy"],
            "excluded_ground_truth_dependent_fields": ["logp_true", "correct"],
            "unavailable_fields_not_fabricated": ["full_candidate_log_probs", "candidate_stgcn_feature", "candidate_pose_confidence"],
        },
        "feature_files": {split: str((feature_dir / f"{split}.jsonl").resolve()) for split in splits},
        "feature_file_sha256": hashes,
        "feature_file_counts": counts,
        "feature_stats": str(stats_path.resolve()),
        "feature_stats_sha256": _sha256(stats_path),
        "source_stage_c_v0_feature_root": str(feature_root.resolve()),
        "source_stage_c_v0_summary": str((feature_root / "stage_c_feature_summary.json").resolve()),
        "source_stage_c_v0_summary_sha256": _sha256(feature_root / "stage_c_feature_summary.json"),
        "source_stage_c_v0_feature_sha256": {
            split: _sha256(feature_root / "features" / f"{split}.jsonl") for split in splits
        },
        "label_mapping": str(Path(source_summary["label_mapping"]).resolve()),
        "label_mapping_sha256": source_summary.get("label_mapping_sha256"),
        "source_stage_b_root": str(stage_b_root.resolve()),
        "source_stage_b_summary": str((stage_b_root / "stage_b_summary.json").resolve()),
        "source_stage_b_summary_sha256": _sha256(stage_b_root / "stage_b_summary.json") if (stage_b_root / "stage_b_summary.json").is_file() else None,
        "source_stage_b_utility_sha256": {
            split: _sha256(stage_b_root / "utility_labels" / f"{split}.jsonl") for split in splits
        },
        "built_splits": list(splits),
        "test_built": False,
    }
    summary_path = output_dir / "stage_c_v3_teacher_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FutureTeacherDataset(Dataset[Dict[str, Any]]):
    """Episode-level teacher cache with Train-only normalization statistics."""

    def __init__(self, path: Path, stats: Mapping[str, Sequence[float]]) -> None:
        self.rows = load_jsonl(path)
        self.current_mean = np.asarray(stats["current_mean"], dtype=np.float32)
        self.current_std = np.asarray(stats["current_std"], dtype=np.float32)
        self.geometry_mean = np.asarray(stats["geometry_mean"], dtype=np.float32)
        self.geometry_std = np.asarray(stats["geometry_std"], dtype=np.float32)
        self.future_mean = np.asarray(stats["future_mean"], dtype=np.float32)
        self.future_std = np.asarray(stats["future_std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        current = (np.asarray(row["current_feature"], dtype=np.float32) - self.current_mean) / self.current_std
        geometry = (np.asarray(row["candidate_geometry"], dtype=np.float32) - self.geometry_mean) / self.geometry_std
        future = (np.asarray(row["future_candidate_perception"], dtype=np.float32) - self.future_mean) / self.future_std
        return {
            "current_feature": torch.from_numpy(current),
            "candidate_geometry": torch.from_numpy(geometry),
            "future_candidate_perception": torch.from_numpy(future),
            "utility_targets": torch.tensor(row["utility_targets"], dtype=torch.float32),
            "candidate_geodesic": torch.tensor(row["candidate_geodesic"], dtype=torch.float32),
            "candidate_ids": [int(value) for value in row["candidate_viewpoint_ids"]],
            "episode_id": str(row["episode_id"]), "record_id": str(row["record_id"]),
            "policy_split": str(row["policy_split"]), "scene_id": str(row["scene_id"]),
            "region": str(row["region"]), "label_id": int(row["label_id"]),
        }


def collate_future_teacher(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    size = len(batch)
    max_candidates = max(len(item["candidate_ids"]) for item in batch)
    geometry = torch.zeros((size, max_candidates, GEOMETRY_DIM), dtype=torch.float32)
    future = torch.zeros((size, max_candidates, FUTURE_DIM), dtype=torch.float32)
    targets = torch.zeros((size, max_candidates), dtype=torch.float32)
    geodesic = torch.zeros((size, max_candidates), dtype=torch.float32)
    mask = torch.zeros((size, max_candidates), dtype=torch.bool)
    ids: list[list[int]] = []
    for row_index, item in enumerate(batch):
        count = len(item["candidate_ids"])
        geometry[row_index, :count] = item["candidate_geometry"]
        future[row_index, :count] = item["future_candidate_perception"]
        targets[row_index, :count] = item["utility_targets"]
        geodesic[row_index, :count] = item["candidate_geodesic"]
        mask[row_index, :count] = True
        ids.append(list(item["candidate_ids"]))
    return {
        "current_feature": torch.stack([item["current_feature"] for item in batch]),
        "candidate_geometry": geometry, "future_candidate_perception": future,
        "utility_targets": targets, "candidate_geodesic": geodesic,
        "candidate_mask": mask, "candidate_ids": ids,
        "episode_id": [str(item["episode_id"]) for item in batch],
        "record_id": [str(item["record_id"]) for item in batch],
        "policy_split": [str(item["policy_split"]) for item in batch],
        "scene_id": [str(item["scene_id"]) for item in batch],
        "region": [str(item["region"]) for item in batch],
        "label_id": torch.tensor([int(item["label_id"]) for item in batch], dtype=torch.long),
    }


def predict_teacher_dataset(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    stage_b_lookup: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> list[Dict[str, Any]]:
    """Materialize standard Stage C prediction rows from teacher utilities."""
    model.eval()
    rows: list[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            predicted = model(
                batch["current_feature"].to(device),
                batch["candidate_geometry"].to(device),
                batch["future_candidate_perception"].to(device),
                batch["candidate_mask"].to(device),
            ).cpu().numpy()
            valid = batch["candidate_mask"].numpy()
            targets = batch["utility_targets"].numpy()
            geodesic = batch["candidate_geodesic"].numpy()
            for index, episode_id in enumerate(batch["episode_id"]):
                ids = batch["candidate_ids"][index]
                target_values = [float(value) for value in targets[index][valid[index]]]
                predicted_values = [float(value) for value in predicted[index][valid[index]]]
                geo_values = [float(value) for value in geodesic[index][valid[index]]]
                stage_b = stage_b_lookup[str(episode_id)]
                by_id = {int(item["viewpoint_id"]): item for item in stage_b["candidates"]}
                predicted_id, max_predicted = _candidate_choice(predicted_values, ids, geo_values)
                stays = max_predicted <= 0.0
                current_id = int(stage_b["current"]["viewpoint_id"])
                selected_id = current_id if stays else predicted_id
                selected = stage_b["current"] if stays else by_id[selected_id]
                oracle = stage_b["oracle"]
                oracle_id = int(oracle["candidate_oracle_viewpoint_id"])
                safe_id = int(oracle["safe_oracle_viewpoint_id"])
                oracle_item = by_id[oracle_id]
                safe_stays = bool(oracle["safe_oracle_stays"])
                safe_item = stage_b["current"] if safe_stays else by_id[safe_id]
                rows.append({
                    "episode_id": str(episode_id), "record_id": str(batch["record_id"][index]),
                    "policy_split": str(batch["policy_split"][index]), "scene_id": str(batch["scene_id"][index]),
                    "region": str(batch["region"][index]), "label_id": int(batch["label_id"][index]),
                    "current_viewpoint_id": current_id, "candidate_viewpoint_ids": ids,
                    "utility_targets": target_values, "predicted_utilities": predicted_values,
                    "predicted_candidate_viewpoint_id": predicted_id,
                    "predicted_action": "stay" if stays else f"candidate:{predicted_id}",
                    "predicted_stays": stays,
                    "selected_true_utility": 0.0 if stays else float(selected["utility"]),
                    "selected_predicted_label_id": int(selected["predicted_label_id"]),
                    "selected_entropy": float(selected["entropy"]),
                    "current_predicted_label_id": int(stage_b["current"]["predicted_label_id"]),
                    "current_entropy": float(stage_b["current"]["entropy"]),
                    "candidate_oracle_viewpoint_id": oracle_id,
                    "candidate_oracle_predicted_label_id": int(oracle_item["predicted_label_id"]),
                    "candidate_oracle_entropy": float(oracle_item["entropy"]),
                    "safe_oracle_viewpoint_id": safe_id, "safe_oracle_stays": safe_stays,
                    "safe_oracle_action": "stay" if safe_stays else f"candidate:{safe_id}",
                    "safe_oracle_utility": float(oracle["safe_oracle_utility"]),
                    "safe_oracle_predicted_label_id": int(safe_item["predicted_label_id"]),
                    "safe_oracle_entropy": float(safe_item["entropy"]),
                    "regret": float(oracle["safe_oracle_utility"]) - (0.0 if stays else float(selected["utility"])),
                })
    return rows


def load_teacher_stats(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}
