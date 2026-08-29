#!/usr/bin/env python3
"""Build leakage-safe Stage C-v2 current-observation arrays from frozen caches."""

from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.action_recognition.st_gcn_model import STGCN
from activeview.active_view.stage_c_v2_features import (
    JOINT_COUNT,
    JOINT_TOKEN_DIM,
    SKELETON_SHAPE,
    extract_frozen_joint_tokens,
    semantic_context_from_current_feature,
    v2_schema,
)
from activeview.active_view.stage_c_v2_dataset import compute_v2_statistics, save_v2_statistics
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root
from activeview.perception.skeleton_definition import get_skeleton_definition


def _load_model(checkpoint: Path, device_name: str) -> tuple[STGCN, torch.device]:
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = STGCN(
        in_channels=3, num_classes=16, graph_strategy="spatial", edge_importance_weighting=True,
        skel_def=get_skeleton_definition(backend="h36m_17"),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, device


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _current_skeleton(episode: Mapping[str, Any]) -> np.ndarray:
    current = episode["current_view"]
    archive_path = Path(current["skeleton_source_path"])
    with np.load(archive_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
    if skeletons.ndim != 4 or skeletons.shape[1:] != SKELETON_SHAPE:
        raise ValueError(f"Invalid skeleton archive shape in {archive_path}: {skeletons.shape}")
    matches = np.flatnonzero(ids == int(current["viewpoint_id"]))
    if matches.size != 1:
        raise ValueError(f"Current viewpoint is not unique in {archive_path}")
    skeleton = skeletons[int(matches[0])]
    if not np.isfinite(skeleton).all():
        raise ValueError(f"Current skeleton is non-finite in {archive_path}")
    return skeleton


def _aligned_rows(stage_a: Path, stage_b: Path, source_features: Path) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    sentry = object()
    streams = (_jsonl_rows(stage_a), _jsonl_rows(stage_b), _jsonl_rows(source_features))
    for index, values in enumerate(zip_longest(*streams, fillvalue=sentry), 1):
        if any(value is sentry for value in values):
            raise ValueError(f"Stage A/B/source feature lengths disagree at line {index}")
        episode, utility, feature = values
        ids = {str(item["episode_id"]) for item in (episode, utility, feature)}
        if len(ids) != 1:
            raise ValueError(f"Stage A/B/source feature alignment mismatch at line {index}")
        feature_by_id = {
            int(candidate_id): float(target)
            for candidate_id, target in zip(feature["candidate_viewpoint_ids"], feature["utility_targets"])
        }
        utility_by_id = {
            int(candidate["viewpoint_id"]): float(candidate["utility"])
            for candidate in utility["candidates"]
        }
        if set(feature_by_id) != set(utility_by_id) or any(
            not np.isclose(feature_by_id[key], utility_by_id[key], atol=1e-7, rtol=0.0)
            for key in feature_by_id
        ):
            raise ValueError(f"Stage B/source feature utility mismatch at line {index}")
        yield episode, utility, feature


def _metadata_row(feature: Mapping[str, Any], index: int) -> Dict[str, Any]:
    current_feature = semantic_context_from_current_feature(feature["current_feature"])
    if current_feature.shape != (19,):
        raise ValueError("Unexpected semantic context shape")
    return {
        "episode_id": str(feature["episode_id"]), "record_id": str(feature["record_id"]),
        "policy_split": str(feature["policy_split"]), "scene_id": str(feature["scene_id"]),
        "region": str(feature["region"]), "label_id": int(feature["label_id"]),
        "current_viewpoint_id": int(feature["current_viewpoint_id"]),
        "current_semantic_context": current_feature.tolist(),
        "current_entropy": float(current_feature[16]), "current_margin": float(current_feature[17]),
        "current_pose_confidence": float(current_feature[18]),
        "joint_tokens_index": int(index), "skeleton_index": int(index),
        "candidate_viewpoint_ids": [int(value) for value in feature["candidate_viewpoint_ids"]],
        "candidate_geometry": feature["candidate_geometry"],
        "utility_targets": [float(value) for value in feature["utility_targets"]],
        "candidate_geodesic": [float(value) for value in feature["candidate_geodesic"]],
    }


def _build_split(
    *, split: str, stage_a: Path, stage_b: Path, source_features: Path,
    output_dir: Path, model: STGCN, device: torch.device, batch_size: int,
) -> tuple[int, Path, Path, Path]:
    count = _count_rows(source_features)
    arrays_dir = output_dir / "arrays"
    feature_dir = output_dir / "features"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    joint_path = arrays_dir / f"{split}_joint_tokens.npy"
    skeleton_path = arrays_dir / f"{split}_skeletons.npy"
    joint_array = open_memmap(joint_path, mode="w+", dtype=np.float32, shape=(count, JOINT_COUNT, JOINT_TOKEN_DIM))
    skeleton_array = open_memmap(skeleton_path, mode="w+", dtype=np.float32, shape=(count, *SKELETON_SHAPE))
    output_path = feature_dir / f"{split}.jsonl"
    pending: list[tuple[int, Mapping[str, Any], np.ndarray]] = []

    def flush(handle) -> None:
        if not pending:
            return
        skeleton_batch = torch.from_numpy(np.stack([item[2] for item in pending]).astype(np.float32)).to(device)
        tokens = extract_frozen_joint_tokens(model, skeleton_batch).cpu().numpy().astype(np.float32)
        for item, token in zip(pending, tokens):
            index, feature, skeleton = item
            joint_array[index] = token
            skeleton_array[index] = skeleton
            handle.write(json.dumps(_metadata_row(feature, index), separators=(",", ":"), ensure_ascii=False) + "\n")
        pending.clear()

    with output_path.open("w", encoding="utf-8") as handle:
        for index, (episode, _utility, feature) in enumerate(_aligned_rows(stage_a, stage_b, source_features)):
            pending.append((index, feature, _current_skeleton(episode)))
            if len(pending) >= batch_size:
                flush(handle)
        flush(handle)
    joint_array.flush(); skeleton_array.flush()
    del joint_array, skeleton_array
    return count, output_path, joint_path, skeleton_path


def build(*, dataset_root: Path, stage_b_root: Path, source_feature_root: Path, output_dir: Path, checkpoint: Path, device_name: str, batch_size: int, splits: Sequence[str]) -> Dict[str, Any]:
    started = time.perf_counter()
    source_summary_path = source_feature_root / "stage_c_feature_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    stage_a_summary_path = dataset_root / "stage_a_summary.json"
    stage_a_summary = json.loads(stage_a_summary_path.read_text(encoding="utf-8"))
    model, device = _load_model(checkpoint, device_name)
    counts: Dict[str, int] = {}
    feature_files: Dict[str, str] = {}
    array_files: Dict[str, Dict[str, str]] = {"joint_tokens": {}, "skeletons": {}}
    for split in splits:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")
        count, feature_path, joint_path, skeleton_path = _build_split(
            split=split,
            stage_a=Path(stage_a_summary["episode_files"][split]),
            stage_b=stage_b_root / "utility_labels" / f"{split}.jsonl",
            source_features=Path(source_summary["feature_files"][split]),
            output_dir=output_dir, model=model, device=device, batch_size=batch_size,
        )
        counts[split] = count
        feature_files[split] = str(feature_path.resolve())
        array_files["joint_tokens"][split] = str(joint_path.resolve())
        array_files["skeletons"][split] = str(skeleton_path.resolve())

    train_rows = list(_jsonl_rows(output_dir / "features/train.jsonl"))
    stats_path = output_dir / "stage_c_v2_feature_stats.json"
    save_v2_statistics(stats_path, compute_v2_statistics(train_rows))
    summary = {
        "protocol": "ACTIVEVIEW Stage C-v2 current-observation representation cache",
        "status": "generated", "schema": v2_schema(), "feature_files": feature_files,
        "feature_file_sha256": {split: file_sha256(Path(path)) for split, path in feature_files.items()},
        "feature_file_counts": counts,
        "arrays": array_files,
        "array_sha256": {kind: {split: file_sha256(Path(path)) for split, path in values.items()} for kind, values in array_files.items()},
        "feature_stats": str(stats_path.resolve()), "feature_stats_sha256": file_sha256(stats_path),
        "source_stage_a_summary": str(stage_a_summary_path.resolve()), "source_stage_a_summary_sha256": file_sha256(stage_a_summary_path),
        "source_stage_b_root": str(stage_b_root.resolve()),
        "source_stage_b_summary_sha256": file_sha256(stage_b_root / "stage_b_summary.json") if (stage_b_root / "stage_b_summary.json").is_file() else None,
        "source_stage_b_utility_sha256": {split: file_sha256(stage_b_root / "utility_labels" / f"{split}.jsonl") for split in splits},
        "source_stage_c_v0_summary": str(source_summary_path.resolve()), "source_stage_c_v0_summary_sha256": file_sha256(source_summary_path),
        "source_stage_c_v0_feature_sha256": {split: file_sha256(Path(source_summary["feature_files"][split])) for split in splits},
        "stgcn_checkpoint": str(checkpoint.resolve()), "stgcn_checkpoint_sha256": file_sha256(checkpoint),
        "label_mapping": source_summary["label_mapping"], "label_mapping_sha256": source_summary["label_mapping_sha256"],
        "canonical_split_counts": {"train": 589, "val": 197, "test": 194},
        "built_splits": list(splits), "test_built": "test" in splits, "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "stage_c_v2_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage_c_v2_feature_summary": str(summary_path.resolve()), "feature_file_counts": counts}, ensure_ascii=False))
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--source-feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    build(dataset_root=args.dataset_root, stage_b_root=args.stage_b_root, source_feature_root=args.source_feature_root, output_dir=args.output_dir, checkpoint=args.checkpoint, device_name=args.device, batch_size=args.batch_size, splits=args.splits)


if __name__ == "__main__":
    main()
