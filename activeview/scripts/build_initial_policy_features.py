#!/usr/bin/env python3
"""Build current-only policy features from accepted Stage A/B artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.action_recognition.st_gcn_model import STGCN
from activeview.active_view.policy_data import feature_statistics, save_feature_statistics
from activeview.active_view.policy_features import (
    CURRENT_FEATURE_DIM,
    candidate_geometry_matrix,
    current_state_features,
    schema_metadata,
)
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root
from activeview.perception.skeleton_definition import get_skeleton_definition


SPLITS = ("train", "val", "test")


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


def _iter_pairs(stage_a_path: Path, stage_b_path: Path) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    with stage_a_path.open(encoding="utf-8") as stage_a, stage_b_path.open(encoding="utf-8") as stage_b:
        for line_number, (left, right) in enumerate(zip(stage_a, stage_b), 1):
            episode = json.loads(left)
            utility = json.loads(right)
            if episode["episode_id"] != utility["episode_id"]:
                raise ValueError(f"Stage A/B Episode mismatch at line {line_number}")
            yield episode, utility
        if next(stage_a, None) is not None or next(stage_b, None) is not None:
            raise ValueError("Stage A and Stage B files have different lengths")


def _make_row(episode: Mapping[str, Any], utility: Mapping[str, Any], current_feature: np.ndarray, placement_position: np.ndarray) -> Dict[str, Any]:
    current = episode["current_view"]
    candidates = episode["candidate_pool"]
    utility_by_id = {int(item["viewpoint_id"]): float(item["utility"]) for item in utility["candidates"]}
    candidate_ids = [int(item["viewpoint_id"]) for item in candidates]
    if set(candidate_ids) != set(utility_by_id):
        raise ValueError(f"Candidate IDs disagree for {episode['episode_id']}")
    geometry = candidate_geometry_matrix(
        candidates,
        current_position=current["agent_position"],
        current_rotation_wxyz=current["rotation_wxyz"],
        placement_position=placement_position,
    )
    return {
        "episode_id": str(episode["episode_id"]), "record_id": str(episode["record_id"]),
        "policy_split": str(episode["policy_split"]), "scene_id": str(episode["scene_id"]),
        "region": str(episode["region"]), "label_id": int(episode["label_id"]),
        "current_viewpoint_id": int(current["viewpoint_id"]),
        "current_feature": current_feature.tolist(),
        "candidate_viewpoint_ids": candidate_ids,
        "candidate_geometry": geometry.tolist(),
        "utility_targets": [utility_by_id[index] for index in candidate_ids],
        "candidate_geodesic": [float(item["geodesic_distance_m"]) for item in candidates],
        "current_pose_confidence_available": bool(current.get("pose_confidence_available", False)),
    }


def _process_batch(
    pending: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray, np.ndarray, float]],
    model: STGCN,
    device: torch.device,
) -> List[Dict[str, Any]]:
    skeletons = torch.from_numpy(np.stack([item[2] for item in pending]).astype(np.float32)).to(device).unsqueeze(-1)
    with torch.inference_mode():
        features = model.forward_features(skeletons)
        log_probs = torch.log_softmax(model.fc(features), dim=-1)
    output: List[Dict[str, Any]] = []
    for index, (episode, utility, _skeleton, placement, confidence) in enumerate(pending):
        current_feature = current_state_features(
            features[index].cpu().numpy(), log_probs[index].cpu().numpy(),
            confidence,
        )
        output.append(_make_row(episode, utility, current_feature, placement))
    return output


def _load_pending(episode: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    path = Path(episode["current_view"]["skeleton_source_path"])
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
        placement = np.asarray(archive["placement_position"], dtype=np.float32)
        confidence = np.asarray(archive["confidence"], dtype=np.float32)
    viewpoint_id = int(episode["current_view"]["viewpoint_id"])
    matches = np.flatnonzero(ids == viewpoint_id)
    if skeletons.shape != (32, 3, 30, 17) or matches.size != 1 or confidence.shape != ids.shape:
        raise ValueError(f"Invalid current skeleton archive {path}")
    skeleton = skeletons[int(matches[0])]
    if not np.isfinite(skeleton).all() or placement.shape != (3,) or not np.isfinite(placement).all():
        raise ValueError(f"Invalid current skeleton/placement values in {path}")
    confidence_value = float(confidence[int(matches[0])])
    if not np.isfinite(confidence_value):
        raise ValueError(f"Non-finite current pose confidence in {path}")
    return skeleton, placement, confidence_value


def build(*, dataset_root: Path, stage_b_root: Path, output_dir: Path, checkpoint: Path, label_mapping: Path, device_name: str, batch_size: int, max_episodes: int | None = None) -> Dict[str, Any]:
    started = time.perf_counter()
    stage_a_summary_path = dataset_root / "stage_a_summary.json"
    stage_a_summary = json.loads(stage_a_summary_path.read_text(encoding="utf-8"))
    model, device = _load_model(checkpoint, device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for split in SPLITS:
        stage_a_path = Path(stage_a_summary["episode_files"][split])
        stage_b_path = stage_b_root / "utility_labels" / f"{split}.jsonl"
        output_path = feature_dir / f"{split}.jsonl"
        count = 0
        pending: List[tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray, np.ndarray, float]] = []
        with output_path.open("w", encoding="utf-8") as target:
            for line_number, (episode, utility) in enumerate(_iter_pairs(stage_a_path, stage_b_path), 1):
                if max_episodes is not None and line_number > max_episodes:
                    break
                skeleton, placement, confidence = _load_pending(episode)
                pending.append((episode, utility, skeleton, placement, confidence))
                if len(pending) >= batch_size:
                    for row in _process_batch(pending, model, device):
                        target.write(json.dumps(row, separators=(",", ":")) + "\n")
                        count += 1
                    pending.clear()
            if pending:
                for row in _process_batch(pending, model, device):
                    target.write(json.dumps(row, separators=(",", ":")) + "\n")
                    count += 1
        counts[split] = count

    train_rows = (json.loads(line) for line in (feature_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    statistics = feature_statistics(train_rows)
    stats_path = output_dir / "stage_c_feature_stats.json"
    save_feature_statistics(stats_path, statistics)
    stage_b_summary_path = stage_b_root / "stage_b_summary.json"
    summary = {
        "protocol": "ACTIVEVIEW v11.5 Stage C current-conditioned features",
        "stage": "C",
        "status": "generated",
        "schema": schema_metadata(),
        "feature_files": {split: str((feature_dir / f"{split}.jsonl").resolve()) for split in SPLITS},
        "feature_file_sha256": {split: file_sha256(feature_dir / f"{split}.jsonl") for split in SPLITS},
        "feature_stats": str(stats_path.resolve()),
        "feature_stats_sha256": file_sha256(stats_path),
        "feature_file_counts": counts,
        "source_stage_a_summary": str(stage_a_summary_path.resolve()),
        "source_stage_a_summary_sha256": file_sha256(stage_a_summary_path),
        "source_stage_a_episode_sha256": {split: file_sha256(Path(stage_a_summary["episode_files"][split])) for split in SPLITS},
        "source_stage_b_summary": str(stage_b_summary_path.resolve()),
        "source_stage_b_summary_sha256": file_sha256(stage_b_summary_path),
        "source_stage_b_utility_sha256": {split: file_sha256(stage_b_root / "utility_labels" / f"{split}.jsonl") for split in SPLITS},
        "stgcn_checkpoint": str(checkpoint.resolve()), "stgcn_checkpoint_sha256": file_sha256(checkpoint),
        "label_mapping": str(label_mapping.resolve()), "label_mapping_sha256": file_sha256(label_mapping),
        "canonical_split_counts": {"train": 589, "val": 197, "test": 194},
        "current_feature_dim": CURRENT_FEATURE_DIM, "candidate_geometry_dim": 11,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "stage_c_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage_c_feature_summary": str(summary_path.resolve()), "feature_file_counts": counts}, ensure_ascii=False))
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--output-dir", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--label-mapping", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional per-split smoke-test limit")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    build(dataset_root=args.dataset_root, stage_b_root=args.stage_b_root, output_dir=args.output_dir, checkpoint=args.checkpoint, label_mapping=args.label_mapping, device_name=args.device, batch_size=args.batch_size, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
