#!/usr/bin/env python3
"""Build Stage B offline Utility labels from frozen ST-GCN predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.action_recognition.st_gcn_model import STGCN
from ea_avs_mvp_v11.active_view.utility_label_builder import (
    build_utility_record,
    file_sha256,
    summarize_utility_records,
)
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition


LOGGER = logging.getLogger("activeview.stage_b_builder")
SPLITS = ("train", "val", "test")
LEGACY_MINIVAL_SCENE = "00800-TEEsavR23oF"


def _load_model(checkpoint: Path, category_count: int, device_name: str) -> tuple[STGCN, torch.device]:
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = STGCN(
        in_channels=3,
        num_classes=category_count,
        graph_strategy="spatial",
        edge_importance_weighting=True,
        skel_def=get_skeleton_definition(backend="h36m_17"),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, device


def _load_archive_predictions(
    path: Path,
    model: STGCN,
    device: torch.device,
    batch_size: int,
    category_count: int,
) -> Dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        viewpoint_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
    if skeleton.shape != (32, 3, 30, 17):
        raise ValueError(f"Unexpected skeleton shape {skeleton.shape} in {path}")
    if viewpoint_ids.shape != (32,) or len(set(viewpoint_ids.tolist())) != 32:
        raise ValueError(f"Invalid viewpoint IDs in {path}")
    finite = np.isfinite(skeleton).all(axis=(1, 2, 3))
    if not finite.any():
        raise ValueError(f"No finite viewpoint skeleton in {path}")
    result: Dict[int, np.ndarray] = {}
    valid_indices = np.flatnonzero(finite)
    with torch.inference_mode():
        for start in range(0, len(valid_indices), batch_size):
            indices = valid_indices[start:start + batch_size]
            batch = torch.from_numpy(skeleton[indices]).to(device=device, dtype=torch.float32).unsqueeze(-1)
            log_probs = torch.log_softmax(model(batch), dim=-1).cpu().numpy().astype(np.float64)
            if log_probs.shape != (len(indices), category_count):
                raise ValueError(f"Unexpected ST-GCN output shape {log_probs.shape} for {path}")
            for index, values in zip(indices.tolist(), log_probs):
                result[int(viewpoint_ids[index])] = values
    return result


def _read_stage_a_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage A summary must be an object: {path}")
    scenes = {str(item) for item in payload.get("scene_ids_used", [])}
    if len(scenes) != 21 or LEGACY_MINIVAL_SCENE in scenes:
        raise ValueError("Stage A summary does not describe the canonical 21 HM3D-train scenes")
    return payload


def build(
    *, dataset_root: Path, output_dir: Path, checkpoint: Path, label_mapping: Path,
    device_name: str, inference_batch_size: int, max_episodes: int | None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    stage_a_summary_path = dataset_root / "stage_a_summary.json"
    stage_a_summary = _read_stage_a_summary(stage_a_summary_path)
    mapping = json.loads(label_mapping.read_text(encoding="utf-8"))
    categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    model, device = _load_model(checkpoint, len(categories), device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_cache: Dict[str, Dict[int, np.ndarray]] = {}
    records_by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
    output_files: Dict[str, str] = {}
    source_episode_files: Dict[str, str] = {}
    source_episode_file_hashes: Dict[str, str] = {}
    for split in SPLITS:
        source_path = Path(stage_a_summary["episode_files"][split])
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_episode_files[split] = str(source_path.resolve())
        source_episode_file_hashes[split] = file_sha256(source_path)
        output_path = output_dir / "utility_labels" / f"{split}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_files[split] = str(output_path.resolve())
        with source_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, 1):
                if max_episodes is not None and line_number > max_episodes:
                    break
                episode = json.loads(line)
                scene_id = str(episode["scene_id"])
                if scene_id not in {str(item) for item in stage_a_summary["scene_ids_used"]}:
                    raise ValueError(f"Episode references non-canonical scene {scene_id}")
                archive_path = str(episode["current_view"]["skeleton_source_path"])
                if archive_path not in archive_cache:
                    LOGGER.info("Caching ST-GCN predictions for %s", archive_path)
                    archive_cache[archive_path] = _load_archive_predictions(
                        Path(archive_path), model, device, inference_batch_size, len(categories),
                    )
                record = build_utility_record(episode, archive_cache[archive_path])
                records_by_split[split].append(record)
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        LOGGER.info("Built Stage B %s records: %d", split, len(records_by_split[split]))

    policy_counts = stage_a_summary.get("policy_split", {}).get("counts", {})
    episode_counts = {split: len(records_by_split[split]) for split in SPLITS}
    pair_counts = {
        split: sum(len(record["candidates"]) for record in records_by_split[split])
        for split in SPLITS
    }
    summary = {
        "protocol": "ACTIVEVIEW v11.5 Stage B offline utility labels",
        "stage": "B",
        "status": "generated",
        "supervision_only": True,
        "source_stage_a_summary": str(stage_a_summary_path.resolve()),
        "source_stage_a_summary_sha256": file_sha256(stage_a_summary_path),
        "source_episode_files": source_episode_files,
        "source_episode_file_sha256": source_episode_file_hashes,
        "utility_label_files": output_files,
        "official_scene_count": len(stage_a_summary["scene_ids_used"]),
        "scene_ids": [str(item) for item in stage_a_summary["scene_ids_used"]],
        "split_counts": {split: int(policy_counts.get(split, 0)) for split in SPLITS},
        "episode_counts": episode_counts,
        "candidate_pair_counts": pair_counts,
        "categories": categories,
        "label_mapping": str(label_mapping.resolve()),
        "label_mapping_sha256": file_sha256(label_mapping),
        "stgcn_checkpoint": str(checkpoint.resolve()),
        "stgcn_checkpoint_sha256": file_sha256(checkpoint),
        "device": str(device),
        "inference_archive_count": len(archive_cache),
        "utility_definition": "log_softmax(candidate_logits)[label_id] - log_softmax(current_logits)[label_id]",
        "oracle_definition": "argmax candidate utility; tie-break by smaller geodesic distance then viewpoint_id",
        "safe_oracle_definition": "stay at current when max candidate utility <= 0, otherwise choose Candidate Oracle",
        "numeric_tolerance": 1e-5,
        "near_zero_tolerance": 1e-6,
        "metrics": summarize_utility_records(records_by_split, categories),
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "stage_b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"stage_b_summary": str(summary_path.resolve()), "episode_counts": episode_counts, "candidate_pair_counts": pair_counts}, ensure_ascii=False))
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--output-dir", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--checkpoint", type=Path, default=data_root / "checkpoints/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled/stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth")
    parser.add_argument("--label-mapping", type=Path, default=data_root / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/label_mapping.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional per-split smoke-test limit")
    args = parser.parse_args()
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    build(
        dataset_root=args.dataset_root, output_dir=args.output_dir, checkpoint=args.checkpoint,
        label_mapping=args.label_mapping, device_name=args.device,
        inference_batch_size=args.inference_batch_size, max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
