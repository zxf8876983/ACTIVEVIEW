import json
from pathlib import Path

import numpy as np

from ea_avs_mvp_v11.active_view.utility_label_builder import (
    build_utility_record,
    file_sha256,
    summarize_utility_records,
)
from ea_avs_mvp_v11.scripts.validate_stage_b import validate


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    dataset_root = tmp_path / "dataset"
    stage_b_root = dataset_root / "stage_b"
    dataset_root.mkdir()
    (stage_b_root / "utility_labels").mkdir(parents=True)
    scene_ids = [f"scene-{index:02d}" for index in range(21)]
    episode = {
        "episode_id": "episode-1",
        "record_id": "record-1",
        "policy_split": "train",
        "scene_id": scene_ids[0],
        "region": "bedroom",
        "label_id": 0,
        "current_view": {"viewpoint_id": 0},
        "candidate_pool": [
            {"viewpoint_id": 1, "geodesic_distance_m": 1.0},
            {"viewpoint_id": 2, "geodesic_distance_m": 2.0},
        ],
    }
    log_probs = {
        0: np.log(np.asarray([0.6, 0.4], dtype=np.float64)),
        1: np.log(np.asarray([0.9, 0.1], dtype=np.float64)),
        2: np.log(np.asarray([0.5, 0.5], dtype=np.float64)),
    }
    record = build_utility_record(episode, log_probs)
    episode_files = {}
    for split in ("train", "val", "test"):
        path = dataset_root / f"{split}_episodes.jsonl"
        path.write_text(
            (json.dumps(episode) + "\n") if split == "train" else "",
            encoding="utf-8",
        )
        episode_files[split] = str(path.resolve())
        (stage_b_root / "utility_labels" / f"{split}.jsonl").write_text(
            (json.dumps(record) + "\n") if split == "train" else "",
            encoding="utf-8",
        )
    stage_a_summary = {
        "scene_ids_used": scene_ids,
        "episode_files": episode_files,
        "policy_split": {"counts": {"train": 1, "val": 0, "test": 0}},
    }
    stage_a_summary_path = dataset_root / "stage_a_summary.json"
    stage_a_summary_path.write_text(json.dumps(stage_a_summary), encoding="utf-8")
    mapping_path = dataset_root / "mapping.json"
    mapping_path.write_text(json.dumps({"class0": 0, "class1": 1}), encoding="utf-8")
    checkpoint_path = dataset_root / "checkpoint.pth"
    checkpoint_path.write_bytes(b"fixture-checkpoint")
    summary = {
        "stage": "B",
        "supervision_only": True,
        "source_stage_a_summary_sha256": file_sha256(stage_a_summary_path),
        "source_episode_file_sha256": {split: file_sha256(Path(path)) for split, path in episode_files.items()},
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "episode_counts": {"train": 1, "val": 0, "test": 0},
        "candidate_pair_counts": {"train": 2, "val": 0, "test": 0},
        "official_scene_count": 21,
        "scene_ids": scene_ids,
        "categories": ["class0", "class1"],
        "label_mapping": str(mapping_path.resolve()),
        "label_mapping_sha256": file_sha256(mapping_path),
        "stgcn_checkpoint": str(checkpoint_path.resolve()),
        "stgcn_checkpoint_sha256": file_sha256(checkpoint_path),
        "numeric_tolerance": 1e-5,
        "metrics": summarize_utility_records({"train": [record], "val": [], "test": []}, ["class0", "class1"]),
    }
    (stage_b_root / "stage_b_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return dataset_root, stage_b_root, summary


def test_validator_recomputes_metrics_and_accepts_matching_fixture(tmp_path):
    dataset_root, stage_b_root, _ = _make_fixture(tmp_path)
    report = validate(dataset_root, stage_b_root)
    assert report["passed"]


def test_validator_rejects_geodesic_mismatch(tmp_path):
    dataset_root, stage_b_root, _ = _make_fixture(tmp_path)
    path = stage_b_root / "utility_labels" / "train.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["candidates"][0]["geodesic_distance_m"] = 9.0
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = validate(dataset_root, stage_b_root)
    assert not report["passed"]
    assert any("candidate_geodesic_mismatch" in str(error) for error in report["errors"])


def test_validator_rejects_metric_corruption_and_stale_source_hash(tmp_path):
    dataset_root, stage_b_root, summary = _make_fixture(tmp_path)
    summary["metrics"]["train"]["policies"]["NoMove"]["accuracy"] = 0.123
    (stage_b_root / "stage_b_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    report = validate(dataset_root, stage_b_root)
    assert not report["passed"]
    assert any("metrics_value_mismatch" in str(error) for error in report["errors"])

    stage_a_path = dataset_root / "stage_a_summary.json"
    stage_a_path.write_text(stage_a_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report = validate(dataset_root, stage_b_root)
    assert not report["passed"]
    assert any(error.get("reason") == "source_stage_a_summary_hash_mismatch" for error in report["errors"])
