#!/usr/bin/env python3
"""Build EXP003 geometry features from the frozen Stage C-v0 cache.

This is a feature-only transformation.  It does not invoke Habitat, pose
estimation, VideoPose3D, or ST-GCN inference; the 275-D current feature and
all supervision fields are copied from the accepted v0 cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_dataset import feature_statistics, save_feature_statistics
from activeview.active_view.stage_c_features import (
    BASE_CANDIDATE_GEOMETRY_DIM,
    CANDIDATE_GEOMETRY_DIM,
    RELATIVE_CANDIDATE_GEOMETRY_DIM,
    candidate_set_relative_features,
    schema_metadata,
)
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


SPLITS = ("train", "val", "test")


def _transform_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
    ids = list(row["candidate_viewpoint_ids"])
    expected_shape = (len(ids), BASE_CANDIDATE_GEOMETRY_DIM)
    if geometry.shape != expected_shape or not np.isfinite(geometry).all():
        raise ValueError(
            f"{row.get('episode_id', '<unknown>')} geometry shape {geometry.shape} "
            f"!= {expected_shape} or contains non-finite values"
        )
    relative = candidate_set_relative_features(geometry)
    transformed = dict(row)
    transformed["candidate_geometry"] = np.concatenate(
        [geometry, relative], axis=1
    ).tolist()
    return transformed


def build(*, source_feature_root: Path, output_dir: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    source_summary_path = source_feature_root / "stage_c_feature_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_schema = source_summary.get("schema", {})
    if int(source_schema.get("candidate_geometry_dim", -1)) != CANDIDATE_GEOMETRY_DIM:
        raise ValueError("EXP003 expects the frozen 11-D Stage C-v0 geometry cache")
    source_feature_dir = source_feature_root / "features"
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    train_rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        source_path = source_feature_dir / f"{split}.jsonl"
        target_path = feature_dir / f"{split}.jsonl"
        count = 0
        with source_path.open(encoding="utf-8") as source, target_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = _transform_row(json.loads(line))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid feature row at {source_path}:{line_number}: {error}") from error
                target.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                count += 1
                if split == "train":
                    train_rows.append(row)
        counts[split] = count

    statistics = feature_statistics(train_rows)
    stats_path = output_dir / "stage_c_feature_stats.json"
    save_feature_statistics(stats_path, statistics)
    schema = schema_metadata(include_relative_features=True)
    summary: Dict[str, Any] = {
        "protocol": "ACTIVEVIEW v11.5 EXP003 relative candidate geometry features",
        "stage": "C",
        "status": "generated",
        "schema": schema,
        "feature_files": {
            split: str((feature_dir / f"{split}.jsonl").resolve()) for split in SPLITS
        },
        "feature_file_sha256": {
            split: file_sha256(feature_dir / f"{split}.jsonl") for split in SPLITS
        },
        "feature_stats": str(stats_path.resolve()),
        "feature_stats_sha256": file_sha256(stats_path),
        "feature_file_counts": counts,
        "source_stage_c_v0_feature_summary": str(source_summary_path.resolve()),
        "source_stage_c_v0_feature_summary_sha256": file_sha256(source_summary_path),
        "source_stage_a_summary": source_summary.get("source_stage_a_summary"),
        "source_stage_a_summary_sha256": source_summary.get("source_stage_a_summary_sha256"),
        "source_stage_a_episode_sha256": source_summary.get("source_stage_a_episode_sha256"),
        "source_stage_b_summary": source_summary.get("source_stage_b_summary"),
        "source_stage_b_summary_sha256": source_summary.get("source_stage_b_summary_sha256"),
        "source_stage_b_utility_sha256": source_summary.get("source_stage_b_utility_sha256"),
        "stgcn_checkpoint": source_summary.get("stgcn_checkpoint"),
        "stgcn_checkpoint_sha256": source_summary.get("stgcn_checkpoint_sha256"),
        "label_mapping": source_summary.get("label_mapping"),
        "label_mapping_sha256": source_summary.get("label_mapping_sha256"),
        "canonical_split_counts": source_summary.get("canonical_split_counts"),
        "current_feature_dim": schema["current_feature_dim"],
        "candidate_geometry_dim": schema["candidate_geometry_dim"],
        "base_candidate_geometry_dim": BASE_CANDIDATE_GEOMETRY_DIM,
        "added_relative_geometry_dim": RELATIVE_CANDIDATE_GEOMETRY_DIM,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "stage_c_feature_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-feature-root",
        type=Path,
        default=data_root / "datasets/policy_v11_5/stage_c",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "datasets/policy_v11_5/stage_c_v1/EXP003_relative_geometry",
    )
    args = parser.parse_args()
    result = build(source_feature_root=args.source_feature_root, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "feature_file_counts": result["feature_file_counts"], "candidate_geometry_dim": result["candidate_geometry_dim"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
