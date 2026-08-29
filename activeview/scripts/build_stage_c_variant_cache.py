#!/usr/bin/env python3
"""Build one independent Stage C-v1 geometry cache from frozen Stage C-v0 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_dataset import feature_statistics, save_feature_statistics
from activeview.active_view.stage_c_experiment_features import (
    VARIANT_NAMES,
    transform_geometry,
    variant_schema,
)
from activeview.active_view.stage_c_features import BASE_CANDIDATE_GEOMETRY_DIM
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


SPLITS = ("train", "val", "test")


def build(*, source_feature_root: Path, output_dir: Path, variant: str) -> Dict[str, Any]:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"variant must be one of {VARIANT_NAMES}")
    started = time.perf_counter()
    source_summary_path = source_feature_root / "stage_c_feature_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_schema = source_summary.get("schema", {})
    if int(source_schema.get("candidate_geometry_dim", -1)) != BASE_CANDIDATE_GEOMETRY_DIM:
        raise ValueError("Experiment variants must start from the frozen 11-D Stage C-v0 cache")
    source_dir = source_feature_root / "features"
    target_dir = output_dir / "features"
    target_dir.mkdir(parents=True, exist_ok=True)
    train_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for split in SPLITS:
        source_path = source_dir / f"{split}.jsonl"
        target_path = target_dir / f"{split}.jsonl"
        count = 0
        with source_path.open(encoding="utf-8") as source, target_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
                expected = (len(row["candidate_viewpoint_ids"]), BASE_CANDIDATE_GEOMETRY_DIM)
                if geometry.shape != expected:
                    raise ValueError(f"{source_path}:{line_number} geometry shape {geometry.shape} != {expected}")
                transformed = dict(row)
                transformed["candidate_geometry"] = transform_geometry(geometry, variant).tolist()
                target.write(json.dumps(transformed, separators=(",", ":"), ensure_ascii=False) + "\n")
                count += 1
                if split == "train":
                    train_rows.append(transformed)
        counts[split] = count
    stats_path = output_dir / "stage_c_feature_stats.json"
    save_feature_statistics(stats_path, feature_statistics(train_rows))
    schema = variant_schema(variant)
    summary: Dict[str, Any] = {
        "protocol": f"ACTIVEVIEW v11.5 Stage C-v1 {variant} geometry",
        "stage": "C",
        "status": "generated",
        "schema": schema,
        "feature_files": {split: str((target_dir / f"{split}.jsonl").resolve()) for split in SPLITS},
        "feature_file_sha256": {split: file_sha256(target_dir / f"{split}.jsonl") for split in SPLITS},
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
        "current_feature_dim": 275,
        "candidate_geometry_dim": schema["candidate_geometry_dim"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output_dir / "stage_c_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANT_NAMES, required=True)
    parser.add_argument("--source-feature-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(source_feature_root=args.source_feature_root, output_dir=args.output_dir, variant=args.variant)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "variant": args.variant, "feature_file_counts": result["feature_file_counts"], "candidate_geometry_dim": result["candidate_geometry_dim"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
