#!/usr/bin/env python3
"""Assemble the reproducible Stage C implementation/evaluation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_features import schema_metadata
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


def build_summary(*, stage_c_root: Path, checkpoint_root: Path, output: Path) -> Dict[str, Any]:
    feature_summary_path = stage_c_root / "stage_c_feature_summary.json"
    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    evaluations: Dict[str, Any] = {}
    training: Dict[str, Any] = {}
    for model_type in ("pairwise_mlp", "set_ranker"):
        eval_path = stage_c_root / "evaluations" / f"{model_type}_evaluation_summary.json"
        train_path = checkpoint_root / f"{model_type}_training_summary.json"
        evaluations[model_type] = json.loads(eval_path.read_text(encoding="utf-8"))
        training[model_type] = json.loads(train_path.read_text(encoding="utf-8"))
    diagnostics_path = stage_c_root / "stage_b_diagnostics.json"
    summary: Dict[str, Any] = {
        "protocol": "ACTIVEVIEW v11.5 Stage C current-conditioned utility prediction",
        "stage": "C",
        "status": "evaluated_ready_for_user_review",
        "scope": {
            "stage_a_unchanged": True,
            "stage_b_unchanged": True,
            "stage_d_started": False,
            "body_yaw_used": False,
            "movement_cost_penalty_used": False,
            "future_candidate_perception_used_as_input": False,
            "stgcn_frozen": True,
        },
        "canonical_split_counts": feature_summary["canonical_split_counts"],
        "feature_schema": feature_summary.get("schema", schema_metadata()),
        "feature_summary": str(feature_summary_path.resolve()),
        "feature_summary_sha256": file_sha256(feature_summary_path),
        "source_stage_a_summary_sha256": feature_summary["source_stage_a_summary_sha256"],
        "source_stage_a_episode_sha256": feature_summary["source_stage_a_episode_sha256"],
        "source_stage_b_summary_sha256": feature_summary["source_stage_b_summary_sha256"],
        "source_stage_b_utility_sha256": feature_summary["source_stage_b_utility_sha256"],
        "stgcn_checkpoint_sha256": feature_summary["stgcn_checkpoint_sha256"],
        "label_mapping_sha256": feature_summary["label_mapping_sha256"],
        "feature_file_counts": feature_summary["feature_file_counts"],
        "diagnostics": str(diagnostics_path.resolve()) if diagnostics_path.exists() else None,
        "training": {
            model: {
                "parameter_count": data["parameter_count"],
                "selected_epoch": data["selected_epoch"],
                "checkpoint": data["checkpoint"],
                "checkpoint_sha256": data["checkpoint_sha256"],
                "checkpoint_selection_metric": data["checkpoint_selection_metric"],
                "sampler": data["sampler"],
                "loss": data["loss"],
            }
            for model, data in training.items()
        },
        "evaluations": {
            model: {
                "summary": str((stage_c_root / "evaluations" / f"{model}_evaluation_summary.json").resolve()),
                "summary_sha256": file_sha256(stage_c_root / "evaluations" / f"{model}_evaluation_summary.json"),
                "metrics": data["metrics"],
                "prediction_files": data["prediction_files"],
            }
            for model, data in evaluations.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--checkpoint-root", type=Path, default=data_root / "checkpoints/stage_c")
    parser.add_argument("--output", type=Path, default=data_root / "datasets/policy_v11_5/stage_c/stage_c_summary.json")
    args = parser.parse_args()
    print(json.dumps(build_summary(stage_c_root=args.stage_c_root, checkpoint_root=args.checkpoint_root, output=args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
