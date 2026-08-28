#!/usr/bin/env python3
"""Independent Stage C feature/prediction provenance and leakage validator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.active_view.stage_c_metrics import summarize_stage_c_predictions
from ea_avs_mvp_v11.active_view.stage_c_features import schema_metadata
from ea_avs_mvp_v11.active_view.utility_label_builder import file_sha256
from ea_avs_mvp_v11.dataset.policy_split import load_policy_splits


SPLITS = ("train", "val", "test")
CANONICAL_COUNTS = {"train": 589, "val": 197, "test": 194}
FORBIDDEN = {"label_id", "action_label", "candidate_utility", "candidate_skeleton", "candidate_confidence", "candidate_log_probs", "candidate_entropy", "candidate_prediction", "gt_correctness", "viewpoint_id"}


def _compare(expected: Any, observed: Any, path: str, errors: List[str], tol: float = 1e-7) -> None:
    if len(errors) >= 100:
        return
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(expected) != set(observed):
            errors.append(f"metric_structure_mismatch:{path}"); return
        for key in expected:
            _compare(expected[key], observed[key], f"{path}.{key}", errors, tol)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isclose(float(expected), float(observed), rel_tol=0.0, abs_tol=tol):
            errors.append(f"metric_value_mismatch:{path}")
        return
    if expected != observed:
        errors.append(f"metric_value_mismatch:{path}")


def validate(*, dataset_root: Path, stage_b_root: Path, stage_c_root: Path, eval_summaries: List[Path] | None = None, report_path: Path | None = None) -> Dict[str, Any]:
    errors: List[str] = []
    feature_summary_path = stage_c_root / "stage_c_feature_summary.json"
    try:
        feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
        stage_a_summary_path = dataset_root / "stage_a_summary.json"
        stage_b_summary_path = stage_b_root / "stage_b_summary.json"
        if feature_summary.get("source_stage_a_summary_sha256") != file_sha256(stage_a_summary_path):
            errors.append("stage_a_summary_hash_mismatch")
        if feature_summary.get("source_stage_b_summary_sha256") != file_sha256(stage_b_summary_path):
            errors.append("stage_b_summary_hash_mismatch")
        splits = load_policy_splits(dataset_root / "splits")
        actual_split_counts = {split: len(splits[split]) for split in SPLITS}
        if actual_split_counts != CANONICAL_COUNTS:
            errors.append(f"noncanonical_policy_split:{actual_split_counts}")
        if feature_summary.get("canonical_split_counts") != CANONICAL_COUNTS:
            errors.append("noncanonical_split_counts")
        stage_a_summary = json.loads(stage_a_summary_path.read_text(encoding="utf-8"))
        expected_episode_counts: Dict[str, int] = {}
        for split in SPLITS:
            episode_path = Path(stage_a_summary["episode_files"][split])
            with episode_path.open(encoding="utf-8") as handle:
                expected_episode_counts[split] = sum(1 for line in handle if line.strip())
        if expected_episode_counts != feature_summary.get("feature_file_counts"):
            errors.append(f"feature_episode_count_mismatch:{expected_episode_counts}")
        expected_utility_counts: Dict[str, int] = {}
        for split in SPLITS:
            utility_path = stage_b_root / "utility_labels" / f"{split}.jsonl"
            with utility_path.open(encoding="utf-8") as handle:
                expected_utility_counts[split] = sum(1 for line in handle if line.strip())
        if expected_utility_counts != expected_episode_counts:
            errors.append(f"stage_a_stage_b_count_mismatch:{expected_episode_counts}:{expected_utility_counts}")
        for split in SPLITS:
            episode_path = Path(stage_a_summary["episode_files"][split])
            utility_path = stage_b_root / "utility_labels" / f"{split}.jsonl"
            if feature_summary.get("source_stage_a_episode_sha256", {}).get(split) != file_sha256(episode_path):
                errors.append(f"stage_a_episode_hash_mismatch:{split}")
            if feature_summary.get("source_stage_b_utility_sha256", {}).get(split) != file_sha256(utility_path):
                errors.append(f"stage_b_utility_hash_mismatch:{split}")
        schema = feature_summary.get("schema")
        if not isinstance(schema, Mapping):
            errors.append("missing_feature_schema")
        else:
            forbidden_inputs = FORBIDDEN.intersection(set(schema.get("input_whitelist", [])))
            if forbidden_inputs:
                errors.append(f"forbidden_feature_whitelist:{sorted(forbidden_inputs)}")
            if schema != schema_metadata():
                errors.append("feature_schema_mismatch")
        for split in SPLITS:
            path = Path(feature_summary["feature_files"][split])
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            if count != int(feature_summary["feature_file_counts"][split]):
                errors.append(f"feature_count_mismatch:{split}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                current = np.asarray(row["current_feature"], dtype=np.float32)
                geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
                if current.shape != (275,) or geometry.ndim != 2 or geometry.shape[1] != 11 or not np.isfinite(current).all() or not np.isfinite(geometry).all():
                    errors.append(f"invalid_feature_values:{split}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"feature_validation_failed:{error}")

    if eval_summaries:
        for summary_path in eval_summaries:
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                model_type = str(summary["model_type"])
                expected_feature_summary = stage_c_root / "stage_c_feature_summary.json"
                if summary.get("feature_summary_sha256") != file_sha256(expected_feature_summary):
                    errors.append(f"feature_summary_hash_mismatch:{model_type}")
                if summary.get("source_stage_a_summary_sha256") != file_sha256(dataset_root / "stage_a_summary.json"):
                    errors.append(f"stage_a_summary_hash_mismatch:{model_type}")
                if summary.get("source_stage_b_summary_sha256") != file_sha256(stage_b_root / "stage_b_summary.json"):
                    errors.append(f"stage_b_summary_hash_mismatch:{model_type}")
                checkpoint_path = Path(summary["checkpoint"])
                if summary.get("checkpoint_sha256") != file_sha256(checkpoint_path):
                    errors.append(f"predictor_checkpoint_hash_mismatch:{model_type}")
                label_mapping_path = Path(summary["label_mapping"])
                if summary.get("label_mapping_sha256") != file_sha256(label_mapping_path):
                    errors.append(f"label_mapping_hash_mismatch:{model_type}")
                if summary.get("stgcn_checkpoint_sha256") != feature_summary.get("stgcn_checkpoint_sha256"):
                    errors.append(f"stgcn_checkpoint_hash_mismatch:{model_type}")
                if summary.get("canonical_split_counts") != CANONICAL_COUNTS:
                    errors.append(f"noncanonical_eval_split_counts:{model_type}")
                feature_summary = json.loads((stage_c_root / "stage_c_feature_summary.json").read_text(encoding="utf-8"))
                if summary.get("feature_file_counts") != feature_summary.get("feature_file_counts"):
                    errors.append(f"feature_file_count_mismatch:{model_type}")
                for split in ("val", "test"):
                    prediction_path = Path(summary["prediction_files"][split])
                    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if not isinstance(summary.get("evaluation_only_fields"), list):
                        errors.append(f"missing_evaluation_only_field_declaration:{model_type}")
                    recomputed = summarize_stage_c_predictions(rows, summary["categories"])
                    metric_errors: List[str] = []
                    _compare(recomputed, summary["metrics"][split], f"{model_type}.{split}", metric_errors)
                    errors.extend(metric_errors)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"evaluation_validation_failed:{summary_path}:{error}")
    combined_summary_path = stage_c_root / "stage_c_summary.json"
    if combined_summary_path.exists():
        try:
            combined = json.loads(combined_summary_path.read_text(encoding="utf-8"))
            if combined.get("canonical_split_counts") != CANONICAL_COUNTS:
                errors.append("combined_summary_noncanonical_split_counts")
            if combined.get("feature_summary_sha256") != file_sha256(feature_summary_path):
                errors.append("combined_summary_feature_hash_mismatch")
            if combined.get("scope", {}).get("stage_d_started") is not False:
                errors.append("combined_summary_stage_d_scope_violation")
            if eval_summaries:
                expected_models = {Path(path).name.split("_evaluation_summary.json")[0] for path in eval_summaries}
                if set(combined.get("evaluations", {})) != expected_models:
                    errors.append("combined_summary_model_set_mismatch")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"combined_summary_validation_failed:{error}")
    report = {"protocol": "ACTIVEVIEW v11.5 Stage C validator", "stage": "C", "passed": not errors, "error_count": len(errors), "errors": errors[:100]}
    output = report_path or stage_c_root / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    from ea_avs_mvp_v11.core.paths import get_data_root
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--stage-c-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--eval-summary", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = validate(dataset_root=args.dataset_root, stage_b_root=args.stage_b_root, stage_c_root=args.stage_c_root, eval_summaries=args.eval_summary, report_path=args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
