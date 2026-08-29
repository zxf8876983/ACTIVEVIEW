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
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_c_metrics import summarize_stage_c_predictions
from activeview.active_view.stage_c_features import candidate_geometry_matrix, schema_metadata
from activeview.active_view.utility_label_builder import file_sha256
from activeview.dataset.policy_split import load_policy_splits


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


def _expected_candidate_id(predicted: List[float], ids: List[int], geodesic: List[float]) -> int:
    if not predicted or len(predicted) != len(ids) or len(ids) != len(geodesic):
        raise ValueError("prediction/candidate arrays are not aligned")
    index = min(range(len(ids)), key=lambda item: (-float(predicted[item]), float(geodesic[item]), int(ids[item])))
    return int(ids[index])


def _validate_independent_decision(row: Mapping[str, Any], stage_b: Mapping[str, Any], contract: Mapping[str, Any], model_type: str, split: str, errors: List[str]) -> None:
    episode_id = str(row["episode_id"])
    ids = [int(value) for value in row.get("candidate_viewpoint_ids", [])]
    predicted = [float(value) for value in row.get("predicted_utilities", [])]
    geodesic = [float(value) for value in contract["candidate_geodesic"]]
    if len(predicted) != len(ids) or not predicted or not np.isfinite(np.asarray(predicted, dtype=np.float64)).all():
        errors.append(f"invalid_predicted_utilities:{model_type}:{split}:{episode_id}")
        return
    expected_candidate_id = _expected_candidate_id(predicted, ids, geodesic)
    if int(row.get("predicted_candidate_viewpoint_id", -1)) != expected_candidate_id:
        errors.append(f"decision_candidate_id_mismatch:{model_type}:{split}:{episode_id}")
    expected_stays = max(predicted) <= 0.0
    if bool(row.get("predicted_stays")) != expected_stays:
        errors.append(f"decision_stay_mismatch:{model_type}:{split}:{episode_id}")
    expected_action = "stay" if expected_stays else f"candidate:{expected_candidate_id}"
    if row.get("predicted_action") != expected_action:
        errors.append(f"decision_action_mismatch:{model_type}:{split}:{episode_id}")
    candidates = {int(item["viewpoint_id"]): item for item in stage_b["candidates"]}
    current = stage_b["current"]
    if int(row.get("current_predicted_label_id", -1)) != int(current["predicted_label_id"]):
        errors.append(f"current_prediction_mismatch:{model_type}:{split}:{episode_id}")
    if not math.isclose(
        float(row.get("current_entropy", float("nan"))),
        float(current["entropy"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        errors.append(f"current_entropy_mismatch:{model_type}:{split}:{episode_id}")
    selected = current if expected_stays else candidates[expected_candidate_id]
    expected_selected_utility = 0.0 if expected_stays else float(selected["utility"])
    if not math.isclose(float(row.get("selected_true_utility", float("nan"))), expected_selected_utility, rel_tol=0.0, abs_tol=1e-6):
        errors.append(f"selected_true_utility_mismatch:{model_type}:{split}:{episode_id}")
    if int(row.get("selected_predicted_label_id", -1)) != int(selected["predicted_label_id"]):
        errors.append(f"selected_prediction_mismatch:{model_type}:{split}:{episode_id}")
    if not math.isclose(float(row.get("selected_entropy", float("nan"))), float(selected["entropy"]), rel_tol=0.0, abs_tol=1e-6):
        errors.append(f"selected_entropy_mismatch:{model_type}:{split}:{episode_id}")
    oracle = stage_b["oracle"]
    candidate_oracle_id = int(oracle["candidate_oracle_viewpoint_id"])
    safe_oracle_id = int(oracle["safe_oracle_viewpoint_id"])
    candidate_oracle = candidates[candidate_oracle_id]
    safe_oracle = current if bool(oracle["safe_oracle_stays"]) else candidates[safe_oracle_id]
    checks = {
        "candidate_oracle_viewpoint_id": candidate_oracle_id,
        "candidate_oracle_predicted_label_id": int(candidate_oracle["predicted_label_id"]),
        "safe_oracle_viewpoint_id": safe_oracle_id,
        "safe_oracle_predicted_label_id": int(safe_oracle["predicted_label_id"]),
    }
    for field, expected in checks.items():
        if int(row.get(field, -1)) != expected:
            errors.append(f"oracle_{field}_mismatch:{model_type}:{split}:{episode_id}")
    if bool(row.get("safe_oracle_stays")) != bool(oracle["safe_oracle_stays"]):
        errors.append(f"oracle_safe_stay_mismatch:{model_type}:{split}:{episode_id}")
    expected_safe_action = "stay" if bool(oracle["safe_oracle_stays"]) else f"candidate:{safe_oracle_id}"
    if row.get("safe_oracle_action") != expected_safe_action:
        errors.append(f"oracle_safe_action_mismatch:{model_type}:{split}:{episode_id}")
    for field, expected in {
        "candidate_oracle_entropy": float(candidate_oracle["entropy"]),
        "safe_oracle_entropy": float(safe_oracle["entropy"]),
        "safe_oracle_utility": float(oracle["safe_oracle_utility"]),
        "regret": float(oracle["safe_oracle_utility"]) - expected_selected_utility,
    }.items():
        if not math.isclose(float(row.get(field, float("nan"))), expected, rel_tol=0.0, abs_tol=1e-6):
            errors.append(f"decision_{field}_mismatch:{model_type}:{split}:{episode_id}")


def validate(*, dataset_root: Path, stage_b_root: Path, stage_c_root: Path, eval_summaries: List[Path] | None = None, report_path: Path | None = None) -> Dict[str, Any]:
    errors: List[str] = []
    feature_contracts: Dict[str, Dict[str, Dict[str, Any]]] = {split: {} for split in SPLITS}
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
            feature_path = Path(feature_summary["feature_files"][split])
            if feature_summary.get("source_stage_a_episode_sha256", {}).get(split) != file_sha256(episode_path):
                errors.append(f"stage_a_episode_hash_mismatch:{split}")
            if feature_summary.get("source_stage_b_utility_sha256", {}).get(split) != file_sha256(utility_path):
                errors.append(f"stage_b_utility_hash_mismatch:{split}")
            if feature_summary.get("feature_file_sha256", {}).get(split) != file_sha256(feature_path):
                errors.append(f"feature_file_hash_mismatch:{split}")
        stats_path = Path(feature_summary["feature_stats"])
        if feature_summary.get("feature_stats_sha256") != file_sha256(stats_path):
            errors.append("feature_stats_hash_mismatch")
        schema = feature_summary.get("schema")
        include_relative_features = False
        if not isinstance(schema, Mapping):
            errors.append("missing_feature_schema")
        else:
            include_relative_features = bool(schema.get("relative_geometry_features", False))
            forbidden_inputs = FORBIDDEN.intersection(set(schema.get("input_whitelist", [])))
            if forbidden_inputs:
                errors.append(f"forbidden_feature_whitelist:{sorted(forbidden_inputs)}")
            if schema != schema_metadata(include_relative_features=include_relative_features):
                errors.append("feature_schema_mismatch")
        for split in SPLITS:
            stage_a_path = Path(stage_a_summary["episode_files"][split])
            stage_b_path = stage_b_root / "utility_labels" / f"{split}.jsonl"
            feature_path = Path(feature_summary["feature_files"][split])
            seen_ids: set[str] = set()
            count = 0
            archive_placements: Dict[str, np.ndarray] = {}
            with stage_a_path.open(encoding="utf-8") as stage_a_handle, stage_b_path.open(encoding="utf-8") as stage_b_handle, feature_path.open(encoding="utf-8") as feature_handle:
                while True:
                    stage_a_line = stage_a_handle.readline()
                    stage_b_line = stage_b_handle.readline()
                    feature_line = feature_handle.readline()
                    if not stage_a_line and not stage_b_line and not feature_line:
                        break
                    if not stage_a_line or not stage_b_line or not feature_line:
                        errors.append(f"stage_alignment_length_mismatch:{split}")
                        break
                    episode = json.loads(stage_a_line)
                    utility = json.loads(stage_b_line)
                    row = json.loads(feature_line)
                    episode_id = str(episode["episode_id"])
                    if episode_id != str(utility.get("episode_id")) or episode_id != str(row.get("episode_id")):
                        errors.append(f"stage_alignment_id_mismatch:{split}:{count}")
                    if episode_id in seen_ids:
                        errors.append(f"duplicate_feature_episode_id:{split}:{episode_id}")
                    seen_ids.add(episode_id)
                    count += 1
                    expected_candidates = episode["candidate_pool"]
                    stage_b_candidates = utility.get("candidates", [])
                    expected_ids = [int(item["viewpoint_id"]) for item in expected_candidates]
                    stage_b_ids = [int(item["viewpoint_id"]) for item in stage_b_candidates]
                    observed_ids = [int(item) for item in row.get("candidate_viewpoint_ids", [])]
                    if expected_ids != stage_b_ids or expected_ids != observed_ids:
                        errors.append(f"candidate_id_alignment_mismatch:{split}:{episode_id}")
                    expected_utilities = [float(item["utility"]) for item in stage_b_candidates]
                    observed_utilities = [float(item) for item in row.get("utility_targets", [])]
                    if len(expected_utilities) != len(observed_utilities) or not np.allclose(expected_utilities, observed_utilities, atol=1e-6, rtol=0.0):
                        errors.append(f"utility_target_mismatch:{split}:{episode_id}")
                    expected_geodesics = [float(item["geodesic_distance_m"]) for item in expected_candidates]
                    observed_geodesics = [float(item) for item in row.get("candidate_geodesic", [])]
                    if len(expected_geodesics) != len(observed_geodesics) or not np.allclose(expected_geodesics, observed_geodesics, atol=1e-7, rtol=0.0):
                        errors.append(f"candidate_geodesic_mismatch:{split}:{episode_id}")
                    current = np.asarray(row["current_feature"], dtype=np.float32)
                    geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
                    geometry_dim = int(schema.get("candidate_geometry_dim", 11)) if isinstance(schema, Mapping) else 11
                    if current.shape != (275,) or geometry.shape != (len(expected_candidates), geometry_dim) or not np.isfinite(current).all() or not np.isfinite(geometry).all():
                        errors.append(f"invalid_feature_values:{split}:{episode_id}")
                    try:
                        archive_path = str(episode["current_view"]["skeleton_source_path"])
                        placement = archive_placements.get(archive_path)
                        if placement is None:
                            with np.load(archive_path, allow_pickle=False) as archive:
                                placement = np.asarray(archive["placement_position"], dtype=np.float32)
                            archive_placements[archive_path] = placement
                        expected_geometry = candidate_geometry_matrix(
                            expected_candidates,
                            current_position=episode["current_view"]["agent_position"],
                            current_rotation_wxyz=episode["current_view"]["rotation_wxyz"],
                            placement_position=placement,
                            include_relative_features=include_relative_features,
                        )
                        if geometry.shape == expected_geometry.shape and not np.allclose(geometry, expected_geometry, atol=1e-5, rtol=0.0):
                            errors.append(f"candidate_geometry_mismatch:{split}:{episode_id}")
                    except (OSError, KeyError, TypeError, ValueError) as error:
                        errors.append(f"candidate_geometry_audit_failed:{split}:{episode_id}:{error}")
                    feature_contracts[split][episode_id] = {
                        "record_id": str(episode["record_id"]), "policy_split": str(episode["policy_split"]),
                        "scene_id": str(episode["scene_id"]), "region": str(episode["region"]),
                        "label_id": int(episode["label_id"]), "current_viewpoint_id": int(episode["current_view"]["viewpoint_id"]),
                        "candidate_viewpoint_ids": expected_ids, "utility_targets": expected_utilities,
                        "candidate_geodesic": expected_geodesics,
                    }
            if count != int(feature_summary["feature_file_counts"][split]):
                errors.append(f"feature_count_mismatch:{split}")
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
                training_summary_path = checkpoint_path.parent / f"{model_type}_training_summary.json"
                if not training_summary_path.exists():
                    errors.append(f"missing_training_summary:{model_type}")
                else:
                    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
                    for field in ("feature_summary_sha256", "feature_file_sha256", "feature_stats_sha256"):
                        if training_summary.get(field) != feature_summary.get(field) and field != "feature_summary_sha256":
                            errors.append(f"training_{field}_mismatch:{model_type}")
                    if training_summary.get("feature_summary_sha256") != file_sha256(stage_c_root / "stage_c_feature_summary.json"):
                        errors.append(f"training_feature_summary_hash_mismatch:{model_type}")
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if payload.get("feature_summary_sha256") != training_summary.get("feature_summary_sha256") or payload.get("feature_file_sha256") != training_summary.get("feature_file_sha256") or payload.get("feature_stats_sha256") != training_summary.get("feature_stats_sha256"):
                        errors.append(f"checkpoint_feature_provenance_mismatch:{model_type}")
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
                if summary.get("feature_file_sha256") != feature_summary.get("feature_file_sha256"):
                    errors.append(f"feature_file_hash_mismatch:{model_type}")
                if summary.get("feature_stats_sha256") != feature_summary.get("feature_stats_sha256"):
                    errors.append(f"feature_stats_hash_mismatch:{model_type}")
                for split in ("val", "test"):
                    prediction_path = Path(summary["prediction_files"][split])
                    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    stage_b_path = stage_b_root / "utility_labels" / f"{split}.jsonl"
                    stage_b_lookup = {str(item["episode_id"]): item for item in (json.loads(line) for line in stage_b_path.read_text(encoding="utf-8").splitlines() if line.strip())}
                    expected = feature_contracts[split]
                    observed_ids: List[str] = []
                    for row in rows:
                        episode_id = str(row.get("episode_id", ""))
                        observed_ids.append(episode_id)
                        contract = expected.get(episode_id)
                        if contract is None:
                            errors.append(f"unexpected_prediction_episode:{model_type}:{split}:{episode_id}")
                            continue
                        for field in ("record_id", "policy_split", "scene_id", "region", "label_id", "current_viewpoint_id", "candidate_viewpoint_ids"):
                            if row.get(field) != contract[field]:
                                errors.append(f"prediction_{field}_mismatch:{model_type}:{split}:{episode_id}")
                        if len(row.get("utility_targets", [])) != len(contract["utility_targets"]) or not np.allclose(row.get("utility_targets", []), contract["utility_targets"], atol=1e-6, rtol=0.0):
                            errors.append(f"prediction_utility_targets_mismatch:{model_type}:{split}:{episode_id}")
                        if len(row.get("candidate_geodesic", contract["candidate_geodesic"])) != len(contract["candidate_geodesic"]):
                            errors.append(f"prediction_candidate_count_mismatch:{model_type}:{split}:{episode_id}")
                        stage_b = stage_b_lookup.get(episode_id)
                        if stage_b is None:
                            errors.append(f"missing_stage_b_episode:{model_type}:{split}:{episode_id}")
                        else:
                            _validate_independent_decision(row, stage_b, contract, model_type, split, errors)
                    if len(observed_ids) != len(set(observed_ids)):
                        errors.append(f"duplicate_prediction_episode_id:{model_type}:{split}")
                    if set(observed_ids) != set(expected):
                        errors.append(f"prediction_episode_coverage_mismatch:{model_type}:{split}")
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
    from activeview.core.paths import get_data_root
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
