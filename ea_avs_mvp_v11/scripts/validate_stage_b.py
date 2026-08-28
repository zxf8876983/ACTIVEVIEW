#!/usr/bin/env python3
"""Independently validate Stage B Utility labels against Stage A Episodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.active_view.utility_label_builder import (
    UTILITY_TOLERANCE,
    file_sha256,
)


SPLITS = ("train", "val", "test")
LEGACY_MINIVAL_SCENE = "00800-TEEsavR23oF"
FORBIDDEN_TOKENS = (
    "skeleton", "rgb", "depth", "logits", "probabilities", "pose_3d",
    "future_skeleton", "candidate_pose", "candidate_entropy_map",
)


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, payload


def _load_stage_a(root: Path) -> tuple[Dict[str, Any], Dict[str, Mapping[str, Any]], Dict[str, int]]:
    summary = json.loads((root / "stage_a_summary.json").read_text(encoding="utf-8"))
    episodes: Dict[str, Mapping[str, Any]] = {}
    counts: Dict[str, int] = {}
    for split in SPLITS:
        path = Path(summary["episode_files"][split])
        count = 0
        for _line, episode in _iter_jsonl(path):
            episode_id = str(episode["episode_id"])
            if episode_id in episodes:
                raise ValueError(f"Duplicate Stage A episode_id: {episode_id}")
            episodes[episode_id] = episode
            count += 1
        counts[split] = count
    return summary, episodes, counts


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_TOKENS):
                return True
            if _contains_forbidden(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden(child) for child in value)
    return False


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _validate_record(
    record: Mapping[str, Any], stage_a: Mapping[str, Any], tolerance: float,
) -> List[str]:
    errors: List[str] = []
    if _contains_forbidden(record):
        errors.append("forbidden_future_perception_field")
    required = {"episode_id", "record_id", "policy_split", "scene_id", "region", "label_id", "current", "candidates", "oracle"}
    missing = sorted(required - set(record))
    if missing:
        return [f"missing_fields:{','.join(missing)}"]
    label_id = int(record["label_id"])
    current = record["current"]
    candidates = record["candidates"]
    oracle = record["oracle"]
    if not isinstance(current, Mapping) or not isinstance(candidates, list) or not isinstance(oracle, Mapping):
        return ["invalid_record_sections"]
    if not all(_finite(current.get(field)) for field in ("logp_true", "entropy")):
        errors.append("nonfinite_current_diagnostics")
    if bool(current.get("correct")) != (int(current.get("predicted_label_id", -1)) == label_id):
        errors.append("current_correctness_mismatch")
    current_id = int(current.get("viewpoint_id", -1))
    stage_a_current = stage_a["current_view"]
    if current_id != int(stage_a_current["viewpoint_id"]):
        errors.append("current_viewpoint_mismatch")
    expected_candidates = {int(item["viewpoint_id"]): item for item in stage_a["candidate_pool"]}
    observed_ids = [int(item.get("viewpoint_id", -1)) for item in candidates if isinstance(item, Mapping)]
    if len(observed_ids) != len(set(observed_ids)):
        errors.append("duplicate_candidate_viewpoints")
    if set(observed_ids) != set(expected_candidates):
        errors.append("candidate_viewpoint_set_mismatch")
    candidate_by_id: Dict[int, Mapping[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, Mapping):
            errors.append("invalid_candidate_record")
            continue
        viewpoint_id = int(item.get("viewpoint_id", -1))
        candidate_by_id[viewpoint_id] = item
        if not all(_finite(item.get(field)) for field in ("logp_true", "entropy", "utility", "geodesic_distance_m")):
            errors.append(f"nonfinite_candidate:{viewpoint_id}")
        if bool(item.get("correct")) != (int(item.get("predicted_label_id", -1)) == label_id):
            errors.append(f"candidate_correctness_mismatch:{viewpoint_id}")
        if _finite(item.get("utility")) and _finite(item.get("logp_true")) and _finite(current.get("logp_true")):
            expected_utility = float(item["logp_true"]) - float(current["logp_true"])
            if not math.isclose(float(item["utility"]), expected_utility, rel_tol=0.0, abs_tol=tolerance):
                errors.append(f"utility_formula_mismatch:{viewpoint_id}")
    if not candidate_by_id:
        errors.append("empty_candidates")
        return errors
    oracle_id = int(oracle.get("candidate_oracle_viewpoint_id", -1))
    if oracle_id not in candidate_by_id:
        errors.append("invalid_candidate_oracle_id")
    else:
        expected_oracle = min(
            candidate_by_id.values(),
            key=lambda item: (-float(item["utility"]), float(item["geodesic_distance_m"]), int(item["viewpoint_id"])),
        )
        if oracle_id != int(expected_oracle["viewpoint_id"]):
            errors.append("candidate_oracle_tiebreak_mismatch")
        if not math.isclose(float(oracle.get("candidate_oracle_utility")), float(expected_oracle["utility"]), rel_tol=0.0, abs_tol=tolerance):
            errors.append("candidate_oracle_utility_mismatch")
    max_utility = max(float(item["utility"]) for item in candidate_by_id.values())
    safe_id = int(oracle.get("safe_oracle_viewpoint_id", -1))
    expected_safe_id = oracle_id if max_utility > 0.0 else current_id
    if safe_id != expected_safe_id:
        errors.append("safe_oracle_viewpoint_mismatch")
    expected_safe_utility = max(0.0, max_utility)
    if not math.isclose(float(oracle.get("safe_oracle_utility")), expected_safe_utility, rel_tol=0.0, abs_tol=tolerance):
        errors.append("safe_oracle_utility_mismatch")
    if bool(oracle.get("safe_oracle_stays")) != (max_utility <= 0.0):
        errors.append("safe_oracle_stay_flag_mismatch")
    return errors


def validate(dataset_root: Path, stage_b_root: Path, max_errors: int = 100) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    counts: Dict[str, Any] = {
        "stage_a_episode_counts": {}, "stage_b_episode_counts": {split: 0 for split in SPLITS},
        "stage_a_candidate_pair_counts": {}, "stage_b_candidate_pair_counts": {split: 0 for split in SPLITS},
        "duplicate_episode_ids": 0, "duplicate_episode_candidate_pairs": 0,
        "missing_stage_b_episodes": 0, "unexpected_stage_b_episodes": 0,
        "record_errors": 0,
    }
    try:
        stage_a_summary, stage_a_episodes, stage_a_counts = _load_stage_a(dataset_root)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        report = {"protocol": "ACTIVEVIEW v11.5 Stage B validator", "stage": "B", "passed": False, "errors": [{"reason": str(error)}], "counts": counts}
        report_path = stage_b_root / "validation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    counts["stage_a_episode_counts"] = stage_a_counts
    counts["stage_a_candidate_pair_counts"] = {
        split: sum(len(episode["candidate_pool"]) for episode in stage_a_episodes.values() if str(episode["policy_split"]) == split)
        for split in SPLITS
    }
    expected_scene_ids = {str(item) for item in stage_a_summary.get("scene_ids_used", [])}
    if len(expected_scene_ids) != 21 or LEGACY_MINIVAL_SCENE in expected_scene_ids:
        errors.append({"reason": "invalid_stage_a_scene_set"})
    summary_path = stage_b_root / "stage_b_summary.json"
    try:
        stage_b_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append({"reason": f"missing_or_invalid_stage_b_summary:{error}"})
        stage_b_summary = {}
    if stage_b_summary.get("stage") != "B":
        errors.append({"reason": "invalid_stage_b_stage_field"})
    if stage_b_summary.get("supervision_only") is not True:
        errors.append({"reason": "stage_b_not_marked_supervision_only"})
    expected_policy_counts = stage_a_summary.get("policy_split", {}).get("counts", {})
    if stage_b_summary.get("split_counts") != expected_policy_counts:
        errors.append({"reason": "summary_policy_split_counts_mismatch"})
    mapping_path = Path(stage_b_summary.get("label_mapping", ""))
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
        if stage_b_summary.get("categories") != categories:
            errors.append({"reason": "label_mapping_mismatch"})
        if stage_b_summary.get("label_mapping_sha256") != file_sha256(mapping_path):
            errors.append({"reason": "label_mapping_hash_mismatch"})
    else:
        errors.append({"reason": "missing_label_mapping"})
    checkpoint_path = Path(stage_b_summary.get("stgcn_checkpoint", ""))
    if not checkpoint_path.exists():
        errors.append({"reason": "missing_stgcn_checkpoint"})
    elif stage_b_summary.get("stgcn_checkpoint_sha256") != file_sha256(checkpoint_path):
        errors.append({"reason": "stgcn_checkpoint_hash_mismatch"})
    seen_episode_ids: set[str] = set()
    seen_pairs: set[Tuple[str, int]] = set()
    for split in SPLITS:
        output_path = stage_b_root / "utility_labels" / f"{split}.jsonl"
        expected_ids = {episode_id for episode_id, episode in stage_a_episodes.items() if str(episode["policy_split"]) == split}
        observed_ids: set[str] = set()
        if not output_path.exists():
            errors.append({"split": split, "reason": "missing_utility_file"})
            continue
        try:
            iterator = _iter_jsonl(output_path)
            for line_number, record in iterator:
                episode_id = str(record.get("episode_id", ""))
                counts["stage_b_episode_counts"][split] += 1
                counts["stage_b_candidate_pair_counts"][split] += len(record.get("candidates", [])) if isinstance(record.get("candidates"), list) else 0
                if episode_id in seen_episode_ids:
                    counts["duplicate_episode_ids"] += 1
                seen_episode_ids.add(episode_id)
                observed_ids.add(episode_id)
                if episode_id not in stage_a_episodes:
                    counts["unexpected_stage_b_episodes"] += 1
                    if len(errors) < max_errors:
                        errors.append({"split": split, "line": line_number, "reason": "unexpected_episode_id"})
                    continue
                stage_a_episode = stage_a_episodes[episode_id]
                if str(record.get("policy_split")) != split:
                    errors.append({"split": split, "line": line_number, "reason": "split_mismatch"})
                for field in ("record_id", "scene_id", "region", "label_id"):
                    if record.get(field) != stage_a_episode.get(field):
                        errors.append({"split": split, "line": line_number, "reason": f"episode_{field}_mismatch"})
                scene_id = str(record.get("scene_id", ""))
                if scene_id not in expected_scene_ids or scene_id == LEGACY_MINIVAL_SCENE:
                    errors.append({"split": split, "line": line_number, "reason": "noncanonical_scene"})
                for candidate in record.get("candidates", []) if isinstance(record.get("candidates"), list) else []:
                    if isinstance(candidate, Mapping):
                        pair = (episode_id, int(candidate.get("viewpoint_id", -1)))
                        if pair in seen_pairs:
                            counts["duplicate_episode_candidate_pairs"] += 1
                        seen_pairs.add(pair)
                record_errors = _validate_record(record, stage_a_episode, float(stage_b_summary.get("numeric_tolerance", UTILITY_TOLERANCE)))
                counts["record_errors"] += len(record_errors)
                if record_errors and len(errors) < max_errors:
                    errors.append({"split": split, "line": line_number, "reason": record_errors[:20]})
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append({"split": split, "reason": f"invalid_utility_jsonl:{error}"})
        counts["missing_stage_b_episodes"] += len(expected_ids - observed_ids)
        counts["unexpected_stage_b_episodes"] += len(observed_ids - expected_ids)

    expected_episode_counts = {split: int(value) for split, value in stage_a_counts.items()}
    if stage_b_summary.get("episode_counts") != counts["stage_b_episode_counts"]:
        errors.append({"reason": "summary_episode_counts_mismatch"})
    if stage_b_summary.get("candidate_pair_counts") != counts["stage_b_candidate_pair_counts"]:
        errors.append({"reason": "summary_candidate_pair_counts_mismatch"})
    if counts["stage_b_candidate_pair_counts"] != counts["stage_a_candidate_pair_counts"]:
        errors.append({"reason": "candidate_pair_coverage_mismatch"})
    if (
        stage_b_summary.get("official_scene_count") != 21
        or set(stage_b_summary.get("scene_ids", [])) != expected_scene_ids
    ):
        errors.append({"reason": "summary_scene_count_mismatch"})
    passed = not errors and counts["stage_b_episode_counts"] == expected_episode_counts and counts["stage_b_candidate_pair_counts"] == counts["stage_a_candidate_pair_counts"] and counts["missing_stage_b_episodes"] == 0 and counts["unexpected_stage_b_episodes"] == 0 and counts["duplicate_episode_ids"] == 0 and counts["duplicate_episode_candidate_pairs"] == 0 and counts["record_errors"] == 0
    report = {
        "protocol": "ACTIVEVIEW v11.5 Stage B validator",
        "stage": "B",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "supervision_only": True,
        "counts": counts,
        "errors": errors[:max_errors],
    }
    report_path = stage_b_root / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    data_root = Path(__file__).resolve().parents[3] / "data" / "ActiveView"
    try:
        from ea_avs_mvp_v11.core.paths import get_data_root
        data_root = get_data_root()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--max-errors", type=int, default=100)
    args = parser.parse_args()
    report = validate(args.dataset_root, args.stage_b_root, max_errors=args.max_errors)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
