#!/usr/bin/env python3
"""Audit exact clean-motion alignment for the reduced12 Val benchmark.

The downstream representation analysis is deliberately gated on an exact
instance-level clean H36M-17 reference.  This script records the available
record/segment provenance and stops without fabricating a clean skeleton when
the formal clean preprocessing implementation or cache is unavailable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root


DATASET_NAME = "reduced12_no_kneel_clean_babel_diversity_v1"
POLICY_NAME = "policy_reduced12_eight_placement_v1"
OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/recognizer_clean_reference_audit"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _record_id(row: Mapping[str, Any]) -> str:
    value = row.get("record_id")
    if value is None or not str(value):
        raise ValueError("row has no non-empty record_id")
    return str(value)


def _check_formal_clean_backend() -> dict[str, Any]:
    module_name = "activeview.data.motion.babel_official150_true_skeleton"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - environment-specific import audit
        return {
            "module": module_name,
            "available": False,
            "converter_symbol": "AMASSTrueSkeletonConverter",
            "error": f"{type(exc).__name__}: {exc}",
        }
    available = hasattr(module, "AMASSTrueSkeletonConverter")
    return {
        "module": module_name,
        "available": bool(available),
        "converter_symbol": "AMASSTrueSkeletonConverter",
        "error": None if available else "AMASSTrueSkeletonConverter is not defined",
    }


def _candidate_clean_paths(dataset_root: Path, record_id: str) -> list[Path]:
    """Return only explicit clean-cache locations; never treat estimated views as clean."""
    relative_paths = (
        Path("raw-val/clean_skeletons") / f"{record_id}.npz",
        Path("raw-val/true_skeletons") / f"{record_id}.npz",
        Path("clean_skeletons") / f"{record_id}.npz",
        Path("true_skeletons") / f"{record_id}.npz",
    )
    return [dataset_root / relative for relative in relative_paths]


def _audit_alignment(data_root: Path) -> dict[str, Any]:
    dataset_root = data_root / "datasets" / DATASET_NAME
    manifest_path = dataset_root / "raw-val" / "official_val.json"
    stage_d_path = data_root / "datasets" / POLICY_NAME / "stage_d" / "features" / "val.jsonl"
    records = _read_json(manifest_path)
    contexts = _read_jsonl(stage_d_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {manifest_path}")

    by_id: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("official_val.json contains a non-object record")
        by_id.setdefault(_record_id(item), []).append(item)

    context_record_ids = [_record_id(row) for row in contexts]
    missing_context_records = sorted(set(context_record_ids) - set(by_id))
    ambiguous_records = sorted(record_id for record_id, items in by_id.items() if len(items) != 1)
    matched_contexts = sum(record_id in by_id and len(by_id[record_id]) == 1 for record_id in context_record_ids)

    source_missing: list[str] = []
    interval_invalid: list[str] = []
    clean_cache_hits: list[str] = []
    for record_id, items in by_id.items():
        if len(items) != 1:
            continue
        record = items[0]
        source_path = Path(str(record.get("source_path", "")))
        if not source_path.is_file():
            source_missing.append(record_id)
        try:
            start = int(record["start_frame"])
            end = int(record["end_frame"])
            source_frames = int(record["source_num_frames"])
            if start < 0 or end < start or end >= source_frames:
                interval_invalid.append(record_id)
        except (KeyError, TypeError, ValueError):
            interval_invalid.append(record_id)
        if any(path.is_file() for path in _candidate_clean_paths(dataset_root, record_id)):
            clean_cache_hits.append(record_id)

    backend = _check_formal_clean_backend()
    exact_record_mapping = not missing_context_records and not ambiguous_records
    exact_segment_mapping = exact_record_mapping and not source_missing and not interval_invalid
    clean_reference_available = bool(clean_cache_hits) or bool(backend["available"])
    blockers = []
    if missing_context_records:
        blockers.append("some Val contexts do not map to official_val.json record instances")
    if ambiguous_records:
        blockers.append("official_val.json contains ambiguous record_id instances")
    if source_missing:
        blockers.append("one or more AMASS source files are missing")
    if interval_invalid:
        blockers.append("one or more record frame intervals are invalid")
    if not clean_cache_hits:
        blockers.append("no explicit clean/true H36M17 skeleton cache was found")
    if not backend["available"]:
        blockers.append("formal AMASS-to-H36M17 clean converter is unavailable")

    return {
        "status": "PASS" if exact_segment_mapping and clean_reference_available else "BLOCKED",
        "total_contexts": len(contexts),
        "unique_context_record_ids": len(set(context_record_ids)),
        "official_val_records": len(records),
        "unique_official_record_ids": len(by_id),
        "record_mapping_contexts": int(matched_contexts),
        "exact_clean_motion_matched": 0,
        "unmatched": len(missing_context_records),
        "ambiguous": len(ambiguous_records),
        "exact_frame_resampling_mapping_confirmed": bool(exact_segment_mapping),
        "source_files_present": len(by_id) - len(source_missing),
        "source_files_missing": sorted(source_missing),
        "invalid_intervals": sorted(interval_invalid),
        "clean_cache_hits": sorted(clean_cache_hits),
        "formal_clean_backend": backend,
        "clean_instance_alignment_confirmed": False,
        "blockers": blockers,
        "paths": {
            "official_val_manifest": str(manifest_path.resolve()),
            "stage_d_val_features": str(stage_d_path.resolve()),
            "dataset_root": str(dataset_root.resolve()),
        },
        "protocol": {
            "split": "val_moving",
            "target_frames": 30,
            "resampling_rule": "np.linspace(start_frame, end_frame, 30, dtype=int64)",
            "test_used": False,
            "training_used": False,
            "new_rgb_rendered": False,
            "new_pose_estimation": False,
            "exact_clean_motion_instance_required": True,
            "clean_motion_used_for_privileged_reference_only": True,
            "clean_reference_used_as_deployable_input": False,
            "deployable": False,
        },
    }


def _blocked_artifact(reason: str, alignment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "alignment_status": alignment.get("status"),
        "test_used": False,
        "training_used": False,
        "metrics_computed": False,
    }


def _write_outputs(alignment: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "alignment_audit.json").write_text(json.dumps(alignment, indent=2, ensure_ascii=False), encoding="utf-8")
    reason = "; ".join(str(item) for item in alignment["blockers"]) or "clean reference unavailable"
    artifacts = {
        "clean_recognition_metrics.json": "clean recognizer metrics",
        "representation_similarity.json": "feature/posterior similarity",
        "ranking_correlations.json": "utility ranking correlations",
        "quadrant_metrics.json": "clean/oracle correctness quadrants",
        "cleanwrong_oraclecorrect.json": "CleanWrong & OracleCorrect analysis",
        "cleancorrect_oraclewrong.json": "CleanCorrect & OracleWrong analysis",
        "privileged_selector_metrics.json": "clean-reference privileged selectors",
        "per_class_metrics.json": "per-class clean-reference metrics",
    }
    for filename, label in artifacts.items():
        (output / filename).write_text(json.dumps(_blocked_artifact(f"alignment gate blocked {label}: {reason}", alignment), indent=2), encoding="utf-8")
    result = {
        "experiment_id": "REDUCED12_RECOGNIZER_CLEAN_REFERENCE_AUDIT",
        "status": "BLOCKED",
        "population": {
            "split": "val_moving",
            "contexts": alignment["total_contexts"],
            "candidate_features_evaluated": 0,
        },
        "alignment_audit": alignment,
        "metrics": {},
        "protocol": alignment["protocol"],
        "reason": "Main instance-level analysis was stopped before any clean skeleton or recognizer metrics were fabricated.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Recognizer-Level Clean Motion Reference Audit",
        "",
        "## Status: BLOCKED before instance-level analysis",
        "",
        "The Val record-to-source audit was run, but the required exact clean H36M17 reference could not be constructed from the repository's current formal pipeline. No clean skeleton, ST-GCN output, similarity, ranking, quadrant, selector, or per-class metric was fabricated.",
        "",
        f"- Val Moving contexts: {alignment['total_contexts']}",
        f"- Unique Val record instances: {alignment['unique_context_record_ids']}",
        f"- Exact record/segment mapping: {'confirmed' if alignment['exact_frame_resampling_mapping_confirmed'] else 'not confirmed'}",
        f"- Exact clean motion instances matched: {alignment['exact_clean_motion_matched']}",
        f"- Blockers: {'; '.join(alignment['blockers'])}",
        "",
        "## What is available",
        "",
        "Every checked Val record carries an AMASS source path, start/end frame, and the deterministic 30-frame `np.linspace` mapping. The archived candidate NPZs are view-dependent estimated skeletons and therefore are not accepted as clean references.",
        "",
        "## Why the analysis stopped",
        "",
        "`activeview.data.motion.babel_official150_true_skeleton` does not define the referenced `AMASSTrueSkeletonConverter`, and no explicit clean/true H36M17 skeleton cache exists at the checked dataset locations. The installed environment also lacks an alternative formal clean FK path. Generating a clean reference by choosing another same-class motion, using a centroid, or treating an estimated candidate as clean would violate exact instance alignment.",
        "",
        "## Required minimum fix",
        "",
        "Restore or provide the project's formal AMASS-SMPL/SMPL-X-to-H36M17 FK converter and its exact normalization (`root_center + torso_scale + yaw_only`), or provide a verified per-record clean skeleton cache keyed by the exact `record_id` and frame segment. Then rerun this script before computing recognizer-level comparisons.",
        "",
        "Flags: `test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_pose_estimation=false`; `exact_clean_motion_instance_required=true`; `clean_motion_used_for_privileged_reference_only=true`; `clean_reference_used_as_deployable_input=false`; `deployable=false`.",
        "",
        "`representation_cases.png` was intentionally not generated because the alignment gate failed and no valid cases exist.",
    ]
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    alignment = _audit_alignment(get_data_root())
    _write_outputs(alignment, args.output)
    print(json.dumps({"status": alignment["status"], "output": str(args.output.resolve()), "blockers": alignment["blockers"]}, ensure_ascii=False))
    return 0 if alignment["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
