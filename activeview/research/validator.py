"""Independent validator for experiment manifests and lifecycle records."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping

from activeview.active_view.utility_label_builder import file_sha256

from .experiment import ExperimentStatus
from .manifest import load_manifest, validate_controlled_config
from .provenance import provenance_complete, verify_frozen_provenance
from .registry import get_experiment
from .test_gate import validate_final_test_authorization


def _config_id(path: Path) -> str | None:
    if not path.is_file():
        return None
    match = re.search(r"^\s+id:\s*(EXP\d+)\s*$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return match.group(1) if match else None


def validate_experiment(
    source_dir: Path, *, registry_path: Path | None = None, require_complete_provenance: bool = True,
) -> Dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = load_manifest(source_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error_count": 1, "warning_count": 0, "errors": [f"manifest_invalid:{error}"], "warnings": []}
    if manifest.get("schema_version") != "stage-c-research-v1":
        errors.append("schema_version_invalid")
    experiment = manifest.get("experiment", {})
    if not isinstance(experiment, Mapping):
        errors.append("experiment_object_missing")
        experiment = {}
    experiment_id = str(experiment.get("experiment_id", ""))
    status = str(experiment.get("status", ""))
    decision = str(experiment.get("decision", "NA"))
    git_metadata = manifest.get("git", {})
    if not isinstance(git_metadata, Mapping):
        errors.append("git_metadata_missing")
        git_metadata = {}
    if not re.fullmatch(r"EXP\d+", experiment_id):
        errors.append("experiment_id_invalid")
    if status not in {item.value for item in ExperimentStatus}:
        errors.append("status_invalid")
    if decision not in {"NA", "ACCEPT", "REJECT", "INCONCLUSIVE"}:
        errors.append("decision_invalid")
    if str(source_dir.resolve()) != str(experiment.get("source_dir", "")):
        errors.append("source_dir_mismatch")
    runtime_dir = Path(str(experiment.get("runtime_dir", "")))
    if not source_dir.is_dir():
        errors.append("source_dir_missing")
    if not runtime_dir.is_dir():
        errors.append("runtime_dir_missing")
    config_path = source_dir / "config.yaml"
    if _config_id(config_path) != experiment_id:
        errors.append("config_experiment_id_mismatch")
    paths = manifest.get("paths", {})
    if not isinstance(paths, Mapping):
        errors.append("paths_missing")
        paths = {}
    if isinstance(paths, Mapping) and config_path.is_file():
        if status == "PLANNED" and paths.get("draft_config_sha256") != file_sha256(config_path):
            warnings.append("draft_config_changed_before_start")
        if status in {"RUNNING", "COMPLETED", "FINAL_FROZEN"} and paths.get("run_config_sha256") != file_sha256(config_path):
            errors.append("run_config_hash_mismatch")
    errors.extend(validate_controlled_config(config_path, experiment_id))
    provenance = manifest.get("provenance", {})
    if require_complete_provenance and not provenance_complete(provenance if isinstance(provenance, Mapping) else {}):
        errors.append("frozen_provenance_incomplete")
    if status in {"RUNNING", "COMPLETED", "FINAL_FROZEN"}:
        errors.extend(verify_frozen_provenance(provenance if isinstance(provenance, Mapping) else {}))
        if not git_metadata.get("run_commit"):
            errors.append("run_commit_missing")
    protocol = manifest.get("protocol", {})
    if not isinstance(protocol, Mapping):
        errors.append("protocol_missing")
        protocol = {}
    if protocol.get("test_locked") is not True:
        errors.append("test_lock_missing")
    if protocol.get("test_used") is not False:
        errors.append("test_used_must_be_false")
    if status not in {"FINAL_FROZEN"} and protocol.get("test_authorized") is not False:
        errors.append("test_authorization_invalid")
    if status == "FINAL_FROZEN":
        errors.extend(validate_final_test_authorization(
            manifest,
            authorization_path=source_dir / "final_test_authorization.json",
            config_path=config_path,
        ))
        authorization_path = source_dir / "final_test_authorization.json"
        if authorization_path.is_file():
            try:
                payload = json.loads(authorization_path.read_text(encoding="utf-8"))
                authorization = payload if isinstance(payload, Mapping) else {}
            except (OSError, ValueError):
                authorization = {}
            if not authorization.get("manifest_sha256_before_authorization"):
                errors.append("final_authorization_manifest_hash_missing")
    if status == "COMPLETED":
        if decision not in {"ACCEPT", "REJECT", "INCONCLUSIVE"}:
            errors.append("completed_decision_invalid")
        for filename in ("val_metrics.json", "analysis.json", "conclusion.md"):
            if not (source_dir / filename).is_file():
                errors.append(f"completed_file_missing:{filename}")
        conclusion = source_dir / "conclusion.md"
        if conclusion.is_file():
            text = conclusion.read_text(encoding="utf-8")
            if "## Decision" not in text or "NA" in text or "TODO" in text:
                errors.append("conclusion_incomplete")
    if status == "PLANNED":
        if decision != "NA":
            errors.append("planned_decision_must_be_na")
        if protocol.get("test_used") is not False:
            errors.append("planned_test_used_must_be_false")
    if status == "RUNNING" and (not git_metadata.get("run_commit") or not paths.get("run_config_sha256")):
        errors.append("running_freeze_metadata_missing")
    if registry_path is not None:
        try:
            row = get_experiment(registry_path, experiment_id)
            if row.get("status") != status:
                errors.append("registry_status_mismatch")
            if row.get("decision", "NA") != decision:
                errors.append("registry_decision_mismatch")
            if row.get("source_dir") != str(source_dir.resolve()):
                errors.append("registry_source_dir_mismatch")
            if row.get("runtime_dir") != str(runtime_dir.resolve()):
                errors.append("registry_runtime_dir_mismatch")
        except (KeyError, ValueError) as error:
            errors.append(f"registry_identity_invalid:{error}")
    return {"passed": not errors, "error_count": len(errors), "warning_count": len(warnings), "errors": errors, "warnings": warnings, "experiment_id": experiment_id, "status": status, "decision": decision}
