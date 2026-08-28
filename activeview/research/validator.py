"""Independent validator for experiment manifests and lifecycle records."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping

from activeview.active_view.utility_label_builder import file_sha256

from .experiment import ExperimentStatus
from .manifest import git_value, load_manifest
from .provenance import provenance_complete
from .registry import get_experiment


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
    if isinstance(paths, Mapping) and config_path.is_file() and paths.get("config_sha256") != file_sha256(config_path):
        errors.append("config_hash_mismatch")
    provenance = manifest.get("provenance", {})
    if require_complete_provenance and not provenance_complete(provenance if isinstance(provenance, Mapping) else {}):
        errors.append("frozen_provenance_incomplete")
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
        if decision != "ACCEPT":
            errors.append("final_frozen_requires_accept")
        if protocol.get("final_model_frozen") is not True or protocol.get("test_authorized") is not True:
            errors.append("final_frozen_protocol_invalid")
        authorization_path = source_dir / "final_test_authorization.json"
        if not authorization_path.is_file():
            errors.append("final_authorization_missing")
        else:
            try:
                authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                authorization = {}
                errors.append("final_authorization_invalid")
            if authorization.get("authorized") is not True or authorization.get("experiment_id") != experiment_id:
                errors.append("final_authorization_identity_invalid")
            if not authorization.get("frozen_git_commit") or authorization.get("frozen_git_commit") != git_value("rev-parse", "HEAD"):
                errors.append("final_authorization_commit_mismatch")
            if authorization.get("config_sha256") != file_sha256(config_path):
                errors.append("final_authorization_config_hash_mismatch")
            if not authorization.get("manifest_sha256_before_authorization"):
                errors.append("final_authorization_manifest_hash_missing")
    if status == "COMPLETED":
        for filename in ("val_metrics.json", "analysis.json", "conclusion.md"):
            if not (source_dir / filename).is_file():
                errors.append(f"completed_file_missing:{filename}")
    if registry_path is not None:
        try:
            row = get_experiment(registry_path, experiment_id)
            if row.get("status") != status:
                errors.append("registry_status_mismatch")
            if row.get("decision", "NA") != decision:
                errors.append("registry_decision_mismatch")
        except (KeyError, ValueError) as error:
            errors.append(f"registry_identity_invalid:{error}")
    return {"passed": not errors, "error_count": len(errors), "warning_count": len(warnings), "errors": errors, "warnings": warnings, "experiment_id": experiment_id, "status": status, "decision": decision}
