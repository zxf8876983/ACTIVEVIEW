"""Fail-closed authorization gate for final Test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from activeview.active_view.utility_label_builder import file_sha256

from .manifest import git_value


class TestGateError(RuntimeError):
    """Raised whenever a Test evaluation lacks explicit final authorization."""

    __test__ = False


def validate_final_test_authorization(
    manifest: Mapping[str, Any], *, authorization_path: Path | None,
    config_path: Path | None, current_commit: str | None = None,
) -> list[str]:
    """Return shared authorization errors for a canonical run manifest."""
    errors: list[str] = []
    experiment = manifest.get("experiment", {})
    protocol = manifest.get("protocol", {})
    if not isinstance(experiment, Mapping) or not isinstance(protocol, Mapping):
        return ["canonical_manifest_sections_missing"]
    experiment_id = str(experiment.get("experiment_id", ""))
    if experiment.get("status") != "FINAL_FROZEN":
        errors.append("experiment_not_final_frozen")
    if experiment.get("decision") != "ACCEPT":
        errors.append("experiment_not_accepted")
    if protocol.get("test_locked") is not True:
        errors.append("test_lock_missing")
    if protocol.get("final_model_frozen") is not True:
        errors.append("final_model_not_frozen")
    # Authorization is deliberately external to the tracked manifest. The
    # runtime authorization artifact below is the sole Test unlock signal.
    if authorization_path is None or not authorization_path.is_file():
        errors.append("authorization_artifact_missing")
        authorization: Mapping[str, Any] = {}
    else:
        try:
            payload = json.loads(authorization_path.read_text(encoding="utf-8"))
            authorization = payload if isinstance(payload, Mapping) else {}
            if not isinstance(payload, Mapping):
                errors.append("authorization_artifact_invalid")
        except (OSError, ValueError):
            authorization = {}
            errors.append("authorization_artifact_invalid")
        if authorization.get("authorized") is not True:
            errors.append("authorization_not_confirmed")
    if authorization.get("experiment_id") != experiment_id:
        errors.append("authorization_experiment_mismatch")
    actual_commit = current_commit or git_value("rev-parse", "HEAD")
    if authorization.get("frozen_git_commit") != actual_commit:
        errors.append("frozen_commit_mismatch")
    if config_path is None or not config_path.is_file():
        errors.append("config_missing")
    elif authorization.get("config_sha256") != file_sha256(config_path):
        errors.append("config_hash_mismatch")
    return errors


def assert_test_allowed(
    manifest: Mapping[str, Any],
    *,
    authorization_path: Path | None = None,
    config_path: Path | None = None,
    current_commit: str | None = None,
    allow_test: bool = False,
) -> Dict[str, Any]:
    """Validate every condition required to run a final Test evaluation.

    ``allow_test`` is intentionally not an override.  It is accepted only for
    CLI compatibility and still requires all authorization conditions below.
    """
    if not isinstance(manifest, Mapping):
        raise TestGateError("Test evaluation denied: canonical_manifest_required")
    errors = validate_final_test_authorization(manifest, authorization_path=authorization_path, config_path=config_path, current_commit=current_commit)
    experiment_payload = manifest.get("experiment")
    experiment_id = str(
        experiment_payload.get("experiment_id", "")
        if isinstance(experiment_payload, Mapping)
        else ""
    )
    actual_commit = current_commit or git_value("rev-parse", "HEAD")
    if errors:
        raise TestGateError("Test evaluation denied: " + ", ".join(errors))
    return {"allowed": True, "experiment_id": experiment_id, "git_commit": actual_commit, "allow_test_flag": bool(allow_test)}
