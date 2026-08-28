"""Fail-closed authorization gate for final Test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from activeview.active_view.utility_label_builder import file_sha256

from .experiment import Experiment
from .manifest import git_value


class TestGateError(RuntimeError):
    """Raised whenever a Test evaluation lacks explicit final authorization."""


def _value(experiment: Experiment | Mapping[str, Any], name: str) -> Any:
    if isinstance(experiment, Experiment):
        value = getattr(experiment, name)
        return getattr(value, "value", value)
    return experiment.get(name)


def assert_test_allowed(
    experiment: Experiment | Mapping[str, Any],
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
    errors: list[str] = []
    if str(_value(experiment, "status")) != "FINAL_FROZEN":
        errors.append("experiment_not_final_frozen")
    if str(_value(experiment, "decision")) != "ACCEPT":
        errors.append("experiment_not_accepted")
    manifest = experiment if isinstance(experiment, Mapping) else {}
    protocol = manifest.get("protocol", {}) if isinstance(manifest, Mapping) else {}
    if protocol.get("test_locked") is not True:
        errors.append("test_lock_missing")
    if protocol.get("final_model_frozen") is not True:
        errors.append("final_model_not_frozen")
    if protocol.get("test_authorized") is not True:
        errors.append("test_not_authorized")
    if authorization_path is None or not authorization_path.is_file():
        errors.append("authorization_artifact_missing")
        authorization: Mapping[str, Any] = {}
    else:
        try:
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            authorization = {}
            errors.append("authorization_artifact_invalid")
        if authorization.get("authorized") is not True:
            errors.append("authorization_not_confirmed")
    expected_id = str(_value(experiment, "experiment_id"))
    if authorization.get("experiment_id") != expected_id:
        errors.append("authorization_experiment_mismatch")
    actual_commit = current_commit or git_value("rev-parse", "HEAD")
    if authorization.get("frozen_git_commit") != actual_commit:
        errors.append("frozen_commit_mismatch")
    if config_path is None or not config_path.is_file():
        errors.append("config_missing")
    elif authorization.get("config_sha256") != file_sha256(config_path):
        errors.append("config_hash_mismatch")
    if errors:
        raise TestGateError("Test evaluation denied: " + ", ".join(errors))
    return {"allowed": True, "experiment_id": expected_id, "git_commit": actual_commit, "allow_test_flag": bool(allow_test)}
