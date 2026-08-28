import json

import pytest

from activeview.active_view.utility_label_builder import file_sha256
from activeview.research.test_gate import TestGateError, assert_test_allowed


def _final_manifest(tmp_path, commit="abc"):
    config = tmp_path / "config.yaml"; config.write_text("experiment:\n  id: EXP001\n", encoding="utf-8")
    auth = tmp_path / "final_test_authorization.json"
    auth.write_text(json.dumps({"experiment_id": "EXP001", "authorized": True, "frozen_git_commit": commit, "config_sha256": file_sha256(config)}), encoding="utf-8")
    manifest = {"experiment_id": "EXP001", "status": "FINAL_FROZEN", "decision": "ACCEPT", "protocol": {"test_locked": True, "final_model_frozen": True, "test_authorized": True}}
    return manifest, auth, config


def test_allow_test_flag_alone_cannot_unlock(tmp_path):
    with pytest.raises(TestGateError):
        assert_test_allowed({"experiment_id": "EXP001", "status": "COMPLETED", "decision": "ACCEPT"}, allow_test=True)


def test_final_frozen_matching_commit_and_config_passes(tmp_path):
    manifest, auth, config = _final_manifest(tmp_path)
    result = assert_test_allowed(manifest, authorization_path=auth, config_path=config, current_commit="abc")
    assert result["allowed"] is True


def test_commit_or_config_mismatch_fails_closed(tmp_path):
    manifest, auth, config = _final_manifest(tmp_path)
    with pytest.raises(TestGateError, match="frozen_commit_mismatch"):
        assert_test_allowed(manifest, authorization_path=auth, config_path=config, current_commit="def")
