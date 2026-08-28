from activeview.research.manifest import (
    load_manifest,
    save_manifest,
    validate_controlled_config,
    write_json_atomic,
)


def test_manifest_json_round_trip(tmp_path):
    source = tmp_path / "EXP001_demo"
    source.mkdir()
    payload = {"schema_version": "stage-c-research-v1", "experiment": {"experiment_id": "EXP001"}}
    save_manifest(source, payload)
    assert load_manifest(source) == payload


def test_atomic_json_write_replaces_previous_payload(tmp_path):
    path = tmp_path / "status.json"
    write_json_atomic(path, {"status": "PLANNED"})
    write_json_atomic(path, {"status": "RUNNING"})
    assert "RUNNING" in path.read_text(encoding="utf-8")


def test_controlled_config_accepts_locked_non_test_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "experiment:\n  id: EXP001\n"
        "evaluation:\n  test: false\n"
        "protocol:\n  test_locked: true\n  test_authorized: false\n",
        encoding="utf-8",
    )
    assert validate_controlled_config(path, "EXP001") == []


def test_controlled_config_rejects_test_unlock_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "experiment:\n  id: EXP001\n"
        "evaluation:\n  test: true\n"
        "protocol:\n  test_locked: false\n  test_authorized: true\n",
        encoding="utf-8",
    )
    errors = validate_controlled_config(path, "EXP001")
    assert "config_evaluation_test_must_be_false" in errors
    assert "config_test_locked_must_be_true" in errors
    assert "config_test_authorized_must_be_false" in errors
