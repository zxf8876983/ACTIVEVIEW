from activeview.research.manifest import load_manifest, save_manifest, write_json_atomic


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
