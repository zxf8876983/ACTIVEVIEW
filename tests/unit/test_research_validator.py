import json

from activeview.research.validator import validate_experiment


def _manifest(source, runtime, status="PLANNED", decision="NA"):
    return {
        "schema_version": "stage-c-research-v1",
        "experiment": {"experiment_id": "EXP001", "name": "demo", "stage": "stage_c_v1", "status": status, "decision": decision, "hypothesis": "h", "motivation": "m", "baseline": "b", "core_change": "c", "frozen_items": [], "metrics": [], "acceptance_criteria": [], "rejection_criteria": [], "source_dir": str(source.resolve()), "runtime_dir": str(runtime.resolve()), "created_at": "2026-01-01T00:00:00Z", "completed_at": None},
        "paths": {"config_sha256": ""},
        "provenance": {},
        "protocol": {"test_locked": True, "test_used": False, "final_model_frozen": False, "test_authorized": False},
    }


def test_validator_rejects_missing_provenance_and_completed_files(tmp_path):
    source, runtime = tmp_path / "EXP001_demo", tmp_path / "runtime"
    source.mkdir(); runtime.mkdir()
    (source / "config.yaml").write_text("experiment:\n  id: EXP001\n", encoding="utf-8")
    (source / "run_manifest.json").write_text(json.dumps(_manifest(source, runtime, "COMPLETED", "ACCEPT")), encoding="utf-8")
    report = validate_experiment(source)
    assert not report["passed"]
    assert "frozen_provenance_incomplete" in report["errors"]
    assert "completed_file_missing:val_metrics.json" in report["errors"]
