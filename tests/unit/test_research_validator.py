import json

from activeview.active_view.utility_label_builder import file_sha256
from activeview.research.provenance import verify_frozen_provenance
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


def test_validator_rehashes_frozen_artifacts(tmp_path):
    source, runtime = tmp_path / "EXP001_demo", tmp_path / "runtime"
    source.mkdir(); runtime.mkdir()
    config = source / "config.yaml"
    config.write_text(
        "experiment:\n  id: EXP001\n"
        "evaluation:\n  test: false\n"
        "protocol:\n  test_locked: true\n  test_authorized: false\n",
        encoding="utf-8",
    )
    hypothesis = source / "hypothesis.md"
    hypothesis.write_text("hypothesis", encoding="utf-8")
    command = source / "command.sh"
    command.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stable")
    recorded = {"path": str(artifact), "exists": True, "sha256": file_sha256(artifact)}
    provenance = {
        "stage_a": {"summary": recorded},
        "stage_b": {"summary": recorded},
        "stage_c_features": {"summary": recorded},
        "stgcn_checkpoint": recorded,
        "label_mapping": recorded,
        "record_split": {"summary": recorded},
    }
    manifest = _manifest(source, runtime, "RUNNING", "NA")
    manifest["git"] = {"run_commit": "abc"}
    manifest["paths"] = {
        "run_config_sha256": file_sha256(config),
        "hypothesis_sha256": file_sha256(hypothesis),
        "command_sha256_at_start": file_sha256(command),
    }
    manifest["provenance"] = provenance
    (source / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_experiment(source)["passed"]
    artifact.write_bytes(b"changed")
    report = validate_experiment(source)
    assert not report["passed"]
    assert "frozen_artifact_hash_mismatch:stage_a.summary" in report["errors"]


def test_validator_rejects_locked_command_or_hypothesis_mutation(tmp_path):
    source, runtime = tmp_path / "EXP001_demo", tmp_path / "runtime"
    source.mkdir(); runtime.mkdir()
    config = source / "config.yaml"
    config.write_text(
        "experiment:\n  id: EXP001\n"
        "evaluation:\n  test: false\n"
        "protocol:\n  test_locked: true\n  test_authorized: false\n",
        encoding="utf-8",
    )
    hypothesis = source / "hypothesis.md"; hypothesis.write_text("h", encoding="utf-8")
    command = source / "command.sh"; command.write_text("c", encoding="utf-8")
    manifest = _manifest(source, runtime, "RUNNING", "NA")
    manifest["git"] = {"run_commit": "abc"}
    manifest["paths"] = {
        "run_config_sha256": file_sha256(config),
        "hypothesis_sha256": file_sha256(hypothesis),
        "command_sha256_at_start": file_sha256(command),
    }
    artifact = tmp_path / "artifact.bin"; artifact.write_bytes(b"x")
    recorded = {"path": str(artifact), "exists": True, "sha256": file_sha256(artifact)}
    manifest["provenance"] = {
        "stage_a": {"summary": recorded}, "stage_b": {"summary": recorded},
        "stage_c_features": {"summary": recorded}, "stgcn_checkpoint": recorded,
        "label_mapping": recorded, "record_split": {"summary": recorded},
    }
    (source / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    command.write_text("changed", encoding="utf-8")
    report = validate_experiment(source)
    assert "locked_artifact_hash_mismatch:command.sh" in report["errors"]
