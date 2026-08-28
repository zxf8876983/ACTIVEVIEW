from activeview.active_view.utility_label_builder import file_sha256
from activeview.research.provenance import verify_frozen_provenance


def _provenance(path):
    artifact = {"path": str(path), "exists": True, "sha256": file_sha256(path)}
    return {
        "stage_a": {"summary": artifact},
        "stage_b": {"summary": artifact},
        "stage_c_features": {"summary": artifact},
        "stgcn_checkpoint": artifact,
        "label_mapping": artifact,
        "record_split": {"summary": artifact},
    }


def test_verify_frozen_provenance_detects_mutation(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    provenance = _provenance(artifact)
    assert verify_frozen_provenance(provenance) == []
    artifact.write_bytes(b"mutated")
    errors = verify_frozen_provenance(provenance)
    assert "frozen_artifact_hash_mismatch:stage_a.summary" in errors


def test_verify_frozen_provenance_reports_missing_artifact(tmp_path):
    artifact = tmp_path / "missing.bin"
    recorded = {"path": str(artifact), "exists": True, "sha256": "0" * 64}
    provenance = {
        "stage_a": {"summary": recorded},
        "stage_b": {"summary": recorded},
        "stage_c_features": {"summary": recorded},
        "stgcn_checkpoint": recorded,
        "label_mapping": recorded,
        "record_split": {"summary": recorded},
    }
    errors = verify_frozen_provenance(provenance)
    assert "frozen_artifact_missing:stage_a.summary" in errors
