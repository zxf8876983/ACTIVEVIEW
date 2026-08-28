import json

from activeview.active_view.utility_label_builder import file_sha256
import pytest

from activeview.research.manifest import load_manifest, save_manifest
from activeview.scripts.create_experiment import create_experiment
from activeview.scripts.finalize_experiment import finalize_experiment
from activeview.scripts.freeze_final_candidate import freeze_final_candidate
from activeview.scripts.start_experiment import start_experiment
from activeview.research.test_gate import TestGateError, assert_test_allowed
from activeview.research.validator import validate_experiment


def _complete_provenance(tmp_path):
    files = []
    for index in range(10):
        path = tmp_path / f"artifact-{index}.dat"; path.write_text(str(index), encoding="utf-8"); files.append(path)
    def artifact(path):
        return {"path": str(path), "exists": True, "sha256": file_sha256(path)}
    return {
        "stage_a": {"summary": artifact(files[0]), "episodes": {s: artifact(files[1]) for s in ("train", "val", "test")}},
        "stage_b": {"summary": artifact(files[2]), "utility": {s: artifact(files[3]) for s in ("train", "val", "test")}},
        "stage_c_features": {"summary": artifact(files[4]), "features": {s: artifact(files[5]) for s in ("train", "val", "test")}, "stats": artifact(files[6])},
        "stgcn_checkpoint": artifact(files[7]), "label_mapping": artifact(files[8]),
        "record_split": {"summary": artifact(files[9]), "canonical_counts": {"train": 589, "val": 197, "test": 194}},
    }


def test_create_start_validate_finalize_and_authorize_lifecycle(tmp_path, monkeypatch):
    result = create_experiment(stage="stage_c_v1", name="demo", hypothesis="one change", core_change="change one", repo_root=tmp_path, data_root=tmp_path / "data")
    assert result["experiment_id"] == "EXP001"
    source = tmp_path / "experiments/stage_c_v1/EXP001_demo"
    manifest = load_manifest(source)
    assert manifest["experiment"]["source_dir"] == "experiments/stage_c_v1/EXP001_demo"
    assert manifest["experiment"]["runtime_dir"] == "experiments/stage_c_v1/EXP001_demo"
    manifest["provenance"] = _complete_provenance(tmp_path)
    save_manifest(source, manifest)
    config_path = source / "config.yaml"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "# human review\n", encoding="utf-8")
    monkeypatch.setattr("activeview.scripts.start_experiment.git_dirty", lambda: False)
    monkeypatch.setattr("activeview.scripts.start_experiment.git_value", lambda *args, **kwargs: "abc")
    start_experiment("EXP001", repo_root=tmp_path, require_clean=False)
    started_manifest = load_manifest(source)
    assert started_manifest["paths"]["run_config_sha256"] == file_sha256(config_path)
    assert started_manifest["paths"]["run_config_sha256"] != started_manifest["paths"]["draft_config_sha256"]
    assert started_manifest["git"]["run_commit"] == "abc"
    (source / "val_metrics.json").write_text(json.dumps({"split": "val", "recognition": {"accuracy": 0.5, "macro_f1": 0.4}, "regret": {"mean": 1.0, "median": 0.1, "p90": 2.0}, "positive_headroom_capture": 0.7}), encoding="utf-8")
    (source / "analysis.json").write_text("{}", encoding="utf-8")
    (source / "conclusion.md").write_text("# Experiment Conclusion\n\n## Observation\nMeasured.\n\n## Interpretation\nInference.\n\n## Decision\nACCEPT\n\n## Next\nNone.\n", encoding="utf-8")
    result = finalize_experiment("EXP001", decision="ACCEPT", repo_root=tmp_path, data_root=tmp_path / "data")
    assert result["status"] == "COMPLETED"
    assert load_manifest(source)["experiment"]["decision"] == "ACCEPT"
    freeze_final_candidate("EXP001", repo_root=tmp_path, data_root=tmp_path / "data")
    frozen_manifest = load_manifest(source)
    assert frozen_manifest["experiment"]["status"] == "FINAL_FROZEN"
    assert frozen_manifest["protocol"]["test_authorized"] is False
    registry = tmp_path / "experiments/stage_c_v1/EXPERIMENT_REGISTRY.csv"
    assert validate_experiment(source, registry_path=registry, data_root=tmp_path / "data")["passed"]
    tracked_manifest_before_authorize = file_sha256(source / "run_manifest.json")
    registry_before_authorize = registry.read_text(encoding="utf-8")
    monkeypatch.setattr("activeview.scripts.authorize_final_test.git_dirty", lambda: False)
    monkeypatch.setattr("activeview.scripts.authorize_final_test.git_value", lambda *args, **kwargs: "def")
    from activeview.scripts.authorize_final_test import authorize_final_test
    authorize_final_test("EXP001", repo_root=tmp_path, data_root=tmp_path / "data", confirm_final_model_frozen=True, require_clean=False)
    manifest = load_manifest(source)
    authorization_path = tmp_path / "data/experiments/stage_c_v1/EXP001_demo/final_test_authorization.json"
    assert_test_allowed(manifest, authorization_path=authorization_path, config_path=source / "config.yaml", current_commit="def")
    assert file_sha256(source / "run_manifest.json") == tracked_manifest_before_authorize
    assert registry.read_text(encoding="utf-8") == registry_before_authorize
    with pytest.raises(TestGateError, match="frozen_commit_mismatch"):
        assert_test_allowed(manifest, authorization_path=authorization_path, config_path=source / "config.yaml", current_commit="abc")
    (source / "config.yaml").write_text((source / "config.yaml").read_text(encoding="utf-8") + "# mutation\n", encoding="utf-8")
    with pytest.raises(TestGateError, match="config_hash_mismatch"):
        assert_test_allowed(manifest, authorization_path=authorization_path, config_path=source / "config.yaml", current_commit="def")


def test_exp001_research_record_is_planned_and_not_started():
    """The checked-in research record must remain a non-running plan."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "experiments/stage_c_v1/EXP001_gap_aware_ranking"
    assert source.is_dir()
    manifest = load_manifest(source)
    assert manifest["experiment"]["status"] == "PLANNED"
    assert manifest["protocol"]["test_used"] is False
    assert manifest["git"]["start_commit"] is None
    assert manifest["git"]["run_commit"] is None
