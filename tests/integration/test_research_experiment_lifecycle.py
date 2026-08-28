import json

from activeview.active_view.utility_label_builder import file_sha256
from activeview.research.manifest import load_manifest, save_manifest
from activeview.scripts.create_experiment import create_experiment
from activeview.scripts.finalize_experiment import finalize_experiment
from activeview.scripts.start_experiment import start_experiment
from activeview.research.test_gate import assert_test_allowed


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


def test_create_start_validate_finalize_lifecycle(tmp_path):
    result = create_experiment(stage="stage_c_v1", name="demo", hypothesis="one change", repo_root=tmp_path, data_root=tmp_path / "data")
    assert result["experiment_id"] == "EXP001"
    source = tmp_path / "experiments/stage_c_v1/EXP001_demo"
    manifest = load_manifest(source)
    manifest["provenance"] = _complete_provenance(tmp_path)
    save_manifest(source, manifest)
    start_experiment("EXP001", repo_root=tmp_path, require_clean=False)
    (source / "val_metrics.json").write_text(json.dumps({"split": "val", "recognition": {"accuracy": 0.5, "macro_f1": 0.4}, "regret": {"mean": 1.0, "median": 0.1, "p90": 2.0}, "positive_headroom_capture": 0.7}), encoding="utf-8")
    (source / "analysis.json").write_text("{}", encoding="utf-8")
    (source / "conclusion.md").write_text("# conclusion\n", encoding="utf-8")
    result = finalize_experiment("EXP001", decision="ACCEPT", repo_root=tmp_path)
    assert result["status"] == "COMPLETED"
    assert load_manifest(source)["experiment"]["decision"] == "ACCEPT"


def test_no_real_experiment_is_created_in_repository():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    assert not [path for path in (repo_root / "experiments/stage_c_v1").glob("EXP*_*") if path.is_dir()]
