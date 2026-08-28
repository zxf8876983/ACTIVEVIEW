import pytest

from activeview.research.registry import list_experiments
from activeview.scripts import create_experiment as create_module


def test_create_rolls_back_new_directories_before_registry_registration(tmp_path, monkeypatch):
    def fail_register(*args, **kwargs):
        raise RuntimeError("simulated registry failure")

    monkeypatch.setattr(create_module, "register_experiment", fail_register)
    with pytest.raises(RuntimeError, match="simulated registry failure"):
        create_module.create_experiment(
            stage="stage_c_v1",
            name="rollback",
            hypothesis="h",
            core_change="c",
            repo_root=tmp_path,
            data_root=tmp_path / "data",
        )
    stage_root = tmp_path / "experiments" / "stage_c_v1"
    assert not [path for path in stage_root.glob("EXP*_*") if path.is_dir()]
    registry = stage_root / "EXPERIMENT_REGISTRY.csv"
    assert list_experiments(registry) == []
    monkeypatch.undo()
    result = create_module.create_experiment(
        stage="stage_c_v1",
        name="after-rollback",
        hypothesis="h",
        core_change="c",
        repo_root=tmp_path,
        data_root=tmp_path / "data",
    )
    assert result["experiment_id"] == "EXP001"
