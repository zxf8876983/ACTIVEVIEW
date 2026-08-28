from activeview.research.registry import REGISTRY_FIELDS, get_experiment, list_experiments, next_experiment_id, register_experiment, update_experiment


def _row(experiment_id):
    return {"experiment_id": experiment_id, "name": experiment_id, "stage": "stage_c_v1", "status": "PLANNED"}


def test_registry_ids_are_monotonic_and_updates_preserve_rows(tmp_path):
    path = tmp_path / "registry.csv"
    assert next_experiment_id(path) == "EXP001"
    register_experiment(path, _row("EXP001"))
    register_experiment(path, _row("EXP003"))
    assert next_experiment_id(path) == "EXP004"
    update_experiment(path, "EXP001", {"status": "RUNNING"})
    rows = list_experiments(path)
    assert [row["experiment_id"] for row in rows] == ["EXP001", "EXP003"]
    assert get_experiment(path, "EXP001")["status"] == "RUNNING"
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == REGISTRY_FIELDS


def test_duplicate_registry_id_is_rejected(tmp_path):
    path = tmp_path / "registry.csv"
    register_experiment(path, _row("EXP001"))
    try:
        register_experiment(path, _row("EXP001"))
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("duplicate ID was accepted")
