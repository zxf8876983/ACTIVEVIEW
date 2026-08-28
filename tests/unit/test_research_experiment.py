from activeview.research.experiment import Decision, Experiment, ExperimentStatus


def _experiment(**overrides):
    values = {
        "experiment_id": "EXP001", "name": "demo", "stage": "stage_c_v1",
        "hypothesis": "h", "source_dir": "/tmp/source", "runtime_dir": "/tmp/runtime",
        "created_at": "2026-01-01T00:00:00Z",
    }
    values.update(overrides)
    return Experiment(**values)


def test_experiment_round_trip_and_enum_validation():
    experiment = _experiment(status=ExperimentStatus.PLANNED, decision=Decision.NA)
    restored = Experiment.from_dict(experiment.to_dict())
    assert restored.experiment_id == "EXP001"
    assert restored.to_dict()["status"] == "PLANNED"


def test_invalid_status_and_id_are_rejected():
    try:
        _experiment(status="BROKEN").validate()
    except ValueError as error:
        assert "status" in str(error)
    else:
        raise AssertionError("invalid status was accepted")
    try:
        _experiment(experiment_id="EXP").validate()
    except ValueError as error:
        assert "experiment_id" in str(error)
    else:
        raise AssertionError("invalid ID was accepted")
