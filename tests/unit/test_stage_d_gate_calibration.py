import json

import numpy as np
import pytest

from activeview.scripts.analyze_stage_d_gate_calibration import build_parser
from activeview.active_view.stage_d_gate_calibration import (
    _threshold_candidates,
    build_thresholded_prediction_rows,
    calibrate_train_threshold,
    candidate_identity_audit,
    gate_metrics,
    load_calibration_artifact,
    select_train_threshold,
    validate_exp017_split,
)
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories


def _cache_row(episode_id: str, *, split: str = "train") -> dict:
    return {
        "episode_id": episode_id,
        "policy_split": split,
        "remaining_candidate_ids": [2, 3],
        "second_step_utility_targets": [1.0, -1.0],
        "second_step_candidate_geodesic": [2.0, 1.0],
    }


def _prediction_row(
    episode_id: str,
    values: list[float],
    *,
    split: str | None = None,
    stays: bool = True,
    selected_id: int | None = None,
) -> dict:
    row = {
        "episode_id": episode_id,
        "remaining_candidate_ids": [2, 3],
        "predicted_utilities": values,
        "predicted_stays": stays,
        "predicted_candidate_viewpoint_id": selected_id,
    }
    if split is not None:
        row["policy_split"] = split
    return row


def test_strict_threshold_semantics_and_tau_changes_only_gate():
    rows = [_prediction_row("e", [0.5, 0.5])]
    cache = [_cache_row("e")]
    assert build_thresholded_prediction_rows(rows, cache, 0.49)[0]["predicted_stays"] is False
    assert build_thresholded_prediction_rows(rows, cache, 0.5)[0]["predicted_stays"] is True
    low = build_thresholded_prediction_rows(rows, cache, -1.0)
    high = build_thresholded_prediction_rows(rows, cache, 1.0)
    assert low[0]["predicted_candidate_viewpoint_id"] == 3
    assert high[0]["predicted_candidate_viewpoint_id"] is None


def test_candidate_identity_is_unchanged_when_both_thresholds_move():
    rows = [_prediction_row("e", [0.5, 0.2])]
    cache = [_cache_row("e")]
    zero = build_thresholded_prediction_rows(rows, cache, 0.0)
    calibrated = build_thresholded_prediction_rows(rows, cache, -1.0)
    audit = candidate_identity_audit(zero, calibrated)
    assert audit["both_move_episode_count"] == 1
    assert audit["candidate_identity_mismatch_count"] == 0
    assert audit["candidate_identity_unchanged"] is True


def test_frozen_v0_stay_never_becomes_second_step_move():
    stage_b = [
        {
            "episode_id": "stay",
            "record_id": "record-0",
            "policy_split": "val",
            "scene_id": "scene-0",
            "region": "bedroom",
            "label_id": 0,
            "current": {"viewpoint_id": 0, "predicted_label_id": 0},
            "candidates": [{"viewpoint_id": 1, "predicted_label_id": 0, "geodesic_distance_m": 1.0, "utility": 5.0}],
            "oracle": {"safe_oracle_utility": 5.0, "safe_oracle_stays": False},
        }
    ]
    v0 = [{"episode_id": "stay", "predicted_stays": True, "predicted_candidate_viewpoint_id": None}]
    for tau in (-10.0, 0.0, 10.0):
        trajectories = build_stage_d_trajectories(stage_b, v0, [], [])
        assert trajectories[0]["moves"] == 0
        assert trajectories[0]["selected_viewpoint_id"] == 0


def test_train_calibration_rejects_val_and_test_rows():
    with pytest.raises(ValueError, match="non-train"):
        calibrate_train_threshold(
            [_prediction_row("e", [1.0, -1.0], split="val")],
            [_cache_row("e", split="train")],
        )
    with pytest.raises(ValueError, match="requires val"):
        validate_exp017_split("test", "val")


def test_calibration_uses_train_rows_and_writes_frozen_artifact_contract():
    predictions = [
        _prediction_row("a", [2.0, 0.0], split="train"),
        _prediction_row("b", [-2.0, -3.0], split="train"),
    ]
    caches = [_cache_row("a"), _cache_row("b")]
    artifact = calibrate_train_threshold(predictions, caches)
    assert artifact["split"] == "train"
    assert artifact["test_used"] is False
    assert np.isfinite(artifact["selected_tau"])
    assert artifact["train_gate_metrics"]["episode_count"] == 2


def test_threshold_candidates_include_zero_and_safe_all_move_all_stay_boundaries():
    candidates = _threshold_candidates([-1.0, 1.0])
    assert 0.0 in candidates
    assert any(value < -1.0 for value in candidates)
    assert any(value > 1.0 for value in candidates)
    assert candidates == sorted(set(candidates))


def test_gate_metrics_confusion_and_strict_move_rule():
    metrics = gate_metrics([0.0, 0.5, -0.5], [False, True, False], 0.0)
    assert metrics["confusion"] == {
        "learned_stay_oracle_stay": 2,
        "learned_stay_oracle_move": 0,
        "learned_move_oracle_stay": 0,
        "learned_move_oracle_move": 1,
    }
    assert metrics["balanced_accuracy"] == 1.0


def test_threshold_tie_breaking_is_deterministic():
    scores = [-1.0, 1.0]
    labels = [False, True]
    first = select_train_threshold(scores, labels)
    second = select_train_threshold(scores, labels)
    assert first[0] == second[0]
    assert first[1] == second[1]


def test_threshold_tie_breaking_prefers_move_f1_then_zero_distance():
    # Both all-Move and all-Stay have BA=0.5; Move-F1 selects the tiny
    # negative all-Move boundary. With only true Stay labels, zero-distance
    # tie-breaking selects tau=max(score), not the nextafter boundary.
    tau, _, _ = select_train_threshold([0.0, 1.0], [True, False])
    assert tau == np.nextafter(0.0, -np.inf)
    stay_tau, _, _ = select_train_threshold([-1.0, 1.0], [False, False])
    assert stay_tau == np.nextafter(1.0, np.inf)


def test_calibration_artifact_rejects_test_or_invalid_schema(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"experiment_id": "EXP017", "split": "train", "selected_tau": 0.0, "test_used": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test_used=false"):
        load_calibration_artifact(path)


def test_exp017_cli_has_no_test_split_entry_point():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--split", "test"])
