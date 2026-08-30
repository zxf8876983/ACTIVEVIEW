import pytest

from activeview.scripts.analyze_stage_d_second_step_errors import build_parser
from activeview.active_view.stage_d_error_decomposition import (
    build_exp016_variant_trajectories,
    second_step_variant_decision,
    validate_exp016_episode_alignment,
    validate_exp016_split,
)
from activeview.active_view.stage_d_evaluation import build_fixed_first_oracle


def _stage_b_record(episode_id: str, label_id: int = 0) -> dict:
    return {
        "episode_id": episode_id,
        "record_id": "record-0",
        "policy_split": "val",
        "scene_id": "scene-0",
        "region": "bedroom",
        "label_id": label_id,
        "current": {
            "viewpoint_id": 0,
            "logp_true": -1.0,
            "predicted_label_id": label_id,
            "correct": True,
            "entropy": 0.1,
        },
        "candidates": [
            {
                "viewpoint_id": 1,
                "logp_true": -1.0,
                "predicted_label_id": label_id,
                "correct": True,
                "entropy": 0.1,
                "geodesic_distance_m": 1.0,
                "utility": 0.0,
            },
            {
                "viewpoint_id": 2,
                "logp_true": -1.5,
                "predicted_label_id": label_id,
                "correct": True,
                "entropy": 0.2,
                "geodesic_distance_m": 2.0,
                "utility": -0.5,
            },
            {
                "viewpoint_id": 3,
                "logp_true": -2.0,
                "predicted_label_id": label_id,
                "correct": True,
                "entropy": 0.3,
                "geodesic_distance_m": 3.0,
                "utility": -1.0,
            },
        ],
        "oracle": {
            "candidate_oracle_viewpoint_id": 1,
            "candidate_oracle_utility": 0.0,
            "safe_oracle_viewpoint_id": 1,
            "safe_oracle_utility": 0.0,
            "safe_oracle_stays": True,
        },
    }


def _move_inputs() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    record = _stage_b_record("move")
    record["oracle"] = {
        "candidate_oracle_viewpoint_id": 2,
        "candidate_oracle_utility": -0.5,
        "safe_oracle_viewpoint_id": 2,
        "safe_oracle_utility": 0.0,
        "safe_oracle_stays": True,
    }
    v0 = [{
        "episode_id": "move",
        "predicted_stays": False,
        "predicted_candidate_viewpoint_id": 1,
        "candidate_viewpoint_ids": [1, 2, 3],
        "predicted_utilities": [1.0, 0.5, 0.1],
    }]
    cache = [{
        "episode_id": "move",
        "remaining_candidate_ids": [2, 3],
        "second_step_utility_targets": [-2.0, -1.0],
        "second_step_candidate_geodesic": [2.0, 3.0],
    }]
    exp014 = [{
        "episode_id": "move",
        "remaining_candidate_ids": [2, 3],
        "predicted_utilities": [0.5, 0.2],
        "predicted_stays": False,
        "predicted_candidate_viewpoint_id": 2,
    }]
    return [record], v0, cache, exp014


def test_v0_stay_is_frozen_for_all_variants():
    record = _stage_b_record("stay")
    record["candidates"][1]["utility"] = 2.0
    stage_b = [record]
    v0 = [{"episode_id": "stay", "predicted_stays": True, "predicted_candidate_viewpoint_id": 1}]
    variants = [("learned", "learned"), ("oracle", "learned"), ("learned", "oracle"), ("oracle", "oracle")]
    for gate, candidate in variants:
        rows, _ = build_exp016_variant_trajectories(
            stage_b_rows=stage_b, v0_prediction_rows=v0, cache_rows=[],
            exp014_prediction_rows=[], gate=gate, candidate=candidate,
        )
        assert rows[0]["moves"] == 0
        assert rows[0]["selected_viewpoint_id"] == 0


def test_oracle_gate_does_not_choose_candidate_identity():
    decision = second_step_variant_decision(
        gate="oracle", candidate="learned", learned_utilities=[10.0, -10.0],
        true_utilities=[-1.0, 2.0], candidate_ids=[2, 3], candidate_geodesics=[2.0, 3.0],
    )
    assert decision["stays"] is False
    assert decision["candidate_id"] == 2


def test_oracle_candidate_does_not_override_learned_gate_on_negative_values():
    decision = second_step_variant_decision(
        gate="learned", candidate="oracle", learned_utilities=[0.5, 0.2],
        true_utilities=[-2.0, -1.0], candidate_ids=[2, 3], candidate_geodesics=[2.0, 3.0],
    )
    assert decision["stays"] is False
    assert decision["candidate_id"] == 3


def test_fixed_first_oracle_stays_when_both_true_u2_are_negative():
    decision = second_step_variant_decision(
        gate="oracle", candidate="oracle", learned_utilities=[0.5, 0.2],
        true_utilities=[-2.0, -1.0], candidate_ids=[2, 3], candidate_geodesics=[2.0, 3.0],
    )
    assert decision["stays"] is True
    assert decision["candidate_id"] is None


def test_learned_and_learned_reproduces_exp014_action_logic():
    stage_b, v0, cache, exp014 = _move_inputs()
    rows, counters = build_exp016_variant_trajectories(
        stage_b_rows=stage_b, v0_prediction_rows=v0, cache_rows=cache,
        exp014_prediction_rows=exp014, gate="learned", candidate="learned",
    )
    assert rows[0]["moves"] == 2
    assert rows[0]["selected_viewpoint_id"] == 2
    assert counters == {"v0_move": 1, "second_move": 1}


def test_fixed_first_oracle_uses_stay_as_zero_utility():
    stage_b, v0, cache, exp014 = _move_inputs()
    rows, _ = build_exp016_variant_trajectories(
        stage_b_rows=stage_b, v0_prediction_rows=v0, cache_rows=cache,
        exp014_prediction_rows=exp014, gate="oracle", candidate="oracle",
    )
    assert rows[0]["moves"] == 1
    assert rows[0]["selected_viewpoint_id"] == 1


def test_ties_use_geodesic_then_viewpoint_id():
    decision = second_step_variant_decision(
        gate="learned", candidate="learned", learned_utilities=[1.0, 1.0],
        true_utilities=[1.0, 1.0], candidate_ids=[3, 2], candidate_geodesics=[1.0, 1.0],
    )
    assert decision["candidate_id"] == 2


def test_oracle_candidate_ties_preserve_frozen_exp015_cache_order():
    decision = second_step_variant_decision(
        gate="learned", candidate="oracle", learned_utilities=[1.0, 1.0],
        true_utilities=[1.0, 1.0], candidate_ids=[3, 2], candidate_geodesics=[10.0, 1.0],
    )
    assert decision["candidate_id"] == 3


def test_exp016_oracle_candidate_matches_frozen_fixed_first_tie_order():
    stage_b, v0, cache, exp014 = _move_inputs()
    cache[0]["remaining_candidate_ids"] = [3, 2]
    cache[0]["second_step_utility_targets"] = [1.0, 1.0]
    cache[0]["second_step_candidate_geodesic"] = [10.0, 1.0]
    exp014[0]["remaining_candidate_ids"] = [3, 2]
    exp014[0]["predicted_utilities"] = [1.0, 1.0]

    decision = second_step_variant_decision(
        gate="oracle", candidate="oracle", learned_utilities=[1.0, 1.0],
        true_utilities=cache[0]["second_step_utility_targets"],
        candidate_ids=cache[0]["remaining_candidate_ids"],
        candidate_geodesics=cache[0]["second_step_candidate_geodesic"],
    )
    frozen = build_fixed_first_oracle(stage_b, v0, cache)
    assert decision["candidate_id"] == 3
    assert frozen[0]["selected_viewpoint_id"] == 3


def test_exp016_requires_exact_episode_universe_and_second_step_subset():
    stage_b, v0, cache, exp014 = _move_inputs()
    alignment = validate_exp016_episode_alignment(
        stage_b_rows=stage_b,
        v0_prediction_rows=v0,
        cache_rows=cache,
        exp014_prediction_rows=exp014,
    )
    assert alignment["stage_b_episode_count"] == 1
    assert alignment["expected_second_step_episode_count"] == 1

    with pytest.raises(ValueError, match="Stage D cache second-step episode IDs mismatch"):
        validate_exp016_episode_alignment(
            stage_b_rows=stage_b,
            v0_prediction_rows=v0,
            cache_rows=cache + [{"episode_id": "unexpected"}],
            exp014_prediction_rows=exp014,
        )

    with pytest.raises(ValueError, match="EXP014 prediction second-step episode IDs mismatch"):
        validate_exp016_episode_alignment(
            stage_b_rows=stage_b,
            v0_prediction_rows=v0,
            cache_rows=cache,
            exp014_prediction_rows=exp014 + [{"episode_id": "unexpected"}],
        )

    with pytest.raises(ValueError, match="Stage B/v0 episode IDs mismatch"):
        validate_exp016_episode_alignment(
            stage_b_rows=stage_b,
            v0_prediction_rows=v0 + [{"episode_id": "unexpected", "predicted_stays": True}],
            cache_rows=cache,
            exp014_prediction_rows=exp014,
        )


def test_exp016_rejects_test_split():
    validate_exp016_split("val")
    with pytest.raises(ValueError, match="Val only"):
        validate_exp016_split("test")


def test_exp016_cli_does_not_accept_test_split():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--split", "test"])
