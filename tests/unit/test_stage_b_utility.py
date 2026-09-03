import numpy as np

from activeview.data.generation.utility_labels import (
    build_utility_record,
    summarize_utility_records,
)


def _episode():
    return {
        "episode_id": "episode-1",
        "record_id": "record-1",
        "policy_split": "train",
        "scene_id": "scene-1",
        "region": "bedroom",
        "label_id": 1,
        "current_view": {"viewpoint_id": 0},
        "candidate_pool": [
            {"viewpoint_id": 2, "geodesic_distance_m": 2.0},
            {"viewpoint_id": 1, "geodesic_distance_m": 1.0},
        ],
    }


def test_utility_uses_true_class_log_probability_and_deterministic_tie_break():
    log_probs = {
        0: np.log(np.asarray([0.1, 0.7, 0.2], dtype=np.float64)),
        1: np.log(np.asarray([0.1, 0.7, 0.2], dtype=np.float64)),
        2: np.log(np.asarray([0.1, 0.7, 0.2], dtype=np.float64)),
    }
    record = build_utility_record(_episode(), log_probs)
    assert record["candidates"][0]["utility"] == 0.0
    assert record["candidates"][1]["utility"] == 0.0
    assert record["oracle"]["candidate_oracle_viewpoint_id"] == 1
    assert record["oracle"]["safe_oracle_viewpoint_id"] == 0
    assert record["oracle"]["safe_oracle_stays"]


def test_summary_reports_rescue_and_positive_headroom():
    log_probs = {
        0: np.log(np.asarray([0.8, 0.1, 0.1], dtype=np.float64)),
        1: np.log(np.asarray([0.1, 0.8, 0.1], dtype=np.float64)),
        2: np.log(np.asarray([0.1, 0.9, 0.0 + 1e-8], dtype=np.float64)),
    }
    record = build_utility_record(_episode(), log_probs)
    summary = summarize_utility_records({"train": [record], "val": [], "test": []}, ["a", "b", "c"])
    assert summary["train"]["headroom"]["positive_headroom_episode_count"] == 1
    assert summary["train"]["rescue"]["rescue_count"] == 1
    assert summary["train"]["policies"]["SafeOracle"]["accuracy"] == 1.0


def test_near_zero_headroom_is_not_positive_and_degradation_is_conditional():
    episode = _episode()
    log_probs = {
        0: np.log(np.asarray([0.1, 0.7, 0.2], dtype=np.float64)),
        1: np.log(np.asarray([0.1, 0.7000002, 0.1999998], dtype=np.float64)),
        2: np.log(np.asarray([0.1, 0.7000003, 0.1999997], dtype=np.float64)),
    }
    record = build_utility_record(episode, log_probs)
    summary = summarize_utility_records({"train": [record], "val": [], "test": []}, ["a", "b", "c"])
    headroom = summary["train"]["headroom"]
    assert headroom["positive_headroom_episode_count"] == 0
    assert headroom["near_zero_ratio"] == 1.0
    assert headroom["candidate_pair_utility"]["positive_ratio"] == 0.0
    assert headroom["candidate_pair_utility"]["near_zero_ratio"] == 1.0
    rescue = summary["train"]["rescue"]
    assert rescue["current_correct_count"] == 1
    assert rescue["degradation_rate_among_current_correct"] == 0.0
