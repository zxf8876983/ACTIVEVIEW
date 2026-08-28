import pytest

from ea_avs_mvp_v11.active_view.stage_c_failure_analysis import analyze_rows, prepare_aligned_rows


def _row(
    episode_id: str,
    record_id: str,
    *,
    safe_stays: bool,
    predicted_stays: bool,
    predicted_id: int = 1,
    safe_id: int = 2,
    regret: float = 0.2,
) -> dict:
    return {
        "episode_id": episode_id,
        "record_id": record_id,
        "policy_split": "test",
        "region": "bedroom",
        "label_id": 0,
        "action_label": "sit",
        "current_predicted_label_id": 0,
        "current_entropy": 0.2,
        "current_margin": 0.7,
        "current_pose_confidence": 0.8,
        "current_logp_true": -0.2,
        "predicted_stays": predicted_stays,
        "predicted_action": "stay" if predicted_stays else f"candidate:{predicted_id}",
        "predicted_candidate_viewpoint_id": predicted_id,
        "safe_oracle_stays": safe_stays,
        "safe_oracle_action": "stay" if safe_stays else f"candidate:{safe_id}",
        "safe_oracle_viewpoint_id": safe_id,
        "candidate_oracle_viewpoint_id": safe_id,
        "candidate_oracle_utility": 1.0,
        "safe_oracle_utility": 1.0,
        "selected_true_utility": 1.0 - regret,
        "selected_predicted_label_id": 0,
        "safe_oracle_predicted_label_id": 0,
        "regret": regret,
        "candidate_viewpoint_ids": [1, 2],
        "predicted_utilities": [0.4, 0.3],
        "utility_targets": [0.2, 1.0],
        "candidate_geometry": [
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.5, 1.5, 0.0],
            [-1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.5, 1.5, 0.0],
        ],
        "current_feature": [0.0] * 273 + [0.7, 0.8],
    }


def test_failure_taxonomy_and_regret_groups_are_disjoint():
    rows = [
        _row("e0", "r0", safe_stays=False, predicted_stays=True, regret=0.0),
        _row("e1", "r1", safe_stays=True, predicted_stays=False, regret=0.2),
        _row("e2", "r2", safe_stays=False, predicted_stays=False, predicted_id=1, safe_id=2, regret=0.2),
        _row("e3", "r3", safe_stays=False, predicted_stays=False, predicted_id=2, safe_id=2, regret=0.0),
    ]
    summary = analyze_rows(rows, ["sit"])
    taxonomy = summary["failure_taxonomy"]
    assert taxonomy["A_missed_move"]["count"] == 1
    assert taxonomy["B_unnecessary_move"]["count"] == 1
    assert taxonomy["C2_wrong_high_utility_loss"]["count"] == 1
    assert taxonomy["D_correct_safe_action"]["count"] == 1
    assert sum(item["count"] for item in taxonomy.values()) == len(rows)
    assert summary["regret"]["G0_near_optimal"]["count"] == 2


def test_candidate_set_gap_and_safe_ratio_are_reported():
    rows = [_row("e0", "r0", safe_stays=False, predicted_stays=False, predicted_id=1, safe_id=2, regret=0.1)]
    summary = analyze_rows(rows, ["sit"])
    assert summary["candidate_set_difficulty"]["overall"]["top1_top2_gap"]["mean"] == pytest.approx(0.8)
    assert summary["candidate_miss"]["miss_count"] == 1
    assert summary["candidate_miss"]["ratio_thresholds"]["selected_at_least_50pct_oracle"] == pytest.approx(1.0)


def test_candidate_set_gap_bins_are_a_partition():
    rows = []
    for index, gap in enumerate((0.1, 0.2, 0.3, 0.4)):
        row = _row(
            f"e{index}",
            f"r{index}",
            safe_stays=False,
            predicted_stays=False,
            predicted_id=2,
            safe_id=2,
            regret=0.0,
        )
        row["utility_targets"] = [0.0, gap]
        rows.append(row)
    difficulty = analyze_rows(rows, ["sit"])["candidate_set_difficulty"]
    assert difficulty["bins_partition"] is True
    assert difficulty["bin_count_sum"] == len(rows)
    assert sum(difficulty[name]["count"] for name in ("very_small", "small", "medium", "large")) == len(rows)


def test_symmetric_analysis_reports_baseline_and_enrichment_ratio():
    rows = [
        _row(
            f"e{index}",
            f"r{index}",
            safe_stays=False,
            predicted_stays=False,
            predicted_id=1,
            safe_id=2,
            regret=10.0 if index == 0 else 0.0,
        )
        for index in range(10)
    ]
    symmetric = analyze_rows(rows, ["sit"])["symmetric_geometry_ambiguity"]
    assert symmetric["high_regret_baseline_rate"] == pytest.approx(0.1)
    assert symmetric["high_regret_given_ambiguity_rate"] == pytest.approx(0.1)
    assert symmetric["enrichment_ratio"] == pytest.approx(1.0)


def test_symmetric_geometry_uses_candidate_radius_not_current_radius():
    different_candidate_radii = _row(
        "e-radius-different",
        "r-radius-different",
        safe_stays=False,
        predicted_stays=False,
        predicted_id=1,
        safe_id=2,
        regret=0.0,
    )
    different_candidate_radii["candidate_geometry"][0][9] = 1.5
    different_candidate_radii["candidate_geometry"][1][9] = 4.5
    different = analyze_rows([different_candidate_radii], ["sit"])["symmetric_geometry_ambiguity"]
    assert different["candidate_pair_count"] == 0

    similar_candidate_radii = _row(
        "e-radius-similar",
        "r-radius-similar",
        safe_stays=False,
        predicted_stays=False,
        predicted_id=1,
        safe_id=2,
        regret=0.0,
    )
    similar_candidate_radii["candidate_geometry"][0][9] = 2.0
    similar_candidate_radii["candidate_geometry"][1][9] = 2.1
    similar = analyze_rows([similar_candidate_radii], ["sit"])["symmetric_geometry_ambiguity"]
    assert similar["candidate_pair_count"] == 1


def test_record_aggregation_does_not_treat_episodes_as_records():
    rows = [
        _row("e0", "r0", safe_stays=False, predicted_stays=False, predicted_id=2, safe_id=2, regret=0.1),
        _row("e1", "r0", safe_stays=False, predicted_stays=False, predicted_id=1, safe_id=2, regret=0.9),
        _row("e2", "r1", safe_stays=True, predicted_stays=True, regret=0.0),
    ]
    summary = analyze_rows(rows, ["sit"])
    assert summary["record_level"]["record_count"] == 2
    assert summary["record_level"]["all_records"][0]["record_id"] == "r0"


def test_regret_percentile_thresholds_are_derived_from_rows():
    rows = [_row(f"e{i}", f"r{i}", safe_stays=False, predicted_stays=False, predicted_id=2, safe_id=2, regret=float(i)) for i in range(1, 6)]
    summary = analyze_rows(rows, ["sit"])
    assert summary["regret"]["thresholds"]["median"] == pytest.approx(3.0)
    assert summary["regret"]["thresholds"]["p90"] == pytest.approx(4.6)


def test_near_zero_safe_oracle_does_not_create_invalid_ratio():
    row = _row("e0", "r0", safe_stays=True, predicted_stays=True, regret=0.0)
    row["safe_oracle_utility"] = 5e-8
    summary = analyze_rows([row], ["sit"])
    assert summary["overall"]["headroom"]["positive_episode_count"] == 0
    assert summary["overall"]["headroom"]["aggregate_capture"] == 0.0


def test_alignment_rejects_duplicate_episode_ids():
    source = [{"episode_id": "a"}, {"episode_id": "a"}]
    with pytest.raises(ValueError, match="duplicate Stage A episode_id"):
        prepare_aligned_rows(source, source, source, source)


def test_alignment_rejects_missing_or_reordered_episode():
    with pytest.raises(ValueError, match="alignment mismatch"):
        prepare_aligned_rows(
            [{"episode_id": "a"}],
            [{"episode_id": "b"}],
            [{"episode_id": "a"}],
            [{"episode_id": "a"}],
        )
