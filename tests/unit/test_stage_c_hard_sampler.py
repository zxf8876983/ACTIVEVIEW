import pytest

from activeview.data.preprocessing.policy_data import HardRecordAwareSampler


def _rows(record_ids, episodes_per_record=20):
    rows = []
    for record_id in record_ids:
        rows.extend({"record_id": record_id} for _ in range(episodes_per_record))
    return rows


def _record_counts(rows, indices):
    counts = {record_id: 0 for record_id in {row["record_id"] for row in rows}}
    for index in indices:
        counts[rows[index]["record_id"]] += 1
    return counts


def test_hard_sampler_uses_separate_exposure_counts():
    rows = _rows(("hard", "normal"))
    sampler = HardRecordAwareSampler(rows, ("hard",), hard_episodes_per_record=32, normal_episodes_per_record=12, seed=7)
    indices = list(sampler)
    assert len(indices) == 44
    assert _record_counts(rows, indices) == {"hard": 32, "normal": 12}


def test_hard_sampler_rejects_unknown_record_ids():
    with pytest.raises(ValueError, match="absent from the Train dataset"):
        HardRecordAwareSampler(_rows(("known",)), ("missing",))


def test_hard_sampler_is_deterministic_per_epoch_and_changes_across_epochs():
    rows = _rows(("hard", "normal"), episodes_per_record=40)
    sampler = HardRecordAwareSampler(rows, ("hard",), seed=11)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert first == list(sampler)
    sampler.set_epoch(4)
    assert first != list(sampler)


def test_hard_sampler_total_volume_matches_record_balanced_baseline():
    rows = _rows(tuple(f"r-{index}" for index in range(10)))
    hard = HardRecordAwareSampler(rows, tuple(f"r-{index}" for index in range(2)), hard_episodes_per_record=32, normal_episodes_per_record=12)
    baseline_count = 10 * 16
    assert abs(len(hard) - baseline_count) <= 4
