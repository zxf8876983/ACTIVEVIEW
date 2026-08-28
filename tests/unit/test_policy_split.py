import json

import pytest

from activeview.dataset.policy_split import (
    RATIOS,
    audit_policy_splits,
    build_policy_splits,
    load_policy_split_summary,
    validate_split_summary_against_files,
)


def _records():
    return [
        {"record_id": f"r{i:02d}", "action_label": "sit" if i < 10 else "lie", "label_id": 0 if i < 10 else 1}
        for i in range(20)
    ]


def test_policy_split_is_deterministic_and_disjoint():
    first = build_policy_splits(_records(), seed=42)
    second = build_policy_splits(_records(), seed=42)
    assert first == second
    audit = audit_policy_splits(first)
    assert not audit["split_overlap"]
    assert audit["same_record_same_split"]


def test_record_keeps_one_split():
    splits = build_policy_splits(_records(), seed=42)
    memberships = {}
    for split, records in splits.items():
        for record in records:
            memberships.setdefault(record["record_id"], set()).add(split)
    assert memberships
    assert all(value == {next(iter(value))} for value in memberships.values())


def test_record_id_cannot_have_conflicting_label_id():
    records = _records()
    records.append({"record_id": "r00", "action_label": "sit", "label_id": 99})
    with pytest.raises(ValueError, match="conflicting label IDs"):
        build_policy_splits(records)


def test_split_audit_reports_label_id_conflict():
    splits = {
        "train": [{"record_id": "r", "action_label": "sit", "label_id": 0}],
        "val": [{"record_id": "r", "action_label": "sit", "label_id": 1}],
        "test": [],
    }
    audit = audit_policy_splits(splits)
    assert not audit["same_record_same_label_id"]


def test_canonical_policy_split_is_six_two_two():
    assert RATIOS == {"train": 0.60, "val": 0.20, "test": 0.20}


def test_episode_builder_source_summary_ratios_are_preserved(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({
        "split_ratios": {"train": 0.60, "val": 0.20, "test": 0.20},
    }), encoding="utf-8")
    summary = load_policy_split_summary(tmp_path)
    assert summary["split_ratios"] == RATIOS


def test_split_summary_validation_rejects_count_mismatch():
    splits = {
        "train": [{"record_id": "a", "action_label": "sit", "label_id": 0}],
        "val": [],
        "test": [],
    }
    summary = {
        "split_ratios": dict(RATIOS),
        "split_counts": {"train": 0, "val": 0, "test": 1},
        "input_sample_count": 1,
        "per_class_split_counts": {"sit": {"train": 1, "val": 0, "test": 0}},
    }
    with pytest.raises(ValueError, match="split_counts"):
        validate_split_summary_against_files(summary, splits)
