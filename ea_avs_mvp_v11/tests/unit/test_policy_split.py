import pytest

from ea_avs_mvp_v11.dataset.policy_split import audit_policy_splits, build_policy_splits


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
