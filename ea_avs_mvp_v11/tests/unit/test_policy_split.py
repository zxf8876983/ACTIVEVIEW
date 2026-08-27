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
