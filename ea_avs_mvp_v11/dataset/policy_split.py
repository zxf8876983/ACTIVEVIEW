"""Deterministic action-sample split for the v11.5 active-view policy."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SPLITS: Tuple[str, ...] = ("train", "val", "test")
# Canonical v11.5 policy split: 60% train, 20% validation, 20% test.
RATIOS: Mapping[str, float] = {"train": 0.60, "val": 0.20, "test": 0.20}


def _allocate_counts(size: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    """Allocate an integer class count using largest remainders."""
    if size < 0:
        raise ValueError("size must be non-negative")
    raw = {name: size * float(ratios[name]) for name in SPLITS}
    counts = {name: int(raw[name]) for name in SPLITS}
    remaining = size - sum(counts.values())
    order = sorted(SPLITS, key=lambda name: (raw[name] - counts[name], -SPLITS.index(name)), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def build_policy_splits(
    records: Sequence[Mapping[str, Any]], *, seed: int = 42,
    ratios: Mapping[str, float] = RATIOS,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split unique action records stratified by ``action_label``.

    Scene and region are deliberately absent from the unit of splitting.  A
    record is assigned exactly once, so every later scene/region replica keeps
    the same policy split.
    """
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("ratios must contain train/val/test and sum to one")
    unique: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        record_id = str(record.get("record_id", ""))
        label = str(record.get("action_label", ""))
        if not record_id or not label:
            raise ValueError("Every record needs record_id and action_label")
        previous = unique.get(record_id)
        if previous is not None:
            if str(previous.get("action_label")) != label:
                raise ValueError(f"record_id has conflicting labels: {record_id}")
            if int(previous.get("label_id", -1)) != int(record.get("label_id", -1)):
                raise ValueError(f"record_id has conflicting label IDs: {record_id}")
        unique[record_id] = record

    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in unique.values():
        grouped[str(record["action_label"])].append(record)

    rng = random.Random(seed)
    output: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SPLITS}
    for label in sorted(grouped):
        group = sorted(grouped[label], key=lambda item: str(item["record_id"]))
        rng.shuffle(group)
        counts = _allocate_counts(len(group), ratios)
        cursor = 0
        for split in SPLITS:
            for source in group[cursor: cursor + counts[split]]:
                item = {
                    "record_id": str(source["record_id"]),
                    "action_label": str(source["action_label"]),
                    "label_id": int(source["label_id"]),
                    "policy_split": split,
                }
                output[split].append(item)
            cursor += counts[split]
    for split in SPLITS:
        output[split].sort(key=lambda item: str(item["record_id"]))
    return output


def audit_policy_splits(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Return overlap and same-record assignment audits."""
    memberships = {name: {str(item["record_id"]) for item in splits.get(name, [])} for name in SPLITS}
    overlaps = {
        f"{left}_{right}": bool(memberships[left] & memberships[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    }
    assignments: Dict[str, str] = {}
    labels: Dict[str, Tuple[str, int]] = {}
    consistent = True
    label_consistent = True
    label_id_consistent = True
    for split in SPLITS:
        for item in splits.get(split, []):
            record_id = str(item["record_id"])
            previous = assignments.setdefault(record_id, split)
            consistent = consistent and previous == split
            current_label = str(item.get("action_label", ""))
            current_label_id = int(item.get("label_id", -1))
            previous_label = labels.setdefault(record_id, (current_label, current_label_id))
            label_consistent = label_consistent and previous_label[0] == current_label
            label_id_consistent = label_id_consistent and previous_label[1] == current_label_id
    return {
        "train_val_overlap": overlaps["train_val"],
        "train_test_overlap": overlaps["train_test"],
        "val_test_overlap": overlaps["val_test"],
        "split_overlap": any(overlaps.values()),
        "same_record_same_split": consistent,
        "same_record_same_label": label_consistent,
        "same_record_same_label_id": label_id_consistent,
        "unique_record_count": len(assignments),
    }


def write_policy_splits(
    records: Sequence[Mapping[str, Any]], output_dir: Path, *, seed: int = 42,
    ratios: Mapping[str, float] = RATIOS,
) -> Dict[str, Any]:
    """Build and persist split JSON files plus an audit summary."""
    splits = build_policy_splits(records, seed=seed, ratios=ratios)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_dir / f"{split}.json").write_text(
            json.dumps(splits[split], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    per_class: Dict[str, Dict[str, int]] = defaultdict(dict)
    for split in SPLITS:
        for item in splits[split]:
            label = str(item["action_label"])
            per_class[label][split] = per_class[label].get(split, 0) + 1
    for counts in per_class.values():
        for split in SPLITS:
            counts.setdefault(split, 0)
    audit = audit_policy_splits(splits)
    summary = {
        "protocol": "ACTIVEVIEW v11.5 Stage A policy split",
        "seed": seed,
        "split_ratios": dict(ratios),
        "input_sample_count": sum(len(splits[name]) for name in SPLITS),
        "split_counts": {name: len(splits[name]) for name in SPLITS},
        "per_class_split_counts": dict(sorted(per_class.items())),
        "overlap_audit": audit,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def load_policy_split_summary(split_dir: Path) -> Dict[str, Any]:
    """Load the persisted split metadata, including the actual ratios used."""
    path = split_dir / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("split_ratios"), Mapping):
        raise ValueError(f"Missing split_ratios in {path}")
    ratios = {str(name): float(value) for name, value in payload["split_ratios"].items()}
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError(f"Invalid split_ratios in {path}: {ratios}")
    payload["split_ratios"] = ratios
    return payload


def validate_split_summary_against_files(
    summary: Mapping[str, Any],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Reject split metadata that disagrees with the serialized split files."""
    actual_counts = {split: len(splits.get(split, [])) for split in SPLITS}
    declared_counts = summary.get("split_counts")
    if not isinstance(declared_counts, Mapping):
        raise ValueError("split summary is missing split_counts")
    normalized_declared = {split: int(declared_counts.get(split, -1)) for split in SPLITS}
    if normalized_declared != actual_counts:
        raise ValueError(
            f"split_counts do not match files: declared={normalized_declared}, actual={actual_counts}"
        )

    record_ids = {
        str(item["record_id"])
        for split in SPLITS
        for item in splits.get(split, [])
    }
    declared_input_count = int(summary.get("input_sample_count", -1))
    if declared_input_count != len(record_ids):
        raise ValueError(
            "input_sample_count does not match unique record IDs: "
            f"declared={declared_input_count}, actual={len(record_ids)}"
        )

    declared_ratios = summary.get("split_ratios")
    if not isinstance(declared_ratios, Mapping):
        raise ValueError("split summary is missing split_ratios")
    normalized_ratios = {split: float(declared_ratios.get(split, float("nan"))) for split in SPLITS}
    if any(not math.isclose(normalized_ratios[split], float(RATIOS[split]), rel_tol=0.0, abs_tol=1e-12) for split in SPLITS):
        raise ValueError(
            f"split_ratios do not match canonical RATIOS: declared={normalized_ratios}, canonical={dict(RATIOS)}"
        )

    actual_per_class: Dict[str, Dict[str, int]] = defaultdict(dict)
    for split in SPLITS:
        for item in splits.get(split, []):
            label = str(item["action_label"])
            actual_per_class[label][split] = actual_per_class[label].get(split, 0) + 1
    for counts in actual_per_class.values():
        for split in SPLITS:
            counts.setdefault(split, 0)
    declared_per_class = summary.get("per_class_split_counts")
    normalized_per_class: Dict[str, Dict[str, int]] = {}
    if isinstance(declared_per_class, Mapping):
        for label, counts in declared_per_class.items():
            if not isinstance(counts, Mapping):
                raise ValueError(f"Invalid per_class_split_counts entry for {label}")
            normalized_per_class[str(label)] = {
                split: int(counts.get(split, 0)) for split in SPLITS
            }
    if normalized_per_class != dict(sorted(actual_per_class.items())):
        raise ValueError(
            "per_class_split_counts do not match files: "
            f"declared={normalized_per_class}, actual={dict(sorted(actual_per_class.items()))}"
        )


def load_policy_splits(split_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load and validate persisted policy split files."""
    summary = load_policy_split_summary(split_dir)
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for split in SPLITS:
        path = split_dir / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list in {path}")
        splits[split] = [dict(item) for item in payload]
    audit = audit_policy_splits(splits)
    if (
        audit["split_overlap"]
        or not audit["same_record_same_split"]
        or not audit["same_record_same_label"]
        or not audit["same_record_same_label_id"]
    ):
        raise ValueError(f"Invalid policy split overlap: {audit}")
    validate_split_summary_against_files(summary, splits)
    return splits
