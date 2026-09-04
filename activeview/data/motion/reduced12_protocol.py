"""Diversity-aware manifest construction for the reduced 12-class protocol."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from activeview.data.motion.babel_official150_true_skeleton import (
    _annotation,
    _frame_interval,
)
from activeview.data.motion.babel_source_utils import (
    _read_source_info,
    _source_lookup,
    resolve_source_path,
)

REDUCED12_LABELS: Tuple[str, ...] = (
    "walk",
    "sit",
    "stand up",
    "bend",
    "squat",
    "lean",
    "stretch",
    "take/pick something up",
    "place something",
    "lift something",
    "clean something",
    "stumble",
)


def _identity(feat_p: str) -> Tuple[str, str]:
    """Return AMASS dataset and stable subject identity from a BABEL path."""
    parts = str(feat_p).replace("\\", "/").strip("/").split("/")
    dataset = parts[0] if parts and parts[0] else "unknown"
    if len(parts) >= 3:
        subject = f"{dataset}/{parts[2]}"
    elif len(parts) >= 2:
        subject = f"{dataset}/{parts[1]}"
    else:
        subject = dataset
    return dataset, subject


def _duration_bin(duration: float) -> int:
    """Use broad fixed duration bins so selection remains reproducible."""
    if duration < 1.0:
        return 0
    if duration < 2.0:
        return 1
    if duration < 4.0:
        return 2
    if duration < 8.0:
        return 3
    return 4


def _record_id(split: str, sid: str, label_index: int, seg_id: str, label: str) -> str:
    digest = hashlib.sha1(f"{split}|{sid}|{label_index}|{seg_id}|{label}".encode()).hexdigest()[:10]
    return f"reduced12_{split}_{int(sid):05d}_{label_index:03d}_{digest}"


def collect_reduced12_records(
    babel_path: Path,
    source_split: str,
    source_lookup: Mapping[str, Path],
    *,
    min_frames_exclusive: int = 30,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect readable, duration-valid records for the requested labels."""
    data = json.loads(babel_path.read_text(encoding="utf-8"))
    wanted = set(REDUCED12_LABELS)
    records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    source_cache: Dict[str, Any] = {}
    for sid, entry in data.items():
        annotation, level = _annotation(entry)
        if annotation is None:
            continue
        feat_p = str(entry.get("feat_p", ""))
        source_path = resolve_source_path(feat_p, source_lookup)
        if source_path is None:
            continue
        try:
            source_key = str(source_path)
            if source_key not in source_cache:
                source_cache[source_key] = _read_source_info(source_path)
            source_info = source_cache[source_key]
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
        for label_index, label_info in enumerate(annotation.get("labels", [])):
            labels = set(str(item) for item in (label_info.get("act_cat") or [])) & wanted
            if not labels:
                continue
            start_frame, end_frame = _frame_interval(
                label_info,
                level,
                float(entry.get("dur", 0.0)),
                source_info.num_frames,
                source_info.fps,
            )
            num_frames = end_frame - start_frame + 1
            if num_frames <= min_frames_exclusive:
                continue
            start_t = float(label_info.get("start_t", 0.0)) if level == "frame_level" else 0.0
            end_t = float(label_info.get("end_t", entry.get("dur", 0.0))) if level == "frame_level" else float(entry.get("dur", 0.0))
            duration = max(0.0, end_t - start_t)
            seg_id = str(label_info.get("seg_id") or f"{sid}:{label_index}")
            dataset, subject = _identity(feat_p)
            for action_label in sorted(labels):
                records.append(
                    {
                        "record_id": _record_id(source_split, str(sid), label_index, seg_id, action_label),
                        "babel_sid": int(sid),
                        "label_index": int(label_index),
                        "seg_id": seg_id,
                        "source_split": source_split,
                        "split": source_split,
                        "annotation_level": level,
                        "action_label": action_label,
                        "feat_p": feat_p,
                        "source_group": str(source_path),
                        "source_path": str(source_path),
                        "amass_dataset": dataset,
                        "subject_id": subject,
                        "source_num_frames": int(source_info.num_frames),
                        "fps": float(source_info.fps),
                        "sequence_duration": float(entry.get("dur", 0.0)),
                        "start_t": start_t,
                        "end_t": end_t,
                        "duration_seconds": duration,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "num_frames": int(num_frames),
                    }
                )
    return records, excluded


def _candidate_key(item: Mapping[str, Any]) -> Tuple[str, str]:
    return str(item["source_split"]), str(item["record_id"])


def select_diverse_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cap_per_class: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Greedily maximize source, subject, dataset and duration-bin novelty."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["action_label"])].append(dict(record))
    selected: List[Dict[str, Any]] = []
    for label in REDUCED12_LABELS:
        candidates = list(grouped.get(label, []))
        rng = random.Random(seed + REDUCED12_LABELS.index(label) * 1009)
        rng.shuffle(candidates)
        chosen: List[Dict[str, Any]] = []
        source_counts: Counter[str] = Counter()
        subject_counts: Counter[str] = Counter()
        dataset_counts: Counter[str] = Counter()
        duration_counts: Counter[int] = Counter()
        while candidates and len(chosen) < int(cap_per_class):
            def score(item: Mapping[str, Any]) -> Tuple[float, float, float, float, str]:
                source = str(item["source_group"])
                subject = str(item["subject_id"])
                dataset = str(item["amass_dataset"])
                duration = _duration_bin(float(item["duration_seconds"]))
                return (
                    float(source_counts[source] == 0) * 1_000_000.0 - source_counts[source] * 1_000.0,
                    float(subject_counts[subject] == 0) * 10_000.0 - subject_counts[subject] * 100.0,
                    float(dataset_counts[dataset] == 0) * 1_000.0 - dataset_counts[dataset] * 10.0,
                    float(duration_counts[duration] == 0) * 100.0 - duration_counts[duration],
                    str(item["record_id"]),
                )

            best = max(candidates, key=score)
            candidates.remove(best)
            chosen.append(best)
            source_counts[str(best["source_group"])] += 1
            subject_counts[str(best["subject_id"])] += 1
            dataset_counts[str(best["amass_dataset"])] += 1
            duration_counts[_duration_bin(float(best["duration_seconds"]))] += 1
        selected.extend(chosen)
    selected.sort(key=lambda item: (REDUCED12_LABELS.index(str(item["action_label"])), str(item["record_id"])))
    return selected


def _split_records(records: Sequence[Mapping[str, Any]], fractions: Sequence[float], seed: int) -> List[List[Dict[str, Any]]]:
    """Stratified deterministic split of already capped records."""
    result: List[List[Dict[str, Any]]] = [[] for _ in fractions]
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[str(record["action_label"])].append(dict(record))
    for label_index, label in enumerate(REDUCED12_LABELS):
        items = by_label.get(label, [])
        rng = random.Random(seed + label_index * 7919)
        rng.shuffle(items)
        n = len(items)
        cuts: List[int] = []
        cumulative = 0.0
        for fraction in fractions[:-1]:
            cumulative += float(fraction)
            cuts.append(int(round(n * cumulative)))
        start = 0
        for bucket, end in enumerate(cuts + [n]):
            result[bucket].extend(items[start:end])
            start = end
    for bucket in result:
        bucket.sort(key=lambda item: str(item["record_id"]))
    return result


def _with_split(records: Iterable[Mapping[str, Any]], split: str, label_to_id: Mapping[str, int]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["split"] = split
        item["label_id"] = int(label_to_id[str(item["action_label"])])
        output.append(item)
    return output


def _diversity(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    durations = [float(item["duration_seconds"]) for item in records]
    return {
        "records": len(records),
        "unique_sources": len({str(item["source_group"]) for item in records}),
        "unique_subjects": len({str(item["subject_id"]) for item in records}),
        "unique_amass_datasets": len({str(item["amass_dataset"]) for item in records}),
        "duration_bins": dict(Counter(_duration_bin(value) for value in durations)),
        "duration_seconds_total": float(sum(durations)),
        "duration_seconds_mean": float(sum(durations) / len(durations)) if durations else 0.0,
    }


def build_reduced12_protocol(
    *,
    output_root: Path,
    babel_dir: Path,
    amass_index_path: Path,
    min_frames_exclusive: int = 30,
    train_cap: int = 300,
    active_val_cap: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build independent ST-GCN-development and ActiveView manifests."""
    index = json.loads(amass_index_path.read_text(encoding="utf-8"))
    lookup = _source_lookup(index)
    train_raw, train_excluded = collect_reduced12_records(
        babel_dir / "train.json", "official_train", lookup, min_frames_exclusive=min_frames_exclusive
    )
    val_raw, val_excluded = collect_reduced12_records(
        babel_dir / "val.json", "official_val", lookup, min_frames_exclusive=min_frames_exclusive
    )
    label_to_id = {label: index for index, label in enumerate(REDUCED12_LABELS)}
    train_selected = select_diverse_records(train_raw, cap_per_class=train_cap, seed=seed)
    val_selected = select_diverse_records(val_raw, cap_per_class=active_val_cap, seed=seed + 1)
    stgcn_train, stgcn_val = _split_records(train_selected, (0.9, 0.1), seed)
    active_train, active_val, active_test = _split_records(val_selected, (0.6, 0.2, 0.2), seed + 1)
    manifests = {
        "stgcn_development": {
            "train": _with_split(stgcn_train, "train", label_to_id),
            "val": _with_split(stgcn_val, "val", label_to_id),
        },
        "activeview": {
            "train": _with_split(active_train, "train", label_to_id),
            "val": _with_split(active_val, "val", label_to_id),
            "test": _with_split(active_test, "test", label_to_id),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for subset, splits in manifests.items():
        subset_root = output_root / subset
        subset_root.mkdir(parents=True, exist_ok=True)
        (subset_root / "label_mapping.json").write_text(json.dumps(label_to_id, indent=2, ensure_ascii=False), encoding="utf-8")
        for split, rows in splits.items():
            (subset_root / f"{split}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "protocol": "reduced12 diversity-aware BABEL protocol",
        "categories": list(REDUCED12_LABELS),
        "label_mapping": label_to_id,
        "seed": seed,
        "caps": {"official_train_per_class": train_cap, "official_val_per_class": active_val_cap},
        "split_definition": {
            "stgcn_development": "official BABEL Train selected per class then 90/10 train/val",
            "activeview": "official BABEL Val selected per class then 60/20/20 train/val/test",
        },
        "raw_counts": {
            "official_train": dict(Counter(str(item["action_label"]) for item in train_raw)),
            "official_val": dict(Counter(str(item["action_label"]) for item in val_raw)),
        },
        "selected_counts": {
            subset: {split: dict(Counter(str(item["action_label"]) for item in rows)) for split, rows in splits.items()}
            for subset, splits in manifests.items()
        },
        "diversity": {
            subset: {split: _diversity(rows) for split, rows in splits.items()}
            for subset, splits in manifests.items()
        },
        "selection_priority": ["unique source", "subject diversity", "AMASS dataset diversity", "duration-bin diversity"],
        "excluded_records": len(train_excluded) + len(val_excluded),
        "source_files": {
            "official_train": str((babel_dir / "train.json").resolve()),
            "official_val": str((babel_dir / "val.json").resolve()),
            "amass_index": str(amass_index_path.resolve()),
        },
    }
    (output_root / "protocol_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


__all__ = ["REDUCED12_LABELS", "build_reduced12_protocol", "collect_reduced12_records", "select_diverse_records"]
