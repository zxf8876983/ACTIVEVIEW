"""BABEL official-category manifest utilities used by selected16.

The selected16 manifest reuses the official category table and its strict
single-label interval filtering.  It does not generate a separate 150-class
dataset; all 150-class training artifacts were removed.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from activeview.dataset.babel_source_utils import (
    _normalise_key,
    _read_source_info,
    _source_lookup,
    resolve_source_path,
)
from activeview.perception.skeleton_definition import get_skeleton_definition
from activeview.perception.skeleton_normalizer import SkeletonNormalizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activeview.dataset.amass_true_skeleton import AMASSTrueSkeletonConverter

LOGGER = logging.getLogger(__name__)


def load_official_categories(path: Path) -> List[str]:
    """Load the ordered official 150-category mapping from its count table."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150:
        raise ValueError(f"Expected 150 official categories, found {len(rows)} in {path}")
    rows.sort(key=lambda row: int(row["official_index"]))
    indices = [int(row["official_index"]) for row in rows]
    if indices != list(range(150)):
        raise ValueError(f"Official indices are not contiguous 0..149: {indices[:5]}...{indices[-5:]}")
    labels = [str(row["action_category"]) for row in rows]
    if len(set(labels)) != 150:
        raise ValueError("Official category names are not unique")
    return labels


def _annotation(entry: Mapping[str, Any]) -> Tuple[Mapping[str, Any] | None, str]:
    """Prefer frame-level Dense labels and fall back to sequence labels."""
    frame_ann = entry.get("frame_ann")
    if frame_ann:
        return frame_ann, "frame_level"
    seq_ann = entry.get("seq_ann")
    if seq_ann:
        return seq_ann, "sequence_level"
    return None, ""


def _frame_interval(
    label: Mapping[str, Any],
    annotation_level: str,
    sequence_duration: float,
    source_frames: int,
    fps: float,
) -> Tuple[int, int]:
    if annotation_level == "frame_level":
        start = int(round(float(label.get("start_t", 0.0)) * fps))
        end = int(round(float(label.get("end_t", sequence_duration)) * fps)) - 1
    else:
        start, end = 0, source_frames - 1
    start = max(0, min(start, source_frames - 1))
    end = max(start, min(end, source_frames - 1))
    return start, end


def collect_official_records(
    babel_path: Path,
    source_split: str,
    source_lookup: Mapping[str, Path],
    official_labels: Sequence[str],
    *,
    min_frames_exclusive: int = 30,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect unambiguous, readable official-category segments."""
    data = json.loads(babel_path.read_text(encoding="utf-8"))
    official_set = set(official_labels)
    records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    source_cache: Dict[str, Any] = {}
    for sid, entry in data.items():
        annotation, annotation_level = _annotation(entry)
        if annotation is None:
            excluded.append({"babel_sid": int(sid), "source_split": source_split, "reason": "annotation_missing"})
            continue
        feat_p = str(entry.get("feat_p", ""))
        source_path = resolve_source_path(feat_p, source_lookup)
        if source_path is None:
            excluded.append({"babel_sid": int(sid), "source_split": source_split, "reason": "source_missing"})
            continue
        source_key = _normalise_key(feat_p)
        try:
            source = source_cache.setdefault(source_key, _read_source_info(source_path))
        except Exception as exc:  # noqa: BLE001 - retain an exclusion audit
            excluded.append(
                {
                    "babel_sid": int(sid),
                    "source_split": source_split,
                    "reason": "source_unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        sequence_duration = float(entry.get("dur", 0.0))
        for label_index, label_info in enumerate(annotation.get("labels", [])):
            act_cat = [str(item) for item in (label_info.get("act_cat") or [])]
            official_hits = sorted(set(act_cat) & official_set)
            base = {
                "babel_sid": int(sid),
                "label_index": label_index,
                "source_split": source_split,
                "act_cat": act_cat,
                "raw_label": str(label_info.get("raw_label", "")),
                "proc_label": str(label_info.get("proc_label", "")),
            }
            if not official_hits:
                excluded.append({**base, "reason": "not_in_official150"})
                continue
            if len(official_hits) != 1:
                excluded.append({**base, "reason": "ambiguous_official150_labels", "official_hits": official_hits})
                continue
            start_frame, end_frame = _frame_interval(
                label_info,
                annotation_level,
                sequence_duration,
                source.num_frames,
                source.fps,
            )
            num_frames = end_frame - start_frame + 1
            if num_frames <= min_frames_exclusive:
                excluded.append({**base, "reason": "too_short", "num_frames": num_frames})
                continue
            records.append(
                {
                    "record_id": f"babel_{source_split}_{int(sid):05d}_{label_index:03d}",
                    "babel_sid": int(sid),
                    "label_index": label_index,
                    "source_split": source_split,
                    "split": source_split,
                    "annotation_level": annotation_level,
                    "action_label": official_hits[0],
                    "official_index": official_labels.index(official_hits[0]),
                    "act_cat": act_cat,
                    "raw_label": str(label_info.get("raw_label", "")),
                    "proc_label": str(label_info.get("proc_label", "")),
                    "feat_p": feat_p,
                    "source_group": source_key,
                    "source_path": str(source.path),
                    "source_num_frames": int(source.num_frames),
                    "fps": float(source.fps),
                    "sequence_duration": sequence_duration,
                    "start_t": float(label_info.get("start_t", 0.0)) if annotation_level == "frame_level" else 0.0,
                    "end_t": float(label_info.get("end_t", sequence_duration)) if annotation_level == "frame_level" else sequence_duration,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "num_frames": num_frames,
                }
            )
    return _deduplicate(records), excluded


def _deduplicate(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        key = (
            record["source_split"],
            record["babel_sid"],
            record["start_frame"],
            record["end_frame"],
            record["action_label"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def cap_records(
    records: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    cap: int,
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample at most ``cap`` records per official category deterministically."""
    grouped: Dict[str, List[Dict[str, Any]]] = {label: [] for label in categories}
    for record in records:
        grouped[str(record["action_label"])].append(dict(record))
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for label in categories:
        items = grouped[label]
        rng.shuffle(items)
        selected.extend(items[: int(cap)])
    selected.sort(key=lambda item: str(item["record_id"]))
    return selected


def write_manifests(
    output_dir: Path,
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    categories: Sequence[str],
    *,
    train_cap: int,
    val_cap: int,
    seed: int,
    min_frames_exclusive: int,
    excluded: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    mapping = {label: index for index, label in enumerate(categories)}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in (("train", train_records), ("val", val_records)):
        materialized = []
        for record in records:
            item = dict(record)
            item["split"] = split
            item["label_id"] = mapping[str(item["action_label"])]
            item["benchmark_skeleton_source"] = "AMASS SMPL pose parameters + deterministic FK; no Habitat/RGB/VideoPose3D"
            materialized.append(item)
        (output_dir / f"{split}.json").write_text(json.dumps(materialized, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "excluded.json").write_text(json.dumps(list(excluded), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "protocol": "official BABEL 150 categories; unambiguous single-official-label segments only",
        "official_category_count": len(categories),
        "categories": list(categories),
        "train_cap_per_class": train_cap,
        "val_cap_per_class": val_cap,
        "seed": seed,
        "min_source_frames_exclusive": min_frames_exclusive,
        "selected_counts": {
            split: dict(Counter(str(item["action_label"]) for item in records))
            for split, records in (("train", train_records), ("val", val_records))
        },
        "selected_samples": {"train": len(train_records), "val": len(val_records)},
        "split_definition": {
            "train": "BABEL official train.json",
        "val": "BABEL official val.json; used only for class screening and post-hoc diagnosis",
            "test": "BABEL test.json is empty; no independent official test score is claimed",
        },
        "source_group_overlap_train_val": len(
            {str(item["source_group"]) for item in train_records}
            & {str(item["source_group"]) for item in val_records}
        ),
        "skeleton_preprocessing": "AMASS FK H36M-17 + root_center + torso_scale + yaw_only_y_up",
        "excluded_records": len(excluded),
        "exclusion_counts": dict(Counter(str(item.get("reason", "unknown")) for item in excluded)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def generate_tensor_split(
    output_dir: Path,
    split: str,
    records: Sequence[Mapping[str, Any]],
    converter: AMASSTrueSkeletonConverter,
    normalizer: SkeletonNormalizer,
    categories: Sequence[str],
    *,
    target_frames: int,
) -> Dict[str, Any]:
    tensors: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[Dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        raw = converter.convert_record(record, target_frames=target_frames)
        normalized = normalizer.normalize_sequence(raw, align_canonical=True)
        tensor = np.transpose(normalized, (2, 0, 1))[:, :, :, np.newaxis].astype(np.float32)
        if tensor.shape != (3, target_frames, 17, 1) or not np.isfinite(tensor).all():
            raise ValueError(f"Invalid {split} tensor for {record['record_id']}: {tensor.shape}")
        tensors.append(tensor)
        labels.append(int(record["label_id"]))
        metadata.append(
            {
                "record_id": str(record["record_id"]),
                "action_label": str(record["action_label"]),
                "official_index": int(record["official_index"]),
                "label_id": int(record["label_id"]),
                "source_group": str(record["source_group"]),
                "source_path": str(record["source_path"]),
                "start_frame": int(record["start_frame"]),
                "end_frame": int(record["end_frame"]),
                "num_frames": int(record["num_frames"]),
                "skeleton_source": "AMASS SMPL pose parameters + deterministic FK",
            }
        )
        if index % 500 == 0:
            LOGGER.info("%s: converted %d/%d", split, index, len(records))
    data = np.stack(tensors, axis=0) if tensors else np.empty((0, 3, target_frames, 17, 1), dtype=np.float32)
    np.save(output_dir / f"{split}_data.npy", data)
    np.save(output_dir / f"{split}_labels.npy", np.asarray(labels, dtype=np.int64))
    (output_dir / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "split": split,
        "samples": len(metadata),
        "data_shape": list(data.shape),
        "class_counts": dict(Counter(int(label) for label in labels)),
        "categories": list(categories),
        "skeleton_source": "AMASS SMPL pose parameters + deterministic FK; no Habitat/RGB/VideoPose3D",
        "skeleton_preprocessing": "root_center + torso_scale + yaw_only_y_up",
    }
    (output_dir / f"{split}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_official150_benchmark(
    *,
    output_dir: Path,
    babel_dir: Path,
    index_path: Path,
    official_mapping_path: Path,
    train_cap: int = 400,
    val_cap: int = 100,
    target_frames: int = 150,
    min_frames_exclusive: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    categories = load_official_categories(official_mapping_path)
    source_lookup = _source_lookup(json.loads(index_path.read_text(encoding="utf-8")))
    train_raw, train_excluded = collect_official_records(
        babel_dir / "train.json", "train", source_lookup, categories, min_frames_exclusive=min_frames_exclusive
    )
    val_raw, val_excluded = collect_official_records(
        babel_dir / "val.json", "val", source_lookup, categories, min_frames_exclusive=min_frames_exclusive
    )
    excluded = train_excluded + val_excluded
    train = cap_records(train_raw, categories, train_cap, seed=seed)
    val = cap_records(val_raw, categories, val_cap, seed=seed + 1)
    for record in train + val:
        record["label_id"] = categories.index(str(record["action_label"]))
    summary = write_manifests(
        output_dir,
        train,
        val,
        categories,
        train_cap=train_cap,
        val_cap=val_cap,
        seed=seed,
        min_frames_exclusive=min_frames_exclusive,
        excluded=excluded,
    )
    converter = AMASSTrueSkeletonConverter()
    normalizer = SkeletonNormalizer(skel_def=get_skeleton_definition(backend="h36m_17"))
    for split, records in (("train", train), ("val", val)):
        generate_tensor_split(
            output_dir,
            split,
            records,
            converter,
            normalizer,
            categories,
            target_frames=target_frames,
        )
    return summary
