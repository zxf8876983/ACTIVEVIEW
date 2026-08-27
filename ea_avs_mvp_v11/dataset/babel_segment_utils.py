"""Generic single-label BABEL segment filtering for the selected16 protocol."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ea_avs_mvp_v11.dataset.babel_source_utils import (
    _normalise_key,
    _read_source_info,
    resolve_source_path,
)

LOCOMOTION_RE = re.compile(r"\b(?:walk|walking|run|running|jog|jogging|crawl|crawling)\b", re.I)


def _annotation(entry: Mapping[str, Any]) -> Tuple[Mapping[str, Any] | None, str]:
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


def _iter_auxiliary_records(
    babel_path: Path,
    source_split: str,
    source_lookup: Mapping[str, Path],
    auxiliary_labels: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect exactly one of the canonical auxiliary labels per segment."""
    data = json.loads(babel_path.read_text(encoding="utf-8"))
    allowed = set(str(label) for label in auxiliary_labels)
    records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    source_cache: Dict[str, Any] = {}
    for sid, entry in data.items():
        annotation, annotation_level = _annotation(entry)
        if annotation is None:
            continue
        feat_p = str(entry.get("feat_p", ""))
        source_path = resolve_source_path(feat_p, source_lookup)
        if source_path is None:
            excluded.append({"babel_sid": sid, "source_split": source_split, "reason": "source_missing"})
            continue
        source_key = _normalise_key(feat_p)
        try:
            source = source_cache.setdefault(source_key, _read_source_info(source_path))
        except Exception as exc:  # noqa: BLE001 - retain exclusion audit
            excluded.append({"babel_sid": sid, "source_split": source_split, "reason": "source_unreadable", "error": str(exc)})
            continue
        sequence_duration = float(entry.get("dur", 0.0))
        for label_index, label_info in enumerate(annotation.get("labels", [])):
            act_cat = [str(item) for item in (label_info.get("act_cat") or [])]
            hits = sorted(set(act_cat) & allowed)
            if len(hits) != 1:
                continue
            action_label = hits[0]
            raw_label = str(label_info.get("raw_label", ""))
            proc_label = str(label_info.get("proc_label", ""))
            if action_label not in {"lie", "stumble"} and LOCOMOTION_RE.search(f"{raw_label} {proc_label}"):
                continue
            start_frame, end_frame = _frame_interval(
                label_info, annotation_level, sequence_duration, source.num_frames, source.fps
            )
            records.append(
                {
                    "record_id": f"babel_{source_split}_{int(sid):05d}_{label_index:03d}",
                    "babel_sid": int(sid),
                    "source_split": source_split,
                    "split": source_split,
                    "annotation_level": annotation_level,
                    "action_label": action_label,
                    "act_cat": act_cat,
                    "target_source_labels": hits,
                    "raw_label": raw_label,
                    "proc_label": proc_label,
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
                    "num_frames": end_frame - start_frame + 1,
                }
            )
    return records, excluded


def _deduplicate(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for record in records:
        key = (
            record["source_split"], record["babel_sid"], record["action_label"],
            record["annotation_level"], record["start_frame"], record["end_frame"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _remove_ambiguous_intervals(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["source_group"]), int(record["start_frame"]), int(record["end_frame"]))
        grouped[key].append(record)
    ambiguous_keys = {
        key for key, items in grouped.items()
        if len({str(item["action_label"]) for item in items}) > 1
    }
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for record in records:
        key = (str(record["source_group"]), int(record["start_frame"]), int(record["end_frame"]))
        if key not in ambiguous_keys:
            kept.append(record)
            continue
        excluded.append({
            "record_id": record["record_id"],
            "source_split": record["source_split"],
            "reason": "ambiguous_target_labels",
            "source_group": record["source_group"],
            "start_frame": record["start_frame"],
            "end_frame": record["end_frame"],
            "target_labels": sorted({str(item["action_label"]) for item in grouped[key]}),
        })
    return kept, excluded


def _filter_short_intervals(
    records: Sequence[Dict[str, Any]],
    min_source_frames: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for record in records:
        if int(record["num_frames"]) > min_source_frames:
            kept.append(record)
            continue
        excluded.append({
            "record_id": record["record_id"],
            "source_split": record["source_split"],
            "reason": "source_interval_too_short",
            "num_frames": record["num_frames"],
            "minimum_exclusive": min_source_frames,
        })
    return kept, excluded
