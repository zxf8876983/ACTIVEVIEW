"""文件用途：
    处理 AMASS/BABEL 动作资产与 Habitat 转换。

主要输入：
    - 动作标注、NPZ 资产和 URDF。
主要输出：
    - 规范化动作、映射或动作清单。
项目角色：
    - 属于 data.motion 数据模块。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from activeview.core.paths import get_data_root
from activeview.data.motion.babel_official150_true_skeleton import (
    cap_records,
    collect_official_records,
    load_official_categories,
)
from activeview.data.motion.babel_segment_utils import (
    _deduplicate,
    _filter_short_intervals,
    _iter_auxiliary_records,
    _remove_ambiguous_intervals,
)
from activeview.data.motion.babel_source_utils import _source_lookup

LOGGER = logging.getLogger(__name__)

SELECTED_OFFICIAL_LABELS: Tuple[str, ...] = (
    "t pose",
    "cartwheel",
    "knock",
    "play instrument",
    "crawl",
    "a pose",
    "kick",
    "sit",
    "move up/down incline",
    "jog",
    "stand up",
    "jump",
    "walk",
    "throw",
)
DEFAULT_AUXILIARY_LABELS: Tuple[str, ...] = ("lie", "stumble")
SELECTED_LABELS: Tuple[str, ...] = SELECTED_OFFICIAL_LABELS + DEFAULT_AUXILIARY_LABELS


def _normalise_aux_records(
    babel_dir: Path,
    source_lookup: Mapping[str, Path],
    *,
    auxiliary_labels: Sequence[str],
    min_frames_exclusive: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for source_split in ("train", "val"):
        current, current_excluded = _iter_auxiliary_records(
            babel_dir / f"{source_split}.json", source_split, source_lookup, auxiliary_labels
        )
        current = _deduplicate(current)
        current, short = _filter_short_intervals(current, min_frames_exclusive)
        current, ambiguous = _remove_ambiguous_intervals(current)
        for item in current:
            item["split"] = source_split
        records.extend(current)
        excluded.extend(current_excluded + short + ambiguous)
    return records, excluded


def build_selected16_manifests(
    *,
    output_dir: Optional[Path] = None,
    babel_dir: Optional[Path] = None,
    amass_index_path: Optional[Path] = None,
    official_mapping_path: Optional[Path] = None,
    train_cap: int = 400,
    val_cap: int = 100,
    min_frames_exclusive: int = 30,
    seed: int = 42,
    auxiliary_labels: Sequence[str] = DEFAULT_AUXILIARY_LABELS,
) -> Dict[str, Any]:
    """Collect, cap, label, and write selected 16-class Train/Val records."""
    auxiliary_labels = tuple(str(label) for label in auxiliary_labels)
    if len(auxiliary_labels) != 2 or len(set(auxiliary_labels)) != 2:
        raise ValueError("Exactly two distinct auxiliary labels are required")
    if set(auxiliary_labels) != set(DEFAULT_AUXILIARY_LABELS):
        raise ValueError("The active selected16 protocol requires exactly lie and stumble; fall is obsolete")
    data_root = get_data_root()
    output_dir = output_dir or data_root / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"
    babel_dir = babel_dir or data_root / "datasets" / "babel" / "babel_v1.0_release"
    amass_index_path = amass_index_path or data_root / "cache" / "amass_download" / "amass_file_index.json"
    official_mapping_path = official_mapping_path or data_root / "cache" / "babel_selection" / "babel_train_val_official150_act_cat_counts.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    official_categories = load_official_categories(official_mapping_path)
    missing = sorted(set(SELECTED_OFFICIAL_LABELS) - set(official_categories))
    if missing:
        raise ValueError(f"Selected labels missing from official mapping: {missing}")
    source_lookup = _source_lookup(json.loads(amass_index_path.read_text(encoding="utf-8")))

    official_by_split: Dict[str, List[Dict[str, Any]]] = {}
    excluded: List[Dict[str, Any]] = []
    for source_split, cap, split_seed in (("train", train_cap, seed), ("val", val_cap, seed + 1)):
        raw, current_excluded = collect_official_records(
            babel_dir / f"{source_split}.json",
            source_split,
            source_lookup,
            official_categories,
            min_frames_exclusive=min_frames_exclusive,
        )
        selected = [item for item in raw if item["action_label"] in SELECTED_OFFICIAL_LABELS]
        official_by_split[source_split] = cap_records(
            selected, SELECTED_OFFICIAL_LABELS, cap, seed=split_seed
        )
        excluded.extend(current_excluded)

    aux_records, aux_excluded = _normalise_aux_records(
        babel_dir,
        source_lookup,
        auxiliary_labels=auxiliary_labels,
        min_frames_exclusive=min_frames_exclusive,
    )
    excluded.extend(aux_excluded)

    # Enforce one target per identical source interval across both official and
    # auxiliary branches before class caps are applied.
    combined, ambiguous = _remove_ambiguous_intervals(
        official_by_split["train"] + official_by_split["val"] + aux_records
    )
    excluded.extend(ambiguous)
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    for item in combined:
        split = str(item["source_split"])
        if split in by_split:
            by_split[split].append(dict(item))

    # Re-apply caps after ambiguity removal so the final manifest obeys the
    # documented per-class limits exactly.
    train = cap_records(
        [item for item in by_split["train"] if item["action_label"] in SELECTED_OFFICIAL_LABELS],
        SELECTED_OFFICIAL_LABELS,
        train_cap,
        seed=seed,
    )
    val = cap_records(
        [item for item in by_split["val"] if item["action_label"] in SELECTED_OFFICIAL_LABELS],
        SELECTED_OFFICIAL_LABELS,
        val_cap,
        seed=seed + 1,
    )
    train.extend(item for item in by_split["train"] if item["action_label"] in auxiliary_labels)
    val.extend(item for item in by_split["val"] if item["action_label"] in auxiliary_labels)
    train.sort(key=lambda item: str(item["record_id"]))
    val.sort(key=lambda item: str(item["record_id"]))
    selected_labels = SELECTED_OFFICIAL_LABELS + auxiliary_labels
    label_mapping = {label: index for index, label in enumerate(selected_labels)}
    for item in train:
        item["split"] = "train"
        item["label_id"] = label_mapping[str(item["action_label"])]
        item["selected16_protocol"] = f"official14_f1_screened_plus_{'_'.join(auxiliary_labels)}"
    for item in val:
        item["split"] = "val"
        item["label_id"] = label_mapping[str(item["action_label"])]
        item["selected16_protocol"] = f"official14_f1_screened_plus_{'_'.join(auxiliary_labels)}"

    (output_dir / "train.json").write_text(json.dumps(train, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "val.json").write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "protocol": "selected 16 classes for pure-color Habitat RGB perception pretraining",
        "selected_labels": list(selected_labels),
        "official_labels": list(SELECTED_OFFICIAL_LABELS),
        "auxiliary_labels": list(auxiliary_labels),
        "train_cap_per_official_class": int(train_cap),
        "val_cap_per_official_class": int(val_cap),
        "min_source_frames_exclusive": int(min_frames_exclusive),
        "seed": int(seed),
        "selected_counts": {
            "train": dict(Counter(str(item["action_label"]) for item in train)),
            "val": dict(Counter(str(item["action_label"]) for item in val)),
        },
        "selected_samples": {"train": len(train), "val": len(val)},
        "split_definition": {"train": "BABEL train.json", "val": "BABEL val.json"},
        "single_target_label": True,
        "ambiguous_records_removed": len(ambiguous),
        "excluded_records": len(excluded),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "excluded.json").write_text(json.dumps(excluded, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Selected16 manifests: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_selected16_manifests()
