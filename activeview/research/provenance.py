"""Frozen-foundation provenance collection for Stage C research experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root


def _artifact(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": file_sha256(resolved) if resolved.is_file() else None,
    }


def _json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def collect_stage_c_research_provenance(data_root: Path | None = None) -> Dict[str, Any]:
    """Collect hashes using the canonical Stage A/B/C artifact layout.

    Missing files are represented explicitly rather than replaced by fake
    hashes.  This keeps creation useful on a fresh checkout while allowing the
    experiment validator to require a complete frozen foundation before run.
    """
    root = (data_root or get_data_root()).expanduser().resolve()
    dataset_root = root / "datasets" / "policy_v11_5"
    stage_a_summary = dataset_root / "stage_a_summary.json"
    stage_b_root = dataset_root / "stage_b"
    stage_b_summary = stage_b_root / "stage_b_summary.json"
    stage_c_root = dataset_root / "stage_c"
    stage_c_summary = stage_c_root / "stage_c_feature_summary.json"
    stage_c_payload = _json(stage_c_summary)
    stage_a_payload = _json(stage_a_summary)
    label_mapping = root / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed" / "label_mapping.json"
    checkpoint = root / "checkpoints" / "stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled" / "stgcn_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed_oversampled_best.pth"
    episode_files = {
        split: Path(str(stage_a_payload.get("episode_files", {}).get(split, dataset_root / f"{split}_episodes.jsonl")))
        for split in ("train", "val", "test")
    }
    feature_files = {
        split: Path(str(stage_c_payload.get("feature_files", {}).get(split, stage_c_root / "features" / f"{split}.jsonl")))
        for split in ("train", "val", "test")
    }
    utility_files = {split: stage_b_root / "utility_labels" / f"{split}.jsonl" for split in ("train", "val", "test")}
    return {
        "stage_a": {"summary": _artifact(stage_a_summary), "episodes": {split: _artifact(path) for split, path in episode_files.items()}},
        "stage_b": {"summary": _artifact(stage_b_summary), "utility": {split: _artifact(path) for split, path in utility_files.items()}},
        "stage_c_features": {
            "summary": _artifact(stage_c_summary),
            "features": {split: _artifact(path) for split, path in feature_files.items()},
            "stats": _artifact(Path(str(stage_c_payload.get("feature_stats", stage_c_root / "stage_c_feature_stats.json")))),
        },
        "stgcn_checkpoint": _artifact(checkpoint),
        "label_mapping": _artifact(label_mapping),
        "record_split": {
            "summary": _artifact(dataset_root / "splits" / "summary.json"),
            "canonical_counts": {"train": 589, "val": 197, "test": 194},
        },
    }


def provenance_complete(provenance: Mapping[str, Any]) -> bool:
    """Return whether all frozen foundation artifacts have real SHA-256 hashes."""
    required = ("stage_a", "stage_b", "stage_c_features", "stgcn_checkpoint", "label_mapping", "record_split")
    if any(not isinstance(provenance.get(key), Mapping) for key in required):
        return False
    def hashes(value: Any) -> bool:
        if isinstance(value, Mapping):
            if "path" in value:
                return bool(value.get("exists") and value.get("sha256"))
            return all(hashes(item) for item in value.values())
        return True
    return all(hashes(provenance[key]) for key in required)


def verify_frozen_provenance(recorded_provenance: Mapping[str, Any]) -> list[str]:
    """Re-hash every recorded frozen artifact and return explicit errors."""
    errors: list[str] = []
    required = ("stage_a", "stage_b", "stage_c_features", "stgcn_checkpoint", "label_mapping", "record_split")

    def visit(value: Any, logical_path: str) -> None:
        if isinstance(value, Mapping):
            if "path" in value:
                path = Path(str(value.get("path", "")))
                if not path.is_file():
                    errors.append(f"frozen_artifact_missing:{logical_path}")
                    return
                expected = value.get("sha256")
                actual = file_sha256(path)
                if not expected or actual != expected:
                    errors.append(f"frozen_artifact_hash_mismatch:{logical_path}")
                return
            for key, child in value.items():
                visit(child, f"{logical_path}.{key}")

    if not isinstance(recorded_provenance, Mapping):
        return ["frozen_provenance_missing"]
    for key in required:
        if key not in recorded_provenance:
            errors.append(f"frozen_provenance_missing:{key}")
        else:
            visit(recorded_provenance[key], key)
    return errors
