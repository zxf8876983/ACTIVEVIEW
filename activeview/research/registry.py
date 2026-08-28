"""CSV registry with monotonic IDs and atomic updates."""

from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .experiment import Experiment


REGISTRY_FIELDS = [
    "experiment_id", "name", "stage", "status", "hypothesis", "core_change",
    "baseline", "created_at", "completed_at", "git_commit_start", "git_commit_end",
    "val_accuracy", "val_macro_f1", "val_mean_regret", "val_median_regret",
    "val_p90_regret", "val_headroom_capture", "decision", "test_used", "source_dir",
    "runtime_dir", "notes",
]
_ID_PATTERN = re.compile(r"^EXP(\d+)$")


def _ensure_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS).writeheader()


def list_experiments(path: Path) -> List[Dict[str, str]]:
    _ensure_registry(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REGISTRY_FIELDS:
            raise ValueError(f"Invalid registry header in {path}")
        return [dict(row) for row in reader]


def next_experiment_id(path: Path) -> str:
    numbers = [int(match.group(1)) for row in list_experiments(path) if (match := _ID_PATTERN.match(str(row.get("experiment_id", ""))))]
    return f"EXP{max(numbers, default=0) + 1:03d}"


def _atomic_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def register_experiment(path: Path, record: Mapping[str, Any]) -> None:
    rows = list_experiments(path)
    experiment_id = str(record.get("experiment_id", ""))
    if not experiment_id:
        raise ValueError("experiment_id is required")
    if any(row.get("experiment_id") == experiment_id for row in rows):
        raise ValueError(f"Duplicate experiment ID: {experiment_id}")
    rows.append({field: record.get(field, "") for field in REGISTRY_FIELDS})
    _atomic_write(path, rows)


def get_experiment(path: Path, experiment_id: str) -> Dict[str, str]:
    matches = [row for row in list_experiments(path) if row.get("experiment_id") == experiment_id]
    if not matches:
        raise KeyError(f"Unknown experiment: {experiment_id}")
    return matches[0]


def update_experiment(path: Path, experiment_id: str, updates: Mapping[str, Any]) -> Dict[str, str]:
    rows = list_experiments(path)
    found = False
    updated: Dict[str, str] = {}
    for row in rows:
        if row.get("experiment_id") == experiment_id:
            found = True
            for field, value in updates.items():
                if field not in REGISTRY_FIELDS or field == "experiment_id":
                    raise ValueError(f"Invalid registry update field: {field}")
                row[field] = "" if value is None else str(value)
            updated = dict(row)
    if not found:
        raise KeyError(f"Unknown experiment: {experiment_id}")
    _atomic_write(path, rows)
    return updated
