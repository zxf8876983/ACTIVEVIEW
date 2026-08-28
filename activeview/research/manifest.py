"""Experiment manifest serialization and immutable-directory helpers."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from activeview.active_view.utility_label_builder import file_sha256

from .experiment import Experiment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str, default: str = "") -> str:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return default
    return result.stdout.strip()


def git_dirty() -> bool:
    return bool(git_value("status", "--porcelain"))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(source_dir: Path) -> Dict[str, Any]:
    path = source_dir / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    return payload


def save_manifest(source_dir: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(source_dir / "run_manifest.json", payload)


def manifest_sha256(source_dir: Path) -> str:
    return file_sha256(source_dir / "run_manifest.json")


def experiment_from_manifest(payload: Mapping[str, Any]) -> Experiment:
    experiment_payload = payload.get("experiment")
    if not isinstance(experiment_payload, Mapping):
        raise ValueError("Manifest is missing experiment object")
    return Experiment.from_dict(experiment_payload)


def status_payload(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    experiment = manifest.get("experiment", {})
    return {
        "experiment_id": experiment.get("experiment_id"),
        "status": experiment.get("status"),
        "decision": experiment.get("decision", "NA"),
        "updated_at": utc_now(),
    }


def write_status(source_dir: Path, manifest: Mapping[str, Any]) -> None:
    write_json_atomic(source_dir / "status.json", status_payload(manifest))
