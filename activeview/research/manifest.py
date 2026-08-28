"""Experiment manifest serialization and immutable-directory helpers."""

from __future__ import annotations

import json
import os
import re
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


def _portable_path(path: Path, root: Path) -> str:
    """Encode a path relative to its owning root for portable manifests."""
    return path.expanduser().resolve().relative_to(root.expanduser().resolve()).as_posix()


def source_path_value(path: Path, repo_root: Path) -> str:
    """Return the canonical repository-relative source path."""
    return _portable_path(path, repo_root)


def runtime_path_value(path: Path, data_root: Path) -> str:
    """Return the canonical data-root-relative runtime path."""
    return _portable_path(path, data_root)


def resolve_source_path(value: str, repo_root: Path) -> Path:
    """Resolve a source path, accepting legacy absolute values read-only."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = repo_root.expanduser().resolve()
    resolved = (root / path).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"source path escapes repository root: {value}")
    return resolved


def resolve_runtime_path(value: str, data_root: Path) -> Path:
    """Resolve a runtime path, accepting legacy absolute values read-only."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = data_root.expanduser().resolve()
    resolved = (root / path).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"runtime path escapes data root: {value}")
    return resolved


def parse_controlled_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse only the small nested subset needed by lifecycle guards."""
    if not path.is_file():
        raise FileNotFoundError(path)
    sections: Dict[str, Dict[str, Any]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        section = re.match(r"^([A-Za-z_][\w-]*):\s*$", line)
        if section:
            current = section.group(1)
            sections.setdefault(current, {})
            continue
        item = re.match(r"^\s{2}([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
        if not item or current is None:
            continue
        raw = item.group(2).strip()
        if raw.lower() in {"true", "false"}:
            value: Any = raw.lower() == "true"
        elif raw in {"", "null", "~"}:
            value = None
        else:
            value = raw.strip("'\"")
        sections[current][item.group(1)] = value
    return sections


def validate_controlled_config(path: Path, expected_experiment_id: str) -> list[str]:
    """Validate Test-lock fields without importing a YAML dependency."""
    try:
        config = parse_controlled_config(path)
    except (OSError, ValueError) as error:
        return [f"config_invalid:{error}"]
    errors: list[str] = []
    if config.get("experiment", {}).get("id") != expected_experiment_id:
        errors.append("config_experiment_id_mismatch")
    if config.get("evaluation", {}).get("test") is not False:
        errors.append("config_evaluation_test_must_be_false")
    if config.get("protocol", {}).get("test_locked") is not True:
        errors.append("config_test_locked_must_be_true")
    if config.get("protocol", {}).get("test_authorized") is not False:
        errors.append("config_test_authorized_must_be_false")
    return errors


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
