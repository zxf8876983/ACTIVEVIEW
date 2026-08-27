"""Shared BABEL-to-AMASS source resolution utilities for the active dataset.

This module contains no action taxonomy.  The selected16 manifest owns the
current label definition; these helpers only resolve source files and interval
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class SourceInfo:
    """Resolved AMASS source metadata."""

    path: Path
    num_frames: int
    fps: float


def _normalise_key(value: str) -> str:
    return value.replace("\\", "/").strip().lstrip("/").lower().replace("-", "_")


def _source_lookup(index: Mapping[str, str]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for key, value in index.items():
        norm = _normalise_key(str(key))
        path = Path(value)
        variants = {
            norm,
            "/".join(norm.split("/")[1:]) if "/" in norm else norm,
            norm.replace("_poses.npz", ".npz"),
            Path(norm).name,
            Path(norm).name.replace("_poses.npz", ".npz"),
        }
        for variant in variants:
            lookup.setdefault(variant, path)
    return lookup


def resolve_source_path(feat_p: str, lookup: Mapping[str, Path]) -> Optional[Path]:
    """Resolve BABEL's path spelling against the local AMASS index."""
    norm = _normalise_key(feat_p)
    parts = norm.split("/")
    variants = [
        norm,
        "/".join(parts[1:]) if len(parts) > 1 else norm,
        norm.replace("_poses.npz", ".npz"),
        "/".join(parts[1:]).replace("_poses.npz", ".npz") if len(parts) > 1 else norm,
        Path(norm).name,
        Path(norm).name.replace("_poses.npz", ".npz"),
    ]
    for variant in variants:
        candidate = lookup.get(variant)
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def _read_source_info(path: Path) -> SourceInfo:
    with np.load(path, allow_pickle=True) as data:
        num_frames = int(np.asarray(data["poses"]).shape[0])
        fps = 30.0
        for key in ("mocap_framerate", "mocap_frame_rate", "frame_rate", "framerate", "fps"):
            if key in data:
                value = data[key]
                fps = float(value) if np.ndim(value) == 0 else float(value[0])
                break
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS {fps} in {path}")
    return SourceInfo(path=path, num_frames=num_frames, fps=fps)
