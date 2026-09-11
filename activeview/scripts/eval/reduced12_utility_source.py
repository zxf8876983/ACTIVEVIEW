"""Shared metadata and sample helpers for reduced12 utility diagnostics.

The module is deliberately read-only: it joins the frozen Stage-D Val rows to
the archived eight-placement candidate metadata and true ST-GCN evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.data.motion.babel_clean_dataset_generator import compose_root_rotation
from activeview.methods.active_view.geometry import viewpoint_azimuth, viewpoint_radius, wrap_relative_azimuth
from activeview.scripts.eval.reduced12_nbv_utils import candidate_logp, gt_margin, legal_ids


VIEW_COUNT = 32
RADIUS_COUNT = 4
AZIMUTH_COUNT = 8
BODY_AZIMUTH_BINS = tuple(range(AZIMUTH_COUNT))


def _wrap(angle: float) -> float:
    return float((float(angle) + 180.0) % 360.0 - 180.0)


def body_azimuth_bin(world_azimuth_deg: float, yaw_deg: float) -> int:
    """Map ``world azimuth - body yaw`` to its nearest 45-degree cell."""
    relative = float(world_azimuth_deg) - float(yaw_deg)
    return int(np.floor((relative % 360.0) / 45.0 + 0.5)) % AZIMUTH_COUNT


def _manifest_for(data_root: Path, scene_id: str) -> Path:
    root = data_root / "datasets/offline/habitat-train/00006-00087"
    path = root / scene_id / "candidate_metadata" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate metadata manifest: {path}")
    return path


def load_scene_metadata(
    data_root: Path, rows: Sequence[Mapping[str, Any]], *, audit_sample_count: int = 10,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Load and validate placement yaw and world candidate azimuth metadata."""
    keys = sorted({(str(row["scene_id"]), str(row["region"])) for row in rows})
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for scene_id, placement_id in keys:
        manifest_path = _manifest_for(data_root, scene_id)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("version") != "furniture-placement-v2":
            raise ValueError(f"unexpected manifest version: {manifest_path}")
        matches = [
            item for item in payload.get("placements_data", [])
            if str(item.get("placement_id") or item.get("region")) == placement_id
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one placement {scene_id}/{placement_id}")
        placement = matches[0]
        yaw = float(placement["yaw_deg"])
        if not np.isfinite(yaw) or int(round(yaw)) % 45 != 0:
            raise ValueError(f"invalid placement yaw for {scene_id}/{placement_id}")
        views = placement.get("viewpoints", [])
        if len(views) != VIEW_COUNT:
            raise ValueError(f"expected 32 viewpoints for {scene_id}/{placement_id}")
        azimuths: dict[int, float] = {}
        radii: dict[int, float] = {}
        position_errors: list[float] = []
        base = np.asarray(placement["position"], dtype=np.float64)
        for view in views:
            viewpoint_id = int(view["viewpoint_id"])
            azimuth = float(view["azimuth_deg"]) % 360.0
            radius = float(view["radius_m"])
            if viewpoint_id in azimuths:
                raise ValueError(f"duplicate viewpoint id {viewpoint_id}")
            if abs(_wrap(azimuth - viewpoint_azimuth(viewpoint_id))) > 1e-5:
                raise ValueError(f"viewpoint id/azimuth mismatch for {scene_id}/{placement_id}/{viewpoint_id}")
            if abs(radius - viewpoint_radius(viewpoint_id)) > 1e-5:
                raise ValueError(f"viewpoint id/radius mismatch for {scene_id}/{placement_id}/{viewpoint_id}")
            position = np.asarray(view["position"], dtype=np.float64)
            expected_azimuth = float(np.degrees(np.arctan2(position[0] - base[0], position[2] - base[2])) % 360.0)
            position_errors.append(abs(_wrap(expected_azimuth - azimuth)))
            azimuths[viewpoint_id] = azimuth
            radii[viewpoint_id] = radius
        if max(position_errors) > 1e-3:
            raise ValueError(f"candidate position/azimuth mismatch for {scene_id}/{placement_id}")
        metadata[(scene_id, placement_id)] = {
            "manifest": str(manifest_path.resolve()),
            "placement_id": placement_id,
            "yaw_deg": yaw,
            "position": base.tolist(),
            "azimuths": azimuths,
            "radii": radii,
            "position_azimuth_max_error_deg": max(position_errors),
        }

    # A numerical audit makes the body-relative convention explicit rather than
    # relying on an undocumented sign choice.  The generator's +Z body-forward
    # basis rotates toward +X for positive yaw.
    rng = np.random.default_rng(42)
    row_indices = rng.choice(len(rows), size=min(audit_sample_count, len(rows)), replace=False)
    samples: list[dict[str, Any]] = []
    sign_errors: list[float] = []
    for index in np.asarray(row_indices).tolist():
        row = rows[int(index)]
        key = (str(row["scene_id"]), str(row["region"]))
        item = metadata[key]
        selected_ids = sorted(item["azimuths"])[:3]
        for viewpoint_id in selected_ids:
            world_azimuth = float(item["azimuths"][viewpoint_id])
            yaw = float(item["yaw_deg"])
            local_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            rotated = compose_root_rotation(np.eye(4, dtype=np.float32), scene_yaw_deg=yaw)[:3, :3] @ local_forward
            forward_azimuth = float(np.degrees(np.arctan2(rotated[0], rotated[2])) % 360.0)
            sign_errors.append(abs(_wrap(forward_azimuth - yaw)))
            samples.append({
                "episode_id": str(row["episode_id"]),
                "scene_id": key[0],
                "placement_id": key[1],
                "viewpoint_id": int(viewpoint_id),
                "world_azimuth_deg": world_azimuth,
                "human_yaw_deg": yaw,
                "body_relative_azimuth_deg": _wrap(world_azimuth - yaw),
                "body_relative_bin": body_azimuth_bin(world_azimuth, yaw),
                "generator_forward_azimuth_deg": forward_azimuth,
            })
    audit = {
        "convention_confirmed": bool(sign_errors and max(sign_errors) <= 1e-4),
        "world_azimuth_reference": "+Z is 0 degrees; atan2(+X,+Z), increasing toward +X",
        "body_yaw_application": "compose_root_rotation prepends positive Y yaw; local +Z maps to world azimuth +yaw",
        "body_relative_formula": "wrap(world_candidate_azimuth_deg - placement_yaw_deg) in [-180,180)",
        "viewpoint_id_formula": "radius_index=viewpoint_id//8, azimuth_index=viewpoint_id%8",
        "max_position_azimuth_error_deg": float(max(item["position_azimuth_max_error_deg"] for item in metadata.values())),
        "max_forward_sign_error_deg": float(max(sign_errors) if sign_errors else float("nan")),
        "sample_count": len(samples),
        "samples": samples,
    }
    if not audit["convention_confirmed"]:
        raise ValueError("could not confirm positive yaw convention")
    return metadata, audit


def legal_ids_for_row(row: Mapping[str, Any]) -> list[int]:
    """Return Stage-C candidate ids when a row carries its candidate list."""
    values = row.get("candidate_viewpoint_ids")
    if values is None:
        raise ValueError("row does not contain candidate_viewpoint_ids")
    return [int(value) for value in values]


def build_utility_samples(
    data: Mapping[str, Any], metadata: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one utility map sample for every moving Val context."""
    rows = data["stage_d_rows"]["val"]
    cache = data["caches"]["val"]
    samples: list[dict[str, Any]] = []
    for row in rows:
        cache_index = int(row["cache_index"])
        scene_id, placement_id = str(row["scene_id"]), str(row["region"])
        meta = metadata[(scene_id, placement_id)]
        candidates = legal_ids(cache, cache_index)
        if not candidates:
            continue
        label = int(row["label_id"])
        utility: dict[tuple[int, int], float] = {}
        correct: dict[tuple[int, int], bool] = {}
        candidate_values: dict[int, dict[str, Any]] = {}
        for viewpoint_id in candidates:
            logp = candidate_logp(cache, cache_index, viewpoint_id)
            radius_index = int(viewpoint_id) // 8
            azimuth_bin = body_azimuth_bin(meta["azimuths"][int(viewpoint_id)], meta["yaw_deg"])
            cell = (radius_index, azimuth_bin)
            margin = gt_margin(logp, label)
            utility[cell] = float(margin)
            correct[cell] = bool(np.argmax(logp) == label)
            candidate_values[int(viewpoint_id)] = {
                "viewpoint_id": int(viewpoint_id),
                "radius_index": radius_index,
                "radius_m": float(meta["radii"][int(viewpoint_id)]),
                "world_azimuth_deg": float(meta["azimuths"][int(viewpoint_id)]),
                "body_relative_azimuth_deg": _wrap(float(meta["azimuths"][int(viewpoint_id)]) - float(meta["yaw_deg"])),
                "body_relative_bin": azimuth_bin,
                "margin": float(margin),
                "correct": bool(np.argmax(logp) == label),
            }
        samples.append({
            "sample_id": f"{row['record_id']}|{scene_id}|{placement_id}",
            "record_id": str(row["record_id"]),
            "scene_id": scene_id,
            "placement_id": placement_id,
            "label": label,
            "h1": int(row["s1_viewpoint_id"]),
            "episode_id": str(row["episode_id"]),
            "utility": utility,
            "correct": correct,
            "candidates": candidate_values,
        })
    return samples


def grouped_by(samples: Sequence[Mapping[str, Any]], key_fn: Any) -> dict[Any, list[Mapping[str, Any]]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[key_fn(sample)].append(sample)
    return dict(groups)


def flatten_cell_values(
    samples: Sequence[Mapping[str, Any]], *, cell: tuple[int, int] | None = None,
) -> list[tuple[Mapping[str, Any], tuple[int, int], float, bool]]:
    values: list[tuple[Mapping[str, Any], tuple[int, int], float, bool]] = []
    for sample in samples:
        utility = sample["utility"]
        for key, value in utility.items():
            if cell is None or tuple(key) == tuple(cell):
                values.append((sample, tuple(key), float(value), bool(sample["correct"][key])))
    return values


__all__ = [
    "AZIMUTH_COUNT", "BODY_AZIMUTH_BINS", "RADIUS_COUNT", "VIEW_COUNT",
    "body_azimuth_bin", "build_utility_samples", "flatten_cell_values",
    "grouped_by", "legal_ids_for_row", "load_scene_metadata",
]
