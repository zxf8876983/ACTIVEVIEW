"""DINOv2 spatial patch-token cache for EXP025.

The cache is restricted to already visited Stage-D s0/s1 observations.  DINO
is used only as a frozen patch feature extractor; the spatial projector and
small regression head are trained by the EXP025 script.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from activeview.active_view.stage_d_rgb_context import (
    DINO_EMBED_DIM,
    DINO_MODEL_NAME,
    RGB_DATASET_VERSION,
    RGB_IMAGE_SIZE,
    RGBObservationKey,
    _load_rgb_images_for_record,
    _preprocess_rgb,
)


SPATIAL_GRID_SIZE = 4
SPATIAL_TOKEN_COUNT = SPATIAL_GRID_SIZE * SPATIAL_GRID_SIZE
SPATIAL_PATCH_GRID = 16


def dino_spatial_embeddings(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    """Return pooled 4x4 DINO patch tokens as float16 [N,16,768]."""
    pixels = _preprocess_rgb(images).to(device)
    with torch.inference_mode():
        output = model(pixel_values=pixels)
        tokens = output.last_hidden_state[:, 1:, :]
    if tokens.shape[1] != SPATIAL_PATCH_GRID * SPATIAL_PATCH_GRID:
        raise ValueError("DINOv2 ViT-B/14 must produce a 16x16 patch grid")
    grid = tokens.transpose(1, 2).reshape(-1, DINO_EMBED_DIM, SPATIAL_PATCH_GRID, SPATIAL_PATCH_GRID)
    pooled = F.adaptive_avg_pool2d(grid, (SPATIAL_GRID_SIZE, SPATIAL_GRID_SIZE))
    result = pooled.flatten(2).transpose(1, 2).detach().cpu().numpy().astype(np.float16)
    expected = (len(images), SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM)
    if result.shape != expected or not np.isfinite(result.astype(np.float32)).all():
        raise ValueError(f"Invalid DINO spatial embedding shape: {result.shape}")
    return result


class SpatialRGBUtilityRegressor(nn.Module):
    """EXP025 spatial projector, encoder and executed-utility head."""

    input_dim = 128 + 1 + 128 + 128 + 128

    def __init__(self) -> None:
        super().__init__()
        self.rgb_projector = nn.Sequential(nn.Linear(DINO_EMBED_DIM, 128), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.regression_head = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM):
            raise ValueError("spatial RGB inputs must have shape (batch, 16, 768)")
        projected = self.rgb_projector(values)
        return self.spatial_encoder(projected).mean(dim=1)

    def forward(
        self,
        contextual_token: torch.Tensor,
        predicted_utility: torch.Tensor,
        rgb_s0: torch.Tensor,
        rgb_s1: torch.Tensor,
    ) -> torch.Tensor:
        if contextual_token.ndim != 2 or contextual_token.size(-1) != 128:
            raise ValueError("contextual_token must have shape (batch, 128)")
        if predicted_utility.ndim == 1:
            predicted_utility = predicted_utility.unsqueeze(1)
        if predicted_utility.shape != (contextual_token.size(0), 1):
            raise ValueError("predicted_utility must have shape (batch, 1)")
        if rgb_s0.shape != (contextual_token.size(0), SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM):
            raise ValueError("rgb_s0 must have shape (batch, 16, 768)")
        if rgb_s1.shape != rgb_s0.shape:
            raise ValueError("rgb_s1 must align with rgb_s0")
        z0 = self._encode(rgb_s0)
        z1 = self._encode(rgb_s1)
        features = torch.cat([contextual_token, predicted_utility, z0, z1, z1 - z0], dim=1)
        if features.size(-1) != self.input_dim:
            raise RuntimeError("EXP025 feature dimension changed unexpectedly")
        return self.regression_head(features).squeeze(-1)


def _manifest_rows(keys: Sequence[RGBObservationKey]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": key.scene_id,
            "region": key.region,
            "record_id": key.record_id,
            "viewpoint_id": key.viewpoint_id,
            "source_rgb_relative_path": key.relative_record_path,
            "rgb_dataset_version": RGB_DATASET_VERSION,
            "embedding_shape": [SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM],
            "dtype": "float16",
        }
        for key in keys
    ]


def _load_valid_cache(cache_dir: Path, keys: Sequence[RGBObservationKey]) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    embeddings_path = cache_dir / "embeddings.npy"
    manifest_path = cache_dir / "manifest.jsonl"
    if not embeddings_path.is_file() or not manifest_path.is_file():
        return None
    try:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if embeddings.shape != (len(keys), SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM) or embeddings.dtype != np.float16:
            return None
        if rows != _manifest_rows(keys):
            return None
        if not np.isfinite(np.asarray(embeddings, dtype=np.float32)).all():
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return embeddings, rows


def build_or_load_spatial_cache(
    *,
    rgb_root: Path,
    cache_dir: Path,
    keys: Sequence[RGBObservationKey],
    model_loader: Callable[[torch.device], tuple[nn.Module, str]],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Build/reuse the deduplicated visited-observation spatial cache."""
    ordered = sorted(keys, key=lambda item: item.tuple)
    if len(set(key.tuple for key in ordered)) != len(ordered):
        raise ValueError("Duplicate RGB observation key")
    cached = _load_valid_cache(cache_dir, ordered)
    if cached is not None:
        embeddings, rows = cached
        return embeddings, rows, {"cache_hit_count": len(ordered), "cache_miss_count": 0, "extraction_time_sec": 0.0, "cache_reused": True}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, model_version = model_loader(device)
    by_path: dict[Path, list[tuple[int, RGBObservationKey]]] = defaultdict(list)
    for index, key in enumerate(ordered):
        by_path[rgb_root / key.relative_record_path].append((index, key))
    result = np.empty((len(ordered), SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM), dtype=np.float16)
    pending_images: list[np.ndarray] = []
    pending_indices: list[int] = []

    def flush() -> None:
        if not pending_images:
            return
        values = dino_spatial_embeddings(model, np.stack(pending_images), device)
        result[np.asarray(pending_indices, dtype=np.int64)] = values
        pending_images.clear()
        pending_indices.clear()

    for path in sorted(by_path):
        for index, image in _load_rgb_images_for_record(path, by_path[path]):
            pending_images.append(image)
            pending_indices.append(index)
            if len(pending_images) >= batch_size:
                flush()
    flush()
    rows = _manifest_rows(ordered)
    np.save(cache_dir / "embeddings.tmp.npy", result)
    (cache_dir / "manifest.tmp.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (cache_dir / "summary.tmp.json").write_text(
        json.dumps(
            {
                "encoder_name": DINO_MODEL_NAME,
                "encoder_version": "transformers.AutoModel",
                "model_config_name": model_version,
                "rgb_dataset_version": RGB_DATASET_VERSION,
                "patch_grid": [SPATIAL_PATCH_GRID, SPATIAL_PATCH_GRID],
                "pooled_grid": [SPATIAL_GRID_SIZE, SPATIAL_GRID_SIZE],
                "embedding_shape": [SPATIAL_TOKEN_COUNT, DINO_EMBED_DIM],
                "dtype": "float16",
                "observation_count": len(ordered),
                "future_candidate_rgb_used": False,
            },
            indent=2,
        ), encoding="utf-8"
    )
    (cache_dir / "embeddings.tmp.npy").replace(cache_dir / "embeddings.npy")
    (cache_dir / "manifest.tmp.jsonl").replace(cache_dir / "manifest.jsonl")
    (cache_dir / "summary.tmp.json").replace(cache_dir / "summary.json")
    return result, rows, {"cache_hit_count": 0, "cache_miss_count": len(ordered), "extraction_time_sec": time.monotonic() - started, "cache_reused": False}


def spatial_embedding_index(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], int]:
    index: dict[tuple[str, str, str, int], int] = {}
    for row_index, row in enumerate(manifest_rows):
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"]))
        if key in index:
            raise ValueError(f"Duplicate spatial cache manifest key: {key}")
        index[key] = row_index
    return index
