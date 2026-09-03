"""文件用途：
    提供 ActiveView 感知处理能力。

主要输入：
    - RGB 或归一化骨架数据。
主要输出：
    - 骨架、归一化表示或 RGB 特征。
项目角色：
    - 属于 perception 感知模块。
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


DINO_MODEL_NAME = "facebook/dinov2-base"
DINO_MODEL_VERSION = "transformers.AutoModel"
DINO_EMBED_DIM = 768
RGB_PROJECTOR_DIM = 128
EXP024_INPUT_DIM = 128 + 1 + 128 + 128 + 128
RGB_DATASET_VERSION = "activeview-rgb-observation-v1"
RGB_IMAGE_SIZE = 256
DINO_IMAGE_SIZE = 224


@dataclass(frozen=True)
class RGBObservationKey:
    """Stable key for one RGB viewpoint in the mirrored record dataset."""

    scene_id: str
    region: str
    record_id: str
    viewpoint_id: int

    @property
    def relative_record_path(self) -> str:
        return f"{self.scene_id}/{self.region}/{self.record_id}.npz"

    @property
    def tuple(self) -> tuple[str, str, str, int]:
        return (self.scene_id, self.region, self.record_id, self.viewpoint_id)


class RGBContextUtilityRegressor(nn.Module):
    """EXP024's trainable RGB projector plus executed-utility regression head."""

    input_dim = EXP024_INPUT_DIM

    def __init__(self) -> None:
        super().__init__()
        self.rgb_projector = nn.Sequential(nn.Linear(DINO_EMBED_DIM, RGB_PROJECTOR_DIM), nn.GELU())
        self.regression_head = nn.Sequential(
            nn.Linear(EXP024_INPUT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

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
        for name, value in (("rgb_s0", rgb_s0), ("rgb_s1", rgb_s1)):
            if value.ndim != 2 or value.shape != (contextual_token.size(0), DINO_EMBED_DIM):
                raise ValueError(f"{name} must have shape (batch, {DINO_EMBED_DIM})")
        z0 = self.rgb_projector(rgb_s0)
        z1 = self.rgb_projector(rgb_s1)
        features = torch.cat([contextual_token, predicted_utility, z0, z1, z1 - z0], dim=1)
        if features.size(-1) != EXP024_INPUT_DIM:
            raise RuntimeError("EXP024 feature dimension changed unexpectedly")
        return self.regression_head(features).squeeze(-1)


def observation_keys_from_feature_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[RGBObservationKey], dict[str, tuple[RGBObservationKey, RGBObservationKey]]]:
    """Return deduplicated s0/s1 keys and episode-to-key mapping.

    The function deliberately reads only ``s0_viewpoint_id`` and
    ``s1_viewpoint_id``; candidate IDs are never turned into RGB requests.
    """
    keys: set[RGBObservationKey] = set()
    episode_keys: dict[str, tuple[RGBObservationKey, RGBObservationKey]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in episode_keys:
            raise ValueError(f"Duplicate Stage-D episode_id: {episode_id}")
        common = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
        s0 = RGBObservationKey(*common, int(row["s0_viewpoint_id"]))
        s1 = RGBObservationKey(*common, int(row["s1_viewpoint_id"]))
        keys.update((s0, s1))
        episode_keys[episode_id] = (s0, s1)
    return sorted(keys, key=lambda item: item.tuple), episode_keys


def _scalar_text(value: np.ndarray) -> str:
    return str(value.item())


def _load_rgb_image(rgb_root: Path, key: RGBObservationKey) -> np.ndarray:
    """Load exactly one requested image from a validated RGB record archive."""
    path = rgb_root / key.relative_record_path
    if not path.is_file():
        raise FileNotFoundError(f"RGB record not found for {key.tuple}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"rgb", "viewpoint_ids", "scene_id", "region", "record_id", "rgb_observation_version"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"RGB record missing fields {sorted(missing)}: {path}")
        rgb = archive["rgb"]
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        if rgb.shape != (32, RGB_IMAGE_SIZE, RGB_IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Invalid RGB schema in {path}: {rgb.shape} {rgb.dtype}")
        if not np.array_equal(ids, np.arange(32, dtype=np.int64)):
            raise ValueError(f"RGB viewpoint IDs are not canonical in {path}")
        if _scalar_text(archive["rgb_observation_version"]) != RGB_DATASET_VERSION:
            raise ValueError(f"Unexpected RGB dataset version in {path}")
        if _scalar_text(archive["scene_id"]) != key.scene_id or _scalar_text(archive["region"]) != key.region:
            raise ValueError(f"RGB scene/region mismatch for {key.tuple}: {path}")
        if _scalar_text(archive["record_id"]) != key.record_id:
            raise ValueError(f"RGB record mismatch for {key.tuple}: {path}")
        return np.asarray(rgb[key.viewpoint_id], dtype=np.uint8)


def _load_rgb_images_for_record(
    path: Path,
    requested: Sequence[tuple[int, RGBObservationKey]],
) -> list[tuple[int, np.ndarray]]:
    """Validate/decompress one record once, then return requested view images."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"rgb", "viewpoint_ids", "scene_id", "region", "record_id", "rgb_observation_version"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"RGB record missing fields {sorted(missing)}: {path}")
        rgb = archive["rgb"]
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        if rgb.shape != (32, RGB_IMAGE_SIZE, RGB_IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Invalid RGB schema in {path}: {rgb.shape} {rgb.dtype}")
        if not np.array_equal(ids, np.arange(32, dtype=np.int64)):
            raise ValueError(f"RGB viewpoint IDs are not canonical in {path}")
        if _scalar_text(archive["rgb_observation_version"]) != RGB_DATASET_VERSION:
            raise ValueError(f"Unexpected RGB dataset version in {path}")
        output: list[tuple[int, np.ndarray]] = []
        for index, key in requested:
            if _scalar_text(archive["scene_id"]) != key.scene_id or _scalar_text(archive["region"]) != key.region or _scalar_text(archive["record_id"]) != key.record_id:
                raise ValueError(f"RGB metadata mismatch for {key.tuple}: {path}")
            output.append((index, np.asarray(rgb[key.viewpoint_id], dtype=np.uint8)))
        return output


def _preprocess_rgb(images: np.ndarray) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(images, dtype=np.uint8)).permute(0, 3, 1, 2).float() / 255.0
    values = F.interpolate(values, size=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE), mode="bilinear", align_corners=False)
    mean = values.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = values.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (values - mean) / std


def load_dinov2(device: torch.device) -> tuple[nn.Module, str]:
    """Load the official Hugging Face DINOv2 ViT-B/14 feature extractor."""
    try:
        from transformers import AutoModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("EXP024 requires transformers with DINOv2 support") from exc
    model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden = int(getattr(model.config, "hidden_size", -1))
    if hidden != DINO_EMBED_DIM:
        raise ValueError(f"DINOv2 hidden size must be {DINO_EMBED_DIM}, got {hidden}")
    return model, str(getattr(model.config, "_name_or_path", DINO_MODEL_NAME))


def _dino_embeddings(model: nn.Module, images: np.ndarray, device: torch.device) -> np.ndarray:
    pixels = _preprocess_rgb(images).to(device)
    with torch.inference_mode():
        output = model(pixel_values=pixels)
        values = output.last_hidden_state[:, 0, :]
    result = values.detach().cpu().numpy().astype(np.float16)
    if result.shape != (len(images), DINO_EMBED_DIM) or not np.isfinite(result).all():
        raise ValueError("DINOv2 output must be finite [N, 768] embeddings")
    return result


def _manifest_rows(keys: Sequence[RGBObservationKey]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": key.scene_id,
            "region": key.region,
            "record_id": key.record_id,
            "viewpoint_id": key.viewpoint_id,
            "source_rgb_relative_path": key.relative_record_path,
            "rgb_dataset_version": RGB_DATASET_VERSION,
            "embedding_shape": [DINO_EMBED_DIM],
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
        expected = _manifest_rows(keys)
        if embeddings.shape != (len(keys), DINO_EMBED_DIM) or embeddings.dtype != np.float16:
            return None
        if rows != expected:
            return None
        if not np.isfinite(np.asarray(embeddings, dtype=np.float32)).all():
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return embeddings, rows


def build_or_load_rgb_cache(
    *,
    rgb_root: Path,
    cache_dir: Path,
    keys: Sequence[RGBObservationKey],
    model_loader: Callable[[torch.device], tuple[nn.Module, str]],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Build or reuse the deduplicated visited-observation embedding cache."""
    ordered = sorted(keys, key=lambda item: item.tuple)
    if len(set(item.tuple for item in ordered)) != len(ordered):
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
    result = np.empty((len(ordered), DINO_EMBED_DIM), dtype=np.float16)
    pending_images: list[np.ndarray] = []
    pending_indices: list[int] = []

    def flush() -> None:
        if not pending_images:
            return
        embedded = _dino_embeddings(model, np.stack(pending_images), device)
        result[np.asarray(pending_indices, dtype=np.int64)] = embedded
        pending_images.clear()
        pending_indices.clear()

    for path in sorted(by_path):
        for index, image in _load_rgb_images_for_record(path, by_path[path]):
            pending_images.append(image)
            pending_indices.append(index)
            if len(pending_images) >= batch_size:
                flush()
    flush()
    manifest_rows = _manifest_rows(ordered)
    np.save(cache_dir / "embeddings.tmp.npy", result)
    (cache_dir / "manifest.tmp.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    (cache_dir / "summary.tmp.json").write_text(
        json.dumps(
            {
                "encoder_name": DINO_MODEL_NAME,
                "encoder_version": DINO_MODEL_VERSION,
                "model_config_name": model_version,
                "rgb_dataset_version": RGB_DATASET_VERSION,
                "embedding_shape": [DINO_EMBED_DIM],
                "dtype": "float16",
                "observation_count": len(ordered),
                "future_candidate_rgb_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (cache_dir / "embeddings.tmp.npy").replace(cache_dir / "embeddings.npy")
    (cache_dir / "manifest.tmp.jsonl").replace(cache_dir / "manifest.jsonl")
    (cache_dir / "summary.tmp.json").replace(cache_dir / "summary.json")
    elapsed = time.monotonic() - started
    return result, manifest_rows, {"cache_hit_count": 0, "cache_miss_count": len(ordered), "extraction_time_sec": elapsed, "cache_reused": False}


def embedding_index(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], int]:
    index: dict[tuple[str, str, str, int], int] = {}
    for row_index, row in enumerate(manifest_rows):
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"]))
        if key in index:
            raise ValueError(f"Duplicate RGB cache manifest key: {key}")
        index[key] = row_index
    return index
