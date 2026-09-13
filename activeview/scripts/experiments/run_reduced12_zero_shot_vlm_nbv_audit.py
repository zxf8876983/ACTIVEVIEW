#!/usr/bin/env python3
"""Zero-shot VLM NBV viability audit on a fixed Moving-Val subset.

The VLM sees only the current frame-0 RGB (where enabled), static Habitat
top-down geometry, the current pose and legal candidate metadata.  Future
candidate observations are consumed only after selection for terminal HAR.
Large image packages and response caches are written below
``ACTIVEVIEW_DATA_ROOT``; only small summaries are tracked in Git.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    _option_geometry,
)
from activeview.scripts.experiments.run_reduced12_map_aware_robust_observability_audit import (
    MAP_DIM,
    SCENE_CLEARANCE_INDEX,
    UtilityMLP,
)
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import _scene_sim

SEED = 42
NUM_CLASSES = 12
NUM_VIEWS = 32
MAX_OPTIONS = 22
FEATURE_DIM = 256
SUBSET_SIZE = 1000
RESOLUTION = 0.20
VLM_DEFAULT_URL = "http://127.0.0.1:18080"
VLM_DEFAULT_MODEL = "Qwen3.5-35B-A3B"
VLM_DEFAULT_PATH = "/home/zxf/llama/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
VLM_DEFAULT_MMPROJ = "/home/zxf/llama/models/Qwen3.5-35B-A3B-GGUF/mmproj-F16.gguf"
POLICY_REL = Path("datasets/policy_reduced12_eight_placement_v1")
RGB_REL = Path("datasets/rgb_reduced12_eight_placement_v1/frame0_current")
MAP_REL = Path("diagnostics/map_aware_robust_observability_audit")
SCALAR_REL = Path("diagnostics/frame0_visibility_predictor_v1")
SHARED_CKPT = Path("checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth")
GEOM_CKPT = Path("checkpoints/policy_reduced12_eight_placement_v1/privileged_information_ladder/geometryonly_utility.pth")
SCALAR_CKPT = Path("checkpoints/policy_reduced12_eight_placement_v1/structured_observability_final_audit/scalarvisibility_geometry.pth")
MAP_CKPT = Path("checkpoints/policy_reduced12_eight_placement_v1/map_aware_robust_observability_audit/mapfeature_utility.pth")
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/zero_shot_vlm_nbv_viability"
PACKAGE_VERSION = "floor_aware_static_map_v2_single_image_vlm"


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _validate_cache(path: Path, meta_path: Path, rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, np.ndarray]:
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"missing {name} cache: {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        raise ValueError(f"{name} cache signature does not match Moving Val rows")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"{name} cache is marked Test-derived")
    return _load_npz(path)


def _actions(row: Mapping[str, Any]) -> list[int]:
    actions = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
    if len(actions) > MAX_OPTIONS or len(actions) != len(set(actions)):
        raise ValueError(f"invalid legal action set for {row['episode_id']}")
    return actions


def _select_subset(rows: Sequence[Mapping[str, Any]], path: Path) -> list[int]:
    if path.is_file():
        values = json.loads(path.read_text(encoding="utf-8"))
        ids = values.get("context_ids", values) if isinstance(values, dict) else values
        by_id = {str(row["episode_id"]): i for i, row in enumerate(rows)}
        selected = [by_id[str(item)] for item in ids]
        if len(selected) != SUBSET_SIZE or len(set(selected)) != SUBSET_SIZE:
            raise ValueError("existing subset must contain exactly 1000 unique Moving-Val contexts")
        return selected
    rng = np.random.default_rng(SEED)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[int(row["label_id"])].append(index)
    selected: list[int] = []
    quotas = [SUBSET_SIZE // NUM_CLASSES + int(i < SUBSET_SIZE % NUM_CLASSES) for i in range(NUM_CLASSES)]
    for label in range(NUM_CLASSES):
        values = np.asarray(grouped[label], dtype=np.int64)
        if values.size < quotas[label]:
            raise ValueError(f"not enough Val contexts for class {label}")
        selected.extend(rng.choice(values, quotas[label], replace=False).tolist())
    rng.shuffle(selected)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": SEED, "context_ids": [str(rows[i]["episode_id"]) for i in selected], "count": len(selected), "test_used": False}, indent=2) + "\n", encoding="utf-8")
    return selected


def _head_logits(head: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIM)
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), 4096):
            values.append(head(torch.from_numpy(flat[start : start + 4096]).to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(values).reshape(*features.shape[:-1], NUM_CLASSES).astype(np.float32)


def _load_head(data_root: Path, device: torch.device) -> SharedHead:
    model = SharedHead().to(device)
    payload = torch.load(data_root / SHARED_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def _load_utility(data_root: Path, checkpoint: Path, input_dim: int, device: torch.device) -> UtilityMLP:
    model = UtilityMLP(input_dim).to(device)
    payload = torch.load(data_root / checkpoint, map_location=device, weights_only=False)
    if int(payload.get("input_dim", input_dim)) != input_dim or bool(payload.get("test_used", True)):
        raise ValueError(f"invalid utility checkpoint metadata: {checkpoint}")
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def _predict_utility(model: UtilityMLP, values: np.ndarray, device: torch.device) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 4096):
            output.append(model(torch.from_numpy(values[start : start + 4096]).to(device, non_blocking=True)).cpu().numpy())
    values = np.concatenate(output)
    return values.squeeze(-1) if values.ndim > 1 else values


def _stable_select(scores: np.ndarray, ids: np.ndarray, mask: np.ndarray, costs: np.ndarray | None = None) -> list[int]:
    actions: list[int] = []
    for index in range(scores.shape[0]):
        valid = np.flatnonzero(mask[index])
        def key(slot: int) -> tuple[float, float, int]:
            cost = 0.0 if costs is None else float(costs[index, slot])
            return (-float(scores[index, slot]), cost, int(ids[index, slot]))
        slot = min(valid.tolist(), key=key)
        actions.append(int(ids[index, slot]))
    return actions


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero(mask[index] & (ids[index] == int(action)))
        if slots.size != 1:
            raise ValueError(f"selected action {action} is not uniquely legal at row {index}")
        predictions.append(int(np.argmax(logp[index, int(slots[0])])))
    return np.asarray(predictions, dtype=np.int64)


def _metric(name: str, labels: np.ndarray, predictions: np.ndarray, rows: Sequence[Mapping[str, Any]], actions: Sequence[int], map_features: np.ndarray | None = None, ids: np.ndarray | None = None, mask: np.ndarray | None = None) -> dict[str, Any]:
    value = classification(labels, predictions)
    moves = np.asarray([int(action) != int(row["current_viewpoint_id"]) for row, action in zip(rows, actions)], dtype=bool)
    result: dict[str, Any] = {"method": name, **value, "move_rate": float(moves.mean()), "stay_rate": float(1.0 - moves.mean()), "test_used": False}
    if map_features is not None and ids is not None and mask is not None:
        costs = []
        for index, action in enumerate(actions):
            slot = int(np.flatnonzero(mask[index] & (ids[index] == int(action)))[0])
            costs.append(float(map_features[index, slot, 42]))
        result["mean_nav_distance_m"] = float(np.mean(costs))
    return result


def _row_archive(row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return _load_npz(Path(str(row["archive_path"])))


def _rgb_path(data_root: Path, row: Mapping[str, Any]) -> Path:
    return data_root / RGB_REL / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz"


def _scene_map(data_root: Path, habitat_root: Path, row: Mapping[str, Any], runtime: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scene_id = str(row["scene_id"])
    archive = _row_archive(row)
    floor_y = float(np.asarray(archive["placement_position"])[1])
    floor_key = f"{floor_y:.2f}"
    cache_key = f"{scene_id}@{floor_key}"
    if cache_key in cache:
        return cache[cache_key]
    map_dir = runtime / "scene_maps"
    map_path = map_dir / f"{scene_id}__y{floor_key.replace('-', 'm').replace('.', 'p')}.npz"
    if map_path.is_file():
        loaded = _load_npz(map_path)
        info = {"grid": loaded["grid"].astype(bool), "lower": loaded["lower"].astype(np.float64), "resolution": float(loaded["resolution"][0]), "available": True}
        cache[cache_key] = info
        return info
    try:
        sim = _scene_sim(habitat_root / "hm3d-train", scene_id, physics=False)
        try:
            grid = np.asarray(sim.pathfinder.get_topdown_view(RESOLUTION, floor_y, eps=0.5), dtype=bool)
            lower, _ = sim.pathfinder.get_bounds()
        finally:
            sim.close()
        if grid.ndim != 2 or not grid.any():
            raise RuntimeError("empty navmesh slice")
        info = {"grid": grid, "lower": np.asarray(lower, dtype=np.float64), "resolution": RESOLUTION, "available": True}
        map_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(map_path, grid=grid.astype(np.uint8), lower=info["lower"], resolution=np.asarray([RESOLUTION]))
    except Exception as exc:  # noqa: BLE001 - map rendering has an explicit, recorded fallback
        info = {"grid": None, "lower": np.zeros(3), "resolution": RESOLUTION, "available": False, "error": str(exc)}
    cache[cache_key] = info
    return info


def _furniture(data_root: Path, scene_id: str) -> list[Mapping[str, Any]]:
    path = data_root / "datasets/offline/habitat-train/00006-00087/semantic_furniture" / scene_id / "furniture_positions.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("objects", []))


def _plot_map(ax: Any, row: Mapping[str, Any], archive: Mapping[str, np.ndarray], scene_info: Mapping[str, Any], data_root: Path, annotate: bool = True) -> None:
    positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
    human = np.asarray(archive["placement_position"], dtype=np.float32)
    current = positions[int(row["current_viewpoint_id"])]
    legal = [int(x) for x in row["candidate_ids"]]
    points = np.vstack([human[None], current[None], positions[legal]])
    extent = max(3.0, float(np.ptp(points[:, [0, 2]], axis=0).max()) * 0.65)
    if scene_info.get("available"):
        grid = np.asarray(scene_info["grid"], dtype=float)
        lower = np.asarray(scene_info["lower"])
        h, w = grid.shape
        ax.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.30, extent=(lower[0], lower[0] + w * scene_info["resolution"], lower[2], lower[2] + h * scene_info["resolution"]))
    for obj in _furniture(data_root, str(row["scene_id"])):
        lo, hi = obj.get("bounds_min_xyz"), obj.get("bounds_max_xyz")
        if lo is None or hi is None:
            continue
        lo, hi = np.asarray(lo), np.asarray(hi)
        ax.add_patch(plt.Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], facecolor="none", edgecolor="0.35", linewidth=0.5))
    ax.scatter(human[0], human[2], marker="*", s=55, c="black", label="H")
    ax.scatter(current[0], current[2], marker="s", s=25, c="tab:blue", label="S")
    for n, viewpoint in enumerate(legal, 1):
        candidate = positions[viewpoint]
        ax.plot([candidate[0], human[0]], [candidate[2], human[2]], color="0.45", linewidth=0.5, linestyle="--")
        ax.text(candidate[0], candidate[2], f"C{n}", fontsize=6, ha="center", va="center", bbox={"facecolor": "white", "alpha": 0.65, "pad": 0.2, "edgecolor": "none"})
    ax.text(current[0], current[2], "STAY", fontsize=6, ha="left", va="bottom")
    ax.set_xlim(float(human[0] - extent), float(human[0] + extent)); ax.set_ylim(float(human[2] - extent), float(human[2] + extent)); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    if annotate:
        ax.set_title("Static Habitat top-down map")


def _draw_current_rgb(data_root: Path, row: Mapping[str, Any], destination: Path) -> None:
    source = _rgb_path(data_root, row)
    if not source.is_file():
        raise FileNotFoundError(f"missing current frame-0 RGB cache: {source}")
    archive = _load_npz(source)
    required = {"rgb", "available_view_mask", "viewpoint_ids", "scene_id", "region", "record_id"}
    if not required.issubset(archive):
        raise ValueError(f"RGB cache missing metadata for {row['episode_id']}")
    rgb = np.asarray(archive["rgb"])
    if rgb.shape != (NUM_VIEWS, 256, 256, 3):
        raise ValueError(f"unexpected RGB shape {rgb.shape} for {row['episode_id']}")
    viewpoint = int(row["current_viewpoint_id"])
    available = np.asarray(archive["available_view_mask"], dtype=bool)
    ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
    if viewpoint >= len(available) or not bool(available[viewpoint]) or int(ids[viewpoint]) != viewpoint:
        raise ValueError(f"current viewpoint {viewpoint} is unavailable in RGB cache for {row['episode_id']}")
    for key in ("scene_id", "region", "record_id"):
        if str(np.asarray(archive[key]).item()) != str(row[key]):
            raise ValueError(f"RGB metadata mismatch ({key}) for {row['episode_id']}")
    Image.fromarray(rgb[viewpoint].astype(np.uint8)).save(destination)


def _draw_board(row: Mapping[str, Any], archive: Mapping[str, np.ndarray], scene_info: Mapping[str, Any], data_root: Path, destination: Path) -> None:
    legal = [int(x) for x in row["candidate_ids"]]
    labels = ["STAY"] + [f"C{i}" for i in range(1, len(legal) + 1)]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), squeeze=False)
    current = int(row["current_viewpoint_id"]); positions = np.asarray(archive["viewpoint_agent_positions"])
    pose = np.asarray(archive["skeleton"])[current, :, 0, :].T
    pose = pose - pose[0]
    for panel, ax in enumerate(axes.flat):
        if panel >= len(labels):
            ax.axis("off"); continue
        _plot_map(ax, row, archive, scene_info, data_root, annotate=False)
        viewpoint = current if panel == 0 else legal[panel - 1]
        camera = positions[viewpoint]; human = np.asarray(archive["placement_position"])
        ax.scatter(camera[0], camera[2], marker="o", s=36, facecolors="none", edgecolors="tab:red")
        ax.plot([camera[0], human[0]], [camera[2], human[2]], color="tab:red", linewidth=1.0)
        # Current frame-0 pose is drawn in a small candidate-facing local inset.
        right = np.asarray((human - camera)[[0, 2]], dtype=np.float64); norm = np.linalg.norm(right)
        if norm > 1e-6:
            right /= norm
            x2 = pose[:, 0] * right[0] + pose[:, 2] * right[1]
            y2 = pose[:, 1]
            x2 = x2 / max(np.ptp(x2), 1e-5) * 0.8 + human[0]; y2 = y2 / max(np.ptp(y2), 1e-5) * 0.8 + human[2]
            edges = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16))
            for left, right_joint in edges:
                ax.plot([x2[left], x2[right_joint]], [y2[left], y2[right_joint]], color="tab:orange", linewidth=0.8)
        ax.set_title(f"{labels[panel]} static geometry", fontsize=8)
    fig.suptitle("Candidate comparison board (current frame-0 pose only; no future observation)", fontsize=10)
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination, dpi=120); plt.close(fig)


def _make_composite(paths: Sequence[Path], destination: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = sum(image.width for image in images); height = max(image.height for image in images)
    if width > 600:
        scale = 600.0 / width
        images = [image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS) for image in images]
        width = sum(image.width for image in images); height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (255, 255, 255)); offset = 0
    for image in images:
        canvas.paste(image, (offset, (height - image.height) // 2)); offset += image.width
    destination.parent.mkdir(parents=True, exist_ok=True); canvas.save(destination, format="PNG")


def _resize_rgb(path: Path, max_width: int = 600) -> None:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            scale = max_width / float(image.width)
            image = image.resize((max_width, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        image.save(path, format="PNG")


def _metadata_text(row: Mapping[str, Any], archive: Mapping[str, np.ndarray]) -> str:
    positions = np.asarray(archive["viewpoint_agent_positions"])
    current = positions[int(row["current_viewpoint_id"])]
    lines = ["STAY: nav_distance=0.00 m"]
    for index, (viewpoint, cost) in enumerate(zip(row["candidate_ids"], row["candidate_geodesic"]), 1):
        point = positions[int(viewpoint)]
        delta = point - current
        azimuth = math.degrees(math.atan2(float(delta[2]), float(delta[0])))
        lines.append(f"C{index}: viewpoint_id={int(viewpoint)}, nav_distance={float(cost):.2f} m, relative_azimuth={azimuth:+.1f} deg")
    return "\n".join(lines)


PROMPT = """The future human action is unknown. Do not guess a specific action. Choose one legal observation waypoint that is most likely to provide a stable, unobstructed, well-framed full-body observation during unknown future motion. Consider static layout, obstacles, viewing corridors, body framing, free space, distance, and robustness to small motion. Observation quality is primary; navigation distance is secondary. Return concise JSON only: {\"ranking\":[\"C1\",\"STAY\"],\"selected\":\"C1\",\"scores\":{}}. Use only supplied IDs; scores may remain empty. No explanation or chain of thought."""


def _image_data(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _request_vlm(url: str, model: str, text: str, images: Sequence[Path], max_tokens: int = 256) -> tuple[str, float]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend({"type": "image_url", "image_url": {"url": _image_data(path)}} for path in images)
    body: dict[str, Any] = {"model": model, "temperature": 0, "seed": SEED, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started
    return str(payload["choices"][0]["message"].get("content", "")), elapsed


def _parse_response(raw: str, allowed: set[str]) -> dict[str, Any] | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    selected = value.get("selected")
    ranking = value.get("ranking", [])
    if selected not in allowed or not isinstance(ranking, list) or any(item not in allowed for item in ranking):
        return None
    scores = value.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    return {"ranking": [str(item) for item in ranking], "selected": str(selected), "scores": scores}


def _infer_branch(branch: str, row: Mapping[str, Any], package: Mapping[str, Path], server: str, model: str, response_path: Path) -> dict[str, Any]:
    allowed = {"STAY"} | {f"C{i}" for i in range(1, len(row["candidate_ids"]) + 1)}
    if response_path.is_file():
        cached = json.loads(response_path.read_text(encoding="utf-8"))
        expected_hash = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
        expected_images = {key: sha256(value) for key, value in package.items() if key in {"rgb", "map", "board", "map_board", "rgb_map"} and isinstance(value, Path) and value.is_file()}
        if cached.get("context_id") == str(row["episode_id"]) and (not cached.get("fallback", False) or cached.get("skip_reason")) and cached.get("branch") == branch and cached.get("model") == model and cached.get("prompt_hash") == expected_hash and cached.get("input_image_hashes") == expected_images and cached.get("selected") in allowed and isinstance(cached.get("parsed"), dict):
            return cached
    images: list[Path] = []
    if branch == "VLM-RGB":
        images.append(package["rgb"])
    elif branch == "VLM-Map":
        # llama.cpp's OpenAI-compatible endpoint accepts one image reliably;
        # the board already contains the static map and all candidate panels.
        images.append(package["board"])
    else:
        images.append(package["rgb_map"])
    prompt = PROMPT + "\n\nCandidate metadata:\n" + str(package["metadata_text"])
    parse_failure = False; repair = False; fallback = False; latency = 0.0
    try:
        raw, latency = _request_vlm(server, model, prompt, images)
        parsed = _parse_response(raw, allowed)
        if parsed is None:
            parse_failure = True; repair = True
            repair_prompt = prompt + "\n\nReturn valid JSON using only the previously supplied candidate IDs. Do not add any new information."
            raw, repair_latency = _request_vlm(server, model, repair_prompt, images, max_tokens=128)
            latency += repair_latency; parsed = _parse_response(raw, allowed)
        if parsed is None:
            fallback = True; parsed = {"ranking": ["STAY"], "selected": "STAY", "scores": {}}
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 413, 422}:
            raise RuntimeError(f"VLM HTTP failure for {branch}/{row['episode_id']}: {exc}") from exc
        parse_failure = True; fallback = True; raw = f"http_error_{exc.code}: {exc}"; parsed = {"ranking": ["STAY"], "selected": "STAY", "scores": {}}
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"VLM request failed for {branch}/{row['episode_id']}: {exc}") from exc
    prompt_hash = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    image_hashes = {key: sha256(value) for key, value in package.items() if key in {"rgb", "map", "board", "map_board", "rgb_map"} and isinstance(value, Path) and value.is_file()}
    result = {"context_id": str(row["episode_id"]), "branch": branch, "model": model, "prompt_hash": prompt_hash, "input_image_hashes": image_hashes, "raw_response": raw, "parsed": parsed, "selected": parsed["selected"], "parse_failure": parse_failure, "repair": repair, "fallback": fallback, "latency_seconds": latency, "test_used": False}
    response_path.parent.mkdir(parents=True, exist_ok=True); response_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _id_to_action(row: Mapping[str, Any], identifier: str) -> int:
    if identifier == "STAY":
        return int(row["current_viewpoint_id"])
    index = int(identifier[1:]) - 1
    return int(row["candidate_ids"][index])


def _ranking_diagnostics(branch: str, records: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], true_logp: np.ndarray) -> dict[str, Any]:
    values: list[float] = []; top1 = []; top3 = []; stay = []; move = []
    for index, record in enumerate(records):
        legal = ["STAY"] + [f"C{i}" for i in range(1, len(rows[index]["candidate_ids"]) + 1)]
        target = np.asarray([true_logp[index, slot, int(rows[index]["label_id"])] for slot in range(len(legal))], dtype=np.float64)
        ranking = [x for x in record["parsed"].get("ranking", []) if x in legal]
        ranking.extend(x for x in legal if x not in ranking)
        predicted = np.asarray([len(ranking) - ranking.index(x) for x in legal], dtype=np.float64)
        values.append(correlation(predicted, target, spearman=True))
        oracle = legal[int(np.argmax(target))]
        top1.append(int(ranking[0] == oracle)); top3.append(int(oracle in ranking[:3])); stay.append(int((ranking[0] == "STAY") == (oracle == "STAY"))); move.append(int((ranking[0] != "STAY") == (oracle != "STAY")))
    return {"branch": branch, "within_context_spearman_mean": float(np.mean(values)), "within_context_spearman_median": float(np.median(values)), "gt_true_logp_top1_overlap": float(np.mean(top1)), "gt_true_logp_top3_overlap": float(np.mean(top3)), "stay_agreement": float(np.mean(stay)), "move_agreement": float(np.mean(move)), "test_used": False}


def _subset_metrics(name: str, metric: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metric[key] for key in ("method", "n", "accuracy", "macro_f1", "move_rate", "stay_rate", "mean_nav_distance_m") if key in metric}


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    baseline = result["matched_baselines"]; branches = result["vlm_branches"]; scalar = float(baseline["ScalarVisibility+Geometry"]["accuracy"])
    best_name, best = max(branches.items(), key=lambda item: float(item[1]["accuracy"]))
    gain = float(best["accuracy"]) - scalar
    decision = "KILL" if gain <= 0.01 else "BORDERLINE" if gain < 0.03 else "KEEP" if gain < 0.05 else "STRONG KEEP"
    lines = ["# Zero-shot VLM NBV Viability Audit", "", "```text", "Subset: 1000 fixed Moving-Val contexts", "Seed: 42", "Training: none", "VLM fine-tuning: false", "Policy Test: false", "Problem: pre-action single-step NBV for HAR", "Future action: unknown", "Known static Habitat map: true", "Current exact frame0 pose: privileged viability input", "Future human motion: false", "Candidate real RGB: false", "Candidate real skeleton: false", "Candidate recognizer output: false", "Action set: Stay + Stage-A legal candidates", "Terminal HAR: selected real viewpoint full 30-frame skeleton -> frozen ST-GCN + shared head", "```", "", "## Matched subset results", "", "| Method | Acc | Macro-F1 | Move rate | Mean nav (m) |", "|---|---:|---:|---:|---:|"]
    for name, metric in {**baseline, **branches}.items():
        lines.append(f"| {name} | {float(metric['accuracy']):.6f} | {float(metric['macro_f1']):.6f} | {float(metric.get('move_rate', 0.0)):.4f} | {float(metric.get('mean_nav_distance_m', 0.0)):.3f} |")
    lines.extend(["", f"Best VLM branch: **{best_name}** ({float(best['accuracy']):.6f}); gain over matched ScalarVisibility+Geometry: {gain * 100:.2f} pp.", f"Gain over matched MapFeature-Utility: {(float(best['accuracy']) - float(baseline['MapFeature-Utility']['accuracy'])) * 100:.2f} pp.", f"MapGain (VLM-Map - VLM-RGB): {(float(branches['VLM-Map']['accuracy']) - float(branches['VLM-RGB']['accuracy'])) * 100:.2f} pp.", f"RGBConditionalGain (VLM-RGB+Map - VLM-Map): {(float(branches['VLM-RGB+Map']['accuracy']) - float(branches['VLM-Map']['accuracy'])) * 100:.2f} pp.", f"Map package availability: {float(result.get('map_available_rate', 0.0)) * 100:.2f}%.", f"Run status: {result.get('status', 'UNKNOWN')}.", "", f"## Decision: {decision}", "", "The current exact frame-0 pose is explicitly privileged for this viability audit; it is not a deployable claim.", "If KILL, do not swap a larger VLM or fine-tune automatically; retain the matched negative result.", "", "## Leakage flags", "", "All formal VLM branches set uses_gt_action=false, uses_future_human_motion=false, uses_candidate_real_rgb=false, uses_candidate_real_skeleton=false and uses_candidate_recognizer_output=false."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(); device = require_cuda(args.device); baseline_device = torch.device(args.baseline_device); data_root = get_data_root(); habitat_root = get_habitat_data_root(); started = time.time()
    train_rows, val_rows = load_rows(data_root)
    if len(val_rows) != 10080:
        raise ValueError(f"expected 10080 Moving-Val contexts, got {len(val_rows)}")
    output = Path(args.output_root) if args.output_root else OUTPUT_ROOT; output.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime_root) if args.runtime_root else data_root / "experiments/reduced12_eight_placement_v1/zero_shot_vlm_nbv_viability_runtime"; runtime.mkdir(parents=True, exist_ok=True)
    subset_path = output / "vlm_subset_context_ids.json"; subset_indices = _select_subset(val_rows, subset_path); rows = [val_rows[i] for i in subset_indices]; labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    shared = _validate_cache(data_root / "diagnostics/reduced12_dual_route_overnight/val_all32.npz", data_root / "diagnostics/reduced12_dual_route_overnight/val_all32.json", val_rows, "all32")
    option_cache = _validate_cache(data_root / "diagnostics/reduced12_dual_route_overnight/val_options.npz", data_root / "diagnostics/reduced12_dual_route_overnight/val_options.json", val_rows, "legal option")
    map_cache = _validate_cache(data_root / MAP_REL / "val.npz", data_root / MAP_REL / "val.json", val_rows, "map")
    scalar_cache = _validate_cache(data_root / SCALAR_REL / "val.npz", data_root / SCALAR_REL / "val.json", val_rows, "scalar visibility")
    if not np.array_equal(option_cache["ids"], map_cache["ids"]) or not np.array_equal(option_cache["mask"], map_cache["mask"]):
        raise ValueError("map cache legal action ids/mask do not match recognizer option cache")
    if not np.array_equal(option_cache["ids"], scalar_cache["ids"]) or not np.array_equal(option_cache["mask"], scalar_cache["mask"]):
        raise ValueError("scalar cache legal action ids/mask do not match recognizer option cache")
    # The local 35B VLM occupies nearly all 24 GB on the RTX 4090.  Keep the
    # tiny frozen-head/baseline computation on CPU by default while the VLM
    # itself remains fully GPU-offloaded in llama-server.
    head = _load_head(data_root, baseline_device)
    all_features = np.asarray(shared["features"], dtype=np.float32)[subset_indices]
    all_logits = _head_logits(head, all_features, baseline_device); all_logp = all_logits - np.logaddexp.reduce(all_logits, axis=-1, keepdims=True)
    option_features = np.asarray(option_cache["features"], dtype=np.float32)[subset_indices]; option_ids = np.asarray(option_cache["ids"], dtype=np.int64)[subset_indices]; option_mask = np.asarray(option_cache["mask"], dtype=bool)[subset_indices]
    option_logits = _head_logits(head, option_features, baseline_device); option_logp = option_logits - np.logaddexp.reduce(option_logits, axis=-1, keepdims=True)
    stats = json.loads((data_root / POLICY_REL / "stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8")); geometry, geom_ids, geom_mask = _option_geometry(rows, stats)
    if not np.array_equal(option_ids, geom_ids) or not np.array_equal(option_mask, geom_mask):
        raise ValueError("candidate geometry and recognizer legal action sets differ")
    costs = np.zeros_like(option_ids, dtype=np.float32)
    for index, row in enumerate(rows):
        costs[index, 1 : len(row["candidate_geodesic"]) + 1] = np.asarray(row["candidate_geodesic"], dtype=np.float32)
    scalar_values = np.asarray(scalar_cache["scores"], dtype=np.float32)[subset_indices][..., None]; map_values = np.asarray(map_cache["features"], dtype=np.float32)[subset_indices]
    geom_model = _load_utility(data_root, GEOM_CKPT, 18, baseline_device); scalar_model = _load_utility(data_root, SCALAR_CKPT, 19, baseline_device); map_model = _load_utility(data_root, MAP_CKPT, 114, baseline_device)
    geom_scores = _predict_utility(geom_model, geometry.reshape(-1, 18), baseline_device).reshape(len(rows), MAX_OPTIONS); scalar_scores = _predict_utility(scalar_model, np.concatenate([geometry, scalar_values], axis=-1).reshape(-1, 19), baseline_device).reshape(len(rows), MAX_OPTIONS); map_scores = _predict_utility(map_model, np.concatenate([np.nan_to_num(map_values, nan=0.0), geometry], axis=-1).reshape(-1, 114), baseline_device).reshape(len(rows), MAX_OPTIONS)
    baseline_actions: dict[str, list[int]] = {"Stay": [int(row["current_viewpoint_id"]) for row in rows], "Random": [], "GeometryOnly Utility": _stable_select(geom_scores, option_ids, option_mask, costs), "ScalarVisibility+Geometry": _stable_select(scalar_scores, option_ids, option_mask, costs), "MapFeature-Utility": _stable_select(map_scores, option_ids, option_mask, costs)}
    rng = np.random.default_rng(SEED); baseline_actions["Random"] = [int(rng.choice(_actions(row))) for row in rows]
    gt_scores = np.asarray([option_logp[index, :, int(row["label_id"])] for index, row in enumerate(rows)], dtype=np.float32); baseline_actions["GT-TrueLogP Oracle"] = _stable_select(gt_scores, option_ids, option_mask, costs)
    baseline_metrics = {name: _metric(name, labels, _terminal(option_logp, option_ids, option_mask, actions), rows, actions, map_values, option_ids, option_mask) for name, actions in baseline_actions.items()}
    (output / "matched_baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2) + "\n", encoding="utf-8")
    package_root = runtime / "observation_packages"; package_root.mkdir(parents=True, exist_ok=True); scene_cache: dict[str, dict[str, Any]] = {}; packages: list[dict[str, Path | str]] = []
    for index, row in enumerate(rows):
        package_dir = package_root / str(row["episode_id"]); package_dir.mkdir(parents=True, exist_ok=True); rgb = package_dir / "current_rgb.png"; topdown = package_dir / "topdown_map.png"; board = package_dir / "candidate_board.png"; archive = _row_archive(row); info = _scene_map(data_root, habitat_root, row, runtime, scene_cache)
        floor_y = float(np.asarray(archive["placement_position"])[1])
        manifest_path = package_dir / "package_manifest.json"
        expected_manifest = {"package_version": PACKAGE_VERSION, "scene_id": str(row["scene_id"]), "floor_key": f"{floor_y:.2f}", "record_id": str(row["record_id"]), "current_viewpoint_id": int(row["current_viewpoint_id"])}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        except json.JSONDecodeError:
            manifest = None
        if manifest != expected_manifest:
            for stale in (topdown, board, package_dir / "map_board.png", package_dir / "rgb_map.png"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
        if not rgb.is_file(): _draw_current_rgb(data_root, row, rgb)
        if not topdown.is_file():
            fig, ax = plt.subplots(figsize=(5, 5)); _plot_map(ax, row, archive, info, data_root); fig.tight_layout(); fig.savefig(topdown, dpi=120); plt.close(fig)
        if not board.is_file(): _draw_board(row, archive, info, data_root, board)
        _resize_rgb(board)
        map_board = package_dir / "map_board.png"; rgb_map = package_dir / "rgb_map.png"
        if not map_board.is_file(): _make_composite([topdown, board], map_board)
        if not rgb_map.is_file(): _make_composite([rgb, map_board], rgb_map)
        manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
        packages.append({"rgb": rgb, "map": topdown, "board": board, "map_board": map_board, "rgb_map": rgb_map, "metadata_text": _metadata_text(row, archive), "map_available": bool(info.get("available"))})
        if (index + 1) % 100 == 0: print(f"[packages] {index + 1}/{len(rows)}", flush=True)
    prompt_path = output / "prompt.txt"; prompt_path.write_text(PROMPT + "\n", encoding="utf-8"); (output / "prompt_hash.txt").write_text(sha256(prompt_path) + "\n", encoding="utf-8")
    branches: dict[str, list[dict[str, Any]]] = {}; server = args.server_url; model_name = args.model_name
    if args.cache_only:
        branch = args.cache_branch
        if branch is None:
            raise ValueError("--cache-branch is required with --cache-only")
        start = max(0, int(args.cache_start)); end = min(len(rows), int(args.cache_end))
        if end < start:
            raise ValueError("--cache-end must be >= --cache-start")
        for index in range(start, end):
            row = rows[index]
            response_path = runtime / "responses" / branch / f"{row['episode_id']}.json"
            _infer_branch(branch, row, packages[index], server, model_name, response_path)
            if (index - start + 1) % 50 == 0:
                print(f"[{branch} cache] {index + 1}/{end}", flush=True)
        return
    for branch in ("VLM-RGB", "VLM-Map", "VLM-RGB+Map"):
        records = []
        for index, row in enumerate(rows):
            record = _infer_branch(branch, row, packages[index], server, model_name, runtime / "responses" / branch / f"{row['episode_id']}.json"); records.append(record)
            if (index + 1) % 50 == 0: print(f"[{branch}] {index + 1}/{len(rows)}", flush=True)
        branches[branch] = records
    branch_metrics: dict[str, dict[str, Any]] = {}; navcost_records: list[dict[str, Any]] = []
    for branch, records in branches.items():
        actions = [_id_to_action(row, record["selected"]) for row, record in zip(rows, records)]
        branch_metrics[branch] = _metric(branch, labels, _terminal(option_logp, option_ids, option_mask, actions), rows, actions, map_values, option_ids, option_mask)
        branch_metrics[branch]["parse_failure_rate"] = float(np.mean([int(x["parse_failure"]) for x in records])); branch_metrics[branch]["repair_rate"] = float(np.mean([int(x["repair"]) for x in records])); branch_metrics[branch]["fallback_rate"] = float(np.mean([int(x["fallback"]) for x in records]))
        branch_metrics[branch]["inference_success_rate"] = float(1.0 - branch_metrics[branch]["fallback_rate"])
        if branch == "VLM-RGB+Map":
            for index, record in enumerate(records):
                scores = np.full(MAX_OPTIONS, -np.inf, dtype=np.float32)
                ranking = [item for item in record["parsed"].get("ranking", []) if isinstance(item, str)]
                ranking.extend(item for item in ["STAY"] + [f"C{i}" for i in range(1, len(rows[index]["candidate_ids"]) + 1)] if item not in ranking)
                for rank, item in enumerate(ranking):
                    slot = 0 if item == "STAY" else int(item[1:])
                    scores[slot] = float(len(ranking) - rank)
                norm_cost = costs[index] / max(float(np.max(costs[index])), 1e-6); scores -= 0.25 * norm_cost; selected_slot = int(np.argmax(scores)); navcost_records.append({"selected": int(option_ids[index, selected_slot]), "score": float(scores[selected_slot])})
            nav_actions = [int(record["selected"]) for record in navcost_records]; branch_metrics["VLM-RGB+Map+NavCost"] = _metric("VLM-RGB+Map+NavCost", labels, _terminal(option_logp, option_ids, option_mask, nav_actions), rows, nav_actions, map_values, option_ids, option_mask)
    ranking_metrics = {branch: _ranking_diagnostics(branch, records, rows, option_logp) for branch, records in branches.items()}
    stay_visibility = np.nanmean(map_values[:, 0, :17], axis=1); high = stay_visibility <= np.quantile(stay_visibility, 1 / 3); clutter = map_values[:, 0, SCENE_CLEARANCE_INDEX]; edges = np.nanquantile(clutter, [1 / 3, 2 / 3]); clutter_masks = {"cluttered": clutter <= edges[0], "medium": (clutter > edges[0]) & (clutter <= edges[1]), "open": clutter > edges[1]}
    subset_methods = {**baseline_actions, **{branch: [_id_to_action(row, record["selected"]) for row, record in zip(rows, records)] for branch, records in branches.items()}}
    occlusion_metrics = {"bottom_tertial_count": int(high.sum()), "methods": {name: _metric(name, labels[high], _terminal(option_logp[high], option_ids[high], option_mask[high], [action for action, flag in zip(actions, high) if flag]), [row for row, flag in zip(rows, high) if flag], [action for action, flag in zip(actions, high) if flag]) for name, actions in subset_methods.items()}}
    scene_clutter_metrics = {level: {name: _metric(name, labels[mask], _terminal(option_logp[mask], option_ids[mask], option_mask[mask], [action for action, flag in zip(actions, mask) if flag]), [row for row, flag in zip(rows, mask) if flag], [action for action, flag in zip(actions, mask) if flag]) for name, actions in subset_methods.items()} for level, mask in clutter_masks.items()}
    parse_stats = {branch: {key: float(np.mean([int(item[key]) for item in records])) for key in ("parse_failure", "repair", "fallback")} for branch, records in branches.items()}
    failure_manifest: list[dict[str, Any]] = []; best_branch = max(branch_metrics, key=lambda name: float(branch_metrics[name]["accuracy"])); scalar_preds = _terminal(option_logp, option_ids, option_mask, baseline_actions["ScalarVisibility+Geometry"]); best_actions = subset_methods[best_branch]; best_preds = _terminal(option_logp, option_ids, option_mask, best_actions); categories = {"vlm_correct_scalar_wrong": [], "vlm_wrong_scalar_correct": [], "both_correct": [], "both_wrong": []}
    for index, (vp, sp) in enumerate(zip(best_preds == labels, scalar_preds == labels)):
        category = "both_correct" if vp and sp else "both_wrong" if not vp and not sp else "vlm_correct_scalar_wrong" if vp else "vlm_wrong_scalar_correct"
        if len(categories[category]) < 5:
            failure_manifest.append({"category": category, "episode_id": rows[index]["episode_id"], "package_dir": str((package_root / str(rows[index]["episode_id"])).resolve()), "vlm_selected": int(best_actions[index]), "scalar_selected": int(baseline_actions["ScalarVisibility+Geometry"][index]), "vlm_correct": bool(vp), "scalar_correct": bool(sp)})
            categories[category].append(index)
    map_available_rate = float(np.mean([int(bool(package.get("map_available", False))) for package in packages]))
    status = "FAILED" if any(float(branch_metrics[name].get("inference_success_rate", 0.0)) < 0.50 for name in ("VLM-RGB", "VLM-Map", "VLM-RGB+Map")) else "COMPLETED"
    latencies = [item["latency_seconds"] for records in branches.values() for item in records]
    result = {"status": status, "experiment": "Zero-shot VLM NBV Viability Audit", "subset_contexts": len(rows), "train_contexts_available": len(train_rows), "moving_val_contexts_available": len(val_rows), "matched_baselines": {name: _subset_metrics(name, metric) for name, metric in baseline_metrics.items()}, "vlm_branches": {name: _subset_metrics(name, metric) for name, metric in branch_metrics.items()}, "ranking_metrics": ranking_metrics, "occlusion_metrics": occlusion_metrics, "scene_clutter_metrics": scene_clutter_metrics, "parse_stats": parse_stats, "failure_gallery_manifest": failure_manifest, "map_available_rate": map_available_rate, "flags": {"policy_test_used": False, "training_used": False, "vlm_fine_tuning": False, "new_rgb_generated": False, "new_skeleton_generated": False, "candidate_real_rgb_used": False, "candidate_real_skeleton_used_before_selection": False, "candidate_recognizer_output_used_before_selection": False, "current_exact_pose_privileged": True, "deployable": False}, "model": {"name": model_name, "path": args.vlm_model, "mmproj": args.mmproj, "temperature": 0, "seed": SEED, "enable_thinking": False, "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(device), "checkpoint_sha256": sha256(Path(args.vlm_model)) if Path(args.vlm_model).is_file() else None, "mmproj_sha256": sha256(Path(args.mmproj)) if Path(args.mmproj).is_file() else None}, "inference_runtime": {"mean_seconds": float(np.mean(latencies)), "median_seconds": float(np.median(latencies)), "p90_seconds": float(np.percentile(latencies, 90)), "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_gpu_memory_scope": "client_torch_process_only; VLM server runs as an external GPU-offloaded process", "responses": int(sum(len(records) for records in branches.values())), "response_compute_seconds": float(np.sum(latencies)), "wallclock_seconds_current_invocation": time.time() - started}, "test_used": False}
    (output / "vlm_branch_metrics.json").write_text(json.dumps(result["vlm_branches"], indent=2) + "\n", encoding="utf-8"); (output / "ranking_metrics.json").write_text(json.dumps(ranking_metrics, indent=2) + "\n", encoding="utf-8"); (output / "occlusion_metrics.json").write_text(json.dumps(occlusion_metrics, indent=2) + "\n", encoding="utf-8"); (output / "scene_clutter_metrics.json").write_text(json.dumps(scene_clutter_metrics, indent=2) + "\n", encoding="utf-8"); (output / "navigation_metrics.json").write_text(json.dumps({name: {"move_rate": metric.get("move_rate", 0.0), "mean_nav_distance_m": metric.get("mean_nav_distance_m", 0.0)} for name, metric in {**baseline_metrics, **branch_metrics}.items()}, indent=2) + "\n", encoding="utf-8"); (output / "parse_stats.json").write_text(json.dumps(parse_stats, indent=2) + "\n", encoding="utf-8"); (output / "failure_gallery_manifest.json").write_text(json.dumps(failure_manifest, indent=2) + "\n", encoding="utf-8"); (output / "leakage_audit.json").write_text(json.dumps({"uses_current_rgb": {"VLM-RGB": True, "VLM-Map": False, "VLM-RGB+Map": True}, "uses_topdown_map": {"VLM-RGB": False, "VLM-Map": True, "VLM-RGB+Map": True}, "uses_candidate_static_render": {"VLM-RGB": False, "VLM-Map": True, "VLM-RGB+Map": True}, "uses_current_pose": True, "uses_gt_action": False, "uses_future_human_motion": False, "uses_candidate_real_rgb": False, "uses_candidate_real_skeleton": False, "uses_candidate_recognizer_output": False, "test_used": False}, indent=2) + "\n", encoding="utf-8"); (output / "inference_runtime.json").write_text(json.dumps(result["inference_runtime"], indent=2) + "\n", encoding="utf-8"); (output / "config.json").write_text(json.dumps({"seed": SEED, "subset_size": SUBSET_SIZE, "server_url": server, "model": model_name, "branches": ["VLM-RGB", "VLM-Map", "VLM-RGB+Map"], "nav_cost_lambda": 0.25, "test_used": False}, indent=2) + "\n", encoding="utf-8"); (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); _write_analysis(result, output / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda:0", help="GPU reserved for the local VLM runtime"); parser.add_argument("--baseline-device", default="cpu", help="device for tiny frozen-head/baseline models (CPU avoids contention with a 35B VLM)"); parser.add_argument("--server-url", default=VLM_DEFAULT_URL); parser.add_argument("--model-name", default=VLM_DEFAULT_MODEL); parser.add_argument("--vlm-model", default=VLM_DEFAULT_PATH); parser.add_argument("--mmproj", default=VLM_DEFAULT_MMPROJ); parser.add_argument("--output-root", default=None); parser.add_argument("--runtime-root", default=None); parser.add_argument("--cache-only", action="store_true", help="populate response cache for a bounded subset without writing final metrics"); parser.add_argument("--cache-branch", choices=["VLM-RGB", "VLM-Map", "VLM-RGB+Map"], default=None); parser.add_argument("--cache-start", type=int, default=0); parser.add_argument("--cache-end", type=int, default=SUBSET_SIZE); args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
