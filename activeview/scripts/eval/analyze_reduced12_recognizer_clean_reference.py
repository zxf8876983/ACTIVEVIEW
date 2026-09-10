#!/usr/bin/env python3
"""Recognizer-level clean-motion reference audit for reduced12 (Val only).

The clean reference is reconstructed from the exact AMASS interval carried by
each raw-val record. It is converted with the project's MotionConverter,
forward-kinematics are read from the Habitat articulated humanoid, and the
result is passed through the same frozen ST-GCN used by archived views.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_humanoid_urdf_path
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.motion.babel_clean_dataset_generator import _load_resampled_motion
from activeview.data.motion.habitat_h36m17_fk import (
    H36M17_LINKS,
    extract_h36m17_from_humanoid,
)
from activeview.data.motion.motion_converter import MotionConverter
from activeview.perception.normalization import SkeletonNormalizer
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced12_scene_joint_visibility_oracle import (
    FRAME_IDS as LEGACY_FRAME_IDS,
    _world_h36m17,
)


NUM_CLASSES = 12
FRAME_COUNT = 30
VIEW_COUNT = 32
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
DATASET_NAME = "reduced12_no_kneel_clean_babel_diversity_v1"
POLICY_NAME = "policy_reduced12_eight_placement_v1"
ARCHIVE_REL = "datasets/offline/habitat-train/00006-00087"
TRUE_CACHE_REL = "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"
SCENE_VIS_PATH = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz"
HUMAN_VIS_PATH = REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_view_observability_oracle/candidate_metrics.npz"
QUALITATIVE_MANIFEST = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization/case_manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/recognizer_clean_reference_audit"
DEFAULT_CHECKPOINT_REL = "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth"
EPS = 1.0e-10

def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in load_jsonl(path)]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _rank(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    sorted_values = values_array[order]
    ranks = np.empty(values_array.size, dtype=np.float64)
    start = 0
    while start < values_array.size:
        end = start + 1
        while end < values_array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _corr(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.size != y.size or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0, 0.0
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = _rank(x), _rank(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > EPS and np.std(ry) > EPS else 0.0
    return pearson, spearman


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
    }


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(labels, dtype=np.int64)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for truth, guess in zip(target.tolist(), pred.tolist()):
        if 0 <= int(truth) < NUM_CLASSES and 0 <= int(guess) < NUM_CLASSES:
            matrix[int(truth), int(guess)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(LABELS):
        tp = float(matrix[class_id, class_id])
        support = float(matrix[class_id].sum())
        predicted = float(matrix[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"label_id": class_id, "count": int(support), "recall": recall, "precision": precision, "f1": f1}
        f1_values.append(f1)
    return {
        "count": int(target.size), "accuracy": float(np.mean(pred == target)) if target.size else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class, "confusion_matrix": matrix.tolist(),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(np.clip(shifted, -80.0, 80.0))
    return values / np.maximum(values.sum(axis=-1, keepdims=True), EPS)


def _jsd(left_logp: np.ndarray, right_logp: np.ndarray) -> float:
    left, right = _softmax(left_logp), _softmax(right_logp)
    mean = 0.5 * (left + right)
    return float(0.5 * (np.sum(left * (np.log(np.maximum(left, EPS)) - np.log(np.maximum(mean, EPS)))) + np.sum(right * (np.log(np.maximum(right, EPS)) - np.log(np.maximum(mean, EPS))))))


def _kl(left_logp: np.ndarray, right_logp: np.ndarray) -> float:
    left, right = _softmax(left_logp), _softmax(right_logp)
    return float(np.sum(left * (np.log(np.maximum(left, EPS)) - np.log(np.maximum(right, EPS)))))


def _scene_sim_none() -> tuple[Any, Any]:
    """Create the same articulated humanoid used by the formal converter path."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = "NONE"
    backend.enable_physics = True
    agent = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    human = sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(get_humanoid_urdf_path("male_0")))
    return sim, human


def _placement(data_root: Path, scene_id: str, placement_id: str) -> Mapping[str, Any]:
    path = data_root / ARCHIVE_REL / "placement_sampling_v2" / scene_id / "placements.json"
    payload = _read_json(path)
    for item in payload.get("placements", []):
        if str(item.get("placement_id")) == str(placement_id):
            return item
    raise KeyError(f"placement {placement_id} missing from {path}")


def _archive_path(data_root: Path, row: Mapping[str, Any]) -> Path:
    path = data_root / ARCHIVE_REL / str(row["scene_id"]) / str(row.get("region")) / f"{row['record_id']}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _metadata_views(data_root: Path, scene_id: str, placement_id: str) -> dict[int, Mapping[str, Any]]:
    path = data_root / ARCHIVE_REL / str(scene_id) / "candidate_metadata" / "manifest.json"
    payload = _read_json(path)
    for item in payload.get("placements_data", []):
        if str(item.get("placement_id")) == str(placement_id):
            views = {int(view["viewpoint_id"]): view for view in item.get("viewpoints", [])}
            if len(views) != VIEW_COUNT:
                raise ValueError(f"expected 32 viewpoints in {path} / {placement_id}")
            return views
    raise KeyError(f"placement {placement_id} missing from {path}")


def _motion_amplitude(skeleton: np.ndarray) -> np.ndarray:
    value = np.asarray(skeleton, dtype=np.float64)
    if value.shape == (3, FRAME_COUNT, 17):
        value = np.transpose(value, (1, 2, 0))
    if value.shape != (FRAME_COUNT, 17, 3):
        raise ValueError(f"unexpected skeleton shape {value.shape}")
    return np.linalg.norm(np.diff(value, axis=0), axis=2).mean(axis=0)


def _validate_fk(data_root: Path, rows: Sequence[Mapping[str, Any]], raw_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the generic helper to the legacy six-frame implementation."""
    selected: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["record_id"]), str(row["scene_id"]), str(row["region"]))
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        raise RuntimeError("fewer than three Val contexts are available for FK validation")
    sim, human = _scene_sim_none()
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    records: list[dict[str, Any]] = []
    try:
        for row in selected:
            record_id = str(row["record_id"])
            placement = _placement(data_root, str(row["scene_id"]), str(row["region"]))
            converted = converter.convert(_load_resampled_motion(raw_records[record_id], FRAME_COUNT))
            legacy = _world_h36m17(human, converted, placement, float(placement["yaw_deg"]))
            generic = extract_h36m17_from_humanoid(human, converted, placement, float(placement["yaw_deg"]), LEGACY_FRAME_IDS)
            delta = float(np.max(np.abs(legacy - generic)))
            pelvis = generic[:, 0]
            hip_midpoint = 0.5 * (generic[:, 1] + generic[:, 4])
            distances = np.linalg.norm(pelvis - hip_midpoint, axis=1)
            records.append({"record_id": record_id, "scene_id": str(row["scene_id"]), "placement_id": str(row["region"]), "max_abs_difference": delta, "pelvis_hip_midpoint_distance": _summary(distances.tolist()), "root_translation_finite": bool(np.isfinite(pelvis).all())})
    finally:
        sim.close()
    passed = all(item["max_abs_difference"] <= 1.0e-5 and item["root_translation_finite"] for item in records)
    return {"status": "PASS" if passed else "BLOCKED", "frame_ids": list(LEGACY_FRAME_IDS), "joint_order": list(H36M17_LINKS), "pelvis_semantics": "human.translation, matching legacy renderer convention", "records": records, "max_abs_difference": float(max(item["max_abs_difference"] for item in records)), "tolerance": 1.0e-5, "clean_fk_valid": bool(passed)}


def _build_clean_reference(data_root: Path, raw_records: Mapping[str, Mapping[str, Any]], record_ids: Sequence[str], cache_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate one exact, camera-independent normalized skeleton per record."""
    sim, human = _scene_sim_none()
    converter = MotionConverter(get_humanoid_urdf_path("male_0"))
    normalizer = SkeletonNormalizer()
    skeletons: list[np.ndarray] = []
    labels: list[int] = []
    started = time.perf_counter()
    try:
        for index, record_id in enumerate(record_ids, 1):
            record = raw_records[record_id]
            converted = converter.convert(_load_resampled_motion(record, FRAME_COUNT))
            world = extract_h36m17_from_humanoid(human, converted, {"position": [0.0, 0.0, 0.0]}, 0.0, range(FRAME_COUNT))
            normalized = normalizer.normalize_sequence(world, align_canonical=True)
            tensor = np.transpose(normalized, (2, 0, 1))[:, :, :, None].astype(np.float32)
            if tensor.shape != (3, FRAME_COUNT, 17, 1) or not np.isfinite(tensor).all():
                raise ValueError(f"invalid clean tensor for {record_id}: {tensor.shape}")
            skeletons.append(tensor)
            labels.append(int(record["label_id"]))
            if index % 10 == 0 or index == len(record_ids):
                print(f"clean-fk: {index}/{len(record_ids)} records", flush=True)
    finally:
        sim.close()
    array = np.stack(skeletons, axis=0)
    payload = {"record_ids": np.asarray(record_ids), "skeletons": array, "labels": np.asarray(labels, dtype=np.int64)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload, {"records": len(record_ids), "shape": list(array.shape), "elapsed_seconds": time.perf_counter() - started, "source": "exact raw-val AMASS interval -> MotionConverter -> Habitat humanoid FK"}


def _artifact_map(path: Path, score_key: str) -> dict[str, dict[int, float]]:
    if not path.is_file():
        return {}
    arrays = _load_npz(path)
    required = {"episode_ids", "candidate_ids", "candidate_mask", score_key}
    if not required.issubset(arrays):
        return {}
    output: dict[str, dict[int, float]] = {}
    for index, episode in enumerate(arrays["episode_ids"].tolist()):
        mask = np.asarray(arrays["candidate_mask"][index], dtype=bool)
        ids = np.asarray(arrays["candidate_ids"][index], dtype=np.int64)[mask]
        values = np.asarray(arrays[score_key][index], dtype=np.float64)[mask]
        output[str(episode)] = {int(candidate): float(value) for candidate, value in zip(ids.tolist(), values.tolist())}
    return output


def _infer(model: Any, arrays: Sequence[np.ndarray], device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            batch = np.stack(arrays[start:start + batch_size]).astype(np.float32, copy=False)
            tensor = torch.from_numpy(batch).to(device)
            feature = model.forward_features(tensor).detach().cpu().numpy()
            logit = model.fc(torch.from_numpy(feature).to(device)).detach().cpu().numpy()
            features.append(feature.astype(np.float32))
            logits.append(logit.astype(np.float32))
            if start == 0 or (start + batch_size) % (batch_size * 10) == 0:
                print(f"stgcn: {min(start + batch_size, len(arrays))}/{len(arrays)}", flush=True)
    feature_array = np.concatenate(features, axis=0) if features else np.empty((0, 256), dtype=np.float32)
    logit_array = np.concatenate(logits, axis=0) if logits else np.empty((0, NUM_CLASSES), dtype=np.float32)
    logp_array = logit_array - np.log(np.maximum(np.exp(logit_array - np.max(logit_array, axis=-1, keepdims=True)).sum(axis=-1, keepdims=True), EPS)) - np.max(logit_array, axis=-1, keepdims=True)
    return feature_array, logp_array.astype(np.float32), _softmax(logit_array)


def _load_candidate_contexts(
    data_root: Path,
    moving_rows: Sequence[Mapping[str, Any]],
    stage_c_rows: Sequence[Mapping[str, Any]],
    true_cache: Mapping[str, np.ndarray],
    model: Any,
    device: torch.device,
    batch_size: int,
    scene_map: Mapping[str, Mapping[int, float]],
    human_map: Mapping[str, Mapping[int, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage_index = {str(row["episode_id"]): index for index, row in enumerate(stage_c_rows)}
    if len(stage_index) != len(stage_c_rows):
        raise ValueError("duplicate Stage-C episode_id")
    views_cache: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    contexts: list[dict[str, Any]] = []
    candidate_arrays: list[np.ndarray] = []
    candidate_refs: list[dict[str, Any]] = []
    cache_diff: list[float] = []
    for row in moving_rows:
        episode = str(row["episode_id"])
        if episode not in stage_index:
            raise ValueError(f"Stage-D episode missing from Stage-C: {episode}")
        stage_i = stage_index[episode]
        stage_row = stage_c_rows[stage_i]
        stage_ids = np.asarray(stage_row["candidate_viewpoint_ids"], dtype=np.int64)
        cache_ids = np.asarray(true_cache["candidate_ids"][stage_i], dtype=np.int64)
        mask = np.asarray(true_cache["candidate_mask"][stage_i], dtype=bool)
        cache_ids = cache_ids[mask]
        if set(stage_ids.tolist()) != set(cache_ids.tolist()):
            raise ValueError(f"candidate identity mismatch for {episode}")
        label = int(row["label_id"])
        if int(true_cache["labels"][stage_i]) != label:
            raise ValueError(f"true-cache label mismatch for {episode}")
        archive_path = _archive_path(data_root, row)
        with np.load(archive_path, allow_pickle=False) as archive:
            skeletons = np.asarray(archive["skeleton"], dtype=np.float32)
            archive_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            confidence = np.asarray(archive["confidence"], dtype=np.float64)
            camera_positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float64)
            placement_position = np.asarray(archive["placement_position"], dtype=np.float64)
        if skeletons.shape != (VIEW_COUNT, 3, FRAME_COUNT, 17):
            raise ValueError(f"invalid archive skeleton shape {archive_path}: {skeletons.shape}")
        archive_index = {int(value): index for index, value in enumerate(archive_ids.tolist())}
        placement_id = str(row["region"])
        view_key = (str(row["scene_id"]), placement_id)
        if view_key not in views_cache:
            views_cache[view_key] = _metadata_views(data_root, *view_key)
        views = views_cache[view_key]
        scene_values = scene_map.get(episode, {})
        human_values = human_map.get(episode, {})
        candidates: list[dict[str, Any]] = []
        for candidate_id in stage_ids.tolist():
            candidate_id = int(candidate_id)
            if candidate_id not in archive_index or candidate_id not in views:
                raise ValueError(f"archive/metadata viewpoint missing for {episode}/{candidate_id}")
            archive_i = archive_index[candidate_id]
            view = views[candidate_id]
            candidate = {
                "candidate_id": candidate_id,
                "radius": float(view.get("radius_m", float("nan"))),
                "azimuth": float(view.get("azimuth_deg", float("nan"))),
                "scene_visibility": float(scene_values.get(candidate_id, float("nan"))),
                "human_observability": float(human_values.get(candidate_id, float("nan"))),
                "pose_confidence": float(confidence[archive_i]),
                "distance_m": float(np.linalg.norm(camera_positions[archive_i] - placement_position)),
                "motion_amp": _motion_amplitude(skeletons[archive_i]),
            }
            candidates.append(candidate)
            candidate_arrays.append(skeletons[archive_i])
            candidate_refs.append(candidate)
        s0_feature = np.asarray(row["s0_feature"], dtype=np.float32)
        s1_feature = np.asarray(row["s1_feature"], dtype=np.float32)
        if s0_feature.size < 256 + NUM_CLASSES or s1_feature.size < 256 + NUM_CLASSES:
            raise ValueError(f"Stage-D feature is shorter than 256+{NUM_CLASSES}: {episode}")
        contexts.append({
            "episode_id": episode, "record_id": str(row["record_id"]),
            "scene_id": str(row["scene_id"]), "placement_id": placement_id,
            "label": label, "s0_feature": s0_feature[:256],
            "s0_logp": s0_feature[256:256 + NUM_CLASSES],
            "s1_feature": s1_feature[:256],
            "s1_logp": s1_feature[256:256 + NUM_CLASSES],
            "frozen_candidate_id": int(row["proposal_rank_1_id"]),
            "candidates": candidates,
        })
    candidate_features, candidate_logits, candidate_probs = _infer(model, candidate_arrays, device, batch_size)
    cache_logits = np.asarray(true_cache["candidate_logp"])
    offset = 0
    for context in contexts:
        for candidate in context["candidates"]:
            candidate["feature"] = candidate_features[offset]
            candidate["logp"] = candidate_logits[offset]
            candidate["prob"] = candidate_probs[offset]
            candidate["prediction"] = int(np.argmax(candidate_logits[offset]))
            candidate["correct"] = bool(candidate["prediction"] == int(context["label"]))
            stage_i = stage_index[context["episode_id"]]
            candidate_id = int(candidate["candidate_id"])
            cache_slots = np.flatnonzero(np.asarray(true_cache["candidate_ids"][stage_i]) == candidate_id)
            if cache_slots.size == 1:
                cache_diff.append(float(np.max(np.abs(cache_logits[stage_i, cache_slots[0]] - candidate_logits[offset]))))
            offset += 1
        amplitudes = np.stack([candidate["motion_amp"] for candidate in context["candidates"]])
        reference = np.median(amplitudes, axis=0)
        for candidate in context["candidates"]:
            candidate["motion_deviation"] = float(np.mean(np.abs(candidate["motion_amp"] - reference) / (reference + EPS)))
    return contexts, {"candidate_count": len(candidate_refs), "candidate_cache_logp_max_abs_diff": float(max(cache_diff) if cache_diff else 0.0), "candidate_cache_logp_mean_abs_diff": float(np.mean(cache_diff) if cache_diff else 0.0)}


def _attach_similarity(contexts: Sequence[dict[str, Any]], clean_features: Mapping[str, np.ndarray], clean_logp: Mapping[str, np.ndarray], clean_predictions: Mapping[str, int]) -> None:
    for context in contexts:
        record_id = str(context["record_id"])
        reference_feature = np.asarray(clean_features[record_id], dtype=np.float64)
        reference_logp = np.asarray(clean_logp[record_id], dtype=np.float64)
        reference_norm = float(np.linalg.norm(reference_feature))
        context["clean_prediction"] = int(clean_predictions[record_id])
        for candidate in context["candidates"]:
            feature = np.asarray(candidate["feature"], dtype=np.float64)
            cosine = float(np.dot(feature, reference_feature) / max(np.linalg.norm(feature) * reference_norm, EPS))
            candidate["feature_cos_clean"] = cosine
            candidate["feature_l2_clean"] = float(np.linalg.norm(feature - reference_feature) / max(reference_norm, EPS))
            candidate["posterior_jsd_clean"] = _jsd(np.asarray(candidate["logp"]), reference_logp)
            candidate["posterior_kl_clean"] = _kl(reference_logp, np.asarray(candidate["logp"]))
            label = int(context["label"])
            candidate["gt_logp"] = float(np.asarray(candidate["logp"])[label])
            candidate["gt_margin"] = float(candidate["gt_logp"] - np.max(np.delete(np.asarray(candidate["logp"]), label)))

def _evaluate_contexts(contexts: Sequence[Mapping[str, Any]], clean_predictions: Mapping[str, int], seed: int = 42) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    names = ("S0-only", "FrozenStageCv0", "Candidate-Conditioned Spatial", "Real-GTMargin Oracle", "AnyCorrect Oracle", "MaxFeatureCos-to-Clean", "MinJSD-to-Clean")
    labels: list[int] = []
    predictions = {name: [] for name in names}
    selected_ids = {name: [] for name in names}
    moved = {name: [] for name in names}
    flat: list[dict[str, Any]] = []
    quadrants: list[dict[str, Any]] = []
    for context in contexts:
        label = int(context["label"])
        labels.append(label)
        candidates = list(context["candidates"])
        s0_prediction = int(np.argmax(np.asarray(context["s0_logp"])))
        s1_prediction = int(np.argmax(np.asarray(context["s1_logp"])))
        by_id = {int(candidate["candidate_id"]): candidate for candidate in candidates}
        frozen = by_id.get(int(context["frozen_candidate_id"]), candidates[0])
        gt_oracle = max(candidates, key=lambda item: (float(item["gt_margin"]), -int(item["candidate_id"])))
        max_cos = max(candidates, key=lambda item: (float(item["feature_cos_clean"]), -int(item["candidate_id"])))
        min_jsd = min(candidates, key=lambda item: (float(item["posterior_jsd_clean"]), int(item["candidate_id"])))
        predictions["S0-only"].append(s0_prediction)
        predictions["FrozenStageCv0"].append(s1_prediction)
        predictions["Candidate-Conditioned Spatial"].append(int(frozen["prediction"]))
        predictions["Real-GTMargin Oracle"].append(int(gt_oracle["prediction"]))
        predictions["MaxFeatureCos-to-Clean"].append(int(max_cos["prediction"]))
        predictions["MinJSD-to-Clean"].append(int(min_jsd["prediction"]))
        any_correct = s0_prediction == label or any(bool(candidate["correct"]) for candidate in candidates)
        predictions["AnyCorrect Oracle"].append(label if any_correct else s0_prediction)
        selected_ids["FrozenStageCv0"].append(int(frozen["candidate_id"]))
        selected_ids["Candidate-Conditioned Spatial"].append(int(frozen["candidate_id"]))
        selected_ids["Real-GTMargin Oracle"].append(int(gt_oracle["candidate_id"]))
        selected_ids["MaxFeatureCos-to-Clean"].append(int(max_cos["candidate_id"]))
        selected_ids["MinJSD-to-Clean"].append(int(min_jsd["candidate_id"]))
        for name in names:
            moved[name].append(name not in {"S0-only", "AnyCorrect Oracle"})
        for candidate in candidates:
            flat.append({"episode_id": context["episode_id"], "record_id": context["record_id"], "label": label, **candidate})
        quadrants.append({
            "episode_id": context["episode_id"], "record_id": context["record_id"], "label": label,
            "clean_prediction": int(context["clean_prediction"]), "oracle_prediction": int(gt_oracle["prediction"]),
            "clean_correct": bool(int(context["clean_prediction"]) == label), "oracle_correct": bool(gt_oracle["correct"]),
            "oracle_candidate_id": int(gt_oracle["candidate_id"]), "oracle_gt_margin": float(gt_oracle["gt_margin"]),
        })
    label_array = np.asarray(labels, dtype=np.int64)
    methods: dict[str, Any] = {}
    for name in names:
        methods[name] = _metrics(predictions[name], label_array)
        methods[name]["move_rate"] = float(np.mean(np.asarray(moved[name], dtype=np.float64))) if moved[name] else 0.0
        methods[name]["selected_candidate_ids"] = selected_ids[name]
    by_method = {name: methods[name]["per_class"] for name in names}
    frozen = methods["FrozenStageCv0"]
    utilities = np.asarray([float(item["gt_logp"]) for item in flat], dtype=np.float64)
    rankings: dict[str, Any] = {}
    score_keys = {
        "FeatureCos": "feature_cos_clean", "NegativeFeatureL2": "feature_l2_clean",
        "NegativePosteriorJSD": "posterior_jsd_clean", "SceneVisibility": "scene_visibility",
        "ProjectedArea": "projected_area", "PoseConfidence": "pose_confidence",
        "NegativeDistance": "distance_m", "NegativeMotionDeviation": "motion_deviation",
    }
    for name, key in score_keys.items():
        values = np.asarray([float(item.get(key, float("nan"))) for item in flat], dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(utilities)
        pearson, spearman = _corr(values[valid], utilities[valid])
        context_values: list[float] = []
        context_utilities: list[float] = []
        for context in contexts:
            candidate_values = [float(item.get(key, float("nan"))) for item in context["candidates"]]
            candidate_utility = [float(item["gt_logp"]) for item in context["candidates"]]
            p, s = _corr(candidate_values, candidate_utility)
            if len(candidate_values) > 1:
                context_values.append(p)
                context_utilities.append(s)
        rankings[name] = {"count": int(valid.sum()), "pearson_global": pearson, "spearman_global": spearman, "mean_within_context_pearson": float(np.mean(context_values)) if context_values else 0.0, "mean_within_context_spearman": float(np.mean(context_utilities)) if context_utilities else 0.0}
    similarity = {
        "candidate_count": len(flat),
        "feature_cos_clean": _summary([float(item["feature_cos_clean"]) for item in flat]),
        "normalized_feature_l2_clean": _summary([float(item["feature_l2_clean"]) for item in flat]),
        "posterior_jsd_clean": _summary([float(item["posterior_jsd_clean"]) for item in flat]),
        "posterior_kl_clean_to_candidate": _summary([float(item["posterior_kl_clean"]) for item in flat]),
        "correct_vs_wrong": {
            "feature_cos_clean_correct": _summary([float(item["feature_cos_clean"]) for item in flat if item["correct"]]),
            "feature_cos_clean_wrong": _summary([float(item["feature_cos_clean"]) for item in flat if not item["correct"]]),
            "posterior_jsd_clean_correct": _summary([float(item["posterior_jsd_clean"]) for item in flat if item["correct"]]),
            "posterior_jsd_clean_wrong": _summary([float(item["posterior_jsd_clean"]) for item in flat if not item["correct"]]),
        },
    }
    clean_correct_mask = np.asarray([bool(item["clean_correct"]) for item in quadrants])
    oracle_correct_mask = np.asarray([bool(item["oracle_correct"]) for item in quadrants])
    quadrant_metrics = {
        "counts": {"clean_correct_oracle_correct": int(np.sum(clean_correct_mask & oracle_correct_mask)), "clean_wrong_oracle_correct": int(np.sum(~clean_correct_mask & oracle_correct_mask)), "clean_correct_oracle_wrong": int(np.sum(clean_correct_mask & ~oracle_correct_mask)), "clean_wrong_oracle_wrong": int(np.sum(~clean_correct_mask & ~oracle_correct_mask))},
        "rates": {"clean_correct": float(np.mean(clean_correct_mask)), "oracle_correct": float(np.mean(oracle_correct_mask))},
    }
    wrong_to_correct = ~clean_correct_mask & oracle_correct_mask
    correct_to_wrong = clean_correct_mask & ~oracle_correct_mask
    oracle_by_id = {str(item["episode_id"]): item for item in quadrants}
    quadrant_metrics["cleanwrong_oraclecorrect"] = {"count": int(wrong_to_correct.sum())}
    quadrant_metrics["cleancorrect_oraclewrong"] = {"count": int(correct_to_wrong.sum())}
    selector_audit = {"candidate_rows": len(flat)}
    regrets: list[float] = []
    selected_gt_margins: list[float] = []
    severe_records: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        candidates = list(context["candidates"])
        gt_best = max(float(item["gt_margin"]) for item in candidates)
        frozen = next(item for item in candidates if int(item["candidate_id"]) == int(selected_ids["FrozenStageCv0"][index]))
        regrets.append(gt_best - float(frozen["gt_margin"]))
        selected_gt_margins.append(float(frozen["gt_margin"]))
        if oracle_correct_mask[index] and not bool(frozen["correct"]):
            severe_records.append({"episode_id": context["episode_id"], "regret": float(gt_best - float(frozen["gt_margin"])), "selected_candidate_id": int(frozen["candidate_id"]), "oracle_candidate_id": int(max(candidates, key=lambda item: float(item["gt_margin"]))["candidate_id"])})
    regret_cut = float(np.median([item["regret"] for item in severe_records])) if severe_records else 0.0
    selector_audit.update({"selected_gt_margin": _summary(selected_gt_margins), "regret_to_gt_margin_oracle": _summary(regrets), "frozen_wrong_oracle_correct_contexts": len(severe_records), "severe_miss_definition": "Frozen wrong while GT-margin oracle correct; severe is regret at or above median of that subset", "severe_miss_contexts": int(sum(float(item["regret"]) >= regret_cut for item in severe_records)), "severe_miss_regret_median": regret_cut})
    selector_audit["cleanwrong_oraclecorrect_deep_dive"] = {
        "count": int(wrong_to_correct.sum()),
        "mean_oracle_feature_cos": float(np.mean([float(max(contexts[i]["candidates"], key=lambda item: item["gt_margin"])["feature_cos_clean"]) for i, keep in enumerate(wrong_to_correct) if keep])) if wrong_to_correct.any() else 0.0,
        "mean_oracle_jsd": float(np.mean([float(max(contexts[i]["candidates"], key=lambda item: item["gt_margin"])["posterior_jsd_clean"]) for i, keep in enumerate(wrong_to_correct) if keep])) if wrong_to_correct.any() else 0.0,
        "small_change_flip_rate_feature_cos_ge_0.95": float(np.mean([float(max(contexts[i]["candidates"], key=lambda item: item["gt_margin"])["feature_cos_clean"]) >= 0.95 for i, keep in enumerate(wrong_to_correct) if keep])) if wrong_to_correct.any() else 0.0,
    }
    selector_audit["cleancorrect_oraclewrong_count"] = int(correct_to_wrong.sum())
    selector_audit["clean_subset_metrics"] = {
        "clean_recognizer": _metrics([int(item["clean_prediction"]) for item, keep in zip(quadrants, clean_correct_mask) if keep], [int(item["label"]) for item, keep in zip(quadrants, clean_correct_mask) if keep]),
        "frozen_stage_cv0": _metrics([int(predictions["FrozenStageCv0"][i]) for i, keep in enumerate(clean_correct_mask) if keep], [int(label_array[i]) for i, keep in enumerate(clean_correct_mask) if keep]),
        "real_gt_margin_oracle": _metrics([int(predictions["Real-GTMargin Oracle"][i]) for i, keep in enumerate(clean_correct_mask) if keep], [int(label_array[i]) for i, keep in enumerate(clean_correct_mask) if keep]),
    }
    selector_audit["privileged_selectors"] = {name: {"accuracy": methods[name]["accuracy"], "macro_f1": methods[name]["macro_f1"], "per_class": methods[name]["per_class"]} for name in ("MaxFeatureCos-to-Clean", "MinJSD-to-Clean")}
    return methods, {"similarity": similarity, "rankings": rankings, "quadrants": quadrant_metrics, "selector_audit": selector_audit, "flat_candidates": flat, "quadrant_rows": quadrants}, by_method


def _clean_metrics(clean_payload: Mapping[str, np.ndarray], clean_predictions: Mapping[str, int], contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray(clean_payload["labels"], dtype=np.int64)
    predictions = np.asarray([int(clean_predictions[str(record_id)]) for record_id in clean_payload["record_ids"].tolist()], dtype=np.int64)
    weighted_labels = np.asarray([int(context["label"]) for context in contexts], dtype=np.int64)
    weighted_predictions = np.asarray([int(clean_predictions[str(context["record_id"])]) for context in contexts], dtype=np.int64)
    return {"unique_record": _metrics(predictions, labels), "moving_context_weighted": _metrics(weighted_predictions, weighted_labels)}


def _per_action_rankings(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for class_id, label in enumerate(LABELS):
        rows = [candidate for context in contexts if int(context["label"]) == class_id for candidate in context["candidates"]]
        values: dict[str, list[float]] = defaultdict(list)
        utilities: dict[str, list[float]] = defaultdict(list)
        for candidate in rows:
            for name, key in (("FeatureCos", "feature_cos_clean"), ("NegativeFeatureL2", "feature_l2_clean"), ("NegativePosteriorJSD", "posterior_jsd_clean"), ("SceneVisibility", "scene_visibility"), ("ProjectedArea", "projected_area"), ("PoseConfidence", "pose_confidence"), ("NegativeDistance", "distance_m"), ("NegativeMotionDeviation", "motion_deviation")):
                value = float(candidate.get(key, float("nan")))
                if np.isfinite(value):
                    values[name].append(value)
                    utilities[name].append(float(candidate["gt_logp"]))
        output[label] = {name: {"count": len(values[name]), "spearman": _corr(values[name], utilities[name])[1]} for name in values}
    return output


def _qualitative_contact_sheet(output: Path) -> Optional[str]:
    if not QUALITATIVE_MANIFEST.is_file():
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    manifest = _read_json(QUALITATIVE_MANIFEST)
    images: list[tuple[str, Any]] = []
    for item in manifest[:8]:
        path = Path(str(item.get("figure_record", {}).get("figure", "")))
        if path.is_file():
            try:
                images.append((str(item.get("case_id", path.stem)), Image.open(path).convert("RGB")))
            except (OSError, ValueError):
                continue
    if not images:
        return None
    thumb_w, thumb_h = 420, 300
    canvas = Image.new("RGB", (thumb_w * 2, (thumb_h + 28) * ((len(images) + 1) // 2)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(images):
        image.thumbnail((thumb_w - 8, thumb_h - 8))
        x = (index % 2) * thumb_w
        y = (index // 2) * (thumb_h + 28)
        canvas.paste(image, (x + (thumb_w - image.width) // 2, y + 4))
        draw.text((x + 5, y + thumb_h + 4), name, fill="black")
    path = output / "representation_cases.png"
    canvas.save(path)
    return str(path.resolve())


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]
    frozen = methods["Candidate-Conditioned Spatial"]
    lines = [
        "# Recognizer-Level Clean Motion Reference Audit", "",
        "Val Moving contexts only. Clean references use exact raw-val AMASS intervals, MotionConverter and Habitat articulated-humanoid FK; no candidate estimated skeleton was used to construct a clean reference.", "",
        "| Method | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|",
    ]
    for name, metric in methods.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    clean = result["clean_recognizer"]
    lines.extend([
        "", "## Clean recognizer", "",
        f"Unique-record clean ST-GCN: Accuracy={clean['unique_record']['accuracy']:.6f}, Macro-F1={clean['unique_record']['macro_f1']:.6f} (N={clean['unique_record']['count']}).",
        f"Moving-context weighted clean ST-GCN: Accuracy={clean['moving_context_weighted']['accuracy']:.6f}, Macro-F1={clean['moving_context_weighted']['macro_f1']:.6f} (N={clean['moving_context_weighted']['count']}).",
        "", "## Representation and ranking", "",
    ])
    for name, values in result["ranking_correlations"].items():
        lines.append(f"- {name}: global Pearson={values['pearson_global']:.6f}, Spearman={values['spearman_global']:.6f}; within-context Spearman mean={values['mean_within_context_spearman']:.6f}.")
    similarity = result["representation_similarity"]
    lines.append(f"- Feature cosine mean={similarity['feature_cos_clean']['mean']:.6f}; normalized feature L2 mean={similarity['normalized_feature_l2_clean']['mean']:.6f}; posterior JSD mean={similarity['posterior_jsd_clean']['mean']:.6f}.")
    quadrants = result["quadrant_metrics"]
    lines.extend([
        "", "## Selected-vs-GT-best audit", "",
        f"Clean-correct rate={quadrants['rates']['clean_correct']:.6f}; GT-margin oracle-correct rate={quadrants['rates']['oracle_correct']:.6f}.",
        f"CleanWrong→OracleCorrect contexts={quadrants['cleanwrong_oraclecorrect']['count']}; CleanCorrect→OracleWrong contexts={quadrants['cleancorrect_oraclewrong']['count']}.",
        f"Frozen selection wrong while oracle correct={result['privileged_selector_metrics']['frozen_wrong_oracle_correct_contexts']}; severe-miss contexts={result['privileged_selector_metrics']['severe_miss_contexts']}.",
        "", "## Scientific interpretation", "",
    ])
    if clean["moving_context_weighted"]["accuracy"] < methods["Candidate-Conditioned Spatial"]["accuracy"]:
        lines.append("The clean recognizer is a representation ceiling reference: its score is reported independently of ActiveView selector decisions. The estimated candidate and clean distributions are not interchangeable, so the remaining gap should be interpreted as both motion/observation mismatch and candidate selection error.")
    else:
        lines.append("The clean reference is not used as deployable input. Its independent recognizer score provides the ceiling against which estimated-view mismatch and selector regret are separated.")
    lines.extend([
        "Candidate-conditioned Spatial is represented by the archived Stage-C proposal_rank_1_id; GT-GTMargin Oracle selects the legal candidate with maximum true-class margin.",
        "", "Flags: `test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_pose_estimation=false`; `clean_h36m17_source=exact AMASS->Habitat FK`; `clean_reference_used_for_privileged_reference_only=true`; `deployable=false`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(data_root: Path, output_dir: Path, device_name: str = "cuda:0", batch_size: int = 256, max_contexts: Optional[int] = None) -> dict[str, Any]:
    """Run the complete Val Moving audit and persist compact JSON artifacts."""
    if not torch.cuda.is_available() and str(device_name).startswith("cuda"):
        raise RuntimeError("CUDA is required for frozen ST-GCN inference")
    policy_root = data_root / "datasets" / POLICY_NAME
    stage_c_path = policy_root / "stage_c" / "features" / "val.jsonl"
    stage_d_path = policy_root / "stage_d" / "features" / "val.jsonl"
    raw_path = data_root / "datasets" / DATASET_NAME / "raw-val" / "official_val.json"
    true_cache_path = data_root / TRUE_CACHE_REL
    stage_c_rows = _read_jsonl(stage_c_path)
    moving_rows = _read_jsonl(stage_d_path)
    raw_payload = _read_json(raw_path)
    raw_records = {str(item["record_id"]): item for item in raw_payload}
    if len(raw_records) != len(raw_payload):
        raise ValueError("duplicate record_id in official_val.json")
    if any(str(row.get("policy_split", "")) != "val" for row in moving_rows):
        raise ValueError("non-Val row found in Stage-D input")
    if max_contexts is not None:
        if max_contexts <= 0:
            raise ValueError("max_contexts must be positive")
        moving_rows = moving_rows[:max_contexts]
    record_ids = sorted({str(row["record_id"]) for row in moving_rows})
    missing = sorted(set(record_ids) - set(raw_records))
    if missing:
        raise ValueError(f"moving rows missing raw records: {missing[:3]}")
    fk_validation = _validate_fk(data_root, moving_rows, raw_records)
    if fk_validation["status"] != "PASS":
        raise RuntimeError("generic Habitat H36M17 FK did not match the legacy helper")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "fk_validation.json", fk_validation)
    source_missing = []
    invalid_intervals = []
    for record_id in record_ids:
        record = raw_records[record_id]
        if not Path(str(record.get("source_path", ""))).is_file():
            source_missing.append(record_id)
        try:
            start = int(record["start_frame"])
            end = int(record["end_frame"])
            total = int(record["source_num_frames"])
            if start < 0 or end < start or end >= total:
                invalid_intervals.append(record_id)
        except (KeyError, TypeError, ValueError):
            invalid_intervals.append(record_id)
    alignment_audit = {
        "status": "PASS" if not source_missing and not invalid_intervals else "BLOCKED",
        "official_val_records": len(raw_payload),
        "val_moving_contexts": len(moving_rows),
        "unique_moving_record_ids": len(record_ids),
        "exact_clean_motion_matched": len(record_ids) if not source_missing and not invalid_intervals else 0,
        "source_files_missing": source_missing,
        "invalid_intervals": invalid_intervals,
        "record_mapping": "moving Stage-D record_id -> unique raw-val official_val.json record",
        "frame_resampling": "np.linspace(start_frame, end_frame, 30, dtype=int64)",
        "clean_instance_alignment_confirmed": not source_missing and not invalid_intervals,
        "protocol": {"split": "val_moving", "target_frames": FRAME_COUNT, "test_used": False, "training_used": False, "new_rgb_rendered": False, "new_pose_estimation": False},
    }
    _write_json(output_dir / "alignment_audit.json", alignment_audit)
    clean_cache_path = data_root / "diagnostics" / "reduced12_recognizer_clean_reference_audit" / "clean_reference.npz"
    if clean_cache_path.is_file():
        clean_payload = _load_npz(clean_cache_path)
        if set(clean_payload.get("record_ids", []).tolist()) != set(record_ids) or clean_payload.get("skeletons", np.empty(0)).shape != (len(record_ids), 3, FRAME_COUNT, 17, 1):
            clean_payload, clean_summary = _build_clean_reference(data_root, raw_records, record_ids, clean_cache_path)
        else:
            clean_summary = {"records": len(record_ids), "shape": list(clean_payload["skeletons"].shape), "reused": True, "cache": str(clean_cache_path.resolve())}
    else:
        clean_payload, clean_summary = _build_clean_reference(data_root, raw_records, record_ids, clean_cache_path)
    checkpoint = data_root / DEFAULT_CHECKPOINT_REL
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model, device = load_checkpoint(checkpoint, NUM_CLASSES, device_name)
    clean_arrays = [np.asarray(array, dtype=np.float32) for array in clean_payload["skeletons"]]
    clean_features_array, clean_logits_array, _ = _infer(model, clean_arrays, device, batch_size)
    clean_map_features = {str(record_id): clean_features_array[index] for index, record_id in enumerate(record_ids)}
    clean_map_logp = {str(record_id): clean_logits_array[index] for index, record_id in enumerate(record_ids)}
    clean_map_predictions = {str(record_id): int(np.argmax(clean_logits_array[index])) for index, record_id in enumerate(record_ids)}
    clean_metric_payload = {"record_ids": np.asarray(record_ids), "skeletons": clean_payload["skeletons"], "labels": clean_payload["labels"], "features": clean_features_array, "logits": clean_logits_array, "predictions": np.asarray([clean_map_predictions[str(record_id)] for record_id in record_ids], dtype=np.int64)}
    clean_metric_path = data_root / "diagnostics" / "reduced12_recognizer_clean_reference_audit" / "clean_reference_stgcn.npz"
    np.savez_compressed(clean_metric_path, **clean_metric_payload)
    true_cache = _load_npz(true_cache_path)
    scene_map = _artifact_map(SCENE_VIS_PATH, "scene_visibility_score")
    human_map = _artifact_map(HUMAN_VIS_PATH, "human_observability")
    projected_map = _artifact_map(HUMAN_VIS_PATH, "projected_area")
    for episode in scene_map:
        if episode in human_map:
            for candidate_id, value in projected_map.get(episode, {}).items():
                human_map[episode].setdefault(candidate_id, human_map[episode].get(candidate_id, value))
    contexts, candidate_summary = _load_candidate_contexts(data_root, moving_rows, stage_c_rows, true_cache, model, device, batch_size, scene_map, human_map)
    for context in contexts:
        for candidate in context["candidates"]:
            candidate["projected_area"] = float(projected_map.get(context["episode_id"], {}).get(int(candidate["candidate_id"]), float("nan")))
    _attach_similarity(contexts, clean_map_features, clean_map_logp, clean_map_predictions)
    methods, diagnostics, per_class = _evaluate_contexts(contexts, clean_map_predictions)
    clean_metrics = _clean_metrics(clean_payload, clean_map_predictions, contexts)
    ranking_correlations = diagnostics.pop("rankings")
    per_action_rank = _per_action_rankings(contexts)
    diagnostics["per_action_ranking_correlations"] = per_action_rank
    diagnostics["candidate_feature_cache"] = str((data_root / "diagnostics/reduced12_recognizer_clean_reference_audit/candidate_features.npz").resolve())
    # Keep the large candidate feature matrix outside Git, but make it reusable.
    flat = diagnostics.pop("flat_candidates")
    np.savez_compressed(data_root / "diagnostics/reduced12_recognizer_clean_reference_audit" / "candidate_features.npz", episode_ids=np.asarray([str(item["episode_id"]) for item in flat]), candidate_ids=np.asarray([int(item["candidate_id"]) for item in flat]), features=np.asarray([item["feature"] for item in flat], dtype=np.float32), logits=np.asarray([item["logp"] for item in flat], dtype=np.float32))
    quadrant_rows = diagnostics.pop("quadrant_rows")
    qualitative_path = _qualitative_contact_sheet(output_dir)
    frozen = methods["Candidate-Conditioned Spatial"]
    result = {
        "experiment_id": "REDUCED12_RECOGNIZER_CLEAN_REFERENCE_AUDIT", "status": "COMPLETED",
        "population": {"val_moving_contexts": len(contexts), "unique_val_records": len(record_ids), "candidate_count": int(candidate_summary["candidate_count"])},
        "labels": list(LABELS), "clean_recognizer": clean_metrics, "methods": methods,
        "deltas_vs_candidate_conditioned_spatial": {name: {"accuracy_pp": 100.0 * (methods[name]["accuracy"] - frozen["accuracy"]), "macro_f1_pp": 100.0 * (methods[name]["macro_f1"] - frozen["macro_f1"])} for name in methods},
        "representation_similarity": diagnostics["similarity"], "ranking_correlations": ranking_correlations,
        "quadrant_metrics": diagnostics["quadrants"], "privileged_selector_metrics": diagnostics["selector_audit"],
        "per_class_metrics": per_class, "protocol": {"split": "val_moving", "target_frames": FRAME_COUNT, "normalization": "SkeletonNormalizer(h36m_17).normalize_sequence(align_canonical=True)", "test_used": False, "training_used": False, "new_rgb_rendered": False, "new_pose_estimation": False, "clean_h36m17_source": "exact AMASS interval -> MotionConverter -> Habitat humanoid FK", "clean_reference_used_for_privileged_reference_only": True, "deployable": False},
        "alignment_audit": {"record_alignment": alignment_audit, "fk_validation": fk_validation, "clean_summary": clean_summary, "candidate_summary": candidate_summary},
        "artifacts": {"checkpoint": str(checkpoint.resolve()), "clean_reference_cache": str(clean_cache_path.resolve()), "clean_stgcn_cache": str(clean_metric_path.resolve()), "candidate_feature_cache": diagnostics["candidate_feature_cache"], "qualitative_rgb_reused": qualitative_path},
    }
    _write_json(output_dir / "clean_recognition_metrics.json", clean_metrics)
    _write_json(output_dir / "representation_similarity.json", diagnostics["similarity"])
    _write_json(output_dir / "ranking_correlations.json", ranking_correlations)
    _write_json(output_dir / "quadrant_metrics.json", diagnostics["quadrants"])
    _write_json(output_dir / "cleanwrong_oraclecorrect.json", diagnostics["selector_audit"]["cleanwrong_oraclecorrect_deep_dive"])
    _write_json(output_dir / "cleancorrect_oraclewrong.json", {"count": diagnostics["selector_audit"]["cleancorrect_oraclewrong_count"]})
    _write_json(output_dir / "privileged_selector_metrics.json", diagnostics["selector_audit"])
    _write_json(output_dir / "per_class_metrics.json", per_class)
    _write_json(output_dir / "result.json", result)
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-contexts", type=int, default=None)
    args = parser.parse_args(argv)
    result = evaluate(args.data_root.resolve(), args.output.resolve(), args.device, args.batch_size, args.max_contexts)
    print(json.dumps({"status": result["status"], "population": result["population"], "clean": result["clean_recognizer"]["moving_context_weighted"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
