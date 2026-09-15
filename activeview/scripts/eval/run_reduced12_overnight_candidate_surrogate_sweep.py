#!/usr/bin/env python3
"""Capacity audit for action-independent candidate observation surrogates.

All selectors are evaluated on reduced12 Moving Val using the frozen Yaw8
encoder/shared head and the unchanged Stage-A legal candidate pool.  Train
data is used only to fit representation manifolds (feature kNN/Mahalanobis and
skeleton PCA).  Policy Test, new perception generation and recognizer updates
are deliberately out of scope.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation, gt_margin
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import VisibilityPredictor, _option_geometry, _predict
from activeview.scripts.experiments.run_reduced12_pepperpose_frame0_confidence_audit import (
    _candidate_slots,
    _evaluate,
    _load_frame0_scene_scores,
    _load_options,
    _random_actions,
    _select,
    _terminal,
)
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import _load_rows
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import _load_existing_dino


SEED = 42
NUM_CLASSES = len(LABELS)
FEATURE_DIM = 256
MAX_KNN = 20
BATCH_SIZE = 512
YAW8_ROOT = Path("datasets/stgcn_babel_reduced12_habitat_yolo26n_yaw8_v1")
YAW8_CHECKPOINT = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
YAW8_HEAD_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
)
POLICY_ROOT = Path("datasets/policy_reduced12_eight_placement_v1")
RGB_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1/RGBGlobal+Geometry.pth"
)
OUTPUT_DEFAULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/overnight_candidate_surrogate_sweep"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _infer_features(model: torch.nn.Module, data: np.ndarray, device: torch.device) -> np.ndarray:
    result: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), BATCH_SIZE):
            batch = torch.from_numpy(np.asarray(data[start : start + BATCH_SIZE])).float().to(device, non_blocking=True)
            result.append(model.forward_features(batch).cpu().numpy())
    return np.concatenate(result, axis=0).astype(np.float32)


def _load_yaw8_train_bank(data_root: Path, model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    root = data_root / YAW8_ROOT
    data = np.load(root / "train_data.npy", mmap_mode="r")
    labels = np.load(root / "train_labels.npy", mmap_mode="r").astype(np.int64)
    if len(data) != len(labels):
        raise ValueError("Yaw8 train skeleton/label count mismatch")
    return _infer_features(model, data, device), np.asarray(labels)


def _candidate_arrays(
    rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    count = int(sum(len(_candidate_slots(options, i)) for i in range(len(rows))))
    features = np.empty((count, FEATURE_DIM), dtype=np.float32)
    skeletons = np.empty((count, 3, 30, 17), dtype=np.float32)
    owners = np.empty(count, dtype=np.int64)
    slots = np.empty(count, dtype=np.int64)
    ids = np.empty(count, dtype=np.int64)
    labels = np.empty(count, dtype=np.int64)
    position = 0
    for index, row in enumerate(rows):
        with np.load(str(row["archive_path"]), allow_pickle=False) as archive:
            archive_ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            archive_skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        if archive_skeleton.shape != (32, 3, 30, 17) or archive_ids.shape != (32,):
            raise ValueError(f"unexpected candidate archive shape: {row['archive_path']}")
        lookup = {int(value): offset for offset, value in enumerate(archive_ids.tolist())}
        for slot in _candidate_slots(options, index):
            view = int(options["ids"][index, slot])
            if view not in lookup:
                raise ValueError(f"candidate viewpoint missing from archive: {row['episode_id']} {view}")
            features[position] = np.asarray(options["features"][index, slot], dtype=np.float32)
            skeletons[position] = archive_skeleton[lookup[view]]
            owners[position] = index
            slots[position] = int(slot)
            ids[position] = view
            labels[position] = int(row["label_id"])
            position += 1
    if position != count:
        raise RuntimeError("candidate flatten count mismatch")
    return {"features": features, "skeletons": skeletons, "owners": owners, "slots": slots, "ids": ids, "labels": labels}


def _matrix_from_flat(values: np.ndarray, arrays: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray]) -> np.ndarray:
    matrix = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    for value, owner, slot in zip(values, arrays["owners"], arrays["slots"]):
        matrix[int(owner), int(slot)] = float(value)
    return matrix


def _row_normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize finite candidate scores independently within each context."""
    output = np.full(values.shape, -np.inf, dtype=np.float64)
    for index in range(values.shape[0]):
        active = np.flatnonzero(mask[index])
        finite = active[np.isfinite(values[index, active])]
        if finite.size == 0:
            continue
        low = float(np.min(values[index, finite]))
        high = float(np.max(values[index, finite]))
        output[index, finite] = 0.0 if high - low <= 1e-12 else (values[index, finite] - low) / (high - low)
    return output


def _feature_record_consensus_matrix(
    options: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Build a candidate score matrix from same-record feature consensus."""
    scores = np.full(options["ids"].shape, -np.inf, dtype=np.float64)
    groups: dict[str, list[tuple[int, int]]] = {}
    for index in range(len(rows)):
        for slot in _candidate_slots(options, index).tolist():
            groups.setdefault(str(rows[index]["record_id"]), []).append((index, int(slot)))
    normalized = np.asarray(options["features"], dtype=np.float32)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=-1, keepdims=True), 1e-8)
    for positions in groups.values():
        centroid = np.mean([normalized[index, slot] for index, slot in positions], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        for index, slot in positions:
            scores[index, slot] = float(np.dot(normalized[index, slot], centroid))
    return scores


def _select_fusion_lambda(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    rgb_scores: np.ndarray,
    surrogate_scores: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Select a fixed RGBGlobal+surrogate lambda on a record holdout."""
    records = sorted({str(row["record_id"]) for row in rows})
    holdout_records = set(records[-max(1, int(round(0.1 * len(records)))) :])
    holdout = np.asarray([str(row["record_id"]) in holdout_records for row in rows], dtype=bool)
    candidate_mask = np.zeros(options["mask"].shape, dtype=bool)
    candidate_mask[:, 1:] = options["mask"][:, 1:]
    rgb_norm = _row_normalize(rgb_scores, candidate_mask)
    surrogate_norm = _row_normalize(surrogate_scores, candidate_mask)
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    candidates: dict[str, Any] = {}
    best: tuple[float, float, float] | None = None
    for value in (0.1, 0.25, 0.5):
        combined = np.full(rgb_norm.shape, -np.inf, dtype=np.float64)
        valid = np.isfinite(rgb_norm) & np.isfinite(surrogate_norm)
        combined[valid] = rgb_norm[valid] + value * surrogate_norm[valid]
        actions = _select(rows, options, combined)
        predictions = _terminal(options, actions)
        accuracy = float(np.mean(predictions[holdout] == labels[holdout])) if holdout.any() else 0.0
        candidates[str(value)] = {"holdout_accuracy": accuracy}
        key = (accuracy, -value, 0.0)
        if best is None or key > best:
            best = (accuracy, -value, value)
    if best is None:
        raise RuntimeError("unable to select fusion lambda")
    selected = float(best[2])
    return selected, {
        "selection_source": "Policy Train record holdout",
        "holdout_records": len(holdout_records),
        "holdout_contexts": int(holdout.sum()),
        "criterion": "highest holdout Accuracy; smallest lambda on tie",
        "candidates": candidates,
        "selected_lambda": selected,
        "test_used": False,
    }


def _knn_distances(train_features: np.ndarray, candidate_features: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    train = torch.from_numpy(train_features).to(device)
    candidate = torch.from_numpy(candidate_features).to(device)
    train = F.normalize(train, dim=1)
    outputs10: list[np.ndarray] = []
    outputs20: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(candidate), BATCH_SIZE):
            similarities = F.normalize(candidate[start : start + BATCH_SIZE], dim=1) @ train.T
            nearest = torch.topk(similarities, MAX_KNN, dim=1, largest=True).values
            outputs10.append((1.0 - nearest[:, :10].mean(dim=1)).cpu().numpy())
            outputs20.append((1.0 - nearest.mean(dim=1)).cpu().numpy())
    return np.concatenate(outputs10).astype(np.float32), np.concatenate(outputs20).astype(np.float32)


def _feature_pca(train_features: np.ndarray, train_labels: np.ndarray, candidate_features: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    train = torch.from_numpy(train_features).to(device)
    mean = train.mean(dim=0)
    centered = train - mean
    covariance = centered.T @ centered / max(len(train) - 1, 1)
    eigenvalues, components = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    components = components[:, order[:64]]
    projected = centered @ components
    projected_np = projected.cpu().numpy()
    global_mean = projected_np.mean(axis=0)
    global_var = np.maximum(projected_np.var(axis=0), 1e-4)
    class_means = np.zeros((NUM_CLASSES, 64), dtype=np.float32)
    class_vars = np.zeros((NUM_CLASSES, 64), dtype=np.float32)
    for class_id in range(NUM_CLASSES):
        values = projected_np[train_labels == class_id]
        if len(values) < 2:
            class_means[class_id] = global_mean
            class_vars[class_id] = global_var
        else:
            class_means[class_id] = values.mean(axis=0)
            class_vars[class_id] = np.maximum(values.var(axis=0), 1e-4)
    candidate = (candidate_features - mean.cpu().numpy()) @ components.cpu().numpy()
    global_distance = np.sum((candidate - global_mean[None]) ** 2 / global_var[None], axis=1)
    class_distance = np.stack([
        np.sum((candidate - class_means[class_id][None]) ** 2 / class_vars[class_id][None], axis=1)
        for class_id in range(NUM_CLASSES)
    ], axis=1)
    return (-global_distance).astype(np.float32), (-np.min(class_distance, axis=1)).astype(np.float32)


def _skeleton_pca(train_data: np.ndarray, candidate_skeletons: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    train_flat = np.asarray(train_data, dtype=np.float32).reshape(len(train_data), -1)
    candidate_flat = candidate_skeletons.reshape(len(candidate_skeletons), -1)
    train_tensor = torch.from_numpy(train_flat).to(device)
    mean = train_tensor.mean(dim=0)
    centered = train_tensor - mean
    _, _, components = torch.pca_lowrank(centered, q=128, center=False, niter=2)
    candidate = torch.from_numpy(candidate_flat).to(device)
    errors: list[np.ndarray] = [[], []]
    with torch.inference_mode():
        for start in range(0, len(candidate), BATCH_SIZE):
            projection = (candidate[start : start + BATCH_SIZE] - mean) @ components[:, :128]
            reconstruction = projection @ components[:, :128].T + mean
            error128 = ((candidate[start : start + BATCH_SIZE] - reconstruction) ** 2).mean(dim=1)
            projection64 = (candidate[start : start + BATCH_SIZE] - mean) @ components[:, :64]
            reconstruction64 = projection64 @ components[:, :64].T + mean
            error64 = ((candidate[start : start + BATCH_SIZE] - reconstruction64) ** 2).mean(dim=1)
            errors[0].append(error64.cpu().numpy())
            errors[1].append(error128.cpu().numpy())
    return -np.concatenate(errors[0]).astype(np.float32), -np.concatenate(errors[1]).astype(np.float32)


def _perturbation_stability(
    model: torch.nn.Module,
    head: torch.nn.Module,
    candidate_skeletons: np.ndarray,
    candidate_features: np.ndarray,
    candidate_logp: np.ndarray,
    sparse_joints: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    stability: list[np.ndarray] = []
    jsd_values: list[np.ndarray] = []
    torch.manual_seed(SEED)
    model.eval()
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(candidate_skeletons), BATCH_SIZE):
            base = torch.from_numpy(candidate_skeletons[start : start + BATCH_SIZE]).to(device)
            base_input = base.unsqueeze(-1)
            base_feature = torch.from_numpy(candidate_features[start : start + BATCH_SIZE]).to(device)
            base_logp = torch.from_numpy(candidate_logp[start : start + BATCH_SIZE]).to(device)
            perturbations: list[torch.Tensor] = []
            for _ in range(4):
                perturbations.append(base_input + torch.randn_like(base_input) * 0.01)
            for perturbation_index in range(4):
                perturbation = base_input.clone()
                joints = sparse_joints[start : start + BATCH_SIZE, perturbation_index]
                for local, joint in enumerate(joints.tolist()):
                    perturbation[local, :, :, int(joint), 0] += torch.randn((3, 30), device=device) * 0.03
                perturbations.append(perturbation)
            stacked = torch.cat(perturbations, dim=0)
            features = model.forward_features(stacked).reshape(8, -1, FEATURE_DIM).transpose(0, 1)
            logits = head(features.reshape(-1, FEATURE_DIM)).reshape(features.shape[0], 8, NUM_CLASSES)
            cosine = F.cosine_similarity(features, base_feature[:, None, :], dim=-1)
            stability.append((1.0 - cosine).mean(dim=1).cpu().numpy())
            logp = torch.log_softmax(logits, dim=-1)
            p0 = base_logp.exp()[:, None, :]
            p1 = logp.exp()
            mean_p = 0.5 * (p0 + p1)
            jsd = 0.5 * (p0 * (base_logp[:, None, :] - mean_p.clamp_min(1e-8).log())).sum(dim=-1)
            jsd += 0.5 * (p1 * (logp - mean_p.clamp_min(1e-8).log())).sum(dim=-1)
            jsd_values.append(jsd.mean(dim=1).cpu().numpy())
    return -np.concatenate(stability).astype(np.float32), np.concatenate(jsd_values).astype(np.float32)


def _consensus_scores(arrays: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = arrays["features"]
    owners = arrays["owners"]
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    feature_medoid = np.full(len(features), -np.inf, dtype=np.float32)
    skeleton_medoid = np.full(len(features), -np.inf, dtype=np.float32)
    row_positions: list[list[int]] = [[] for _ in rows]
    for position, owner in enumerate(owners.tolist()):
        row_positions[int(owner)].append(position)
    pose = arrays["skeletons"].transpose(0, 2, 3, 1).reshape(len(features), 30 * 17, 3)
    pose_norm = np.maximum(np.linalg.norm(pose, axis=2).mean(axis=1), 1e-6)
    for positions in row_positions:
        if not positions:
            continue
        values = normalized[positions]
        if len(positions) == 1:
            feature_medoid[positions[0]] = 0.0
            skeleton_medoid[positions[0]] = 0.0
            continue
        distances = 1.0 - values @ values.T
        np.fill_diagonal(distances, np.nan)
        feature_medoid[positions] = -np.nanmedian(distances, axis=1)
        for local, position in enumerate(positions):
            delta = np.linalg.norm(pose[positions] - pose[position][None], axis=2).mean(axis=1)
            scale = 0.5 * (pose_norm[positions] + pose_norm[position])
            delta[local] = np.nan
            skeleton_medoid[position] = -float(np.nanmedian(delta / np.maximum(scale, 1e-6)))
    by_record: dict[str, list[int]] = {}
    for position, owner in enumerate(owners.tolist()):
        by_record.setdefault(str(rows[int(owner)]["record_id"]), []).append(position)
    record_consensus = np.empty(len(features), dtype=np.float32)
    for positions in by_record.values():
        centroid = normalized[positions].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        record_consensus[positions] = normalized[positions] @ centroid
    return feature_medoid, skeleton_medoid, record_consensus


def _within_context(values: np.ndarray, arrays: Mapping[str, Any], true_logp: np.ndarray, margins: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    per_row: list[list[int]] = [[] for _ in rows]
    for position, owner in enumerate(arrays["owners"].tolist()):
        per_row[int(owner)].append(position)
    rho_logp: list[float] = []
    rho_margin: list[float] = []
    for positions in per_row:
        if len(positions) >= 2:
            rho_logp.append(correlation(values[positions], true_logp[positions], spearman=True))
            rho_margin.append(correlation(values[positions], margins[positions], spearman=True))
    return (float(np.mean(rho_logp)) if rho_logp else 0.0, float(np.mean(rho_margin)) if rho_margin else 0.0)


def _top3_coverage(values: np.ndarray, arrays: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], correct: np.ndarray) -> float:
    per_row: list[list[int]] = [[] for _ in rows]
    for position, owner in enumerate(arrays["owners"].tolist()):
        per_row[int(owner)].append(position)
    covered = 0
    for positions in per_row:
        ordered = sorted(positions, key=lambda p: (-float(values[p]), int(arrays["ids"][p])))
        covered += int(np.any(correct[ordered[:3]])) if ordered else 0
    return float(covered / len(rows)) if rows else 0.0


def _surrogate_metrics(
    name: str,
    values: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    arrays: Mapping[str, Any],
    true_logp: np.ndarray,
    margins: np.ndarray,
    correct: np.ndarray,
    high_mask: np.ndarray,
) -> tuple[dict[str, Any], list[int]]:
    scores = _matrix_from_flat(values, arrays, rows, options)
    actions = _select(rows, options, scores)
    metric = _evaluate(name, rows, options, actions)
    selected_predictions = _terminal(options, actions)
    high_indices = np.flatnonzero(high_mask)
    high_metric = classification(
        np.asarray([int(rows[i]["label_id"]) for i in high_indices], dtype=np.int64), selected_predictions[high_indices]
    ) if high_indices.size else {"accuracy": 0.0, "macro_f1": 0.0}
    within_logp, within_margin = _within_context(values, arrays, true_logp, margins, rows)
    metric.update({
        "global_spearman_true_logp": correlation(values, true_logp, spearman=True),
        "global_spearman_gt_margin": correlation(values, margins, spearman=True),
        "within_context_spearman_true_logp": within_logp,
        "within_context_spearman_gt_margin": within_margin,
        "high_occlusion_accuracy": float(high_metric["accuracy"]),
        "high_occlusion_macro_f1": float(high_metric["macro_f1"]),
        "top3_har_correct_coverage": _top3_coverage(values, arrays, rows, correct),
        "candidate_count": int(len(values)),
        "test_used": False,
        "verdict": "PROMOTE" if metric["accuracy"] >= 0.65 else ("WEAK SURROGATE" if metric["accuracy"] >= 0.60 else "KILL"),
    })
    return metric, actions


def _reference_metrics(name: str, rows: Sequence[Mapping[str, Any]], options: Mapping[str, np.ndarray], actions: Sequence[int]) -> dict[str, Any]:
    metric = _evaluate(name, rows, options, actions)
    metric.update({"global_spearman_true_logp": None, "global_spearman_gt_margin": None, "within_context_spearman_true_logp": None, "within_context_spearman_gt_margin": None, "high_occlusion_accuracy": None, "high_occlusion_macro_f1": None, "top3_har_correct_coverage": None, "verdict": "REFERENCE"})
    return metric


def _clean_audit(data_root: Path) -> dict[str, Any]:
    recovery = REPO_ROOT / "experiments/reduced12_eight_placement_v1/clean_perception_reference_recovery/result.json"
    if not recovery.is_file():
        return {"available": False, "reason": "clean-perception recovery report is absent; no rendering attempted", "test_used": False}
    payload = _read_json(recovery)
    flags = payload.get("flags", {})
    return {
        "available": False,
        "reason": "no exact per-candidate HM3D-vs-clean perception cache; historical Moving mapping is blocked",
        "recovery_status": payload.get("status"),
        "moving_clean_perception_asset_missing": flags.get("moving_clean_perception_asset_missing", True),
        "exact_record_mapping": payload.get("mapping_summary", {}).get("exact_matches", 0),
        "rendering_attempted": False,
        "F1_F2_F3": {
            "F1 Matched Skeleton Error": {"status": "UNAVAILABLE"},
            "F2 Matched Feature Distance": {"status": "UNAVAILABLE"},
            "F3 Layerwise Feature Distance": {"status": "UNAVAILABLE"},
        },
        "test_used": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    _seed()
    device = _device(args.device)
    data_root = get_data_root()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    train_rows, moving_rows = _load_rows(data_root)
    model, _ = load_checkpoint(data_root / YAW8_CHECKPOINT, NUM_CLASSES, str(device))
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(data_root / YAW8_HEAD_CHECKPOINT, map_location=device, weights_only=False)["state_dict"])
    train_features, train_labels = _load_yaw8_train_bank(data_root, model, device)
    yaw8_train_data = np.load(data_root / YAW8_ROOT / "train_data.npy", mmap_mode="r")
    train_options, train_cache_meta = _load_options(data_root, train_rows, "train", device)
    moving_options, moving_cache_meta = _load_options(data_root, moving_rows, "moving_val", device)
    arrays = _candidate_arrays(moving_rows, moving_options)
    candidate_logp = np.asarray([moving_options["logp"][int(owner), int(slot)] for owner, slot in zip(arrays["owners"], arrays["slots"])], dtype=np.float32)
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    true_logp = candidate_logp[np.arange(len(candidate_logp)), arrays["labels"]]
    margins = np.asarray([gt_margin(logp, int(label)) for logp, label in zip(candidate_logp, arrays["labels"])], dtype=np.float32)
    correct = np.argmax(candidate_logp, axis=1) == arrays["labels"]

    knn10, knn20 = _knn_distances(train_features, arrays["features"], device)
    global_mahal, class_mahal = _feature_pca(train_features, train_labels, arrays["features"], device)
    pca64, pca128 = _skeleton_pca(yaw8_train_data, arrays["skeletons"], device)
    sparse_rng = np.random.default_rng(SEED)
    sparse_joints = sparse_rng.integers(1, 17, size=(len(arrays["features"]), 4), endpoint=False)
    stability, jsd = _perturbation_stability(model, head, arrays["skeletons"], arrays["features"], candidate_logp, sparse_joints, device)
    feature_medoid, skeleton_medoid, record_consensus = _consensus_scores(arrays, moving_rows)

    stats = _read_json(data_root / POLICY_ROOT / "stage_c/stage_c_feature_stats.json")
    moving_geometry, moving_ids, moving_mask = _option_geometry(moving_rows, stats)
    if not np.array_equal(moving_ids, moving_options["ids"]) or not np.array_equal(moving_mask, moving_options["mask"]):
        raise ValueError("candidate geometry/action IDs do not match Yaw8Fair cache")
    dino, dino_meta = _load_existing_dino(data_root, moving_rows, "val")
    rgb_model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    rgb_model.load_state_dict(torch.load(data_root / RGB_CHECKPOINT, map_location=device, weights_only=False)["state_dict"])
    rgb_scores = _predict(rgb_model, moving_geometry, dino, device)
    scene_scores = _load_frame0_scene_scores(data_root, moving_rows)
    high_mask = scene_scores[:, 0] <= float(np.quantile(scene_scores[:, 0], 1.0 / 3.0))

    score_values = {
        "A Feature-kNN k10": knn10 * -1.0,
        "A Feature-kNN k20": knn20 * -1.0,
        "B Global Mahalanobis": global_mahal,
        "B MinClass Mahalanobis": class_mahal,
        "C PCA64": pca64,
        "C PCA128": pca128,
        "D Perturbation Stability": stability,
        "E Feature Medoid": feature_medoid,
        "E Skeleton Medoid": skeleton_medoid,
        "E Record Consensus": record_consensus,
    }
    surrogate_metrics: dict[str, Any] = {}
    actions: dict[str, list[int]] = {}
    for name, values in score_values.items():
        surrogate_metrics[name], actions[name] = _surrogate_metrics(name, values, moving_rows, moving_options, arrays, true_logp, margins, correct, high_mask)

    true_scores = _matrix_from_flat(true_logp, arrays, moving_rows, moving_options)
    random_actions = _random_actions(moving_rows)
    static_sums = np.zeros(32, dtype=np.float64)
    static_counts = np.zeros(32, dtype=np.int64)
    for index, row in enumerate(train_rows):
        for slot in _candidate_slots(train_options, index):
            view = int(train_options["ids"][index, slot])
            static_sums[view] += gt_margin(train_options["logp"][index, slot], int(row["label_id"]))
            static_counts[view] += 1
    fallback = float(static_sums.sum() / max(int(static_counts.sum()), 1))
    static_map = {view: float(static_sums[view] / static_counts[view]) if static_counts[view] else fallback for view in range(32)}
    static_values = np.asarray([static_map[int(view)] for view in arrays["ids"]], dtype=np.float32)
    reference_actions = {
        "Random": random_actions,
        "StaticPrior": _select(moving_rows, moving_options, _matrix_from_flat(static_values, arrays, moving_rows, moving_options)),
        "RGBGlobal": _select(moving_rows, moving_options, rgb_scores),
        "Frame0SceneVisibility": _select(moving_rows, moving_options, scene_scores),
        "GT-TrueLogP Oracle": _select(moving_rows, moving_options, true_scores),
    }
    reference_metrics = {name: _reference_metrics(name, moving_rows, moving_options, selected) for name, selected in reference_actions.items()}
    any_correct = np.asarray([np.any(correct[arrays["owners"] == index]) for index in range(len(moving_rows))], dtype=bool)
    reference_metrics["AnyCorrect Coverage"] = {"method": "AnyCorrect Coverage", "n": len(moving_rows), "coverage_rate": float(any_correct.mean()), "coverage_count": int(any_correct.sum()), "test_used": False, "verdict": "REFERENCE"}

    all_metrics = {**reference_metrics, **surrogate_metrics}
    clean_audit = _clean_audit(data_root)
    eligible = {name: item for name, item in surrogate_metrics.items()}
    best_name = max(eligible, key=lambda name: (eligible[name]["accuracy"], eligible[name]["macro_f1"], name)) if eligible else None
    strongest_rho = max((max(abs(float(item["within_context_spearman_true_logp"])), abs(float(item["within_context_spearman_gt_margin"]))) for item in eligible.values()), default=0.0)
    if best_name is not None and eligible[best_name]["accuracy"] >= 0.65:
        final_decision = "PROMOTE"
    elif strongest_rho >= 0.50:
        final_decision = "STRONG SURROGATE"
    elif best_name is not None and eligible[best_name]["accuracy"] >= 0.60:
        final_decision = "WEAK SURROGATE"
    else:
        final_decision = "NO SIMPLE ACTION-INDEPENDENT OBSERVATION-QUALITY SCALAR HAS SUFFICIENT CAPACITY"
    fusion = {"status": "SKIPPED_ALL_SURROGATES_BELOW_0_60", "reason": "conditional RGBGlobal+surrogate fusion gate was not reached", "test_used": False}
    fusion_method_name: str | None = None
    train_dino_meta: dict[str, Any] | None = None
    if best_name is not None and eligible[best_name]["accuracy"] >= 0.60:
        train_geometry, train_ids, train_mask = _option_geometry(train_rows, stats)
        train_dino, train_dino_meta = _load_existing_dino(data_root, train_rows, "train")
        train_rgb = _predict(rgb_model, train_geometry, train_dino, device)
        train_consensus = _feature_record_consensus_matrix(train_options, train_rows)
        selected_lambda, fusion_selection = _select_fusion_lambda(train_rows, train_options, train_rgb, train_consensus)
        moving_candidate_mask = np.zeros(moving_options["mask"].shape, dtype=bool)
        moving_candidate_mask[:, 1:] = moving_options["mask"][:, 1:]
        moving_rgb_norm = _row_normalize(rgb_scores, moving_candidate_mask)
        moving_surrogate = _matrix_from_flat(
            score_values[best_name], arrays, moving_rows, moving_options
        )
        moving_surrogate_norm = _row_normalize(moving_surrogate, moving_candidate_mask)
        fused_scores = np.full(moving_rgb_norm.shape, -np.inf, dtype=np.float64)
        valid = np.isfinite(moving_rgb_norm) & np.isfinite(moving_surrogate_norm)
        fused_scores[valid] = moving_rgb_norm[valid] + selected_lambda * moving_surrogate_norm[valid]
        fused_name = f"RGBGlobal + {best_name} (lambda={selected_lambda:g})"
        fusion_method_name = fused_name
        # ``_surrogate_metrics`` expects a flat candidate vector.
        fused_flat = np.asarray([
            fused_scores[int(owner), int(slot)]
            for owner, slot in zip(arrays["owners"], arrays["slots"])
        ], dtype=np.float32)
        fused_metric, fused_action = _surrogate_metrics(
            fused_name, fused_flat, moving_rows, moving_options, arrays,
            true_logp, margins, correct, high_mask,
        )
        all_metrics[fused_name] = fused_metric
        surrogate_metrics[fused_name] = fused_metric
        actions[fused_name] = fused_action
        fusion = {
            "status": "COMPLETED",
            "surrogate": best_name,
            "selected_lambda": selected_lambda,
            "selection": fusion_selection,
            "test_used": False,
            "training_used": False,
            "note": "Diagnostic fusion uses actual candidate observations and is not deployable",
        }
    summary = {
        "experiment_id": "REDUCED12_OVERNIGHT_CANDIDATE_SURROGATE_SWEEP",
        "status": "COMPLETED",
        "population": {"policy_train_contexts": len(train_rows), "moving_val_contexts": len(moving_rows), "moving_val_candidate_samples": len(arrays["features"]), "yaw8_train_observations": len(train_features)},
        "methods": all_metrics,
        "references": {"Random": all_metrics["Random"], "StaticPrior": all_metrics["StaticPrior"], "RGBGlobal": all_metrics["RGBGlobal"], "Frame0SceneVisibility": all_metrics["Frame0SceneVisibility"], "GT-TrueLogP Oracle": all_metrics["GT-TrueLogP Oracle"], "AnyCorrect Coverage": all_metrics["AnyCorrect Coverage"]},
        "surrogate_gate": {"best_surrogate": best_name, "best_accuracy": float(eligible[best_name]["accuracy"]) if best_name else None, "strongest_within_context_abs_spearman": strongest_rho, "hard_gate": "<0.60 KILL; 0.60-0.65 WEAK SURROGATE; >=0.65 PROMOTE; >=0.68 or within-rho>=0.50 STRONG SURROGATE", "decision": final_decision},
        "conditional_fusion": fusion,
        "clean_perception_audit": clean_audit,
        "flags": {"policy_test_used": False, "training_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "frozen_stgcn_modified": False, "future_candidate_observation_used_for_selection": True, "future_candidate_observation_used_for_train_targets_only": False, "deployable": False},
        "recognizer": {"stgcn_checkpoint": str((data_root / YAW8_CHECKPOINT).resolve()), "shared_head_checkpoint": str((data_root / YAW8_HEAD_CHECKPOINT).resolve()), "protocol": "Frozen Yaw8 encoder + Yaw8Fair shared head"},
        "cache_metadata": {"train": train_cache_meta, "moving_val": moving_cache_meta, "moving_val_dino": dino_meta, "train_dino": train_dino_meta},
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.monotonic() - started},
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifold_metrics.json", {name: surrogate_metrics[name] for name in score_values if name.startswith(("A ", "B ", "C "))})
    _write_json(output / "stability_metrics.json", {"D Perturbation Stability": surrogate_metrics["D Perturbation Stability"], "posterior_jsd_mean": float(np.mean(jsd)), "perturbation_definition": "4 Gaussian coordinate noise (sigma=0.01) + 4 sparse joint coordinate noise (sigma=0.03), seed=42"})
    _write_json(output / "consensus_metrics.json", {name: surrogate_metrics[name] for name in ("E Feature Medoid", "E Skeleton Medoid", "E Record Consensus")})
    _write_json(output / "matched_clean_perception_metrics.json", clean_audit)
    _write_json(output / "high_occlusion.json", {"definition": "Moving contexts at or below the bottom tertile of existing Frame0SceneVisibility stay score", "count": int(high_mask.sum()), "methods": {name: {"accuracy": item.get("high_occlusion_accuracy"), "macro_f1": item.get("high_occlusion_macro_f1")} for name, item in all_metrics.items()}})

    lines = [
        "# Overnight Candidate Utility Surrogate Sweep", "",
        f"Moving Val contains {len(moving_rows):,} contexts and {len(arrays['features']):,} legal candidate observations. The frozen Yaw8 encoder/shared head and Stage-A legal candidate pool were reused; Policy Test was not read and no model or perception data was generated.", "",
        "## Capacity table", "", "| Method | Acc | Macro-F1 | Within rho(TrueLogP) | Within rho(Margin) | HighOcc Acc | Verdict |", "|---|---:|---:|---:|---:|---:|---|",
    ]
    table_order = ["StaticPrior", "RGBGlobal", "Frame0SceneVisibility", *score_values.keys()]
    if fusion_method_name is not None:
        table_order.append(fusion_method_name)
    table_order.append("GT-TrueLogP Oracle")
    for name in table_order:
        item = all_metrics[name]
        if item.get("verdict") == "REFERENCE":
            lines.append(f"| {name} | {item.get('accuracy', 0.0):.6f} | {item.get('macro_f1', 0.0):.6f} | — | — | — | reference |")
        else:
            lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['within_context_spearman_true_logp']:.6f} | {item['within_context_spearman_gt_margin']:.6f} | {item['high_occlusion_accuracy']:.6f} | {item['verdict']} |")
    for name in ("F1 Matched Skeleton Error", "F2 Matched Feature Distance", "F3 Layerwise Feature Distance"):
        lines.append(f"| {name} | unavailable | unavailable | — | — | — | unavailable |")
    lines.extend([
        "", f"Reference Random selector: Acc {all_metrics['Random']['accuracy']:.6f}, Macro-F1 {all_metrics['Random']['macro_f1']:.6f}; AnyCorrect Coverage: {all_metrics['AnyCorrect Coverage']['coverage_rate']:.6f} ({all_metrics['AnyCorrect Coverage']['coverage_count']:,}/{len(moving_rows):,} contexts).",
        "", f"Best surrogate by Moving Accuracy: {best_name} ({eligible[best_name]['accuracy']:.6f})" if best_name else "No surrogate was evaluated.",
        f"Strongest within-context absolute Spearman among A–E: {strongest_rho:.6f}.",
        "", "## Matched clean-perception branch", "",
        "F1/F2/F3 were not scored: the existing recovery audit reports no exact per-candidate HM3D-vs-clean perception cache and a blocked Moving mapping. No clean rendering was attempted, so this branch is explicitly unavailable rather than approximated.", "",
        f"Final gate decision: **{final_decision}**.", "",
        "## Scientific answers", "",
        f"1. A >=65% action-independent surrogate: {'yes' if best_name and eligible[best_name]['accuracy'] >= 0.65 else 'no'}.",
        "2. The table separates training-manifold membership (A/B), skeleton plausibility (C), local recognizer robustness (D), cross-view consensus (E), and matched perception error (F unavailable).",
        f"3. A future Frame0 predictor target is warranted only if a surrogate passes the hard gate; in this run the registered decision is {final_decision}.",
        "4. If all A–E remain below 0.60, stop scalar-surrogate search and move to a low-cost candidate glimpse/active-probe information structure.",
        "", "Flags: `policy_test_used=false`; `training_used=false`; `new_rgb_generated=false`; `new_skeleton_generated=false`; `frozen_stgcn_modified=false`; `deployable=false`.",
    ])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
