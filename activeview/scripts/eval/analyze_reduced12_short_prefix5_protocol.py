#!/usr/bin/env python3
"""Audit the reduced12 short-prefix (L=5) mixed-view protocol on Moving Val.

This is a read-only, Val-only diagnostic.  A candidate action observes frames
0:5 at the current view and frames 5:30 at that candidate; the terminal
recognizer always consumes the resulting real archived skeleton.  The action
set is exactly Stay plus the Stage-A legal candidate pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    classification,
    correlation,
    gt_margin,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    load_rows,
)

SEED = 42
NUM_CLASSES = 12
NUM_VIEWS = 32
FEATURE_DIM = 256
FRAME_COUNT = 30
PREFIX = 5
MAX_OPTIONS = 22
INFERENCE_BATCH = 256
STGCN_RELATIVE = Path(
    "checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/"
    "stgcn_reduced12_no_kneel_clean_best.pth"
)
HEAD_RELATIVE = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "view_agnostic_frozen_encoder_head/shared_head_best.pth"
)
CACHE_RELATIVE = Path("diagnostics/reduced12_dual_route_overnight/val_options.npz")
VISIBILITY_RELATIVE = Path("diagnostics/frame0_visibility_predictor_v1/val.npz")
OUTPUT_RELATIVE = Path(
    "experiments/reduced12_eight_placement_v1/short_prefix5_protocol_feasibility"
)

# The canonical H36M-17 topology in configs/skeleton_definition_h36m.json.
H36M17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)


def _seed() -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_options_cache(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    cache = _load_npz(data_root / CACHE_RELATIVE)
    expected = (len(rows), MAX_OPTIONS)
    if cache["features"].shape != (len(rows), MAX_OPTIONS, FEATURE_DIM):
        raise ValueError(f"unexpected option feature shape {cache['features'].shape}; expected {expected}")
    if cache["logits"].shape != (len(rows), MAX_OPTIONS, NUM_CLASSES):
        raise ValueError(f"unexpected option logits shape {cache['logits'].shape}")
    if cache["ids"].shape != cache["mask"].shape or cache["ids"].shape != (len(rows), MAX_OPTIONS):
        raise ValueError("option id/mask shape mismatch")
    for index, row in enumerate(rows):
        options = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
        options = list(dict.fromkeys(options))
        valid = np.flatnonzero(cache["mask"][index]).astype(int)
        cached = cache["ids"][index, valid].astype(int).tolist()
        if cached != options:
            raise ValueError(f"candidate/action mismatch at row {index}: {cached} != {options}")
    return cache


def _load_head(data_root: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = data_root / HEAD_RELATIVE
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    head = SharedHead().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    return head, {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)}


def _apply_head(head: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIM)
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), INFERENCE_BATCH):
            batch = torch.from_numpy(flat[start : start + INFERENCE_BATCH]).to(device, non_blocking=True)
            output.append(head(batch).cpu().numpy())
    return np.concatenate(output).reshape(*features.shape[:-1], NUM_CLASSES).astype(np.float32)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)


def _infer_mixed(
    rows: Sequence[Mapping[str, Any]],
    model: nn.Module,
    head: nn.Module,
    device: torch.device,
    batch_rows: int = 128,
) -> np.ndarray:
    """Infer candidate mixed sequences, retaining only current first 5 frames."""
    mixed_logits = np.zeros((len(rows), MAX_OPTIONS, NUM_CLASSES), dtype=np.float32)
    for start in range(0, len(rows), batch_rows):
        group = rows[start : start + batch_rows]
        sequences: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for row in group:
            archive_path = Path(str(row["archive_path"]))
            if archive_path.stem != str(row["record_id"]):
                raise ValueError(f"archive record mismatch: {archive_path} vs {row['record_id']}")
            if archive_path.parent.name != str(row["region"]) or archive_path.parent.parent.name != str(row["scene_id"]):
                raise ValueError(f"archive scene/region mismatch at {row['episode_id']}")
            archive = _load_npz(archive_path)
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            if skeleton.shape != (NUM_VIEWS, 3, FRAME_COUNT, 17) or not np.isfinite(skeleton).all():
                raise ValueError(f"invalid skeleton archive {archive_path}")
            if not np.array_equal(ids, np.arange(NUM_VIEWS)):
                raise ValueError(f"invalid viewpoint ids in {archive_path}")
            current = int(row["current_viewpoint_id"])
            candidates = [int(x) for x in row["candidate_ids"]]
            offset = len(sequences)
            current_skeleton = skeleton[current]
            for candidate in candidates:
                mixed = np.array(current_skeleton, copy=True)
                mixed[:, PREFIX:] = skeleton[candidate, :, PREFIX:]
                sequences.append(mixed)
            spans.append((offset, len(candidates)))
        if sequences:
            encoded: list[np.ndarray] = []
            with torch.inference_mode():
                values = np.stack(sequences).astype(np.float32)
                for offset in range(0, len(values), INFERENCE_BATCH):
                    batch = torch.from_numpy(values[offset : offset + INFERENCE_BATCH]).to(device, non_blocking=True)
                    encoded.append(model.forward_features(batch).cpu().numpy())
            logits = _apply_head(head, np.concatenate(encoded, axis=0), device)
            for local, (offset, length) in enumerate(spans):
                mixed_logits[start + local, 1 : length + 1] = logits[offset : offset + length]
        if (start + len(group)) % 512 == 0 or start + len(group) == len(rows):
            print(f"[mixed-inference] contexts={start + len(group)}/{len(rows)}", flush=True)
    return mixed_logits


def _slot_map(ids: np.ndarray, mask: np.ndarray, row: int) -> dict[int, int]:
    return {int(value): int(slot) for slot, value in enumerate(ids[row]) if bool(mask[row, slot])}


def _actions_metric(
    labels: np.ndarray,
    logits: np.ndarray,
    ids: np.ndarray,
    mask: np.ndarray,
    actions: Sequence[int],
    name: str,
) -> dict[str, Any]:
    predictions: list[int] = []
    moves = 0
    for index, action in enumerate(actions):
        slots = _slot_map(ids, mask, index)
        if int(action) not in slots:
            raise ValueError(f"selected action {action} is not legal at row {index}")
        predictions.append(int(np.argmax(logits[index, slots[int(action)]])))
        if int(action) != int(ids[index, 0]):
            moves += 1
    result = classification(labels.tolist(), predictions)
    result.update({"method": name, "move_rate": float(moves / len(labels)), "stay_rate": 1.0 - float(moves / len(labels))})
    return result


def _oracle_actions(logits: np.ndarray, ids: np.ndarray, mask: np.ndarray, labels: np.ndarray, mode: str) -> list[int]:
    actions: list[int] = []
    for index, label in enumerate(labels):
        valid = np.flatnonzero(mask[index])
        values = logits[index, valid, int(label)]
        if mode == "margin":
            values = np.asarray([gt_margin(logits[index, slot], int(label)) for slot in valid])
        selected = int(valid[int(np.argmax(values))])
        actions.append(int(ids[index, selected]))
    return actions


def _any_correct(logits: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> tuple[int, float]:
    hits = [bool(np.any(np.argmax(logits[index, np.flatnonzero(mask[index])], axis=1) == int(label))) for index, label in enumerate(labels)]
    return int(sum(hits)), float(np.mean(hits))


def _random_actions(ids: np.ndarray, mask: np.ndarray, seed: int = SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    actions: list[int] = []
    for index in range(ids.shape[0]):
        valid = np.flatnonzero(mask[index])
        actions.append(int(rng.choice(ids[index, valid])))
    return actions


def _bone_lengths(frame: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(frame[left] - frame[right]) for left, right in H36M17_EDGES], dtype=np.float32)


def _transition_stats(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    displacement = np.linalg.norm(second - first, axis=1)
    bone_delta = np.abs(_bone_lengths(second) - _bone_lengths(first))
    return {
        "mean_joint_displacement": float(np.mean(displacement)),
        "max_joint_displacement": float(np.max(displacement)),
        "root_pelvis_displacement": float(displacement[0]),
        "mean_bone_length_discontinuity": float(np.mean(bone_delta)),
        "max_bone_length_discontinuity": float(np.max(bone_delta)),
    }


def _aggregate_stats(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("mean_joint_displacement", "max_joint_displacement", "root_pelvis_displacement", "mean_bone_length_discontinuity", "max_bone_length_discontinuity")}
    return {key: float(np.mean([item[key] for item in values])) for key in values[0]}


def _boundary_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normal: dict[str, list[dict[str, float]]] = {"3_to_4": [], "4_to_5": [], "5_to_6": []}
    switch: list[dict[str, float]] = []
    for row in rows:
        archive = _load_npz(Path(str(row["archive_path"])))
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        current = skeleton[int(row["current_viewpoint_id"])].transpose(1, 2, 0)
        for key, left, right in (("3_to_4", 3, 4), ("4_to_5", 4, 5), ("5_to_6", 5, 6)):
            normal[key].append(_transition_stats(current[left], current[right]))
        for candidate in row["candidate_ids"]:
            mixed = np.array(current, copy=True)
            candidate_sequence = skeleton[int(candidate)].transpose(1, 2, 0)
            mixed[5:] = candidate_sequence[5:]
            switch.append(_transition_stats(mixed[4], mixed[5]))
    normal_summary = {key: _aggregate_stats(values) for key, values in normal.items()}
    switch_summary = _aggregate_stats(switch)
    base = normal_summary["4_to_5"]["mean_joint_displacement"]
    return {
        "normal_same_view": normal_summary,
        "mixed_candidate_switch_4_to_5": switch_summary,
        "switch_to_normal_4_to_5_amplification": float(switch_summary["mean_joint_displacement"] / max(base, 1e-12)),
        "normal_transition_samples": {key: len(value) for key, value in normal.items()},
        "mixed_switch_samples": len(switch),
        "edges": [list(edge) for edge in H36M17_EDGES],
        "root_joint": 0,
    }


def _ranking_diagnostics(mixed: np.ndarray, full: np.ndarray, ids: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    spearman: list[float] = []
    top1: list[bool] = []
    top3: list[float] = []
    same_stay_move: list[bool] = []
    same_view: list[bool] = []
    for index, label in enumerate(labels):
        valid = np.flatnonzero(mask[index])
        mixed_score = mixed[index, valid, int(label)]
        full_score = full[index, valid, int(label)]
        spearman.append(correlation(mixed_score, full_score, spearman=True))
        mixed_order = valid[np.argsort(-mixed_score, kind="mergesort")]
        full_order = valid[np.argsort(-full_score, kind="mergesort")]
        top1.append(int(mixed_order[0]) == int(full_order[0]))
        top3.append(float(len(set(mixed_order[:3].tolist()) & set(full_order[:3].tolist()))) / min(3, len(valid)))
        same_view.append(int(ids[index, mixed_order[0]]) == int(ids[index, full_order[0]]))
        same_stay_move.append((int(mixed_order[0]) == 0) == (int(full_order[0]) == 0))
    return {
        "within_context_spearman_mean": float(np.mean(spearman)),
        "within_context_spearman_median": float(np.median(spearman)),
        "top1_slot_agreement": float(np.mean(top1)),
        "top1_viewpoint_agreement": float(np.mean(same_view)),
        "top3_overlap_mean": float(np.mean(top3)),
        "same_stay_move_decision": float(np.mean(same_stay_move)),
        "contexts": len(labels),
    }


def _candidate_transition_metrics(mixed: np.ndarray, full: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    counts = {key: 0 for key in ("full_correct_mixed_correct", "full_correct_mixed_wrong", "full_wrong_mixed_correct", "full_wrong_mixed_wrong")}
    logp_delta: list[float] = []
    margin_delta: list[float] = []
    for index, label in enumerate(labels):
        for slot in np.flatnonzero(mask[index])[1:]:
            full_correct = int(np.argmax(full[index, slot])) == int(label)
            mixed_correct = int(np.argmax(mixed[index, slot])) == int(label)
            key = f"full_{'correct' if full_correct else 'wrong'}_mixed_{'correct' if mixed_correct else 'wrong'}"
            counts[key] += 1
            logp_delta.append(float(mixed[index, slot, int(label)] - full[index, slot, int(label)]))
            margin_delta.append(float(gt_margin(mixed[index, slot], int(label)) - gt_margin(full[index, slot], int(label))))
    total = sum(counts.values())
    full_correct = counts["full_correct_mixed_correct"] + counts["full_correct_mixed_wrong"]
    full_wrong = counts["full_wrong_mixed_correct"] + counts["full_wrong_mixed_wrong"]
    return {"candidate_samples": total, "counts": counts, "ratios": {key: float(value / max(total, 1)) for key, value in counts.items()}, "conditional_ratios": {"mixed_wrong_given_full_correct": float(counts["full_correct_mixed_wrong"] / max(full_correct, 1)), "mixed_correct_given_full_wrong": float(counts["full_wrong_mixed_correct"] / max(full_wrong, 1))}, "mean_true_logp_mixed_minus_full": float(np.mean(logp_delta)), "mean_margin_mixed_minus_full": float(np.mean(margin_delta))}


def _per_class(metrics: Mapping[str, Mapping[str, Any]], methods: Sequence[str]) -> dict[str, Any]:
    return {label: {method: metrics[method].get("per_class", {}).get(label, {}) for method in methods} for label in LABELS}


def _high_occlusion(rows: Sequence[Mapping[str, Any]], data_root: Path, labels: np.ndarray, mixed: np.ndarray, full: np.ndarray, ids: np.ndarray, mask: np.ndarray, random_actions: Sequence[int]) -> dict[str, Any]:
    visibility = _load_npz(data_root / VISIBILITY_RELATIVE)
    if visibility["scores"].shape[1:] != (MAX_OPTIONS,) or visibility["scores"].shape[0] < len(rows):
        raise ValueError("frame0 visibility target has incompatible shape")
    visibility_ids = visibility["ids"][: len(rows)]
    visibility_mask = visibility["mask"][: len(rows)]
    if not np.array_equal(visibility_ids, ids) or not np.array_equal(visibility_mask, mask):
        raise ValueError("frame0 visibility target is not aligned with Val action rows")
    target = np.asarray(visibility["scores"][: len(rows), 0], dtype=np.float32)
    cutoff = float(np.quantile(target, 1.0 / 3.0))
    subset = np.flatnonzero(target <= cutoff)
    sub_labels = labels[subset]
    sub_ids, sub_mask = ids[subset], mask[subset]
    sub_mixed, sub_full = mixed[subset], full[subset]
    mixed_true = _oracle_actions(sub_mixed, sub_ids, sub_mask, sub_labels, "true_logp")
    full_true = _oracle_actions(sub_full, sub_ids, sub_mask, sub_labels, "true_logp")
    mixed_hits, mixed_rate = _any_correct(sub_mixed, sub_mask, sub_labels)
    return {"definition": "bottom tertile of frame-0 Stay SceneVisibility", "cutoff": cutoff, "contexts": int(len(subset)), "metrics": {"Stay": _actions_metric(sub_labels, sub_full, sub_ids, sub_mask, [int(x) for x in sub_ids[:, 0]], "Stay"), "Random-ShortPrefix5": _actions_metric(sub_labels, sub_mixed, sub_ids, sub_mask, [int(x) for x in np.asarray(random_actions)[subset]], "Random-ShortPrefix5"), "Mixed5-GT-TrueLogP": _actions_metric(sub_labels, sub_mixed, sub_ids, sub_mask, mixed_true, "Mixed5-GT-TrueLogP"), "FullView-GT-TrueLogP": _actions_metric(sub_labels, sub_full, sub_ids, sub_mask, full_true, "FullView-GT-TrueLogP"), "Mixed5-AnyCorrect Coverage": {"count": mixed_hits, "coverage": mixed_rate}}, "test_used": False}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    data_root = get_data_root()
    device = _device(args.device)
    started = time.time()
    train_rows, val_rows = load_rows(data_root)
    if args.limit is not None:
        val_rows = val_rows[: int(args.limit)]
    cache = _load_options_cache(data_root, val_rows if args.limit is None else load_rows(data_root)[1])
    if args.limit is not None:
        cache = {key: value[: len(val_rows)] for key, value in cache.items()}
    labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    model, _ = load_checkpoint(data_root / STGCN_RELATIVE, NUM_CLASSES, str(device))
    model.eval()
    head, head_info = _load_head(data_root, device)
    full_logits = _apply_head(head, cache["features"], device)
    mixed_logits = _infer_mixed(val_rows, model, head, device)
    # Stay is the unchanged current-view 30-frame sequence, not a zero-filled
    # mixed sequence.  Reuse the same full-view recognizer output so every
    # selector compares Stay and candidates on the identical action set.
    mixed_logits[:, 0] = full_logits[:, 0]
    full_logp = _log_softmax(full_logits)
    mixed_logp = _log_softmax(mixed_logits)
    ids, mask = np.asarray(cache["ids"]), np.asarray(cache["mask"], dtype=bool)
    stay_actions = ids[:, 0].astype(int).tolist()
    random_actions = _random_actions(ids, mask)
    methods: dict[str, dict[str, Any]] = {}
    methods["Stay"] = _actions_metric(labels, full_logp, ids, mask, stay_actions, "Stay")
    methods["Random-ShortPrefix5"] = _actions_metric(labels, mixed_logp, ids, mask, random_actions, "Random-ShortPrefix5")
    mixed_true_actions = _oracle_actions(mixed_logp, ids, mask, labels, "true_logp")
    mixed_margin_actions = _oracle_actions(mixed_logp, ids, mask, labels, "margin")
    full_true_actions = _oracle_actions(full_logp, ids, mask, labels, "true_logp")
    full_margin_actions = _oracle_actions(full_logp, ids, mask, labels, "margin")
    methods["Mixed5-GT-TrueLogP"] = _actions_metric(labels, mixed_logp, ids, mask, mixed_true_actions, "Mixed5-GT-TrueLogP")
    methods["Mixed5-GT-Margin"] = _actions_metric(labels, mixed_logp, ids, mask, mixed_margin_actions, "Mixed5-GT-Margin")
    methods["FullView-GT-TrueLogP"] = _actions_metric(labels, full_logp, ids, mask, full_true_actions, "FullView-GT-TrueLogP")
    methods["FullView-GT-Margin"] = _actions_metric(labels, full_logp, ids, mask, full_margin_actions, "FullView-GT-Margin")
    stay_acc = methods["Stay"]["accuracy"]
    if abs(stay_acc - 0.30257936507936506) > 1e-6 and args.limit is None:
        raise RuntimeError(f"Stay baseline mismatch: {stay_acc:.9f} (expected 0.302579365)")
    mixed_any_count, mixed_any_rate = _any_correct(mixed_logp, mask, labels)
    full_any_count, full_any_rate = _any_correct(full_logp, mask, labels)
    candidate_metrics = _candidate_transition_metrics(mixed_logp, full_logp, mask, labels)
    ranking = _ranking_diagnostics(mixed_logp, full_logp, ids, mask, labels)
    boundary = _boundary_audit(val_rows)
    high_occ = _high_occlusion(val_rows, data_root, labels, mixed_logp, full_logp, ids, mask, random_actions)
    methods["Mixed5-AnyCorrect Coverage"] = {"contexts": len(labels), "count": mixed_any_count, "coverage": mixed_any_rate, "deployable": False}
    methods["FullView-AnyCorrect Coverage"] = {"contexts": len(labels), "count": full_any_count, "coverage": full_any_rate, "deployable": False}
    metrics_for_classes = {name: value for name, value in methods.items() if "per_class" in value}
    output = data_root.parent / "dummy" if False else REPO_ROOT / OUTPUT_RELATIVE
    _write(output / "config.json", {"split": "Moving Val only", "contexts": len(val_rows), "prefix_frames": "current[0:5]+candidate[5:30]", "stay_frames": "current[0:30]", "formal_action_set": "Stay + Stage-A legal candidate_pool", "all32_formal": False, "recognizer": "frozen reduced12 ST-GCN + frozen shared adapted head", "checkpoint": {"stgcn": str((data_root / STGCN_RELATIVE).resolve()), "stgcn_sha256": _sha256(data_root / STGCN_RELATIVE), "shared_head": head_info}, "seed": SEED, "test_used": False, "training_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "discrete_time_view_switch_approximation": True, "continuous_navigation_claimed": False})
    _write(output / "per_method_metrics.json", methods)
    _write(output / "per_class_metrics.json", _per_class(metrics_for_classes, list(metrics_for_classes)))
    _write(output / "switch_boundary_audit.json", boundary)
    _write(output / "candidate_transition_metrics.json", candidate_metrics)
    _write(output / "ranking_diagnostics.json", ranking)
    _write(output / "occlusion_stratified_metrics.json", high_occ)
    _write(output / "coverage_audit.json", {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "candidate_count_min": int(np.min(mask[:, 1:].sum(axis=1))), "candidate_count_max": int(np.max(mask[:, 1:].sum(axis=1))), "candidate_count_mean": float(np.mean(mask[:, 1:].sum(axis=1))), "legal_action_count_min": int(np.min(mask.sum(axis=1))), "legal_action_count_max": int(np.max(mask.sum(axis=1))), "legal_action_count_mean": float(np.mean(mask.sum(axis=1))), "archive_shape_checked": [NUM_VIEWS, 3, FRAME_COUNT, 17], "action_set_identity_checked": True, "mixed_any_correct_count": mixed_any_count, "mixed_any_correct_coverage": mixed_any_rate, "full_any_correct_count": full_any_count, "full_any_correct_coverage": full_any_rate, "test_used": False})
    result = {"protocol": {"split": "Moving Val only", "train_contexts_loaded": len(train_rows), "contexts": len(val_rows), "prefix": "current frames 0..4 then selected candidate frames 5..29", "stay": "current frames 0..29", "formal_action_set": "Stay/current + Stage-A legal candidate_pool", "interpretation": "discrete-time view-switch approximation; not continuous navigation"}, "metrics": methods, "mixed5_any_correct": {"count": mixed_any_count, "coverage": mixed_any_rate}, "fullview_any_correct": {"count": full_any_count, "coverage": full_any_rate}, "ceiling_drop": {"accuracy": float(methods["FullView-GT-TrueLogP"]["accuracy"] - methods["Mixed5-GT-TrueLogP"]["accuracy"]), "macro_f1": float(methods["FullView-GT-TrueLogP"]["macro_f1"] - methods["Mixed5-GT-TrueLogP"]["macro_f1"])}, "candidate_transition": candidate_metrics, "ranking": ranking, "boundary": boundary, "high_occlusion": high_occ, "flags": {"policy_test_used": False, "training_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "future_candidate_skeleton_used_only_for_terminal_evaluation": True, "gt_label_used_for_oracle_only": True, "deployable": False}, "elapsed_seconds": float(time.time() - started)}
    _write(output / "result.json", result)
    _write_analysis(output / "analysis.md", result)
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]
    lines = [
        "# Short-prefix5 mixed-view protocol feasibility audit", "",
        "Split: Moving Val only; contexts: 10,080 when run without `--limit`; training: none; Policy Test: false.",
        "Recognizer: frozen reduced12 ST-GCN plus frozen shared adapted head. Prefix5 uses current frames 0..4 and selected candidate frames 5..29; Stay uses current frames 0..29.",
        "This is a discrete-time view-switch approximation, not continuous navigation.", "",
        "## Results", "", "| Method | Accuracy | Macro-F1 | Move rate |", "|---|---:|---:|---:|",
    ]
    for name in ("Stay", "Random-ShortPrefix5", "Mixed5-GT-TrueLogP", "Mixed5-GT-Margin", "FullView-GT-TrueLogP", "FullView-GT-Margin"):
        value = metrics[name]
        lines.append(f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} | {value.get('move_rate', 0.0):.6f} |")
    lines.extend([
        f"| Mixed5 AnyCorrect Coverage | {result['mixed5_any_correct']['coverage']:.6f} | — | — |",
        f"| FullView AnyCorrect Coverage | {result['fullview_any_correct']['coverage']:.6f} | — | — |", "",
        f"Mixed5-to-FullView GT-TrueLogP ceiling drop: {result['ceiling_drop']['accuracy']:.6f} Acc / {result['ceiling_drop']['macro_f1']:.6f} F1.",
        f"Candidate ranking: within-context Spearman mean/median {result['ranking']['within_context_spearman_mean']:.6f}/{result['ranking']['within_context_spearman_median']:.6f}; top-1 viewpoint agreement {result['ranking']['top1_viewpoint_agreement']:.6f}; top-3 overlap {result['ranking']['top3_overlap_mean']:.6f}.",
        f"Full-correct→mixed-wrong: {result['candidate_transition']['conditional_ratios']['mixed_wrong_given_full_correct']:.6f} conditional ({result['candidate_transition']['ratios']['full_correct_mixed_wrong']:.6f} of all candidate samples); switch/normal 4→5 mean displacement amplification: {result['boundary']['switch_to_normal_4_to_5_amplification']:.6f}.", "",
        "## High-occlusion subset", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|",
    ])
    for name in ("Stay", "Random-ShortPrefix5", "Mixed5-GT-TrueLogP", "FullView-GT-TrueLogP"):
        value = result["high_occlusion"]["metrics"][name]
        lines.append(f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} |")
    lines.append(f"| Mixed5 AnyCorrect Coverage | {result['high_occlusion']['metrics']['Mixed5-AnyCorrect Coverage']['coverage']:.6f} | — |")
    mixed_class = metrics["Mixed5-GT-TrueLogP"]["per_class"]
    full_class = metrics["FullView-GT-TrueLogP"]["per_class"]
    drops = sorted(((float(full_class[label]["f1"] - mixed_class[label]["f1"]), label) for label in LABELS), reverse=True)
    lines.extend(["", "Largest FullView→Mixed5 GT-TrueLogP per-class F1 drops: " + ", ".join(f"{label} {drop:.6f}" for drop, label in drops[:4]) + ".", "", "## Decision", ""])
    mixed_acc = float(metrics["Mixed5-GT-TrueLogP"]["accuracy"])
    drop = float(result["ceiling_drop"]["accuracy"])
    if mixed_acc < 0.58 or drop >= 0.15:
        decision = "KILL"
    elif mixed_acc < 0.65:
        decision = "BORDERLINE"
    elif mixed_acc >= 0.70 and drop < 0.10:
        decision = "STRONG KEEP"
    else:
        decision = "KEEP"
    lines.append(f"Decision: **{decision}** under the preregistered thresholds (Mixed5 oracle Acc={mixed_acc:.6f}, FullView−Mixed5={drop:.6f}).")
    lines.append("A subsequent selector-training experiment is allowed only as a separately authorized step; this audit itself trained no selector.")
    if result["boundary"]["switch_to_normal_4_to_5_amplification"] > 2.0:
        lines.append("The mixed-view boundary is substantially larger than normal same-view transitions; if the short-prefix ceiling is low, representation discontinuity is a plausible protocol ceiling.")
    lines.extend(["", "The candidate skeleton is used only to construct the privileged mixed terminal evaluation and oracle targets. No future candidate observation is available to a deployable selector in this audit.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="debug-only Val prefix; full audit omits this")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
