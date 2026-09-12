#!/usr/bin/env python3
"""Val-only causal mixed-view oracle sweep for several prefix lengths.

For a legal candidate, the only mixed sequence evaluated is
``current[0:L] + candidate[L:30]``.  Stay is always ``current[0:30]``.
This script trains nothing and never opens Policy Test; it records oracle
ceilings and protocol diagnostics for L=5, 8, 10, 12 and 15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation, gt_margin
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    head_logits,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import _option_geometry

SEED = 42
NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32
MAX_OPTIONS = 22
FRAME_COUNT = 30
FEATURE_DIM = 256
PREFIX_LENGTHS = (5, 8, 10, 12, 15)
INFERENCE_BATCH = 2048
OPTION_GEOMETRY_DIM = 18

POLICY_RELATIVE = Path("datasets/policy_reduced12_eight_placement_v1")
DIAGNOSTIC_RELATIVE = Path("diagnostics/reduced12_dual_route_overnight")
VISIBILITY_RELATIVE = Path("diagnostics/frame0_visibility_predictor_v1/val.npz")
STGCN_RELATIVE = Path("checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
SHARED_HEAD_RELATIVE = Path("checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth")
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/short_prefix_length_oracle_sweep"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _log_softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - array.max(axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)


def _action_set(row: Mapping[str, Any]) -> list[int]:
    return list(dict.fromkeys([int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]))


def _load_option_cache(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str, head: SharedHead, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = data_root / DIAGNOSTIC_RELATIVE
    metadata = json.loads((root / f"{split}_options.json").read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows):
        raise ValueError(f"{split} option cache signature mismatch")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"{split} option cache is marked as Test-derived")
    with np.load(root / f"{split}_options.npz", allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        ids = np.asarray(archive["ids"], dtype=np.int64)
        mask = np.asarray(archive["mask"], dtype=bool)
    if features.shape != (len(rows), MAX_OPTIONS, FEATURE_DIM) or ids.shape != mask.shape or ids.shape != (len(rows), MAX_OPTIONS):
        raise ValueError(f"invalid {split} option cache shape")
    for index, row in enumerate(rows):
        expected = _action_set(row)
        valid = np.flatnonzero(mask[index]).tolist()
        if ids[index, valid].tolist() != expected:
            raise ValueError(f"{split} action set mismatch at {row['episode_id']}")
    logits = head_logits(head, features, device)
    return _log_softmax(logits), ids, mask


def _load_all32(data_root: Path, rows: Sequence[Mapping[str, Any]], head: SharedHead, device: torch.device) -> np.ndarray:
    root = data_root / DIAGNOSTIC_RELATIVE
    metadata = json.loads((root / "val_all32.json").read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != row_signature(rows) or not bool(metadata.get("all32")):
        raise ValueError("Val all32 cache metadata mismatch")
    if bool(metadata.get("test_used", True)):
        raise ValueError("all32 cache is marked as Test-derived")
    with np.load(root / "val_all32.npz", allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        ids = np.asarray(archive["ids"], dtype=np.int64)
        mask = np.asarray(archive["mask"], dtype=bool)
    if features.shape != (len(rows), NUM_VIEWS, FEATURE_DIM) or ids.shape != (len(rows), NUM_VIEWS) or mask.shape != ids.shape or not np.all(mask):
        raise ValueError("invalid Val all32 cache shape")
    if not np.array_equal(ids, np.tile(np.arange(NUM_VIEWS, dtype=np.int64), (len(rows), 1))):
        raise ValueError("all32 viewpoint identity mismatch")
    return _log_softmax(head_logits(head, features, device))


def _load_current_skeletons(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    output = np.empty((len(rows), 3, FRAME_COUNT, 17), dtype=np.float32)
    for index, row in enumerate(rows):
        path = Path(str(row["archive_path"]))
        if path.stem != str(row["record_id"]) or path.parent.name != str(row["region"]) or path.parent.parent.name != str(row["scene_id"]):
            raise ValueError(f"archive identity mismatch at {row['episode_id']}: {path}")
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            if ids.shape != (NUM_VIEWS,) or not np.array_equal(ids, np.arange(NUM_VIEWS)) or skeleton.shape != (NUM_VIEWS, 3, FRAME_COUNT, 17) or not np.isfinite(skeleton).all():
                raise ValueError(f"invalid skeleton archive {path}")
            output[index] = skeleton[int(row["current_viewpoint_id"])]
    return output


def _infer_mixed_length(rows: Sequence[Mapping[str, Any]], current: np.ndarray, prefix_len: int, recognizer: nn.Module, head: nn.Module, device: torch.device) -> tuple[np.ndarray, list[float], list[float]]:
    logits = np.zeros((len(rows), MAX_OPTIONS, NUM_CLASSES), dtype=np.float32)
    switch_jumps: list[float] = []
    normal_jumps: list[float] = []
    for start in range(0, len(rows), 128):
        group = rows[start : start + 128]
        sequences: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for local, row in enumerate(group):
            path = Path(str(row["archive_path"]))
            with np.load(path, allow_pickle=False) as archive:
                ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                values = np.asarray(archive["skeleton"], dtype=np.float32)
                if ids.shape != (NUM_VIEWS,) or not np.array_equal(ids, np.arange(NUM_VIEWS)) or values.shape != (NUM_VIEWS, 3, FRAME_COUNT, 17) or not np.isfinite(values).all():
                    raise ValueError(f"invalid skeleton archive {path}")
                current_view = int(row["current_viewpoint_id"])
                if not np.allclose(values[current_view], current[start + local], rtol=1e-5, atol=1e-6):
                    raise ValueError(f"current/candidate skeleton identity mismatch at {row['episode_id']}")
                candidates = [int(value) for value in row["candidate_ids"]]
                offset = len(sequences)
                for candidate in candidates:
                    mixed = np.array(current[start + local], copy=True)
                    mixed[:, prefix_len:] = values[candidate, :, prefix_len:]
                    sequences.append(mixed)
                    if prefix_len < FRAME_COUNT:
                        switch_jump = float(np.linalg.norm(current[start + local][:, prefix_len - 1] - values[candidate, :, prefix_len], axis=0).mean())
                        normal_jump = float(np.linalg.norm(current[start + local][:, prefix_len - 1] - current[start + local][:, prefix_len], axis=0).mean())
                        switch_jumps.append(switch_jump)
                        normal_jumps.append(normal_jump)
                spans.append((offset, len(candidates)))
        if sequences:
            encoded: list[np.ndarray] = []
            sequence_values = np.stack(sequences).astype(np.float32)
            with torch.inference_mode():
                for offset in range(0, len(sequence_values), INFERENCE_BATCH):
                    batch = torch.from_numpy(sequence_values[offset : offset + INFERENCE_BATCH]).to(device, non_blocking=True)
                    encoded.append(recognizer.forward_features(batch).cpu().numpy())
            sequence_logits = head_logits(head, np.concatenate(encoded, axis=0), device)
            for local, (offset, length) in enumerate(spans):
                logits[start + local, 1 : length + 1] = sequence_logits[offset : offset + length]
        if (start + len(group)) % 512 == 0 or start + len(group) == len(rows):
            print(f"[mixed-L{prefix_len}] contexts={start + len(group)}/{len(rows)}", flush=True)
    return logits, switch_jumps, normal_jumps


def _utilities(logp: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    utility = np.zeros(logp.shape[:2], dtype=np.float32)
    margin = np.zeros_like(utility)
    for index, label in enumerate(labels):
        for slot in np.flatnonzero(mask[index]):
            utility[index, slot] = float(logp[index, slot, int(label)])
            margin[index, slot] = gt_margin(logp[index, slot], int(label))
    return utility, margin


def _select(scores: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> list[int]:
    selected: list[int] = []
    for index in range(scores.shape[0]):
        active = np.flatnonzero(mask[index])
        ordered = sorted(active.tolist(), key=lambda slot: (-float(scores[index, slot]), int(ids[index, slot])))
        selected.append(int(ids[index, ordered[0]]))
    return selected


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    output: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((ids[index] == int(action)) & mask[index])
        if slots.size != 1:
            raise ValueError(f"selected action {action} is not legal at context {index}")
        output.append(int(np.argmax(logp[index, int(slots[0])])))
    return np.asarray(output, dtype=np.int64)


def _metrics(name: str, rows: Sequence[Mapping[str, Any]], labels: np.ndarray, predictions: np.ndarray, actions: Sequence[int]) -> dict[str, Any]:
    move = float(np.mean([int(action) != int(row["current_viewpoint_id"]) for action, row in zip(actions, rows)]))
    result = classification(labels, predictions)
    result.update({"method": name, "move_rate": move, "stay_rate": 1.0 - move, "test_used": False})
    return result


def _ranking(values: np.ndarray, reference: np.ndarray, ids: np.ndarray, mask: np.ndarray, candidate_only: bool = True) -> dict[str, float]:
    within: list[float] = []
    top1: list[int] = []
    top3: list[int] = []
    for index in range(values.shape[0]):
        active = np.flatnonzero(mask[index])
        if candidate_only:
            active = active[1:]
        if active.size < 2:
            continue
        within.append(correlation(values[index, active], reference[index, active], spearman=True))
        left = active[np.argsort(-values[index, active], kind="mergesort")]
        right = active[np.argsort(-reference[index, active], kind="mergesort")]
        top1.append(int(left[0] == right[0]))
        top3.append(int(right[0] in left[:3]))
    return {"within_context_spearman_mean": float(np.mean(within)), "within_context_spearman_median": float(np.median(within)), "top1_selected_action_agreement": float(np.mean(top1)), "top3_overlap": float(np.mean(top3)), "contexts_with_two_candidates": int(len(within))}


def _motion_evidence(current: np.ndarray, prefix_len: int) -> dict[str, float]:
    prefix = current[:, :, :prefix_len]
    velocity = np.diff(prefix, axis=2)
    velocity_mag = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=2)
    acceleration_mag = np.linalg.norm(acceleration, axis=1)
    return {"prefix_frames": prefix_len, "mean_joint_velocity_magnitude": float(velocity_mag.mean()), "mean_joint_acceleration_magnitude": float(acceleration_mag.mean()) if acceleration_mag.size else 0.0, "mean_total_displacement": float(velocity_mag.sum(axis=(1, 2)).mean()), "mean_joint_temporal_variance": float(prefix.var(axis=2).mean()), "remaining_candidate_frames": FRAME_COUNT - prefix_len}


def _load_high_occlusion(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    with np.load(data_root / VISIBILITY_RELATIVE, allow_pickle=False) as archive:
        scores = np.asarray(archive["scores"])
        ids = np.asarray(archive["ids"])
        mask = np.asarray(archive["mask"])
    if scores.shape != (len(rows), MAX_OPTIONS) or ids.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("frame-0 visibility cache shape mismatch")
    return scores[:, 0] < np.quantile(scores[:, 0], 1.0 / 3.0)


def _analysis(result: Mapping[str, Any]) -> str:
    sweep = result["prefix_sweep_metrics"]
    oracle_acc = {int(length): float(values["true_logp_oracle"]["accuracy"]) for length, values in sweep.items()}
    qualifying = [length for length in sorted(oracle_acc) if oracle_acc[length] >= 0.65 and values_drop(result, length) < 0.10]
    longest = max(qualifying) if qualifying else None
    l10_note = "viable next selector length" if oracle_acc.get(10, 0.0) >= 0.65 else "below the 0.65 viability threshold"
    if not qualifying:
        decision = "KILL longer-prefix route"
    else:
        decision = f"KEEP longer-prefix family; longest qualifying prefix is L={longest}"
    lines = ["# Causal short-prefix length oracle sweep", "", "Split: Moving Val", "Contexts: 10080", "Training: none", "Policy Test: false", "Prefix lengths: 5,8,10,12,15", "Recognizer: frozen ST-GCN + frozen shared head", "Action set: Stay + Stage-A legal candidates", "Mixed protocol: current[0:L] + candidate[L:30]", "Stay: current[0:30]", "Continuous navigation: not modeled", "Protocol: discrete-time view-switch approximation", "", "| L | Random Acc/F1 | TrueLogP Acc/F1 | Margin/AnyCorrect Acc/F1 | Drop vs FullView |", "|---:|---:|---:|---:|---:|"]
    for length in PREFIX_LENGTHS:
        values = sweep[str(length)]
        lines.append(f"| {length} | {values['random']['accuracy']:.6f}/{values['random']['macro_f1']:.6f} | {values['true_logp_oracle']['accuracy']:.6f}/{values['true_logp_oracle']['macro_f1']:.6f} | {values['margin_oracle']['accuracy']:.6f}/{values['margin_oracle']['macro_f1']:.6f} | {values_drop(result, length):.6f} |")
    lines += ["", f"FullView GT-TrueLogP Oracle: {result['fullview']['true_logp_oracle']['accuracy']:.6f}/{result['fullview']['true_logp_oracle']['macro_f1']:.6f}.", f"L10 is {l10_note}.", f"High-occlusion best oracle prefix: L={result['high_occlusion']['best_length']}.", f"Recommendation: **{decision}**.", "", "The prefix trade-off is explicit: longer L adds current action evidence but leaves fewer candidate frames (30−L) for the mixed observation.", "Per-class best lengths and boundary amplification are diagnostic only; no deployment policy was selected per class.", "", "```text", "policy_test_used=false", "training_used=false", "new_rgb_generated=false", "new_skeleton_generated=false", "frozen_stgcn_modified=false", "gt_label_used_for_oracle_only=true", "deployable=false", "```", ""]
    return "\n".join(lines)


def values_drop(result: Mapping[str, Any], length: int) -> float:
    return float(result["fullview"]["true_logp_oracle"]["accuracy"] - result["prefix_sweep_metrics"][str(length)]["true_logp_oracle"]["accuracy"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    _, val_rows = load_rows(data_root)
    stgcn_path = data_root / STGCN_RELATIVE
    head_path = data_root / SHARED_HEAD_RELATIVE
    recognizer, _ = load_checkpoint(stgcn_path, NUM_CLASSES, str(device))
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    current_logp, val_ids, val_mask = _load_option_cache(data_root, val_rows, "val", head, device)
    labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    current_skeletons = _load_current_skeletons(val_rows)
    stats = json.loads((data_root / POLICY_RELATIVE / "stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8"))
    val_geometry, geometry_ids, geometry_mask = _option_geometry(val_rows, stats)
    if not np.array_equal(val_ids, geometry_ids) or not np.array_equal(val_mask, geometry_mask):
        raise ValueError("Stage-A geometry action set differs from frozen recognizer cache")
    full_logp = _load_all32(data_root, val_rows, head, device)
    full_utility = np.zeros((len(val_rows), MAX_OPTIONS), dtype=np.float32)
    full_margin = np.zeros_like(full_utility)
    for index, label in enumerate(labels):
        for slot in np.flatnonzero(val_mask[index]):
            viewpoint = int(val_ids[index, slot])
            full_utility[index, slot] = full_logp[index, viewpoint, int(label)]
            full_margin[index, slot] = gt_margin(full_logp[index, viewpoint], int(label))
    full_oracle_actions = _select(full_utility, val_ids, val_mask)
    # The all32 cache is indexed directly by viewpoint, so assemble a compact
    # legal-action view before terminal lookup.
    legal_full_logp = np.zeros((len(val_rows), MAX_OPTIONS, NUM_CLASSES), dtype=np.float32)
    for index in range(len(val_rows)):
        for slot in np.flatnonzero(val_mask[index]):
            legal_full_logp[index, slot] = full_logp[index, int(val_ids[index, slot])]
    full_oracle_pred = _terminal(legal_full_logp, val_ids, val_mask, full_oracle_actions)
    full_metric = _metrics("FullView GT-TrueLogP Oracle", val_rows, labels, full_oracle_pred, full_oracle_actions)
    high_mask = _load_high_occlusion(data_root, val_rows)
    sweep: dict[str, Any] = {}
    ranking: dict[str, Any] = {}
    boundary: dict[str, Any] = {}
    motion: dict[str, Any] = {}
    utility_by_length: dict[int, np.ndarray] = {}
    action_by_length: dict[int, list[int]] = {}
    high_oracle_acc: dict[int, float] = {}
    for length in PREFIX_LENGTHS:
        raw_logits, switch_jumps, normal_jumps = _infer_mixed_length(val_rows, current_skeletons, length, recognizer, head, device)
        mixed_logp = _log_softmax(raw_logits)
        mixed_logp[:, 0] = current_logp[:, 0]
        mixed_logp = _log_softmax(mixed_logp)
        utility, margin = _utilities(mixed_logp, labels, val_mask)
        utility_by_length[length] = utility
        true_actions = _select(utility, val_ids, val_mask)
        margin_actions = _select(margin, val_ids, val_mask)
        rng = np.random.default_rng(SEED if length == 5 else SEED + length)
        random_actions = [int(val_ids[index, int(rng.choice(np.flatnonzero(val_mask[index]).tolist()))]) for index in range(len(val_rows))]
        true_pred = _terminal(mixed_logp, val_ids, val_mask, true_actions)
        margin_pred = _terminal(mixed_logp, val_ids, val_mask, margin_actions)
        random_pred = _terminal(mixed_logp, val_ids, val_mask, random_actions)
        true_metric = _metrics(f"Mixed{length} GT-TrueLogP Oracle", val_rows, labels, true_pred, true_actions)
        margin_metric = _metrics(f"Mixed{length} GT-Margin/AnyCorrect", val_rows, labels, margin_pred, margin_actions)
        random_metric = _metrics(f"Random-Mixed-{length}", val_rows, labels, random_pred, random_actions)
        any_correct = float(np.mean([
            np.any(np.argmax(mixed_logp[index, np.flatnonzero(val_mask[index])], axis=1) == labels[index])
            for index in range(len(val_rows))
        ]))
        sweep[str(length)] = {"random": random_metric, "true_logp_oracle": true_metric, "margin_oracle": margin_metric, "any_correct_coverage": any_correct, "margin_anycorrect_difference": float(margin_metric["accuracy"] - any_correct), "remaining_candidate_frames": FRAME_COUNT - length}
        action_by_length[length] = true_actions
        high_indices = np.flatnonzero(high_mask)
        high_true = _metrics(f"Mixed{length} high-occlusion oracle", [val_rows[int(index)] for index in high_indices], labels[high_mask], true_pred[high_mask], [true_actions[int(index)] for index in high_indices])
        high_oracle_acc[length] = float(high_true["accuracy"])
        candidate_mask = np.array(val_mask, copy=True)
        candidate_mask[:, 0] = False
        ranking[str(length)] = {"vs_fullview": _ranking(utility, full_utility, val_ids, val_mask), "candidate_utility_spearman_vs_fullview": correlation(utility[candidate_mask], full_utility[candidate_mask], spearman=True)}
        ratios = np.asarray(switch_jumps, dtype=np.float64) / np.maximum(np.asarray(normal_jumps, dtype=np.float64), 1e-8)
        finite_ratios = ratios[np.asarray(normal_jumps, dtype=np.float64) > 1e-5]
        boundary[str(length)] = {"sample_count": len(switch_jumps), "mean_switch_displacement": float(np.mean(switch_jumps)), "mean_normal_displacement": float(np.mean(normal_jumps)), "mean_amplification": float(np.mean(switch_jumps) / max(float(np.mean(normal_jumps)), 1e-8)), "median_amplification": float(np.median(finite_ratios)) if finite_ratios.size else 0.0, "ratio_samples_with_nontrivial_normal": int(finite_ratios.size), "normalization": "aggregate candidate mixed boundary jump / aggregate same-view current jump; median uses nontrivial denominators", "prefix_boundary": f"frame {length - 1} -> {length}"}
        motion[str(length)] = _motion_evidence(current_skeletons, length)
    for left, right in zip(PREFIX_LENGTHS[:-1], PREFIX_LENGTHS[1:]):
        ranking[f"L{left}_vs_L{right}"] = _ranking(utility_by_length[left], utility_by_length[right], val_ids, val_mask)
    ranking["definition"] = "legal candidate utility rankings exclude Stay slot; FullView reference uses candidate[0:30]"
    full_action_stability: dict[str, Any] = {}
    for left, right in ((5, 10), (5, 15), (10, 15), (10, "FullView"), (15, "FullView")):
        right_actions = full_oracle_actions if right == "FullView" else action_by_length[int(right)]
        left_actions = action_by_length[int(left)]
        full_action_stability[f"L{left}_vs_{right}"] = {"same_viewpoint_fraction": float(np.mean(np.asarray(left_actions) == np.asarray(right_actions))), "same_stay_move_fraction": float(np.mean(np.asarray([int(a) == int(r["current_viewpoint_id"]) for a, r in zip(left_actions, val_rows)]) == np.asarray([int(a) == int(r["current_viewpoint_id"]) for a, r in zip(right_actions, val_rows)])))}
    per_class: dict[str, Any] = {}
    for length in PREFIX_LENGTHS:
        per_class[str(length)] = sweep[str(length)]["true_logp_oracle"]["per_class"]
    best_per_class = {label: max(PREFIX_LENGTHS, key=lambda length: float(per_class[str(length)][label]["f1"])) for label in LABELS}
    high_best = max(PREFIX_LENGTHS, key=lambda length: high_oracle_acc[length])
    result: dict[str, Any] = {"experiment_id": "REDUCED12_SHORT_PREFIX_LENGTH_ORACLE_SWEEP", "status": "COMPLETED", "population": {"moving_val_contexts": len(val_rows), "moving_val_records": len({str(row['record_id']) for row in val_rows})}, "labels": list(LABELS), "prefix_sweep_metrics": sweep, "fullview": {"true_logp_oracle": full_metric, "definition": "Stay=current[0:30]; candidate=candidate[0:30]; diagnostic reference only"}, "ranking_stability": ranking, "oracle_action_stability": full_action_stability, "high_occlusion": {"definition": "frame-0 Stay SceneVisibility bottom tertile", "count": int(high_mask.sum()), "by_length": high_oracle_acc, "best_length": int(high_best)}, "per_class": {"by_length": per_class, "best_prefix_length_by_f1": best_per_class}, "boundary_audit": boundary, "motion_evidence_statistics": motion, "coverage_audit": {"action_set": "Stay/current + Stage-A legal candidate_pool", "mean_legal_candidate_count": float(val_mask[:, 1:].sum(axis=1).mean()), "min_legal_candidate_count": int(val_mask[:, 1:].sum(axis=1).min()), "max_legal_candidate_count": int(val_mask[:, 1:].sum(axis=1).max()), "margin_anycorrect_max_abs_difference": float(max(abs(float(value["margin_anycorrect_difference"])) for value in sweep.values()))}, "protocol": {"split": "Moving Val", "prefix_lengths": list(PREFIX_LENGTHS), "mixed_sequence": "current[0:L] + candidate[L:30]", "stay": "current[0:30]", "continuous_navigation_modeled": False, "discrete_time_view_switch_approximation": True, "recognizer": "frozen reduced12 ST-GCN + frozen shared head", "training": "none"}, "flags": {"policy_test_used": False, "training_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "frozen_stgcn_modified": False, "gt_label_used_for_oracle_only": True, "deployable": False}, "artifacts": {"stgcn_checkpoint": {"path": str(stgcn_path.resolve()), "sha256": _sha256(stgcn_path)}, "shared_head_checkpoint": {"path": str(head_path.resolve()), "sha256": _sha256(head_path)}}}
    output_root.mkdir(parents=True, exist_ok=True)
    _write(output_root / "result.json", result)
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    _write(output_root / "config.json", {"prefix_lengths": list(PREFIX_LENGTHS), "action_set": "Stay + Stage-A legal candidate_pool", "mixed_sequence": "current[0:L] + candidate[L:30]", "test_used": False, "training_used": False})
    _write(output_root / "prefix_sweep_metrics.json", sweep)
    _write(output_root / "ranking_stability.json", ranking)
    _write(output_root / "oracle_action_stability.json", full_action_stability)
    _write(output_root / "boundary_audit.json", boundary)
    _write(output_root / "occlusion_stratified_metrics.json", result["high_occlusion"])
    _write(output_root / "per_class_metrics.json", result["per_class"])
    _write(output_root / "motion_evidence_statistics.json", motion)
    _write(output_root / "coverage_audit.json", result["coverage_audit"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
