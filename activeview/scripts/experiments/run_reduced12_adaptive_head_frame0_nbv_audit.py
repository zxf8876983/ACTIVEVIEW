#!/usr/bin/env python3
"""Audit adaptive frozen-recognizer heads with the strict Frame0 NBV selector.

Only Policy Train is used to fit the legal-candidate-balanced lightweight head;
Moving Val is used for model selection and reporting.  The selector sees the
current frame-0 DINO image context, legal geometry and the existing frame-0
visibility auxiliary, while selected candidates are evaluated from their real
archived observations (O1 alone).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    load_rows,
)
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import (
    TaskUtilityPredictor,
    _load_visibility_targets,
    _predict_task,
    _task_targets,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    EVAL_BATCH,
    OPTION_GEOMETRY_DIM,
    _load_frame0_dino,
    _method_metrics,
    _option_geometry,
    _select,
)
from activeview.scripts.experiments.run_reduced12_single_view_classifier_adaptation import (
    LightweightClassifier,
)

NUM_CLASSES = len(LABELS)
MAX_OPTIONS = 22
FEATURE_DIM = 256
SEED = 42
HEAD_EPOCHS = 20
HEAD_BATCH = 1024
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
OBS_PER_RECORD = 16
DIAGNOSTIC_REL = Path("diagnostics/reduced12_dual_route_overnight")
VIS_REL = Path("diagnostics/frame0_visibility_predictor_v1")
POLICY_REL = Path("datasets/policy_reduced12_eight_placement_v1")
SHARED_HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "view_agnostic_frozen_encoder_head/shared_head_best.pth"
)
OLD_HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "single_view_classifier_adaptation/learned_head_best.pth"
)
SELECTOR_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "frame0_task_utility_predictor_v1/RGBGlobal-TrueLogP+VisibilityAux.pth"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "adaptive_head_frame0_nbv_audit"
)
DEFAULT_RUNTIME = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "adaptive_head_frame0_nbv_audit"
)


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


def _row_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["episode_id"]) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_raw_options(
    data_root: Path, rows: Sequence[Mapping[str, Any]], split: str,
) -> dict[str, np.ndarray]:
    root = data_root / DIAGNOSTIC_REL
    meta = json.loads((root / f"{split}_options.json").read_text(encoding="utf-8"))
    if int(meta.get("rows", -1)) != len(rows) or meta.get("signature") != _row_signature(rows):
        raise ValueError(f"{split} option-cache signature mismatch")
    if bool(meta.get("test_used", True)):
        raise ValueError(f"{split} option cache is marked Test-derived")
    with np.load(root / f"{split}_options.npz", allow_pickle=False) as archive:
        cache = {key: np.asarray(archive[key]) for key in archive.files}
    if cache["features"].shape != (len(rows), MAX_OPTIONS, FEATURE_DIM):
        raise ValueError(f"unexpected {split} feature shape")
    if cache["ids"].shape != (len(rows), MAX_OPTIONS) or cache["mask"].shape != (len(rows), MAX_OPTIONS):
        raise ValueError(f"unexpected {split} option shape")
    for index, row in enumerate(rows):
        active = np.flatnonzero(cache["mask"][index])
        expected = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
        if cache["ids"][index, active].astype(int).tolist() != expected:
            raise ValueError(f"action-set mismatch at {split}/{row['episode_id']}")
    return cache


def _load_head(path: Path, device: torch.device) -> nn.Module:
    model: nn.Module = LightweightClassifier(FEATURE_DIM).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def _head_logits(cache: Mapping[str, np.ndarray], model: nn.Module, device: torch.device) -> np.ndarray:
    features = np.asarray(cache["features"], dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=bool)
    result = np.zeros((features.shape[0], features.shape[1], NUM_CLASSES), dtype=np.float32)
    rows, slots = np.nonzero(mask)
    with torch.inference_mode():
        for start in range(0, len(rows), 8192):
            batch_rows = rows[start : start + 8192]
            batch_slots = slots[start : start + 8192]
            values = torch.from_numpy(features[batch_rows, batch_slots]).to(device, non_blocking=True)
            result[batch_rows, batch_slots] = model(values).cpu().numpy()
    return result


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)


def _s1_predictions(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, logp: np.ndarray, labels: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    slots: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        matches = np.flatnonzero(mask[index] & (ids[index] == int(row["s1_viewpoint_id"])))
        if matches.size != 1:
            raise ValueError(f"s1 viewpoint is not in legal cache: {row['episode_id']}")
        slots.append((index, int(matches[0])))
    predictions = np.asarray([int(np.argmax(logp[row, slot])) for row, slot in slots], dtype=np.int64)
    return classification(labels, predictions), predictions


def _load_stage_d_s1_features(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    path = data_root / POLICY_REL / "stage_d/features/val.jsonl"
    by_episode = {
        str(item["episode_id"]): item for item in (
            json.loads(line) for line in path.open(encoding="utf-8") if line.strip()
        )
    }
    features: list[np.ndarray] = []
    for row in rows:
        item = by_episode.get(str(row["episode_id"]))
        if item is None:
            raise ValueError(f"missing Stage-D s1 feature for {row['episode_id']}")
        vector = np.asarray(item["s1_feature"], dtype=np.float32)
        if vector.shape != (FEATURE_DIM + NUM_CLASSES + 3,):
            raise ValueError(f"unexpected Stage-D s1 feature shape {vector.shape}")
        features.append(vector[:FEATURE_DIM])
    return np.stack(features).astype(np.float32)


def _direct_s1_logp(features: np.ndarray, model: nn.Module, device: torch.device) -> np.ndarray:
    """Evaluate the exact Stage-D s1 feature distribution (protocol matched)."""
    features = np.asarray(features, dtype=np.float32)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            batch = torch.from_numpy(features[start : start + 8192]).to(device, non_blocking=True)
            outputs.append(model(batch).cpu().numpy())
    return _log_softmax(np.concatenate(outputs, axis=0))


def _candidate_metrics(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, rows: Sequence[Mapping[str, Any]], labels: np.ndarray) -> dict[str, Any]:
    target: list[int] = []
    pred: list[int] = []
    per_view: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for slot in np.flatnonzero(mask[index])[1:]:
            target.append(int(labels[index]))
            prediction = int(np.argmax(logp[index, slot]))
            pred.append(prediction)
            per_view[str(int(ids[index, slot]))].append(int(prediction == labels[index]))
    metrics = classification(np.asarray(target), np.asarray(pred))
    metrics["per_viewpoint_accuracy"] = {key: float(np.mean(value)) for key, value in sorted(per_view.items(), key=lambda item: int(item[0]))}
    return metrics


def _oracle_actions(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, labels: np.ndarray, include_stay: bool, criterion: str) -> tuple[list[int], np.ndarray]:
    actions: list[int] = []
    any_correct: list[bool] = []
    for index in range(logp.shape[0]):
        active = np.flatnonzero(mask[index])
        if not include_stay:
            active = active[active != 0]
        values = logp[index, active]
        label = int(labels[index])
        if criterion == "true_logp":
            score = values[:, label]
        elif criterion == "margin":
            score = values[:, label] - np.max(np.delete(values, label, axis=1), axis=1)
        elif criterion == "confidence":
            score = np.max(np.exp(values), axis=1)
        else:
            raise ValueError(f"unknown oracle criterion {criterion}")
        selected = int(active[int(np.argmax(score))])
        actions.append(int(ids[index, selected]))
        candidate_slots = np.flatnonzero(mask[index])[1:]
        any_correct.append(bool(np.any(np.argmax(logp[index, candidate_slots], axis=1) == label)))
    return actions, np.asarray(any_correct, dtype=bool)


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, rows: Sequence[Mapping[str, Any]], actions: Sequence[int], labels: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    predictions: list[int] = []
    moves = 0
    for index, action in enumerate(actions):
        slots = np.flatnonzero(mask[index] & (ids[index] == int(action)))
        if slots.size != 1:
            raise ValueError(f"selected action is outside legal set at {rows[index]['episode_id']}: {action}")
        moves += int(int(action) != int(rows[index]["current_viewpoint_id"]))
        predictions.append(int(np.argmax(logp[index, int(slots[0])])) )
    result = classification(labels, np.asarray(predictions, dtype=np.int64))
    result["move_rate"] = float(moves / len(rows)) if rows else 0.0
    result["stay_rate"] = 1.0 - result["move_rate"]
    return result, np.asarray(predictions, dtype=np.int64)


def _safe_rank(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    context_values: list[float] = []
    top1: list[int] = []
    top3: list[int] = []
    for index in range(values.shape[0]):
        active = np.flatnonzero(mask[index])[1:]
        if len(active) < 2:
            continue
        context_values.append(correlation(values[index, active, 0], values[index, active, 1], spearman=True))
        left = active[np.argsort(-values[index, active, 0], kind="mergesort")]
        right = active[np.argsort(-values[index, active, 1], kind="mergesort")]
        top1.append(int(left[0] == right[0]))
        top3.append(int(right[0] in left[:3]))
    return {
        "within_context_spearman_mean": float(np.mean(context_values)) if context_values else 0.0,
        "within_context_spearman_median": float(np.median(context_values)) if context_values else 0.0,
        "oracle_top1_agreement": float(np.mean(top1)) if top1 else 0.0,
        "oracle_top3_overlap": float(np.mean(top3)) if top3 else 0.0,
    }


def _ranking_comparison(logps: Mapping[str, np.ndarray], mask: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    names = list(logps)
    targets = {name: _task_targets(logps[name], labels, mask)[0] for name in names}
    result: dict[str, Any] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pair = np.stack((targets[left], targets[right]), axis=-1)
            valid = mask & np.isfinite(pair).all(axis=-1)
            flat = pair[valid]
            result[f"{left}_vs_{right}"] = {
                "candidate_level_spearman": correlation(flat[:, 0], flat[:, 1], spearman=True),
                **_safe_rank(pair, mask),
            }
    return result


def _train_balanced_head(
    train_cache: Mapping[str, np.ndarray], train_rows: Sequence[Mapping[str, Any]],
    val_cache: Mapping[str, np.ndarray], val_rows: Sequence[Mapping[str, Any]],
    device: torch.device, checkpoint: Path, summary_path: Path,
) -> tuple[LightweightClassifier, dict[str, Any]]:
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = LightweightClassifier(FEATURE_DIM).to(device)
        model.load_state_dict(payload["state_dict"])
        return model.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    _seed()
    model = LightweightClassifier(FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    class_counts: defaultdict[int, int] = defaultdict(int)
    viewpoint_counts: defaultdict[int, int] = defaultdict(int)
    for index, row in enumerate(train_rows):
        for slot in np.flatnonzero(train_cache["mask"][index])[1:]:
            pair = (index, int(slot))
            groups[str(row["record_id"])].append(pair)
            class_counts[int(row["label_id"])] += 1
            viewpoint_counts[int(train_cache["ids"][index, slot])] += 1
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    best_loss = float("inf")
    best_epoch = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, HEAD_EPOCHS + 1):
        sampled: list[tuple[int, int]] = []
        for record in sorted(groups):
            values = groups[record]
            weights = np.asarray([
                1.0 / (class_counts[int(train_rows[row]["label_id"])] * viewpoint_counts[int(train_cache["ids"][row, slot])])
                for row, slot in values
            ], dtype=np.float64)
            weights /= weights.sum()
            selected = rng.choice(len(values), OBS_PER_RECORD, replace=len(values) < OBS_PER_RECORD, p=weights)
            sampled.extend(values[int(i)] for i in selected)
        losses: list[float] = []
        model.train()
        for start in range(0, len(sampled), HEAD_BATCH):
            selected = sampled[start : start + HEAD_BATCH]
            row_idx = np.asarray([item[0] for item in selected], dtype=np.int64)
            slot_idx = np.asarray([item[1] for item in selected], dtype=np.int64)
            x = torch.from_numpy(np.asarray(train_cache["features"][row_idx, slot_idx], dtype=np.float32)).to(device, non_blocking=True)
            y = torch.from_numpy(np.asarray([int(train_rows[i]["label_id"]) for i in row_idx], dtype=np.int64)).to(device, non_blocking=True)
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            rows, slots = np.nonzero(np.asarray(val_cache["mask"], dtype=bool) & (np.arange(MAX_OPTIONS)[None, :] != 0))
            for start in range(0, len(rows), 8192):
                rr, ss = rows[start : start + 8192], slots[start : start + 8192]
                x = torch.from_numpy(np.asarray(val_cache["features"][rr, ss], dtype=np.float32)).to(device, non_blocking=True)
                y = torch.from_numpy(np.asarray([int(val_rows[i]["label_id"]) for i in rr], dtype=np.int64)).to(device, non_blocking=True)
                val_losses.append(float(nn.functional.cross_entropy(model(x), y).cpu()))
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "val_candidate_ce": float(np.mean(val_losses)),
            "observations": len(sampled), "records": len(groups),
            "unique_classes": len(class_counts), "unique_viewpoints": len(viewpoint_counts),
        }
        history.append(record)
        print(f"[H_adapt_balanced] epoch={epoch:02d}/{HEAD_EPOCHS} train={record['train_loss']:.6f} val_ce={record['val_candidate_ce']:.6f}", flush=True)
        if record["val_candidate_ce"] < best_loss:
            best_loss = float(record["val_candidate_ce"])
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "feature_dim": FEATURE_DIM, "hidden_dim": 256, "num_classes": NUM_CLASSES, "epoch": epoch, "seed": SEED}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {
        "branch": "H_adapt_balanced", "seed": SEED, "epochs": HEAD_EPOCHS,
        "observations_per_record": OBS_PER_RECORD, "best_epoch": best_epoch,
        "best_val_candidate_ce": best_loss, "checkpoint_selection": "minimum Val legal-candidate CE",
        "sampling": "record-balanced 16 observations; inverse class and viewpoint frequency within record",
        "train_class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "train_viewpoint_counts": {str(k): int(v) for k, v in sorted(viewpoint_counts.items())},
        "history": history, "checkpoint": str(checkpoint.resolve()), "test_used": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model.eval(), summary


def _method_per_class(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return classification(labels, predictions)["per_class"]


def _analysis(result: Mapping[str, Any]) -> str:
    heads = result["head_distribution_audit"]
    methods = result["selector_metrics"]
    balanced = methods["H_frame0_adapt_balanced"]
    strict = methods["D_frame0_shared"]
    random_shared = methods["A_random_shared"]
    gain = float(balanced["accuracy"] - strict["accuracy"])
    old_specialized = bool(heads["specialization_audit"]["old_adaptive_is_distribution_specialized"])
    if gain < 0.01:
        decision = "KILL ADAPTIVE-HEAD × NBV COMBINATION"
    elif gain < 0.02:
        decision = "WEAK KEEP ADAPTIVE-HEAD × NBV COMBINATION"
    elif gain >= 0.03 and balanced["macro_f1"] >= strict["macro_f1"]:
        decision = "STRONG KEEP ADAPTIVE-HEAD × NBV COMBINATION"
    else:
        decision = "KEEP ADAPTIVE-HEAD × NBV COMBINATION"
    lines = [
        "# Adaptive Head × Frame0 NBV Combination Audit", "",
        "Task: strict single-step next-best-view selection", "",
        "Input before selection: current frame0 RGB + legal candidate geometry",
        "Full current action before selection: false", "Candidate observation before selection: false",
        "Final evaluation: selected O1 alone", "Multi-view fusion: false", "Policy Test: false",
        "Encoder: frozen ST-GCN", "Compared recognition heads: shared / historical adaptive / legal-candidate-balanced adaptive",
        "Selector architecture: historical Frame0 RGBGlobal + Geometry + VisibilityAux", "",
        f"Train contexts: {result['population']['train_contexts']}; Moving Val contexts: {result['population']['moving_val_contexts']}; records: {result['population']['train_records']} / {result['population']['val_records']}",
        "", "## Moving-Val strict results", "", "| Method | Head | Accuracy | Macro-F1 | Move rate |", "|---|---|---:|---:|---:|",
    ]
    for name, metric in methods.items():
        if "accuracy" in metric:
            lines.append(f"| {name} | {metric.get('head', '—')} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend(["", "## Head distribution", "", f"H_shared s1: {heads['H_shared']['s1']['accuracy']:.6f} / {heads['H_shared']['s1']['macro_f1']:.6f}; legal candidates: {heads['H_shared']['legal_candidates']['accuracy']:.6f} / {heads['H_shared']['legal_candidates']['macro_f1']:.6f}.", f"H_adapt_old s1: {heads['H_adapt_old']['s1']['accuracy']:.6f} / {heads['H_adapt_old']['s1']['macro_f1']:.6f}; legal candidates: {heads['H_adapt_old']['legal_candidates']['accuracy']:.6f} / {heads['H_adapt_old']['legal_candidates']['macro_f1']:.6f}.", f"H_adapt_balanced s1: {heads['H_adapt_balanced']['s1']['accuracy']:.6f} / {heads['H_adapt_balanced']['s1']['macro_f1']:.6f}; legal candidates: {heads['H_adapt_balanced']['legal_candidates']['accuracy']:.6f} / {heads['H_adapt_balanced']['legal_candidates']['macro_f1']:.6f}.", f"Specialization flag: {'OLD ADAPTIVE HEAD IS DISTRIBUTION-SPECIALIZED' if old_specialized else 'not triggered'}.", "", "## Gain decomposition", "", f"Random shared baseline: {random_shared['accuracy']:.6f}; head gain (random balanced − random shared): {result['gain_decomposition']['head_gain']:+.6f}; historical selector gain: {result['gain_decomposition']['selector_gain_shared']:+.6f}; adaptive selector gain: {result['gain_decomposition']['selector_gain_adaptive']:+.6f}; combined gain: {result['gain_decomposition']['combined_gain']:+.6f}; synergy: {result['gain_decomposition']['synergy']:+.6f}.", f"Balanced adaptive-aware selector versus historical strict selector: {gain:+.6f} Accuracy.", f"Decision: **{decision}**.", "", "## Interpretation", "", "The old and balanced heads are evaluated on the same current/Stay + Stage-A legal candidate action set. Candidate observations and recognizer outputs are never selector inputs; they are used only for Train targets or terminal/oracle diagnostics."])
    lines.extend(["", "```text", "policy_test_used=false", "training_split=Policy Train", "evaluation_split=Moving Val", "future_candidate_observation_used_for_selector=false", "future_candidate_recognizer_output_used_for_selector=false", "gt_action_used_for_selector=false", "frozen_stgcn_encoder_modified=false", "deployable_selector_inputs=true", "```", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    runtime_root = (data_root / args.runtime_relative).resolve()
    train_rows, val_rows = load_rows(data_root)
    val_s1_features = _load_stage_d_s1_features(data_root, val_rows)
    train_cache = _load_raw_options(data_root, train_rows, "train")
    val_cache = _load_raw_options(data_root, val_rows, "val")
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    shared_path, old_path, selector_path = data_root / SHARED_HEAD_REL, data_root / OLD_HEAD_REL, data_root / SELECTOR_REL
    for path in (shared_path, old_path, selector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    shared = SharedHead().to(device)
    shared.load_state_dict(torch.load(shared_path, map_location=device, weights_only=False)["state_dict"])
    old = _load_head(old_path, device)
    balanced, balanced_summary = _train_balanced_head(train_cache, train_rows, val_cache, val_rows, device, runtime_root / "H_adapt_balanced.pth", output_root / "H_adapt_balanced_training.json")
    models: dict[str, nn.Module] = {"H_shared": shared.eval(), "H_adapt_old": old, "H_adapt_balanced": balanced}
    logits = {name: _head_logits(val_cache, model, device) for name, model in models.items()}
    logps = {name: _log_softmax(values) for name, values in logits.items()}
    head_audit: dict[str, Any] = {}
    for name, values in logps.items():
        s1_logp = _direct_s1_logp(val_s1_features, models[name], device)
        s1 = classification(val_labels, np.argmax(s1_logp, axis=1))
        legal = _candidate_metrics(values, val_cache["ids"], val_cache["mask"], val_rows, val_labels)
        legal_true_logp, legal_margin = _task_targets(values, val_labels, val_cache["mask"])
        candidate_mask = val_cache["mask"].copy()
        candidate_mask[:, 0] = False
        legal["mean_gt_true_logp"] = float(np.nanmean(legal_true_logp[candidate_mask]))
        legal["mean_gt_margin"] = float(np.nanmean(legal_margin[candidate_mask]))
        head_audit[name] = {"s1": s1, "legal_candidates": legal}
    old_s1 = head_audit["H_adapt_old"]["s1"]["accuracy"]
    old_legal = head_audit["H_adapt_old"]["legal_candidates"]["accuracy"]
    shared_legal = head_audit["H_shared"]["legal_candidates"]["accuracy"]
    head_audit["specialization_audit"] = {
        "old_adaptive_is_distribution_specialized": bool(old_s1 - old_legal > 0.01 and old_legal <= shared_legal + 0.01),
        "old_s1_minus_old_legal_accuracy": float(old_s1 - old_legal),
        "old_legal_minus_shared_legal_accuracy": float(old_legal - shared_legal),
    }
    stats = json.loads((data_root / POLICY_REL / "stage_c/stage_c_feature_stats.json").read_text(encoding="utf-8"))
    train_geom, _, _ = _option_geometry(train_rows, stats)
    val_geom, _, _ = _option_geometry(val_rows, stats)
    train_tokens, train_dino_meta = _load_frame0_dino(data_root, train_rows, "train", device)
    val_tokens, val_dino_meta = _load_frame0_dino(data_root, val_rows, "val", device)
    train_vis = _load_visibility_targets(data_root, train_rows, "train")
    val_vis = _load_visibility_targets(data_root, val_rows, "val")
    if not np.array_equal(val_vis["ids"], val_cache["ids"]) or not np.array_equal(val_vis["mask"], val_cache["mask"]):
        raise ValueError("visibility and recognizer action-set mismatch")
    selector_path_payload = torch.load(selector_path, map_location=device, weights_only=False)
    historical_selector = TaskUtilityPredictor(True, True).to(device)
    historical_selector.load_state_dict(selector_path_payload["state_dict"])
    historical_selector.eval()
    historical_scores, _ = _predict_task(historical_selector, val_geom, val_tokens, device)
    selector_metrics: dict[str, Any] = {}
    actions: dict[str, list[int]] = {}
    predictions: dict[str, np.ndarray] = {}
    random_rng = np.random.default_rng(SEED)
    random_actions = [int(random_rng.choice([int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]])) for row in val_rows]
    def add_method(name: str, head_name: str, selected: Sequence[int], selector_name: str) -> None:
        metric, prediction = _terminal(logps[head_name], val_cache["ids"], val_cache["mask"], val_rows, selected, val_labels)
        metric["head"] = head_name
        metric["selector"] = selector_name
        selector_metrics[name] = metric
        actions[name] = list(map(int, selected))
        predictions[name] = prediction
    add_method("A_random_shared", "H_shared", random_actions, "fixed-seed random legal")
    add_method("B_random_adapt_old", "H_adapt_old", random_actions, "fixed-seed random legal")
    add_method("C_random_adapt_balanced", "H_adapt_balanced", random_actions, "fixed-seed random legal")
    shared_actions = _select(val_rows, historical_scores, val_cache["mask"])
    add_method("D_frame0_shared", "H_shared", shared_actions, "historical RGBGlobal+TrueLogP+VisibilityAux")
    add_method("E_frame0_shared_selector_adapt_old", "H_adapt_old", shared_actions, "same historical selector checkpoint")
    add_method("F_frame0_shared_selector_adapt_balanced", "H_adapt_balanced", shared_actions, "same historical selector checkpoint")
    train_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in models:
        train_logits = _head_logits(train_cache, models[name], device)
        train_logp = _log_softmax(train_logits)
        train_targets[name] = _task_targets(train_logp, train_labels, train_cache["mask"])
    adaptive_old_model = __import__("activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor", fromlist=["_train_branch"])._train_branch(
        "RGBGlobal-TrueLogP+VisibilityAux", True, True, train_rows, val_rows, train_geom, val_geom, train_tokens, val_tokens,
        train_targets["H_adapt_old"][0], _task_targets(logps["H_adapt_old"], val_labels, val_cache["mask"])[0], train_vis["scores"], val_vis["scores"],
        train_cache["mask"], val_cache["mask"], device, runtime_root / "selector_adapt_old.pth", output_root / "selector_adapt_old_training.json")
    adaptive_old_scores, _ = _predict_task(adaptive_old_model, val_geom, val_tokens, device)
    add_method("G_frame0_adaptive_old_selector", "H_adapt_old", _select(val_rows, adaptive_old_scores, val_cache["mask"]), "retrained target(H_adapt_old)")
    adaptive_bal_model = __import__("activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor", fromlist=["_train_branch"])._train_branch(
        "RGBGlobal-TrueLogP+VisibilityAux", True, True, train_rows, val_rows, train_geom, val_geom, train_tokens, val_tokens,
        train_targets["H_adapt_balanced"][0], _task_targets(logps["H_adapt_balanced"], val_labels, val_cache["mask"])[0], train_vis["scores"], val_vis["scores"],
        train_cache["mask"], val_cache["mask"], device, runtime_root / "selector_adapt_balanced.pth", output_root / "selector_adapt_balanced_training.json")
    adaptive_bal_scores, _ = _predict_task(adaptive_bal_model, val_geom, val_tokens, device)
    add_method("H_frame0_adapt_balanced", "H_adapt_balanced", _select(val_rows, adaptive_bal_scores, val_cache["mask"]), "retrained target(H_adapt_balanced)")
    oracle_metrics: dict[str, Any] = {}
    for head_name, values in logps.items():
        oracle_metrics[head_name] = {}
        for include_stay, scope in ((False, "candidate_only"), (True, "stay_plus_candidate")):
            oracle_metrics[head_name][scope] = {}
            for criterion in ("true_logp", "margin", "confidence"):
                selected, any_correct = _oracle_actions(values, val_cache["ids"], val_cache["mask"], val_labels, include_stay, criterion)
                metric, _ = _terminal(values, val_cache["ids"], val_cache["mask"], val_rows, selected, val_labels)
                metric["any_correct_coverage"] = float(np.mean(any_correct))
                oracle_metrics[head_name][scope][criterion] = metric
    ranking_metrics = _ranking_comparison(logps, val_cache["mask"], val_labels)
    selector_ranking: dict[str, Any] = {}
    for name, score, head_name in (("D_frame0_shared", historical_scores, "H_shared"), ("G_frame0_adaptive_old_selector", adaptive_old_scores, "H_adapt_old"), ("H_frame0_adapt_balanced", adaptive_bal_scores, "H_adapt_balanced")):
        target, _ = _task_targets(logps[head_name], val_labels, val_cache["mask"])
        valid = val_cache["mask"] & np.isfinite(target)
        selector_ranking[name] = {"candidate_level_spearman": correlation(score[valid], target[valid], spearman=True), **_safe_rank(np.stack((score, target), axis=-1), val_cache["mask"])}
    historical_expected = {"accuracy": 0.5061507936507936, "macro_f1": 0.5037217440127477}
    frame0_reproduction = {"actual": selector_metrics["D_frame0_shared"], "historical_reference": historical_expected, "accuracy_abs_error": float(abs(selector_metrics["D_frame0_shared"]["accuracy"] - historical_expected["accuracy"])), "macro_f1_abs_error": float(abs(selector_metrics["D_frame0_shared"]["macro_f1"] - historical_expected["macro_f1"])), "passed_0_5pp_gate": bool(abs(selector_metrics["D_frame0_shared"]["accuracy"] - historical_expected["accuracy"]) <= 0.005)}
    current_vis = val_vis["scores"][:, 0]
    bottom = current_vis < np.quantile(current_vis, 1.0 / 3.0)
    occlusion = {"definition": "bottom tertile of frame-0 Stay SceneVisibility", "count": int(bottom.sum()), "methods": {}}
    for name in ("A_random_shared", "D_frame0_shared", "C_random_adapt_balanced", "H_frame0_adapt_balanced"):
        metric = selector_metrics[name]
        occlusion["methods"][name] = classification(val_labels[bottom], predictions[name][bottom])
    gain = {"baseline_random_shared": selector_metrics["A_random_shared"]["accuracy"], "head_gain": selector_metrics["C_random_adapt_balanced"]["accuracy"] - selector_metrics["A_random_shared"]["accuracy"], "selector_gain_shared": selector_metrics["D_frame0_shared"]["accuracy"] - selector_metrics["A_random_shared"]["accuracy"], "selector_gain_adaptive": selector_metrics["H_frame0_adapt_balanced"]["accuracy"] - selector_metrics["C_random_adapt_balanced"]["accuracy"], "combined_gain": selector_metrics["H_frame0_adapt_balanced"]["accuracy"] - selector_metrics["A_random_shared"]["accuracy"]}
    gain["synergy"] = gain["combined_gain"] - gain["head_gain"] - gain["selector_gain_shared"]
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_ADAPTIVE_HEAD_FRAME0_NBV_AUDIT", "status": "COMPLETED",
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(r['record_id']) for r in train_rows}), "val_records": len({str(r['record_id']) for r in val_rows})},
        "labels": list(LABELS), "head_distribution_audit": head_audit, "selector_metrics": selector_metrics,
        "oracle_metrics": oracle_metrics, "ranking_metrics": ranking_metrics, "selector_ranking_metrics": selector_ranking,
        "frame0_reproduction": frame0_reproduction, "occlusion_metrics": occlusion, "gain_decomposition": gain,
        "balanced_head_training": balanced_summary,
        "flags": {"policy_test_used": False, "train_supervision_used": True, "new_rgb_generated": False, "new_skeleton_generated": False, "future_candidate_observation_used_for_selector": False, "future_candidate_feature_used_for_selector": False, "future_candidate_logits_used_for_selector": False, "gt_action_used_for_selector": False, "frozen_stgcn_encoder_modified": False, "multi_view_fusion": False, "selected_observation_protocol": "O1-alone"},
        "protocol": {"action_set": "current/Stay + Stage-A legal candidate_pool", "selector_input": "current frame0 DINO spatial context + option geometry + existing VisibilityAux", "selector_model": "historical Frame0 RGBGlobal+Geometry+VisibilityAux", "terminal": "selected real archived candidate observation through the nominated head", "test_used": False},
        "artifacts": {"train_options": str((data_root / DIAGNOSTIC_REL / "train_options.npz").resolve()), "val_options": str((data_root / DIAGNOSTIC_REL / "val_options.npz").resolve()), "shared_head": str(shared_path.resolve()), "old_head": str(old_path.resolve()), "historical_selector": str(selector_path.resolve()), "balanced_head": str((runtime_root / "H_adapt_balanced.pth").resolve()), "frame0_dino_train": train_dino_meta, "frame0_dino_val": val_dino_meta},
        "checkpoint_sha256": {"shared_head": _sha256(shared_path), "old_head": _sha256(old_path), "historical_selector": _sha256(selector_path), "balanced_head": _sha256(runtime_root / "H_adapt_balanced.pth")},
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda": torch.version.cuda, "elapsed_seconds": time.monotonic() - started, "seed": SEED},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"seed": SEED, "epochs": HEAD_EPOCHS, "obs_per_record": OBS_PER_RECORD, "option_geometry_dim": OPTION_GEOMETRY_DIM, "test_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "head_distribution_audit.json").write_text(json.dumps(head_audit, indent=2) + "\n", encoding="utf-8")
    (output_root / "head_alllegal_metrics.json").write_text(json.dumps({name: value["legal_candidates"] for name, value in head_audit.items() if name.startswith("H_")}, indent=2) + "\n", encoding="utf-8")
    (output_root / "head_per_view_metrics.json").write_text(json.dumps({name: value["legal_candidates"]["per_viewpoint_accuracy"] for name, value in head_audit.items() if name.startswith("H_")}, indent=2) + "\n", encoding="utf-8")
    (output_root / "head_per_class_metrics.json").write_text(json.dumps({name: {"s1": value["s1"]["per_class"], "legal_candidates": value["legal_candidates"]["per_class"]} for name, value in head_audit.items() if name.startswith("H_")}, indent=2) + "\n", encoding="utf-8")
    (output_root / "candidate_ranking_comparison.json").write_text(json.dumps(ranking_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "selector_metrics.json").write_text(json.dumps(selector_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "selector_ranking_metrics.json").write_text(json.dumps(selector_ranking, indent=2) + "\n", encoding="utf-8")
    training_summaries = {"H_adapt_balanced": balanced_summary}
    for branch_name in ("selector_adapt_old", "selector_adapt_balanced"):
        summary_file = output_root / f"{branch_name}_training.json"
        if summary_file.is_file():
            training_summaries[branch_name] = json.loads(summary_file.read_text(encoding="utf-8"))
    (output_root / "training_summary.json").write_text(json.dumps(training_summaries, indent=2) + "\n", encoding="utf-8")
    (output_root / "oracle_metrics.json").write_text(json.dumps(oracle_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "occlusion_metrics.json").write_text(json.dumps(occlusion, indent=2) + "\n", encoding="utf-8")
    (output_root / "per_class_combined_metrics.json").write_text(json.dumps({name: _method_per_class(val_labels, pred) for name, pred in predictions.items()}, indent=2) + "\n", encoding="utf-8")
    (output_root / "gain_decomposition.json").write_text(json.dumps(gain, indent=2) + "\n", encoding="utf-8")
    (output_root / "leakage_audit.json").write_text(json.dumps(result["flags"], indent=2) + "\n", encoding="utf-8")
    (output_root / "runtime_summary.json").write_text(json.dumps(result["runtime"], indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-relative", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output_root.resolve()), "methods": {name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]} for name, value in result["selector_metrics"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
