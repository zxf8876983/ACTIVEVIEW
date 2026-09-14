#!/usr/bin/env python3
"""Yaw8Fair strict Frame-0 one-step NBV re-baseline.

The experiment evaluates only the causal protocol ``Frame0 -> one Stage-A
legal candidate -> O1-alone HAR``.  Frozen Yaw8 features and the matched
Policy-balanced head are reused.  Selector checkpoints are trained from
Policy Train and selected on a deterministic record holdout inside that split;
Moving Val is never used for training or checkpoint selection and Policy Test
is never opened.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.eval.reduced12_nbv_utils import (  # noqa: E402
    LABELS,
    NUM_CLASSES,
    correlation,
)
from activeview.scripts.eval.reduced12_utility_source import (  # noqa: E402
    body_azimuth_bin,
    load_scene_metadata,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (  # noqa: E402
    SharedHead,
)
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import (  # noqa: E402
    TaskUtilityPredictor,
    _load_visibility_targets,
    _predict_task,
    _task_targets,
    _utility_loss,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (  # noqa: E402
    EVAL_BATCH,
    VisibilityPredictor,
    _method_metrics,
    _option_geometry,
    _predict,
    _terminal,
)
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (  # noqa: E402
    _any_correct,
    _load_rows,
    _oracle_actions,
    _read_npz,
    _sha256,
    _signature,
    _train_prior,
)
from activeview.scripts.experiments.run_reduced12_yaw8_shared_head_fairness_audit import (  # noqa: E402
    _build_options,
    _head_logits,
)
from activeview.scripts.experiments.yaw8_strict_frame0_rebaseline_report import render_analysis
SEED = 42
EPOCHS = 12
OBS_PER_RECORD = 16
TRAIN_BATCH = 1024
FEATURE_DIM = 256
MAX_OPTIONS = 22
WEIGHT_DECAY = 1e-4
POLICY_REL = Path("datasets/policy_reduced12_eight_placement_v1")
FAIR_CACHE_REL = Path("diagnostics/policy_features_shared_head_fairness/yaw8")
DINO_REL = Path("features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/frame0_current")
YAW8_STGCN_REL = Path("checkpoints/stgcn_reduced12_yaw8_v1/best.pt")
YAW8_HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "yaw8_shared_head_fairness_audit/yaw8_shared_head_best.pth"
)
OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/"
    "yaw8_strict_frame0_full_rebaseline"
)
CHECKPOINT_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "yaw8_strict_frame0_full_rebaseline"
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
def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
def _load_existing_dino(
    data_root: Path, rows: Sequence[Mapping[str, Any]], split: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the existing frame-0 cache; never invoke RGB/DINO generation."""
    path = data_root / DINO_REL / f"{split}.npy"
    metadata_path = data_root / DINO_REL / f"{split}.json"
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing existing frame-0 DINO cache: {path}")
    metadata = _read_json(metadata_path)
    if int(metadata.get("rows", -1)) != len(rows):
        raise ValueError(f"DINO row count mismatch: {split}")
    if metadata.get("signature") != _signature(rows):
        raise ValueError(f"DINO signature mismatch: {split}")
    if bool(metadata.get("test_used", True)) or bool(metadata.get("candidate_rgb_used", True)):
        raise ValueError(f"DINO cache is not strict current-frame cache: {split}")
    values = np.load(path, mmap_mode="r")
    if values.shape != (len(rows), 16, 768) or values.dtype != np.float16:
        raise ValueError(f"unexpected DINO shape/dtype: {split} {values.shape} {values.dtype}")
    return np.asarray(values), metadata
def _load_yaw8_options(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    head: nn.Module,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_split = "train" if split == "train" else "moving_val"
    root = data_root / FAIR_CACHE_REL
    cache_path, metadata_path = root / f"{cache_split}.npz", root / f"{cache_split}.json"
    metadata = _read_json(metadata_path)
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != _signature(rows):
        raise ValueError(f"Yaw8Fair cache signature/count mismatch: {split}")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"Yaw8Fair cache marked Test-derived: {split}")
    cache = _read_npz(cache_path)
    expected = (len(rows), MAX_OPTIONS)
    if cache["features"].shape != (*expected, FEATURE_DIM):
        raise ValueError(f"Yaw8Fair feature shape mismatch: {split}")
    if cache["ids"].shape != expected or cache["mask"].shape != expected:
        raise ValueError(f"Yaw8Fair action shape mismatch: {split}")
    for index, row in enumerate(rows):
        active = np.flatnonzero(cache["mask"][index])
        wanted = [int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]]
        if cache["ids"][index, active].astype(int).tolist() != wanted:
            raise ValueError(f"candidate action mismatch in Yaw8Fair cache: {row['episode_id']}")
    logits = _head_logits(head, np.asarray(cache["features"], dtype=np.float32), device)
    return _build_options(cache, logits), metadata


def _candidate_mask(mask: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    result[:, 0] = False
    return result


def _select_candidate(
    rows: Sequence[Mapping[str, Any]], scores: np.ndarray, mask: np.ndarray
) -> list[int]:
    actions: list[int] = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        # Callers may pass either the full current+candidate mask or an
        # explicit candidate-only mask.  Drop slot zero only in the former.
        if bool(mask[index, 0]):
            active = active[1:]
        if active.size == 0:
            actions.append(int(row["current_viewpoint_id"]))
            continue
        ordered = sorted(
            active.tolist(),
            key=lambda slot: (-float(scores[index, slot]), int(mask[index, slot]), int(slot)),
        )
        actions.append(int(scores.shape[1] and row["candidate_ids"][ordered[0] - 1]))
    return actions


def _candidate_random_actions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    rng = np.random.default_rng(SEED)
    result: list[int] = []
    for row in rows:
        candidates = [int(x) for x in row["candidate_ids"]]
        result.append(int(rng.choice(candidates)) if candidates else int(row["current_viewpoint_id"]))
    return result


def _prior_scores(
    ids: np.ndarray, mask: np.ndarray, prior: Mapping[int, float]
) -> np.ndarray:
    scores = np.full(ids.shape, -1e9, dtype=np.float32)
    for index in range(ids.shape[0]):
        for slot in np.flatnonzero(mask[index]):
            scores[index, slot] = float(prior.get(int(ids[index, slot]), prior.get(-1, 0.0)))
    return scores


def _record_holdout(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    records = sorted({str(row["record_id"]) for row in rows})
    holdout_count = max(1, int(round(0.1 * len(records))))
    holdout = set(records[-holdout_count:])
    hold_idx = np.asarray(
        [i for i, row in enumerate(rows) if str(row["record_id"]) in holdout], dtype=np.int64
    )
    fit_idx = np.asarray(
        [i for i, row in enumerate(rows) if str(row["record_id"]) not in holdout], dtype=np.int64
    )
    return fit_idx, hold_idx


def _sample_fit_rows(
    rows: Sequence[Mapping[str, Any]], fit_idx: np.ndarray, epoch: int
) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in fit_idx.tolist():
        groups[str(rows[index]["record_id"])].append(int(index))
    rng = np.random.default_rng(SEED + epoch)
    selected: list[int] = []
    for record in sorted(groups):
        values = np.asarray(groups[record], dtype=np.int64)
        selected.extend(rng.choice(values, OBS_PER_RECORD, replace=len(values) < OBS_PER_RECORD).tolist())
    return np.asarray(selected, dtype=np.int64)


def _eval_utility_loss(
    model: TaskUtilityPredictor,
    indices: np.ndarray,
    geometry: np.ndarray,
    tokens: np.ndarray | None,
    target: np.ndarray,
    mask: np.ndarray,
    auxiliary_target: np.ndarray | None,
    device: torch.device,
) -> tuple[float, float | None]:
    losses: list[float] = []
    aux_losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), EVAL_BATCH):
            batch = indices[start : start + EVAL_BATCH]
            geom = torch.from_numpy(geometry[batch]).to(device, non_blocking=True)
            tok = None if tokens is None else torch.from_numpy(np.asarray(tokens[batch], dtype=np.float32)).to(device, non_blocking=True)
            pred, pred_aux = model(geom, tok)
            tgt = torch.from_numpy(target[batch]).to(device, non_blocking=True)
            valid = torch.from_numpy(mask[batch]).to(device, non_blocking=True)
            task = _utility_loss(pred, tgt, valid, 0.5)[0]
            total = task
            if pred_aux is not None and auxiliary_target is not None:
                aux = torch.from_numpy(auxiliary_target[batch]).to(device, non_blocking=True)
                aux_loss = _utility_loss(pred_aux, aux, valid, 0.1)[0]
                total = total + 0.5 * aux_loss
                aux_losses.append(float(aux_loss.cpu()))
            losses.append(float(total.cpu()))
    return float(np.mean(losses)) if losses else 0.0, float(np.mean(aux_losses)) if aux_losses else None


def _train_utility_branch(
    name: str,
    model: TaskUtilityPredictor,
    rows: Sequence[Mapping[str, Any]],
    geometry: np.ndarray,
    tokens: np.ndarray | None,
    target: np.ndarray,
    mask: np.ndarray,
    auxiliary_target: np.ndarray | None,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
) -> tuple[TaskUtilityPredictor, dict[str, Any]]:
    fit_idx, hold_idx = _record_holdout(rows)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval(), _read_json(summary_path)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    best = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        sampled = _sample_fit_rows(rows, fit_idx, epoch)
        model.train()
        train_losses: list[float] = []
        for start in range(0, len(sampled), TRAIN_BATCH):
            batch = sampled[start : start + TRAIN_BATCH]
            geom = torch.from_numpy(geometry[batch]).to(device, non_blocking=True)
            tok = None if tokens is None else torch.from_numpy(np.asarray(tokens[batch], dtype=np.float32)).to(device, non_blocking=True)
            tgt = torch.from_numpy(target[batch]).to(device, non_blocking=True)
            valid = torch.from_numpy(mask[batch]).to(device, non_blocking=True)
            pred, pred_aux = model(geom, tok)
            total = _utility_loss(pred, tgt, valid, 0.5)[0]
            if pred_aux is not None and auxiliary_target is not None:
                aux = torch.from_numpy(auxiliary_target[batch]).to(device, non_blocking=True)
                total = total + 0.5 * _utility_loss(pred_aux, aux, valid, 0.1)[0]
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(total.detach().cpu()))
        hold_loss, hold_aux = _eval_utility_loss(
            model, hold_idx, geometry, tokens, target, mask, auxiliary_target, device
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "internal_policy_train_holdout_loss": hold_loss,
            "internal_policy_train_holdout_aux_loss": hold_aux,
            "fit_records": int(len({str(rows[i]['record_id']) for i in fit_idx.tolist()})),
            "holdout_records": int(len({str(rows[i]['record_id']) for i in hold_idx.tolist()})),
            "sampled_contexts": int(len(sampled)),
        }
        history.append(record)
        print(f"[{name}] epoch={epoch:02d} train={record['train_loss']:.6f} holdout={hold_loss:.6f}", flush=True)
        if hold_loss < best:
            best, best_epoch = hold_loss, epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "branch": name, "epoch": epoch, "seed": SEED}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    summary = {
        "branch": name,
        "seed": SEED,
        "epochs": EPOCHS,
        "observations_per_record": OBS_PER_RECORD,
        "best_epoch": best_epoch,
        "best_internal_policy_train_holdout_loss": best,
        "checkpoint_selection": "minimum objective on deterministic 10% Policy-Train record holdout",
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
        "moving_val_used_for_training_or_selection": False,
    }
    _write_json(summary_path, summary)
    return model.eval(), summary


def _subset_metric(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    actions: Sequence[int],
) -> dict[str, Any]:
    result = _method_metrics(name, rows, labels, predictions, actions)
    result["n"] = int(len(rows))
    return result


def _ranking_metrics(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    true_logp: np.ndarray,
    ids: np.ndarray,
) -> dict[str, Any]:
    candidate = _candidate_mask(mask)
    valid_pred, valid_target = predicted[candidate], target[candidate]
    within: list[float] = []
    top1: list[int] = []
    top3: list[int] = []
    selected_true: list[float] = []
    regrets: list[float] = []
    for index in range(len(rows)):
        active = np.flatnonzero(candidate[index])
        if active.size == 0:
            continue
        pred_order = active[np.argsort(-predicted[index, active], kind="mergesort")]
        target_order = active[np.argsort(-target[index, active], kind="mergesort")]
        if active.size >= 2:
            within.append(correlation(predicted[index, active], target[index, active], spearman=True))
        top1.append(int(pred_order[0] == target_order[0]))
        top3.append(int(target_order[0] in pred_order[:3]))
        selected_true.append(float(true_logp[index, pred_order[0]]))
        regrets.append(float(true_logp[index, target_order[0]] - true_logp[index, pred_order[0]]))
    del ids
    return {
        "branch": name,
        "candidate_count": int(valid_pred.size),
        "candidate_level_spearman": correlation(valid_pred, valid_target, spearman=True),
        "within_context_spearman_mean": float(np.mean(within)) if within else 0.0,
        "within_context_spearman_median": float(np.median(within)) if within else 0.0,
        "within_context_count": int(len(within)),
        "oracle_top1_agreement": float(np.mean(top1)) if top1 else 0.0,
        "oracle_top3_hit": float(np.mean(top3)) if top3 else 0.0,
        "mean_selected_gt_true_logp": float(np.mean(selected_true)) if selected_true else 0.0,
        "mean_oracle_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mae": float(np.mean(np.abs(valid_pred - valid_target))) if valid_pred.size else 0.0,
    }


def _historical_inventory(repo_root: Path) -> dict[str, Any]:
    old_root = repo_root / "experiments/reduced12_eight_placement_v1"

    def metric(path: Path, method: str) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        payload = _read_json(path)
        values = payload.get("methods", {}).get(method)
        if isinstance(values, dict):
            return {"accuracy": values.get("accuracy"), "macro_f1": values.get("macro_f1")}
        return None

    entries = [
        {"method": "B0 Stay", "historical": metric(old_root / "frame0_task_utility_predictor_v1/result.json", "Stay"), "strict_frame0_causal": False, "rerun": False, "reason": "diagnostic only; not a next-view action"},
        {"method": "B1 Random legal", "historical": metric(old_root / "frame0_task_utility_predictor_v1/result.json", "Random legal"), "strict_frame0_causal": True, "rerun": True, "reason": "formal baseline"},
        {"method": "B2 StaticViewPrior", "historical": metric(old_root / "prior_residual_frame0_nbv/result.json", "StaticViewPrior + old adaptive"), "strict_frame0_causal": True, "rerun": True, "reason": "Train-derived fixed viewpoint prior"},
        {"method": "Visibility family V1-V4", "historical": metric(old_root / "frame0_visibility_predictor_v1/result.json", "RGBGlobal+Geometry"), "selector_inputs": "frame0 scene/geometry and/or current DINO", "rerun": True, "reason": "exact cache/checkpoint available"},
        {"method": "Utility family U1-U4", "historical": metric(old_root / "frame0_task_utility_predictor_v1/result.json", "RGBGlobal-TrueLogP+VisibilityAux"), "selector_inputs": "frame0 DINO + geometry", "rerun": True, "reason": "targets must be rebuilt for Yaw8Fair"},
        {"method": "Prior residual P1-P3", "historical": metric(old_root / "prior_residual_frame0_nbv/result.json", "Prior+RGBResidual λ=1.0"), "selector_inputs": "Train prior + frame0 evidence", "rerun": True, "reason": "utility landscape changed under Yaw8Fair"},
    ]
    excluded = [
        "full O0/full temporal visibility", "short-prefix or full-sequence fusion", "B2-B4/OrderedConcat/streaming", "VLM/map/structured observability/GT-action/pair/world-model", "old adaptive or balanced adaptive head", "full ST-GCN fine-tuning",
    ]
    return {"entries": entries, "excluded_routes": excluded, "historical_results_are_reference_only": True}


def _distribution(
    rows: Sequence[Mapping[str, Any]],
    actions_by_method: Mapping[str, Sequence[int]],
    metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    static_actions: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, actions in actions_by_method.items():
        views = np.asarray(actions, dtype=np.int64)
        counts = Counter(str(v) for v in views.tolist())
        radii = Counter()
        yaw_bins = Counter()
        for row, view in zip(rows, views.tolist()):
            meta = metadata[(str(row["scene_id"]), str(row["region"]))]
            radii[str(meta["radii"][int(view)])] += 1
            yaw_bins[str(body_azimuth_bin(meta["azimuths"][int(view)], meta["yaw_deg"]))] += 1
        probs = np.asarray(list(counts.values()), dtype=np.float64)
        probs /= max(probs.sum(), 1.0)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        result[name] = {
            "viewpoint_histogram": dict(sorted(counts.items(), key=lambda x: int(x[0]))),
            "radius_histogram": dict(sorted(radii.items(), key=lambda x: float(x[0]))),
            "relative_yaw_bin_histogram": dict(sorted(yaw_bins.items(), key=lambda x: int(x[0]))),
            "selection_entropy": entropy,
            "top1_agreement_with_static": float(np.mean(views == np.asarray(static_actions, dtype=np.int64))),
        }
    return result


def _switch_analysis(
    rows: Sequence[Mapping[str, Any]],
    static_actions: Sequence[int],
    learned_actions: Sequence[int],
    static_pred: np.ndarray,
    learned_pred: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    static_a, learned_a = np.asarray(static_actions), np.asarray(learned_actions)
    switched = static_a != learned_a
    return {
        "switch_count": int(switched.sum()),
        "switch_rate": float(switched.mean()),
        "switch_contexts": [str(rows[i]["episode_id"]) for i in np.flatnonzero(switched)[:100]],
        "static_selected_accuracy_on_switch": float(np.mean(static_pred[switched] == labels[switched])) if switched.any() else 0.0,
        "learned_selected_accuracy_on_switch": float(np.mean(learned_pred[switched] == labels[switched])) if switched.any() else 0.0,
        "static_wrong_learned_correct": int(np.sum((static_pred != labels) & (learned_pred == labels) & switched)),
        "static_correct_learned_wrong": int(np.sum((static_pred == labels) & (learned_pred != labels) & switched)),
        "both_correct": int(np.sum((static_pred == labels) & (learned_pred == labels))),
        "both_wrong": int(np.sum((static_pred != labels) & (learned_pred != labels))),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    started = time.monotonic()
    train_rows, moving_rows = _load_rows(data_root)
    labels_train = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    labels_val = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    head_path = data_root / YAW8_HEAD_REL
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    train_options, train_cache_meta = _load_yaw8_options(data_root, train_rows, "train", head, device)
    val_options, val_cache_meta = _load_yaw8_options(data_root, moving_rows, "moving_val", head, device)
    train_tokens, train_dino_meta = _load_existing_dino(data_root, train_rows, "train")
    val_tokens, val_dino_meta = _load_existing_dino(data_root, moving_rows, "val")
    stats_path = data_root / POLICY_REL / "stage_c/stage_c_feature_stats.json"
    stats = _read_json(stats_path)
    train_geometry, train_ids, train_geom_mask = _option_geometry(train_rows, stats)
    val_geometry, val_ids, val_geom_mask = _option_geometry(moving_rows, stats)
    if not np.array_equal(train_ids, train_options["ids"]) or not np.array_equal(val_ids, val_options["ids"]):
        raise ValueError("geometry and Yaw8Fair action IDs differ")
    if not np.array_equal(train_geom_mask, train_options["mask"]) or not np.array_equal(val_geom_mask, val_options["mask"]):
        raise ValueError("geometry and Yaw8Fair action masks differ")
    train_visibility = _load_visibility_targets(data_root, train_rows, "train")
    val_visibility = _load_visibility_targets(data_root, moving_rows, "val")
    train_true, train_margin = _task_targets(train_options["logp"], labels_train, train_options["mask"])
    val_true, val_margin = _task_targets(val_options["logp"], labels_val, val_options["mask"])
    prior, prior_meta = _train_prior(train_rows, train_options)
    train_q = _prior_scores(train_options["ids"], train_options["mask"], prior)
    val_q = _prior_scores(val_options["ids"], val_options["mask"], prior)
    train_residual = train_margin - train_q
    val_residual = val_margin - val_q
    candidate_mask_train = _candidate_mask(train_options["mask"])
    candidate_mask_val = _candidate_mask(val_options["mask"])
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = data_root / CHECKPOINT_REL

    methods: dict[str, Any] = {}
    actions_by_method: dict[str, list[int]] = {}
    predictions_by_method: dict[str, np.ndarray] = {}

    def add_method(name: str, actions: Sequence[int]) -> None:
        predictions = _terminal(val_options["logp"], val_options["ids"], val_options["mask"], actions, labels_val)
        actions_by_method[name] = list(actions)
        predictions_by_method[name] = predictions
        methods[name] = _method_metrics(name, moving_rows, labels_val, predictions, actions)

    add_method("Stay", [int(row["current_viewpoint_id"]) for row in moving_rows])
    add_method("Random legal", _candidate_random_actions(moving_rows))
    static_actions = _select_candidate(moving_rows, val_q, candidate_mask_val)
    add_method("StaticViewPrior", static_actions)

    # Visibility family: these checkpoints/targets are exact historical
    # frame-0 artifacts and do not depend on the recognizer head.
    visibility_specs = {
        "Frame0SceneVisibility": (None, val_visibility["scores"]),
        "GeometryOnly-Visibility": ("GeometryOnly", None),
        "RGBGlobal-Visibility": ("RGBGlobal+Geometry", None),
        "RGBSpatial-Visibility": ("RGBSpatial+Geometry", None),
    }
    visibility_predictions: dict[str, np.ndarray] = {}
    for name, (branch, direct) in visibility_specs.items():
        if direct is not None:
            scores = np.asarray(direct, dtype=np.float32)
        else:
            checkpoint = data_root / "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1" / f"{branch}.pth"
            model = VisibilityPredictor(branch).to(device)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
            scores = _predict(model.eval(), val_geometry, None if branch == "GeometryOnly" else val_tokens, device)
        visibility_predictions[name] = scores
        add_method(name, _select_candidate(moving_rows, scores, candidate_mask_val))

    # Utility and residual branches are trained on Yaw8Fair Train targets.
    branch_specs: list[tuple[str, bool, bool, np.ndarray, np.ndarray, np.ndarray | None]] = [
        ("GeometryOnly-TrueLogP", False, False, train_true, val_true, None),
        ("RGBGlobal-TrueLogP", True, False, train_true, val_true, None),
        ("RGBGlobal-Margin", True, False, train_margin, val_margin, None),
        ("RGBGlobal-TrueLogP+VisibilityAux", True, True, train_true, val_true, train_visibility["scores"]),
        ("Prior+GeometryResidual", False, False, train_residual, val_residual, None),
        ("Prior+RGBResidual λ=0.5", True, False, train_residual, val_residual, None),
        ("Prior+RGBResidual λ=1.0", True, False, train_residual, val_residual, None),
    ]
    models: dict[str, TaskUtilityPredictor] = {}
    utility_predictions: dict[str, np.ndarray] = {}
    ranking_metrics: dict[str, Any] = {}
    training_summaries: dict[str, Any] = {}
    for name, visual, auxiliary, target_train, target_val, aux_train in branch_specs:
        model, summary = _train_utility_branch(
            name,
            TaskUtilityPredictor(visual, auxiliary),
            train_rows,
            train_geometry,
            train_tokens if visual else None,
            target_train,
            train_options["mask"],
            aux_train,
            device,
            runtime / (name.replace(" ", "_").replace("λ", "lambda") + ".pth"),
            output_root / (name.replace(" ", "_").replace("λ", "lambda") + "_training.json"),
        )
        models[name] = model
        scores, _ = _predict_task(model, val_geometry, val_tokens if visual else None, device)
        utility_predictions[name] = scores
        effective_scores = scores
        target_for_ranking = val_true if "Margin" not in name and "Residual" not in name else (val_margin if "Residual" not in name else val_residual)
        if "Residual" in name:
            effective_scores = val_q + scores
        add_method(name, _select_candidate(moving_rows, effective_scores, candidate_mask_val))
        ranking_metrics[name] = _ranking_metrics(name, moving_rows, scores, target_for_ranking, val_options["mask"], val_true, val_options["ids"])
        training_summaries[name] = summary

    # Candidate-only Yaw8Fair oracle references.
    true_actions = _oracle_actions(val_options, moving_rows, "true_logp", candidate_only=True)
    margin_actions = _oracle_actions(val_options, moving_rows, "margin", candidate_only=True)
    add_method("GT-TrueLogP Oracle", true_actions)
    add_method("GT-Margin Oracle", margin_actions)
    any_correct = _any_correct(val_options, moving_rows, candidate_only=True)
    methods["AnyCorrect Coverage"] = {
        "n": len(moving_rows),
        "coverage_count": int(any_correct.sum()),
        "coverage_rate": float(any_correct.mean()),
        "test_used": False,
    }

    # Historical old-recognizer transfer diagnostics (never promoted to the
    # formal table).  The architecture is identical, so only selected O1 is
    # evaluated with the new Yaw8Fair recognizer.
    transfer: dict[str, Any] = {}
    old_u4_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/frame0_task_utility_predictor_v1/RGBGlobal-TrueLogP+VisibilityAux.pth"
    if old_u4_path.is_file():
        old_model = TaskUtilityPredictor(True, True).to(device)
        old_model.load_state_dict(torch.load(old_u4_path, map_location=device, weights_only=False)["state_dict"])
        old_scores, _ = _predict_task(old_model.eval(), val_geometry, val_tokens, device)
        old_actions = _select_candidate(moving_rows, old_scores, candidate_mask_val)
        old_pred = _terminal(val_options["logp"], val_options["ids"], val_options["mask"], old_actions, labels_val)
        transfer["OldTargetSelector_to_Yaw8Fair"] = _method_metrics("OldTargetSelector→Yaw8Fair", moving_rows, labels_val, old_pred, old_actions)
    old_res_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/prior_residual_frame0_nbv/rgb_residual.pth"
    if old_res_path.is_file():
        old_model = TaskUtilityPredictor(True, True).to(device)
        old_model.load_state_dict(torch.load(old_res_path, map_location=device, weights_only=False)["state_dict"])
        old_scores, _ = _predict_task(old_model.eval(), val_geometry, val_tokens, device)
        old_actions = _select_candidate(moving_rows, val_q + old_scores, candidate_mask_val)
        old_pred = _terminal(val_options["logp"], val_options["ids"], val_options["mask"], old_actions, labels_val)
        transfer["OldResidualSelector_to_Yaw8Fair"] = _method_metrics("OldResidualSelector→Yaw8Fair", moving_rows, labels_val, old_pred, old_actions)

    # RGB shuffle and one geometry shuffle diagnostic.
    shuffle: dict[str, Any] = {}
    permutation = np.random.default_rng(SEED).permutation(len(moving_rows))
    rgb_branches = ["RGBGlobal-TrueLogP", "RGBGlobal-Margin", "RGBGlobal-TrueLogP+VisibilityAux", "Prior+RGBResidual λ=0.5", "Prior+RGBResidual λ=1.0"]
    best_rgb_drop = -float("inf")
    for name in rgb_branches:
        model = models[name]
        shuffled, _ = _predict_task(model, val_geometry, val_tokens[permutation], device)
        if "Residual" in name:
            shuffled = val_q + shuffled
        shuffled_actions = _select_candidate(moving_rows, shuffled, candidate_mask_val)
        shuffled_pred = _terminal(val_options["logp"], val_options["ids"], val_options["mask"], shuffled_actions, labels_val)
        shuffled_metric = _method_metrics(name + " RGB-shuffled", moving_rows, labels_val, shuffled_pred, shuffled_actions)
        drop = 100.0 * (methods[name]["accuracy"] - shuffled_metric["accuracy"])
        shuffle[name] = {"normal": methods[name], "rgb_shuffled": shuffled_metric, "accuracy_drop_pp": drop, "permutation_seed": SEED}
        best_rgb_drop = max(best_rgb_drop, drop)
    learned_names = [spec[0] for spec in branch_specs]
    strict_names = ["Frame0SceneVisibility", "GeometryOnly-Visibility", "RGBGlobal-Visibility", "RGBSpatial-Visibility", *learned_names]
    best_name = max(learned_names, key=lambda n: float(methods[n]["accuracy"]))
    best_strict_name = max(strict_names, key=lambda n: float(methods[n]["accuracy"]))
    best_model = models[best_name]
    shuffled_geometry, _ = _predict_task(best_model, val_geometry[permutation], val_tokens if best_model.visual else None, device)
    if "Residual" in best_name:
        shuffled_geometry = val_q + shuffled_geometry
    shuffled_actions = _select_candidate(moving_rows, shuffled_geometry, candidate_mask_val)
    shuffled_pred = _terminal(val_options["logp"], val_options["ids"], val_options["mask"], shuffled_actions, labels_val)
    geometry_shuffle = _method_metrics(best_name + " geometry-shuffled", moving_rows, labels_val, shuffled_pred, shuffled_actions)
    shuffle["best_geometry"] = {"branch": best_name, "normal": methods[best_name], "geometry_shuffled": geometry_shuffle, "accuracy_drop_pp": 100.0 * (methods[best_name]["accuracy"] - geometry_shuffle["accuracy"]), "permutation_seed": SEED}

    # Scene visibility bottom tertile, exactly the existing frame-0 subset.
    stay_visibility = np.asarray(val_visibility["scores"][:, 0], dtype=np.float64)
    high_mask = stay_visibility < np.quantile(stay_visibility, 1.0 / 3.0)
    high_methods: dict[str, Any] = {}
    high_names = ["Random legal", "StaticViewPrior", "Frame0SceneVisibility", "GeometryOnly-TrueLogP", "RGBGlobal-TrueLogP", "RGBGlobal-Margin", "RGBGlobal-TrueLogP+VisibilityAux", "Prior+RGBResidual λ=1.0", "GT-TrueLogP Oracle", "GT-Margin Oracle"]
    for name in high_names:
        indices = np.flatnonzero(high_mask)
        subset_rows = [moving_rows[int(i)] for i in indices]
        high_methods[name] = _subset_metric(name, subset_rows, labels_val[high_mask], predictions_by_method[name][high_mask], [actions_by_method[name][int(i)] for i in indices])
    learned_high = max((high_methods[n]["accuracy"] for n in high_names if n in learned_names), default=0.0)
    static_high = float(high_methods["StaticViewPrior"]["accuracy"])

    metadata, scene_audit = load_scene_metadata(data_root, list(train_rows) + list(moving_rows), audit_sample_count=10)
    distribution_names = {"StaticViewPrior": static_actions, "RGBGlobal-TrueLogP": actions_by_method["RGBGlobal-TrueLogP"], "RGBGlobal-Margin": actions_by_method["RGBGlobal-Margin"], "RGBGlobal-TrueLogP+VisibilityAux": actions_by_method["RGBGlobal-TrueLogP+VisibilityAux"], best_name: actions_by_method[best_name]}
    distribution = _distribution(moving_rows, distribution_names, metadata, static_actions)
    switch = _switch_analysis(moving_rows, static_actions, actions_by_method[best_name], predictions_by_method["StaticViewPrior"], predictions_by_method[best_name], labels_val)

    gate_expected = {"GT-TrueLogP": 0.760714, "GT-Margin": 0.776190, "AnyCorrect": 0.776190}
    gate_observed = {"GT-TrueLogP": methods["GT-TrueLogP Oracle"]["accuracy"], "GT-Margin": methods["GT-Margin Oracle"]["accuracy"], "AnyCorrect": methods["AnyCorrect Coverage"]["coverage_rate"]}
    gate_delta = {key: float(gate_observed[key] - gate_expected[key]) for key in gate_expected}
    oracle_metrics = {"candidate_only": {"GT-TrueLogP": methods["GT-TrueLogP Oracle"], "GT-Margin": methods["GT-Margin Oracle"], "AnyCorrect": methods["AnyCorrect Coverage"]}, "candidate_anycorrect_rate": float(any_correct.mean()), "sanity_expected": gate_expected, "sanity_observed": gate_observed, "sanity_delta": gate_delta, "sanity_status": "PASS" if max(abs(x) for x in gate_delta.values()) <= 0.002 else "FAIL"}
    if oracle_metrics["sanity_status"] != "PASS":
        raise RuntimeError(f"Yaw8Fair candidate-only oracle sanity failed: {gate_delta}")

    candidate_count_mean = float(np.mean(val_options["mask"][:, 1:].sum(axis=1)))
    candidate_count_min = int(np.min(val_options["mask"][:, 1:].sum(axis=1)))
    candidate_count_max = int(np.max(val_options["mask"][:, 1:].sum(axis=1)))
    candidate_empty = int(np.sum(val_options["mask"][:, 1:].sum(axis=1) == 0))
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_YAW8FAIR_STRICT_FRAME0_FULL_REBASELINE",
        "status": "COMPLETED",
        "population": {
            "policy_train_contexts": len(train_rows), "moving_val_contexts": len(moving_rows),
            "policy_train_records": len({str(r["record_id"]) for r in train_rows}), "moving_val_records": len({str(r["record_id"]) for r in moving_rows}),
            "legal_candidate_count_mean": candidate_count_mean, "legal_candidate_count_min": candidate_count_min, "legal_candidate_count_max": candidate_count_max, "empty_candidate_pool_count": candidate_empty,
        },
        "labels": list(LABELS),
        "main_metrics": methods,
        "best_strict_method": best_strict_name,
        "best_learned_method": best_name,
        "oracle_metrics": oracle_metrics,
        "transfer_diagnostics": transfer,
        "ranking_metrics": ranking_metrics,
        "training_summaries": training_summaries,
        "prior": {"metadata": prior_meta, "values": {str(k): float(v) for k, v in prior.items()}},
        "shuffle_metrics": {"branches": shuffle, "best_rgb_drop_pp": float(best_rgb_drop if np.isfinite(best_rgb_drop) else 0.0), "best_geometry_drop_pp": float(shuffle["best_geometry"]["accuracy_drop_pp"])},
        "occlusion_metrics": {"definition": "bottom tertile of existing frame-0 Stay scene visibility", "count": int(high_mask.sum()), "threshold": float(np.quantile(stay_visibility, 1.0 / 3.0)), "methods": high_methods, "best_learned_gain_over_static_pp": float(100.0 * (learned_high - static_high))},
        "viewpoint_distribution": distribution,
        "switch_analysis": switch,
        "scene_audit": scene_audit,
        "historical_method_inventory": _historical_inventory(REPO_ROOT),
        "protocol_audit": {
            "formal_task": "Frame0 -> exactly one Stage-A legal candidate -> selected real O1 alone",
            "action_set": "candidate-only Stage-A legal candidate_pool; current/stay excluded from main action set",
            "candidate_count_gate": {"mean": candidate_count_mean, "min": candidate_count_min, "max": candidate_count_max, "empty": candidate_empty},
            "yaw8_stgcn_checkpoint_sha256": _sha256(data_root / YAW8_STGCN_REL),
            "yaw8_shared_head_checkpoint_sha256": _sha256(head_path),
            "yaw8_cache_train": train_cache_meta,
            "yaw8_cache_moving_val": val_cache_meta,
            "frame0_dino_train": train_dino_meta,
            "frame0_dino_moving_val": val_dino_meta,
            "visibility_target_signature_checked": True,
            "test_used": False,
        },
        "leakage_audit": {
            "policy_test_read": False, "moving_val_used_for_training": False, "moving_val_used_for_checkpoint_selection": False,
            "candidate_observation_used_before_selection": False, "candidate_feature_or_logits_used_before_selection": False,
            "gt_action_used_at_inference": False, "future_o0_frames_used": False, "new_rgb_generated": False, "new_skeleton_generated": False,
            "yaw8_encoder_modified": False, "yaw8_shared_head_modified": False, "future_candidate_outputs_used_for_train_targets_or_terminal_only": True,
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "seed": SEED, "epochs": EPOCHS, "elapsed_seconds": time.monotonic() - started},
    }
    historical = result["historical_method_inventory"]
    _write_json(output_root / "historical_method_inventory.json", historical)
    _write_json(output_root / "config.json", {"seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "formal_action_set": "candidate-only Stage-A legal candidate_pool", "recognizer": "frozen Yaw8 ST-GCN + frozen matched Policy-balanced head", "test_used": False})
    _write_json(output_root / "protocol_audit.json", result["protocol_audit"])
    utility_summary = {
        "definition": "Yaw8Fair frozen matched-head candidate utility; candidate-only slots",
        "train": {"contexts": len(train_rows), "candidate_count": int(candidate_mask_train.sum()), "true_logp_mean": float(np.nanmean(train_true[candidate_mask_train])), "true_logp_std": float(np.nanstd(train_true[candidate_mask_train])), "margin_mean": float(np.nanmean(train_margin[candidate_mask_train])), "margin_std": float(np.nanstd(train_margin[candidate_mask_train]))},
        "moving_val": {"contexts": len(moving_rows), "candidate_count": int(candidate_mask_val.sum()), "true_logp_mean": float(np.nanmean(val_true[candidate_mask_val])), "true_logp_std": float(np.nanstd(val_true[candidate_mask_val])), "margin_mean": float(np.nanmean(val_margin[candidate_mask_val])), "margin_std": float(np.nanstd(val_margin[candidate_mask_val]))},
        "test_used": False,
    }
    result["utility_summary"] = utility_summary
    _write_json(output_root / "yaw8_policy_utility_train.json", utility_summary["train"])
    _write_json(output_root / "yaw8_policy_utility_val_summary.json", utility_summary["moving_val"])
    _write_json(output_root / "baseline_metrics.json", {k: methods[k] for k in ("Stay", "Random legal", "StaticViewPrior")})
    _write_json(output_root / "visibility_family_metrics.json", {k: methods[k] for k in visibility_specs})
    _write_json(output_root / "utility_family_metrics.json", {k: methods[k] for k, *_ in branch_specs[:4]})
    _write_json(output_root / "prior_residual_metrics.json", {k: methods[k] for k, *_ in branch_specs[4:]})
    _write_json(output_root / "selector_training_metrics.json", training_summaries)
    _write_json(output_root / "ranking_metrics.json", ranking_metrics)
    _write_json(output_root / "shuffle_metrics.json", result["shuffle_metrics"])
    _write_json(output_root / "switch_analysis.json", switch)
    _write_json(output_root / "viewpoint_distribution.json", distribution)
    _write_json(output_root / "occlusion_metrics.json", result["occlusion_metrics"])
    _write_json(output_root / "per_class_metrics.json", {k: methods[k]["per_class"] for k in methods if isinstance(methods[k], dict) and "per_class" in methods[k]})
    _write_json(output_root / "oracle_metrics.json", oracle_metrics)
    _write_json(output_root / "historical_vs_yaw8_comparison.json", {"historical_inventory": historical, "yaw8_formal": {k: methods[k] for k in methods if isinstance(methods[k], dict) and "accuracy" in methods[k]}, "old_values_are_reference_only": True})
    _write_json(output_root / "leakage_audit.json", result["leakage_audit"])
    _write_json(output_root / "result.json", result)
    (output_root / "analysis.md").write_text(render_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.data_root is None:
        from activeview.core.paths import get_data_root

        args.data_root = get_data_root()
    result = run(args)
    best = max(
        ((k, v["accuracy"]) for k, v in result["main_metrics"].items()
         if isinstance(v, dict) and "accuracy" in v),
        key=lambda item: item[1],
    )
    print(json.dumps({"status": result["status"], "output": str(args.output_root.resolve()), "best": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
