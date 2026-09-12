#!/usr/bin/env python3
"""Train action-utility predictors from causal frame-0 visual evidence.

The predictor inputs are deliberately identical to the frame-0 visibility
experiment: current-view DINO global features and Stage-A legal geometry.
Frozen shared recognizer outputs are used only to construct Train targets and
to evaluate a selected real archived viewpoint.  Policy Test is never read.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    head_logits,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    EVAL_BATCH,
    OPTION_GEOMETRY_DIM,
    TRAIN_BATCH,
    _load_frame0_dino,
    _method_metrics,
    _option_geometry,
    _predict,
    _select,
    _terminal,
    _transitions,
    VisibilityPredictor,
)


NUM_CLASSES = len(LABELS)
MAX_OPTIONS = 22
SEED = 42
EPOCHS = 12
OBS_PER_RECORD = 16
TASK_TAU = 0.5
VIS_TAU = 0.1
VIS_AUX_WEIGHT = 0.5
DIAGNOSTIC_REL = "diagnostics/reduced12_dual_route_overnight"
VIS_TARGET_REL = "diagnostics/frame0_visibility_predictor_v1"
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/frame0_task_utility_predictor_v1"
CHECKPOINT_REL = "checkpoints/policy_reduced12_eight_placement_v1/frame0_task_utility_predictor_v1"
OLD_VIS_CHECKPOINT_REL = "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1/RGBGlobal+Geometry.pth"
SHARED_HEAD_REL = "checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth"
HISTORICAL_RESULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/historical_route1_shared_head_synergy/result.json"


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


def _load_options(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    shared_head: SharedHead,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = data_root / DIAGNOSTIC_REL
    path, meta_path = root / f"{split}_options.npz", root / f"{split}_options.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or str(metadata.get("signature")) != row_signature(rows):
        raise ValueError(f"{split} recognizer cache signature mismatch")
    if bool(metadata.get("test_used", True)):
        raise ValueError(f"{split} recognizer cache is marked as Test-derived")
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        logits = np.asarray(archive["logits"], dtype=np.float32)
        ids = np.asarray(archive["ids"], dtype=np.int64)
        mask = np.asarray(archive["mask"], dtype=bool)
    expected_shape = (len(rows), MAX_OPTIONS)
    if features.shape != (len(rows), MAX_OPTIONS, 256) or logits.shape != (*expected_shape, NUM_CLASSES):
        raise ValueError(f"unexpected {split} recognizer cache shape")
    if ids.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError(f"unexpected {split} recognizer action shape")
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if ids[index, valid].tolist() != expected:
            raise ValueError(f"action-set mismatch at {split}/{row['episode_id']}")
    # The option cache's raw logits are frozen ST-GCN outputs.  The formal
    # protocol uses the separately frozen shared adapted head, so recompute
    # these logits from the cached penultimate features before constructing
    # targets or terminal predictions.
    shared_head.eval()
    logits = head_logits(shared_head, features, device)
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    logp = (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)
    return logp, ids, mask


def _load_visibility_targets(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, np.ndarray]:
    root = data_root / VIS_TARGET_REL
    metadata = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or str(metadata.get("signature")) != row_signature(rows):
        raise ValueError(f"{split} visibility target signature mismatch")
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        target = {key: np.asarray(archive[key]) for key in archive.files}
    if target["scores"].shape != (len(rows), MAX_OPTIONS) or target["mask"].shape != (len(rows), MAX_OPTIONS):
        raise ValueError(f"unexpected {split} visibility target shape")
    if target["ids"].shape != (len(rows), MAX_OPTIONS) or not np.all(np.isfinite(target["scores"])[target["mask"]]):
        raise ValueError("invalid visibility target values")
    for index, row in enumerate(rows):
        valid = np.flatnonzero(target["mask"][index])
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if target["ids"][index, valid].astype(int).tolist() != expected:
            raise ValueError(f"visibility target action-set mismatch at {split}/{row['episode_id']}")
    return target


def _task_targets(logp: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_logp = np.full(logp.shape[:2], np.nan, dtype=np.float32)
    margin = np.full_like(true_logp, np.nan)
    for index in range(logp.shape[0]):
        label = int(labels[index])
        for slot in np.flatnonzero(mask[index]):
            values = logp[index, slot]
            true_logp[index, slot] = values[label]
            margin[index, slot] = values[label] - np.max(np.delete(values, label))
    return true_logp, margin


def _target_stats(values: np.ndarray, ids: np.ndarray, mask: np.ndarray, rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    valid = np.asarray(values)[mask]
    best_actions: list[int] = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        slot = int(active[np.argmax(values[index, active])])
        best_actions.append(int(ids[index, slot]))
    counts = Counter(str(value) for value in best_actions)
    return {
        "name": name,
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "best_action_distribution": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "oracle_stay_rate": float(np.mean([int(action == int(row["current_viewpoint_id"])) for action, row in zip(best_actions, rows)])),
    }


class TaskUtilityPredictor(nn.Module):
    """RGB-global/geometry scorer with a separate explicit Stay output."""

    def __init__(self, visual: bool, auxiliary_visibility: bool) -> None:
        super().__init__()
        self.visual = visual
        self.auxiliary_visibility = auxiliary_visibility
        context_dim = 768 if visual else 0
        output_dim = 2 if auxiliary_visibility else 1
        self.candidate_head = nn.Sequential(
            nn.Linear(context_dim + OPTION_GEOMETRY_DIM, 128), nn.GELU(), nn.Linear(128, output_dim)
        )
        self.stay_head = nn.Sequential(
            nn.Linear(context_dim + OPTION_GEOMETRY_DIM, 128), nn.GELU(), nn.Linear(128, output_dim)
        )

    def forward(self, geometry: torch.Tensor, tokens: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.visual:
            if tokens is None:
                raise ValueError("visual branch requires current DINO tokens")
            context = tokens.mean(dim=1)
        else:
            context = geometry.new_zeros((geometry.shape[0], 0))
        expanded = context.unsqueeze(1).expand(-1, geometry.shape[1], -1)
        candidate = self.candidate_head(torch.cat((expanded, geometry), dim=-1))
        stay = self.stay_head(torch.cat((context, geometry[:, 0]), dim=-1))
        output = torch.cat((stay[:, None], candidate[:, 1:]), dim=1)
        if self.auxiliary_visibility:
            return output[..., 0], output[..., 1]
        return output[..., 0], None


def _utility_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(pred[mask], target[mask])
    ranking_terms: list[torch.Tensor] = []
    for index in range(pred.shape[0]):
        active = mask[index]
        if int(active.sum()) >= 2:
            target_prob = torch.softmax(target[index, active] / tau, dim=0)
            ranking_terms.append(-(target_prob * torch.log_softmax(pred[index, active] / tau, dim=0)).sum())
    ranking = torch.stack(ranking_terms).mean() if ranking_terms else regression.new_zeros(())
    return regression + 0.5 * ranking, regression, ranking


def _sample_rows(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["record_id"]), []).append(index)
    selected: list[int] = []
    for record in sorted(groups):
        candidates = np.asarray(groups[record], dtype=np.int64)
        selected.extend(rng.choice(candidates, OBS_PER_RECORD, replace=len(candidates) < OBS_PER_RECORD).tolist())
    return np.asarray(selected, dtype=np.int64)


def _predict_task(model: TaskUtilityPredictor, geometry: np.ndarray, tokens: np.ndarray | None, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    utilities: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(geometry), EVAL_BATCH):
            sl = slice(start, start + EVAL_BATCH)
            geom = torch.from_numpy(geometry[sl]).to(device, non_blocking=True)
            tok = None if tokens is None else torch.from_numpy(np.asarray(tokens[sl], dtype=np.float32)).to(device, non_blocking=True)
            utility, auxiliary = model(geom, tok)
            utilities.append(utility.cpu().numpy())
            if auxiliary is not None:
                visibility.append(auxiliary.cpu().numpy())
    return np.concatenate(utilities), (np.concatenate(visibility) if visibility else None)


def _train_branch(
    name: str,
    visual: bool,
    auxiliary: bool,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    train_geometry: np.ndarray,
    val_geometry: np.ndarray,
    train_tokens: np.ndarray | None,
    val_tokens: np.ndarray | None,
    train_utility: np.ndarray,
    val_utility: np.ndarray,
    train_visibility: np.ndarray | None,
    val_visibility: np.ndarray | None,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    device: torch.device,
    checkpoint: Path,
    summary_path: Path,
) -> TaskUtilityPredictor:
    model = TaskUtilityPredictor(visual, auxiliary).to(device)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("recognizer") == "frozen_stgcn_plus_shared_head":
            model.load_state_dict(payload["state_dict"])
            return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    best = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sampled = _sample_rows(train_rows, rng)
        train_losses: list[float] = []
        for start in range(0, len(sampled), TRAIN_BATCH):
            index = sampled[start : start + TRAIN_BATCH]
            geometry = torch.from_numpy(train_geometry[index]).to(device, non_blocking=True)
            utility = torch.from_numpy(train_utility[index]).to(device, non_blocking=True)
            mask = torch.from_numpy(train_mask[index]).to(device, non_blocking=True)
            tokens = None if train_tokens is None else torch.from_numpy(np.asarray(train_tokens[index], dtype=np.float32)).to(device, non_blocking=True)
            predicted, predicted_visibility = model(geometry, tokens)
            total, _, _ = _utility_loss(predicted, utility, mask, TASK_TAU)
            if auxiliary and predicted_visibility is not None and train_visibility is not None:
                visibility = torch.from_numpy(train_visibility[index]).to(device, non_blocking=True)
                total = total + VIS_AUX_WEIGHT * _utility_loss(predicted_visibility, visibility, mask, VIS_TAU)[0]
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(total.detach().cpu()))
        model.eval()
        val_total: list[float] = []
        val_task: list[float] = []
        val_aux: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_rows), EVAL_BATCH):
                sl = slice(start, start + EVAL_BATCH)
                geometry = torch.from_numpy(val_geometry[sl]).to(device, non_blocking=True)
                utility = torch.from_numpy(val_utility[sl]).to(device, non_blocking=True)
                mask = torch.from_numpy(val_mask[sl]).to(device, non_blocking=True)
                tokens = None if val_tokens is None else torch.from_numpy(np.asarray(val_tokens[sl], dtype=np.float32)).to(device, non_blocking=True)
                predicted, predicted_visibility = model(geometry, tokens)
                task_loss = _utility_loss(predicted, utility, mask, TASK_TAU)[0]
                total_loss = task_loss
                if auxiliary and predicted_visibility is not None and val_visibility is not None:
                    visibility = torch.from_numpy(val_visibility[sl]).to(device, non_blocking=True)
                    aux_loss = _utility_loss(predicted_visibility, visibility, mask, VIS_TAU)[0]
                    total_loss = total_loss + VIS_AUX_WEIGHT * aux_loss
                    val_aux.append(float(aux_loss.cpu()))
                val_task.append(float(task_loss.cpu()))
                val_total.append(float(total_loss.cpu()))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_total_loss": float(np.mean(val_total)),
            "val_utility_loss": float(np.mean(val_task)),
            "val_visibility_loss": float(np.mean(val_aux)) if val_aux else None,
            "observations": int(len(sampled)),
            "records": int(len({str(row['record_id']) for row in train_rows})),
        }
        history.append(record)
        print(f"[{name}] epoch={epoch:02d} train={record['train_loss']:.6f} val_total={record['val_total_loss']:.6f}", flush=True)
        if record["val_total_loss"] < best:
            best = float(record["val_total_loss"])
            best_epoch = epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "branch": name,
                    "epoch": epoch,
                    "seed": SEED,
                    "recognizer": "frozen_stgcn_plus_shared_head",
                },
                checkpoint,
            )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {
        "branch": name,
        "seed": SEED,
        "epochs": EPOCHS,
        "observations_per_record": OBS_PER_RECORD,
        "best_epoch": best_epoch,
        "best_val_total_loss": best,
        "checkpoint_selection": "minimum Val total task loss (utility loss + fixed visibility auxiliary where applicable)",
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model.eval()


def _ranking_metrics(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    valid_pred, valid_target = predicted[mask], target[mask]
    within: list[float] = []
    top1, top3 = [], []
    stay_move: list[int] = []
    for index in range(len(rows)):
        active = np.flatnonzero(mask[index])
        if active.size < 2:
            continue
        within.append(correlation(predicted[index, active], target[index, active], spearman=True))
        target_order = active[np.argsort(-target[index, active])]
        predicted_order = active[np.argsort(-predicted[index, active])]
        top1.append(int(predicted_order[0] == target_order[0]))
        top3.append(int(target_order[0] in predicted_order[:3]))
        stay_move.append(int((predicted_order[0] == 0) == (target_order[0] == 0)))
    target_best_stay = []
    for index, row in enumerate(rows):
        active = np.flatnonzero(mask[index])
        target_best_stay.append(int(active[np.argmax(target[index, active])] == 0))
    return {
        "branch": name,
        "mae": float(np.mean(np.abs(valid_pred - valid_target))),
        "rmse": float(np.sqrt(np.mean((valid_pred - valid_target) ** 2))),
        "candidate_level_spearman": correlation(valid_pred, valid_target, spearman=True),
        "within_context_spearman_mean": float(np.mean(within)) if within else 0.0,
        "within_context_spearman_median": float(np.median(within)) if within else 0.0,
        "gt_utility_oracle_top1_overlap": float(np.mean(top1)) if top1 else 0.0,
        "gt_utility_oracle_top3_overlap": float(np.mean(top3)) if top3 else 0.0,
        "stay_move_agreement": float(np.mean(stay_move)) if stay_move else 0.0,
        "oracle_stay_rate": float(np.mean(target_best_stay)),
    }


def _historical_actions(rows: Sequence[Mapping[str, Any]]) -> list[int] | None:
    if not HISTORICAL_RESULT.is_file():
        return None
    payload = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
    actions = payload.get("selected_actions")
    if not isinstance(actions, list) or len(actions) != len(rows):
        return None
    result = [int(value) for value in actions]
    for row, action in zip(rows, result):
        legal = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if action not in legal:
            raise ValueError(f"historical action outside legal set: {row['episode_id']}/{action}")
    return result


def _analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    branches = ["GeometryOnly-TrueLogP", "RGBGlobal-TrueLogP", "RGBGlobal-Margin", "RGBGlobal-TrueLogP+VisibilityAux"]
    best = max(branches, key=lambda name: float(methods[name]["accuracy"]))
    best_acc = float(methods[best]["accuracy"])
    if best_acc <= 0.505:
        decision = "KILL frame0 task-utility route"
        next_step = "move to short-prefix active recognition"
    elif best_acc < 0.53:
        decision = "WEAK KEEP frame0 task-utility route"
        next_step = "retain as baseline; do not add complexity"
    else:
        decision = "STRONG KEEP frame0 task-utility route"
        next_step = "continue frame0 task-aware research"
    rgb_gain = methods["RGBGlobal-TrueLogP"]["accuracy"] - methods["GeometryOnly-TrueLogP"]["accuracy"]
    aux_gain = methods["RGBGlobal-TrueLogP+VisibilityAux"]["accuracy"] - methods["RGBGlobal-TrueLogP"]["accuracy"]
    lines = [
        "# Frame-0 Task-Utility Predictor",
        "",
        "Training split: Policy Train",
        "Model-selection split: Moving Val",
        "Policy Test used: false",
        "",
        "Input:",
        "strict current frame-0 RGB DINO global feature + candidate/current geometry",
        "",
        "Future motion input: false",
        "candidate observation input: false",
        "current full-action feature input: false",
        "GT action input: false",
        "recognizer output input: false",
        "",
        "GT action / candidate recognizer outputs: training supervision only",
        "Final recognizer: frozen ST-GCN encoder + frozen shared head",
        "",
        "## Moving-Val results",
        "",
        "| Method | Target | Accuracy | Macro-F1 | Move rate |",
        "|---|---|---:|---:|---:|",
    ]
    for name, target in (
        ("Stay", "—"), ("Random legal", "—"), ("RGBGlobal Visibility", "visibility"),
        *[(branch, "GT-TrueLogP" if "Margin" not in branch else "GT-Margin") for branch in branches],
        ("Historical Route-1 + Shared", "old task-aware"), ("GT-TrueLogP Oracle", "privileged"),
    ):
        metric = methods.get(name)
        if metric is not None:
            lines.append(f"| {name} | {target} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('move_rate', 0.0):.6f} |")
    lines.extend([
        "",
        "## Utility ranking",
        "",
        "| Branch | MAE | RMSE | Candidate Spearman | Within-context Spearman | Top-1 overlap | Top-3 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for branch in branches:
        metric = result["ranking_metrics"][branch]
        lines.append(f"| {branch} | {metric['mae']:.6f} | {metric['rmse']:.6f} | {metric['candidate_level_spearman']:.6f} | {metric['within_context_spearman_mean']:.6f} | {metric['gt_utility_oracle_top1_overlap']:.6f} | {metric['gt_utility_oracle_top3_overlap']:.6f} |")
    audit = result["target_audit"]
    train_target = audit["train_gt_true_logp"]
    val_target = audit["val_gt_true_logp"]
    lines.extend([
        "",
        "## Supervision audit",
        "",
        f"Train/Val contexts: {audit['train_contexts']} / {audit['val_contexts']}; records: {result['population']['train_records']} / {result['population']['val_records']}.",
        f"Stay coverage: {audit['train_stay_coverage']:.6f} (Train), {audit['val_stay_coverage']:.6f} (Val); candidate-slot coverage: {audit['train_candidate_coverage']:.6f}, {audit['val_candidate_coverage']:.6f}.",
        f"GT-TrueLogP mean/std: {train_target['mean']:.6f}/{train_target['std']:.6f} (Train), {val_target['mean']:.6f}/{val_target['std']:.6f} (Val); ranges: [{val_target['min']:.6f}, {val_target['max']:.6f}] on Val.",
        "Action-set viewpoint identity was checked against the recognizer and visibility caches; all target recognizer outputs are supervision/terminal-only.",
        "",
        "## High-occlusion subset",
        "",
        f"Definition: bottom tertile of frame-0 Stay visibility ({result['occlusion_stratified_metrics']['count']} contexts).",
        "",
        "| Selector | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ])
    high_names = ["Random legal", "RGBGlobal Visibility", *branches, "Historical Route-1 + Shared", "GT-TrueLogP Oracle"]
    for name in high_names:
        metric = result["occlusion_stratified_metrics"]["methods"].get(name)
        if metric is not None:
            lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    visibility_transition = result["context_transitions"]["best_task_vs_rgb_visibility"]
    historical_transition = result["context_transitions"]["best_task_vs_historical"]
    disagreement = result["context_transitions"].get("visibility_oracle_disagreement", {})
    shortcut = result["shortcut_audit"]
    lines.extend([
        "",
        "## Error transitions and shortcut audit",
        "",
        f"Best task vs RGBGlobal Visibility: task-correct/visibility-wrong={visibility_transition['left_correct_right_wrong']}, task-wrong/visibility-correct={visibility_transition['left_wrong_right_correct']}, both-correct={visibility_transition['both_correct']}, both-wrong={visibility_transition['both_wrong']}.",
        f"Best task vs Historical Route-1: new-correct/old-wrong={historical_transition['left_correct_right_wrong']}, new-wrong/old-correct={historical_transition['left_wrong_right_correct']}, both-correct={historical_transition['both_correct']}, both-wrong={historical_transition['both_wrong']}.",
        f"On {disagreement.get('count', 0)} disagreements with the frame-0 visibility oracle, task-correct/visibility-wrong={disagreement.get('task_correct_visibility_wrong', 0)}, task-wrong/visibility-correct={disagreement.get('task_wrong_visibility_correct', 0)}, mean selected GT-TrueLogP difference={disagreement.get('mean_gt_true_logp_difference_task_minus_visibility', 0.0):+.6f}.",
        f"RGB shuffle diagnostic: normal Accuracy={shortcut['normal']['accuracy']:.6f}, shuffled Accuracy={shortcut['rgb_shuffled']['accuracy']:.6f}, change={shortcut['rgb_shuffled']['accuracy'] - shortcut['normal']['accuracy']:+.6f}.",
        "",
        f"Best new task-aware branch: **{best}** (Accuracy {best_acc:.6f}).",
        f"RGB contribution (RGBGlobal-TrueLogP minus GeometryOnly-TrueLogP): {rgb_gain:+.6f} Accuracy.",
        f"Visibility auxiliary contribution: {aux_gain:+.6f} Accuracy.",
        f"Decision: **{decision}**.",
        f"Next step: **{next_step}** (no automatic follow-up experiment).",
        "",
        "High-occlusion, transition and RGB-shuffle details are stored in the companion JSON files.",
        "",
        "```text",
        "policy_test_used=false",
        "training_split=Policy Train",
        "evaluation_split=Moving Val",
        "future_candidate_observation_used=false",
        "gt_action_used_for_predictor=false",
        "recognizer_output_used_for_predictor=false",
        "```",
    ])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    train_rows, val_rows = load_rows(data_root)
    shared_head_checkpoint = data_root / SHARED_HEAD_REL
    if not shared_head_checkpoint.is_file():
        raise FileNotFoundError(shared_head_checkpoint)
    shared_head = SharedHead().to(device)
    shared_head.load_state_dict(torch.load(shared_head_checkpoint, map_location=device, weights_only=False)["state_dict"])
    shared_head.eval()
    train_logp, train_ids, train_mask = _load_options(data_root, train_rows, "train", shared_head, device)
    val_logp, val_ids, val_mask = _load_options(data_root, val_rows, "val", shared_head, device)
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    train_utility, train_margin = _task_targets(train_logp, train_labels, train_mask)
    val_utility, val_margin = _task_targets(val_logp, val_labels, val_mask)
    train_visibility = _load_visibility_targets(data_root, train_rows, "train")
    val_visibility = _load_visibility_targets(data_root, val_rows, "val")
    stats_path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/stage_c_feature_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_geometry, _, _ = _option_geometry(train_rows, stats)
    val_geometry, _, _ = _option_geometry(val_rows, stats)
    train_tokens, train_dino_meta = _load_frame0_dino(data_root, train_rows, "train", device)
    val_tokens, val_dino_meta = _load_frame0_dino(data_root, val_rows, "val", device)
    target_audit = {
        "train_contexts": len(train_rows),
        "val_contexts": len(val_rows),
        "train_stay_coverage": float(train_mask[:, 0].mean()),
        "val_stay_coverage": float(val_mask[:, 0].mean()),
        "train_candidate_coverage": float(train_mask[:, 1:].mean()),
        "val_candidate_coverage": float(val_mask[:, 1:].mean()),
        "train_gt_true_logp": _target_stats(train_utility, train_ids, train_mask, train_rows, "GT-TrueLogP"),
        "val_gt_true_logp": _target_stats(val_utility, val_ids, val_mask, val_rows, "GT-TrueLogP"),
        "train_gt_margin": _target_stats(train_margin, train_ids, train_mask, train_rows, "GT-Margin"),
        "val_gt_margin": _target_stats(val_margin, val_ids, val_mask, val_rows, "GT-Margin"),
        "action_set": "current/Stay + Stage-A legal candidate_pool",
        "target_viewpoint_identity_checked": True,
        "recognizer": "frozen ST-GCN encoder + frozen shared head",
        "test_used": False,
    }
    methods: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    actions_by_method: dict[str, list[int]] = {}
    stay_actions = [int(row["current_viewpoint_id"]) for row in val_rows]
    rng = np.random.default_rng(SEED)
    random_actions = [int(rng.choice([int(row["current_viewpoint_id"])] + [int(x) for x in row["candidate_ids"]])) for row in val_rows]
    # Reuse the previous visibility checkpoint; this is a fixed baseline, not retraining.
    visibility_model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    visibility_checkpoint = data_root / OLD_VIS_CHECKPOINT_REL
    if not visibility_checkpoint.is_file():
        raise FileNotFoundError(visibility_checkpoint)
    visibility_model.load_state_dict(torch.load(visibility_checkpoint, map_location=device, weights_only=False)["state_dict"])
    visibility_scores = _predict(visibility_model.eval(), val_geometry, val_tokens, device)
    frame0_actions = _select(val_rows, visibility_scores, val_visibility["mask"])
    gt_actions = _select(val_rows, val_utility, val_mask)
    baseline_actions = [("Stay", stay_actions), ("Random legal", random_actions), ("RGBGlobal Visibility", _select(val_rows, visibility_scores, val_mask)), ("GT-TrueLogP Oracle", gt_actions)]
    historical = _historical_actions(val_rows)
    if historical is not None:
        baseline_actions.append(("Historical Route-1 + Shared", historical))
    for name, actions in baseline_actions:
        actions_by_method[name] = list(actions)
        predictions[name] = _terminal(val_logp, val_ids, val_mask, actions, val_labels)
        methods[name] = _method_metrics(name, val_rows, val_labels, predictions[name], actions)
    output_root.mkdir(parents=True, exist_ok=True)
    branch_specs = (
        ("GeometryOnly-TrueLogP", False, False, train_utility, val_utility, None, None),
        ("RGBGlobal-TrueLogP", True, False, train_utility, val_utility, None, None),
        ("RGBGlobal-Margin", True, False, train_margin, val_margin, None, None),
        ("RGBGlobal-TrueLogP+VisibilityAux", True, True, train_utility, val_utility, train_visibility["scores"], val_visibility["scores"]),
    )
    ranking_metrics: dict[str, Any] = {}
    for name, visual, auxiliary, target_train, target_val, aux_train, aux_val in branch_specs:
        train_tok = train_tokens if visual else None
        val_tok = val_tokens if visual else None
        checkpoint = data_root / CHECKPOINT_REL / f"{name}.pth"
        model = _train_branch(name, visual, auxiliary, train_rows, val_rows, train_geometry, val_geometry, train_tok, val_tok, target_train, target_val, aux_train, aux_val, train_mask, val_mask, device, checkpoint, output_root / f"{name}_training.json")
        utility_scores, _ = _predict_task(model, val_geometry, val_tok, device)
        actions = _select(val_rows, utility_scores, val_mask)
        actions_by_method[name] = list(actions)
        predictions[name] = _terminal(val_logp, val_ids, val_mask, actions, val_labels)
        methods[name] = _method_metrics(name, val_rows, val_labels, predictions[name], actions)
        ranking_metrics[name] = _ranking_metrics(name, val_rows, utility_scores, target_val, val_mask)
    branches = [spec[0] for spec in branch_specs]
    methods["RGBGlobal Visibility"]["selector"] = "RGBGlobal+Geometry visibility checkpoint"
    high_occlusion = np.asarray(val_visibility["scores"])[:, 0] < np.quantile(np.asarray(val_visibility["scores"])[:, 0], 1.0 / 3.0)
    occlusion_methods = ["Random legal", "RGBGlobal Visibility", *branches, "Historical Route-1 + Shared", "GT-TrueLogP Oracle"]
    occlusion: dict[str, Any] = {"definition": "bottom tertile of frame-0 Stay scene visibility", "count": int(high_occlusion.sum()), "methods": {}}
    indices = np.flatnonzero(high_occlusion)
    for name in occlusion_methods:
        if name in methods:
            occlusion["methods"][name] = _method_metrics(name, [val_rows[int(i)] for i in indices], val_labels[high_occlusion], predictions[name][high_occlusion], [actions_by_method[name][int(i)] for i in indices])
    best_branch = max(branches, key=lambda name: float(methods[name]["accuracy"]))
    transitions = {
        "best_task_vs_rgb_visibility": _transitions(val_labels, predictions["RGBGlobal Visibility"], predictions[best_branch]),
        "best_task_vs_historical": _transitions(val_labels, predictions.get("Historical Route-1 + Shared", predictions["RGBGlobal Visibility"]), predictions[best_branch]),
        "visibility_oracle_disagreement": {},
    }
    visibility_oracle_actions = frame0_actions
    disagreement = np.asarray(actions_by_method[best_branch]) != np.asarray(visibility_oracle_actions)
    if disagreement.any():
        task_correct = predictions[best_branch][disagreement] == val_labels[disagreement]
        visibility_correct = predictions["RGBGlobal Visibility"][disagreement] == val_labels[disagreement]

        def _selected_target(row_indices: Sequence[int], actions: Sequence[int], values: np.ndarray) -> np.ndarray:
            selected: list[float] = []
            for subset_index, (row_index, action) in enumerate(zip(row_indices, actions)):
                slots = np.flatnonzero((val_ids[row_index] == int(action)) & val_mask[row_index])
                if slots.size != 1:
                    raise ValueError(f"cannot resolve selected action {action}")
                selected.append(float(values[subset_index, int(slots[0])]))
            return np.asarray(selected, dtype=np.float32)

        disagree_indices = np.flatnonzero(disagreement)
        task_selected = [actions_by_method[best_branch][int(i)] for i in disagree_indices]
        visibility_selected = [visibility_oracle_actions[int(i)] for i in disagree_indices]
        task_values = _selected_target(disagree_indices, task_selected, val_utility[disagree_indices])
        visibility_values = _selected_target(disagree_indices, visibility_selected, val_utility[disagree_indices])
        transitions["visibility_oracle_disagreement"] = {
            "count": int(disagreement.sum()),
            "task_correct_visibility_wrong": int(np.sum(task_correct & ~visibility_correct)),
            "task_wrong_visibility_correct": int(np.sum(~task_correct & visibility_correct)),
            "both_correct": int(np.sum(task_correct & visibility_correct)),
            "both_wrong": int(np.sum(~task_correct & ~visibility_correct)),
            "mean_gt_true_logp_difference_task_minus_visibility": float(np.mean(task_values - visibility_values)),
            "mean_gt_true_logp_task_selected": float(np.mean(task_values)),
            "mean_gt_true_logp_visibility_selected": float(np.mean(visibility_values)),
        }
    # Shuffle only current RGB/DINO context for the trained RGBGlobal TrueLogP branch.
    shuffle_rng = np.random.default_rng(SEED)
    permutation = shuffle_rng.permutation(len(val_rows))
    rgb_logp_model = TaskUtilityPredictor(True, False).to(device)
    rgb_logp_checkpoint = data_root / CHECKPOINT_REL / "RGBGlobal-TrueLogP.pth"
    rgb_logp_model.load_state_dict(torch.load(rgb_logp_checkpoint, map_location=device, weights_only=False)["state_dict"])
    shuffled_scores, _ = _predict_task(rgb_logp_model.eval(), val_geometry, val_tokens[permutation], device)
    shuffled_actions = _select(val_rows, shuffled_scores, val_mask)
    shuffled_predictions = _terminal(val_logp, val_ids, val_mask, shuffled_actions, val_labels)
    shortcut = {
        "shuffle": "current frame-0 DINO global feature permuted across contexts; geometry unchanged",
        "normal": methods["RGBGlobal-TrueLogP"],
        "rgb_shuffled": _method_metrics("RGBGlobal-TrueLogP RGB-shuffled", val_rows, val_labels, shuffled_predictions, shuffled_actions),
        "permutation_seed": SEED,
        "test_used": False,
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_FRAME0_TASK_UTILITY_PREDICTOR_V1",
        "status": "COMPLETED",
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(r['record_id']) for r in train_rows}), "val_records": len({str(r['record_id']) for r in val_rows})},
        "labels": list(LABELS),
        "methods": methods,
        "ranking_metrics": ranking_metrics,
        "target_audit": target_audit,
        "occlusion_stratified_metrics": occlusion,
        "context_transitions": transitions,
        "shortcut_audit": shortcut,
        "baseline_reference": {"RGBGlobal Visibility": {"accuracy": 0.494841, "macro_f1": 0.493881}, "Historical Route-1 + Shared": {"accuracy": 0.509524, "macro_f1": 0.509806}, "GT-TrueLogP Oracle": {"accuracy": 0.753175, "macro_f1": 0.755358}},
        "task_gain_vs_visibility": {name: float(methods[name]["accuracy"] - methods["RGBGlobal Visibility"]["accuracy"]) for name in branches},
        "protocol": {"input": "current frame-0 DINO global + candidate/current geometry", "action_set": "current/Stay + Stage-A legal candidate_pool", "loss": "SmoothL1 + 0.5 listwise; tau=0.5", "visibility_aux_weight": VIS_AUX_WEIGHT, "checkpoint_selection": "minimum Val total task loss", "terminal": "selected real archived 30-frame skeleton through frozen ST-GCN + shared head"},
        "flags": {"policy_test_used": False, "train_supervision_used": True, "new_rgb_generated": False, "new_skeleton_generated": False, "future_candidate_rgb_used": False, "future_candidate_skeleton_used_for_predictor": False, "current_full_sequence_feature_used": False, "current_action_posterior_used": False, "gt_action_used_for_predictor": False, "recognizer_output_used_for_predictor": False, "frozen_stgcn_modified": False, "deployable": True},
        "artifacts": {"frame0_dino_train": train_dino_meta, "frame0_dino_val": val_dino_meta, "train_options": str((data_root / DIAGNOSTIC_REL / "train_options.npz").resolve()), "val_options": str((data_root / DIAGNOSTIC_REL / "val_options.npz").resolve()), "old_visibility_checkpoint": str(visibility_checkpoint.resolve()), "shared_head_checkpoint": str(shared_head_checkpoint.resolve())},
        "runtime": {"device": str(device), "seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD},
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"branches": branches, "option_geometry_dim": OPTION_GEOMETRY_DIM, "task_tau": TASK_TAU, "visibility_aux_weight": VIS_AUX_WEIGHT, "seed": SEED, "epochs": EPOCHS, "test_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "training_summary.json").write_text(json.dumps({name: json.loads((output_root / f"{name}_training.json").read_text(encoding="utf-8")) for name in branches}, indent=2) + "\n", encoding="utf-8")
    (output_root / "per_branch_metrics.json").write_text(json.dumps({name: {"downstream": methods[name], "ranking": ranking_metrics[name], "gain_vs_visibility": result["task_gain_vs_visibility"][name]} for name in branches}, indent=2) + "\n", encoding="utf-8")
    (output_root / "ranking_metrics.json").write_text(json.dumps(ranking_metrics, indent=2) + "\n", encoding="utf-8")
    (output_root / "occlusion_stratified_metrics.json").write_text(json.dumps(occlusion, indent=2) + "\n", encoding="utf-8")
    (output_root / "context_transitions.json").write_text(json.dumps(transitions, indent=2) + "\n", encoding="utf-8")
    (output_root / "shortcut_audit.json").write_text(json.dumps(shortcut, indent=2) + "\n", encoding="utf-8")
    (output_root / "target_audit.json").write_text(json.dumps(target_audit, indent=2) + "\n", encoding="utf-8")
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
