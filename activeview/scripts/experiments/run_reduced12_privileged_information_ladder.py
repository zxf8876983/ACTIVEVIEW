#!/usr/bin/env python3
"""Train/Val-only privileged-information ladder for reduced12 frame-0 NBV.

The ladder keeps the action set and frozen recognizer fixed while adding
privileged action identity and frame-0 scene visibility to a small utility
predictor.  Candidate recognizer outputs are targets/oracle values only and
are never supplied as learned-branch inputs.  Policy Test is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
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
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import (
    SharedHead,
    load_rows,
    row_signature,
)
from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import (
    _load_options,
    _load_visibility_targets,
)
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import _option_geometry
from activeview.scripts.experiments.privileged_information_ladder_support import (
    action_view_preference as _action_view_preference,
    analysis as _analysis,
    json_matrix as _json_matrix,
    ranking_metrics as _ranking_metrics,
    target_stats as _target_stats,
    visibility_interaction as _visibility_interaction,
)

NUM_CLASSES = len(LABELS)
MAX_OPTIONS = 22
NUM_VIEWS = 32
SEED = 42
EPOCHS = 12
OBS_PER_RECORD = 16
TRAIN_BATCH = 512
EVAL_BATCH = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4
LISTWISE_TAU = 0.5

SHARED_HEAD_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/"
    "view_agnostic_frozen_encoder_head/shared_head_best.pth"
)
CHECKPOINT_REL = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/privileged_information_ladder"
)
OUTPUT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/privileged_information_ladder"
VISIBILITY_REL = Path("diagnostics/frame0_visibility_predictor_v1")
HISTORICAL_RESULT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/frame0_task_utility_predictor_v1/result.json"

LEARNED_BRANCHES = (
    "GeometryOnly Utility",
    "RealVisibility+Geometry",
    "GTAction+Geometry",
    "GTAction+RealVisibility",
    "GTAction+RealVisibility+Geometry",
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


def _slug(name: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")


def _validate_action_cache(
    rows: Sequence[Mapping[str, Any]],
    ids: np.ndarray,
    mask: np.ndarray,
    name: str,
) -> None:
    if ids.shape != (len(rows), MAX_OPTIONS) or mask.shape != ids.shape:
        raise ValueError(f"{name} action cache shape mismatch: {ids.shape}/{mask.shape}")
    for index, row in enumerate(rows):
        valid = np.flatnonzero(mask[index])
        expected = [int(row["current_viewpoint_id"])] + [int(value) for value in row["candidate_ids"]]
        if ids[index, valid].astype(int).tolist() != expected:
            raise ValueError(f"{name} action-set mismatch for {row['episode_id']}")


def _load_visibility(data_root: Path, rows: Sequence[Mapping[str, Any]], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    root = data_root / VISIBILITY_REL
    metadata = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
    if int(metadata.get("rows", -1)) != len(rows) or str(metadata.get("signature")) != row_signature(rows):
        raise ValueError(f"{split} frame-0 visibility signature mismatch")
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        scores = np.asarray(archive["scores"], dtype=np.float32)
        ids = np.asarray(archive["ids"], dtype=np.int64)
        mask = np.asarray(archive["mask"], dtype=bool)
    _validate_action_cache(rows, ids, mask, f"{split} visibility")
    if scores.shape != (len(rows), MAX_OPTIONS) or not np.isfinite(scores[mask]).all():
        raise ValueError(f"{split} visibility values are invalid")
    return scores, ids, mask, metadata


def _utility_targets(logp: np.ndarray, labels: np.ndarray) -> np.ndarray:
    target = np.zeros(logp.shape[:2], dtype=np.float32)
    for index, label in enumerate(labels):
        target[index] = logp[index, :, int(label)]
    if not np.isfinite(target).all():
        raise ValueError("non-finite utility target")
    return target


def _build_inputs(
    branch: str,
    geometry: np.ndarray,
    visibility: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    n_rows, n_slots = geometry.shape[:2]
    if geometry.shape[:2] != visibility.shape:
        raise ValueError("geometry/visibility action shape mismatch")
    if branch == "GeometryOnly Utility":
        return geometry.astype(np.float32, copy=False)
    # Visibility targets use NaN only in padded action slots.  Padding is
    # excluded by the action mask, but must still be finite during the MLP
    # forward pass to avoid contaminating shared weights with NaN gradients.
    visibility_feature = np.nan_to_num(visibility[..., None].astype(np.float32), nan=0.0)
    if branch == "RealVisibility+Geometry":
        return np.concatenate((geometry, visibility_feature), axis=-1)
    one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[labels][:, None, :]
    one_hot = np.broadcast_to(one_hot, (n_rows, n_slots, NUM_CLASSES))
    if branch == "GTAction+Geometry":
        return np.concatenate((one_hot, geometry), axis=-1)
    stay = np.zeros((n_rows, n_slots, 1), dtype=np.float32)
    stay[:, 0, 0] = 1.0
    if branch == "GTAction+RealVisibility":
        return np.concatenate((one_hot, visibility_feature, stay), axis=-1)
    if branch == "GTAction+RealVisibility+Geometry":
        return np.concatenate((one_hot, visibility_feature, geometry), axis=-1)
    raise ValueError(f"unknown branch: {branch}")


class UtilityMLP(nn.Module):
    """Shared architecture for every learned ladder branch."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shape = inputs.shape[:-1]
        return self.network(inputs.reshape(-1, inputs.shape[-1])).reshape(shape)


def _utility_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression = F.smooth_l1_loss(predicted[mask], target[mask])
    ranking_terms: list[torch.Tensor] = []
    for index in range(predicted.shape[0]):
        active = mask[index]
        if int(active.sum()) >= 2:
            target_prob = torch.softmax(target[index, active] / LISTWISE_TAU, dim=0)
            ranking_terms.append(
                -(target_prob * torch.log_softmax(predicted[index, active] / LISTWISE_TAU, dim=0)).sum()
            )
    ranking = torch.stack(ranking_terms).mean() if ranking_terms else regression.new_zeros(())
    return regression + 0.5 * ranking, regression, ranking


def _sample_rows(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["record_id"])].append(index)
    selected: list[int] = []
    for record in sorted(groups):
        candidates = np.asarray(groups[record], dtype=np.int64)
        replace = len(candidates) < OBS_PER_RECORD
        selected.extend(rng.choice(candidates, OBS_PER_RECORD, replace=replace).tolist())
    return np.asarray(selected, dtype=np.int64)


def _predict(model: UtilityMLP, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), EVAL_BATCH):
            batch = torch.from_numpy(np.asarray(inputs[start : start + EVAL_BATCH], dtype=np.float32)).to(
                device, non_blocking=True
            )
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _train_branch(
    branch: str,
    train_inputs: np.ndarray,
    val_inputs: np.ndarray,
    train_target: np.ndarray,
    val_target: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    output_root: Path,
    device: torch.device,
) -> tuple[UtilityMLP, dict[str, Any]]:
    slug = _slug(branch)
    checkpoint = data_root / CHECKPOINT_REL / f"{slug}.pth"
    summary_path = output_root / f"{slug}_training.json"
    model = UtilityMLP(train_inputs.shape[-1]).to(device)
    if checkpoint.is_file() and summary_path.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if int(payload.get("input_dim", -1)) != train_inputs.shape[-1] or bool(payload.get("test_used", True)):
            raise ValueError(f"checkpoint metadata mismatch for {branch}")
        model.load_state_dict(payload["state_dict"])
        return model.eval(), json.loads(summary_path.read_text(encoding="utf-8"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        sampled = _sample_rows(train_rows, rng)
        order = rng.permutation(len(sampled))
        model.train()
        train_losses: list[float] = []
        train_regression: list[float] = []
        train_ranking: list[float] = []
        for start in range(0, len(order), TRAIN_BATCH):
            indices = sampled[order[start : start + TRAIN_BATCH]]
            inputs = torch.from_numpy(train_inputs[indices]).to(device, non_blocking=True)
            target = torch.from_numpy(train_target[indices]).to(device, non_blocking=True)
            mask = torch.from_numpy(train_mask[indices]).to(device, non_blocking=True)
            predicted = model(inputs)
            total, regression, ranking = _utility_loss(predicted, target, mask)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(total.detach().cpu()))
            train_regression.append(float(regression.detach().cpu()))
            train_ranking.append(float(ranking.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        val_regression: list[float] = []
        val_ranking: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(val_inputs), EVAL_BATCH):
                sl = slice(start, start + EVAL_BATCH)
                inputs = torch.from_numpy(val_inputs[sl]).to(device, non_blocking=True)
                target = torch.from_numpy(val_target[sl]).to(device, non_blocking=True)
                mask = torch.from_numpy(val_mask[sl]).to(device, non_blocking=True)
                total, regression, ranking = _utility_loss(model(inputs), target, mask)
                val_losses.append(float(total.cpu()))
                val_regression.append(float(regression.cpu()))
                val_ranking.append(float(ranking.cpu()))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_regression": float(np.mean(train_regression)),
            "train_listwise": float(np.mean(train_ranking)),
            "val_total_loss": float(np.mean(val_losses)),
            "val_regression": float(np.mean(val_regression)),
            "val_listwise": float(np.mean(val_ranking)),
            "sampled_contexts": int(len(sampled)),
            "unique_records": int(len({str(row['record_id']) for row in train_rows})),
        }
        history.append(record)
        print(
            f"[{branch}] epoch={epoch:02d} train={record['train_loss']:.6f} "
            f"val={record['val_total_loss']:.6f}",
            flush=True,
        )
        if record["val_total_loss"] < best_loss:
            best_loss = float(record["val_total_loss"])
            best_epoch = epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "branch": branch,
                    "input_dim": int(train_inputs.shape[-1]),
                    "epoch": epoch,
                    "seed": SEED,
                    "test_used": False,
                },
                checkpoint,
            )
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    summary = {
        "branch": branch,
        "seed": SEED,
        "epochs": EPOCHS,
        "input_dim": int(train_inputs.shape[-1]),
        "architecture": "Linear(input_dim,256)-GELU-Linear(256,256)-GELU-Linear(256,1)",
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "observations_per_record": OBS_PER_RECORD,
        "checkpoint_selection": "minimum Moving-Val utility prediction loss",
        "best_epoch": best_epoch,
        "best_val_total_loss": best_loss,
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "test_used": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return model.eval(), summary


def _select(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> list[int]:
    actions: list[int] = []
    del rows
    for index in range(ids.shape[0]):
        valid = np.flatnonzero(mask[index])
        ordered = sorted(
            valid.tolist(),
            key=lambda slot: (-float(scores[index, slot]), 0 if slot == 0 else 1, int(ids[index, slot])),
        )
        actions.append(int(ids[index, ordered[0]]))
    return actions


def _random_actions(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray) -> list[int]:
    rng = np.random.default_rng(SEED)
    return [int(rng.choice(ids[index, np.flatnonzero(mask[index])])) for index in range(len(rows))]


def _terminal(logp: np.ndarray, ids: np.ndarray, mask: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    predictions: list[int] = []
    for index, action in enumerate(actions):
        slots = np.flatnonzero((ids[index] == int(action)) & mask[index])
        if slots.size != 1:
            raise ValueError(f"selected action is not uniquely legal at row {index}: {action}")
        predictions.append(int(np.argmax(logp[index, int(slots[0])])))
    return np.asarray(predictions, dtype=np.int64)


def _metrics(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    actions: Sequence[int],
    deployable: bool,
) -> dict[str, Any]:
    result = classification(labels, predictions)
    move_rate = float(np.mean([int(action) != int(row["current_viewpoint_id"]) for action, row in zip(actions, rows)]))
    result.update(
        {
            "selector": name,
            "move_rate": move_rate,
            "stay_rate": 1.0 - move_rate,
            "deployable": bool(deployable),
            "test_used": False,
        }
    )
    return result


def _oracle_actions(target: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> list[int]:
    return _select([], target, ids, mask)  # rows are unused by _select


def _prior_tables(
    utility: np.ndarray,
    ids: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    global_sum = np.zeros(NUM_VIEWS, dtype=np.float64)
    global_count = np.zeros(NUM_VIEWS, dtype=np.int64)
    class_sum = np.zeros((NUM_CLASSES, NUM_VIEWS), dtype=np.float64)
    class_count = np.zeros((NUM_CLASSES, NUM_VIEWS), dtype=np.int64)
    class_all_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_all_count = np.zeros(NUM_CLASSES, dtype=np.int64)
    for index, label in enumerate(labels):
        for slot in np.flatnonzero(mask[index]):
            view = int(ids[index, slot])
            value = float(utility[index, slot])
            global_sum[view] += value
            global_count[view] += 1
            class_sum[int(label), view] += value
            class_count[int(label), view] += 1
            class_all_sum[int(label)] += value
            class_all_count[int(label)] += 1
    global_mean = np.divide(global_sum, global_count, out=np.full(NUM_VIEWS, np.nan), where=global_count > 0)
    class_mean = np.divide(class_sum, class_count, out=np.full_like(class_sum, np.nan), where=class_count > 0)
    class_fallback = np.divide(class_all_sum, class_all_count, out=np.full(NUM_CLASSES, np.nan), where=class_all_count > 0)
    overall = float(np.mean(utility[mask]))
    return {
        "global_mean": global_mean,
        "global_count": global_count,
        "class_mean": class_mean,
        "class_count": class_count,
        "class_fallback": class_fallback,
        "overall_mean": np.asarray(overall),
    }


def _select_global_prior(rows: Sequence[Mapping[str, Any]], ids: np.ndarray, mask: np.ndarray, table: Mapping[str, np.ndarray]) -> tuple[list[int], int]:
    values = np.asarray(table["global_mean"])
    fallback = float(table["overall_mean"])
    scores = np.asarray([[float(values[int(view)]) if np.isfinite(values[int(view)]) else fallback for view in row_ids] for row_ids in ids], dtype=np.float32)
    scores[~mask] = -np.inf
    return _select(rows, scores, ids, mask), int(np.sum(~np.isfinite(values)))


def _select_class_prior(
    rows: Sequence[Mapping[str, Any]],
    ids: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    table: Mapping[str, np.ndarray],
) -> tuple[list[int], int, int]:
    class_mean = np.asarray(table["class_mean"])
    class_fallback = np.asarray(table["class_fallback"])
    overall = float(table["overall_mean"])
    scores = np.zeros(ids.shape, dtype=np.float32)
    missing_pairs = 0
    for index, label in enumerate(labels):
        fallback = float(class_fallback[int(label)]) if np.isfinite(class_fallback[int(label)]) else overall
        for slot in np.flatnonzero(mask[index]):
            value = class_mean[int(label), int(ids[index, slot])]
            if not np.isfinite(value):
                missing_pairs += 1
                value = fallback
            scores[index, slot] = float(value)
    return _select(rows, scores, ids, mask), missing_pairs, int(mask.sum())


def run(args: argparse.Namespace) -> dict[str, Any]:
    _seed()
    device = _device(args.device)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    train_rows, val_rows = load_rows(data_root)
    train_labels = np.asarray([int(row["label_id"]) for row in train_rows], dtype=np.int64)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    shared_head_path = data_root / SHARED_HEAD_REL
    if not shared_head_path.is_file():
        raise FileNotFoundError(shared_head_path)
    shared_head = SharedHead().to(device)
    shared_head.load_state_dict(torch.load(shared_head_path, map_location=device, weights_only=False)["state_dict"])
    shared_head.eval()
    from activeview.scripts.experiments.run_reduced12_frame0_task_utility_predictor import _load_options as load_option_logp

    train_logp, train_ids, train_mask = load_option_logp(data_root, train_rows, "train", shared_head, device)
    val_logp, val_ids, val_mask = load_option_logp(data_root, val_rows, "val", shared_head, device)
    _validate_action_cache(train_rows, train_ids, train_mask, "train recognizer")
    _validate_action_cache(val_rows, val_ids, val_mask, "val recognizer")
    train_visibility, train_vis_ids, train_vis_mask, train_vis_meta = _load_visibility(data_root, train_rows, "train")
    val_visibility, val_vis_ids, val_vis_mask, val_vis_meta = _load_visibility(data_root, val_rows, "val")
    if not np.array_equal(train_ids, train_vis_ids) or not np.array_equal(train_mask, train_vis_mask):
        raise ValueError("train recognizer/visibility action cache mismatch")
    if not np.array_equal(val_ids, val_vis_ids) or not np.array_equal(val_mask, val_vis_mask):
        raise ValueError("val recognizer/visibility action cache mismatch")
    stats_path = data_root / "datasets/policy_reduced12_eight_placement_v1/stage_c/stage_c_feature_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    train_geometry, train_geom_ids, train_geom_mask = _option_geometry(train_rows, stats)
    val_geometry, val_geom_ids, val_geom_mask = _option_geometry(val_rows, stats)
    if not np.array_equal(train_ids, train_geom_ids) or not np.array_equal(train_mask, train_geom_mask):
        raise ValueError("train recognizer/geometry action cache mismatch")
    if not np.array_equal(val_ids, val_geom_ids) or not np.array_equal(val_mask, val_geom_mask):
        raise ValueError("val recognizer/geometry action cache mismatch")
    train_target = _utility_targets(train_logp, train_labels)
    val_target = _utility_targets(val_logp, val_labels)
    output_root.mkdir(parents=True, exist_ok=True)
    train_inputs = {branch: _build_inputs(branch, train_geometry, train_visibility, train_labels) for branch in LEARNED_BRANCHES}
    val_inputs = {branch: _build_inputs(branch, val_geometry, val_visibility, val_labels) for branch in LEARNED_BRANCHES}
    models: dict[str, UtilityMLP] = {}
    training: dict[str, Any] = {}
    predicted_scores: dict[str, np.ndarray] = {}
    for branch in LEARNED_BRANCHES:
        model, summary = _train_branch(
            branch,
            train_inputs[branch],
            val_inputs[branch],
            train_target,
            val_target,
            train_mask,
            val_mask,
            train_rows,
            data_root,
            output_root,
            device,
        )
        models[branch] = model
        training[branch] = summary
        predicted_scores[branch] = _predict(model, val_inputs[branch], device)
    prior = _prior_tables(train_target, train_ids, train_mask, train_labels)
    global_actions, missing_global_views = _select_global_prior(val_rows, val_ids, val_mask, prior)
    class_actions, missing_class_pairs, class_pair_total = _select_class_prior(val_rows, val_ids, val_mask, val_labels, prior)
    gt_actions = _oracle_actions(val_target, val_ids, val_mask)
    visibility_actions = _select(val_rows, val_visibility, val_ids, val_mask)
    random_actions = _random_actions(val_rows, val_ids, val_mask)
    stay_actions = [int(row["current_viewpoint_id"]) for row in val_rows]
    action_map: dict[str, list[int]] = {
        "Stay": stay_actions,
        "Random": random_actions,
        "Real Frame0 Visibility argmax": visibility_actions,
        "Global Viewpoint Prior": global_actions,
        "GTAction+ViewpointPrior": class_actions,
        "GT-TrueLogP Oracle": gt_actions,
    }
    action_map.update({branch: _select(val_rows, predicted_scores[branch], val_ids, val_mask) for branch in LEARNED_BRANCHES})
    methods: dict[str, Any] = {}
    method_information: dict[str, Any] = {
        "Stay": {"information": "none", "deployable": True},
        "Random": {"information": "legal action set", "deployable": True},
        "GeometryOnly Utility": {"information": "G", "deployable": True},
        "RGBGlobal Task+VisibilityAux (historical)": {"information": "RGB+G", "deployable": True},
        "Real Frame0 Visibility argmax": {"information": "V", "deployable": False},
        "RealVisibility+Geometry": {"information": "V+G", "deployable": False},
        "Global Viewpoint Prior": {"information": "global view prior", "deployable": True},
        "GTAction+ViewpointPrior": {"information": "Y+view ID", "deployable": False},
        "GTAction+Geometry": {"information": "Y+G", "deployable": False},
        "GTAction+RealVisibility": {"information": "Y+V", "deployable": False},
        "GTAction+RealVisibility+Geometry": {"information": "Y+V+G", "deployable": False},
        "GT-TrueLogP Oracle": {"information": "exact candidate utility", "deployable": False},
    }
    for name, actions in action_map.items():
        predictions = _terminal(val_logp, val_ids, val_mask, actions)
        methods[name] = _metrics(name, val_rows, val_labels, predictions, actions, method_information[name]["deployable"])
    historical: dict[str, Any] = {}
    if HISTORICAL_RESULT.is_file():
        payload = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
        historical_metric = payload.get("methods", {}).get("RGBGlobal-TrueLogP+VisibilityAux")
        if isinstance(historical_metric, dict):
            historical = {
                "n": int(historical_metric.get("n", len(val_rows))),
                "accuracy": float(historical_metric["accuracy"]),
                "macro_f1": float(historical_metric["macro_f1"]),
                "move_rate": float(historical_metric.get("move_rate", 0.0)),
                "stay_rate": float(historical_metric.get("stay_rate", 1.0)),
                "selector": "RGBGlobal Task+VisibilityAux (historical)",
                "deployable": True,
                "test_used": False,
                "source": str(HISTORICAL_RESULT),
            }
    if not historical:
        raise FileNotFoundError(f"historical deployable baseline missing: {HISTORICAL_RESULT}")
    methods["RGBGlobal Task+VisibilityAux (historical)"] = historical
    ladder_names = (
        "Stay",
        "Random",
        "GeometryOnly Utility",
        "RealVisibility+Geometry",
        "Global Viewpoint Prior",
        "GTAction+ViewpointPrior",
        "GTAction+Geometry",
        "GTAction+RealVisibility",
        "GTAction+RealVisibility+Geometry",
        "GT-TrueLogP Oracle",
    )
    ladder = {name: methods[name] for name in ladder_names}
    ranking: dict[str, Any] = {}
    for branch in LEARNED_BRANCHES:
        ranking[branch] = _ranking_metrics(branch, val_rows, predicted_scores[branch], val_target, val_ids, val_mask, gt_actions)
    for name, scores in (("Real Frame0 Visibility argmax", val_visibility), ("Global Viewpoint Prior", np.asarray([[prior["global_mean"][int(v)] if np.isfinite(prior["global_mean"][int(v)]) else float(prior["overall_mean"]) for v in row_ids] for row_ids in val_ids], dtype=np.float32)), ("GTAction+ViewpointPrior", np.asarray([[prior["class_mean"][int(label), int(v)] if np.isfinite(prior["class_mean"][int(label), int(v)]) else float(prior["class_fallback"][int(label)]) for v in row_ids] for label, row_ids in zip(val_labels, val_ids)], dtype=np.float32))):
        scores = np.asarray(scores, dtype=np.float32)
        scores[~val_mask] = -np.inf
        ranking[name] = _ranking_metrics(name, val_rows, scores, val_target, val_ids, val_mask, gt_actions)
    # Match the existing frame-0 audits: the bottom tertile is the strict
    # lower tail, excluding values tied exactly at the quantile boundary.
    high_occlusion = val_visibility[:, 0] < np.quantile(val_visibility[:, 0], 1.0 / 3.0)
    high_indices = np.flatnonzero(high_occlusion)
    high_rows = [val_rows[int(index)] for index in high_indices]
    high_methods: dict[str, Any] = {}
    for name in ("Stay", "Random", "GeometryOnly Utility", "RealVisibility+Geometry", "GTAction+Geometry", "GTAction+RealVisibility+Geometry", "GT-TrueLogP Oracle"):
        actions = [action_map[name][int(index)] for index in high_indices]
        high_methods[name] = _metrics(name, high_rows, val_labels[high_occlusion], _terminal(val_logp[high_occlusion], val_ids[high_occlusion], val_mask[high_occlusion], actions), actions, method_information[name]["deployable"])
    high_methods["DeployableBest"] = {
        "accuracy": float(json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))["occlusion_stratified_metrics"]["methods"]["RGBGlobal-TrueLogP+VisibilityAux"]["accuracy"]),
        "macro_f1": float(json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))["occlusion_stratified_metrics"]["methods"]["RGBGlobal-TrueLogP+VisibilityAux"]["macro_f1"]),
        "n": int(high_occlusion.sum()),
        "source": str(HISTORICAL_RESULT),
        "deployable": True,
        "test_used": False,
    }
    action_view_preference = _action_view_preference(train_target, train_logp, train_ids, train_mask, train_labels)
    interaction = _visibility_interaction(val_target, val_visibility, val_mask, val_labels)
    target_audit = {
        "train_contexts": len(train_rows),
        "moving_val_contexts": len(val_rows),
        "train_records": len({str(row["record_id"]) for row in train_rows}),
        "moving_val_records": len({str(row["record_id"]) for row in val_rows}),
        "train_utility": _target_stats(train_target, train_mask),
        "val_utility": _target_stats(val_target, val_mask),
        "train_action_count": int(train_mask.sum()),
        "val_action_count": int(val_mask.sum()),
        "action_set": "Stay/current + Stage-A legal candidate_pool",
        "recognizer": "frozen ST-GCN encoder + frozen shared head",
        "shared_head_sha256": _sha256(shared_head_path),
        "visibility_train_metadata": train_vis_meta,
        "visibility_val_metadata": val_vis_meta,
        "test_used": False,
    }
    if HISTORICAL_RESULT.is_file():
        previous = json.loads(HISTORICAL_RESULT.read_text(encoding="utf-8"))
        previous_method = previous.get("methods", {}).get("GeometryOnly-TrueLogP", {})
        previous_protocol = previous.get("protocol", {})
        target_audit["existing_geometry_only_reference"] = {
            "accuracy": float(previous_method.get("accuracy", 0.0)),
            "macro_f1": float(previous_method.get("macro_f1", 0.0)),
            "action_set": previous_protocol.get("action_set"),
            "target": "GT-TrueLogP",
            "train_contexts": int(previous.get("population", {}).get("train_contexts", 0)),
            "moving_val_contexts": int(previous.get("population", {}).get("moving_val_contexts", 0)),
            "seed": int(previous.get("runtime", {}).get("seed", -1)),
            "observations_per_record": int(previous.get("runtime", {}).get("observations_per_record", -1)),
            "protocol_match": bool(
                previous_protocol.get("action_set") == "current/Stay + Stage-A legal candidate_pool"
                and int(previous.get("population", {}).get("train_contexts", -1)) == len(train_rows)
                and int(previous.get("population", {}).get("moving_val_contexts", -1)) == len(val_rows)
                and int(previous.get("runtime", {}).get("seed", -1)) == SEED
                and int(previous.get("runtime", {}).get("observations_per_record", -1)) == OBS_PER_RECORD
                and bool(previous.get("flags", {}).get("policy_test_used", True)) is False
            ),
            "source": str(HISTORICAL_RESULT),
        }
    information_ladder = {
        name: {**methods[name], "information": method_information[name]["information"]}
        for name in ladder_names
    }
    g = float(methods["GeometryOnly Utility"]["accuracy"])
    vg = float(methods["RealVisibility+Geometry"]["accuracy"])
    yg = float(methods["GTAction+Geometry"]["accuracy"])
    yvg = float(methods["GTAction+RealVisibility+Geometry"]["accuracy"])
    oracle = float(methods["GT-TrueLogP Oracle"]["accuracy"])
    gaps = {
        "G": float(g),
        "VG": float(vg),
        "YG": float(yg),
        "YV": float(methods["GTAction+RealVisibility"]["accuracy"]),
        "YVG": float(yvg),
        "Oracle": float(oracle),
        "VisibilityGain_over_G": float(vg - g),
        "ActionGain_over_G": float(yg - g),
        "ConditionalVisibilityGain": float(yvg - yg),
        "ConditionalActionGain": float(yvg - vg),
        "ResidualGap": float(oracle - yvg),
    }
    leakage = {
        name: {
            "uses_gt_action_input": name.startswith("GTAction"),
            "uses_real_candidate_visibility": name in ("RealVisibility+Geometry", "GTAction+RealVisibility", "GTAction+RealVisibility+Geometry"),
            "uses_future_candidate_recognizer_target": False,
            "future_candidate_recognizer_output_used_as_input": False,
            "deployable": method_information[name]["deployable"],
            "test_used": False,
        }
        for name in LEARNED_BRANCHES
    }
    leakage.update(
        {
            "GT-TrueLogP Oracle": {"uses_gt_action_input": True, "uses_real_candidate_visibility": False, "uses_future_candidate_recognizer_target": True, "future_candidate_recognizer_output_used_as_input": True, "deployable": False, "test_used": False},
            "GTAction+ViewpointPrior": {"uses_gt_action_input": True, "uses_real_candidate_visibility": False, "uses_future_candidate_recognizer_target": False, "future_candidate_recognizer_output_used_as_input": False, "deployable": False, "test_used": False},
            "Global Viewpoint Prior": {"uses_gt_action_input": False, "uses_real_candidate_visibility": False, "uses_future_candidate_recognizer_target": True, "future_candidate_recognizer_output_used_as_input": False, "deployable": True, "test_used": False},
        }
    )
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_PRIVILEGED_INFORMATION_LADDER",
        "status": "COMPLETED",
        "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": target_audit["train_records"], "moving_val_records": target_audit["moving_val_records"]},
        "labels": list(LABELS),
        "methods": methods,
        "method_information": method_information,
        "information_ladder": information_ladder,
        "gaps": gaps,
        "ranking_metrics": ranking,
        "target_audit": target_audit,
        "viewpoint_prior_metrics": {"global_mean_utility": _json_matrix(np.asarray([prior["global_mean"]], dtype=np.float64))[0], "global_count": prior["global_count"].tolist(), "class_view_mean_utility": _json_matrix(prior["class_mean"]), "class_view_count": prior["class_count"].tolist(), "global_missing_viewpoints": missing_global_views, "gt_action_view_prior_missing_val_pairs": missing_class_pairs, "gt_action_view_prior_val_pair_total": class_pair_total},
        "action_view_preference": action_view_preference,
        "visibility_action_interaction": interaction,
        "occlusion_stratified_metrics": {"definition": "bottom tertile of current frame-0 Stay SceneVisibility", "count": int(high_occlusion.sum()), "methods": high_methods},
        "per_class_decomposition": {name: methods[name]["per_class"] for name in ("GeometryOnly Utility", "Real Frame0 Visibility argmax", "GTAction+ViewpointPrior", "GTAction+Geometry", "GTAction+RealVisibility+Geometry", "GT-TrueLogP Oracle")},
        "training": training,
        "leakage_audit": leakage,
        "flags": {"policy_test_used": False, "new_rgb_generated": False, "new_dino_generated": False, "new_skeleton_generated": False, "frozen_stgcn_modified": False, "gt_label_used_for_privileged_diagnostics_only": True, "real_candidate_visibility_used_for_privileged_diagnostics_only": True, "future_candidate_recognizer_target_only": True, "deployable": False},
        "runtime": {"device": str(device), "seed": SEED, "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "action_set": "Stay/current + Stage-A legal candidate_pool"},
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    (output_root / "config.json").write_text(json.dumps({"branches": list(LEARNED_BRANCHES), "architecture": "Linear(input_dim,256)-GELU-Linear(256,256)-GELU-Linear(256,1)", "epochs": EPOCHS, "observations_per_record": OBS_PER_RECORD, "lr": LR, "weight_decay": WEIGHT_DECAY, "listwise_weight": 0.5, "listwise_tau": LISTWISE_TAU, "checkpoint_selection": "minimum Moving-Val utility prediction loss", "test_used": False}, indent=2) + "\n", encoding="utf-8")
    (output_root / "information_ladder_metrics.json").write_text(json.dumps(information_ladder, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "ranking_metrics.json").write_text(json.dumps(ranking, indent=2) + "\n", encoding="utf-8")
    (output_root / "per_class_decomposition.json").write_text(json.dumps(result["per_class_decomposition"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "viewpoint_prior_metrics.json").write_text(json.dumps(result["viewpoint_prior_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "action_view_preference.json").write_text(json.dumps(action_view_preference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "visibility_action_interaction.json").write_text(json.dumps(interaction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "occlusion_stratified_metrics.json").write_text(json.dumps(result["occlusion_stratified_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "target_audit.json").write_text(json.dumps(target_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "leakage_audit.json").write_text(json.dumps(leakage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "training_summary.json").write_text(json.dumps(training, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
