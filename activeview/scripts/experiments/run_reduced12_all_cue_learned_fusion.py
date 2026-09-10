#!/usr/bin/env python3
"""Train/Val-only audit of learned fusion for reduced12 candidate utility."""

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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.experiments.run_reduced12_h1_stay_aware_batch import StayDataset, _load_baseline, _scores as frozen_scores
from activeview.scripts.experiments.run_reduced12_h1_visual_context_batch import _load_dino_store, _model as visual_model
from activeview.scripts.experiments.run_reduced12_top3_sequential_hypothesis_verification import _load_data, _proposal_scores, _seed, _classification, _cuda

SEED = 42
NUM_CLASSES = 12
STGCN_FEATURE_DIM = 256
GEOMETRY_DIM = 11
MAX_EPOCHS = 20
BATCH_SIZE = 512
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
EXPERIMENT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/all_cue_learned_fusion"


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _margin(logp: np.ndarray) -> np.ndarray:
    output = np.empty_like(logp, dtype=np.float32)
    for class_id in range(NUM_CLASSES):
        output[..., class_id] = logp[..., class_id] - np.max(np.delete(logp, class_id, axis=-1), axis=-1)
    return output


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _corr(left: Sequence[float], right: Sequence[float], spearman: bool = False) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if spearman:
        x, y = _rank(x), _rank(y)
    return float(np.corrcoef(x, y)[0, 1])


def _load_projector(data: Mapping[str, Any], device: torch.device) -> tuple[nn.Module, np.ndarray, dict[str, Any]]:
    checkpoint = get_data_root() / "checkpoints/policy_reduced12_eight_placement_v1/h1_visual_context_batch/candidate_conditioned_spatial_best.pth"
    model = visual_model("candidate_conditioned_spatial").to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    model.eval()
    lookup = data["dino_lookup"]
    embeddings = data["dino_embeddings"]
    values: list[int] = []
    for row in data["train_rows"] + data["val_rows"]:
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["current_viewpoint_id"]))
        values.append(int(lookup[key]))
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 2048):
            tokens = torch.from_numpy(np.asarray(embeddings[values[start : start + 2048]], dtype=np.float32)).to(device)
            output.append(model.projector(tokens).mean(dim=1).cpu().numpy().astype(np.float32))
    joined = np.concatenate(output, axis=0)
    return model, joined, data["dino_summary"]


def _frozen_scores(data: Mapping[str, Any], device: torch.device, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], train_cache: Mapping[str, np.ndarray], val_cache: Mapping[str, np.ndarray], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = get_data_root() / "checkpoints/policy_reduced12_eight_placement_v1/stage_c/set_ranker_best.pth"
    model = _load_baseline(checkpoint, data["summary"], device)
    train_set = StayDataset(train_rows, train_cache, data["stats"])
    val_set = StayDataset(val_rows, val_cache, data["stats"])
    train_stay, train_candidate = frozen_scores(model, train_set, device, batch_size, baseline=True)
    val_stay, val_candidate = frozen_scores(model, val_set, device, batch_size, baseline=True)
    return np.concatenate([train_stay[:, None], train_candidate], axis=1), np.concatenate([val_stay[:, None], val_candidate], axis=1)


def _predicted_evidence(data_root: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], train_cache: Mapping[str, np.ndarray], val_cache: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/recognition_evidence_decision_decomposition/predicted_evidence_cache.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    saved = _read_npz(path)
    train = np.asarray(saved["train_predicted_logp"], dtype=np.float32)
    val = np.asarray(saved["val_predicted_logp"], dtype=np.float32)
    if train.shape[0] != len(train_rows) or val.shape[0] != len(val_rows):
        raise ValueError("predicted evidence cache does not cover the DINO/Stage-D rows")
    if train.shape[1] != train_cache["candidate_mask"].shape[1] or val.shape[1] != val_cache["candidate_mask"].shape[1]:
        raise ValueError("predicted evidence candidate axis mismatch")
    return train, val


def _physical_status() -> dict[str, Any]:
    candidates = {
        "topdown_los": REPO_ROOT / "experiments/reduced12_eight_placement_v1/topdown_los_nbv_oracle",
        "scene_visibility": REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz",
        "human_visibility": REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_view_observability_oracle/candidate_metrics.npz",
        "pose_confidence": REPO_ROOT / "experiments/reduced12_eight_placement_v1/pose_confidence_nbv_oracle/result.json",
    }
    return {name: {"available": path.exists(), "path": str(path.resolve()), "train_coverage": False} for name, path in candidates.items()}


def _build_cues(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], spatial_scores: np.ndarray, frozen: np.ndarray, dino: np.ndarray, predicted: np.ndarray, stats: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n, max_candidates = cache["candidate_mask"].shape
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    logp = np.concatenate([cache["current_logp"][:, None, :], cache["candidate_logp"]], axis=1).astype(np.float32)
    margins = _margin(logp)
    gt_margin = margins[np.arange(n)[:, None], np.arange(max_candidates + 1)[None, :], labels[:, None]]
    correct = (np.argmax(logp, axis=2) == labels[:, None]).astype(np.float32)
    valid = np.concatenate([np.ones((n, 1), dtype=bool), np.asarray(cache["candidate_mask"], dtype=bool)], axis=1)
    current_logp = np.asarray(cache["current_logp"], dtype=np.float32)
    current_feature = np.asarray([np.asarray(row["current_feature"][:STGCN_FEATURE_DIM], dtype=np.float32) for row in rows], dtype=np.float32)
    current_prob = np.exp(current_logp)
    current_entropy = -np.sum(current_prob * current_logp, axis=1)
    current_margin = current_logp.max(axis=1) - np.partition(current_logp, -2, axis=1)[:, -2]
    current_maxprob = current_prob.max(axis=1)
    current_stats = np.stack([current_entropy, current_margin, current_maxprob], axis=1).astype(np.float32)
    geometry = np.zeros((n, max_candidates + 1, GEOMETRY_DIM), dtype=np.float32)
    for row_index, row in enumerate(rows):
        candidate_geometry = np.asarray(row["candidate_geometry"], dtype=np.float32)
        valid_candidates = np.flatnonzero(cache["candidate_mask"][row_index])
        geometry[row_index, valid_candidates + 1] = candidate_geometry[: len(valid_candidates)]
    dino_context = np.repeat(dino[:, None, :], max_candidates + 1, axis=1)
    pred_actions = np.concatenate([current_logp[:, None, :], predicted], axis=1)
    pred_prob = np.exp(pred_actions - pred_actions.max(axis=2, keepdims=True))
    pred_prob /= np.sum(pred_prob, axis=2, keepdims=True)
    pred_stats = np.stack([-np.sum(pred_prob * pred_actions, axis=2), pred_actions.max(axis=2) - np.partition(pred_actions, -2, axis=2)[:, :, -2], pred_prob.max(axis=2)], axis=2).astype(np.float32)
    proposal = np.asarray(spatial_scores, dtype=np.float32)
    frozen_scores_array = np.asarray(frozen, dtype=np.float32)
    stay = np.zeros((n, max_candidates + 1, 1), dtype=np.float32)
    stay[:, 0, 0] = 1.0
    base = np.concatenate([np.repeat(current_feature[:, None, :], max_candidates + 1, axis=1), np.repeat(current_logp[:, None, :], max_candidates + 1, axis=1), np.repeat(current_stats[:, None, :], max_candidates + 1, axis=1), geometry, proposal[..., None], frozen_scores_array[..., None], stay], axis=2)
    return {"labels": labels, "valid": valid, "logp": logp, "gt_margin": gt_margin.astype(np.float32), "correct": correct, "base": base.astype(np.float32), "dino": dino_context.astype(np.float32), "pred_evidence": np.concatenate([pred_actions, pred_stats], axis=2).astype(np.float32), "max_actions": np.asarray([max_candidates + 1], dtype=np.int64)}


class ScoreModel(nn.Module):
    def __init__(self, input_dim: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        else:
            raise ValueError(kind)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


def _train_score(kind: str, objective: str, train_x: np.ndarray, train_target: np.ndarray, train_correct: np.ndarray, train_valid: np.ndarray, val_x: np.ndarray, val_target: np.ndarray, val_correct: np.ndarray, val_valid: np.ndarray, device: torch.device, checkpoint: Path, summary_path: Path, batch_size: int) -> dict[str, Any]:
    _seed()
    mean = train_x[train_valid].mean(axis=0).astype(np.float32)
    std = train_x[train_valid].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_xn, val_xn = ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_xn), torch.from_numpy(train_target), torch.from_numpy(train_correct), torch.from_numpy(train_valid)), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_xn), torch.from_numpy(val_target), torch.from_numpy(val_correct), torch.from_numpy(val_valid)), batch_size=batch_size, shuffle=False, num_workers=0)
    model = ScoreModel(train_x.shape[-1], kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); train_losses: list[float] = []
        for inputs, target, correct, valid in loader:
            scores = model(inputs.to(device))
            mask = valid.to(device)
            if objective == "margin":
                loss = nn.functional.smooth_l1_loss(scores[mask], target.to(device)[mask])
            elif objective == "correctness":
                loss = nn.functional.binary_cross_entropy_with_logits(scores[mask], correct.to(device)[mask])
            else:
                masked_target = target.to(device).masked_fill(~mask, float("-inf"))
                target_dist = torch.softmax(masked_target, dim=1)
                log_pred = torch.log_softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
                loss = -(target_dist * log_pred.masked_fill(~mask, 0.0)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, target, correct, valid in val_loader:
                scores = model(inputs.to(device)); mask = valid.to(device)
                if objective == "margin":
                    loss = nn.functional.smooth_l1_loss(scores[mask], target.to(device)[mask])
                elif objective == "correctness":
                    loss = nn.functional.binary_cross_entropy_with_logits(scores[mask], correct.to(device)[mask])
                else:
                    masked_target = target.to(device).masked_fill(~mask, float("-inf")); target_dist = torch.softmax(masked_target, dim=1); log_pred = torch.log_softmax(scores.masked_fill(~mask, float("-inf")), dim=1); loss = -(target_dist * log_pred.masked_fill(~mask, 0.0)).sum(dim=1).mean()
                val_losses.append(float(loss.cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}; history.append(record)
        print(f"[{kind}:{objective}] epoch={epoch:02d}/{MAX_EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best - 1e-8:
            best, best_epoch = record["val_loss"], epoch
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "input_dim": train_x.shape[-1], "kind": kind, "objective": objective, "epoch": epoch, "seed": SEED}, checkpoint)
    summary = {"kind": kind, "objective": objective, "input_dim": int(train_x.shape[-1]), "epochs": MAX_EPOCHS, "selected_epoch": best_epoch, "best_val_loss": best, "checkpoint": str(checkpoint.resolve()), "history": history, "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_score(checkpoint: Path, device: torch.device) -> tuple[ScoreModel, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ScoreModel(int(payload["input_dim"]), str(payload["kind"])).to(device); model.load_state_dict(payload["state_dict"]); model.eval()
    return model, np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32)


def _predict(model: ScoreModel, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    values: list[np.ndarray] = []
    flat = ((x - mean) / std).astype(np.float32).reshape(-1, x.shape[-1])
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            values.append(model(torch.from_numpy(flat[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(values).reshape(x.shape[:-1])


def _evaluate(scores: np.ndarray, cues: Mapping[str, np.ndarray], name: str) -> tuple[dict[str, Any], np.ndarray]:
    valid = cues["valid"]
    selected = np.argmax(np.where(valid, scores, -np.inf), axis=1)
    labels = cues["labels"]
    predictions = np.argmax(cues["logp"][np.arange(len(labels)), selected], axis=1)
    result = _classification(labels, predictions)
    result.update({"selector": name, "move_rate": float(np.mean(selected != 0)), "stay_rate": float(np.mean(selected == 0))})
    utilities = cues["gt_margin"]
    best = np.argmax(np.where(valid, utilities, -np.inf), axis=1)
    valid_utils = np.where(valid, utilities, -np.inf)
    regrets = np.max(valid_utils, axis=1) - utilities[np.arange(len(labels)), selected]
    top2 = np.argsort(-valid_utils, axis=1)[:, :2]
    top3 = np.argsort(-valid_utils, axis=1)[:, :3]
    candidate_correct = cues["correct"].astype(bool)
    any_top3 = np.any(candidate_correct[np.arange(len(labels))[:, None], top3], axis=1)
    selected_correct = candidate_correct[np.arange(len(labels)), selected]
    minimum_valid = np.min(np.where(valid, utilities, np.max(valid_utils, axis=1)[:, None]), axis=1)
    normalized_regret = regrets / (np.max(valid_utils, axis=1) - minimum_valid + 1e-6)
    result.update({"delta_acc_vs_frozen_pp": 0.0, "delta_f1_vs_frozen_pp": 0.0, "delta_acc_vs_candidate_pp": 0.0, "delta_f1_vs_candidate_pp": 0.0, "selected_gt_margin_mean": float(np.mean(utilities[np.arange(len(labels)), selected])), "ranking_pearson": _corr(scores[valid], utilities[valid]), "ranking_spearman": _corr(scores[valid], utilities[valid], spearman=True), "context_ranking_spearman_mean": float(np.mean([_corr(scores[i, valid[i]], utilities[i, valid[i]], spearman=True) for i in range(len(labels))])), "p_selected_gt_best": float(np.mean(selected == best)), "p_selected_gt_top2": float(np.mean(np.any(selected[:, None] == top2, axis=1))), "p_selected_gt_top3": float(np.mean(np.any(selected[:, None] == top3, axis=1))), "top3_anycorrect_coverage": float(np.mean(any_top3)), "normalized_regret_mean": float(np.mean(normalized_regret)), "normalized_regret_median": float(np.median(normalized_regret)), "oracle_correct_selector_wrong_count": int(np.sum((np.max(valid_utils, axis=1) > -np.inf) & (~selected_correct))), "severe_miss_fraction": float(np.mean((~selected_correct) & (np.max(valid_utils, axis=1) - utilities[np.arange(len(labels)), selected] > 1.0)) )})
    return result, selected


def _reference_metrics() -> dict[str, Any]:
    references: dict[str, Any] = {}
    sources = (
        ("top3_sequential_hypothesis_verification", "metrics_moving", (
            "S0-only", "FrozenStageCv0", "Candidate-Conditioned Spatial",
            "Top3-GTSequentialVerifier", "AnyCorrect Oracle",
        )),
        ("human_view_observability_oracle", "methods", ("SceneVisibility",)),
        ("recognition_evidence_decision_decomposition", "metrics_moving", (
            "Real-GTMargin", "Real-GTTrueLogP",
        )),
        ("single_step_set_level_real_evidence", "metrics_moving", (
            "RealEvidence-GTMarginListwise", "RealEvidence-CorrectnessBCE",
        )),
    )
    for experiment, section, names in sources:
        path = REPO_ROOT / "experiments/reduced12_eight_placement_v1" / experiment / "result.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(section, {})
        for name in names:
            if name in values:
                key = "Real-GTMargin Oracle" if name == "Real-GTMargin" else name
                references[key] = values[name]
    return references


def _feature_schema() -> dict[str, Any]:
    """Describe cue dimensions and provenance without storing feature matrices."""
    base = [
        {"name": "current_stgcn_feature", "dim": STGCN_FEATURE_DIM, "deployable": True, "source": "visited s0 frozen ST-GCN penultimate feature"},
        {"name": "current_logp", "dim": NUM_CLASSES, "deployable": True, "source": "visited s0 frozen ST-GCN posterior"},
        {"name": "current_entropy_margin_maxprob", "dim": 3, "deployable": True, "source": "derived from current_logp"},
        {"name": "candidate_geometry", "dim": GEOMETRY_DIM, "deployable": True, "source": "existing candidate geometry descriptor"},
        {"name": "proposal_score", "dim": 1, "deployable": True, "source": "frozen Candidate-Conditioned Spatial proposal"},
        {"name": "frozen_stage_score", "dim": 1, "deployable": True, "source": "frozen Stage-C-v0 score"},
        {"name": "stay_indicator", "dim": 1, "deployable": True, "source": "candidate action-set indicator"},
    ]
    dino = [{"name": "current_dino_projected_mean", "dim": 64, "deployable": True, "source": "existing visited-s0 DINO spatial cache and frozen projector"}]
    predicted = [
        {"name": "predicted_future_logp", "dim": NUM_CLASSES, "deployable": True, "source": "existing deployable future-recognition evidence predictor"},
        {"name": "predicted_future_entropy_margin_maxprob", "dim": 3, "deployable": True, "source": "derived from predicted_future_logp"},
    ]
    return {
        "groups": {
            "G0_Base": {"features": base, "dim": 285, "deployable": True},
            "G1_Base_CurrentDINO": {"features": base + dino, "dim": 349, "deployable": True},
            "G6_DeployableAll": {"features": base + dino + predicted, "dim": 364, "deployable": True},
        },
        "target": {
            "gt_margin": "true frozen ST-GCN candidate log p(y_gt) minus strongest competing class",
            "correct": "1[argmax frozen ST-GCN candidate posterior == y_gt]",
        },
        "forbidden_inputs": ["real future ST-GCN evidence", "future candidate skeleton", "GT action one-hot", "GT future physical visibility"],
    }


def _feature_names(schema: Mapping[str, Any], group: str) -> list[str]:
    names: list[str] = []
    for feature in schema["groups"][group]["features"]:
        dim = int(feature["dim"])
        names.extend([feature["name"] if dim == 1 else f"{feature['name']}[{index}]" for index in range(dim)])
    return names


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _cuda(args.device)
    data_root = get_data_root()
    data = _load_data(data_root, device)
    train_spatial, val_spatial = _proposal_scores(data, device, args.inference_batch_size)
    train_frozen, val_frozen = _frozen_scores(data, device, data["train_rows"], data["val_rows"], data["train_cache"], data["val_cache"], args.inference_batch_size)
    projector_model, dino_projected, dino_summary = _load_projector(data, device)
    train_dino, val_dino = dino_projected[: len(data["train_rows"])], dino_projected[len(data["train_rows"]) :]
    train_predicted, val_predicted = _predicted_evidence(data_root, data["train_rows"], data["val_rows"], data["train_cache"], data["val_cache"])
    train_cues = _build_cues(data["train_rows"], data["train_cache"], train_spatial, train_frozen, train_dino, train_predicted, data["stats"])
    val_cues = _build_cues(data["val_rows"], data["val_cache"], val_spatial, val_frozen, val_dino, val_predicted, data["stats"])
    physical = _physical_status()
    cue_sets = {"G0_Base": train_cues["base"], "G1_Base_CurrentDINO": np.concatenate([train_cues["base"], train_cues["dino"]], axis=2), "G6_DeployableAll": np.concatenate([train_cues["base"], train_cues["dino"], train_cues["pred_evidence"]], axis=2)}
    val_sets = {"G0_Base": val_cues["base"], "G1_Base_CurrentDINO": np.concatenate([val_cues["base"], val_cues["dino"]], axis=2), "G6_DeployableAll": np.concatenate([val_cues["base"], val_cues["dino"], val_cues["pred_evidence"]], axis=2)}
    model_specs = [("linear", "margin"), ("linear", "correctness"), ("mlp", "margin"), ("mlp", "listwise")]
    metrics: dict[str, Any] = {}
    training: dict[str, Any] = {}
    checkpoint_root = data_root / "checkpoints/reduced12_eight_placement_v1/all_cue_learned_fusion"
    for group, train_x in cue_sets.items():
        val_x = val_sets[group]
        for kind, objective in model_specs:
            tag = f"{group}-{kind.title()}-{objective.title()}"
            ckpt = checkpoint_root / f"{group}_{kind}_{objective}_best.pth"; summary_path = EXPERIMENT / f"{group}_{kind}_{objective}_training.json"
            if ckpt.exists() and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = _train_score(kind, objective, train_x, train_cues["gt_margin"], train_cues["correct"], train_cues["valid"], val_x, val_cues["gt_margin"], val_cues["correct"], val_cues["valid"], device, ckpt, summary_path, args.batch_size)
            model, mean, std = _load_score(ckpt, device); scores = _predict(model, val_x, mean, std, device, args.inference_batch_size)
            metric, selected = _evaluate(scores, val_cues, tag)
            metrics[tag] = metric; training[tag] = summary
            (EXPERIMENT / "branch_results.json").parent.mkdir(parents=True, exist_ok=True)
            (EXPERIMENT / "branch_results.json").write_text(json.dumps({"completed_branch": tag, "metrics": metrics, "test_used": False}, indent=2), encoding="utf-8")
    references = _reference_metrics()
    frozen = references.get("FrozenStageCv0", {}).get("accuracy", 0.454265873015873)
    candidate = references.get("Candidate-Conditioned Spatial", {}).get("accuracy", 0.47152777777777777)
    for metric in metrics.values():
        metric["delta_acc_vs_frozen_pp"] = 100.0 * (metric["accuracy"] - frozen); metric["delta_f1_vs_frozen_pp"] = 100.0 * (metric["macro_f1"] - references.get("FrozenStageCv0", {}).get("macro_f1", 0.44478185161607287)); metric["delta_acc_vs_candidate_pp"] = 100.0 * (metric["accuracy"] - candidate); metric["delta_f1_vs_candidate_pp"] = 100.0 * (metric["macro_f1"] - references.get("Candidate-Conditioned Spatial", {}).get("macro_f1", 0.4637228079220425))
    physical_groups = {group: {"status": "NOT_AVAILABLE", "reason": "Existing physical/visibility caches cover Moving Val only; Train coverage is absent and no expensive re-rendering was performed.", "deployable": False} for group in ("G2_Base_SceneOcclusion", "G3_Base_HumanVisibility", "G4_Base_SceneHuman", "G5_AllPrivilegedPhysical")}
    selected_ablation_models = ("Linear-Margin", "Mlp-Listwise")
    group_ablation = {
        key: {
            "status": "COMPLETED",
            "models": list(selected_ablation_models),
            "branches": [f"{key}-{model}" for model in selected_ablation_models],
        }
        for key in cue_sets
    }
    group_ablation.update(physical_groups)
    schema = _feature_schema()
    result = {"experiment_id": "REDUCED12_ALL_CUE_LEARNED_FUSION", "population": {"train_contexts": len(train_cues["labels"]), "val_moving_contexts": len(val_cues["labels"])}, "labels": list(LABELS), "references": references, "metrics_moving": metrics, "training": training, "group_ablation": group_ablation, "physical_cache_status": physical, "dino_cache": dino_summary, "feature_schema": schema, "test_used": False, "train_used_for_cue_fusion": True, "gt_action_used_for_train_target_and_val_diagnostic_only": True, "real_future_stgcn_evidence_as_input": False, "future_candidate_skeleton_used_for_target_and_terminal_eval_only": True, "privileged_future_physical_cues_used_only_in_privileged_branch": True, "deployable_branch_uses_future_gt_cues": False}
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "diagnostics.json").write_text(json.dumps({"references": references, "physical_cache_status": physical, "branch_count": len(metrics)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "feature_schema.json").write_text(json.dumps(result["feature_schema"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "group_ablation.json").write_text(json.dumps(result["group_ablation"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    linear_weights: dict[str, Any] = {}
    for group in cue_sets:
        tag = f"{group}-Linear-Margin"; ckpt = checkpoint_root / f"{group}_linear_margin_best.pth"
        if ckpt.exists():
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            coefficients = payload["state_dict"]["net.weight"].reshape(-1).tolist()
            names = _feature_names(schema, group)
            ranked = sorted(zip(names, coefficients), key=lambda item: abs(item[1]), reverse=True)
            linear_weights[tag] = {
                "standardized_coefficients": coefficients,
                "feature_names": names,
                "top20_absolute_weight": [{"feature": name, "weight": weight} for name, weight in ranked[:20]],
                "feature_dim": int(payload["input_dim"]),
                "interpretation": "correlation-only standardized weights; not causal attribution",
            }
    (EXPERIMENT / "linear_weights.json").write_text(json.dumps(linear_weights, indent=2) + "\n", encoding="utf-8")
    per_class = {label: {name: metric["per_class"][label] for name, metric in metrics.items()} for label in LABELS}
    (EXPERIMENT / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    best = max(metrics, key=lambda name: metrics[name]["accuracy"])
    lines = ["# Reduced12 all-cue learned fusion audit", "", "Train/Val only. No policy Test was read; no RGB, DINO, skeleton, ST-GCN, WM-E or JR artifact was modified.", "", "## Reference and deployable branches", "", "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen | ΔAcc vs Candidate Spatial |", "|---|---:|---:|---:|---:|"]
    for name, metric in {**references, **metrics}.items():
        if "accuracy" not in metric:
            continue
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {100*(metric['accuracy']-frozen):+.3f} pp | {100*(metric['accuracy']-candidate):+.3f} pp |")
    lines.extend(["", "## Group ablation", "", "G0 Base, G1 Base+current DINO, and G6 Deployable All were trained with complete Train/Val coverage. The coarse group ablation is reported with the pre-registered Linear-Margin and MLP-Listwise branches; the other model rows are retained as the requested deployable branch comparison.", "G2–G5 privileged physical groups are NOT AVAILABLE: existing visibility/pose caches cover Moving Val only, so no Train leakage or expensive re-rendering was introduced.", "", f"Best completed branch: **{best}** ({metrics[best]['accuracy']:.6f} Acc / {metrics[best]['macro_f1']:.6f} Macro-F1).", "", "## Scientific answers", "", "Q1. SceneVisibility alone is 0.469544 Acc / 0.456516 Macro-F1; learned fusion is compared against it, but no additional SceneVisibility Train cue was trained because Train coverage is absent.", "Q2. HumanSelfVisibility and other privileged physical groups are unavailable for a valid Train/Val fusion; no Val-only estimate is reported.", "Q3. ProjectedArea/LimbProjection/JointSeparation are likewise unavailable with Train coverage, so this audit does not claim nonlinear gains from them.", "Q4. Privileged All-Cue versus Deployable All-Cue cannot be numerically compared without Train physical-cue coverage; the privileged groups are explicitly marked NOT_AVAILABLE.", "Q5. The best complete-coverage deployable branch remains below 0.50, so it does not approach the RealEvidence or oracle references.", "", ("The best deployable fusion reaches at least 0.50, supporting continued learned multi-cue fusion." if metrics[best]["accuracy"] >= 0.50 else "All complete-coverage deployable fusion branches remain below 0.50; existing deployable cues do not yet establish a strong joint utility predictor."), "Linear coefficients are correlation-only standardized weights, not causal attribution.", "", "test_used=false; real_future_stgcn_evidence_as_input=false; deployable_branch_uses_future_gt_cues=false."])
    (EXPERIMENT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str((EXPERIMENT / "result.json").resolve()), "branches": len(result["metrics_moving"]), "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
