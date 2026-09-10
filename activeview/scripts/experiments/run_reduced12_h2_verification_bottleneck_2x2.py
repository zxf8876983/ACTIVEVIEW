#!/usr/bin/env python3
"""Train/Val-only decomposition of the reduced12 H2 verification bottleneck.

The frozen Candidate-Conditioned Spatial model proposes rank1/rank2/rank3.
After real H1 observation, this script separates action-identity inference
from candidate-utility prediction without reading Test or regenerating data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.scripts.experiments.run_reduced12_top3_sequential_hypothesis_verification import (
    EPOCHS,
    EXPERIMENT as SEQUENTIAL_EXPERIMENT,
    HIDDEN_DIM,
    LABELS,
    NUM_CLASSES,
    _build_sequential_arrays,
    _classification,
    _cuda,
    _load_data,
    _proposal_scores,
    _seed,
    _terminal_metrics,
    _utility,
)
from activeview.core.paths import get_data_root

OBSERVED_INPUT_DIM = 2 * (NUM_CLASSES + 256)
UTILITY_INPUT_DIM = 550 + NUM_CLASSES
EXPERIMENT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/h2_verification_bottleneck_2x2"


class ObservedActionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBSERVED_INPUT_DIM, 256), nn.GELU(), nn.Linear(256, NUM_CLASSES))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class UtilityPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(UTILITY_INPUT_DIM, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


def _train_classifier(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, device: torch.device, checkpoint: Path, summary_path: Path, batch_size: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _seed()
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_norm = ((train_x - mean) / std).astype(np.float32)
    val_norm = ((val_x - mean) / std).astype(np.float32)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_norm), torch.from_numpy(train_y)), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_norm), torch.from_numpy(val_y)), batch_size=batch_size, shuffle=False, num_workers=0)
    model = ObservedActionClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for inputs, labels in train_loader:
            loss = nn.functional.cross_entropy(model(inputs.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, labels in val_loader:
                val_losses.append(float(nn.functional.cross_entropy(model(inputs.to(device)), labels.to(device)).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        print(f"[observed-action] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss = record["val_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "input_mean": mean, "input_std": std, "epoch": epoch, "seed": 42}, checkpoint)
    summary = {"model": "ObservedActionClassifier", "input_dim": OBSERVED_INPUT_DIM, "hidden_dim": 256, "epochs": EPOCHS, "selected_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "history": history, "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, mean, std


def _load_classifier(checkpoint: Path, device: torch.device) -> tuple[ObservedActionClassifier, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ObservedActionClassifier().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["input_mean"], dtype=np.float32), np.asarray(payload["input_std"], dtype=np.float32)


def _observed_inputs(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([arrays["s0_logp"], arrays["s0_feature"], arrays["h1_logp"], arrays["h1_feature"]], axis=1).astype(np.float32)


def _margins(logp: np.ndarray) -> np.ndarray:
    output = np.empty_like(logp, dtype=np.float32)
    for class_id in range(NUM_CLASSES):
        other = np.delete(logp, class_id, axis=-1)
        output[..., class_id] = logp[..., class_id] - np.max(other, axis=-1)
    return output


def _real_action_selection(arrays: Mapping[str, np.ndarray], action_scores: np.ndarray, name: str) -> dict[str, Any]:
    selected = np.argmax(action_scores, axis=1).astype(np.int64)
    result = _terminal_metrics(arrays, selected, name=name)
    return result


def _utility_inputs(arrays: Mapping[str, np.ndarray], action_class: np.ndarray) -> np.ndarray:
    n = len(arrays["labels"])
    base = np.concatenate([arrays["s0_logp"], arrays["s0_feature"], arrays["h1_logp"], arrays["h1_feature"]], axis=1)
    output = np.zeros((n, 3, UTILITY_INPUT_DIM), dtype=np.float32)
    for row_index in range(n):
        for slot in range(3):
            action = np.zeros(NUM_CLASSES, dtype=np.float32)
            action[int(action_class[row_index])] = 1.0
            output[row_index, slot] = np.concatenate([base[row_index], arrays["geometry"][row_index, int(arrays["ranks"][row_index, slot])] if int(arrays["ranks"][row_index, slot]) >= 0 else np.zeros(11, dtype=np.float32), np.asarray([float(arrays["proposal_scores"][row_index, int(arrays["ranks"][row_index, slot])]), float(slot + 1), float(slot == 0)], dtype=np.float32), action])
    return output


def _train_utility(train_inputs: np.ndarray, train_targets: np.ndarray, val_inputs: np.ndarray, val_targets: np.ndarray, device: torch.device, checkpoint: Path, summary_path: Path, batch_size: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _seed()
    mean = train_inputs.reshape(-1, UTILITY_INPUT_DIM).mean(axis=0).astype(np.float32)
    std = train_inputs.reshape(-1, UTILITY_INPUT_DIM).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_norm = ((train_inputs - mean) / std).astype(np.float32)
    val_norm = ((val_inputs - mean) / std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_norm), torch.from_numpy(train_targets)), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_norm), torch.from_numpy(val_targets)), batch_size=batch_size, shuffle=False, num_workers=0)
    model = UtilityPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for inputs, targets in loader:
            loss = nn.functional.smooth_l1_loss(model(inputs.to(device)), targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses: list[float] = []
        with torch.inference_mode():
            for inputs, targets in val_loader:
                val_losses.append(float(nn.functional.smooth_l1_loss(model(inputs.to(device)), targets.to(device)).cpu()))
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        print(f"[utility-predictor] epoch={epoch:02d}/{EPOCHS} train_loss={record['train_loss']:.6f} val_loss={record['val_loss']:.6f}", flush=True)
        if record["val_loss"] < best_loss - 1e-8:
            best_loss = record["val_loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "input_mean": mean, "input_std": std, "epoch": epoch, "seed": 42}, checkpoint)
    summary = {"model": "GTActionConditionedUtilityPredictor", "input_dim": UTILITY_INPUT_DIM, "hidden_dims": [256, 128], "epochs": EPOCHS, "selected_epoch": best_epoch, "best_val_loss": best_loss, "checkpoint": str(checkpoint.resolve()), "history": history, "test_used": False}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, mean, std


def _load_utility(checkpoint: Path, device: torch.device) -> tuple[UtilityPredictor, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = UtilityPredictor().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["input_mean"], dtype=np.float32), np.asarray(payload["input_std"], dtype=np.float32)


def _classifier_metrics(arrays: Mapping[str, np.ndarray], probabilities: np.ndarray) -> dict[str, Any]:
    labels = arrays["labels"]
    s0 = np.argmax(arrays["s0_logp"], axis=1)
    h1 = np.argmax(arrays["h1_logp"], axis=1)
    mean = np.argmax(np.mean([arrays["s0_logp"], arrays["h1_logp"]], axis=0), axis=1)
    return {
        "S0 ST-GCN": _classification(labels, s0),
        "H1 ST-GCN": _classification(labels, h1),
        "Mean(s0,h1)": _classification(labels, mean),
        "PredAction": _classification(labels, np.argmax(probabilities, axis=1)),
    }


def _utility_diagnostics(arrays: Mapping[str, np.ndarray], predicted: np.ndarray, action_class: np.ndarray, focus: np.ndarray) -> dict[str, Any]:
    real = arrays["utilities"]
    records: dict[str, Any] = {}
    for slot, name in enumerate(("STOP", "rank2", "rank3")):
        p = predicted[:, slot]
        r = real[:, slot]
        records[name] = {"pearson": float(np.corrcoef(p, r)[0, 1]) if np.std(p) > 1e-12 and np.std(r) > 1e-12 else 0.0, "spearman": float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(r)))[0, 1]) if np.std(p) > 1e-12 and np.std(r) > 1e-12 else 0.0, "mae": float(np.mean(np.abs(p - r)))}
    rank_corr = []
    top1 = []
    top2 = []
    for p, r in zip(predicted, real):
        rank_corr.append(float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(r)))[0, 1]) if np.std(p) > 1e-12 and np.std(r) > 1e-12 else 0.0)
        top1.append(int(np.argmax(p) == np.argmax(r)))
        top2.append(int(np.argmax(r) in np.argsort(-p)[:2]))
    records["context_ranking"] = {"spearman_mean": float(np.mean(rank_corr)), "predicted_argmax_is_real_best": float(np.mean(top1)), "real_best_in_predicted_top2": float(np.mean(top2))}
    if np.any(focus):
        records["h1_wrong_top3_contains_correct"] = {"n": int(np.sum(focus)), "predicted_action_class_accuracy": float(np.mean(action_class[focus] == arrays["labels"][focus])), "predicted_utility_top1_is_real_best": float(np.mean(np.argmax(predicted[focus], axis=1) == np.argmax(real[focus], axis=1)))}
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _cuda(args.device)
    data_root = get_data_root()
    data = _load_data(data_root, device)
    proposal_train, proposal_val = _proposal_scores(data, device, args.inference_batch_size)
    train_arrays = _build_sequential_arrays(data["train_rows"], data["train_cache"], data["train_target"], proposal_train, data["stats"])
    val_arrays = _build_sequential_arrays(data["val_rows"], data["val_cache"], data["val_target"], proposal_val, data["stats"])
    train_obs = np.concatenate([train_arrays["s0_logp"], train_arrays["s0_feature"], train_arrays["h1_logp"], train_arrays["h1_feature"]], axis=1).astype(np.float32)
    val_obs = np.concatenate([val_arrays["s0_logp"], val_arrays["s0_feature"], val_arrays["h1_logp"], val_arrays["h1_feature"]], axis=1).astype(np.float32)
    classifier_ckpt = data_root / "checkpoints/reduced12_eight_placement_v1/h2_verification_bottleneck_2x2/observed_action_classifier_best.pth"
    classifier_summary_path = EXPERIMENT / "action_classifier_training.json"
    if classifier_ckpt.exists() and classifier_summary_path.exists():
        classifier_summary = json.loads(classifier_summary_path.read_text(encoding="utf-8"))
    else:
        classifier_summary, _, _ = _train_classifier(train_obs, train_arrays["labels"], val_obs, val_arrays["labels"], device, classifier_ckpt, classifier_summary_path, args.batch_size)
    classifier, obs_mean, obs_std = _load_classifier(classifier_ckpt, device)
    with torch.inference_mode():
        action_logits = classifier(torch.from_numpy(((val_obs - obs_mean) / obs_std).astype(np.float32)).to(device)).cpu().numpy()
    action_probabilities = np.exp(action_logits - action_logits.max(axis=1, keepdims=True))
    action_probabilities /= action_probabilities.sum(axis=1, keepdims=True)
    train_pred_action = np.asarray(train_arrays["labels"], dtype=np.int64)
    train_utility_inputs = _utility_inputs(train_arrays, train_pred_action)
    val_gt_inputs = _utility_inputs(val_arrays, val_arrays["labels"])
    val_pred_inputs = _utility_inputs(val_arrays, np.argmax(action_probabilities, axis=1))
    soft_inputs = np.stack([_utility_inputs(val_arrays, np.full(len(val_arrays["labels"]), class_id, dtype=np.int64)) for class_id in range(NUM_CLASSES)], axis=-1)
    train_utility_targets = train_arrays["utilities"][..., 0:3]
    utility_ckpt = data_root / "checkpoints/reduced12_eight_placement_v1/h2_verification_bottleneck_2x2/utility_predictor_best.pth"
    utility_summary_path = EXPERIMENT / "utility_predictor_training.json"
    if utility_ckpt.exists() and utility_summary_path.exists():
        utility_summary = json.loads(utility_summary_path.read_text(encoding="utf-8"))
    else:
        utility_summary, _, _ = _train_utility(train_utility_inputs, train_utility_targets, val_gt_inputs, val_arrays["utilities"], device, utility_ckpt, utility_summary_path, args.batch_size)
    utility_model, util_mean, util_std = _load_utility(utility_ckpt, device)
    with torch.inference_mode():
        gt_pred = utility_model(torch.from_numpy(((val_gt_inputs - util_mean) / util_std).astype(np.float32)).to(device)).cpu().numpy().reshape(-1, 3)
        pred_pred = utility_model(torch.from_numpy(((val_pred_inputs - util_mean) / util_std).astype(np.float32)).to(device)).cpu().numpy().reshape(-1, 3)
        soft_pred = np.zeros_like(gt_pred)
        for class_id in range(NUM_CLASSES):
            current = utility_model(torch.from_numpy(((soft_inputs[..., class_id] - util_mean) / util_std).astype(np.float32)).to(device)).cpu().numpy().reshape(-1, 3)
            soft_pred += action_probabilities[:, class_id, None] * current
    real_margins = _margins(np.concatenate([val_arrays["h1_logp"][:, None], val_arrays["action_logp"][:, 1:]], axis=1))
    real_gt_actions = np.argmax(val_arrays["utilities"], axis=1)
    pred_action_scores = np.stack([np.sum(action_probabilities[:, None, :] * real_margins, axis=2)[:, slot] for slot in range(3)], axis=1)
    hard_action_scores = np.stack([real_margins[np.arange(len(real_margins)), slot, np.argmax(action_probabilities, axis=1)] for slot in range(3)], axis=1)
    metrics = {
        "PredAction-RealEvidence": _real_action_selection(val_arrays, hard_action_scores, "PredAction-RealEvidence"),
        "SoftPredAction-RealEvidence": _real_action_selection(val_arrays, pred_action_scores, "SoftPredAction-RealEvidence"),
        "GTAction-RealEvidence": _real_action_selection(val_arrays, val_arrays["utilities"], "GTAction-RealEvidence"),
        "GTAction-PredUtility": _real_action_selection(val_arrays, gt_pred, "GTAction-PredUtility"),
        "PredAction-PredUtility": _real_action_selection(val_arrays, pred_pred, "PredAction-PredUtility"),
        "SoftPredAction-PredUtility": _real_action_selection(val_arrays, soft_pred, "SoftPredAction-PredUtility"),
    }
    classifier_metrics = _classifier_metrics(val_arrays, action_probabilities)
    focus = (np.argmax(val_arrays["h1_logp"], axis=1) != val_arrays["labels"]) & np.any(np.argmax(val_arrays["action_logp"], axis=2) == val_arrays["labels"][:, None], axis=1)
    predicted_action = np.argmax(action_probabilities, axis=1)
    utility_diagnostics = _utility_diagnostics(val_arrays, gt_pred, predicted_action, focus)
    pred_real_slot = np.argmax(hard_action_scores, axis=1)
    pred_real_prediction = np.asarray([np.argmax(val_arrays["action_logp"][i, pred_real_slot[i]]) for i in range(len(pred_real_slot))], dtype=np.int64)
    pred_action_correct = predicted_action == val_arrays["labels"]
    utility_diagnostics["focus_action_inference"] = {
        "n": int(np.sum(focus)),
        "pred_action_correct_rate": float(np.mean(pred_action_correct[focus])) if np.any(focus) else 0.0,
        "pred_action_real_evidence_success_if_pred_action_correct": float(np.mean(pred_real_prediction[focus & pred_action_correct] == val_arrays["labels"][focus & pred_action_correct])) if np.any(focus & pred_action_correct) else 0.0,
        "pred_action_real_evidence_success_if_pred_action_wrong": float(np.mean(pred_real_prediction[focus & ~pred_action_correct] == val_arrays["labels"][focus & ~pred_action_correct])) if np.any(focus & ~pred_action_correct) else 0.0,
    }
    gt_acc = metrics["GTAction-RealEvidence"]["accuracy"]
    pred_real_acc = metrics["PredAction-RealEvidence"]["accuracy"]
    gt_pred_acc = metrics["GTAction-PredUtility"]["accuracy"]
    pred_pred_acc = metrics["PredAction-PredUtility"]["accuracy"]
    gaps = {"total_verification_gap_pp": 100.0 * (gt_acc - pred_pred_acc), "action_inference_gap_pp": 100.0 * (gt_acc - pred_real_acc), "utility_prediction_gap_pp": 100.0 * (gt_acc - gt_pred_acc), "deployable_interaction_gap_pp": 100.0 * (min(pred_real_acc, gt_pred_acc) - pred_pred_acc)}
    previous_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/top3_sequential_hypothesis_verification/result.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    candidate_reference = previous.get("metrics_moving", {}).get("Candidate-Conditioned Spatial", _terminal_metrics(val_arrays, np.zeros(len(val_arrays["labels"]), dtype=np.int64), name="Candidate-Conditioned Spatial"))
    for metric in metrics.values():
        metric["delta_acc_vs_candidate_pp"] = 100.0 * (metric["accuracy"] - candidate_reference["accuracy"])
        metric["delta_f1_vs_candidate_pp"] = 100.0 * (metric["macro_f1"] - candidate_reference["macro_f1"])
    reference_metrics = {key: previous["metrics_moving"][key] for key in ("FrozenStageCv0", "Learned Sequential Verifier H2-only", "Top3-GTSequentialVerifier", "AnyCorrect Oracle") if key in previous.get("metrics_moving", {})}
    reference_metrics["Candidate-Conditioned Spatial"] = candidate_reference
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    result = {"experiment_id": "REDUCED12_H2_VERIFICATION_BOTTLENECK_2X2", "labels": list(LABELS), "population": {"train_contexts": len(train_arrays["labels"]), "val_moving_contexts": len(val_arrays["labels"])}, "action_classifier": classifier_metrics, "reference_metrics": reference_metrics, "metrics_moving": metrics, "quadrant_metrics": {key: metrics[key] for key in ("GTAction-RealEvidence", "PredAction-RealEvidence", "GTAction-PredUtility", "PredAction-PredUtility", "SoftPredAction-RealEvidence", "SoftPredAction-PredUtility")}, "utility_diagnostics": utility_diagnostics, "gaps_pp": gaps, "training": {"action_classifier": classifier_summary, "utility_predictor": utility_summary}, "test_used": False, "test_read": False, "future_rgb_used": False, "future_dino_used": False, "future_rank2_rank3_evidence_used_as_deployable_input": False, "gt_action_used_only_for_train_supervision_and_privileged_quadrants": True}
    (EXPERIMENT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "diagnostics.json").write_text(json.dumps({"gaps_pp": gaps, "utility_diagnostics": utility_diagnostics, "focus_contexts": int(np.sum(focus))}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "action_inference_metrics.json").write_text(json.dumps(classifier_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "utility_prediction_metrics.json").write_text(json.dumps(utility_diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXPERIMENT / "quadrant_metrics.json").write_text(json.dumps(result["quadrant_metrics"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_class = {label: {method: metric["per_class"][label] for method, metric in metrics.items()} for label in LABELS}
    (EXPERIMENT / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reference = metrics["GTAction-RealEvidence"]
    lines = ["# Reduced12 H2 verification bottleneck 2×2 decomposition", "", "Train/Val only. No policy Test was read; no RGB, DINO, skeleton, WM-E, JR, or ST-GCN artifact was modified.", "", "## Action inference", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for name, metric in classifier_metrics.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend(["", "## H2 2×2", "", "| Method | Accuracy | Macro-F1 | H2 move | STOP | ΔAcc vs Candidate Spatial |", "|---|---:|---:|---:|---:|---:|"])
    for name, metric in {**reference_metrics, **metrics}.items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {metric.get('h2_move_rate', metric.get('move_rate', 0.0)):.6f} | {metric.get('stop_rate', metric.get('stay_rate', 0.0)):.6f} | {100.0 * (metric['accuracy'] - candidate_reference['accuracy']):+.3f} pp |")
    conditional = utility_diagnostics["focus_action_inference"]
    lines.extend(["", "## Gaps", "", f"Total verification gap: {gaps['total_verification_gap_pp']:+.3f} pp", f"Action inference gap: {gaps['action_inference_gap_pp']:+.3f} pp", f"Utility prediction gap: {gaps['utility_prediction_gap_pp']:+.3f} pp", f"Deployable interaction gap: {gaps['deployable_interaction_gap_pp']:+.3f} pp", "", f"On H1-wrong / Top3-correct contexts (n={conditional['n']}), PredAction accuracy={conditional['pred_action_correct_rate']:.6f}; PredAction-RealEvidence success when PredAction is correct={conditional['pred_action_real_evidence_success_if_pred_action_correct']:.6f}, when wrong={conditional['pred_action_real_evidence_success_if_pred_action_wrong']:.6f}.", "", "These gaps are diagnostic decompositions and are not assumed to be strictly additive because action and utility errors interact.", "", "## Scientific decision", "", ("The action-inference gap is the larger component; prioritize temporal/sequential action evidence." if gaps["action_inference_gap_pp"] > gaps["utility_prediction_gap_pp"] else "The utility-prediction gap is the larger component; prioritize candidate-specific future utility representation."), ("GT-action predicted utility remains at or above 60%, indicating future utility is predictable once action identity is known." if gt_pred_acc >= 0.60 else "GT-action predicted utility remains below 60%; observed state and candidate geometry do not yet provide a strong utility predictor."), "", "test_used=false; future_rank2_rank3_evidence_used_as_deployable_input=false."])
    (EXPERIMENT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=2048)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str((EXPERIMENT / "result.json").resolve()), "val_moving": result["population"]["val_moving_contexts"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
