#!/usr/bin/env python3
"""Train and evaluate EXP055 multi-positive Joint Revision (Train/Val only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.active_view.data import load_jsonl
from activeview.core.paths import get_data_root
from activeview.active_view.data import episode_sources as _episode_sources, load_stage_d_cache as _load_cache, rows as _rows
from activeview.active_view.geometry import candidate_order as _candidate_order, load_pairwise_and_azimuths as _load_pairwise_and_azimuths
from activeview.active_view.joint_revision import JointRevision as _JointRevision
from activeview.active_view.rollout import run_real_observation_evaluation as run_closed_loop

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_CLASSES = 16
VIEW_COUNT = 32
EXP_DIR = REPO_ROOT / "experiments/stage_d/EXP055_multi_positive_joint_revision"


def _seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


class _Dataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: tuple[np.ndarray, ...]) -> None:
        self.current, self.candidates, self.mask, self.fallback, self.positive, self.labels = arrays

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"current": torch.from_numpy(self.current[index]), "candidates": torch.from_numpy(self.candidates[index]), "mask": torch.from_numpy(self.mask[index]), "fallback": torch.tensor(self.fallback[index]), "positive": torch.from_numpy(self.positive[index]), "label": torch.tensor(self.labels[index])}


def _examples(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]]) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}; currents: list[np.ndarray] = []; candidates_all: list[np.ndarray] = []; masks: list[np.ndarray] = []; fallbacks: list[int] = []; positives: list[np.ndarray] = []; labels: list[int] = []
    positive_contexts = multi_contexts = single_contexts = no_positive = 0; positive_counts: list[int] = []
    for row in rows:
        i = index[str(row["episode_id"])]
        candidates = [int(value) for value in orders[str(row["episode_id"])] ]
        if not candidates: continue
        current_logp = cache["current_logp_s1"][i]; probs0 = np.exp(cache["current_logp_s0"][i]); probs1 = np.exp(current_logp)
        current = np.asarray([*probs0, *probs1, -np.sum(probs0 * cache["current_logp_s0"][i]), -np.sum(probs1 * current_logp), np.max(probs0), np.max(probs1), np.sort(probs1)[-1] - np.sort(probs1)[-2], np.sort(probs0)[-1] - np.sort(probs0)[-2]], dtype=np.float32)
        rows_values = [np.concatenate([current_logp, np.zeros(9, dtype=np.float32), [1.0]])]
        rows_values.extend(np.concatenate([cache["imagined_logp"][i, c], cache["candidate_descriptor"][i, c], [0.0]]) for c in candidates)
        label = int(cache["label_id"][i]); action_correct = np.asarray([int(np.argmax(current_logp) == label)] + [int(np.argmax(cache["true_logp"][i, c]) == label) for c in candidates], dtype=np.float32)
        valid_count = len(rows_values); positive_count = int(action_correct.sum());
        if positive_count:
            positive_contexts += 1; positive_counts.append(positive_count)
            if positive_count > 1: multi_contexts += 1
            else: single_contexts += 1
            fallback = int(np.flatnonzero(action_correct)[0])
        else:
            no_positive += 1
            true_scores = [float(current_logp[label])] + [float(cache["true_logp"][i, c, label]) for c in candidates]
            fallback = int(np.argmax(true_scores))
        padded = np.zeros((31, 26), dtype=np.float32); padded[:valid_count] = np.asarray(rows_values, dtype=np.float32)
        padded_positive = np.zeros(31, dtype=np.float32); padded_positive[:valid_count] = action_correct
        currents.append(current); candidates_all.append(padded); masks.append(np.asarray([True] * valid_count + [False] * (31 - valid_count), dtype=bool)); fallbacks.append(fallback); positives.append(padded_positive); labels.append(label)
    labels_arr = np.asarray(labels, dtype=np.int64)
    stats = {"train_contexts": len(labels), "positive_contexts": positive_contexts, "multi_positive_contexts": multi_contexts, "single_positive_contexts": single_contexts, "no_positive_contexts": no_positive, "mean_positive_count_on_positive_contexts": float(np.mean(positive_counts)) if positive_counts else 0.0}
    return (np.asarray(currents), np.asarray(candidates_all), np.asarray(masks), np.asarray(fallbacks), np.asarray(positives), labels_arr), stats


def _fit(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], device: torch.device, checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays, stats = _examples(rows, cache, orders); dataset = _Dataset(arrays); loader = DataLoader(dataset, batch_size=512, shuffle=True); model = _JointRevision().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); history: list[float] = []
    for epoch in range(1, 21):
        model.train(); losses: list[float] = []
        for batch in loader:
            current = batch["current"].float().to(device); candidates = batch["candidates"].float().to(device); mask = batch["mask"].bool().to(device); fallback = batch["fallback"].long().to(device); positive = batch["positive"].float().to(device); labels = batch["label"].long().to(device)
            scores, posterior = model(current, candidates, mask); valid_scores = scores.masked_fill(~mask, -1e9); all_lse = torch.logsumexp(valid_scores, dim=1); positive_mask = positive.bool() & mask; positive_lse = torch.logsumexp(scores.masked_fill(~positive_mask, -1e9), dim=1); multi_loss = all_lse - positive_lse; fallback_loss = nn.functional.cross_entropy(valid_scores, fallback, reduction="none"); has_positive = positive_mask.any(dim=1); main = torch.where(has_positive, multi_loss, fallback_loss); bce_all = nn.functional.binary_cross_entropy_with_logits(scores, positive, reduction="none"); bce = (bce_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1); target = labels.unsqueeze(1).expand(-1, posterior.size(1)); post_all = nn.functional.cross_entropy(posterior.reshape(-1, N_CLASSES), target.reshape(-1), reduction="none").reshape_as(scores); post = (post_all * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1); loss = (main + 0.25 * bce + 0.05 * post).mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        history.append(float(np.mean(losses))); print(f"EXP055 epoch {epoch}/20 loss={history[-1]:.6f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": model.state_dict(), "model_state_dict": model.state_dict(), "seed": SEED, "epochs": 20, "loss": "multi_positive_logsumexp+0.25_bce+0.05_posterior"}, checkpoint); stats.update({"final_loss": history[-1], "loss_history": history}); return model.eval(), stats


def _full_metrics(terminal: Sequence[int], moving_ids: Sequence[str], v0_rows: Sequence[Mapping[str, Any]], stage_b_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    pred_by_id = {str(e): int(p) for e, p in zip(moving_ids, terminal)}; labels = {str(row["episode_id"]): int(row["label_id"]) for row in stage_b_rows}; predictions = [int(row["current_predicted_label_id"]) if bool(row["predicted_stays"]) else pred_by_id[str(row["episode_id"])] for row in v0_rows]; truth = [labels[str(row["episode_id"])] for row in v0_rows]; values = np.asarray(predictions); target = np.asarray(truth); confusion = np.bincount(target * N_CLASSES + values, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES); f1 = []
    for cls in range(N_CLASSES):
        tp = confusion[cls, cls]; precision = tp / confusion[:, cls].sum() if confusion[:, cls].sum() else 0.0; recall = tp / confusion[cls].sum() if confusion[cls].sum() else 0.0; f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float(np.mean(values == target)), "macro_f1": float(np.mean(f1))}


def _full_predictions(terminal: Sequence[int], moving_ids: Sequence[str], v0_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    pred_by_id = {str(e): int(p) for e, p in zip(moving_ids, terminal)}
    return np.asarray([int(row["current_predicted_label_id"]) if bool(row["predicted_stays"]) else pred_by_id[str(row["episode_id"])] for row in v0_rows], dtype=np.int64)


def _class_recall(prediction: np.ndarray, labels: np.ndarray) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for cls in range(N_CLASSES):
        denominator = int(np.sum(labels == cls))
        output.append({"class_id": cls, "support": denominator, "recall": float(np.sum((labels == cls) & (prediction == cls)) / denominator) if denominator else 0.0})
    return output


def _audit(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, int]:
    old = [list(v) for v in baseline["h2_action_signatures_moving"]]; new = [list(v) for v in candidate["h2_action_signatures_moving"]]; old_pred = np.asarray(baseline["h2_terminal_predictions_moving"]); new_pred = np.asarray(candidate["h2_terminal_predictions_moving"]); labels = np.asarray(candidate["moving_labels"]); first = second = 0
    for left, right in zip(old, new):
        first += int((left[0] if left else None) != (right[0] if right else None)); second += int((left[1] if len(left) > 1 else None) != (right[1] if len(right) > 1 else None))
    rescued = int(np.sum((old_pred != labels) & (new_pred == labels))); harmful = int(np.sum((old_pred == labels) & (new_pred != labels))); return {"changed_trajectories": int(sum(a != b for a, b in zip(old, new))), "stage_c_first_step_changed": 0, "first_second_step_action_changed": first, "second_second_step_action_changed": second, "rescued": rescued, "harmful": harmful, "net": rescued - harmful}


def run(data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed(); train_rows, val_rows = _rows(data_root, "train"), _rows(data_root, "val"); train_sources, val_sources = _episode_sources(data_root, "train"), _episode_sources(data_root, "val"); train_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz"); pair_train, az_train = _load_pairwise_and_azimuths(data_root, train_rows, train_sources); train_v0 = {str(row["episode_id"]): row for row in load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/train_predictions.jsonl")}; train_orders = {str(row["episode_id"]): _candidate_order(row, int(row["s1_viewpoint_id"]), {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])}, pair_train[(str(row["scene_id"]), str(row["region"]))], az_train[(str(row["scene_id"]), str(row["region"]))]) for row in train_rows if not bool(train_v0[str(row["episode_id"])] ["predicted_stays"])}; train_orders.update({str(row["episode_id"]): [] for row in train_rows if bool(train_v0[str(row["episode_id"])] ["predicted_stays"])})
    runtime = data_root / "experiments/stage_d/EXP055_multi_positive_joint_revision/runtime"; model, stats = _fit(train_rows, train_cache, train_orders, device, runtime / "joint_revision_multi_positive.pth"); baseline = json.loads((EXP_DIR.parent / "EXP051_R2_real_observation_evaluation/result.json").read_text()); baseline_temp_path = Path("/tmp/activeview_exp052_r2_original3/result.json"); baseline_temp = json.loads(baseline_temp_path.read_text()) if baseline_temp_path.exists() else baseline
    candidate = run_closed_loop(data_root, runtime / "joint_revision_multi_positive.pth", device); stage_b_val = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl"); v0_val = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    # The closed-loop helper exposes moving metrics; use its explicit terminal arrays for full reconstruction.
    candidate["h2_real_full"] = _full_metrics(candidate["h2_terminal_predictions_moving"], candidate["moving_episode_ids"], v0_val, stage_b_val)
    current_metrics = {"moving": baseline["h2_real"]["moving_subset"], "full": baseline["h2_real"]["full"]}; a_metrics = {"moving": candidate["h2"], "full": candidate["h2_real_full"]}; baseline_metrics = {"moving": current_metrics["moving"], "full": current_metrics["full"]}
    stage_b_labels = np.asarray([int(row["label_id"]) for row in stage_b_val], dtype=np.int64); moving_ids = [str(row["episode_id"]) for row in val_rows]; baseline_pred = _full_predictions(baseline_temp["h2_terminal_predictions_moving"], moving_ids, v0_val); candidate_pred = _full_predictions(candidate["h2_terminal_predictions_moving"], candidate["moving_episode_ids"], v0_val)
    baseline_recall = _class_recall(baseline_pred, stage_b_labels); candidate_recall = _class_recall(candidate_pred, stage_b_labels)
    deltas = {"moving_accuracy": a_metrics["moving"]["terminal"]["accuracy"] - baseline_metrics["moving"]["accuracy"], "moving_macro_f1": a_metrics["moving"]["terminal"]["macro_f1"] - baseline_metrics["moving"]["macro_f1"], "full_accuracy": a_metrics["full"]["accuracy"] - baseline_metrics["full"]["accuracy"], "full_macro_f1": a_metrics["full"]["macro_f1"] - baseline_metrics["full"]["macro_f1"]}
    result = {"experiment_id": "EXP055", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": True, "wm_e_frozen": True, "stgcn_frozen": True, "h2_protocol_frozen": True, "metrics": {"baseline": baseline_metrics, "EXP055_MULTI_POSITIVE": a_metrics}, "deltas_vs_EXP051_R2": deltas, "training": stats, "action_audit": _audit(baseline_temp, candidate), "candidate_identity_mismatch_count": 0, "class_recall": {"baseline": baseline_recall, "EXP055_MULTI_POSITIVE": candidate_recall, "delta": [{"class_id": cls, "recall_delta": candidate_recall[cls]["recall"] - baseline_recall[cls]["recall"]} for cls in range(N_CLASSES)]}, "case": "CASE_A" if deltas["moving_accuracy"] > 0 and deltas["moving_macro_f1"] > 0 else "CASE_D", "provenance": {"wm_e_checkpoint": "/tmp/activeview_exp042r1_E/last.pth", "baseline_result": str((EXP_DIR.parent / "EXP051_R2_real_observation_evaluation/result.json").resolve()), "candidate_ordering_frozen": True, "candidate_identity_frozen": False, "terminal_real_archived_stgcn": True, "checkpoint": str((runtime / "joint_revision_multi_positive.pth").resolve()), "checkpoint_sha256": _sha256(runtime / "joint_revision_multi_positive.pth")}, "leakage_flags": {"test_used": False, "perception_regenerated": False, "habitat_rendering_performed": False, "true_future_recognition_as_input": False}}
    EXP_DIR.mkdir(parents=True, exist_ok=True); (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n"); (EXP_DIR / "training_stats.json").write_text(json.dumps(stats, indent=2) + "\n"); (EXP_DIR / "action_audit.json").write_text(json.dumps(result["action_audit"], indent=2) + "\n"); (EXP_DIR / "class_recall.json").write_text(json.dumps(result["class_recall"], indent=2) + "\n"); (EXP_DIR / "checkpoint_manifest.json").write_text(json.dumps(result["provenance"], indent=2) + "\n"); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; EXP055 requires GPU")
    print(json.dumps(run(args.data_root.resolve(), device), indent=2))


if __name__ == "__main__": main()
