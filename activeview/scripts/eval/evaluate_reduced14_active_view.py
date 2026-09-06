#!/usr/bin/env python3
"""Evaluate reduced14 ActiveView methods on Val or Test without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import candidate_order, load_pairwise_and_azimuths
from activeview.methods.joint_revision.model import JointRevision, select_actions

NUM_CLASSES = 14


def _metrics(pred: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    p = np.asarray(pred, dtype=np.int64); y = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(y * NUM_CLASSES + p, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls]); precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0; recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"count": int(len(y)), "accuracy": float(np.mean(p == y)) if len(y) else 0.0, "macro_f1": float(np.mean(f1))}


def _source_map(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    return {(str(r["scene_id"]), str(r["region"]), str(r["record_id"])): str(root / str(r["scene_id"]) / str(r["region"]) / f"{r['record_id']}.npz") for r in rows}


def _orders(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    source = _source_map(data_root, rows)
    pair, az = load_pairwise_and_azimuths(data_root, rows, source, pair_root=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1/pairwise_viewpoint_geodesic")
    result: dict[str, list[int]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]))
        result[str(row["episode_id"])] = candidate_order(row, int(row["s1_viewpoint_id"]), {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])}, pair[key], az[key])
    return result


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def evaluate(data_root: Path, split: str, device: torch.device) -> dict[str, Any]:
    root = data_root / "datasets/policy_reduced14_kneel_eight_placement_v1"
    all_v0 = load_jsonl(root / "stage_c/v0_predictions" / f"{split}_predictions.jsonl")
    moving_rows = load_jsonl(root / "stage_d/features" / f"{split}.jsonl")
    cache = _load_cache(root / "counterfactual_cache" / f"{split}.npz")
    if {str(r["episode_id"]) for r in moving_rows} != set(str(v) for v in cache["episode_ids"].tolist()):
        raise ValueError(f"{split} Stage-D/cache episode IDs are not aligned")
    v0_by_id = {str(r["episode_id"]): r for r in all_v0}
    moving_ids = [str(r["episode_id"]) for r in moving_rows]
    moving_set = set(moving_ids)
    if any(bool(v0_by_id[e]["predicted_stays"]) for e in moving_ids):
        raise ValueError("Stage-D rows must be exactly frozen-v0 moving contexts")
    labels_full = [int(r["label_id"]) for r in all_v0]
    labels_moving = [int(v0_by_id[e]["label_id"]) for e in moving_ids]
    index = {e: i for i, e in enumerate(cache["episode_ids"].tolist())}
    current_s0_pred = {e: int(v0_by_id[e]["current_predicted_label_id"]) for e in v0_by_id}
    current_s1_pred = {e: int(np.argmax(cache["current_logp_s1"][index[e]])) for e in moving_ids}
    no_move_m = [current_s0_pred[e] for e in moving_ids]
    frozen_m = current_s1_pred.copy()
    orders = _orders(data_root, moving_rows)
    order_by_index = [orders[e] for e in moving_ids]
    random_rng = np.random.default_rng(42)
    random_m: dict[str, int] = {}
    safe_m: dict[str, int] = {}
    candidate_m: dict[str, int] = {}
    for e, candidates, label in zip(moving_ids, order_by_index, labels_moving):
        i = index[e]; current = float(cache["current_logp_s1"][i, label]); utilities = [(int(c), float(cache["true_logp"][i, c, label] - current)) for c in candidates]
        random_choice = int(random_rng.integers(0, len(candidates) + 1))
        selected_random = None if random_choice == 0 else candidates[random_choice - 1]
        random_m[e] = int(np.argmax(cache["current_logp_s1"][i])) if selected_random is None else int(np.argmax(cache["true_logp"][i, selected_random]))
        best = max(utilities, key=lambda item: (item[1], -candidates.index(item[0]))) if utilities else (None, -np.inf)
        safe_m[e] = int(np.argmax(cache["current_logp_s1"][i])) if best[1] <= 0.0 else int(np.argmax(cache["true_logp"][i, best[0]]))
        candidate_m[e] = int(np.argmax(cache["true_logp"][i, max(candidates, key=lambda c: float(cache["true_logp"][i, c, label]))])) if candidates else int(np.argmax(cache["current_logp_s1"][i]))
    wm = JointRevision(num_classes=NUM_CLASSES).to(device)
    checkpoint = data_root / "checkpoints/activeview_reduced14_eight_placement_v1/joint_revision_multi_positive.pth"
    payload = torch.load(checkpoint, map_location=device, weights_only=False); wm.load_state_dict(payload.get("model_state_dict", payload["state_dict"])); wm.eval()
    jr_selected = select_actions(wm, cache, moving_rows, orders, budget="ALL_LEGAL", device=device)
    jr_m = {e: int(np.argmax(cache["current_logp_s1"][index[e]])) if action is None else int(np.argmax(cache["true_logp"][index[e], int(action)])) for e, action in zip(moving_ids, jr_selected)}

    moving_predictions = {"NoMove": no_move_m, "FrozenStageCv0": list(frozen_m.values()), "Random": [random_m[e] for e in moving_ids], "SafeOracle": [safe_m[e] for e in moving_ids], "CandidateOracle": [candidate_m[e] for e in moving_ids], "Multi-positive H2": [jr_m[e] for e in moving_ids]}
    full_predictions: dict[str, list[int]] = {}
    for name, moving_pred in moving_predictions.items():
        by_id = dict(zip(moving_ids, moving_pred)); full_predictions[name] = [by_id[e] if e in moving_set else current_s0_pred[e] for e in (str(r["episode_id"]) for r in all_v0)]
    result: dict[str, Any] = {"experiment_id": "REDUCED14_ACTIVE_VIEW", "status": "COMPLETED", "split": split, "test_used": split == "test", "population": {"full": len(all_v0), "moving": len(moving_rows), "split_source": "raw-val Train/Val/Test=357/120/120, no scene split"}, "methods": {"full": {name: _metrics(pred, labels_full) for name, pred in full_predictions.items()}, "moving": {name: _metrics(pred, labels_moving) for name, pred in moving_predictions.items()}}, "protocol": {"taxonomy": "reduced14_kneel", "placements_per_scene": 8, "candidate_budget": "ALL_LEGAL", "terminal_observation": "real archived skeleton through frozen ST-GCN", "seed": 42}, "leakage_flags": {"test_used": split == "test", "true_future_recognition_as_model_input": False, "future_candidate_rgb_used": False, "habitat_rendering_performed": False}}
    out = data_root / "experiments/reduced14_eight_placement_v1/active_view_evaluation"; out.mkdir(parents=True, exist_ok=True); (out / f"{split}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--split", choices=("val", "test"), required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; evaluator requires GPU")
    print(json.dumps(evaluate(args.data_root.resolve(), args.split, device), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
