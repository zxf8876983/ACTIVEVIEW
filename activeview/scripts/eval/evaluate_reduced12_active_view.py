#!/usr/bin/env python3
"""Train/Val-only benchmark for the reduced12 eight-placement protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import candidate_order, context_key, load_pairwise_and_azimuths
from activeview.methods.joint_revision.model import JointRevision, select_actions

NUM_CLASSES = 12


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.int64); target = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(target * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls]); precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0; recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0
        f1.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"count": int(target.size), "accuracy": float(np.mean(pred == target)) if target.size else 0.0, "macro_f1": float(np.mean(f1)) if f1 else 0.0}


def _sources(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/habitat-train/00006-00087"
    return {context_key(row): str(root / context_key(row)[0] / context_key(row)[1] / f"{context_key(row)[2]}.npz") for row in rows}


def _orders(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    source = _sources(data_root, rows); pair, az = load_pairwise_and_azimuths(data_root, rows, source, pair_root=data_root / "datasets/policy_reduced12_eight_placement_v1/pairwise_viewpoint_geodesic")
    return {str(row["episode_id"]): candidate_order(row, int(row["s1_viewpoint_id"]), {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])}, pair[(str(row["scene_id"]), str(row["region"]))], az[(str(row["scene_id"]), str(row["region"]))]) for row in rows}


def _cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _rank_corr(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(left, dtype=np.float64); y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0, 0.0
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


def _h0_predictions(utility_rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, Any]]:
    random_pred: dict[str, int] = {}; candidate_pred: dict[str, int] = {}; safe_pred: dict[str, int] = {}; diagnostics: list[dict[str, float | int]] = []
    for row in utility_rows:
        label = int(row["label_id"]); current = row["current"]; candidates = list(row["candidates"]); current_pred = int(current["predicted_label_id"])
        legal_correct = [int(int(item["predicted_label_id"]) == label) for item in candidates]
        true_scores = [float(item["logp_true"]) for item in candidates]
        choice = int(rng.integers(0, len(candidates) + 1)) if candidates else 0
        random_pred[str(row["episode_id"])] = current_pred if choice == 0 else int(candidates[choice - 1]["predicted_label_id"])
        best_index = int(np.argmax(true_scores)) if candidates else -1
        candidate_pred[str(row["episode_id"])] = current_pred if best_index < 0 else int(candidates[best_index]["predicted_label_id"])
        current_score = float(current["logp_true"]); best_score = true_scores[best_index] if best_index >= 0 else -np.inf
        safe_pred[str(row["episode_id"])] = current_pred if best_score <= current_score else int(candidates[best_index]["predicted_label_id"])
        diagnostics.append({"any_correct": int(bool(current["correct"]) or any(legal_correct)), "correct_view_ratio": float(np.mean(legal_correct)) if legal_correct else 0.0, "num_correct_legal_views": int(sum(legal_correct)), "legal_view_count": int(len(candidates)), "legal_view_accuracy": float(np.mean(legal_correct)) if legal_correct else 0.0})
    aggregate = {name: float(np.mean([float(item[name]) for item in diagnostics])) for name in ("any_correct", "correct_view_ratio", "num_correct_legal_views", "legal_view_accuracy")}
    aggregate["any_correct_rate"] = aggregate.pop("any_correct")
    aggregate["mean_legal_view_count"] = float(np.mean([item["legal_view_count"] for item in diagnostics]))
    return random_pred, candidate_pred, safe_pred, aggregate


def evaluate(
    data_root: Path,
    device: torch.device,
    *,
    counterfactual_root: Path | None = None,
    wm_checkpoint: Path | None = None,
    jr_checkpoint: Path | None = None,
) -> dict[str, Any]:
    root = data_root / "datasets/policy_reduced12_eight_placement_v1"; utility_rows = load_jsonl(root / "stage_b/utility_labels/val.jsonl"); v0_rows = load_jsonl(root / "stage_c/predictions/val_predictions.jsonl"); moving_rows = load_jsonl(root / "stage_d/features/val.jsonl"); cache = _cache((counterfactual_root or (root / "counterfactual_cache")) / "val.npz")
    train_rows = load_jsonl(root / "stage_d/features/train.jsonl")
    v0_by_id = {str(row["episode_id"]): row for row in v0_rows}; utility_by_id = {str(row["episode_id"]): row for row in utility_rows}; moving_ids = [str(row["episode_id"]) for row in moving_rows]; moving_set = set(moving_ids); cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
    if moving_set != set(cache_index): raise ValueError("Val Stage-D and counterfactual cache IDs are not aligned")
    labels_full = [int(row["label_id"]) for row in v0_rows]; labels_moving = [int(v0_by_id[e]["label_id"]) for e in moving_ids]
    s0_pred = {str(row["episode_id"]): int(row["current_predicted_label_id"]) for row in v0_rows}
    rng = np.random.default_rng(42); random_full, candidate_full, safe_full, h0_diag = _h0_predictions(utility_rows, rng)
    orders = _orders(data_root, moving_rows); order_by_id = {episode: orders[episode] for episode in moving_ids}
    random_m = [random_full[e] for e in moving_ids]; candidate_m = [candidate_full[e] for e in moving_ids]; safe_m = [safe_full[e] for e in moving_ids]
    h1_pred = [int(np.argmax(cache["current_logp_s1"][cache_index[e]])) for e in moving_ids]
    safe_h2: list[int] = []
    for episode in moving_ids:
        i = cache_index[episode]; label = int(v0_by_id[episode]["label_id"]); current_score = float(cache["current_logp_s1"][i, label]); candidates = order_by_id[episode]; best = max(candidates, key=lambda c: float(cache["true_logp"][i, c, label]), default=None)
        safe_h2.append(int(np.argmax(cache["current_logp_s1"][i])) if best is None or float(cache["true_logp"][i, best, label]) <= current_score else int(np.argmax(cache["true_logp"][i, best])))
    jr_path = jr_checkpoint or (data_root / "checkpoints/activeview_reduced12_eight_placement_v1/joint_revision_multi_positive.pth")
    model = JointRevision(num_classes=NUM_CLASSES).to(device); payload = torch.load(jr_path, map_location=device, weights_only=False); model.load_state_dict(payload.get("model_state_dict", payload["state_dict"])); model.eval(); jr_actions = select_actions(model, cache, moving_rows, order_by_id, budget="ALL_LEGAL", device=device)
    jr_m = [int(np.argmax(cache["current_logp_s1"][cache_index[e]])) if action is None else int(np.argmax(cache["true_logp"][cache_index[e], int(action)])) for e, action in zip(moving_ids, jr_actions)]
    frozen_m = h1_pred; no_move_m = [s0_pred[e] for e in moving_ids]
    moving_predictions = {"NoMove": no_move_m, "FrozenStageCv0": frozen_m, "Random": random_m, "H0 CandidateOracle": candidate_m, "H0 SafeOracle": safe_m, "Multi-positive H2": jr_m, "FixedH1-H2 SafeOracle": safe_h2}
    full_predictions: dict[str, list[int]] = {}
    for name, moving_prediction in moving_predictions.items():
        by_id = dict(zip(moving_ids, moving_prediction)); full_predictions[name] = [by_id[e] if e in moving_set else s0_pred[e] for e in (str(row["episode_id"]) for row in v0_rows)]
    # H0 methods are defined for every Val episode, not only v0-moving rows.
    full_predictions["Random"] = [random_full[str(row["episode_id"])] for row in v0_rows]
    full_predictions["H0 CandidateOracle"] = [candidate_full[str(row["episode_id"])] for row in v0_rows]
    full_predictions["H0 SafeOracle"] = [safe_full[str(row["episode_id"])] for row in v0_rows]
    imagined_values: list[float] = []; true_values: list[float] = []; agreement_values: list[float] = []; top1_hits: list[float] = []; top3_hits: list[float] = []; oracle_contexts = 0
    for episode in moving_ids:
        i = cache_index[episode]; label = int(v0_by_id[episode]["label_id"]); legal = order_by_id[episode]
        if not legal:
            continue
        predicted = np.asarray(cache["imagined_logp"][i, legal], dtype=np.float64); truth = np.asarray(cache["true_logp"][i, legal], dtype=np.float64)
        agreement_values.extend((np.argmax(predicted, axis=1) == np.argmax(truth, axis=1)).astype(np.float64).tolist())
        imagined_values.extend(predicted[:, label].tolist()); true_values.extend(truth[:, label].tolist())
        positives = np.flatnonzero(np.argmax(truth, axis=1) == label)
        if positives.size:
            oracle_contexts += 1
            order = np.argsort(-predicted[:, label]); top1_hits.append(float(order[0] in positives)); top3_hits.append(float(np.any(np.isin(order[:3], positives))))
    pearson, spearman = _rank_corr(imagined_values, true_values)
    wm_diagnostics = {"legal_candidate_count": len(imagined_values), "recognition_agreement": float(np.mean(agreement_values)) if agreement_values else 0.0, "true_class_pearson": pearson, "true_class_spearman": spearman, "top1_positive_hit": float(np.mean(top1_hits)) if top1_hits else 0.0, "top3_positive_hit": float(np.mean(top3_hits)) if top3_hits else 0.0, "oracle_positive_contexts": oracle_contexts}
    h0_metrics = {"full": {"H0 CandidateOracle": _metrics([candidate_full[str(row["episode_id"])] for row in v0_rows], labels_full), "H0 SafeOracle": _metrics([safe_full[str(row["episode_id"])] for row in v0_rows], labels_full)}, "diagnostics": h0_diag}
    mapping = json.loads((data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-train/label_mapping.json").read_text(encoding="utf-8"))
    labels_by_id = {str(value): key for key, value in mapping.items()}
    wm_path = wm_checkpoint or (data_root / "checkpoints/activeview_reduced12_eight_placement_v1/wm_e/wm_e_best.pth")
    result = {"experiment_id": "REDUCED12_EIGHT_PLACEMENT_ACTIVE_VIEW", "status": "COMPLETED", "test_used": False, "population": {"full_val": len(v0_rows), "moving_val": len(moving_rows), "train_contexts": len(train_rows), "split_source": "reduced12 raw-val Train/Val only"}, "labels": labels_by_id, "class_counts": {"train": {labels_by_id[str(k)]: v for k, v in sorted(Counter(int(row["label_id"]) for row in train_rows).items())}, "val": {labels_by_id[str(k)]: v for k, v in sorted(Counter(int(row["label_id"]) for row in v0_rows).items())}, "moving_val": {labels_by_id[str(k)]: v for k, v in sorted(Counter(int(row["label_id"]) for row in moving_rows).items())}}, "methods": {"full": {name: _metrics(pred, labels_full) for name, pred in full_predictions.items()}, "moving": {name: _metrics(pred, labels_moving) for name, pred in moving_predictions.items()}}, "h0_benchmark": h0_metrics, "wm_diagnostics": wm_diagnostics, "protocol": {"taxonomy": "reduced12_no_kneel_clean", "placements_per_scene": 8, "candidate_budget": "ALL_LEGAL", "terminal_observation": "real archived skeleton through frozen 12-class ST-GCN", "wm_e_checkpoint": str(wm_path.resolve()), "jr_checkpoint": str(jr_path.resolve()), "rgb_dino_available": counterfactual_root is not None}, "leakage_flags": {"test_used": False, "true_future_recognition_as_model_input": False, "future_candidate_rgb_used": False, "habitat_rendering_performed": False}}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--counterfactual-root", type=Path, default=None); parser.add_argument("--wm-checkpoint", type=Path, default=None); parser.add_argument("--jr-checkpoint", type=Path, default=None); parser.add_argument("--output-dir", type=Path, default=None); args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; evaluator requires GPU")
    result = evaluate(args.data_root.resolve(), device, counterfactual_root=args.counterfactual_root, wm_checkpoint=args.wm_checkpoint, jr_checkpoint=args.jr_checkpoint); output = (args.output_dir or (Path(__file__).resolve().parents[3] / "experiments/reduced12_eight_placement_v1/active_view_retraining")).resolve(); output.mkdir(parents=True, exist_ok=True); (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
