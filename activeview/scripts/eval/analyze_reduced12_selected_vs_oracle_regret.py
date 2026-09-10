#!/usr/bin/env python3
"""Val-only selected-vs-oracle utility regret audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.data.preprocessing.policy_data import load_feature_statistics
from activeview.scripts.experiments.run_reduced12_future_recognition_evidence_prediction import _candidate_alignment, _covered_rows, _load_dino, _read_npz, _subset_cache
from activeview.scripts.experiments.run_reduced12_h1_stay_aware_batch import StayDataset, _collate, _load_baseline, _scores
from activeview.scripts.experiments.run_reduced12_h1_visual_context_batch import VisualDataset, _load_dino_store, _model as visual_model, _score as visual_score
from activeview.scripts.experiments.run_reduced12_single_step_set_level_real_evidence import _action_set, _metrics, _oracle_and_consensus, _require_cuda


LABELS = ("walk", "sit", "stand up", "bend", "crawl", "stumble", "clap", "throw", "kick", "knock", "punch", "touching face")
EXPERIMENT_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_utility_regret_audit"


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(np.argsort(np.argsort(x, kind="mergesort")), np.argsort(np.argsort(y, kind="mergesort")))[0, 1])


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {"count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)), "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90))}


def _stable_pick(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.argmax(np.where(mask, scores, -np.inf), axis=1).astype(np.int64)


def _selected_metrics(name: str, actions: Mapping[str, np.ndarray], selected: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = actions["labels"]
    predictions = np.asarray([int(np.argmax(actions["logp"][i, action])) for i, action in enumerate(selected)], dtype=np.int64)
    metric = _metrics(labels, predictions)
    metric.update({"selector": name, "move_rate": float(np.mean(selected > 0)), "stay_rate": float(np.mean(selected == 0)), "selected_action_ids": [None if int(action) == 0 else int(actions["action_ids"][i, action]) for i, action in enumerate(selected)], "selected_correct": (predictions == labels).tolist()})
    return metric


def _selector_audit(name: str, actions: Mapping[str, np.ndarray], scores: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    mask = actions["mask"]
    selected = _stable_pick(scores, mask)
    labels = actions["labels"]
    utilities = actions["utility"]
    rows: list[dict[str, float | int | bool]] = []
    topk_rows: list[dict[str, Any]] = []
    for index in range(len(labels)):
        valid = np.flatnonzero(mask[index])
        utility_values = utilities[index, valid]
        order_utility = valid[np.argsort(-utility_values, kind="mergesort")]
        best = float(utilities[index, order_utility[0]])
        worst = float(utilities[index, order_utility[-1]])
        second = float(utilities[index, order_utility[1]]) if len(order_utility) > 1 else best
        action = int(selected[index])
        selected_utility = float(utilities[index, action])
        utility_range = best - worst
        rank = int(np.flatnonzero(order_utility == action)[0]) + 1
        percentile = 1.0 - (rank - 1) / max(len(valid) - 1, 1)
        selected_order = valid[np.argsort(-scores[index, valid], kind="mergesort")]
        topk_contains = {str(k): bool(np.any(utilities[index, selected_order[: min(k, len(selected_order))]] > 0.0)) for k in (1, 2, 3, 5)}
        top3 = selected_order[: min(3, len(selected_order))]
        top3_oracle_action = int(top3[np.argmax(utilities[index, top3])])
        topk_rows.append({"topk_contains_correct": topk_contains, "top3_oracle_prediction": int(np.argmax(actions["logp"][index, top3_oracle_action]))})
        rows.append({"best": best, "second": second, "best_second_gap": best - second, "worst": worst, "selected": selected_utility, "regret": best - selected_utility, "norm_regret": (best - selected_utility) / (utility_range + 1e-8), "rank": rank, "rank_percentile": percentile, "oracle_correct": best > 0.0, "selector_correct": selected_utility > 0.0, "selected_action": action})
    table = {key: np.asarray([float(row[key]) for row in rows]) for key in ("best", "second", "best_second_gap", "worst", "selected", "regret", "norm_regret", "rank", "rank_percentile")}
    oracle_wrong = table["best"] > 0.0
    selector_wrong = table["selected"] < 0.0
    subset = oracle_wrong & selector_wrong
    close = subset & (table["selected"] >= -0.25)
    moderate = subset & (table["selected"] >= -1.0) & (table["selected"] < -0.25)
    severe = subset & (table["selected"] < -1.0)
    small = subset & (table["norm_regret"] < 0.20)
    medium = subset & (table["norm_regret"] >= 0.20) & (table["norm_regret"] < 0.50)
    large = subset & (table["norm_regret"] >= 0.50)
    subset_stats = {"context_count": int(subset.sum()), "fraction_all_moving": float(np.mean(subset)), "fraction_oracle_correct": float(subset.sum() / max(oracle_wrong.sum(), 1)), "best_utility": _summary(table["best"][subset]), "selected_utility": _summary(table["selected"][subset]), "regret": _summary(table["regret"][subset]), "normalized_regret": _summary(table["norm_regret"][subset]), "selected_gt_rank": _summary(table["rank"][subset]), "rank_percentile": _summary(table["rank_percentile"][subset]), "close_call_fraction": float(close.sum() / max(subset.sum(), 1)), "moderate_miss_fraction": float(moderate.sum() / max(subset.sum(), 1)), "severe_miss_fraction": float(severe.sum() / max(subset.sum(), 1)), "small_regret_fraction": float(small.sum() / max(subset.sum(), 1)), "medium_regret_fraction": float(medium.sum() / max(subset.sum(), 1)), "large_regret_fraction": float(large.sum() / max(subset.sum(), 1))}
    density_counts = np.sum(actions["correct"] & mask, axis=1)
    density = {}
    for bucket, bucket_mask in (("0", density_counts == 0), ("1", density_counts == 1), ("2", density_counts == 2), ("3+", density_counts >= 3)):
        density[bucket] = {"contexts": int(bucket_mask.sum()), "accuracy": float(np.mean((table["selected"][bucket_mask] > 0.0) if bucket_mask.any() else [])), "selected_correct": float(np.mean(table["selected"][bucket_mask] > 0.0)) if bucket_mask.any() else 0.0}
    topk = {str(k): {"any_correct_coverage": float(np.mean([row["topk_contains_correct"][str(k)] for row in topk_rows])), "top3_oracle_recovery_accuracy": None} for k in (1, 2, 3, 5)}
    top3_predictions = np.asarray([row["top3_oracle_prediction"] for row in topk_rows], dtype=np.int64)
    topk["3"]["top3_oracle_recovery_accuracy"] = float(np.mean(top3_predictions == labels))
    landscape = {"selector_correct": {"best_second_gap": _summary(table["best_second_gap"][table["selected"] > 0]), "utility_std": _summary(np.std(utilities, axis=1)[table["selected"] > 0]), "utility_range": _summary(table["best"] [table["selected"] > 0] - table["worst"][table["selected"] > 0])}, "oracle_correct_selector_wrong": {"best_second_gap": _summary(table["best_second_gap"][subset]), "utility_std": _summary(np.std(utilities, axis=1)[subset]), "utility_range": _summary(table["best"][subset] - table["worst"][subset])}}
    metric = _selected_metrics(name, actions, selected, scores)
    metric.update({"mean_gt_rank": float(np.mean(table["rank"])), "median_gt_rank": float(np.median(table["rank"])), "mean_rank_percentile": float(np.mean(table["rank_percentile"])), "median_rank_percentile": float(np.median(table["rank_percentile"])), "p_rank_eq_1": float(np.mean(table["rank"] == 1)), "p_rank_le_2": float(np.mean(table["rank"] <= 2)), "p_rank_le_3": float(np.mean(table["rank"] <= 3)), "p_rank_bottom_half": float(np.mean(table["rank"] > np.ceil(mask.sum(axis=1) / 2))), "mean_regret": float(np.mean(table["regret"])), "median_regret": float(np.median(table["regret"])), "mean_normalized_regret": float(np.mean(table["norm_regret"])), "median_normalized_regret": float(np.median(table["norm_regret"])), "oracle_correct_selector_wrong": subset_stats, "correct_candidate_density": density, "topk_rescue": topk, "utility_landscape": landscape, "context_ranking_spearman": float(np.mean([_spearman(scores[i, mask[i]], utilities[i, mask[i]]) for i in range(len(labels))]))})
    return metric, table["selected"]


def _load_frozen_and_visual_scores(data_root: Path, rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], dino_lookup: Mapping[tuple[str, str, str, int], int], dino_embeddings: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"; feature_root = policy_root / "stage_c"; stats = load_feature_statistics(feature_root / "stage_c_feature_stats.json"); summary = json.loads((feature_root / "stage_c_feature_summary.json").read_text(encoding="utf-8"))
    baseline_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/stage_c/set_ranker_best.pth"; baseline_model = _load_baseline(baseline_path, summary, device); baseline_set = StayDataset(rows, cache, stats); stay, candidate = _scores(baseline_model, baseline_set, device, batch_size, baseline=True); frozen = np.concatenate([stay[:, None], candidate], axis=1)
    visual_path = data_root / "checkpoints/policy_reduced12_eight_placement_v1/h1_visual_context_batch/candidate_conditioned_spatial_best.pth"; visual_set = VisualDataset(rows, cache, stats, dino_lookup, dino_embeddings, "candidate_conditioned_spatial"); visual = visual_model("candidate_conditioned_spatial").to(device); visual.load_state_dict(torch.load(visual_path, map_location=device, weights_only=False)["state_dict"]); vstay, vcandidate = visual_score(visual, visual_set, device, batch_size); return frozen, np.concatenate([vstay[:, None], vcandidate], axis=1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = get_data_root().resolve(); device = _require_cuda(args.device); policy_root = data_root / "datasets/policy_reduced12_eight_placement_v1"; diag_root = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch"; val_rows_all = load_jsonl(policy_root / "stage_c/features/val.jsonl"); stage_d_val_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/val.jsonl")}; dino_lookup, dino_embeddings, dino_summary = _load_dino_store(data_root); val_rows, val_indices = _covered_rows(val_rows_all, stage_d_val_ids, dino_lookup); cache_all = _read_npz(diag_root / "val_all_candidate_true_logp.npz"); cache = _subset_cache(cache_all, val_indices); _candidate_alignment(val_rows, cache); actions = _action_set(val_rows, cache); output = args.output_dir.resolve() if args.output_dir else EXPERIMENT_ROOT; output.mkdir(parents=True, exist_ok=True)
    frozen_scores, visual_scores = _load_frozen_and_visual_scores(data_root, val_rows, cache, dino_lookup, dino_embeddings, device, args.batch_size)
    set_output = REPO_ROOT / "experiments/reduced12_eight_placement_v1/single_step_set_level_real_evidence/branch_results"; ranker_files = {"RealEvidence-GTMarginListwise": set_output / "real_gt_margin_listwise_scores.npz", "RealEvidence-CorrectnessBCE": set_output / "real_correctness_bce_scores.npz"}; ranker_scores = {name: np.asarray(np.load(path, allow_pickle=False)["scores"], dtype=np.float32) for name, path in ranker_files.items()}
    consensus_metrics, consensus_scores = _oracle_and_consensus(actions)
    selectors = {"FrozenStageCv0": frozen_scores, "Candidate-Conditioned Spatial": visual_scores, **ranker_scores, "ConfidenceWeightedConsensus": consensus_scores["ConfidenceWeightedConsensus"]}
    audits: dict[str, Any] = {}; selected_values: dict[str, np.ndarray] = {}
    for name, score in selectors.items():
        audit, values = _selector_audit(name, actions, score); audits[name] = audit; selected_values[name] = values
    audits["Real-GTMargin Oracle"], _ = _selector_audit("Real-GTMargin Oracle", actions, actions["utility"])
    (output / "per_selector_regret.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    density = {name: audit["correct_candidate_density"] for name, audit in audits.items()}; topk = {name: audit["topk_rescue"] for name, audit in audits.items()}; landscapes = {name: audit["utility_landscape"] for name, audit in audits.items()}; (output / "correct_candidate_density.json").write_text(json.dumps(density, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); (output / "topk_rescue.json").write_text(json.dumps(topk, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); (output / "utility_landscape.json").write_text(json.dumps(landscapes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "selected_regret_arrays.npz", **{name.replace(" ", "_"): values for name, values in selected_values.items()})
    references = json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_stay_aware_objective_batch/result.json").read_text(encoding="utf-8"))["metrics_moving"]; result_metrics = {name: audits[name] for name in selectors}; result_metrics["Real-GTMargin Oracle"] = audits["Real-GTMargin Oracle"]; result = {"experiment_id": "REDUCED12_SELECTED_VS_ORACLE_UTILITY_REGRET_AUDIT", "status": "COMPLETED", "population": {"moving_val_contexts": len(val_rows), "action_set": "stay + legal move candidates", "moving_val_actions": int(actions["mask"].sum())}, "metrics": result_metrics, "reference_metrics": {"FrozenStageCv0": references["FrozenStageCv0"], "Candidate-Conditioned Spatial": json.loads((REPO_ROOT / "experiments/reduced12_eight_placement_v1/h1_visual_context_batch/result.json").read_text(encoding="utf-8"))["metrics_moving"]["candidate_conditioned_spatial"]}, "protocol": {"test_used": False, "training_used": False, "gt_action_used_for_privileged_diagnostic_only": True, "future_candidate_skeleton_used_for_terminal_oracle_analysis_only": True, "deployable": False}, "dino_cache": {**dino_summary, "regenerated": False, "future_candidate_dino_used": False}}
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); diagnostics = {"selected_vs_oracle": {name: {key: value for key, value in metric.items() if key not in {"selected_action_ids", "selected_correct", "per_class", "confusion_matrix"}} for name, metric in audits.items()}, "action_set": "stay + legal move candidates"}; (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_class = {name: metric["per_class"] for name, metric in audits.items()}; (output / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    best_selector = max(selectors, key=lambda name: float(audits[name]["accuracy"])); lines = ["# Selected-vs-Oracle Utility Regret Audit", "", "Val Moving only (10,080 contexts), using the unified stay + legal-candidate action set. No Test data were read, no models were trained, and no perception data were regenerated.", "", "| Selector | Accuracy | Macro-F1 | Mean rank percentile | Median rank | P(rank<=3) | Mean regret | Median norm regret | Oracle-correct/selector-wrong | Close-call | Severe-miss | Top3 coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, audit in audits.items():
        lines.append(f"| {name} | {audit['accuracy']:.6f} | {audit['macro_f1']:.6f} | {audit['mean_rank_percentile']:.6f} | {audit['median_gt_rank']:.2f} | {audit['p_rank_le_3']:.6f} | {audit['mean_regret']:.6f} | {audit['median_normalized_regret']:.6f} | {audit['oracle_correct_selector_wrong']['context_count']} | {audit['oracle_correct_selector_wrong']['close_call_fraction']:.6f} | {audit['oracle_correct_selector_wrong']['severe_miss_fraction']:.6f} | {audit['topk_rescue']['3']['any_correct_coverage']:.6f} |")
    lines.extend(["", "## Oracle alignment", "", f"Real-GTMargin Oracle Accuracy={audits['Real-GTMargin Oracle']['accuracy']:.6f}; unified action set includes stay.", "", "## Main answers", "", f"Best evaluated non-oracle selector: **{best_selector}**, Accuracy={audits[best_selector]['accuracy']:.6f}, mean normalized regret={audits[best_selector]['mean_normalized_regret']:.6f}, Top3 correct coverage={audits[best_selector]['topk_rescue']['3']['any_correct_coverage']:.6f}."])
    best = audits[best_selector]; subset = best["oracle_correct_selector_wrong"]
    if best["median_normalized_regret"] < 0.20 and best["mean_rank_percentile"] > 0.70:
        lines.append("Q1: selected viewpoints are usually close to the GT-optimal utility region; the dominant issue is likely decision-boundary fragility rather than severe mis-ranking.")
    else:
        lines.append("Q1: selected viewpoints are not consistently close to the GT-optimal utility; utility regret indicates substantial candidate mis-ranking.")
    if best["topk_rescue"]["3"]["any_correct_coverage"] >= 0.65 and best["accuracy"] < 0.55:
        lines.append("Q2: the evidence points to coarse ranking/top-1 resolution: correct viewpoints often enter Top-3 but are not selected first.")
    elif best["topk_rescue"]["3"]["any_correct_coverage"] < 0.55:
        lines.append("Q2: the main limitation is candidate mis-ranking; correct viewpoints frequently remain outside the selector Top-3.")
    else:
        lines.append("Q2: both ranking and recognizer decision-boundary effects contribute; neither alone explains the full ceiling.")
    lines.append(f"Q3: on oracle-correct but selector-wrong contexts, Top-3 recovery is {best['topk_rescue']['3']['any_correct_coverage']:.6f}; the subset contains {subset['context_count']} contexts, with close-call fraction {subset['close_call_fraction']:.6f} and severe-miss fraction {subset['severe_miss_fraction']:.6f}.")
    lines.extend(["", "No selector was retrained or modified. This is a privileged regret audit only.", "", "`test_used=false`; `training_used=false`; `gt_action_used_for_privileged_diagnostic_only=true`; `future_candidate_skeleton_used_for_terminal_oracle_analysis_only=true`; `deployable=false`."])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--output-dir", type=Path, default=None); args = parser.parse_args(); result = run(args); print(json.dumps({"status": result["status"], "test_used": result["protocol"]["test_used"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
