"""Reporting helpers for the reduced12 privileged-information ladder."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from activeview.scripts.eval.reduced12_nbv_utils import LABELS, correlation

NUM_CLASSES = len(LABELS)
NUM_VIEWS = 32


def json_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [[float(value) if np.isfinite(value) else None for value in row] for row in np.asarray(values)]


def target_stats(target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(target)[np.asarray(mask, dtype=bool)]
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)), "count": int(values.size)}


def ranking_metrics(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    predicted: np.ndarray,
    target: np.ndarray,
    ids: np.ndarray,
    mask: np.ndarray,
    oracle_actions: Sequence[int],
) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)
    candidate_valid = valid.copy()
    candidate_valid[:, 0] = False
    within: list[float] = []
    candidate_within: list[float] = []
    top1: list[int] = []
    top3: list[int] = []
    stay_move: list[int] = []
    for index in range(len(rows)):
        all_slots = np.flatnonzero(valid[index])
        candidate_slots = np.flatnonzero(candidate_valid[index])
        if all_slots.size >= 2:
            within.append(correlation(predicted[index, all_slots], target[index, all_slots], spearman=True))
            order = all_slots[np.argsort(-predicted[index, all_slots], kind="mergesort")]
            oracle_slot = int(np.flatnonzero(ids[index] == int(oracle_actions[index]))[0])
            top1.append(int(order[0] == oracle_slot))
            top3.append(int(oracle_slot in order[:3]))
            stay_move.append(int((order[0] == 0) == (oracle_slot == 0)))
        if candidate_slots.size >= 2:
            candidate_within.append(correlation(predicted[index, candidate_slots], target[index, candidate_slots], spearman=True))
    return {
        "branch": name,
        "candidate_level_spearman_all_legal": correlation(predicted[valid], target[valid], spearman=True),
        "candidate_level_spearman_candidate_only": correlation(predicted[candidate_valid], target[candidate_valid], spearman=True),
        "within_context_spearman_mean": float(np.mean(within)) if within else 0.0,
        "within_context_spearman_median": float(np.median(within)) if within else 0.0,
        "candidate_only_within_context_spearman_mean": float(np.mean(candidate_within)) if candidate_within else 0.0,
        "candidate_only_within_context_spearman_median": float(np.median(candidate_within)) if candidate_within else 0.0,
        "gt_true_logp_oracle_top1_overlap": float(np.mean(top1)) if top1 else 0.0,
        "gt_true_logp_oracle_top3_overlap": float(np.mean(top3)) if top3 else 0.0,
        "stay_move_agreement": float(np.mean(stay_move)) if stay_move else 0.0,
        "test_used": False,
    }


def action_view_preference(
    utility: np.ndarray,
    logp: np.ndarray,
    ids: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    sums = np.zeros((NUM_CLASSES, NUM_VIEWS), dtype=np.float64)
    counts = np.zeros((NUM_CLASSES, NUM_VIEWS), dtype=np.int64)
    correct = np.zeros((NUM_CLASSES, NUM_VIEWS), dtype=np.int64)
    for index, label in enumerate(labels):
        for slot in np.flatnonzero(mask[index]):
            view = int(ids[index, slot])
            sums[int(label), view] += float(utility[index, slot])
            counts[int(label), view] += 1
            correct[int(label), view] += int(int(np.argmax(logp[index, slot])) == int(label))
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    accuracy = np.divide(correct, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    summaries: list[dict[str, Any]] = []
    best_ids: list[int | None] = []
    for label, action in enumerate(LABELS):
        available = np.flatnonzero(np.isfinite(means[label]))
        best = int(available[np.argmax(means[label, available])]) if available.size else None
        worst = int(available[np.argmin(means[label, available])]) if available.size else None
        best_ids.append(best)
        summaries.append({
            "label": action,
            "best_mean_viewpoint": best,
            "worst_mean_viewpoint": worst,
            "best_worst_utility_gap": float(means[label, best] - means[label, worst]) if best is not None and worst is not None else None,
            "best_worst_accuracy_gap": float(accuracy[label, best] - accuracy[label, worst]) if best is not None and worst is not None else None,
            "available_viewpoint_count": int(available.size),
        })
    return {
        "mean_gt_true_logp_by_action_view": json_matrix(means),
        "accuracy_by_action_view": json_matrix(accuracy),
        "count_by_action_view": counts.tolist(),
        "summary": summaries,
        "best_viewpoint_ids": best_ids,
        "unique_best_viewpoint_ids": sorted({value for value in best_ids if value is not None}),
        "interpretation": "sparse legal-action Train aggregation; entries are reported only where a viewpoint occurs",
    }


def visibility_interaction(
    utility: np.ndarray,
    visibility: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    candidate = np.asarray(mask, dtype=bool).copy()
    candidate[:, 0] = False
    values = np.asarray(visibility)[candidate]
    edges = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]) if values.size else np.asarray([0.0, 0.0])
    result: dict[str, Any] = {"visibility_tertile_edges": [float(x) for x in edges], "by_action": {}}
    for label, action in enumerate(LABELS):
        bins: dict[str, dict[str, float | int]] = {}
        for bin_index, bin_name in enumerate(("low", "medium", "high")):
            selected: list[float] = []
            for index in np.flatnonzero(labels == label):
                for slot in np.flatnonzero(candidate[index]):
                    if int(np.digitize(visibility[index, slot], edges, right=False)) == bin_index:
                        selected.append(float(utility[index, slot]))
            bins[bin_name] = {"count": len(selected), "mean_gt_true_logp": float(np.mean(selected)) if selected else 0.0}
        result["by_action"][action] = bins
    return result


def analysis(result: Mapping[str, Any]) -> str:
    methods = result["methods"]
    ladder = result["information_ladder"]
    g = float(ladder["GeometryOnly Utility"]["accuracy"])
    vg = float(ladder["RealVisibility+Geometry"]["accuracy"])
    yg = float(ladder["GTAction+Geometry"]["accuracy"])
    yvg = float(ladder["GTAction+RealVisibility+Geometry"]["accuracy"])
    oracle = float(ladder["GT-TrueLogP Oracle"]["accuracy"])
    if yg >= 0.60 or yg - g >= 0.08:
        conclusion = "A. PRE-ACTION LIMIT IS MAINLY UNKNOWN ACTION IDENTITY"
    elif yvg - yg >= 0.05:
        conclusion = "B. PRE-ACTION LIMIT IS MAINLY OBSERVABILITY PREDICTION"
    elif oracle - yvg >= 0.12:
        conclusion = "D. LARGE RESIDUAL REMAINS EVEN WITH ACTION+VISIBILITY; PRE-ACTION ORACLE IS NOT REALISTICALLY PREDICTABLE"
    else:
        conclusion = "C. BOTH ACTION IDENTITY AND OBSERVABILITY MATTER STRONGLY"
    lines = [
        "# Privileged Information Ladder / Oracle Gap Decomposition", "", "Protocol:",
        "pre-action / frame-0 full-view terminal recognition", "", "Train:", "Policy Train", "",
        "Val:", "Moving Val", "", "Policy Test:", "false", "", "Action set:",
        "Stay + Stage-A legal candidates", "", "Recognizer:", "frozen ST-GCN + frozen shared head", "",
        "Terminal evaluation:", "selected viewpoint's real full 30-frame skeleton", "", "GT action:",
        "privileged diagnostic input only where explicitly stated", "", "Real candidate Frame0SceneVisibility:",
        "privileged diagnostic input only where explicitly stated", "", "Candidate recognizer output:",
        "training target / oracle only, never learned-branch inference input", "",
        f"Moving Val contexts: {result['population']['moving_val_contexts']}; Train contexts: {result['population']['train_contexts']}.",
        f"Existing GeometryOnly-TrueLogP reference protocol_match={result['target_audit']['existing_geometry_only_reference']['protocol_match']}; reference Acc/F1={result['target_audit']['existing_geometry_only_reference']['accuracy']:.6f}/{result['target_audit']['existing_geometry_only_reference']['macro_f1']:.6f}. This ladder re-runs GeometryOnly with the unified architecture used by all learned branches.",
        "", "## Information ladder (Moving Val)", "", "| Method | Information | Deployable? | Accuracy | Macro-F1 |", "|---|---|---:|---:|---:|",
    ]
    info = result["method_information"]
    names = ("Stay", "Random", "GeometryOnly Utility", "RGBGlobal Task+VisibilityAux (historical)", "Real Frame0 Visibility argmax", "RealVisibility+Geometry", "Global Viewpoint Prior", "GTAction+ViewpointPrior", "GTAction+Geometry", "GTAction+RealVisibility", "GTAction+RealVisibility+Geometry", "GT-TrueLogP Oracle")
    for name in names:
        metric = methods[name]
        lines.append(f"| {name} | {info[name]['information']} | {str(info[name]['deployable'])} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend(["", "## Descriptive gap quantities", "", "These are descriptive information gains, not additive causal effects.", "", f"- VisibilityGain_over_G = {vg - g:+.6f}", f"- ActionGain_over_G = {yg - g:+.6f}", f"- ConditionalVisibilityGain = {yvg - yg:+.6f}", f"- ConditionalActionGain = {yvg - vg:+.6f}", f"- ResidualGap = {oracle - yvg:+.6f}", f"- GTAction+Geometry - DeployableBest = {yg - methods['RGBGlobal Task+VisibilityAux (historical)']['accuracy']:+.6f}", f"- GTAction+Visibility+Geometry - DeployableBest = {yvg - methods['RGBGlobal Task+VisibilityAux (historical)']['accuracy']:+.6f}", f"- Oracle - DeployableBest = {oracle - methods['RGBGlobal Task+VisibilityAux (historical)']['accuracy']:+.6f}", "", "## Utility ranking", "", "The ranking table reports frozen target utility U(c)=log p_y(c); future recognizer output is not an inference feature.", "", "| Branch | Candidate Spearman | Within-context Spearman | Top-1 overlap | Top-3 overlap |", "|---|---:|---:|---:|---:|"])
    for name, metric in result["ranking_metrics"].items():
        lines.append(f"| {name} | {metric['candidate_level_spearman_candidate_only']:.6f} | {metric['candidate_only_within_context_spearman_mean']:.6f} | {metric['gt_true_logp_oracle_top1_overlap']:.6f} | {metric['gt_true_logp_oracle_top3_overlap']:.6f} |")
    lines.extend(["", "## High-occlusion subset", "", f"Bottom tertile of current frame-0 Stay visibility: {result['occlusion_stratified_metrics']['count']} contexts.", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name, metric in result["occlusion_stratified_metrics"]["methods"].items():
        lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} |")
    lines.extend(["", "## Action-specific viewpoint preference", "", f"Train legal-action aggregation yields best viewpoint IDs {result['action_view_preference']['unique_best_viewpoint_ids']}; these are descriptive and sparse where a viewpoint is observed.", "", "## Main conclusion", "", f"**{conclusion}**", "", "GTAction branches are not candidate deployment methods. They quantify how much of the oracle gap would become predictable if action identity were known at selection time.", "The gap values should not be interpreted as a strict additive causal decomposition because action, visibility and geometry interact.", "No follow-up method was started automatically.", "", "```text", "policy_test_used=false", "training_split=Policy Train", "evaluation_split=Moving Val", "future_candidate_recognizer_output_used_as_input=false", "gt_action_used_as_input_only_for_privileged_branches=true", "real_candidate_visibility_used_as_input_only_for_privileged_branches=true", "```"])
    return "\n".join(lines) + "\n"


__all__ = ["action_view_preference", "analysis", "json_matrix", "ranking_metrics", "target_stats", "visibility_interaction"]
