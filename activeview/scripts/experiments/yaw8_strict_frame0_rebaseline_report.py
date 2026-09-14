"""Small reporting helpers for the Yaw8Fair strict Frame-0 re-baseline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.scripts.eval.reduced12_utility_source import body_azimuth_bin


def historical_inventory(repo_root: Path) -> dict[str, Any]:
    old_root = repo_root / "experiments/reduced12_eight_placement_v1"

    def metric(path: Path, method: str) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("methods", {}).get(method)
        if isinstance(values, dict):
            return {"accuracy": values.get("accuracy"), "macro_f1": values.get("macro_f1")}
        return None

    entries = [
        {"method": "B0 Stay", "historical": metric(old_root / "frame0_task_utility_predictor_v1/result.json", "Stay"), "selector_inputs": "none (current O0 reference)", "training_target": "none", "architecture": "none", "loss": "none", "seed": None, "candidate_protocol": "stay/current", "strict_frame0_causal": False, "exact_implementation_recoverable": True, "rerun": False, "reason": "diagnostic only; not a next-view action"},
        {"method": "B1 Random legal", "historical": metric(old_root / "frame0_task_utility_predictor_v1/result.json", "Random legal"), "selector_inputs": "legal action IDs", "training_target": "none", "architecture": "random sampler", "loss": "none", "seed": 42, "candidate_protocol": "historical stay+candidate; formal rerun is candidate-only", "strict_frame0_causal": True, "exact_implementation_recoverable": True, "rerun": True, "reason": "formal baseline"},
        {"method": "B2 StaticViewPrior", "historical": metric(old_root / "prior_residual_frame0_nbv/result.json", "StaticViewPrior + old adaptive"), "selector_inputs": "Train-derived viewpoint prior", "training_target": "mean GT-Margin", "architecture": "lookup table over 32 viewpoints", "loss": "mean utility aggregation", "seed": 42, "candidate_protocol": "historical stay+candidate; formal rerun is candidate-only", "strict_frame0_causal": True, "exact_implementation_recoverable": True, "rerun": True, "reason": "Train-derived fixed viewpoint prior"},
        {"method": "V1 Frame0SceneVisibility", "historical": metric(old_root / "frame0_visibility_predictor_v1/result.json", "Frame0SceneVisibility Oracle"), "selector_inputs": "frame-0 scene visibility target", "training_target": "H36M17 scene LOS at frame 0", "architecture": "direct visibility score", "loss": "none", "seed": None, "candidate_protocol": "strict frame-0 candidate-only", "strict_frame0_causal": True, "exact_implementation_recoverable": True, "rerun": True, "reason": "existing exact target cache"},
        {"method": "V2/V3/V4 Visibility predictors", "historical": metric(old_root / "frame0_visibility_predictor_v1/result.json", "RGBGlobal+Geometry"), "selector_inputs": "candidate geometry plus current DINO (global/spatial)", "training_target": "frame-0 scene visibility", "architecture": "128-unit MLP; V4 64-D spatial projector", "loss": "SmoothL1 + listwise", "seed": 42, "candidate_protocol": "strict frame-0 candidate-only", "strict_frame0_causal": True, "exact_implementation_recoverable": True, "rerun": True, "reason": "exact checkpoints and current-frame cache available"},
        {"method": "U1-U4 Task utility", "historical": metric(old_root / "frame0_task_utility_predictor_v1/result.json", "RGBGlobal-TrueLogP+VisibilityAux"), "selector_inputs": "current DINO + candidate geometry", "training_target": "GT-TrueLogP (U1/U2/U4) or GT-Margin (U3); U4 visibility auxiliary", "architecture": "TaskUtilityPredictor 128-unit MLP with explicit stay head", "loss": "SmoothL1 + 0.5 listwise (+ fixed visibility auxiliary for U4)", "seed": 42, "candidate_protocol": "strict frame-0 candidate-only", "strict_frame0_causal": True, "exact_implementation_recoverable": True, "rerun": True, "reason": "targets rebuilt for Yaw8Fair"},
        {"method": "P1-P3 Prior residual", "historical": metric(old_root / "prior_residual_frame0_nbv/result.json", "Prior+RGBResidual λ=1.0"), "selector_inputs": "Train prior + current DINO/geometry", "training_target": "GT-Margin minus Train prior", "architecture": "TaskUtilityPredictor residual MLP", "loss": "SmoothL1 + 0.5 listwise", "seed": 42, "candidate_protocol": "strict frame-0 candidate-only", "strict_frame0_causal": True, "exact_implementation_recoverable": True, "rerun": True, "reason": "utility landscape changed under Yaw8Fair"},
    ]
    excluded = [
        "full O0/full temporal visibility", "short-prefix or full-sequence fusion", "B2-B4/OrderedConcat/streaming", "VLM/map/structured observability/GT-action/pair/world-model", "old adaptive or balanced adaptive head", "full ST-GCN fine-tuning",
    ]
    return {"entries": entries, "excluded_routes": excluded, "historical_results_are_reference_only": True}


def selection_distribution(
    rows: Sequence[Mapping[str, Any]],
    actions_by_method: Mapping[str, Sequence[int]],
    metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    static_actions: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, actions in actions_by_method.items():
        views = np.asarray(actions, dtype=np.int64)
        counts = Counter(str(v) for v in views.tolist())
        radii: Counter[str] = Counter()
        yaw_bins: Counter[str] = Counter()
        for row, view in zip(rows, views.tolist()):
            meta = metadata[(str(row["scene_id"]), str(row["region"]))]
            radii[str(meta["radii"][int(view)])] += 1
            yaw_bins[str(body_azimuth_bin(meta["azimuths"][int(view)], meta["yaw_deg"]))] += 1
        probs = np.asarray(list(counts.values()), dtype=np.float64)
        probs /= max(probs.sum(), 1.0)
        result[name] = {
            "viewpoint_histogram": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
            "radius_histogram": dict(sorted(radii.items(), key=lambda item: float(item[0]))),
            "relative_yaw_bin_histogram": dict(sorted(yaw_bins.items(), key=lambda item: int(item[0]))),
            "selection_entropy": float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None)))),
            "top1_agreement_with_static": float(np.mean(views == np.asarray(static_actions, dtype=np.int64))),
        }
    return result


def switch_analysis(
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
        "switch_count": int(switched.sum()), "switch_rate": float(switched.mean()),
        "switch_contexts": [str(rows[i]["episode_id"]) for i in np.flatnonzero(switched)[:100]],
        "static_selected_accuracy_on_switch": float(np.mean(static_pred[switched] == labels[switched])) if switched.any() else 0.0,
        "learned_selected_accuracy_on_switch": float(np.mean(learned_pred[switched] == labels[switched])) if switched.any() else 0.0,
        "static_wrong_learned_correct": int(np.sum((static_pred != labels) & (learned_pred == labels) & switched)),
        "static_correct_learned_wrong": int(np.sum((static_pred == labels) & (learned_pred != labels) & switched)),
        "both_correct": int(np.sum((static_pred == labels) & (learned_pred == labels))),
        "both_wrong": int(np.sum((static_pred != labels) & (learned_pred != labels))),
    }


def render_analysis(result: Mapping[str, Any]) -> str:
    methods = result["main_metrics"]
    static = methods["StaticViewPrior"]
    learned_names = ["GeometryOnly-TrueLogP", "RGBGlobal-TrueLogP", "RGBGlobal-Margin", "RGBGlobal-TrueLogP+VisibilityAux", "Prior+GeometryResidual", "Prior+RGBResidual λ=0.5", "Prior+RGBResidual λ=1.0"]
    strict_names = ["Frame0SceneVisibility", "GeometryOnly-Visibility", "RGBGlobal-Visibility", "RGBSpatial-Visibility", *learned_names]
    best_name = str(result.get("best_strict_method", max(strict_names, key=lambda name: float(methods[name]["accuracy"]))))
    best_learned_name = str(result.get("best_learned_method", max(learned_names, key=lambda name: float(methods[name]["accuracy"]))))
    best = methods[best_name]
    best_learned = methods[best_learned_name]
    gain = 100.0 * (float(best["accuracy"]) - float(static["accuracy"]))
    learned_gain = 100.0 * (float(best_learned["accuracy"]) - float(static["accuracy"]))
    high_gain = float(result["occlusion_metrics"]["best_learned_gain_over_static_pp"])
    if gain >= 3.0 and float(best["accuracy"]) >= 0.58 and high_gain >= 2.0:
        decision = "STRONG KEEP"
    elif gain >= 1.5 and float(best["macro_f1"]) >= float(static["macro_f1"]):
        decision = "KEEP INSTANCE-CONDITIONED FRAME0 NBV"
    elif gain >= 0.5:
        decision = "WEAK KEEP"
    else:
        decision = "KILL FURTHER FRAME0 SELECTOR DEVELOPMENT"
    lines = [
        "# Yaw8Fair Strict-Frame0 NBV Full Re-baseline", "",
        "Formal task: Frame0 → exactly one Stage-A legal candidate → selected real O1 alone → Yaw8Fair HAR.",
        "Formal recognizer: frozen Yaw8 ST-GCN encoder + frozen matched Policy-balanced shared head.",
        "Policy Test: locked; Moving Val is evaluation-only; no new perception data were generated.", "",
        "## Main Moving-Val results", "",
        "| Method | Accuracy | Macro-F1 | Gain vs Random | Gain vs StaticPrior |", "|---|---:|---:|---:|---:|",
    ]
    random_acc = float(methods["Random legal"]["accuracy"])
    static_acc = float(static["accuracy"])
    for name, metric in methods.items():
        if isinstance(metric, dict) and "accuracy" in metric:
            lines.append(f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | {(100*(metric['accuracy']-random_acc)):+.3f}pp | {(100*(metric['accuracy']-static_acc)):+.3f}pp |")
    lines.extend([
        "", "## Re-baseline decision", "",
        f"Best strict method: **{best_name}**, Accuracy/F1={best['accuracy']:.6f}/{best['macro_f1']:.6f}; gain over StaticPrior={gain:+.3f}pp.",
        f"Best learned branch: **{best_learned_name}**, Accuracy/F1={best_learned['accuracy']:.6f}/{best_learned['macro_f1']:.6f}; gain over StaticPrior={learned_gain:+.3f}pp.",
        f"Pre-registered decision: **{decision}**.",
        f"Candidate-only oracle sanity: GT-TrueLogP={methods['GT-TrueLogP Oracle']['accuracy']:.6f}, GT-Margin={methods['GT-Margin Oracle']['accuracy']:.6f}, AnyCorrect={result['oracle_metrics']['candidate_anycorrect_rate']:.6f}.",
        "", "## Causal and training protocol", "",
        f"Policy Train contexts={result['population']['policy_train_contexts']}; Moving Val contexts={result['population']['moving_val_contexts']}; mean legal candidates={result['population']['legal_candidate_count_mean']:.3f}.",
        "All selector branches use a deterministic 10% Policy-Train record holdout for checkpoint selection; Moving Val was not used for training or selection.",
        "Candidate observations/features/logits and GT labels are never selector inputs; they are target/oracle/terminal-only.",
        "", "## RGB / geometry shuffle and occlusion", "",
        f"Best learned RGB shuffle drop={result['shuffle_metrics'].get('best_rgb_drop_pp', 0.0):+.3f}pp; geometry shuffle drop={result['shuffle_metrics'].get('best_geometry_drop_pp', 0.0):+.3f}pp.",
        f"High-occlusion subset size={result['occlusion_metrics']['count']}; best learned gain over StaticPrior={high_gain:+.3f}pp.",
        "", "## Scientific conclusion", "",
        "Historical old-recognizer values are reference-only; the formal comparison is entirely within Yaw8Fair.",
        f"Conclusion: **{decision}**. No automatic follow-up method was started.", "", "```text",
        "policy_test_used=false", "moving_val_used_for_training_or_selection=false", "candidate_observation_used_for_selector=false",
        "gt_action_used_for_selector=false", "new_rgb_generated=false", "new_skeleton_generated=false", "yaw8_encoder_modified=false", "yaw8_shared_head_modified=false", "```", "",
    ])
    return "\n".join(lines)
