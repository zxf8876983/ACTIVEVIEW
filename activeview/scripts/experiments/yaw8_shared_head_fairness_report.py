"""Markdown report renderer for the matched Old/Yaw8 shared-head audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def write_analysis(result: Mapping[str, Any], path: Path) -> None:
    old_hist = result["old_historical_reproduction"]
    old = result["oldfair_metrics"]
    yaw = result["yaw8fair_metrics"]
    native = result["yaw8_native_comparison"]
    gains = result["gain_decomposition"]
    rel = result["relative_yaw_metrics"]
    lines = [
        "# Yaw8 Encoder + Policy-Balanced Shared Head Fairness Audit", "",
        "Purpose: remove the classifier-head training-distribution mismatch from the previous Old-vs-Yaw8 policy landscape comparison.", "",
        "Protocol: both ST-GCN encoders are frozen; each head is Linear(256,256) → GELU → Linear(256,12), trained on identical Policy-Train record-balanced samples (16 observations per record per epoch). A deterministic 10% Policy-Train record holdout selects checkpoints. Moving Val is evaluation-only and Policy Test was not read.", "",
        "## Historical Old shared-head reproduction", "",
        f"Stay={old_hist['Stay/current']['accuracy']:.6f}, legal micro={old_hist['Legal micro']['accuracy']:.6f}, Random={old_hist['Random legal']['accuracy']:.6f}; the historical reproduction gate is within ±0.2pp.", "",
        "## Main matched-head Moving-Val results", "",
        "| Method | OldFair Acc/F1 | Yaw8Fair Acc/F1 | Yaw8Native Acc/F1 |", "|---|---:|---:|---:|",
    ]
    for method in ("Stay/current", "Random legal", "StaticPrior", "GT-TrueLogP Oracle", "GT-Margin Oracle"):
        old_v = old[method]
        yaw_v = yaw[method]
        native_v = native.get(method, {})
        lines.append(f"| {method} | {old_v['accuracy']:.6f}/{old_v['macro_f1']:.6f} | {yaw_v['accuracy']:.6f}/{yaw_v['macro_f1']:.6f} | {native_v.get('accuracy', 0.0):.6f}/{native_v.get('macro_f1', 0.0):.6f} |")
    lines.extend([
        f"| Legal candidate micro | {result['legal_micro_metrics']['old']['accuracy']:.6f}/{result['legal_micro_metrics']['old']['macro_f1']:.6f} | {result['legal_micro_metrics']['yaw8']['accuracy']:.6f}/{result['legal_micro_metrics']['yaw8']['macro_f1']:.6f} | {native['Legal micro']['accuracy']:.6f}/{native['Legal micro']['macro_f1']:.6f} |", "",
        f"Candidate-only oracle (primary legal action scope): OldFair GT-TrueLogP={result['candidate_only_oracle_metrics']['oldfair']['GT-TrueLogP']['accuracy']:.6f}, GT-Margin={result['candidate_only_oracle_metrics']['oldfair']['GT-Margin']['accuracy']:.6f}, AnyCorrect={result['candidate_only_oracle_metrics']['oldfair']['AnyCorrect Coverage']['coverage_rate']:.6f}; Yaw8Fair GT-TrueLogP={result['candidate_only_oracle_metrics']['yaw8fair']['GT-TrueLogP']['accuracy']:.6f}, GT-Margin={result['candidate_only_oracle_metrics']['yaw8fair']['GT-Margin']['accuracy']:.6f}, AnyCorrect={result['candidate_only_oracle_metrics']['yaw8fair']['AnyCorrect Coverage']['coverage_rate']:.6f}.", "",
        "## Fair encoder and policy-head gains", "",
        f"Yaw8Fair − OldFair: Stay {gains['stay_accuracy_pp']:+.3f}pp, legal micro {gains['legal_accuracy_pp']:+.3f}pp, Random {gains['random_accuracy_pp']:+.3f}pp, StaticPrior {gains['static_accuracy_pp']:+.3f}pp.",
        f"Yaw8 shared head − Yaw8 native FC: legal micro {gains['yaw8_policy_head_gain_legal_pp']:+.3f}pp, Random {gains['yaw8_policy_head_gain_random_pp']:+.3f}pp.", "",
        "## Relative-yaw audit", "",
        f"OldFair best-worst candidate accuracy gap={rel['old_relative_yaw_gap'] * 100:.3f}pp; Yaw8Fair gap={rel['yaw8_relative_yaw_gap'] * 100:.3f}pp. These policy yaw bins are confounded by scene geometry/occlusion and are not a pure yaw-invariance test.", "",
        "## Per-class observations", "",
        "Matched shared-head Random changes are reported in `per_class_metrics.json`; the key native-FC declines should not be attributed to the encoder until this matched comparison is considered.", "",
        "## Historical controlled clean-yaw evidence (not rerun)", "",
        "Old Yaw8 Val = 58.3163/54.5710; Yaw8 model = 72.5510/72.7956; MeanYawGain = +14.2347pp; WorstYawGain = +20.4082pp. This controlled clean-yaw result is separate from scene-conditioned policy relative-angle behavior.", "",
        "## Decisions", "",
    ])
    legal_gain = gains["legal_accuracy_pp"]
    random_gain = gains["random_accuracy_pp"]
    critical_drop = gains["max_key_class_random_decline_pp"]
    if legal_gain >= 4.0 and random_gain >= 4.0 and gains["high_occlusion_random_gain_pp"] >= 0:
        promotion = "STRONG PROMOTE YAW8 ENCODER"
    elif legal_gain >= 2.0 and random_gain >= 2.0 and critical_drop <= 10.0:
        promotion = "PROMOTE YAW8 ENCODER"
    elif legal_gain >= 1.0 and random_gain >= 1.0 and critical_drop <= 10.0:
        promotion = "BORDERLINE — KEEP AS ALTERNATIVE"
    else:
        promotion = "DO NOT PROMOTE YAW8 ENCODER"
    yaw_headroom = result["headroom_metrics"]["yaw8fair"]
    largest_gap = max(float(yaw_headroom["static_to_gt_true_logp"]), float(yaw_headroom["anycorrect_minus_static"]))
    viability = "SUBSTANTIAL NBV HEADROOM REMAINS" if largest_gap >= 0.12 else "MODERATE NBV HEADROOM" if largest_gap >= 0.08 else "NBV HEADROOM MOSTLY COLLAPSED"
    lines.extend([
        f"**Promotion: {promotion}.** This decision uses only matched-head legal/Random gains, Macro-F1 behavior, and the largest key-class Random decline ({critical_drop:.3f}pp).",
        f"**NBV viability: {viability}.** Yaw8Fair StaticPrior to GT-TrueLogP gap={yaw_headroom['static_to_gt_true_logp'] * 100:.3f}pp; AnyCorrect to StaticPrior gap={yaw_headroom['anycorrect_minus_static'] * 100:.3f}pp.", "",
        "A. Under fair Policy-balanced head adaptation, Yaw8 is compared to Old only through the frozen encoder representation; the head architecture, sampler, optimizer, seed and evaluation action set are identical.", "",
        "B. If a future run promotes Yaw8, freeze the encoder and matched head, rebuild Policy-Train utility targets, then retrain NBV. No selector was trained in this audit.", "",
        "```text", "policy_test_used=false", "moving_val_used_for_head_training_or_selection=false", "new_rgb_generated=false", "new_skeleton_generated=false", "yolo_rerun=false", "videopose3d_rerun=false", "old_encoder_modified=false", "yaw8_encoder_modified=false", "```", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
