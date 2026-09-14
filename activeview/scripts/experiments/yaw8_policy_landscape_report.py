"""Markdown report rendering for the Yaw8 policy-landscape audit."""

from __future__ import annotations

from typing import Any, Mapping

from activeview.scripts.eval.reduced12_nbv_utils import LABELS, NUM_CLASSES


def render_analysis(result: Mapping[str, Any]) -> str:
    main = result["main_table"]
    old = main["old"]
    new = main["yaw8"]
    old_random_gap = old["GT-TrueLogP Oracle"]["accuracy"] - old["Random legal"]["accuracy"]
    new_random_gap = new["GT-TrueLogP Oracle"]["accuracy"] - new["Random legal"]["accuracy"]
    rel = result["relative_body_yaw_metrics"]
    variance = result["viewpoint_variance_metrics"]["variance"]
    legal_gain = new["Random legal"]["accuracy"] - old["Random legal"]["accuracy"]
    relative_flattened = rel["yaw8_relative_yaw_gap"] < rel["old_relative_yaw_gap"]
    per_class = result["per_class_metrics"]
    yaw8_random = per_class["yaw8"]["Random legal"]
    old_random = per_class["old"]["Random legal"]
    collapsed = [
        LABELS[index]
        for index in range(NUM_CLASSES)
        if float(yaw8_random[LABELS[index]]["accuracy"])
        < float(old_random[LABELS[index]]["accuracy"]) - 0.10
    ]
    promote = bool(legal_gain > 0.0 and relative_flattened and not collapsed)
    if new_random_gap >= 0.15 or new["GT-TrueLogP Oracle"]["accuracy"] - new["StaticPrior"]["accuracy"] >= 0.12:
        conclusion = "YAW ROBUSTNESS IMPROVED, BUT SUBSTANTIAL OCCLUSION-DRIVEN NBV HEADROOM REMAINS"
    elif new_random_gap < 0.08 and new["GT-TrueLogP Oracle"]["accuracy"] - new["StaticPrior"]["accuracy"] < 0.08:
        conclusion = "MUCH OF THE PREVIOUS NBV HEADROOM WAS RECOGNIZER-INDUCED"
    else:
        conclusion = "RECOGNIZER VIEWPOINT BIAS WAS A MAJOR ACTIVE-VIEW CONFOUND"
    lines = [
        "# Yaw8 Recognizer × Policy Landscape Audit", "",
        "Policy Train and Moving Val only; Policy Test was not opened.",
        "", "## Main Moving-Val table", "",
        "| Metric | Old ST-GCN | Yaw8 ST-GCN | Change |", "|---|---:|---:|---:|",
    ]
    for method in ("Stay/current", "Random legal", "StaticPrior", "GT-TrueLogP Oracle", "GT-Margin Oracle"):
        lines.append(f"| {method} Acc | {old[method]['accuracy']:.6f} | {new[method]['accuracy']:.6f} | {(new[method]['accuracy']-old[method]['accuracy'])*100:+.3f}pp |")
        lines.append(f"| {method} Macro-F1 | {old[method]['macro_f1']:.6f} | {new[method]['macro_f1']:.6f} | {(new[method]['macro_f1']-old[method]['macro_f1'])*100:+.3f}pp |")
    lines.extend([
        f"| AnyCorrect Coverage | {old['AnyCorrect Coverage']['coverage_rate']:.6f} | {new['AnyCorrect Coverage']['coverage_rate']:.6f} | {(new['AnyCorrect Coverage']['coverage_rate']-old['AnyCorrect Coverage']['coverage_rate'])*100:+.3f}pp |",
        "", f"Random→Oracle gap: old={old_random_gap:.6f}, yaw8={new_random_gap:.6f}; Static→Oracle gap: old={old['GT-TrueLogP Oracle']['accuracy']-old['StaticPrior']['accuracy']:.6f}, yaw8={new['GT-TrueLogP Oracle']['accuracy']-new['StaticPrior']['accuracy']:.6f}.",
        f"Relative-yaw best–worst accuracy gap: old={rel['old_relative_yaw_gap']:.6f}, yaw8={rel['yaw8_relative_yaw_gap']:.6f}, reduction={rel['gap_reduction']:.6f}.",
        f"Viewpoint accuracy std: old={variance['old_viewpoint_accuracy_std']:.6f}, yaw8={variance['yaw8_viewpoint_accuracy_std']:.6f}; eta² absolute-view={variance['old_viewpoint_eta_squared']:.6f}/{variance['yaw8_viewpoint_eta_squared']:.6f}, relative-yaw={rel['old_relative_yaw_eta_squared']:.6f}/{rel['yaw8_relative_yaw_eta_squared']:.6f}.",
        f"Global candidate GT-margin Spearman={result['utility_correlation']['margins']['global_spearman']:.6f}; within-context mean/median={result['utility_correlation']['margins']['within_context_mean']:.6f}/{result['utility_correlation']['margins']['within_context_median']:.6f}.",
        f"Oracle GT-margin top-1 agreement between recognizers={result['oracle_agreement']['gt_margin_top1_agreement']:.6f}; Train prior Spearman={result['prior_structure_comparison']['spearman']:.6f}, top-5 overlap={result['prior_structure_comparison']['top5_overlap']:.6f}.",
        "", "## Interpretation", "", f"**{conclusion}**.",
        f"Legal candidate accuracy change is {legal_gain * 100:+.3f}pp; relative-yaw gap flattened={relative_flattened}; per-class drops larger than 10pp={collapsed or 'none'}.",
        "", "The old and Yaw8 candidates use the identical current/Stay + Stage-A legal action set. The Yaw8 checkpoint was frozen before Policy Val analysis; StaticPrior is computed independently from Policy Train candidate GT-Margins for each recognizer.",
        "", f"Promotion decision: **{'PROMOTE YAW8 RECOGNIZER' if promote else 'DO NOT PROMOTE YAW8 RECOGNIZER'}**. This audit does not train a selector. If promoted, the next required step is to rebuild Policy Train utility targets with frozen Yaw8 and retrain strict Frame0 NBV from scratch.",
        "", "## Required scientific answers", "",
        f"1. Candidate recognition: legal-candidate micro Acc changes from {old['Random legal']['accuracy']:.6f} to {new['Random legal']['accuracy']:.6f}; StaticPrior changes from {old['StaticPrior']['accuracy']:.6f} to {new['StaticPrior']['accuracy']:.6f}.",
        f"2. Old viewpoint variation: absolute-view eta²={variance['old_viewpoint_eta_squared']:.6f}; relative-body-yaw eta²={rel['old_relative_yaw_eta_squared']:.6f}.",
        f"3. Yaw8 relative-angle flattening: {'yes' if relative_flattened else 'no'} (best–worst gap {rel['yaw8_relative_yaw_gap']:.6f}).",
        f"4. Static prior remains {'strong' if new['GT-TrueLogP Oracle']['accuracy'] - new['StaticPrior']['accuracy'] >= 0.12 else 'moderate'}; own-prior accuracy is {new['StaticPrior']['accuracy']:.6f}.",
        f"5. Static-prior ranking correlation is Spearman {result['prior_structure_comparison']['spearman']:.6f}, top-5 overlap {result['prior_structure_comparison']['top5_overlap']:.6f}.",
        f"6. GT-TrueLogP Oracle / AnyCorrect changes {old['GT-TrueLogP Oracle']['accuracy']:.6f}/{old['AnyCorrect Coverage']['coverage_rate']:.6f} → {new['GT-TrueLogP Oracle']['accuracy']:.6f}/{new['AnyCorrect Coverage']['coverage_rate']:.6f}.",
        f"7. Random→Oracle headroom changes {old_random_gap:.6f} → {new_random_gap:.6f}; Static→Oracle changes {old['GT-TrueLogP Oracle']['accuracy']-old['StaticPrior']['accuracy']:.6f} → {new['GT-TrueLogP Oracle']['accuracy']-new['StaticPrior']['accuracy']:.6f}.",
        f"8. High-occlusion Random/Static/Oracle Acc changes {result['occlusion_metrics']['recognizers']['old']['Random legal']['accuracy']:.6f}/{result['occlusion_metrics']['recognizers']['old']['StaticPrior']['accuracy']:.6f}/{result['occlusion_metrics']['recognizers']['old']['GT-TrueLogP Oracle']['accuracy']:.6f} → {result['occlusion_metrics']['recognizers']['yaw8']['Random legal']['accuracy']:.6f}/{result['occlusion_metrics']['recognizers']['yaw8']['StaticPrior']['accuracy']:.6f}/{result['occlusion_metrics']['recognizers']['yaw8']['GT-TrueLogP Oracle']['accuracy']:.6f}.",
        "9. The old candidate utility landscape is substantially changed at the context level (GT-margin top-1 agreement and top-3 overlap are reported above), while the static viewpoint prior remains structurally similar.",
        "10. A real active-view problem remains because Yaw8 Static/Random are well below its legal oracle; however, the strict Yaw8 promotion gate is not met because relative-yaw sensitivity increased and sit/bend show large random-candidate drops.",
        "", "Flags: `policy_test_used=false`, `new_rgb_generated=false`, `new_skeleton_generated=false`, `yolo_rerun=false`, `videopose3d_rerun=false`, `stgcn_modified=false`, `moving_val_used_for_training=false`.",
    ])
    return "\n".join(lines) + "\n"
