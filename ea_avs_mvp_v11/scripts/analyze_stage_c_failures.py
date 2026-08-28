#!/usr/bin/env python3
"""Generate read-only Stage C-v0 failure-analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.active_view.stage_c_failure_analysis import analyze_rows, load_jsonl, prepare_aligned_rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _write_episode_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["episode_id", "record_id", "action_label", "scene_id", "region", "regret", "regret_group", "failure_type", "current_entropy", "current_margin", "current_pose_confidence", "predicted_stays", "safe_oracle_stays", "selected_true_utility", "safe_oracle_utility", "candidate_count", "selected_geodesic", "oracle_geodesic", "selected_azimuth", "oracle_azimuth", "candidate_viewpoint_ids", "predicted_candidate_viewpoint_id", "candidate_oracle_viewpoint_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            selected = row.get("selected_geometry") or {}
            oracle = row.get("oracle_geometry") or {}
            writer.writerow({
                "episode_id": row["episode_id"], "record_id": row["record_id"], "action_label": row["action_label"], "scene_id": row["scene_id"], "region": row["region"], "regret": row["regret"], "regret_group": row["regret_group"], "failure_type": row["failure_type"], "current_entropy": row["current_entropy"], "current_margin": row["current_margin"], "current_pose_confidence": row["current_pose_confidence"], "predicted_stays": row["predicted_stays"], "safe_oracle_stays": row["safe_oracle_stays"], "selected_true_utility": row["selected_true_utility"], "safe_oracle_utility": row["safe_oracle_utility"], "candidate_count": len(row["candidate_viewpoint_ids"]), "selected_geodesic": selected.get("geodesic", ""), "oracle_geodesic": oracle.get("geodesic", ""), "selected_azimuth": selected.get("signed_azimuth_deg", ""), "oracle_azimuth": oracle.get("signed_azimuth_deg", ""), "candidate_viewpoint_ids": json.dumps(row["candidate_viewpoint_ids"]), "predicted_candidate_viewpoint_id": row["predicted_candidate_viewpoint_id"], "candidate_oracle_viewpoint_id": row["candidate_oracle_viewpoint_id"],
            })


def _write_record_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["record_id"]), []).append(row)
    fields = ["record_id", "action_label", "episode_count", "mean_regret", "p90_regret", "Set_accuracy", "NoMove_accuracy", "SafeOracle_accuracy", "headroom_capture", "catastrophic_ratio"]
    records = []
    for record_id, members in grouped.items():
        catastrophic = [row for row in members if row.get("catastrophic_top5pct")]
        safe_positive = [row for row in members if float(row["safe_oracle_utility"]) > 1e-6]
        safe_sum = sum(float(row["safe_oracle_utility"]) for row in safe_positive)
        selected_sum = sum(max(0.0, float(row["selected_true_utility"])) for row in safe_positive)
        records.append({"record_id": record_id, "action_label": members[0]["action_label"], "episode_count": len(members), "mean_regret": float(np.mean([float(row["regret"]) for row in members])), "p90_regret": float(np.percentile([float(row["regret"]) for row in members], 90)), "Set_accuracy": float(np.mean([int(row["selected_predicted_label_id"]) == int(row["label_id"]) for row in members])), "NoMove_accuracy": float(np.mean([int(row["current_predicted_label_id"]) == int(row["label_id"]) for row in members])), "SafeOracle_accuracy": float(np.mean([int(row["safe_oracle_predicted_label_id"]) == int(row["label_id"]) for row in members])), "headroom_capture": selected_sum / safe_sum if safe_sum else 0.0, "catastrophic_ratio": len(catastrophic) / len(members)})
    records.sort(key=lambda value: (-value["mean_regret"], value["record_id"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)


def _summary_reference(eval_summary: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = eval_summary["metrics"]["test"]
    recognition = metrics["recognition"]
    return {"model_type": eval_summary["model_type"], "episode_count": metrics["episode_count"], "recognition": {name: {key: recognition[name][key] for key in ("accuracy", "macro_f1")} for name in ("NoMove", "StageC", "CandidateOracle", "SafeOracle")}, "candidate_oracle_hit_rate": metrics["candidate_oracle_hit_rate"], "safe_action_match_rate": metrics["safe_action_match_rate"], "decision_regret": metrics["decision_regret"], "positive_headroom_capture": metrics["positive_headroom_capture"]}


def _write_report(path: Path, summary: Mapping[str, Any], references: Sequence[Mapping[str, Any]], paths: Mapping[str, str]) -> None:
    thresholds = summary["regret"]["thresholds"]
    taxonomy = summary["failure_taxonomy"]
    action = summary["action_class"]
    hardest = sorted(action.items(), key=lambda item: (item[1]["Set_accuracy"], item[0]))[:5]
    highest_regret = sorted(action.items(), key=lambda item: (-item[1]["regret"]["mean"], item[0]))[:5]
    correlations = summary["current_state"]["correlations_with_regret"]
    strongest = sorted(((name, value["spearman"]) for name, value in correlations.items()), key=lambda item: -abs(item[1]))
    high_actions = ", ".join(f"{name} ({count})" for name, count in sorted(summary["high_regret_action_counts"].items(), key=lambda item: -item[1])[:5]) or "none"
    high_regions = ", ".join(f"{name} ({count})" for name, count in sorted(summary["high_regret_region_counts"].items(), key=lambda item: -item[1])) or "none"
    lines = ["# Stage C-v0 Failure Analysis (Set Ranker Test)", "", "## Scope and frozen-artifact checks", "", f"- Test episodes: **{summary['episode_count']}**; independent motion records: **{summary['record_count']}**.", "- Analysis uses frozen Stage A/B/C JSONL and Set Ranker predictions only; no Habitat, YOLO, VideoPose3D, retraining or upstream regeneration was run.", "- The Stage C validator report was checked before analysis and must be `passed=true`, `error_count=0`.", "- Episode-level rows are repeated observations, not IID samples; record-level aggregation is reported separately.", "", "## Regret distribution and groups", "", f"- Thresholds derived from Test regret: p50={thresholds['median']:.6f}, p75={thresholds['p75']:.6f}, p90={thresholds['p90']:.6f}, p95={thresholds['p95']:.6f}, p99={thresholds['p99']:.6f}.", f"- G0 (≤1e-3): {summary['regret']['G0_near_optimal']['count']} ({summary['regret']['G0_near_optimal']['ratio']:.2%}); G1 (1e-3–p75): {summary['regret']['G1_low_regret']['count']} ({summary['regret']['G1_low_regret']['ratio']:.2%}); G2 (p75–p90]: {summary['regret']['G2_moderate_regret']['count']} ({summary['regret']['G2_moderate_regret']['ratio']:.2%}); G3 (>p90): {summary['regret']['G3_high_regret']['count']} ({summary['regret']['G3_high_regret']['ratio']:.2%}).", f"- Catastrophic top 5%: {summary['regret']['G4_catastrophic_top5pct']['count']} ({summary['regret']['G4_catastrophic_top5pct']['ratio']:.2%}); extreme top 1%: {summary['regret']['G4_extreme_top1pct']['count']} ({summary['regret']['G4_extreme_top1pct']['ratio']:.2%}).", "", "## Decision-failure taxonomy", "", "| Type | Meaning | Count | Ratio | Mean regret | P90 regret |", "|---|---|---:|---:|---:|---:|"]
    labels = {"A_missed_move": "Move required, model stayed", "B_unnecessary_move": "Stay required, model moved", "C1_wrong_near_optimal": "Move/move wrong candidate, near-equivalent", "C2_wrong_high_utility_loss": "Move/move wrong candidate, high loss", "D_correct_safe_action": "SafeOracle action matched"}
    for key in labels:
        item = taxonomy[key]
        lines.append(f"| {key} | {labels[key]} | {item['count']} | {item['ratio']:.2%} | {item['regret']['mean']:.4f} | {item['regret']['p90']:.4f} |")
    lines += ["", "The dominant failure type is **" + max(labels, key=lambda key: taxonomy[key]["count"]) + "**. High-regret episodes are concentrated in: " + high_actions + ". Region counts among G3 are: " + high_regions + ".", "", "## Candidate miss versus utility quality", "", f"- Candidate exact miss rate: {summary['candidate_miss']['miss_ratio']:.2%} ({summary['candidate_miss']['miss_count']} episodes).", f"- For misses, CandidateOracle utility minus selected utility: mean {summary['candidate_miss']['absolute_utility_gap']['mean']:.4f}, median {summary['candidate_miss']['absolute_utility_gap']['median']:.4f}, p90 {summary['candidate_miss']['absolute_utility_gap']['p90']:.4f}.", f"- Among misses with Oracle utility >1e-6, selected utility reaches ≥90%/75%/50% of Oracle in {summary['candidate_miss']['ratio_thresholds']['selected_at_least_90pct_oracle']:.2%}/{summary['candidate_miss']['ratio_thresholds']['selected_at_least_75pct_oracle']:.2%}/{summary['candidate_miss']['ratio_thresholds']['selected_at_least_50pct_oracle']:.2%}.", "This directly tests whether low exact hit is mainly near-equivalent selection or materially bad utility loss.", "", "## Action-class and region findings", "", "Lowest Set Accuracy classes: " + ", ".join(f"{name} ({value['Set_accuracy']:.2%})" for name, value in hardest) + ".", "Highest mean-regret classes: " + ", ".join(f"{name} ({value['regret']['mean']:.3f})" for name, value in highest_regret) + ".", "Region breakdown is in the machine-readable summary; it should not be treated as a scene-held-out result.", "", "## Current state and geometry diagnostics", "", "- Spearman correlations with regret (absolute ordering): " + ", ".join(f"{name}={value:+.3f}" for name, value in strongest) + ".", "- State-group summaries compare entropy, margin, pose confidence, current correctness, move rate and SafeOracle headroom in the JSON artifact.", "- Geometry analysis includes absolute azimuth bins, geodesic/radius distributions, selected-versus-Oracle geometry and candidate-set utility-gap bins.", "- Symmetric-geometry analysis uses explicit tolerances and a data-derived q90 utility-difference threshold; it is indirect evidence only and does not estimate body yaw.", "", "## Record-level concentration", "", f"- Top 10% records account for {summary['record_level']['catastrophic_episode_share_in_top10pct_records']:.2%} of catastrophic top-5% episodes.", "- Worst and best 20 motion records are included in `record_failure_table.csv` and the JSON summary.", "", "## Scientific answers", "", f"1. High regret is {'concentrated' if summary['record_level']['catastrophic_episode_share_in_top10pct_records'] > 0.5 else 'not strongly concentrated'} in a small record subset under the top-10% diagnostic.", f"2. The dominant decision failure is {max(labels, key=lambda key: taxonomy[key]['count'])}; wrong-candidate severity is separated into C1/C2.", f"3. Exact candidate hit versus aggregate headroom is explained by the miss utility-gap and ratio statistics above.", f"4. The hardest action classes by Set accuracy are {', '.join(name for name, _ in hardest)}.", f"5. The strongest state correlation by absolute Spearman magnitude is {strongest[0][0]} ({strongest[0][1]:+.3f}); this is descriptive, not causal.", "6. Geometry and azimuth results are descriptive binned evidence; no single geometry variable is assumed causal.", f"7. Symmetric-geometry ambiguity affects {summary['symmetric_geometry_ambiguity']['episode_count']} episodes under the stated tolerance, with {summary['symmetric_geometry_ambiguity']['large_difference_pair_count']} large-difference pairs; enrichment is descriptive.", "8. Evidence for adding perceived body orientation is **weak / inconclusive** unless symmetric ambiguity is clearly enriched among high-regret cases; no body yaw was added here.", "9. Current evidence most directly supports hard-example/long-tail analysis (E) and improved current-state/candidate-set representation (B/A) as hypotheses for later review. It does not justify changing Stage C-v0 in this task.", "", "## Artifacts", ""]
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "### Per-action detail", "", "| Action | n | NoMove Acc | Set Acc | SafeOracle Acc | Set gain | Mean regret | P90 regret | Headroom |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in sorted(action.items(), key=lambda item: item[1]["Set_accuracy"]):
        lines.append(f"| {name} | {value['n']} | {value['NoMove_accuracy']:.2%} | {value['Set_accuracy']:.2%} | {value['SafeOracle_accuracy']:.2%} | {value['Set_gain_vs_NoMove']:+.2%} | {value['regret']['mean']:.3f} | {value['regret']['p90']:.3f} | {value['positive_headroom']['aggregate_capture']:.2%} |")
    lines += ["", "### Semantic-region detail", "", "| Region | n | NoMove Acc | Set Acc | SafeOracle Acc | Set gain | Mean regret | P90 regret | Headroom | Set stay | Safe stay |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in sorted(summary["region"].items()):
        lines.append(f"| {name} | {value['n']} | {value['NoMove_accuracy']:.2%} | {value['Set_accuracy']:.2%} | {value['SafeOracle_accuracy']:.2%} | {value['Set_gain_vs_NoMove']:+.2%} | {value['regret']['mean']:.3f} | {value['regret']['p90']:.3f} | {value['positive_headroom']['aggregate_capture']:.2%} | {value['Set_stay_rate']:.2%} | {value['SafeOracle_stay_rate']:.2%} |")
    lines += ["", "### Current-state group detail", "", "| Group | n | Entropy mean | Margin mean | Pose confidence mean | Current correct | Move rate | Safe move rate | Safe utility mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("G0_near_optimal", "G1_low_regret", "G2_moderate_regret", "G3_high_regret"):
        value = summary["current_state"][name]
        lines.append(f"| {name} | {value['count']} | {value['current_entropy']['mean']:.3f} | {value['current_margin']['mean']:.3f} | {value['current_pose_confidence']['mean']:.3f} | {value['current_correct_ratio']:.2%} | {value['move_rate']:.2%} | {value['safe_oracle_move_rate']:.2%} | {value['safe_oracle_utility']['mean']:.3f} |")
    geometry = summary["geometry"]
    lines += ["", "### Geometry detail", "", f"- Oracle absolute azimuth mean/median: {geometry['oracle_selected_geometry']['oracle']['abs_azimuth_deg']['mean']:.2f}° / {geometry['oracle_selected_geometry']['oracle']['abs_azimuth_deg']['median']:.2f}°; model-selected move: {geometry['oracle_selected_geometry']['model_selected_move']['abs_azimuth_deg']['mean']:.2f}° / {geometry['oracle_selected_geometry']['model_selected_move']['abs_azimuth_deg']['median']:.2f}°.", f"- Oracle geodesic mean/median: {geometry['oracle_selected_geometry']['oracle']['geodesic']['mean']:.3f} / {geometry['oracle_selected_geometry']['oracle']['geodesic']['median']:.3f} m; model-selected move: {geometry['oracle_selected_geometry']['model_selected_move']['geodesic']['mean']:.3f} / {geometry['oracle_selected_geometry']['model_selected_move']['geodesic']['median']:.3f} m.", f"- Oracle radius direction counts (closer/same/farther): {geometry['radius_direction']['oracle']}; model-selected move: {geometry['radius_direction']['model_selected_move']}.", "", "| Oracle geodesic bin | n | Mean regret | Headroom capture |", "|---|---:|---:|---:|"]
    for name, value in geometry["geodesic_bins_by_oracle"].items():
        lines.append(f"| {name} | {value['count']} | {value['regret']['mean']:.3f} | {value['headroom']['aggregate_capture']:.2%} |")
    difficulty = summary["candidate_set_difficulty"]
    lines += ["", "### Candidate-set difficulty", "", "| Gap bin | n | Exact hit | Mean regret | P90 regret | Headroom |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("very_small", "small", "medium", "large"):
        value = difficulty[name]
        lines.append(f"| {name} | {value['count']} | {value['candidate_hit_rate']:.2%} | {value['regret']['mean']:.3f} | {value['regret']['p90']:.3f} | {value['headroom']['aggregate_capture']:.2%} |")
    lines += ["", "### Representative cases", ""]
    for name, value in summary["representative_cases"].items():
        lines.append(f"- `{name}`: `{value['episode_id']}`, action={value['action_label']}, region={value['region']}, regret={float(value['regret']):.4f}, predicted={value['predicted_action']}, SafeOracle={value['safe_oracle_action']}.")
    lines += ["", "## Auxiliary Pairwise reference", ""]
    for reference in references:
        recognition = reference["recognition"]
        lines.append(f"- {reference['model_type']}: Test StageC {recognition['StageC']['accuracy']:.2%} Accuracy / {recognition['StageC']['macro_f1']:.2%} Macro-F1; regret mean {reference['decision_regret']['mean']:.4f}; aggregate headroom {reference['positive_headroom_capture']['aggregate_positive_clipped_ratio']:.2%}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    regrets = np.asarray([float(row["regret"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(regrets, bins=60, color="#355C7D", alpha=0.85); axes[0].set_xlabel("Regret"); axes[0].set_ylabel("Episodes"); axes[0].set_title("Set Ranker regret")
    sorted_values = np.sort(regrets); axes[1].plot(sorted_values, np.arange(1, len(sorted_values) + 1) / len(sorted_values), color="#C06C84"); axes[1].set_xlabel("Regret"); axes[1].set_ylabel("ECDF"); axes[1].set_title("Regret ECDF"); axes[1].set_xlim(left=0)
    fig.tight_layout(); target = plot_dir / "regret_distribution.png"; fig.savefig(target, dpi=160); plt.close(fig); paths.append(str(target))
    names = list(summary["action_class"]); x = np.arange(len(names)); width = 0.25
    fig, ax = plt.subplots(figsize=(12, 4)); ax.bar(x - width, [summary["action_class"][n]["NoMove_accuracy"] for n in names], width, label="NoMove"); ax.bar(x, [summary["action_class"][n]["Set_accuracy"] for n in names], width, label="Set Ranker"); ax.bar(x + width, [summary["action_class"][n]["SafeOracle_accuracy"] for n in names], width, label="SafeOracle"); ax.set_xticks(x, names, rotation=60, ha="right"); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.legend(); ax.set_title("Per-action recognition"); fig.tight_layout(); target = plot_dir / "per_action_accuracy.png"; fig.savefig(target, dpi=160); plt.close(fig); paths.append(str(target))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter([float(row["current_entropy"]) for row in rows], regrets, s=2, alpha=0.18, color="#6C5B7B"); ax.set_xlabel("Current entropy"); ax.set_ylabel("Regret"); ax.set_title("Regret vs current entropy"); fig.tight_layout(); target = plot_dir / "regret_vs_entropy.png"; fig.savefig(target, dpi=160); plt.close(fig); paths.append(str(target))
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter([float(row["current_pose_confidence"]) for row in rows], regrets, s=2, alpha=0.18, color="#F67280"); ax.set_xlabel("Current pose confidence"); ax.set_ylabel("Regret"); ax.set_title("Regret vs pose confidence"); fig.tight_layout(); target = plot_dir / "regret_vs_pose_confidence.png"; fig.savefig(target, dpi=160); plt.close(fig); paths.append(str(target))
    taxonomy = summary["failure_taxonomy"]; labels = list(taxonomy); values = [taxonomy[key]["count"] for key in labels]
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(labels, values, color="#99B898"); ax.set_ylabel("Episodes"); ax.set_title("Decision failure taxonomy"); ax.tick_params(axis="x", rotation=45); fig.tight_layout(); target = plot_dir / "failure_taxonomy.png"; fig.savefig(target, dpi=160); plt.close(fig); paths.append(str(target))
    return paths


def run(*, dataset_root: Path, stage_b_root: Path, stage_c_root: Path, output_dir: Path, model_type: str = "set_ranker") -> Dict[str, Any]:
    import json
    validation = json.loads((stage_c_root / "validation_report.json").read_text(encoding="utf-8"))
    if validation.get("passed") is not True or validation.get("error_count") != 0:
        raise RuntimeError("Frozen Stage C validator is not passed; refusing failure analysis")
    combined = json.loads((stage_c_root / "stage_c_summary.json").read_text(encoding="utf-8"))
    eval_path = stage_c_root / "evaluations" / f"{model_type}_evaluation_summary.json"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    if evaluation.get("model_type") != model_type:
        raise RuntimeError(f"Unexpected model type in {eval_path}")
    stage_a = json.loads((dataset_root / "stage_a_summary.json").read_text(encoding="utf-8"))
    stage_a_rows = load_jsonl(stage_a["episode_files"]["test"])
    stage_b_rows = load_jsonl(stage_b_root / "utility_labels" / "test.jsonl")
    feature_summary = json.loads((stage_c_root / "stage_c_feature_summary.json").read_text(encoding="utf-8"))
    feature_rows = load_jsonl(feature_summary["feature_files"]["test"])
    prediction_path = Path(evaluation["prediction_files"]["test"])
    prediction_rows = load_jsonl(prediction_path)
    if len(prediction_rows) != 13774 or len({str(row["record_id"]) for row in prediction_rows}) != 194:
        raise RuntimeError(f"Frozen Test coverage mismatch: episodes={len(prediction_rows)}, records={len({str(row['record_id']) for row in prediction_rows})}")
    rows = prepare_aligned_rows(stage_a_rows, stage_b_rows, feature_rows, prediction_rows)
    summary = analyze_rows(rows, evaluation["categories"])
    summary["artifact_validation"] = {"stage_c_validator": validation, "stage_c_summary": str(stage_c_root / "stage_c_summary.json"), "set_ranker_evaluation": str(eval_path), "pairwise_reference": _summary_reference(json.loads((stage_c_root / "evaluations/pairwise_mlp_evaluation_summary.json").read_text(encoding="utf-8"))), "stage_c_combined_status": combined.get("status")}
    summary["plots"] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "stage_c_failure_summary.json", summary)
    _write_episode_csv(output_dir / "episode_failure_table.csv", rows)
    _write_record_csv(output_dir / "record_failure_table.csv", rows)
    plot_paths = _write_plots(output_dir, summary, rows)
    summary["plots"] = plot_paths
    _write_json(output_dir / "stage_c_failure_summary.json", summary)
    references = [summary["artifact_validation"]["pairwise_reference"], _summary_reference(evaluation)]
    paths = {"summary": str(output_dir / "stage_c_failure_summary.json"), "episode_table": str(output_dir / "episode_failure_table.csv"), "record_table": str(output_dir / "record_failure_table.csv"), "figures": str(output_dir / "figures")}
    _write_report(output_dir / "stage_c_failure_report.md", summary, references, paths)
    return summary


def main() -> int:
    from ea_avs_mvp_v11.core.paths import get_data_root
    data_root = get_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=data_root / "datasets/policy_v11_5")
    parser.add_argument("--stage-b-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_b")
    parser.add_argument("--stage-c-root", type=Path, default=data_root / "datasets/policy_v11_5/stage_c")
    parser.add_argument("--output-dir", type=Path, default=data_root / "datasets/policy_v11_5/stage_c/failure_analysis")
    parser.add_argument("--model-type", choices=("set_ranker", "pairwise_mlp"), default="set_ranker")
    args = parser.parse_args()
    summary = run(dataset_root=args.dataset_root, stage_b_root=args.stage_b_root, stage_c_root=args.stage_c_root, output_dir=args.output_dir, model_type=args.model_type)
    print(json.dumps({"output_dir": str(args.output_dir), "episode_count": summary["episode_count"], "record_count": summary["record_count"], "regret": summary["regret"], "failure_taxonomy": summary["failure_taxonomy"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
