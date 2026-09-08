#!/usr/bin/env python3
"""Val-only audit of direct soft-belief candidate scoring.

This diagnostic consumes the frozen reduced12 RGB-restored cache and the
already-trained visited-history belief estimator.  It does not train a model,
regenerate data, or read policy Test artifacts.  Every selected candidate is
evaluated with its real archived skeleton through the frozen ST-GCN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.action_belief import ActionBeliefEstimator
from activeview.scripts.experiments.run_reduced12_action_belief_estimator import (
    _belief_logits,
    _history_inputs,
    _load_dino_store,
)
from activeview.scripts.experiments.run_reduced12_selector_ceiling_decomposition import (
    _cache,
    _diagnostics,
    _full_predictions,
    _metrics,
    _orders,
    _reference_actions,
)


SEED = 42
NUM_CLASSES = 12
OUTPUT_DIR = Path(__file__).resolve().parents[3] / (
    "experiments/reduced12_eight_placement_v1/soft_belief_candidate_scoring"
)


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return (exp_values / np.sum(exp_values, axis=1, keepdims=True)).astype(np.float32)


def _one_hot(labels: Sequence[int]) -> np.ndarray:
    result = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    result[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)] = 1.0
    return result


def _load_belief(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the frozen estimator and infer visited-history beliefs for Val."""
    dino_lookup, dino_embeddings, dino_summary = _load_dino_store(data_root)
    inputs, input_info = _history_inputs(rows, cache, dino_lookup, dino_embeddings)
    checkpoint = data_root / (
        "checkpoints/activeview_reduced12_action_belief_estimator_v1/"
        "action_belief_best.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing frozen action belief checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ActionBeliefEstimator(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_classes=int(payload["num_classes"]),
    ).to(device)
    if model.num_classes != NUM_CLASSES:
        raise ValueError(f"expected reduced12 belief checkpoint, got {model.num_classes} classes")
    model.load_state_dict(payload["state_dict"])
    mean = np.asarray(payload["input_mean"], dtype=np.float32)
    std = np.asarray(payload["input_std"], dtype=np.float32)
    logits = _belief_logits(
        model,
        inputs,
        mean,
        std,
        device=device,
        batch_size=batch_size,
    )
    return _stable_softmax(logits), {
        "checkpoint": str(checkpoint.resolve()),
        "dino_summary": dino_summary,
        "input_alignment": input_info,
        "input_dim": int(inputs.shape[1]),
    }


def _select_direct(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    beliefs: np.ndarray,
    *,
    evidence: str,
) -> tuple[list[int], list[int]]:
    """Select directly from Stay plus every legal candidate.

    Action ``-1`` is the Stay sentinel because viewpoint id 0 is legal.  The
    score is an expected recognition probability, not an expected log-prob.
    """
    if evidence not in {"imagined", "real"}:
        raise ValueError(f"unsupported evidence: {evidence}")
    cache_index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    evidence_key = "imagined_logp" if evidence == "imagined" else "true_logp"
    actions: list[int] = []
    terminal: list[int] = []
    for row, belief in zip(rows, np.asarray(beliefs, dtype=np.float32)):
        episode_id = str(row["episode_id"])
        index = cache_index[episode_id]
        legal = [int(value) for value in orders[episode_id]]
        stay_probability = np.exp(np.asarray(cache["current_logp_s1"][index], dtype=np.float32))
        scores = [float(np.dot(belief, stay_probability))]
        for candidate in legal:
            candidate_probability = np.exp(
                np.asarray(cache[evidence_key][index, candidate], dtype=np.float32)
            )
            scores.append(float(np.dot(belief, candidate_probability)))
        choice = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        if choice == 0:
            actions.append(-1)
            terminal.append(int(np.argmax(cache["current_logp_s1"][index])))
        else:
            action = legal[choice - 1]
            actions.append(action)
            terminal.append(int(np.argmax(cache["true_logp"][index, action])))
    return actions, terminal


def _method_result(
    actions: Sequence[int],
    terminal: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    orders: Mapping[str, Sequence[int]],
    oracle_actions: Sequence[int],
    full_rows: Sequence[Mapping[str, Any]],
    fallback_s0: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    moving = _diagnostics(
        actions,
        terminal,
        rows,
        cache,
        orders=orders,
        oracle_actions=oracle_actions,
    )
    full_predictions, full_labels = _full_predictions(
        full_rows,
        rows,
        terminal,
        fallback_s0,
    )
    full = {
        "terminal": _metrics(full_predictions, full_labels),
        **{key: value for key, value in moving.items() if key != "terminal"},
    }
    return moving, full


def _copy_reference(
    previous: Mapping[str, Any],
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_name = {
        "Current JR": "Imagined+Inferred JR",
        "FixedH1-H2 Oracle": "FixedH1-H2 Oracle",
    }[name]
    return (
        dict(previous["methods"]["moving"][old_name]),
        dict(previous["methods"]["full"][old_name]),
    )


def _terminal(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return metrics["terminal"]


def _gap_pp(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        "accuracy_pp": 100.0 * (float(left["accuracy"]) - float(right["accuracy"])),
        "macro_f1_pp": 100.0 * (float(left["macro_f1"]) - float(right["macro_f1"])),
    }


def _belief_summary(belief: np.ndarray, labels: Sequence[int]) -> dict[str, float | int]:
    predictions = np.argmax(belief, axis=1)
    safe = np.clip(np.asarray(belief, dtype=np.float64), 1e-12, 1.0)
    return {
        **_metrics(predictions.tolist(), list(labels)),
        "mean_entropy": float(np.mean(-np.sum(safe * np.log(safe), axis=1))),
    }


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    order = result["method_order"]
    moving = result["methods"]["moving"]
    full = result["methods"]["full"]
    lines = [
        "# Reduced12 soft-belief candidate scoring",
        "",
        "Val-only diagnostic. No policy Test data was read, no model was trained, and no data was regenerated.",
        "Terminal predictions always come from the selected real archived observation passed through frozen reduced12 ST-GCN.",
        "",
        "## Moving Val",
        "",
        "| Method | Accuracy | Macro-F1 | Positive action hit | Stay rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in order:
        value = moving[name]
        terminal = value["terminal"]
        lines.append(
            f"| {name} | {terminal['accuracy']:.6f} | {terminal['macro_f1']:.6f} | "
            f"{value.get('positive_action_hit_rate', float('nan')):.6f} | "
            f"{value.get('stay_rate', float('nan')):.6f} |"
        )
    lines.extend([
        "",
        "## Full Val",
        "",
        "| Method | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ])
    for name in order:
        terminal = full[name]["terminal"]
        lines.append(f"| {name} | {terminal['accuracy']:.6f} | {terminal['macro_f1']:.6f} |")
    gaps = result["gaps_pp"]
    lines.extend([
        "",
        "## Requested comparisons (Moving Val, percentage points)",
        "",
        f"- LearnedBelief+Imagined − Current JR: **{gaps['learned_soft_imagined_minus_current_jr']['accuracy_pp']:.3f} pp Accuracy / {gaps['learned_soft_imagined_minus_current_jr']['macro_f1_pp']:.3f} pp Macro-F1**.",
        f"- LearnedBelief-hard − LearnedBelief-soft (Imagined): **{gaps['learned_hard_minus_soft_imagined']['accuracy_pp']:.3f} pp / {gaps['learned_hard_minus_soft_imagined']['macro_f1_pp']:.3f} pp**.",
        f"- LearnedBelief-hard − LearnedBelief-soft (Real): **{gaps['learned_hard_minus_soft_real']['accuracy_pp']:.3f} pp / {gaps['learned_hard_minus_soft_real']['macro_f1_pp']:.3f} pp**.",
        f"- LearnedBelief+Real − GT+Real: **{gaps['learned_real_minus_gt_real']['accuracy_pp']:.3f} pp / {gaps['learned_real_minus_gt_real']['macro_f1_pp']:.3f} pp**.",
        f"- GT+Imagined − GT+Real: **{gaps['gt_imagined_minus_gt_real']['accuracy_pp']:.3f} pp / {gaps['gt_imagined_minus_gt_real']['macro_f1_pp']:.3f} pp**.",
        "",
        "## Scientific interpretation",
        "",
        result["scientific_conclusion"],
        "",
        "## Protocol",
        "",
        "- taxonomy: reduced12 (walk, sit, stand up, bend, crawl, stumble, clap, throw, kick, knock, punch, touching face)",
        "- candidate budget: ALL_LEGAL; stay plus every dynamically legal H2 candidate",
        "- score: sum_y belief[y] * exp(candidate_logp[y])",
        "- `test_used=false`; no Test paths or artifacts were read",
        "- no training, checkpoint modification, or data regeneration",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_root: Path, device: torch.device, output_dir: Path, *, batch_size: int) -> dict[str, Any]:
    root = data_root / "datasets/policy_reduced12_eight_placement_v1"
    rgb_root = data_root / "datasets/policy_reduced12_eight_placement_v1_rgb_restored"
    moving_rows = load_jsonl(root / "stage_d/features/val.jsonl")
    full_rows = load_jsonl(root / "stage_c/predictions/val_predictions.jsonl")
    val_cache = _cache(rgb_root / "counterfactual_cache/val.npz")
    val_orders = _orders(data_root, moving_rows)
    cache_ids = {str(value) for value in val_cache["episode_ids"].tolist()}
    row_ids = {str(row["episode_id"]) for row in moving_rows}
    if cache_ids != row_ids:
        raise ValueError("Val Stage-D/cache episode IDs are not aligned")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [int(row["label_id"]) for row in moving_rows]
    learned_soft, belief_provenance = _load_belief(
        data_root,
        moving_rows,
        val_cache,
        device=device,
        batch_size=batch_size,
    )
    if learned_soft.shape != (len(moving_rows), NUM_CLASSES):
        raise ValueError(f"unexpected learned belief shape: {learned_soft.shape}")
    learned_hard = _one_hot(np.argmax(learned_soft, axis=1).tolist())
    s1_posterior = np.exp(np.asarray(val_cache["current_logp_s1"], dtype=np.float32))
    gt_onehot = _one_hot(labels)

    oracle_actions, _, _ = _reference_actions(
        moving_rows,
        val_cache,
        val_orders,
        mode="gt_label_real",
    )
    fallback_s0 = {
        str(row["episode_id"]): int(row["current_predicted_label_id"])
        for row in full_rows
    }
    previous_path = Path(__file__).resolve().parents[3] / (
        "experiments/reduced12_eight_placement_v1/selector_ceiling_decomposition/result.json"
    )
    previous = json.loads(previous_path.read_text(encoding="utf-8"))

    method_order = [
        "Current JR",
        "S1Posterior + Imagined",
        "LearnedBelief + Imagined",
        "LearnedBelief-hard + Imagined",
        "GT-onehot + Imagined",
        "LearnedBelief + Real",
        "LearnedBelief-hard + Real",
        "GT-onehot + Real",
        "FixedH1-H2 Oracle",
    ]
    methods_moving: dict[str, Any] = {}
    methods_full: dict[str, Any] = {}
    current_moving, current_full = _copy_reference(previous, "Current JR")
    oracle_moving, oracle_full = _copy_reference(previous, "FixedH1-H2 Oracle")
    methods_moving["Current JR"] = current_moving
    methods_full["Current JR"] = current_full
    methods_moving["FixedH1-H2 Oracle"] = oracle_moving
    methods_full["FixedH1-H2 Oracle"] = oracle_full

    beliefs = {
        "S1Posterior": s1_posterior,
        "LearnedBelief": learned_soft,
        "LearnedBelief-hard": learned_hard,
        "GT-onehot": gt_onehot,
    }
    method_specs = [
        ("S1Posterior + Imagined", "S1Posterior", "imagined"),
        ("LearnedBelief + Imagined", "LearnedBelief", "imagined"),
        ("LearnedBelief-hard + Imagined", "LearnedBelief-hard", "imagined"),
        ("GT-onehot + Imagined", "GT-onehot", "imagined"),
        ("LearnedBelief + Real", "LearnedBelief", "real"),
        ("LearnedBelief-hard + Real", "LearnedBelief-hard", "real"),
        ("GT-onehot + Real", "GT-onehot", "real"),
    ]
    for display_name, belief_name, evidence in method_specs:
        actions, terminal = _select_direct(
            moving_rows,
            val_cache,
            val_orders,
            beliefs[belief_name],
            evidence=evidence,
        )
        moving, full = _method_result(
            actions,
            terminal,
            moving_rows,
            val_cache,
            val_orders,
            oracle_actions,
            full_rows,
            fallback_s0,
        )
        methods_moving[display_name] = moving
        methods_full[display_name] = full

    methods_moving = {name: methods_moving[name] for name in method_order}
    methods_full = {name: methods_full[name] for name in method_order}
    moving_terminal = {name: _terminal(methods_moving[name]) for name in method_order}
    gaps = {
        "learned_soft_imagined_minus_current_jr": _gap_pp(
            moving_terminal["LearnedBelief + Imagined"],
            moving_terminal["Current JR"],
        ),
        "learned_hard_minus_soft_imagined": _gap_pp(
            moving_terminal["LearnedBelief-hard + Imagined"],
            moving_terminal["LearnedBelief + Imagined"],
        ),
        "learned_hard_minus_soft_real": _gap_pp(
            moving_terminal["LearnedBelief-hard + Real"],
            moving_terminal["LearnedBelief + Real"],
        ),
        "learned_real_minus_gt_real": _gap_pp(
            moving_terminal["LearnedBelief + Real"],
            moving_terminal["GT-onehot + Real"],
        ),
        "gt_imagined_minus_gt_real": _gap_pp(
            moving_terminal["GT-onehot + Imagined"],
            moving_terminal["GT-onehot + Real"],
        ),
    }
    learned_gain = gaps["learned_soft_imagined_minus_current_jr"]
    hard_gain = gaps["learned_hard_minus_soft_imagined"]
    learned_identity_gap = -gaps["learned_real_minus_gt_real"]["accuracy_pp"]
    wm_gap = -gaps["gt_imagined_minus_gt_real"]["accuracy_pp"]
    if learned_gain["accuracy_pp"] > 2.0:
        interface_conclusion = (
            f"Direct soft-belief scoring exceeds Current JR by {learned_gain['accuracy_pp']:.3f} pp Accuracy, "
            "supporting a belief-selector interface bottleneck."
        )
    else:
        interface_conclusion = (
            f"Direct soft-belief scoring changes Current JR by only {learned_gain['accuracy_pp']:.3f} pp Accuracy, "
            "so it does not show a large belief-selector interface gain."
        )
    if learned_identity_gap > 10.0:
        identity_conclusion = (
            f"LearnedBelief + Real remains {learned_identity_gap:.3f} pp below GT-onehot + Real, "
            "indicating that the learned identity estimator remains a major bottleneck."
        )
    else:
        identity_conclusion = (
            f"LearnedBelief + Real is within {learned_identity_gap:.3f} pp of GT-onehot + Real, "
            "so identity estimation is not the dominant remaining gap."
        )
    if hard_gain["accuracy_pp"] > 0.0:
        hardness = "Hard belief is better than soft belief on the imagined branch."
    else:
        hardness = "Soft belief is at least as effective as hard belief on the imagined branch."
    result = {
        "experiment_id": "REDUCED12_SOFT_BELIEF_CANDIDATE_SCORING",
        "status": "COMPLETED",
        "test_used": False,
        "population": {
            "moving_val_contexts": len(moving_rows),
            "full_val_episodes": len(full_rows),
        },
        "method_order": method_order,
        "methods": {"moving": methods_moving, "full": methods_full},
        "belief_metrics": {
            "s1_posterior": _belief_summary(s1_posterior, labels),
            "learned_belief": _belief_summary(learned_soft, labels),
            "learned_belief_hard": _belief_summary(learned_hard, labels),
            "gt_onehot": _belief_summary(gt_onehot, labels),
        },
        "gaps_pp": gaps,
        "belief_provenance": belief_provenance,
        "scientific_conclusion": " ".join([interface_conclusion, hardness, identity_conclusion, f"The GT imagined-to-real gap is {wm_gap:.3f} pp Accuracy, quantifying the remaining candidate-evidence/WM gap."]),
        "protocol": {
            "taxonomy": "reduced12_no_kneel_clean",
            "candidate_budget": "ALL_LEGAL",
            "score": "sum_y belief[y] * exp(candidate_logp[y])",
            "terminal_observation": "selected real archived skeleton through frozen reduced12 ST-GCN",
            "evidence": "imagined_logp or true_logp from frozen counterfactual cache",
            "seed": SEED,
        },
        "leakage_flags": {
            "test_used": False,
            "test_paths_read": False,
            "training_performed": False,
            "stgcn_modified": False,
            "wm_modified": False,
            "jr_modified": False,
            "data_regenerated": False,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    device = torch.device(args.device)
    result = run(args.data_root, device, args.output_dir, batch_size=args.batch_size)
    print(json.dumps({"status": result["status"], "test_used": result["test_used"]}, indent=2))


if __name__ == "__main__":
    main()
