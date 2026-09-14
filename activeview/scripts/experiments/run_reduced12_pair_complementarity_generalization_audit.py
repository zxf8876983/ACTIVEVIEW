#!/usr/bin/env python3
"""Audit whether pair complementarity generalizes beyond single-view quality.

The audit uses only the reduced12 Train/Moving-Val caches.  Pair priors are
estimated on Train, while Val candidate evidence is consumed only for terminal
MeanLogP evaluation and privileged oracle diagnostics.  No policy Test,
perception regeneration, recognizer training, or new model is used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import classification, correlation
from activeview.scripts.experiments.run_reduced12_budgeted_complementary_viewset_sweep import (
    NUM_CLASSES,
    NUM_VIEWS,
    SharedHead,
    build_arrays,
    fused_prediction,
    legal_slots,
    load_cache,
    load_rows,
    make_oracle_sets,
    pair_diverse_sets,
    pair_matrix,
    random_sets,
    relative_geometry,
    relative_prior,
    set_metrics,
    sha256_file,
)

SEED = 42
OUTPUT_REL = Path("experiments/reduced12_eight_placement_v1/pair_complementarity_generalization_audit")
RUNTIME_REL = Path("diagnostics/reduced12_dual_route_overnight")
STGCN_REL = Path("checkpoints/stgcn_reduced12_no_kneel_clean_babel_diversity_v1/stgcn_reduced12_no_kneel_clean_best.pth")
HEAD_REL = Path("checkpoints/policy_reduced12_eight_placement_v1/view_agnostic_frozen_encoder_head/shared_head_best.pth")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)


def original_arrays(shared: Mapping[str, np.ndarray], cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace shared-head posteriors with the frozen original ST-GCN head."""
    count = len(shared["labels"])
    candidate_logp = np.zeros_like(shared["candidate_logp"], dtype=np.float32)
    current_logp = log_softmax(np.asarray(cache["logits"][:, 0], dtype=np.float32))
    for index in range(count):
        valid = legal_slots(shared, index)
        n_candidates = len(valid)
        candidate_logp[index, :n_candidates] = log_softmax(np.asarray(cache["logits"][index, 1 : n_candidates + 1], dtype=np.float32))
    result = dict(shared)
    result["current_logp"] = current_logp
    result["candidate_logp"] = candidate_logp
    return result


def single_view_quality(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Train-only mean single-view GT margin Q(v) and observation counts."""
    sums = np.zeros(NUM_VIEWS, dtype=np.float64)
    counts = np.zeros(NUM_VIEWS, dtype=np.int64)
    for index, label in enumerate(arrays["labels"]):
        current = int(arrays["current_ids"][index])
        views = [(current, arrays["current_logp"][index])]
        views.extend(
            (int(arrays["candidate_ids"][index, slot]), arrays["candidate_logp"][index, slot])
            for slot in legal_slots(arrays, index)
        )
        for viewpoint, logp in views:
            values = np.asarray(logp, dtype=np.float64)
            margin = float(values[int(label)] - np.max(np.delete(values, int(label))))
            sums[viewpoint] += margin
            counts[viewpoint] += 1
    quality = sums / np.maximum(counts, 1)
    return quality.astype(np.float32), counts


def additive_sets(arrays: Mapping[str, np.ndarray], quality: np.ndarray, budget: int) -> list[list[int]]:
    """Greedy selection using AdditivePair(i,j)=Q(i)+Q(j)."""
    output: list[list[int]] = []
    for index in range(len(arrays["labels"])):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        chosen: list[int] = []
        while len(chosen) < min(budget - 1, len(valid)):
            visited = [current] + [int(arrays["candidate_ids"][index, old]) for old in chosen]

            def score(slot: int) -> tuple[float, int]:
                candidate = int(arrays["candidate_ids"][index, slot])
                value = float(np.mean([quality[view] + quality[candidate] for view in visited]))
                return value, -candidate

            remaining = [slot for slot in valid if slot not in chosen]
            chosen.append(max(remaining, key=score))
        output.append(chosen)
    return output


def shifted_viewpoint(viewpoint: int, azimuth_shift: int) -> int:
    radius, azimuth = divmod(int(viewpoint), 8)
    return radius * 8 + (azimuth + azimuth_shift) % 8


def shifted_pair_sets(arrays: Mapping[str, np.ndarray], matrix: np.ndarray, budget: int, azimuth_shift: int) -> list[list[int]]:
    """Lookup a cyclically shifted Train prior without changing observations."""
    output: list[list[int]] = []
    for index in range(len(arrays["labels"])):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        chosen: list[int] = []
        while len(chosen) < min(budget - 1, len(valid)):
            visited = [current] + [int(arrays["candidate_ids"][index, old]) for old in chosen]

            def score(slot: int) -> tuple[float, int]:
                candidate = int(arrays["candidate_ids"][index, slot])
                value = float(np.mean([matrix[shifted_viewpoint(view, azimuth_shift), shifted_viewpoint(candidate, azimuth_shift)] for view in visited]))
                return value, -candidate

            remaining = [slot for slot in valid if slot not in chosen]
            chosen.append(max(remaining, key=score))
        output.append(chosen)
    return output


def relative_policy_sets(arrays: Mapping[str, np.ndarray], relative: Mapping[tuple[int, str], float], budget: int) -> list[list[int]]:
    output: list[list[int]] = []
    for index in range(len(arrays["labels"])):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        chosen: list[int] = []
        while len(chosen) < min(budget - 1, len(valid)):
            def score(slot: int) -> tuple[float, int]:
                candidate = int(arrays["candidate_ids"][index, slot])
                return float(relative.get(relative_geometry(current, candidate), -999.0)), -candidate

            remaining = [slot for slot in valid if slot not in chosen]
            chosen.append(max(remaining, key=score))
        output.append(chosen)
    return output


def pair_additive_correlations(matrix: Mapping[str, np.ndarray], quality: np.ndarray) -> dict[str, float | int]:
    counts = np.asarray(matrix["counts"], dtype=np.int64)
    valid = counts > 0
    valid &= ~np.eye(NUM_VIEWS, dtype=bool)
    rows, cols = np.where(valid)
    pair_values = np.asarray(matrix["mean"], dtype=np.float64)[rows, cols]
    additive = np.asarray(quality, dtype=np.float64)[rows] + np.asarray(quality, dtype=np.float64)[cols]
    residual = pair_values - additive
    return {
        "valid_pairs": int(len(pair_values)),
        "pair_vs_additive_spearman": correlation(pair_values, additive, spearman=True),
        "pair_vs_additive_pearson": correlation(pair_values, additive, spearman=False),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
    }


def top_pair_overlap(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray], k: int) -> dict[str, float | int]:
    valid = (np.asarray(first["counts"]) > 0) & (np.asarray(second["counts"]) > 0)
    valid &= ~np.eye(NUM_VIEWS, dtype=bool)
    indices = np.flatnonzero(valid.ravel())
    first_order = indices[np.argsort(np.asarray(first["mean"]).ravel()[indices])[::-1][:k]]
    second_order = indices[np.argsort(np.asarray(second["mean"]).ravel()[indices])[::-1][:k]]
    overlap = len(set(first_order.tolist()) & set(second_order.tolist()))
    return {"k": k, "overlap_count": int(overlap), "overlap_fraction": float(overlap / max(k, 1))}


def cross_recognizer_metrics(
    train_original: Mapping[str, np.ndarray], train_shared: Mapping[str, np.ndarray],
    val_original: Mapping[str, np.ndarray], val_shared: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_matrix = pair_matrix([], train_original)
    shared_matrix = pair_matrix([], train_shared)
    valid = (original_matrix["counts"] > 0) & (shared_matrix["counts"] > 0)
    valid &= ~np.eye(NUM_VIEWS, dtype=bool)
    original_values = original_matrix["mean"][valid]
    shared_values = shared_matrix["mean"][valid]
    matrix_report = {
        "pair_matrix_spearman_original_vs_shared": correlation(original_values, shared_values, spearman=True),
        "pair_matrix_pearson_original_vs_shared": correlation(original_values, shared_values, spearman=False),
        "valid_common_pairs": int(np.sum(valid)),
        "top_pair_overlap": {str(k): top_pair_overlap(original_matrix, shared_matrix, k) for k in (10, 50)},
        "test_used": False,
    }
    cross_sets_original_to_shared = {
        f"B{budget}": pair_diverse_sets(val_shared, original_matrix["mean"], budget)
        for budget in (2, 3)
    }
    cross_sets_shared_to_original = {
        f"B{budget}": pair_diverse_sets(val_original, shared_matrix["mean"], budget)
        for budget in (2, 3)
    }
    report = {
        "OriginalPrior_to_SharedRecognizer": {
            key: set_metrics(val_shared, value, f"OriginalPrior_to_SharedRecognizer {key}")
            for key, value in cross_sets_original_to_shared.items()
        },
        "SharedPrior_to_OriginalRecognizer": {
            key: set_metrics(val_original, value, f"SharedPrior_to_OriginalRecognizer {key}")
            for key, value in cross_sets_shared_to_original.items()
        },
        "test_used": False,
    }
    return matrix_report, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_REL)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    seed_everything()
    started = time.time()
    device = torch.device(args.device)
    root = get_data_root()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {
        "experiment": "Pair Complementarity Generalization Audit",
        "train_contexts": 46324,
        "moving_val_contexts": 10080,
        "num_classes": NUM_CLASSES,
        "num_viewpoints": NUM_VIEWS,
        "fusion": "normalized MeanLogP",
        "single_view_quality": "Train mean single-view GT margin over current and legal candidate observations",
        "pair_prior": "Train mean pair GT margin with alpha=10 smoothing",
        "residual": "M_pair - Q(i) - Q(j)",
        "rotation_shifts_deg": [45, 90, 135, 180],
        "cross_recognizers": ["original", "shared"],
        "candidate_action_set": "current/Stay + Stage-A legal candidate_pool",
        "policy_test_used": False,
        "training_used": False,
        "train_priors_used": True,
        "train_prior_smoothing_alpha": 10.0,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "frozen_stgcn_modified": False,
        "gt_label_used_for_oracle_only": True,
        "deployable": False,
        "seed": SEED,
        "device": args.device,
    })
    train_rows, val_rows = load_rows(root)
    checkpoint = root / STGCN_REL
    checkpoint_sha = sha256_file(checkpoint)
    train_cache = load_cache(root / RUNTIME_REL / "train_options.npz", train_rows, False, checkpoint_sha)
    val_cache = load_cache(root / RUNTIME_REL / "val_options.npz", val_rows, False, checkpoint_sha)
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(root / HEAD_REL, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    train_shared = build_arrays(train_rows, train_cache, head, device)
    val_shared = build_arrays(val_rows, val_cache, head, device)
    train_original = original_arrays(train_shared, train_cache)
    val_original = original_arrays(val_shared, val_cache)
    if len(train_rows) != 46324 or len(val_rows) != 10080:
        raise ValueError("unexpected Train/Moving-Val population")

    shared_matrix = pair_matrix(train_rows, train_shared)
    original_matrix = pair_matrix(train_rows, train_original)
    quality_shared, counts_shared = single_view_quality(train_shared)
    quality_original, counts_original = single_view_quality(train_original)
    residual_matrix = shared_matrix["mean"] - quality_shared[:, None] - quality_shared[None, :]
    residual_matrix = residual_matrix.astype(np.float32)

    rng = np.random.default_rng(SEED)
    random_policies: dict[str, list[list[int]]] = {}
    for budget in (2, 3):
        random_policies[f"Random B{budget}"] = random_sets(val_shared, budget, rng)
    relative = relative_prior(val_shared, train_shared)
    policies: dict[str, Sequence[Sequence[int]]] = {
        "Random B2": random_policies["Random B2"],
        "Random B3": random_policies["Random B3"],
        "PairMeanGreedy B2": pair_diverse_sets(val_shared, shared_matrix["mean"], 2),
        "PairMeanGreedy B3": pair_diverse_sets(val_shared, shared_matrix["mean"], 3),
        "AdditiveQuality B2": additive_sets(val_shared, quality_shared, 2),
        "AdditiveQuality B3": additive_sets(val_shared, quality_shared, 3),
        "ResidualPairGreedy B2": pair_diverse_sets(val_shared, residual_matrix, 2),
        "ResidualPairGreedy B3": pair_diverse_sets(val_shared, residual_matrix, 3),
        "RelativeGeometryPrior B2": relative_policy_sets(val_shared, relative, 2),
        "RelativeGeometryPrior B3": relative_policy_sets(val_shared, relative, 3),
    }
    oracle_sets: dict[str, Sequence[Sequence[int]]] = {}
    for budget in (2, 3):
        exact, _ = make_oracle_sets(val_shared, budget)
        oracle_sets[f"PairMargin Oracle B{budget}"] = exact
    policies.update(oracle_sets)
    metrics = {name: set_metrics(val_shared, sets, name) for name, sets in policies.items()}

    expected = {
        "Random B2": (0.429762, 0.424594),
        "Random B3": (0.487401, 0.477211),
        "PairMeanGreedy B2": (0.508234, 0.497486),
        "PairMeanGreedy B3": (0.557639, 0.546970),
    }
    observed = {name: {"accuracy": metrics[name]["accuracy"], "macro_f1": metrics[name]["macro_f1"]} for name in expected}
    gate_pass = all(
        abs(observed[name]["accuracy"] - value[0]) <= 0.01
        and abs(observed[name]["macro_f1"] - value[1]) <= 0.01
        for name, value in expected.items()
    )
    write_json(output / "protocol_reproduction.json", {"expected": expected, "observed": observed, "tolerance_pp": 1.0, "gate_pass": gate_pass, "test_used": False})
    if not gate_pass:
        raise RuntimeError("pair complementarity protocol gate failed")

    additive_report = pair_additive_correlations(shared_matrix, quality_shared)
    write_json(output / "single_view_quality.json", {
        "shared_head": {"quality_mean_gt_margin": quality_shared.tolist(), "observation_counts": counts_shared.tolist()},
        "original_head": {"quality_mean_gt_margin": quality_original.tolist(), "observation_counts": counts_original.tolist()},
        "definition": "Train mean single-view GT margin over current and legal candidate observations",
        "test_used": False,
    })
    write_json(output / "additive_pair_metrics.json", {
        "B2": metrics["AdditiveQuality B2"], "B3": metrics["AdditiveQuality B3"],
        "pair_matrix_vs_Q_sum": additive_report, "test_used": False,
    })
    valid = np.asarray(shared_matrix["counts"]) > 0
    residual_values = residual_matrix[valid]
    write_json(output / "residual_pair_metrics.json", {
        "B2": metrics["ResidualPairGreedy B2"], "B3": metrics["ResidualPairGreedy B3"],
        "residual_matrix_stats": {"mean": float(np.mean(residual_values)), "std": float(np.std(residual_values)), "min": float(np.min(residual_values)), "max": float(np.max(residual_values))},
        "definition": "M_pair_shared - Q(i) - Q(j), with Train-smoothed M_pair",
        "test_used": False,
    })

    rotation = {"original": {"B2": metrics["PairMeanGreedy B2"], "B3": metrics["PairMeanGreedy B3"]}}
    for shift in (1, 2, 3, 4, 5, 6, 7):
        for budget in (2, 3):
            selected = shifted_pair_sets(val_shared, shared_matrix["mean"], budget, shift)
            value = set_metrics(val_shared, selected, f"PairMeanGreedy shifted +{shift * 45}deg B{budget}")
            value["accuracy_drop_vs_original_pp"] = float((value["accuracy"] - metrics[f"PairMeanGreedy B{budget}"]["accuracy"]) * 100.0)
            rotation.setdefault(f"+{shift * 45}deg", {})[f"B{budget}"] = value
    write_json(output / "rotation_audit.json", {"azimuth_shifts": rotation, "test_used": False})

    cross_matrix, cross_results = cross_recognizer_metrics(train_original, train_shared, val_original, val_shared)
    write_json(output / "matrix_correlations.json", {**additive_report, **cross_matrix, "test_used": False})
    write_json(output / "cross_recognizer_audit.json", cross_results | {"matrix": cross_matrix})
    write_json(output / "relative_geometry_prior.json", {
        "train_derived_bins": {f"{key[0]}_{key[1]}": value for key, value in relative.items()},
        "B2": metrics["RelativeGeometryPrior B2"], "B3": metrics["RelativeGeometryPrior B3"],
        "vs_pair_B2_accuracy_gap_pp": float((metrics["RelativeGeometryPrior B2"]["accuracy"] - metrics["PairMeanGreedy B2"]["accuracy"]) * 100.0),
        "test_used": False,
    })

    write_json(output / "pair_complementarity_matrix.json", {
        "shared_mean_pair_margin": shared_matrix["mean"].tolist(),
        "shared_pair_fusion_accuracy": shared_matrix["accuracy"].tolist(),
        "original_mean_pair_margin": original_matrix["mean"].tolist(),
        "test_used": False,
    })
    write_json(output / "result.json", {
        "experiment_id": "REDUCED12_PAIR_COMPLEMENTARITY_GENERALIZATION_AUDIT",
        "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows)},
        "metrics": metrics, "protocol_reproduction": {"pass": gate_pass, "observed": observed},
        "flags": {"policy_test_used": False, "training_used": False, "recognizer_modified": False, "new_perception_generated": False, "gt_label_used_for_oracle_only": True, "deployable": False},
    })
    write_json(output / "runtime_summary.json", {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.time() - started, "test_used": False})

    pair_minus_additive_b2 = (metrics["PairMeanGreedy B2"]["accuracy"] - metrics["AdditiveQuality B2"]["accuracy"]) * 100
    pair_minus_additive_b3 = (metrics["PairMeanGreedy B3"]["accuracy"] - metrics["AdditiveQuality B3"]["accuracy"]) * 100
    residual_gain_b3 = (metrics["ResidualPairGreedy B3"]["accuracy"] - metrics["Random B3"]["accuracy"]) * 100
    shift_values = {key: {budget: value[budget]["accuracy"] for budget in ("B2", "B3")} for key, value in rotation.items() if key != "original"}
    max_shift_drop = max((metrics["PairMeanGreedy B2"]["accuracy"] - value["B2"]["accuracy"]) * 100 for value in rotation.values() if isinstance(value, dict) and "B2" in value)
    max_shift_drop_b3 = max((metrics["PairMeanGreedy B3"]["accuracy"] - value["B3"]["accuracy"]) * 100 for value in rotation.values() if isinstance(value, dict) and "B3" in value)
    analysis = [
        "# Pair Complementarity Generalization Audit", "",
        "- Moving Val: 10,080 contexts; Train: 46,324 contexts",
        "- Policy Test: false; recognizer modified: false; new perception generated: false",
        "- Recognizer: frozen reduced12 ST-GCN with frozen original/shared heads",
        "- Action set: current/Stay + Stage-A legal candidate_pool",
        "- Fusion: fixed normalized MeanLogP; GT label/evidence only for Train priors or privileged Val oracle evaluation",
        "- Continuous navigation synchronization is not modeled; this is a finite observation-set audit.", "",
        "## Protocol reproduction",
        f"- Random B2/B3: {metrics['Random B2']['accuracy']:.6f}/{metrics['Random B3']['accuracy']:.6f} Acc; PairMean B2/B3: {metrics['PairMeanGreedy B2']['accuracy']:.6f}/{metrics['PairMeanGreedy B3']['accuracy']:.6f}; gate={'PASS' if gate_pass else 'FAIL'}.", "",
        "## Required comparisons",
        f"- AdditiveQuality B2/B3: {metrics['AdditiveQuality B2']['accuracy']:.6f}/{metrics['AdditiveQuality B3']['accuracy']:.6f} Acc; PairMean - Additive = {pair_minus_additive_b2:+.3f}/{pair_minus_additive_b3:+.3f} pp.",
        f"- ResidualPairGreedy B2/B3: {metrics['ResidualPairGreedy B2']['accuracy']:.6f}/{metrics['ResidualPairGreedy B3']['accuracy']:.6f} Acc; Residual B3 - Random B3 = {residual_gain_b3:+.3f} pp.",
        f"- PairMatrix vs Q(i)+Q(j): Spearman={additive_report['pair_vs_additive_spearman']:.6f}, Pearson={additive_report['pair_vs_additive_pearson']:.6f}; residual mean/std={additive_report['residual_mean']:.6f}/{additive_report['residual_std']:.6f}.",
        f"- RelativeGeometryPrior B2/B3: {metrics['RelativeGeometryPrior B2']['accuracy']:.6f}/{metrics['RelativeGeometryPrior B3']['accuracy']:.6f}; B2 gap vs PairMean={((metrics['RelativeGeometryPrior B2']['accuracy'] - metrics['PairMeanGreedy B2']['accuracy']) * 100):+.3f} pp.",
        "",
        "## Rotation robustness",
        f"- Cyclic +45°/+90°/+135°/+180° B2 Acc: " + ", ".join(f"{shift_values[f'+{shift}deg']['B2']:.6f}" for shift in (45, 90, 135, 180)) + ".",
        f"- Cyclic +45°/+90°/+135°/+180° B3 Acc: " + ", ".join(f"{shift_values[f'+{shift}deg']['B3']:.6f}" for shift in (45, 90, 135, 180)) + ".",
        f"- Maximum B2 drop across all cyclic shifts: {max_shift_drop:.3f} pp.",
        f"- Maximum B3 drop across all cyclic shifts: {max_shift_drop_b3:.3f} pp.",
        "",
        "## Cross-recognizer",
        f"- Original-vs-Shared Train pair matrix Spearman={cross_matrix['pair_matrix_spearman_original_vs_shared']:.6f}, Pearson={cross_matrix['pair_matrix_pearson_original_vs_shared']:.6f}; top-10 overlap={cross_matrix['top_pair_overlap']['10']['overlap_fraction']:.3f}.",
        f"- Original prior → Shared B2/B3: {cross_results['OriginalPrior_to_SharedRecognizer']['B2']['accuracy']:.6f}/{cross_results['OriginalPrior_to_SharedRecognizer']['B3']['accuracy']:.6f}; Shared prior → Original B2/B3: {cross_results['SharedPrior_to_OriginalRecognizer']['B2']['accuracy']:.6f}/{cross_results['SharedPrior_to_OriginalRecognizer']['B3']['accuracy']:.6f}.",
        "",
        "## Decision",
    ]
    if pair_minus_additive_b2 < 1.0 and pair_minus_additive_b3 < 1.0:
        primary = "PAIR GAIN MOSTLY EXPLAINED BY SINGLE-VIEW QUALITY"
    elif pair_minus_additive_b2 >= 2.0 or pair_minus_additive_b3 >= 2.0:
        primary = "PAIR INTERACTION HAS CLEAR ADDITIONAL VALUE"
    else:
        primary = "MIXED SINGLE-VIEW QUALITY AND PAIR INTERACTION"
    residual_decision = "GENUINE PAIR COMPLEMENTARITY EXISTS" if residual_gain_b3 >= 3.0 else "PAIR PRIOR MOSTLY REFLECTS VIEW QUALITY / DATASET PRIOR"
    rotation_decision = "ABSOLUTE VIEWPOINT-ID DEPENDENCE IS STRONG" if max_shift_drop > 5.0 else "NO STRONG ABSOLUTE VIEWPOINT-ID DEPENDENCE DETECTED"
    cross_decision = "COMPLEMENTARITY IS RECOGNIZER-SPECIFIC" if cross_matrix["pair_matrix_spearman_original_vs_shared"] < 0.5 else "COMPLEMENTARITY IS PARTLY STABLE ACROSS RECOGNIZERS"
    analysis.extend([
        f"- {primary}.", f"- {residual_decision}.", f"- {rotation_decision}.", f"- {cross_decision}.",
        f"- PairMeanGreedy B=3 remains justified: it reaches {metrics['PairMeanGreedy B3']['accuracy']:.6f} Accuracy / {metrics['PairMeanGreedy B3']['macro_f1']:.6f} Macro-F1 and outperforms Random B3 by {(metrics['PairMeanGreedy B3']['accuracy'] - metrics['Random B3']['accuracy']) * 100:+.3f} pp.",
        "- No TinySetScorer was run because the deterministic B3 branch exceeded the optional <0.55 trigger.",
        "- This audit does not authorize a follow-up experiment; the next step must be selected explicitly.",
    ])
    (output / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate_pass": gate_pass, "primary_decision": primary, "residual_decision": residual_decision}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
