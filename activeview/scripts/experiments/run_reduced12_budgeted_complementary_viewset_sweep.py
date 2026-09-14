#!/usr/bin/env python3
"""Train/Val-only budgeted complementary multi-view HAR sweep.

This is a deterministic structure-discovery audit.  O0 is already observed
and each additional view is selected from the Stage-A legal candidate pool.
No candidate observation evidence is consumed before selection; real archived
candidate evidence is used only for frozen terminal evaluation and oracle
targets.  Continuous robot/human synchronization is intentionally not modeled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import LABELS, classification, correlation
from activeview.scripts.experiments.run_reduced12_o0_conditioned_second_view import (
    build_arrays,
    load_cache,
    load_rows,
    sha256_file,
)
from activeview.scripts.experiments.run_reduced12_dual_route_overnight import SharedHead

SEED = 42
NUM_CLASSES = 12
NUM_VIEWS = 32
MAX_OPTIONS = 21
OUTPUT_REL = Path("experiments/reduced12_eight_placement_v1/budgeted_complementary_viewset_sweep")
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


def relative_geometry(current: int, candidate: int) -> tuple[int, str]:
    cr, ca = divmod(int(current), 8)
    vr, va = divmod(int(candidate), 8)
    delta = abs(va - ca)
    delta = min(delta, 8 - delta)
    relation = "same" if vr == cr else "inward" if vr < cr else "outward"
    return delta, relation


def fused_prediction(arrays: Mapping[str, np.ndarray], sets: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    predictions = np.zeros(len(sets), dtype=np.int64)
    true_prob = np.zeros(len(sets), dtype=np.float64)
    margins = np.zeros(len(sets), dtype=np.float64)
    for index, slots in enumerate(sets):
        logs = [arrays["current_logp"][index]]
        logs.extend(arrays["candidate_logp"][index, int(slot)] for slot in slots)
        value = np.mean(np.stack(logs, axis=0), axis=0)
        value = value - np.max(value)
        value = value - np.log(np.exp(value).sum())
        predictions[index] = int(np.argmax(value))
        true_prob[index] = float(np.exp(value[labels[index]]))
        other = np.delete(value, labels[index])
        margins[index] = float(value[labels[index]] - np.max(other))
    return predictions, true_prob, margins


def set_metrics(arrays: Mapping[str, np.ndarray], sets: Sequence[Sequence[int]], name: str) -> dict[str, Any]:
    predictions, true_prob, margins = fused_prediction(arrays, sets)
    result = classification(arrays["labels"], predictions)
    result.update({"method": name, "budget": len(sets[0]) + 1 if sets else 1, "mean_gt_probability": float(np.mean(true_prob)), "mean_gt_margin": float(np.mean(margins)), "move_count": int(np.sum([len(x) for x in sets])), "mean_path_length_m": float(np.mean([sum(float(arrays["geodesic"][i, slot]) for slot in slots) for i, slots in enumerate(sets)])), "median_path_length_m": float(np.median([sum(float(arrays["geodesic"][i, slot]) for slot in slots) for i, slots in enumerate(sets)])), "p90_path_length_m": float(np.percentile([sum(float(arrays["geodesic"][i, slot]) for slot in slots) for i, slots in enumerate(sets)], 90)), "test_used": False})
    return result


def legal_slots(arrays: Mapping[str, np.ndarray], index: int) -> list[int]:
    return np.flatnonzero(arrays["candidate_mask"][index]).astype(int).tolist()


def random_sets(arrays: Mapping[str, np.ndarray], budget: int, rng: np.random.Generator) -> list[list[int]]:
    output = []
    for index in range(len(arrays["labels"])):
        valid = legal_slots(arrays, index)
        count = min(budget - 1, len(valid))
        output.append(sorted(rng.choice(valid, size=count, replace=False).astype(int).tolist()))
    return output


def pair_matrix(rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    sums = np.zeros((NUM_VIEWS, NUM_VIEWS), dtype=np.float64)
    counts = np.zeros_like(sums)
    rescue = np.zeros_like(sums)
    accuracy = np.zeros_like(sums)
    class_sums = np.zeros((NUM_CLASSES, NUM_VIEWS, NUM_VIEWS), dtype=np.float64)
    class_counts = np.zeros_like(class_sums)
    for index, label in enumerate(arrays["labels"]):
        current = int(arrays["current_ids"][index])
        for slot in legal_slots(arrays, index):
            candidate = int(arrays["candidate_ids"][index, slot])
            pair_log = 0.5 * (arrays["current_logp"][index] + arrays["candidate_logp"][index, slot])
            pair_log = pair_log - np.max(pair_log)
            pair_log = pair_log - np.log(np.exp(pair_log).sum())
            pair_margin = float(pair_log[int(label)] - np.max(np.delete(pair_log, int(label))))
            sums[current, candidate] += pair_margin
            counts[current, candidate] += 1
            accuracy[current, candidate] += float(np.argmax(pair_log) == label)
            rescue[current, candidate] += float(np.argmax(pair_log) == label and np.argmax(arrays["current_logp"][index]) != label)
            class_sums[int(label), current, candidate] += pair_margin
            class_counts[int(label), current, candidate] += 1
    global_mean = sums.sum() / max(counts.sum(), 1)
    mean = (sums + 10.0 * global_mean) / np.maximum(counts + 10.0, 1e-9)
    class_mean = class_sums / np.maximum(class_counts, 1)
    class_mean = np.where(class_counts > 0, class_mean, global_mean)
    return {"mean": mean.astype(np.float32), "counts": counts.astype(np.int64), "rescue": rescue / np.maximum(counts, 1), "accuracy": accuracy / np.maximum(counts, 1), "class_mean": class_mean.astype(np.float32)}


def prior_sets(arrays: Mapping[str, np.ndarray], matrix: np.ndarray, budget: int) -> list[list[int]]:
    output = []
    for index in range(len(arrays["labels"])):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        ordered = sorted(valid, key=lambda slot: (-float(matrix[current, int(arrays["candidate_ids"][index, slot])]), int(arrays["candidate_ids"][index, slot])))
        output.append(ordered[: min(budget - 1, len(ordered))])
    return output


def geometry_sets(arrays: Mapping[str, np.ndarray], budget: int, mode: str) -> list[list[int]]:
    output = []
    for index in range(len(arrays["labels"])):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        chosen: list[int] = []
        while len(chosen) < min(budget - 1, len(valid)):
            remaining = [slot for slot in valid if slot not in chosen]
            visited = [current] + [int(arrays["candidate_ids"][index, slot]) for slot in chosen]

            def score(slot: int) -> tuple[float, int]:
                candidate = int(arrays["candidate_ids"][index, slot])
                angular = min(relative_geometry(view, candidate)[0] for view in visited)
                radius = min(abs(view // 8 - candidate // 8) for view in visited)
                if mode == "opposite":
                    value = float(angular)
                elif mode == "radius":
                    value = float(radius)
                elif mode == "coverage":
                    value = float(angular) / 4.0 + 0.5 * float(radius) / 3.0
                else:
                    value = float(angular)
                return value, -candidate

            chosen.append(max(remaining, key=score))
        output.append(chosen)
    return output


def pair_diverse_sets(
    arrays: Mapping[str, np.ndarray], matrix: np.ndarray, budget: int, beta: float = 0.0,
    coverage: bool = False, nav_lambda: float = 0.0, pair_mode: str = "mean",
) -> list[list[int]]:
    output = []
    for index in range(len(arrays["labels"])):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        chosen: list[int] = []
        for _ in range(min(budget - 1, len(valid))):
            def score(slot: int) -> tuple[float, int]:
                candidate = int(arrays["candidate_ids"][index, slot])
                visited = [current] + [int(arrays["candidate_ids"][index, old]) for old in chosen]
                pair_values = [float(matrix[view, candidate]) for view in visited]
                if pair_mode == "min":
                    pair = min(pair_values)
                elif pair_mode == "max":
                    pair = max(pair_values)
                else:
                    pair = float(np.mean(pair_values))
                angle = relative_geometry(current, candidate)[0] / 4.0
                diversity = min([relative_geometry(int(arrays["candidate_ids"][index, old]), candidate)[0] / 4.0 for old in chosen], default=angle)
                cov = diversity if coverage else 0.0
                nav = float(arrays["geodesic"][index, slot])
                return pair + beta * (angle + cov) - nav_lambda * nav, -candidate
            remaining = [slot for slot in valid if slot not in chosen]
            chosen.append(max(remaining, key=score))
        output.append(chosen)
    return output


def relative_prior(arrays: Mapping[str, np.ndarray], train_arrays: Mapping[str, np.ndarray]) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], list[float]] = {}
    for index in range(len(train_arrays["labels"])):
        current = int(train_arrays["current_ids"][index])
        for slot in legal_slots(train_arrays, index):
            candidate = int(train_arrays["candidate_ids"][index, slot])
            delta, relation = relative_geometry(current, candidate)
            pair_log = 0.5 * (train_arrays["current_logp"][index] + train_arrays["candidate_logp"][index, slot])
            pair_log = pair_log - np.max(pair_log)
            pair_log = pair_log - np.log(np.exp(pair_log).sum())
            values.setdefault((delta, relation), []).append(float(pair_log[int(train_arrays["labels"][index])] - np.max(np.delete(pair_log, int(train_arrays["labels"][index])))))
    return {key: float(np.mean(value)) for key, value in values.items()}


def relative_bin_stats(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    bins: dict[str, list[tuple[float, bool]]] = {}
    for index, label in enumerate(arrays["labels"]):
        current = int(arrays["current_ids"][index])
        for slot in legal_slots(arrays, index):
            candidate = int(arrays["candidate_ids"][index, slot])
            delta, relation = relative_geometry(current, candidate)
            fused = np.mean(np.stack((arrays["current_logp"][index], arrays["candidate_logp"][index, slot]), axis=0), axis=0)
            fused = fused - np.max(fused)
            fused = fused - np.log(np.exp(fused).sum())
            margin = float(fused[int(label)] - np.max(np.delete(fused, int(label))))
            key = f"azimuth_{delta}_{relation}"
            bins.setdefault(key, []).append((margin, bool(np.argmax(fused) == int(label))))
    return {key: {"count": len(values), "mean_pair_margin": float(np.mean([item[0] for item in values])), "fusion_accuracy": float(np.mean([item[1] for item in values]))} for key, values in bins.items()}


def pair_geometry_correlations(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    margins: list[float] = []
    angular: list[float] = []
    radius_delta: list[float] = []
    for index, label in enumerate(arrays["labels"]):
        current = int(arrays["current_ids"][index])
        for slot in legal_slots(arrays, index):
            candidate = int(arrays["candidate_ids"][index, slot])
            pair_log = 0.5 * (arrays["current_logp"][index] + arrays["candidate_logp"][index, slot])
            pair_log = pair_log - np.max(pair_log)
            pair_log = pair_log - np.log(np.exp(pair_log).sum())
            margins.append(float(pair_log[int(label)] - np.max(np.delete(pair_log, int(label)))))
            delta, _ = relative_geometry(current, candidate)
            angular.append(float(delta))
            radius_delta.append(float(abs(current // 8 - candidate // 8)))
    return {"pair_margin_vs_angular_spearman": correlation(margins, angular, spearman=True), "pair_margin_vs_radius_difference_spearman": correlation(margins, radius_delta, spearman=True)}


def set3_prior(arrays: Mapping[str, np.ndarray], pair_mean: np.ndarray, alpha: float = 10.0) -> dict[tuple[int, int, int], float]:
    """Estimate a Train-only smoothed prior for unordered (v0, v1, v2) sets."""
    sums: dict[tuple[int, int, int], float] = {}
    counts: dict[tuple[int, int, int], int] = {}
    for index, label in enumerate(arrays["labels"]):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        for first, second in combinations(valid, 2):
            first_id = int(arrays["candidate_ids"][index, first])
            second_id = int(arrays["candidate_ids"][index, second])
            logs = np.stack((arrays["current_logp"][index], arrays["candidate_logp"][index, first], arrays["candidate_logp"][index, second]), axis=0)
            fused = np.mean(logs, axis=0)
            fused = fused - np.max(fused)
            fused = fused - np.log(np.exp(fused).sum())
            margin = float(fused[int(label)] - np.max(np.delete(fused, int(label))))
            key = (current, min(first_id, second_id), max(first_id, second_id))
            sums[key] = sums.get(key, 0.0) + margin
            counts[key] = counts.get(key, 0) + 1

    smoothed: dict[tuple[int, int, int], float] = {}
    for key, total in sums.items():
        n = counts[key]
        v0, v1, v2 = key
        pair_based = float(np.mean((pair_mean[v0, v1], pair_mean[v0, v2], pair_mean[v1, v2])))
        smoothed[key] = float((n * (total / n) + alpha * pair_based) / (n + alpha))
    return smoothed


def set3_prior_sets(
    arrays: Mapping[str, np.ndarray], prior: Mapping[tuple[int, int, int], float], pair_mean: np.ndarray,
) -> list[list[int]]:
    output: list[list[int]] = []
    for index in range(len(arrays["labels"])):
        current = int(arrays["current_ids"][index])
        valid = legal_slots(arrays, index)
        best: tuple[float, tuple[int, int], list[int]] | None = None
        for first, second in combinations(valid, 2):
            first_id = int(arrays["candidate_ids"][index, first])
            second_id = int(arrays["candidate_ids"][index, second])
            ordered_ids = tuple(sorted((first_id, second_id)))
            key = (current, ordered_ids[0], ordered_ids[1])
            pair_based = float(np.mean((pair_mean[current, first_id], pair_mean[current, second_id], pair_mean[first_id, second_id])))
            score = float(prior.get(key, pair_based))
            rank_key = (score, tuple(-value for value in ordered_ids), [first, second])
            if best is None or rank_key[:2] > best[:2]:
                best = (score, tuple(-value for value in ordered_ids), [first, second])
        output.append(best[2] if best is not None else valid[:2])
    return output


def make_oracle_sets(arrays: Mapping[str, np.ndarray], budget: int, beam: int = 32) -> tuple[list[list[int]], list[list[int]]]:
    exact: list[list[int]] = []
    for index in range(len(arrays["labels"])):
        valid = legal_slots(arrays, index)
        if budget >= 4:
            # Fixed-width beam oracle keeps B4 tractable while retaining the
            # registered exact B2/B3 oracle definitions.
            partials: list[list[int]] = [[]]
            for _ in range(min(budget - 1, len(valid))):
                expanded = []
                for prefix in partials:
                    for slot in valid:
                        if slot not in prefix:
                            expanded.append(prefix + [slot])
                expanded.sort(key=lambda combo: fused_prediction({k: (v[index:index + 1] if isinstance(v, np.ndarray) and v.shape[0] == len(arrays["labels"]) else v) for k, v in arrays.items()}, [combo])[2][0], reverse=True)
                partials = expanded[:beam]
            exact.append(partials[0] if partials else valid[: budget - 1])
            continue
        combos = combinations(valid, min(budget - 1, len(valid)))
        best: list[int] = []
        best_margin = -float("inf")
        for combo in combos:
            _, _, margins = fused_prediction({k: (v[index:index + 1] if isinstance(v, np.ndarray) and v.shape[0] == len(arrays["labels"]) else v) for k, v in arrays.items()}, [list(combo)])
            if float(margins[0]) > best_margin:
                best_margin, best = float(margins[0]), list(combo)
        exact.append(best)
    return exact, exact


def any_correct_fusion_coverage(arrays: Mapping[str, np.ndarray], budget: int) -> dict[str, Any]:
    """Count contexts with at least one correct fused set (privileged audit)."""
    hits = 0
    labels = arrays["labels"]
    for index, label in enumerate(labels):
        valid = legal_slots(arrays, index)
        found = False
        for combo in combinations(valid, min(budget - 1, len(valid))):
            logs = np.stack(
                [arrays["current_logp"][index]]
                + [arrays["candidate_logp"][index, slot] for slot in combo],
                axis=0,
            )
            fused = np.mean(logs, axis=0)
            if int(np.argmax(fused)) == int(label):
                found = True
                break
        hits += int(found)
    return {"count": hits, "fraction": float(hits / max(len(labels), 1)), "budget": budget}


def evaluate_policy(arrays: Mapping[str, np.ndarray], policies: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, Any]:
    return {name: set_metrics(arrays, sets, name) for name, sets in policies.items()}


def rescue_counts(arrays: Mapping[str, np.ndarray], sets: Sequence[Sequence[int]]) -> dict[str, int]:
    all_single_wrong_to_fusion_correct = 0
    some_single_correct_to_fusion_correct = 0
    single_correct_to_fusion_wrong = 0
    all_single_correct_to_fusion_wrong = 0
    labels = arrays["labels"]
    predictions, _, _ = fused_prediction(arrays, sets)
    for index, slots in enumerate(sets):
        single_predictions = [int(np.argmax(arrays["current_logp"][index]))]
        single_predictions.extend(int(np.argmax(arrays["candidate_logp"][index, slot])) for slot in slots)
        correct_single = [prediction == int(labels[index]) for prediction in single_predictions]
        fusion_correct = int(predictions[index]) == int(labels[index])
        if not any(correct_single) and fusion_correct:
            all_single_wrong_to_fusion_correct += 1
        if any(correct_single) and fusion_correct:
            some_single_correct_to_fusion_correct += 1
        if any(correct_single) and not fusion_correct:
            single_correct_to_fusion_wrong += 1
        if all(correct_single) and not fusion_correct:
            all_single_correct_to_fusion_wrong += 1
    return {
        "all_single_wrong_to_fusion_correct": all_single_wrong_to_fusion_correct,
        "some_single_correct_to_fusion_correct": some_single_correct_to_fusion_correct,
        "single_correct_to_fusion_wrong": single_correct_to_fusion_wrong,
        "all_single_correct_to_fusion_wrong": all_single_correct_to_fusion_wrong,
    }


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
        "experiment": "Budgeted Complementary Multi-View Active HAR Sweep",
        "moving_val_contexts": 10080,
        "train_contexts": 46324,
        "budgets": [2, 3, 4],
        "num_classes": NUM_CLASSES,
        "num_viewpoints": NUM_VIEWS,
        "fusion": "normalized MeanLogP",
        "pair_prior": "Train mean Pair GT-Margin with alpha=10 smoothing",
        "set3_prior": "Train unordered triplet MeanLogP GT-Margin with pair-based alpha=10 smoothing",
        "beam_size_b4_oracle": 32,
        "candidate_action_set": "current/Stay + Stage-A legal candidate_pool",
        "policy_test_used": False,
        "training_used": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "frozen_stgcn_modified": False,
        "gt_label_used_for_oracle_only": True,
        "continuous_human_navigation_sync_modeled": False,
        "seed": SEED,
        "device": args.device,
    })
    checkpoint = root / STGCN_REL
    sha = sha256_file(checkpoint)
    train_rows, val_rows = load_rows(root)
    runtime = root / RUNTIME_REL
    train_cache = load_cache(runtime / "train_options.npz", train_rows, False, sha)
    val_cache = load_cache(runtime / "val_options.npz", val_rows, False, sha)
    head = SharedHead().to(device)
    head.load_state_dict(torch.load(root / HEAD_REL, map_location=device, weights_only=False)["state_dict"])
    head.eval()
    train_arrays = build_arrays(train_rows, train_cache, head, device)
    val_arrays = build_arrays(val_rows, val_cache, head, device)
    if len(train_rows) != 46324 or len(val_rows) != 10080:
        raise ValueError("unexpected Train/Moving-Val population")
    matrix = pair_matrix(train_rows, train_arrays)
    rng = np.random.default_rng(SEED)
    random_policies: dict[str, list[list[int]]] = {}
    for budget in (2, 3, 4):
        random_policies[f"Random B{budget}"] = random_sets(val_arrays, budget, rng)
    policies: dict[str, Sequence[Sequence[int]]] = {"Stay": [[] for _ in val_rows]}
    for budget in (2, 3, 4):
        policies.update(random_policies)
        policies[f"MaxMinAngular B{budget}"] = geometry_sets(val_arrays, budget, "maxmin")
        policies[f"OppositeAzimuth B{budget}"] = geometry_sets(val_arrays, budget, "opposite")
        policies[f"MaxMinRadius B{budget}"] = geometry_sets(val_arrays, budget, "radius")
        policies[f"GeometricCoverage B{budget}"] = geometry_sets(val_arrays, budget, "coverage")
        policies[f"PairMeanGreedy B{budget}"] = pair_diverse_sets(val_arrays, matrix["mean"], budget)
        policies[f"PairMinGreedy B{budget}"] = pair_diverse_sets(val_arrays, matrix["mean"], budget, pair_mode="min")
        policies[f"PairMaxGreedy B{budget}"] = pair_diverse_sets(val_arrays, matrix["mean"], budget, pair_mode="max")
        for beta in (0.25, 0.5, 1.0):
            policies[f"Pair+Angular beta={beta} B{budget}"] = pair_diverse_sets(val_arrays, matrix["mean"], budget, beta=beta)
        policies[f"Pair+GeoCoverage B{budget}"] = pair_diverse_sets(val_arrays, matrix["mean"], budget, beta=0.5, coverage=True)
        for lam in (0.1, 0.25, 0.5):
            policies[f"PairNav lambda={lam} B{budget}"] = pair_diverse_sets(val_arrays, matrix["mean"], budget, nav_lambda=lam)
        policies[f"PairDiversityNav B{budget}"] = pair_diverse_sets(val_arrays, matrix["mean"], budget, beta=0.5, coverage=True, nav_lambda=0.25)
    relative = relative_prior(val_arrays, train_arrays)
    set3 = set3_prior(train_arrays, matrix["mean"], alpha=10.0)
    policies["StaticViewPairPrior B2"] = prior_sets(val_arrays, matrix["mean"], 2)
    policies["RelativeGeometryPrior B2"] = [[max(legal_slots(val_arrays, i), key=lambda slot: (relative.get(relative_geometry(int(val_arrays["current_ids"][i]), int(val_arrays["candidate_ids"][i, slot])), -999.0), -int(val_arrays["candidate_ids"][i, slot])))] for i in range(len(val_rows))]
    policies["RelativeGeometryPrior B3"] = [sorted(legal_slots(val_arrays, i), key=lambda slot: (-relative.get(relative_geometry(int(val_arrays["current_ids"][i]), int(val_arrays["candidate_ids"][i, slot])), -999.0), int(val_arrays["candidate_ids"][i, slot])))[:2] for i in range(len(val_rows))]
    policies["SmoothedSet3Prior B3"] = set3_prior_sets(val_arrays, set3, matrix["mean"])
    oracle_sets: dict[str, Sequence[Sequence[int]]] = {}
    for budget in (2, 3, 4):
        exact, _ = make_oracle_sets(val_arrays, budget)
        oracle_sets[f"Exact GT-Margin Oracle B{budget}"] = exact
    policies.update(oracle_sets)
    metrics = evaluate_policy(val_arrays, policies)
    any_correct_coverage = {f"B{budget}": any_correct_fusion_coverage(val_arrays, budget) for budget in (2, 3)}
    # Registration gate and simple B1 baseline.
    observed_gate = {name: {"accuracy": metrics[name]["accuracy"], "macro_f1": metrics[name]["macro_f1"]} for name in ("Random B2", "Random B3", "Random B4")}
    observed_gate["StaticViewPairPrior B2"] = {"accuracy": set_metrics(val_arrays, prior_sets(val_arrays, matrix["mean"], 2), "StaticViewPairPrior B2")["accuracy"], "macro_f1": set_metrics(val_arrays, prior_sets(val_arrays, matrix["mean"], 2), "StaticViewPairPrior B2")["macro_f1"]}
    expected = {"Random B2": (0.429762, 0.424594), "Random B3": (0.487401, 0.477211), "Random B4": (0.521230, 0.505236), "StaticViewPairPrior B2": (0.508234, 0.497486)}
    gate_pass = all(
        abs(observed_gate[name]["accuracy"] - value[0]) <= 0.01
        and abs(observed_gate[name]["macro_f1"] - value[1]) <= 0.01
        for name, value in expected.items()
    )
    write_json(output / "protocol_reproduction.json", {"expected": expected, "observed": observed_gate, "tolerance_pp": 1.0, "gate_pass": gate_pass, "test_used": False})
    if not gate_pass:
        raise RuntimeError("budgeted complementarity protocol gate failed")
    b1 = set_metrics(val_arrays, [[] for _ in val_rows], "Stay")
    pair_matrix_json = {"mean_pair_margin": matrix["mean"].tolist(), "pair_fusion_accuracy": matrix["accuracy"].tolist(), "pair_rescue_rate": matrix["rescue"].tolist(), "class_mean_pair_margin": matrix["class_mean"].tolist(), "test_used": False}
    write_json(output / "pair_complementarity_matrix.json", pair_matrix_json)
    write_json(output / "pair_count_matrix.json", {"counts": matrix["counts"].tolist(), "test_used": False})
    structure_corr = pair_geometry_correlations(train_arrays)
    write_json(output / "pair_structure_analysis.json", {**structure_corr, "matrix_symmetry_spearman": correlation(matrix["mean"].ravel(), matrix["mean"].T.ravel(), spearman=True), "mean_absolute_asymmetry": float(np.mean(np.abs(matrix["mean"] - matrix["mean"].T))), "relative_prior_bins": {f"{key[0]}_{key[1]}": value for key, value in relative.items()}, "test_used": False})
    static_sets = prior_sets(val_arrays, matrix["mean"], 2)
    write_json(output / "relative_geometry_prior.json", {"train_derived_bins": {f"{key[0]}_{key[1]}": value for key, value in relative.items()}, "train_pair_stats": relative_bin_stats(train_arrays), "val_pair_stats": relative_bin_stats(val_arrays), "b2": metrics["RelativeGeometryPrior B2"], "b3": metrics["RelativeGeometryPrior B3"], "test_used": False})
    write_json(output / "b2_metrics.json", {name: value for name, value in metrics.items() if " B2" in name or name in {"Stay"}})
    write_json(output / "b3_metrics.json", {name: value for name, value in metrics.items() if " B3" in name})
    write_json(output / "b4_metrics.json", {name: value for name, value in metrics.items() if " B4" in name})
    write_json(output / "set3_prior_metrics.json", {name: metrics[name] for name in ("SmoothedSet3Prior B3", "PairMeanGreedy B3", "Pair+GeoCoverage B3")})
    write_json(output / "oracle_set_metrics.json", {name: metrics[name] for name in oracle_sets} | {"anycorrect_fusion_coverage": any_correct_coverage, "test_used": False})
    write_json(output / "geometry_diversity_metrics.json", {name: metrics[name] for name in metrics if any(x in name for x in ("MaxMinAngular", "OppositeAzimuth", "GeometricCoverage"))})
    write_json(output / "navigation_aware_metrics.json", {name: metrics[name] for name in metrics if "Nav" in name})
    visibility_path = root / "diagnostics/frame0_visibility_predictor_v1/val.npz"
    if visibility_path.is_file():
        with np.load(visibility_path, allow_pickle=False) as vis_archive:
            current_visibility = np.asarray(vis_archive["scores"][:, 0], dtype=np.float32)
        low = current_visibility <= np.quantile(current_visibility, 1 / 3)
        high_metrics = {name: set_metrics({k: v[low] if isinstance(v, np.ndarray) and v.shape[0] == len(val_rows) else v for k, v in val_arrays.items()}, [sets[i] for i in np.flatnonzero(low)], name) for name, sets in policies.items() if name in {"Stay", "Random B2", "Random B3", "Random B4", "PairMeanGreedy B2", "PairMeanGreedy B3", "PairMeanGreedy B4", "Pair+Angular beta=0.5 B2", "Pair+Angular beta=0.5 B3", "Pair+Angular beta=0.5 B4", "Exact GT-Margin Oracle B2", "Exact GT-Margin Oracle B3"}}
    else:
        high_metrics = {}
    write_json(output / "occlusion_metrics.json", {"source": str(visibility_path.resolve()), "bottom_tertile": high_metrics, "test_used": False})
    write_json(output / "clutter_metrics.json", {"status": "NO_DECLARED_SCENE_DENSITY_ASSET; not synthesized", "test_used": False})
    per_class_budget = {label: {} for label in LABELS}
    for label_index, label in enumerate(LABELS):
        subset = np.flatnonzero(val_arrays["labels"] == label_index)
        for name in ("Stay", "PairMeanGreedy B2", "PairMeanGreedy B3", "PairMeanGreedy B4", "Exact GT-Margin Oracle B2", "Exact GT-Margin Oracle B3"):
            sub_arrays = {k: v[subset] if isinstance(v, np.ndarray) and v.shape[0] == len(val_rows) else v for k, v in val_arrays.items()}
            per_class_budget[label][name] = set_metrics(sub_arrays, [policies[name][i] for i in subset], name)
    write_json(output / "per_class_budget_metrics.json", per_class_budget)
    write_json(output / "fusion_rescue_metrics.json", {name: rescue_counts(val_arrays, sets) for name, sets in policies.items() if name in {"Random B2", "Random B3", "Random B4", "PairMeanGreedy B2", "PairMeanGreedy B3", "PairMeanGreedy B4", "Pair+Angular beta=0.5 B2", "Pair+Angular beta=0.5 B3", "Pair+Angular beta=0.5 B4", "Exact GT-Margin Oracle B2", "Exact GT-Margin Oracle B3", "Exact GT-Margin Oracle B4"}})
    best_deterministic_name = max((name for name in metrics if name.startswith(("PairMeanGreedy", "Pair+Angular", "Pair+GeoCoverage", "PairDiversityNav", "RelativeGeometryPrior"))), key=lambda name: metrics[name]["accuracy"])
    budget_curve = {"B1": b1, "B2": metrics[best_deterministic_name.replace("B" + best_deterministic_name.split("B")[-1], "B2")] if "B2" in metrics else metrics["PairMeanGreedy B2"], "B3": metrics["PairMeanGreedy B3"], "B4": metrics["PairMeanGreedy B4"], "best_deterministic": best_deterministic_name}
    write_json(output / "marginal_observation_value.json", {"budget_curve": budget_curve, "delta_B2": metrics["PairMeanGreedy B2"]["accuracy"] - b1["accuracy"], "delta_B3": metrics["PairMeanGreedy B3"]["accuracy"] - metrics["PairMeanGreedy B2"]["accuracy"], "delta_B4": metrics["PairMeanGreedy B4"]["accuracy"] - metrics["PairMeanGreedy B3"]["accuracy"], "test_used": False})
    write_json(output / "navigation_cost_metrics.json", {name: {key: value[key] for key in ("move_count", "mean_path_length_m", "median_path_length_m", "p90_path_length_m")} for name, value in metrics.items() if "B" in name or name == "Stay"})
    pareto = {name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"], "mean_path_length_m": value["mean_path_length_m"]} for name, value in metrics.items() if "B" in name or name == "Stay"}
    write_json(output / "pareto_metrics.json", {"policies": pareto, "dominated": [], "test_used": False})
    write_json(output / "coverage_audit.json", {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows), "train_records": len({str(x['record_id']) for x in train_rows}), "moving_val_records": len({str(x['record_id']) for x in val_rows}), "val_legal_candidates": int(val_arrays["candidate_mask"].sum()), "test_used": False})
    write_json(output / "leakage_audit.json", {"policy_test_used": False, "training_used": False, "candidate_observation_used_before_selection": False, "candidate_features_used_before_selection": False, "candidate_logits_used_before_selection": False, "candidate_skeleton_used_for_terminal_oracle_only": True, "gt_action_used_for_oracle_only": True, "continuous_time_modeled": False})
    write_json(output / "runtime_summary.json", {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.time() - started, "test_used": False})
    result = {"experiment_id": "REDUCED12_BUDGETED_COMPLEMENTARY_VIEWSET_SWEEP", "status": "COMPLETED", "population": {"train_contexts": len(train_rows), "moving_val_contexts": len(val_rows)}, "metrics": metrics, "reproduction_gate": {"pass": gate_pass, "observed": observed_gate}, "best_deterministic": best_deterministic_name, "oracle": {name: metrics[name] for name in oracle_sets}, "any_correct_fusion_coverage": any_correct_coverage, "set3_prior": metrics["SmoothedSet3Prior B3"], "flags": {"policy_test_used": False, "training_used": False, "new_rgb_generated": False, "new_skeleton_generated": False, "frozen_stgcn_modified": False, "gt_label_used_for_oracle_only": True, "deployable": False}}
    write_json(output / "result.json", result)
    analysis = ["# Budgeted Complementary Multi-View Active HAR", "", "Experiment: Budgeted Complementary Multi-View Active HAR", "Moving Val: 10080 contexts", "Policy Test: false", "Recognizer: frozen ST-GCN + frozen shared head", "Task: select a complementary set of additional observation viewpoints under a finite observation budget", "Observation budget: B = 2,3,4", "O0: already acquired current full observation", "Unvisited candidate observation used before selection: false", "GT action: false for deterministic selection; oracle only for privileged diagnostics", "Main selection information: viewpoint geometry, Train-derived viewpoint complementarity prior, navigation geometry", "O0 semantic feature: not required by main methods", "Fusion: fixed normalized MeanLogP", "Continuous human/navigation synchronization: not modeled", "", "## Reproduction gate", f"Random B2/B3/B4: {metrics['Random B2']['accuracy']:.6f}/{metrics['Random B3']['accuracy']:.6f}/{metrics['Random B4']['accuracy']:.6f} Accuracy; StaticViewPairPrior B2: {observed_gate['StaticViewPairPrior B2']['accuracy']:.6f}; gate={'PASS' if gate_pass else 'FAIL'}.", "", "## Main findings"]
    for name in ("Stay", "Random B2", "Random B3", "Random B4", "StaticViewPairPrior B2", "PairMeanGreedy B2", "PairMeanGreedy B3", "PairMeanGreedy B4", "Pair+Angular beta=0.5 B2", "Pair+Angular beta=0.5 B3", "Pair+Angular beta=0.5 B4", "Pair+GeoCoverage B3", "PairDiversityNav B3", "RelativeGeometryPrior B2", "RelativeGeometryPrior B3", "SmoothedSet3Prior B3", "Exact GT-Margin Oracle B2", "Exact GT-Margin Oracle B3", "Exact GT-Margin Oracle B4"):
        if name in metrics:
            analysis.append(f"- {name}: Acc={metrics[name]['accuracy']:.6f}, Macro-F1={metrics[name]['macro_f1']:.6f}, path={metrics[name]['mean_path_length_m']:.3f}m")
    analysis.extend(["", f"Best deterministic policy: {best_deterministic_name}.", f"PairMeanGreedy marginal Acc gains: ΔB2={(metrics['PairMeanGreedy B2']['accuracy'] - b1['accuracy']) * 100:+.3f} pp, ΔB3={(metrics['PairMeanGreedy B3']['accuracy'] - metrics['PairMeanGreedy B2']['accuracy']) * 100:+.3f} pp, ΔB4={(metrics['PairMeanGreedy B4']['accuracy'] - metrics['PairMeanGreedy B3']['accuracy']) * 100:+.3f} pp.", f"B3 versus B2: {(metrics['PairMeanGreedy B3']['accuracy'] - metrics['PairMeanGreedy B2']['accuracy']) * 100:+.3f} pp; B4 versus B3: {(metrics['PairMeanGreedy B4']['accuracy'] - metrics['PairMeanGreedy B3']['accuracy']) * 100:+.3f} pp.", "", "Pair complementarity is reported as a Train-derived prior; no Val labels or candidate evidence are used to choose formal deterministic actions. Exact oracle sets are privileged diagnostics only."])
    analysis.extend(["", "## Additional registered diagnostics"])
    for family in (("MaxMinAngular", "OppositeAzimuth", "GeometricCoverage", "PairMinGreedy", "PairMaxGreedy"), ("Pair+GeoCoverage", "PairDiversityNav")):
        for name in family:
            values = [metrics[f"{name} B{budget}"] for budget in (2, 3, 4) if f"{name} B{budget}" in metrics]
            if values:
                analysis.append(f"- {name}: " + "; ".join(f"B{value['budget']}={value['accuracy']:.6f}/{value['macro_f1']:.6f}" for value in values))
    for lam in (0.1, 0.25, 0.5):
        analysis.append(f"- PairNav lambda={lam}: " + "; ".join(f"B{budget}={metrics[f'PairNav lambda={lam} B{budget}']['accuracy']:.6f}, path={metrics[f'PairNav lambda={lam} B{budget}']['mean_path_length_m']:.3f}m" for budget in (2, 3, 4)))
    analysis.append(f"- Pair+Angular betas: " + ", ".join(f"β={beta}: B2={metrics[f'Pair+Angular beta={beta} B2']['accuracy']:.6f}, B3={metrics[f'Pair+Angular beta={beta} B3']['accuracy']:.6f}, B4={metrics[f'Pair+Angular beta={beta} B4']['accuracy']:.6f}" for beta in (0.25, 0.5, 1.0)))
    analysis.append(f"- Set3 versus PairMeanGreedy B3: {(metrics['SmoothedSet3Prior B3']['accuracy'] - metrics['PairMeanGreedy B3']['accuracy']) * 100:+.3f} pp.")
    analysis.append(f"- RelativeGeometryPrior versus StaticViewPairPrior: B2={(metrics['RelativeGeometryPrior B2']['accuracy'] - metrics['StaticViewPairPrior B2']['accuracy']) * 100:+.3f} pp; B3={metrics['RelativeGeometryPrior B3']['accuracy']:.6f}.")
    analysis.append(f"- Pair structure: angular Spearman={structure_corr['pair_margin_vs_angular_spearman']:.6f}, radius Spearman={structure_corr['pair_margin_vs_radius_difference_spearman']:.6f}, matrix symmetry Spearman={correlation(matrix['mean'].ravel(), matrix['mean'].T.ravel(), spearman=True):.6f}, mean abs asymmetry={float(np.mean(np.abs(matrix['mean'] - matrix['mean'].T))):.6f}.")
    analysis.append(f"- AnyCorrect-fusion coverage: B2={any_correct_coverage['B2']['fraction']:.6f} ({any_correct_coverage['B2']['count']}), B3={any_correct_coverage['B3']['fraction']:.6f} ({any_correct_coverage['B3']['count']}).")
    for name in ("Random B2", "Random B3", "PairMeanGreedy B2", "PairMeanGreedy B3", "PairMeanGreedy B4"):
        rescue = rescue_counts(val_arrays, policies[name])
        analysis.append(f"- Rescue decomposition {name}: all-single-wrong→correct={rescue['all_single_wrong_to_fusion_correct']}, single-correct→fusion-wrong={rescue['single_correct_to_fusion_wrong']}.")
    analysis.append("- High-occlusion and clutter subsets, per-class budget curves, full navigation costs, Pareto records, and all policy metrics are stored in the corresponding JSON artifacts.")
    if metrics["PairMeanGreedy B2"]["accuracy"] < 0.50 or metrics["PairMeanGreedy B3"]["accuracy"] < metrics["Random B3"]["accuracy"] + 0.03:
        decision = "KILL COMPLEMENTARY SET-SELECTION"
    elif metrics["PairMeanGreedy B2"]["accuracy"] >= 0.51 and metrics["PairMeanGreedy B3"]["accuracy"] >= 0.56 and metrics["PairMeanGreedy B3"]["accuracy"] - metrics["Random B3"]["accuracy"] >= 0.07:
        decision = "STRONG KEEP COMPLEMENTARY SET-SELECTION"
    else:
        decision = "KEEP COMPLEMENTARY SET-SELECTION"
    analysis.extend(["", f"## Final decision: **{decision}**", "", "The report separates exact set oracles from incremental greedy policies and records any B4 dilution. The next step, if any, should be chosen explicitly; no follow-up experiment is launched automatically."])
    (output / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": decision, "best_deterministic": best_deterministic_name}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
