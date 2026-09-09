#!/usr/bin/env python3
"""Val-only top-down navmesh LOS diagnostic for reduced12.

This is a privileged geometry audit.  It uses Habitat's top-down navmesh
slice only to rank already archived legal viewpoints; no perception artifact
or model is regenerated.  Terminal labels are read from the existing frozen
ST-GCN candidate cache.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.data.preprocessing.cache import load_jsonl

NUM_CLASSES = 12
SEED = 42
LOS_RESOLUTION_M = 0.20
LABELS = (
    "walk", "sit", "stand up", "bend", "crawl", "stumble", "clap",
    "throw", "kick", "knock", "punch", "touching face",
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/topdown_los_nbv_oracle"
)
DEFAULT_POLICY_ROOT = "datasets/policy_reduced12_eight_placement_v1"
DEFAULT_TRUE_CACHE = (
    "diagnostics/reduced12_h1_discriminative_objective_batch/"
    "val_all_candidate_true_logp.npz"
)
ARCHIVE_ROOT_REL = "datasets/offline/habitat-train/00006-00087"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.int64)
    target = np.asarray(labels, dtype=np.int64)
    if pred.shape != target.shape or pred.ndim != 1:
        raise ValueError("predictions and labels must be aligned one-dimensional arrays")
    if target.size and ((target < 0).any() or (target >= NUM_CLASSES).any()):
        raise ValueError("target label outside reduced12 range")
    if pred.size and ((pred < 0).any() or (pred >= NUM_CLASSES).any()):
        raise ValueError("prediction outside reduced12 range")
    matrix = np.bincount(
        target * NUM_CLASSES + pred,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for class_id, name in enumerate(LABELS):
        tp = float(matrix[class_id, class_id])
        support = float(matrix[class_id].sum())
        predicted = float(matrix[:, class_id].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        f1_values.append(f1)
        per_class[name] = {
            "label_id": class_id,
            "count": int(support),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return {
        "count": int(target.size),
        "accuracy": float(np.mean(pred == target)) if target.size else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _archive_path(archive_root: Path, row: Mapping[str, Any]) -> Path:
    scene = str(row["scene_id"])
    placement = str(row.get("placement_id") or row.get("region"))
    record = str(row["record_id"])
    path = archive_root / scene / placement / f"{record}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing archived skeleton: {path}")
    return path


def _scene_file(scene_root: Path, scene_id: str, suffix: str) -> Path:
    directory = scene_root / scene_id
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {suffix} asset for {scene_id}, found {len(matches)}"
        )
    return matches[0]


def _make_sim(scene_root: Path, scene_id: str) -> Any:
    # Imported lazily so metadata-only imports/py_compile do not require Habitat.
    import habitat_sim

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(_scene_file(scene_root, scene_id, ".basis.glb"))
    config.enable_physics = False
    agent = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    if not sim.pathfinder.is_loaded:
        navmesh = _scene_file(scene_root, scene_id, ".basis.navmesh")
        if not sim.pathfinder.load_nav_mesh(str(navmesh)):
            sim.close()
            raise RuntimeError(f"failed to load navmesh for {scene_id}")
    return sim


def _grid_coordinates(
    point: np.ndarray,
    lower: np.ndarray,
    resolution: float,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    column = int(np.floor((float(point[0]) - float(lower[0])) / resolution))
    row = int(np.floor((float(point[2]) - float(lower[2])) / resolution))
    if row < 0 or column < 0 or row >= shape[0] or column >= shape[1]:
        return None
    return row, column


def _line_of_sight(
    start: np.ndarray,
    end: np.ndarray,
    grid: np.ndarray,
    lower: np.ndarray,
    resolution: float,
) -> bool:
    distance = float(np.linalg.norm((end - start)[[0, 2]]))
    steps = max(1, int(np.ceil(distance / (resolution * 0.5))))
    points = np.linspace(start, end, steps + 1)
    for point in points:
        cell = _grid_coordinates(point, lower, resolution, grid.shape)
        if cell is None or not bool(grid[cell]):
            return False
    return True


def _topdown_grid(
    sim: Any,
    floor_y: float,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    lower, _ = sim.pathfinder.get_bounds()
    lower_array = np.asarray(lower, dtype=np.float64)
    grid = np.asarray(
        sim.pathfinder.get_topdown_view(resolution, float(floor_y), eps=0.5),
        dtype=bool,
    )
    if grid.ndim != 2 or not grid.any():
        raise RuntimeError(
            f"empty top-down navmesh slice at floor height {floor_y:.3f}"
        )
    return grid, lower_array, float(grid.mean())


def _collect_bundles(
    data_root: Path,
    stage_c_rows: Sequence[Mapping[str, Any]],
    moving_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    scene_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cache["current_logp"].shape[0] != len(stage_c_rows):
        raise ValueError("true-logp cache is not positionally aligned with Stage-C rows")
    stage_c_index = {str(row["episode_id"]): i for i, row in enumerate(stage_c_rows)}
    if len(stage_c_index) != len(stage_c_rows):
        raise ValueError("duplicate Stage-C episode_id")
    grouped: dict[str, list[tuple[int, Mapping[str, Any], int]]] = defaultdict(list)
    bundles: list[dict[str, Any] | None] = [None] * len(moving_rows)
    for moving_index, row in enumerate(moving_rows):
        episode_id = str(row["episode_id"])
        if episode_id not in stage_c_index:
            raise ValueError(f"moving episode missing from Stage-C rows: {episode_id}")
        cache_index = stage_c_index[episode_id]
        valid_slots = np.flatnonzero(np.asarray(cache["candidate_mask"][cache_index], dtype=bool))
        candidate_ids = np.asarray(cache["candidate_ids"][cache_index], dtype=np.int64)[valid_slots]
        stage_ids = np.asarray(stage_c_rows[cache_index]["candidate_viewpoint_ids"], dtype=np.int64)
        if set(candidate_ids.tolist()) != set(stage_ids.tolist()):
            raise ValueError(f"candidate identity mismatch for {episode_id}")
        grouped[str(row["scene_id"])].append((moving_index, row, cache_index))

    archive_root = data_root / ARCHIVE_ROOT_REL
    scene_grid_info: dict[str, Any] = {}
    for scene_id, scene_items in sorted(grouped.items()):
        sim = _make_sim(scene_root, scene_id)
        grids: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
        try:
            for moving_index, row, cache_index in scene_items:
                archive_data = _load_npz(_archive_path(archive_root, row))
                viewpoint_ids = np.asarray(archive_data["viewpoint_ids"], dtype=np.int64)
                positions = np.asarray(archive_data["viewpoint_agent_positions"], dtype=np.float64)
                human = np.asarray(archive_data["placement_position"], dtype=np.float64)
                if viewpoint_ids.shape != (32,) or positions.shape != (32, 3):
                    raise ValueError(f"invalid viewpoint position schema for {row['episode_id']}")
                if human.shape != (3,) or not np.isfinite(human).all():
                    raise ValueError(f"invalid placement position for {row['episode_id']}")
                floor_key = round(float(human[1]), 3)
                if floor_key not in grids:
                    grids[floor_key] = _topdown_grid(sim, floor_key, LOS_RESOLUTION_M)
                grid, lower, occupancy = grids[floor_key]
                slots = np.flatnonzero(np.asarray(cache["candidate_mask"][cache_index], dtype=bool))
                ids = np.asarray(cache["candidate_ids"][cache_index], dtype=np.int64)[slots]
                candidate_positions: list[np.ndarray] = []
                los_values: list[int] = []
                for viewpoint_id in ids.tolist():
                    matches = np.flatnonzero(viewpoint_ids == int(viewpoint_id))
                    if matches.size != 1:
                        raise ValueError(
                            f"viewpoint alignment failure for {row['episode_id']}: {viewpoint_id}"
                        )
                    candidate = positions[int(matches[0])]
                    candidate_positions.append(candidate)
                    los_values.append(int(_line_of_sight(candidate, human, grid, lower, LOS_RESOLUTION_M)))
                bundles[moving_index] = {
                    "row": row,
                    "cache_index": cache_index,
                    "slots": slots,
                    "candidate_ids": ids,
                    "candidate_geodesic": np.asarray(cache["candidate_geodesic"][cache_index], dtype=np.float64)[slots],
                    "candidate_los": np.asarray(los_values, dtype=np.int64),
                    "grid_occupancy": occupancy,
                }
        finally:
            sim.close()
        scene_grid_info[scene_id] = {
            "floor_slices": len(grids),
            "occupancy_mean": float(np.mean([item[2] for item in grids.values()])),
            "resolution_m": LOS_RESOLUTION_M,
        }
    if any(bundle is None for bundle in bundles):
        raise RuntimeError("failed to build a top-down bundle for one or more contexts")
    return [bundle for bundle in bundles if bundle is not None], scene_grid_info


def _select_and_diagnose(
    moving_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, np.ndarray],
    bundles: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    predictions = {
        name: np.empty(labels.size, dtype=np.int64)
        for name in ("S0-only", "FrozenStageCv0", "Random", "TopDown-LOS", "AnyCorrect Oracle")
    }
    moved = {name: np.zeros(labels.size, dtype=bool) for name in predictions}
    rng = np.random.default_rng(SEED)
    correct_los_one = correct_los_zero = wrong_los_one = wrong_los_zero = 0
    context_fractions: list[float] = []
    all_same_count = 0
    topdown_selected_los: list[int] = []
    for i, (row, bundle) in enumerate(zip(moving_rows, bundles)):
        cache_index = int(bundle["cache_index"])
        current_logp = np.asarray(cache["current_logp"][cache_index], dtype=np.float64)
        slots = np.asarray(bundle["slots"], dtype=np.int64)
        candidate_logp = np.asarray(cache["candidate_logp"][cache_index, slots], dtype=np.float64)
        candidate_pred = np.argmax(candidate_logp, axis=1)
        candidate_los = np.asarray(bundle["candidate_los"], dtype=np.int64)
        candidate_geodesic = np.asarray(bundle["candidate_geodesic"], dtype=np.float64)
        label = int(labels[i])
        s0_prediction = int(np.argmax(current_logp))
        predictions["S0-only"][i] = s0_prediction
        # The archived s1 prediction is the frozen Stage-C-v0 action result.
        s1_logp = np.asarray(row["s1_feature"], dtype=np.float64)[256:256 + NUM_CLASSES]
        if s1_logp.shape != (NUM_CLASSES,):
            raise ValueError(f"invalid s1 feature/logp schema for {row['episode_id']}")
        predictions["FrozenStageCv0"][i] = int(np.argmax(s1_logp))
        moved["FrozenStageCv0"][i] = True
        choice = int(rng.integers(0, slots.size + 1)) if slots.size else 0
        if choice == 0:
            predictions["Random"][i] = s0_prediction
        else:
            predictions["Random"][i] = int(candidate_pred[choice - 1])
            moved["Random"][i] = True
        if slots.size:
            order = sorted(
                range(slots.size),
                key=lambda j: (-int(candidate_los[j]), float(candidate_geodesic[j]), int(bundle["candidate_ids"][j])),
            )
            topdown_index = order[0]
            predictions["TopDown-LOS"][i] = int(candidate_pred[topdown_index])
            moved["TopDown-LOS"][i] = True
            topdown_selected_los.append(int(candidate_los[topdown_index]))
        else:
            predictions["TopDown-LOS"][i] = s0_prediction
        candidate_correct = candidate_pred == label
        context_fractions.append(float(np.mean(candidate_los)) if candidate_los.size else 0.0)
        if candidate_los.size and np.all(candidate_los == candidate_los[0]):
            all_same_count += 1
        correct_los_one += int(np.sum((candidate_los == 1) & candidate_correct))
        correct_los_zero += int(np.sum((candidate_los == 0) & candidate_correct))
        wrong_los_one += int(np.sum((candidate_los == 1) & ~candidate_correct))
        wrong_los_zero += int(np.sum((candidate_los == 0) & ~candidate_correct))
        if s0_prediction == label:
            predictions["AnyCorrect Oracle"][i] = label
        else:
            correct = np.flatnonzero(candidate_correct)
            if correct.size:
                predictions["AnyCorrect Oracle"][i] = label
                moved["AnyCorrect Oracle"][i] = True
            else:
                predictions["AnyCorrect Oracle"][i] = s0_prediction

    los_one = correct_los_one + wrong_los_one
    los_zero = correct_los_zero + wrong_los_zero
    diagnostics = {
        "context_los1_fraction": {
            "mean": float(np.mean(context_fractions)) if context_fractions else 0.0,
            "median": float(np.median(context_fractions)) if context_fractions else 0.0,
            "count": len(context_fractions),
        },
        "candidate_correctness_by_los": {
            "los1_count": los_one,
            "los0_count": los_zero,
            "correct_los1_count": correct_los_one,
            "correct_los0_count": correct_los_zero,
            "wrong_los1_count": wrong_los_one,
            "wrong_los0_count": wrong_los_zero,
            "p_correct_given_los1": float(correct_los_one / los_one) if los_one else 0.0,
            "p_correct_given_los0": float(correct_los_zero / los_zero) if los_zero else 0.0,
            "los1_fraction_among_correct": float(correct_los_one / (correct_los_one + correct_los_zero)) if correct_los_one + correct_los_zero else 0.0,
            "los1_fraction_among_wrong": float(wrong_los_one / (wrong_los_one + wrong_los_zero)) if wrong_los_one + wrong_los_zero else 0.0,
        },
        "contexts_all_candidate_los_equal": {
            "count": all_same_count,
            "rate": float(all_same_count / len(moving_rows)) if moving_rows else 0.0,
        },
        "topdown_selected_los1_rate": float(np.mean(topdown_selected_los)) if topdown_selected_los else 0.0,
        "candidate_count": int(los_one + los_zero),
        "legal_candidates_per_context": float((los_one + los_zero) / len(moving_rows)) if moving_rows else 0.0,
    }
    move_rates = {
        name: float(np.mean(values))
        for name, values in moved.items()
    }
    return predictions, diagnostics, move_rates


def _write_analysis(result: Mapping[str, Any], path: Path) -> None:
    methods = result["methods"]
    frozen = methods["FrozenStageCv0"]
    topdown = methods["TopDown-LOS"]
    diag = result["diagnostics"]
    los = diag["candidate_correctness_by_los"]
    same = diag["contexts_all_candidate_los_equal"]
    gain_acc = 100.0 * (float(topdown["accuracy"]) - float(frozen["accuracy"]))
    gain_f1 = 100.0 * (float(topdown["macro_f1"]) - float(frozen["macro_f1"]))
    discrimination = float(los["p_correct_given_los1"] - los["p_correct_given_los0"])
    lines = [
        "# Reduced12 Top-down LOS NBV Oracle",
        "",
        "Val Moving contexts only. This is a privileged geometry diagnostic: Habitat GT navmesh top-down slices rank archived legal candidates. It is not a joint-visibility measurement and no model, perception cache, RGB, skeleton, DINO or Test artifact was generated/read.",
        "",
        "## Moving metrics",
        "",
        "| Method | Accuracy | Macro-F1 | ΔAcc vs Frozen (pp) | ΔF1 vs Frozen (pp) | Move rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("S0-only", "FrozenStageCv0", "Random", "TopDown-LOS", "AnyCorrect Oracle"):
        metric = methods[name]
        move_rate = result["move_rates"][name]
        lines.append(
            f"| {name} | {metric['accuracy']:.6f} | {metric['macro_f1']:.6f} | "
            f"{100.0 * (float(metric['accuracy']) - float(frozen['accuracy'])):+.3f} | "
            f"{100.0 * (float(metric['macro_f1']) - float(frozen['macro_f1'])):+.3f} | {move_rate:.6f} |"
        )
    lines.extend([
        "",
        "## LOS diagnostics",
        "",
        f"Mean legal-candidate LOS=1 fraction: {diag['context_los1_fraction']['mean']:.6f}; median: {diag['context_los1_fraction']['median']:.6f}.",
        f"P(correct | LOS=1) = {los['p_correct_given_los1']:.6f}; P(correct | LOS=0) = {los['p_correct_given_los0']:.6f}; difference = {discrimination:+.6f}.",
        f"{same['count']} / {same['rate']:.6f} of contexts have identical LOS for all legal candidates.",
        "",
        "## Interpretation",
    ])
    if gain_acc >= 1.0 or gain_f1 >= 1.0:
        lines.append(f"TopDown-LOS shows a material gain over the frozen baseline ({gain_acc:+.3f} pp Accuracy, {gain_f1:+.3f} pp Macro-F1); a finer multi-height LOS diagnostic is warranted before a deployable predictor.")
    else:
        lines.append(f"TopDown-LOS does not materially exceed FrozenStageCv0 ({gain_acc:+.3f} pp Accuracy, {gain_f1:+.3f} pp Macro-F1).")
    if discrimination >= 0.02:
        lines.append("LOS separates correct from wrong archived candidates to a noticeable degree, although it remains a privileged cue.")
    else:
        lines.append("LOS has weak candidate-level discrimination between correct and wrong archived ST-GCN predictions.")
    if same["rate"] >= 0.5:
        lines.append("Because at least half of contexts have identical LOS, the single-floor top-down navmesh signal is too coarse for reliable NBV ranking.")
    else:
        lines.append("Most contexts have at least some LOS variation, so the coarse signal is not entirely degenerate; its policy value is nevertheless limited by the measured gain.")
    lines.extend([
        "",
        "All oracle quantities use only archived terminal recognition for evaluation. `test_used=false`, `training_used=false`, `future_candidate_rgb_used=false`, `future_candidate_skeleton_used_only_for_terminal_evaluation=true`, and `habitat_gt_geometry_used_for_oracle_diagnostic=true`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    data_root: Path,
    *,
    scene_root: Path,
    stage_c_path: Path,
    stage_d_path: Path,
    true_cache_path: Path,
) -> dict[str, Any]:
    stage_c_rows = load_jsonl(stage_c_path)
    moving_rows = load_jsonl(stage_d_path)
    if not stage_c_path.as_posix().endswith("/val.jsonl") or not stage_d_path.as_posix().endswith("/val.jsonl"):
        raise ValueError("this diagnostic is Val-only")
    cache = _load_npz(true_cache_path)
    required = {"current_logp", "candidate_logp", "candidate_mask", "candidate_ids", "candidate_geodesic"}
    if not required.issubset(cache):
        raise ValueError(f"true cache missing keys: {sorted(required - set(cache))}")
    bundles, scene_grid_info = _collect_bundles(data_root, stage_c_rows, moving_rows, cache, scene_root)
    predictions, diagnostics, move_rates = _select_and_diagnose(
        moving_rows, cache, bundles
    )
    labels = np.asarray([int(row["label_id"]) for row in moving_rows], dtype=np.int64)
    methods = {name: _metrics(prediction, labels) for name, prediction in predictions.items()}
    result = {
        "experiment_id": "REDUCED12_TOPDOWN_LOS_NBV_ORACLE",
        "status": "COMPLETED",
        "population": {
            "val_moving_contexts": len(moving_rows),
            "stage_c_val_contexts": len(stage_c_rows),
            "scene_count": len(scene_grid_info),
        },
        "labels": list(LABELS),
        "methods": methods,
        "move_rates": move_rates,
        "diagnostics": diagnostics,
        "scene_grid": scene_grid_info,
        "protocol": {
            "candidate_set": "existing legal candidates from reduced12 Stage-C cache",
            "los_score": "binary top-down navmesh LOS; ties by candidate geodesic then viewpoint id",
            "resolution_m": LOS_RESOLUTION_M,
            "terminal": "selected real archived skeleton through frozen reduced12 ST-GCN cache",
            "seed": SEED,
        },
        "leakage_flags": {
            "test_used": False,
            "training_used": False,
            "future_candidate_rgb_used": False,
            "future_candidate_skeleton_used_only_for_terminal_evaluation": True,
            "habitat_gt_geometry_used_for_oracle_diagnostic": True,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--scene-root", type=Path, default=get_habitat_data_root() / "hm3d-train")
    parser.add_argument("--stage-c", type=Path, default=None)
    parser.add_argument("--stage-d", type=Path, default=None)
    parser.add_argument("--true-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    policy_root = data_root / DEFAULT_POLICY_ROOT
    stage_c_path = (args.stage_c or policy_root / "stage_c/features/val.jsonl").resolve()
    stage_d_path = (args.stage_d or policy_root / "stage_d/features/val.jsonl").resolve()
    true_cache_path = (args.true_cache or data_root / DEFAULT_TRUE_CACHE).resolve()
    result = evaluate(
        data_root,
        scene_root=args.scene_root.resolve(),
        stage_c_path=stage_c_path,
        stage_d_path=stage_d_path,
        true_cache_path=true_cache_path,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(result["diagnostics"], indent=2) + "\n", encoding="utf-8")
    per_class = {name: result["methods"][name]["per_class"] for name in result["methods"]}
    (output_dir / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2) + "\n", encoding="utf-8")
    _write_analysis(result, output_dir / "analysis.md")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
