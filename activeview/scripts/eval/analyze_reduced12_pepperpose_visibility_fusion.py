#!/usr/bin/env python3
"""Audit complementarity between visibility and the frozen PepperPose prior.

The prior table is loaded from the preceding true-facing/frame-0 audit.  This
script performs no RGB generation or training: a deterministic Policy-Train
record holdout chooses the rank-fusion weight, and Moving Val is evaluated once.
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
from activeview.scripts.eval.reduced12_nbv_utils import classification
from activeview.scripts.eval.reduced12_utility_source import load_scene_metadata
from activeview.scripts.experiments.run_reduced12_frame0_visibility_predictor import (
    VisibilityPredictor,
    _option_geometry,
    _predict,
)
from activeview.scripts.experiments.run_reduced12_pepperpose_frame0_confidence_audit import (
    MAX_OPTIONS,
    SEED,
    YAW_DEGREES,
    _angle_scores,
    _candidate_slots,
    _fuse,
    _load_options,
    _load_rows,
    _root_yaw_map,
    _select,
    _terminal,
)
from activeview.scripts.experiments.run_reduced12_yaw8_strict_frame0_rebaseline import (
    _record_holdout,
)
from activeview.scripts.experiments.run_reduced12_yaw8_policy_landscape_audit import (
    _read_npz,
    _signature,
)


OUTPUT_DEFAULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/pepperpose_visibility_fusion_audit"
)
PRIOR_RESULT = REPO_ROOT / (
    "experiments/reduced12_eight_placement_v1/pepperpose_frame0_confidence_audit"
)
POLICY_ROOT = Path("datasets/policy_reduced12_eight_placement_v1")
VISIBILITY_ROOT = Path("diagnostics/frame0_visibility_predictor_v1")
DINO_ROOT = Path("features/dinov2_vitb14_spatial4x4_reduced12_eight_placement/frame0_current")
VISIBILITY_CHECKPOINT = Path(
    "checkpoints/policy_reduced12_eight_placement_v1/frame0_visibility_predictor_v1/"
    "RGBGlobal+Geometry.pth"
)
STATS_PATH = POLICY_ROOT / "stage_c/stage_c_feature_stats.json"
LAMBDA_VALUES = (0.10, 0.25, 0.50)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _load_prior_table() -> np.ndarray:
    landscape = _read_json(PRIOR_RESULT / "internal_confidence_landscape.json")
    values = np.asarray(
        [float(landscape[str(angle)]["mean_frame0_confidence"]) for angle in YAW_DEGREES],
        dtype=np.float64,
    )
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("invalid frozen PepperPose prior table")
    return values


def _load_scene_scores(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    valid_mask: np.ndarray,
) -> np.ndarray:
    root = data_root / VISIBILITY_ROOT
    metadata = _read_json(root / f"{split}.json")
    if int(metadata.get("rows", -1)) != len(rows) or metadata.get("signature") != _signature(rows):
        raise ValueError(f"frame0 visibility cache signature/count mismatch: {split}")
    scores = np.asarray(_read_npz(root / f"{split}.npz")["scores"], dtype=np.float64)
    if scores.shape != (len(rows), MAX_OPTIONS) or not np.isfinite(scores[valid_mask]).all():
        raise ValueError(f"unexpected frame0 visibility score shape: {split} {scores.shape}")
    return scores


def _load_rgb_scores(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    split: str,
    device: torch.device,
) -> np.ndarray:
    stats = _read_json(data_root / STATS_PATH)
    geometry, ids, mask = _option_geometry(rows, stats)
    if not np.array_equal(ids, options["ids"]) or not np.array_equal(mask, options["mask"]):
        raise ValueError(f"candidate geometry/action IDs mismatch: {split}")
    dino_meta = _read_json(data_root / DINO_ROOT / f"{split}.json")
    if int(dino_meta.get("rows", -1)) != len(rows) or dino_meta.get("signature") != _signature(rows):
        raise ValueError(f"frame0 DINO cache signature/count mismatch: {split}")
    if bool(dino_meta.get("test_used", True)) or bool(dino_meta.get("candidate_rgb_used", True)):
        raise ValueError(f"DINO cache is not a strict current-frame cache: {split}")
    tokens = np.load(data_root / DINO_ROOT / f"{split}.npy", mmap_mode="r")
    if tokens.shape != (len(rows), 16, 768):
        raise ValueError(f"unexpected DINO shape: {split} {tokens.shape}")
    model = VisibilityPredictor("RGBGlobal+Geometry").to(device)
    checkpoint = data_root / VISIBILITY_CHECKPOINT
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    scores = np.asarray(_predict(model.eval(), geometry, np.asarray(tokens), device), dtype=np.float64)
    del model, tokens, geometry
    if scores.shape != options["ids"].shape or not np.isfinite(scores).all():
        raise ValueError(f"unexpected RGBGlobal score shape: {split} {scores.shape}")
    return scores


def _slice_options(options: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in options.items()}


def _metric(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    actions: Sequence[int],
    indices: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    selected_rows = rows if indices is None else [rows[int(i)] for i in indices]
    selected_options = options if indices is None else _slice_options(options, indices)
    selected_actions = list(actions) if indices is None else [actions[int(i)] for i in indices]
    labels = np.asarray([int(row["label_id"]) for row in selected_rows], dtype=np.int64)
    predictions = _terminal(selected_options, selected_actions)
    current = np.asarray([int(row["current_viewpoint_id"]) for row in selected_rows], dtype=np.int64)
    result = classification(labels, predictions)
    result.update({
        "method": name,
        "move_rate": float(np.mean(np.asarray(selected_actions, dtype=np.int64) != current)),
        "stay_rate": float(np.mean(np.asarray(selected_actions, dtype=np.int64) == current)),
        "test_used": False,
    })
    return result, predictions


def _lambda_selection(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    primary: np.ndarray,
    confidence: np.ndarray,
    holdout: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    subset_rows = [rows[int(i)] for i in holdout]
    subset_options = _slice_options(options, holdout)
    entries: dict[str, Any] = {}
    for weight in LAMBDA_VALUES:
        scores = _fuse(primary, confidence, options, weight)[holdout]
        actions = _select(subset_rows, subset_options, scores)
        metric, _ = _metric(f"{name}+Confidence lambda={weight:.2f}", subset_rows, subset_options, actions)
        entries[f"{weight:.2f}"] = metric
    best = max(LAMBDA_VALUES, key=lambda weight: (entries[f"{weight:.2f}"]["accuracy"], -weight))
    return float(best), {
        "criterion": "highest HAR Accuracy on deterministic 10% Policy-Train record holdout; ties choose smaller lambda",
        "selected_lambda": float(best),
        "candidates": entries,
        "holdout_contexts": int(holdout.size),
        "holdout_records": len({str(rows[int(i)]["record_id"]) for i in holdout}),
        "test_used": False,
    }


def _pair_audit(
    rows: Sequence[Mapping[str, Any]],
    primary_name: str,
    confidence_name: str,
    primary_actions: Sequence[int],
    confidence_actions: Sequence[int],
    fused_actions: Sequence[int],
    options: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    primary_pred = _terminal(options, primary_actions)
    confidence_pred = _terminal(options, confidence_actions)
    fused_pred = _terminal(options, fused_actions)
    primary_correct = primary_pred == labels
    confidence_correct = confidence_pred == labels
    return {
        "primary": primary_name,
        "confidence": confidence_name,
        "contexts": len(rows),
        "selected_view_agreement_count": int(np.sum(np.asarray(primary_actions) == np.asarray(confidence_actions))),
        "selected_view_agreement_rate": float(np.mean(np.asarray(primary_actions) == np.asarray(confidence_actions))),
        "primary_correct_confidence_wrong": int(np.sum(primary_correct & ~confidence_correct)),
        "primary_wrong_confidence_correct": int(np.sum(~primary_correct & confidence_correct)),
        "both_correct": int(np.sum(primary_correct & confidence_correct)),
        "both_wrong": int(np.sum(~primary_correct & ~confidence_correct)),
        "pair_any_correct_count": int(np.sum(primary_correct | confidence_correct)),
        "pair_any_correct_rate": float(np.mean(primary_correct | confidence_correct)),
        "fused_accuracy": float(np.mean(fused_pred == labels)),
        "test_used": False,
    }


def _disagreement(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    primary_actions: Sequence[int],
    confidence_actions: Sequence[int],
    fused_actions: Sequence[int],
) -> dict[str, Any]:
    mask = np.asarray(primary_actions, dtype=np.int64) != np.asarray(confidence_actions, dtype=np.int64)
    indices = np.flatnonzero(mask)
    primary_metric, _ = _metric("primary", rows, options, primary_actions, indices)
    confidence_metric, _ = _metric("confidence", rows, options, confidence_actions, indices)
    fused_metric, _ = _metric("fused", rows, options, fused_actions, indices)
    return {
        "count": int(indices.size),
        "fraction": float(np.mean(mask)),
        "primary": primary_metric,
        "confidence": confidence_metric,
        "fused": fused_metric,
        "test_used": False,
    }


def _high_occlusion(
    rows: Sequence[Mapping[str, Any]],
    options: Mapping[str, np.ndarray],
    scene_scores: np.ndarray,
    methods: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    stay = np.asarray(scene_scores[:, 0], dtype=np.float64)
    threshold = float(np.quantile(stay, 1.0 / 3.0))
    mask = stay < threshold
    if int(mask.sum()) != 3271:
        raise ValueError(f"fixed high-occlusion subset changed: expected 3271, got {int(mask.sum())}")
    output: dict[str, Any] = {
        "definition": "bottom tertile of frame-0 Stay scene visibility (strict lower tail)",
        "count": int(mask.sum()),
        "threshold": threshold,
        "methods": {},
        "test_used": False,
    }
    indices = np.flatnonzero(mask)
    for name, actions in methods.items():
        output["methods"][name] = _metric(name, rows, options, actions, indices)[0]
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    _seed()
    device = _device(args.device)
    data_root = get_data_root()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    prior = _load_prior_table()
    train_rows, moving_rows = _load_rows(data_root)
    raw_val = _read_json(data_root / "datasets/reduced12_no_kneel_clean_babel_diversity_v1/raw-val/official_val.json")
    # Policy rows intentionally omit the raw motion ``source_path``.  Resolve
    # every policy record against the canonical raw-val manifest, which covers
    # both the Policy-Train contexts and Moving-Val contexts.
    root_yaws = _root_yaw_map(raw_val)
    metadata, metadata_audit = load_scene_metadata(data_root, list(train_rows) + list(moving_rows), audit_sample_count=1)
    train_options, train_cache_meta = _load_options(data_root, train_rows, "train", device)
    moving_options, moving_cache_meta = _load_options(data_root, moving_rows, "moving_val", device)
    train_scene = _load_scene_scores(data_root, train_rows, "train", train_options["mask"])
    moving_scene = _load_scene_scores(data_root, moving_rows, "val", moving_options["mask"])
    train_rgb = _load_rgb_scores(data_root, train_rows, train_options, "train", device)
    moving_rgb = _load_rgb_scores(data_root, moving_rows, moving_options, "val", device)
    train_conf = _angle_scores(train_rows, train_options, metadata, root_yaws, prior)
    moving_conf = _angle_scores(moving_rows, moving_options, metadata, root_yaws, prior)

    _, holdout = _record_holdout(train_rows)
    lambda_rgb, rgb_selection = _lambda_selection("RGBGlobal-Visibility", train_rows, train_options, train_rgb, train_conf, holdout)
    lambda_scene, scene_selection = _lambda_selection("Frame0SceneVisibility", train_rows, train_options, train_scene, train_conf, holdout)

    rgb_actions = _select(moving_rows, moving_options, moving_rgb)
    scene_actions = _select(moving_rows, moving_options, moving_scene)
    confidence_actions = _select(moving_rows, moving_options, moving_conf)
    rgb_fused_scores = _fuse(moving_rgb, moving_conf, moving_options, lambda_rgb)
    scene_fused_scores = _fuse(moving_scene, moving_conf, moving_options, lambda_scene)
    rgb_fused_actions = _select(moving_rows, moving_options, rgb_fused_scores)
    scene_fused_actions = _select(moving_rows, moving_options, scene_fused_scores)
    oracle_scores = np.full(moving_options["ids"].shape, -np.inf, dtype=np.float64)
    for index, row in enumerate(moving_rows):
        label = int(row["label_id"])
        for slot in _candidate_slots(moving_options, index):
            oracle_scores[index, slot] = float(moving_options["logp"][index, slot, label])
    oracle_actions = _select(moving_rows, moving_options, oracle_scores)

    moving_methods = {
        "RGBGlobal-Visibility": rgb_actions,
        "RGBGlobal-Visibility+Confidence": rgb_fused_actions,
        "Frame0SceneVisibility": scene_actions,
        "Frame0SceneVisibility+Confidence": scene_fused_actions,
        "GTYaw-PoseConfidencePrior": confidence_actions,
        "GT-TrueLogP Oracle": oracle_actions,
    }
    moving_metrics = {name: _metric(name, moving_rows, moving_options, actions)[0] for name, actions in moving_methods.items()}
    fusion_metrics = {
        "population": {
            "policy_train_contexts": len(train_rows),
            "policy_train_records": len({str(row["record_id"]) for row in train_rows}),
            "moving_val_contexts": len(moving_rows),
            "moving_val_candidate_samples": int(sum(len(row["candidate_ids"]) for row in moving_rows)),
        },
        "frozen_prior": {
            "source": str((PRIOR_RESULT / "internal_confidence_landscape.json").resolve()),
            "angle_degrees": list(YAW_DEGREES),
            "values": prior.tolist(),
            "recomputed": False,
        },
        "lambda_selection": {"RGBGlobal-Visibility": rgb_selection, "Frame0SceneVisibility": scene_selection},
        "moving_val": {"methods": moving_metrics},
        "gains_vs_baseline": {
            "RGBGlobal-Visibility": {
                "lambda": lambda_rgb,
                "accuracy_gain_pp": 100.0 * (moving_metrics["RGBGlobal-Visibility+Confidence"]["accuracy"] - moving_metrics["RGBGlobal-Visibility"]["accuracy"]),
                "macro_f1_gain_pp": 100.0 * (moving_metrics["RGBGlobal-Visibility+Confidence"]["macro_f1"] - moving_metrics["RGBGlobal-Visibility"]["macro_f1"]),
            },
            "Frame0SceneVisibility": {
                "lambda": lambda_scene,
                "accuracy_gain_pp": 100.0 * (moving_metrics["Frame0SceneVisibility+Confidence"]["accuracy"] - moving_metrics["Frame0SceneVisibility"]["accuracy"]),
                "macro_f1_gain_pp": 100.0 * (moving_metrics["Frame0SceneVisibility+Confidence"]["macro_f1"] - moving_metrics["Frame0SceneVisibility"]["macro_f1"]),
            },
        },
        "cache_audit": {"train_options": train_cache_meta, "moving_options": moving_cache_meta, "scene_metadata": metadata_audit},
        "flags": {
            "policy_test_used": False,
            "training_used": False,
            "new_rgb_generated": False,
            "new_dino_generated": False,
            "gt_body_yaw_used": True,
            "future_candidate_observation_used_for_selection": False,
            "deployable": False,
        },
        "runtime": {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "seed": SEED, "elapsed_seconds": time.monotonic() - started},
    }
    _write_json(output / "fusion_metrics.json", fusion_metrics)

    pairs = {
        "RGBGlobal_vs_Confidence": _pair_audit(moving_rows, "RGBGlobal-Visibility", "GTYaw-PoseConfidencePrior", rgb_actions, confidence_actions, rgb_fused_actions, moving_options),
        "Frame0SceneVisibility_vs_Confidence": _pair_audit(moving_rows, "Frame0SceneVisibility", "GTYaw-PoseConfidencePrior", scene_actions, confidence_actions, scene_fused_actions, moving_options),
    }
    _write_json(output / "complementarity.json", pairs)
    disagreements = {
        "RGBGlobal_vs_Confidence": _disagreement(moving_rows, moving_options, rgb_actions, confidence_actions, rgb_fused_actions),
        "Frame0SceneVisibility_vs_Confidence": _disagreement(moving_rows, moving_options, scene_actions, confidence_actions, scene_fused_actions),
    }
    _write_json(output / "disagreement_subset.json", disagreements)
    high = _high_occlusion(
        moving_rows,
        moving_options,
        moving_scene,
        {
            "RGBGlobal": rgb_actions,
            "RGBGlobal+Confidence": rgb_fused_actions,
            "Frame0SceneVisibility": scene_actions,
            "Frame0SceneVisibility+Confidence": scene_fused_actions,
            "GTYaw-PoseConfidencePrior": confidence_actions,
            "GT-TrueLogP Oracle": oracle_actions,
        },
    )
    _write_json(output / "high_occlusion.json", high)

    rgb_gain = fusion_metrics["gains_vs_baseline"]["RGBGlobal-Visibility"]["accuracy_gain_pp"]
    scene_gain = fusion_metrics["gains_vs_baseline"]["Frame0SceneVisibility"]["accuracy_gain_pp"]
    if rgb_gain >= 1.0 or scene_gain >= 1.0:
        decision = "STRONG PEPPERPOSE COMPLEMENTARITY" if rgb_gain >= 1.0 and moving_metrics["RGBGlobal-Visibility+Confidence"]["accuracy"] >= 0.58 else "KEEP PEPPERPOSE COMPLEMENTARITY"
    elif rgb_gain >= 0.5 or scene_gain >= 0.5:
        decision = "WEAK PEPPERPOSE COMPLEMENTARITY"
    else:
        decision = "KILL ENTIRE PEPPERPOSE BRANCH"
    lines = [
        "# PepperPose confidence + visibility fusion audit",
        "",
        "This is a privileged, non-deployable diagnostic. The true-facing frame-0 confidence table is loaded from the preceding audit; no angle sanity or confidence landscape was recomputed.",
        "",
        "## Moving Val results",
        "",
        "| Method | Accuracy | Macro-F1 | Move rate |",
        "|---|---:|---:|---:|",
    ]
    for name in moving_methods:
        item = moving_metrics[name]
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} |")
    lines.extend([
        "",
        f"RGBGlobal selected λ={lambda_rgb:.2f}; Moving gain={rgb_gain:+.3f} pp Accuracy.",
        f"Frame0SceneVisibility selected λ={lambda_scene:.2f}; Moving gain={scene_gain:+.3f} pp Accuracy.",
        "",
        "## Complementarity",
        "",
    ])
    for key, pair in pairs.items():
        lines.append(f"- **{key}**: selected-view agreement {pair['selected_view_agreement_rate']:.6f}; primary-correct/confidence-wrong {pair['primary_correct_confidence_wrong']}; primary-wrong/confidence-correct {pair['primary_wrong_confidence_correct']}; both wrong {pair['both_wrong']}; pair AnyCorrect {pair['pair_any_correct_count']} ({pair['pair_any_correct_rate']:.6f}).")
    lines.extend(["", "## Disagreement subsets", ""])
    for key, item in disagreements.items():
        lines.append(f"- **{key}** ({item['count']} contexts): primary Acc={item['primary']['accuracy']:.6f}, confidence Acc={item['confidence']['accuracy']:.6f}, fused Acc={item['fused']['accuracy']:.6f}.")
    lines.extend(["", "## High-occlusion subset", "", f"Strict lower tertile of current frame-0 Stay SceneVisibility: n={high['count']}.", "", "| Method | Accuracy | Macro-F1 |", "|---|---:|---:|"])
    for name, item in high["methods"].items():
        lines.append(f"| {name} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} |")
    lines.extend([
        "",
        f"## Decision: {decision}",
        "",
        "The confidence prior is evaluated with GT body yaw and is therefore not deployable. Lambda selection used only a deterministic Policy-Train record holdout; Moving Val was not used for tuning. The policy archive confidence limitation from the previous audit remains: it stores sequence-level `(32,)` confidence, while the frozen prior table itself is frame-0-derived from the internal Yaw8 archive.",
        "",
        "Flags: `policy_test_used=false`, `training_used=false`, `new_rgb_generated=false`, `future_candidate_observation_used_for_selection=false`, `deployable=false`.",
    ])
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"fusion_metrics": fusion_metrics, "complementarity": pairs, "disagreement": disagreements, "high_occlusion": high, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
