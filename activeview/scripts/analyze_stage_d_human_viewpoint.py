#!/usr/bin/env python3
"""EXP031 offline human-view geometry and perception-quality audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_body_view import FEATURE_NAMES, body_view_features
from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_predictability import oracle_action_index, oracle_margin
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root, get_humanoid_urdf_path
from activeview.dataset.babel_clean_dataset_generator import MotionConverter, URDF_PATH, _load_resampled_motion, compose_root_rotation
from activeview.motion.amass_loader import NormalizedMotion
from activeview.scripts.analyze_stage_d_semantic_bev import _records_by_identity
from activeview.scripts.generate_hm3d_train_rgb_observations import _load_skeleton_metadata

EXP_ID = "EXP031"
TARGET_FRAMES = 30
FRAME_INDEX = 15
QUALITY_NAMES = ("mean_pose_confidence", "valid_joint_fraction", "skeleton_spread", "temporal_motion_energy")


def _assert_split(rows: Sequence[Mapping[str, Any]], expected: str, name: str) -> None:
    """Fail closed when an input row is not explicitly from the expected split."""
    invalid = [str(row.get("policy_split", "<missing>")).lower() for row in rows
               if str(row.get("policy_split", "")).lower() != expected]
    if invalid:
        raise ValueError(f"{name} must explicitly declare policy_split={expected}: {sorted(set(invalid))}")


def _anchors(record: Any, metadata: Mapping[str, Any], converter: MotionConverter) -> tuple[np.ndarray, np.ndarray]:
    motion = _load_resampled_motion(record.motion, TARGET_FRAMES)
    # MotionConverter is frame-independent; converting only the protocol's
    # fixed frame 15 avoids an unnecessary 30-frame conversion per record.
    frame_motion = NormalizedMotion(
        translation=motion.translation[FRAME_INDEX:FRAME_INDEX + 1],
        root_rotation=motion.root_rotation[FRAME_INDEX:FRAME_INDEX + 1],
        body_pose=motion.body_pose[FRAME_INDEX:FRAME_INDEX + 1],
        fps=motion.fps,
        num_frames=1,
        metadata=dict(motion.metadata),
    )
    converted = converter.convert(frame_motion)
    joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)[0].reshape(-1, 3)
    root_transform = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)[0]
    rotation = compose_root_rotation(root_transform)[:3, :3]
    placement = np.asarray(metadata["placement_position"], dtype=np.float32)
    values = placement[None, :] + joints @ rotation.T
    if len(values) > 17:
        values = values[np.linspace(0, len(values) - 1, 17, dtype=np.int64)]
    return values.astype(np.float32), (placement + rotation @ joints.mean(axis=0)).astype(np.float32)


def _load_quality_archive(archive: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(archive, allow_pickle=False) as data:
        ids = np.asarray(data["viewpoint_ids"], dtype=np.int64)
        skeleton = np.asarray(data["skeleton"], dtype=np.float32)
        confidence = np.asarray(data["confidence"], dtype=np.float32)
    if skeleton.shape != (32, 3, 30, 17) or confidence.shape != (32,):
        raise ValueError(f"Invalid canonical skeleton archive: {archive}")
    return ids, skeleton, confidence


def _quality(quality_archive: tuple[np.ndarray, np.ndarray, np.ndarray], viewpoint_id: int) -> np.ndarray:
    ids, skeleton, confidence = quality_archive
    match = np.flatnonzero(ids == int(viewpoint_id))
    if match.size != 1:
        raise ValueError(f"Missing viewpoint {viewpoint_id} in canonical skeleton archive")
    pose = skeleton[int(match[0])]
    finite = np.isfinite(pose)
    temporal = np.diff(pose, axis=1)
    return np.asarray([float(confidence[int(match[0])]), float(np.mean(finite)), float(np.std(pose)), float(np.mean(np.abs(temporal)))], dtype=np.float32)


def _records(rows: Sequence[Mapping[str, Any]], source_root: Path, motion_root: Path) -> dict[tuple[str, str, str], Any]:
    return _records_by_identity(source_root, motion_root, sorted({str(row["scene_id"]) for row in rows}))


def _make_features(rows: Sequence[Mapping[str, Any]], source_root: Path, motion_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _records(rows, source_root, motion_root); converter = MotionConverter(URDF_PATH); body_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, Path, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}; output: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
        record = records.get(key)
        if record is None:
            raise ValueError(f"Missing source record {key}")
        if key not in body_cache:
            metadata = _load_skeleton_metadata(record.source_path); anchors, root = _anchors(record, metadata, converter); body_cache[key] = (anchors, root, record.source_path, metadata, _load_quality_archive(record.source_path))
        anchors, root, archive, metadata, quality_archive = body_cache[key]; ids = [int(v) for v in row["remaining_candidate_ids"]]; candidates = {}
        qualities = {}
        positions = np.asarray(metadata["viewpoint_agent_positions"], dtype=np.float32); rotations = np.asarray(metadata["viewpoint_rotations_wxyz"], dtype=np.float32)
        for candidate_id in ids:
            candidates[str(candidate_id)] = body_view_features(anchors, root, positions[candidate_id], rotations[candidate_id]).tolist(); qualities[str(candidate_id)] = _quality(quality_archive, candidate_id).tolist()
        output.append({"episode_id": str(row["episode_id"]), "scene_id": str(row["scene_id"]), "region": str(row["region"]), "record_id": str(row["record_id"]), "policy_split": str(row["policy_split"]), "label_id": int(row["label_id"]), "remaining_candidate_ids": ids, "second_step_utility_targets": [float(v) for v in row["second_step_utility_targets"]], "body_view_features": candidates, "pose_quality_features": qualities})
    return output, {"record_count": len(body_cache), "archive_count": len(body_cache)}


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0); std = train.std(axis=0); std[std < 1e-6] = 1.0
    return (train - mean) / std, (values - mean) / std


def _corr(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    a = np.asarray(x, dtype=np.float64); b = np.asarray(y, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])}


def _regress(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn
    torch.manual_seed(42); base_dim = train_x.shape[1] - len(FEATURE_NAMES)
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.base = nn.Linear(base_dim, 128); self.body = nn.Linear(len(FEATURE_NAMES), 64); self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(torch.cat([nn.functional.gelu(self.base(x[:, :base_dim])), nn.functional.gelu(self.body(x[:, base_dim:]))], dim=1))
    model = Model(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.SmoothL1Loss(); tx = torch.from_numpy(train_x.astype(np.float32)); ty = torch.from_numpy(train_y.astype(np.float32)).reshape(-1, 1)
    for _ in range(20):
        order = torch.randperm(len(tx))
        for start in range(0, len(tx), 512):
            idx = order[start : start + 512]; loss = loss_fn(model(tx[idx]), ty[idx]); optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.inference_mode():
        pred = model(torch.from_numpy(val_x.astype(np.float32))).reshape(-1).numpy(); train_pred = model(tx).reshape(-1).numpy()
    return pred, {"architecture": f"base Linear({base_dim},128) + body_view Linear({len(FEATURE_NAMES)},64) -> concat(192) -> Linear(192,128)->GELU->Linear(128,64)->GELU->Linear(64,1)", "epochs": 20, "batch_size": 512, "learning_rate": 1e-3, "loss": "SmoothL1Loss", "train_final_mae": float(np.mean(np.abs(train_pred - train_y)))}


def _quality_summary(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in QUALITY_NAMES}; utilities: list[float] = []; winner_hits: list[bool] = []
    correlations: dict[str, Any] = {}
    for row in val_rows:
        targets = [float(v) for v in row["second_step_utility_targets"]]; ids = row["remaining_candidate_ids"]
        for i, candidate_id in enumerate(ids):
            quality = row["pose_quality_features"][str(candidate_id)]
            for j, name in enumerate(QUALITY_NAMES):
                values[name].append(float(quality[j]))
            utilities.append(targets[i])
        if len(ids) == 2 and oracle_action_index(targets) > 0:
            q = [float(np.mean(row["pose_quality_features"][str(candidate_id)])) for candidate_id in ids]; winner_hits.append(int(np.argmax(q)) == int(np.argmax(np.asarray(targets))))
    winner_by_feature: dict[str, list[bool]] = {name: [] for name in QUALITY_NAMES}
    winner_by_margin: dict[str, dict[str, list[bool]]] = {name: {str(t): [] for t in (0.25, 0.5, 1.0, 2.0)} for name in QUALITY_NAMES}
    for row in val_rows:
        targets = np.asarray(row["second_step_utility_targets"], dtype=np.float64)
        if len(row["remaining_candidate_ids"]) == 2 and oracle_action_index(targets) > 0:
            for index, name in enumerate(QUALITY_NAMES):
                q = [float(row["pose_quality_features"][str(cid)][index]) for cid in row["remaining_candidate_ids"]]
                winner_by_feature[name].append(int(np.argmax(q)) == int(np.argmax(targets)))
                margin = oracle_margin(targets)["margin_1"]
                for threshold in (0.25, 0.5, 1.0, 2.0):
                    if margin >= threshold:
                        winner_by_margin[name][str(threshold)].append(int(np.argmax(q)) == int(np.argmax(targets)))
    for name in QUALITY_NAMES:
        correlations[name] = _corr(values[name], utilities)
    return {"available": True, "feature_names": QUALITY_NAMES, "feature_correlations": correlations, "oracle_move_winner_accuracy": float(np.mean(winner_hits)) if winner_hits else None, "oracle_move_winner_accuracy_by_feature": {name: float(np.mean(hits)) if hits else None for name, hits in winner_by_feature.items()}, "high_margin_winner_accuracy_by_feature": {name: {threshold: float(np.mean(hits)) if hits else None for threshold, hits in margins.items()} for name, margins in winner_by_margin.items()}, "oracle_move_episode_count": len(winner_hits), "future_candidate_skeleton_used": True, "oracle_perception_quality_upper_bound": True, "deployable": False, "source_artifact_granularity": "viewpoint-level mean sequence confidence; per-joint confidence is unavailable; valid_joint_fraction is derived from finite 3D pose values"}


def analyze(*, cache_root: Path, output: Path, runtime: Path, source_root: Path, motion_root: Path, train_limit: int | None = None, val_limit: int | None = None) -> dict[str, Any]:
    summary = json.loads((cache_root / "stage_d_feature_summary.json").read_text()); train_rows = load_jsonl(Path(summary["feature_files"]["train"])); val_rows = load_jsonl(Path(summary["feature_files"]["val"]))
    _assert_split(train_rows, "train", "train features")
    _assert_split(val_rows, "val", "val features")
    if train_limit is not None: train_rows = train_rows[:train_limit]
    if val_limit is not None: val_rows = val_rows[:val_limit]
    train_features, train_meta = _make_features(train_rows, source_root, motion_root); val_features, val_meta = _make_features(val_rows, source_root, motion_root)
    def base(row: Mapping[str, Any]) -> np.ndarray:
        return np.asarray([*row["s0_feature"], *row["s1_feature"], *row["delta_semantic"]], dtype=np.float32)
    base_train, base_val = _standardize(np.stack([base(r) for r in train_rows]), np.stack([base(r) for r in val_rows]))
    train_x: list[np.ndarray] = []; val_x: list[np.ndarray] = []; train_y: list[float] = []; val_y: list[float]
    train_y = []; val_y = []
    for source_rows, feature_rows, bases, xs, ys in ((train_rows, train_features, base_train, train_x, train_y), (val_rows, val_features, base_val, val_x, val_y)):
        for index, feature in enumerate(feature_rows):
            for candidate_id, target in zip(feature["remaining_candidate_ids"], feature["second_step_utility_targets"]):
                xs.append(np.concatenate([bases[index], np.asarray(feature["body_view_features"][str(candidate_id)], dtype=np.float32)])); ys.append(float(target))
    train_arr, val_arr = _standardize(np.asarray(train_x, dtype=np.float32), np.asarray(val_x, dtype=np.float32)); pred, model_info = _regress(train_arr, np.asarray(train_y, dtype=np.float32), val_arr); true = np.asarray(val_y, dtype=np.float64)
    by_episode: dict[str, list[float]] = {}; offset = 0
    for row in val_features:
        count = len(row["remaining_candidate_ids"]); by_episode[str(row["episode_id"])] = pred[offset : offset + count].tolist(); offset += count
    episode_accuracy = []; binary = []; candidate_hits = []; selected_utility = []; harmful = 0; missed = 0; margins: dict[str, Any] = {}
    for row in val_features:
        target = [float(v) for v in row["second_step_utility_targets"]]; guess = int(np.argmax(np.asarray([0.0, *by_episode[str(row["episode_id"])]], dtype=np.float64))); oracle = oracle_action_index(target); episode_accuracy.append(guess == oracle); binary.append((guess > 0) == (oracle > 0)); selected_utility.append(0.0 if guess == 0 else target[guess - 1]); harmful += int(guess > 0 and selected_utility[-1] <= 0.0); missed += int(guess == 0 and oracle > 0)
        if guess > 0 and oracle > 0: candidate_hits.append(guess == oracle)
    q_summary = _quality_summary(train_features, val_features)
    body_correlations = {}
    body_utilities = [float(t) for r in val_features for t in r["second_step_utility_targets"]]
    for i, name in enumerate(FEATURE_NAMES):
        body_values = [float(r["body_view_features"][str(cid)][i]) for r in val_features for cid in r["remaining_candidate_ids"]]
        body_correlations[name] = _corr(body_values, body_utilities)
    strongest_index = max(range(len(FEATURE_NAMES)), key=lambda i: abs(float(body_correlations[FEATURE_NAMES[i]]["pearson"] or 0.0)))
    strongest_name = FEATURE_NAMES[strongest_index]
    oracle_move_rows = [r for r in val_features if len(r["remaining_candidate_ids"]) == 2 and oracle_action_index(r["second_step_utility_targets"]) > 0]
    body_winner = [int(np.argmax([float(r["body_view_features"][str(cid)][strongest_index]) for cid in r["remaining_candidate_ids"]])) == int(np.argmax(np.asarray(r["second_step_utility_targets"], dtype=np.float64))) for r in oracle_move_rows]
    for threshold in (0.25, 0.5, 1.0, 2.0):
        chosen = [r for r in oracle_move_rows if oracle_margin(r["second_step_utility_targets"])["margin_1"] >= threshold]
        margins[str(threshold)] = {"count": len(chosen), "analytic_feature": strongest_name, "three_way_accuracy": float(np.mean([int(np.argmax([0.0, *[float(r["body_view_features"][str(cid)][strongest_index]) for cid in r["remaining_candidate_ids"]]]) == oracle_action_index(r["second_step_utility_targets"])) for r in chosen])) if chosen else None}
    d_high_margin: dict[str, Any] = {}
    for threshold in (0.25, 0.5, 1.0, 2.0):
        chosen = [r for r in val_features if oracle_margin(r["second_step_utility_targets"])["margin_1"] >= threshold]
        d_high_margin[str(threshold)] = {"count": len(chosen), "three_way_accuracy": float(np.mean([int(np.argmax([0.0, *by_episode[str(r["episode_id"])]]) == oracle_action_index(r["second_step_utility_targets"])) for r in chosen])) if chosen else None, "binary_accuracy": float(np.mean([int((np.argmax([0.0, *by_episode[str(r["episode_id"])]]) > 0) == (oracle_action_index(r["second_step_utility_targets"]) > 0)) for r in chosen])) if chosen else None}
    result: dict[str, Any] = {
        "experiment_id": EXP_ID, "status": "COMPLETED", "split": "val", "test_used": False,
        "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": False,
        "stgcn_retrained": False, "eligible_episode_counts": {"train": len(train_rows), "val": len(val_rows)},
        "method_a_body_view_geometry": {"feature_names": FEATURE_NAMES, "feature_dim": len(FEATURE_NAMES), "oracle_human_geometry_used": True, "human_geometry_source": "frame-15 MotionConverter kinematic anchors transformed by canonical placement/root rotation; no Habitat rendering", "oracle_move_episode_count": len(oracle_move_rows), "feature_correlations": body_correlations, "analytic_winner_feature": strongest_name, "winner_accuracy": float(np.mean(body_winner)) if body_winner else None, "high_margin": margins, "future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_semantic_used": False, "future_candidate_skeleton_used": False, "test_used": False},
        "method_b_pose_quality": q_summary,
        "method_c_future_recognition": {"status": "METHOD_C_NOT_AVAILABLE", "reason": "canonical skeleton archives contain no per-view recognition probabilities, entropy, or GT-class outputs"},
        "method_d_body_view_regression": {"model": model_info, "candidate_level": {"n": len(true), "mae": float(np.mean(np.abs(pred - true))), "rmse": float(np.sqrt(np.mean((pred - true) ** 2))), **_corr(pred, true)}, "episode_level": {"three_way_accuracy": float(np.mean(episode_accuracy)), "binary_move_stay_accuracy": float(np.mean(binary)), "both_move_candidate_hit": float(np.mean(candidate_hits)) if candidate_hits else None, "selected_action_mean_true_utility": float(np.mean(selected_utility)), "harmful_move_count": harmful, "missed_beneficial_move_count": missed}, "high_margin": d_high_margin, "future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_semantic_used": False, "future_candidate_skeleton_used": False, "test_used": False},
        "comparison_references": {"exp030_a0_winner": 0.509744, "exp030_method_d_full_scene_winner": 0.559037, "exp030_method_b_candidate_hit": 0.582401},
        "leakage_flags": {"future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_semantic_used": False, "future_candidate_skeleton_used": True, "oracle_perception_quality_upper_bound": True, "deployable": False, "test_used": False},
        "provenance": {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_d_train_features_sha256": file_sha256(Path(summary["feature_files"]["train"])), "stage_d_val_features_sha256": file_sha256(Path(summary["feature_files"]["val"])), "runtime": str(runtime)},
    }
    body_score = result["method_a_body_view_geometry"]["winner_accuracy"]
    quality_score = result["method_b_pose_quality"]["oracle_move_winner_accuracy"]
    d_score = result["method_d_body_view_regression"]["episode_level"]["both_move_candidate_hit"]
    if body_score is not None and body_score > result["comparison_references"]["exp030_method_b_candidate_hit"] and (d_score or 0.0) > result["comparison_references"]["exp030_method_b_candidate_hit"]:
        case = "CASE_A"
    elif (body_score is None or body_score <= result["comparison_references"]["exp030_method_b_candidate_hit"]) and quality_score is not None and quality_score > result["comparison_references"]["exp030_method_b_candidate_hit"]:
        case = "CASE_B"
    elif result["method_c_future_recognition"]["status"] != "METHOD_C_NOT_AVAILABLE":
        case = "CASE_C"
    else:
        case = "CASE_D"
    result["scientific_decision"] = {"case": case, "basis": "descriptive comparison with frozen EXP030 candidate-hit reference; no Val tuning or arbitrary optimization threshold", "trajectory_rollout_performed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_name("feature_summary.json").write_text(json.dumps({"experiment_id": EXP_ID, "body_view_feature_names": FEATURE_NAMES, "pose_quality_feature_names": QUALITY_NAMES, "method_c_status": "METHOD_C_NOT_AVAILABLE", "train_records": train_meta["record_count"], "val_records": val_meta["record_count"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_name("analysis.md").write_text("# EXP031 — Human Viewpoint Sensitivity / Perception-Quality Audit\n\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    data = get_data_root(); parser = argparse.ArgumentParser(); parser.add_argument("--cache-root", type=Path, default=data / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"); parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiments/stage_d/EXP031_human_viewpoint_sensitivity/result.json"); parser.add_argument("--runtime", type=Path, default=data / "datasets/policy_v11_5/experiments/stage_d/EXP031_human_viewpoint_sensitivity"); parser.add_argument("--source-root", type=Path, default=data / "datasets/offline/hm3d-train"); parser.add_argument("--motion-root", type=Path, default=data / "datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"); parser.add_argument("--train-limit", type=int, default=None); parser.add_argument("--val-limit", type=int, default=None); args = parser.parse_args(); print(json.dumps({"experiment_id": EXP_ID, "status": analyze(**vars(args))["status"], "test_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
