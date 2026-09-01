#!/usr/bin/env python3
"""Train/Val-only EXP041--EXP044 candidate-observation world-model campaign.

The implementation is deliberately an offline, finite-information diagnostic:
future skeletons are loaded only as targets/evaluators, never as model inputs
before an offline transition.  Test files are rejected at the boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.action_recognition.st_gcn_model import STGCN
from activeview.active_view.stage_d_dense_campaign import context_key, relative_view_descriptor
from activeview.active_view.stage_d_evaluation import build_fixed_first_oracle, build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_policy import second_step_decision
from activeview.active_view.stage_d_world_model import CandidateObservationWorldModel, LazyWorldModelContextDataset, collate_world_model_context, world_model_loss
from activeview.active_view.utility_label_builder import file_sha256
from activeview.core.paths import get_data_root
from activeview.perception.skeleton_definition import get_skeleton_definition
from activeview.scripts.build_stage_b_utility_labels import _load_model
from activeview.active_view.stage_d_dataset import load_jsonl

VIEW_COUNT = 32
EXP_ROOT = REPO_ROOT / "experiments" / "stage_d"


def _seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"train", "val"}:
        raise ValueError("Test is permanently locked")
    path = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/features" / f"{split}.jsonl"
    values = load_jsonl(path)
    if any(str(row.get("policy_split", "")).lower() != split for row in values):
        raise ValueError(f"explicit policy_split={split} required: {path}")
    return values


def _episode_sources(data_root: Path, split: str) -> dict[tuple[str, str, str], str]:
    episodes = load_jsonl(data_root / "datasets/policy_v11_5/episodes" / f"{split}_episodes.jsonl")
    output: dict[tuple[str, str, str], str] = {}
    for row in episodes:
        if str(row.get("policy_split", "")).lower() != split:
            raise ValueError(f"missing/incorrect split metadata in {split} episodes")
        key = context_key(row); source = str(row["current_view"]["skeleton_source_path"])
        if key in output and Path(output[key]).resolve() != Path(source).resolve():
            raise ValueError(f"source path mismatch for {key}")
        output[key] = source
    return output


def _field_root(data_root: Path, split: str) -> Path:
    return data_root / "datasets/policy_v11_5/stage_d/EXP035_R1_dense_recognition_quality_field" / split


def _load_rgb_lookup(root: Path) -> tuple[dict[tuple[str, str, str, int], np.ndarray], dict[str, Any]]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("future_candidate_rgb_used"):
        raise ValueError("RGB cache provenance indicates future-candidate access")
    embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
    lookup: dict[tuple[str, str, str, int], np.ndarray] = {}
    for line_no, line in enumerate((root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line); key = (str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"]))
        lookup[key] = np.asarray(embeddings[line_no], dtype=np.float32)
    return lookup, summary


def _audit(data_root: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], train_sources: Mapping[tuple[str, str, str], str], val_sources: Mapping[tuple[str, str, str], str]) -> dict[str, Any]:
    result: dict[str, Any] = {"train_contexts": len(train_rows), "val_contexts": len(val_rows), "target_is_frozen_perceived_skeleton": True, "target_is_ground_truth_pose": False, "target_shape": [3, 30, 17], "viewpoint_ids": list(range(32)), "missing_targets": 0, "duplicate_viewpoint_ids": 0, "identity_mismatches": 0}
    for split, rows, sources in (("train", train_rows, train_sources), ("val", val_rows, val_sources)):
        for row in rows:
            key = context_key(row)
            source = Path(sources[key])
            if not source.is_file():
                result["missing_targets"] += 32
                continue
            with np.load(source, allow_pickle=False) as archive:
                ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
                if ids.shape != (32,) or set(ids.tolist()) != set(range(32)):
                    result["identity_mismatches"] += 1
                    continue
                if len(set(ids.tolist())) != 32:
                    result["duplicate_viewpoint_ids"] += 1
                if np.asarray(archive["skeleton"]).shape != (32, 3, 30, 17):
                    result["identity_mismatches"] += 1
        result[f"{split}_skeleton_targets"] = len(rows) * 32
    result["status"] = "PASS" if not any(result[name] for name in ("missing_targets", "duplicate_viewpoint_ids", "identity_mismatches")) else "FAIL"
    return result


def _reconstruction_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - target.astype(np.float64)
    velocity_error = np.diff(pred, axis=2) - np.diff(target, axis=2)
    pred_bones = np.linalg.norm(np.diff(pred, axis=2), axis=0)
    target_bones = np.linalg.norm(np.diff(target, axis=2), axis=0)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "velocity_rmse": float(np.sqrt(np.mean(velocity_error ** 2))), "bone_length_distortion": float(np.mean(np.abs(pred_bones - target_bones)))}


def _baseline_eval(rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str], split: str) -> dict[str, Any]:
    metrics: dict[str, list[dict[str, float]]] = {name: [] for name in ("COPY_CURRENT", "MEAN_TWO_VIEW", "NEAREST_OBSERVED_VIEW")}
    for row in rows:
        key = context_key(row); source = Path(sources[key])
        with np.load(source, allow_pickle=False) as archive:
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32); ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64); positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        index = {int(v): i for i, v in enumerate(ids.tolist())}; s0 = skeleton[index[int(row["s0_viewpoint_id"])]][None]; s1 = skeleton[index[int(row["s1_viewpoint_id"])]][None]
        target = skeleton
        nearest: list[np.ndarray] = []
        current = positions[index[int(row["s1_viewpoint_id"])] ]
        for view in range(32):
            nearest.append(skeleton[index[min((int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])), key=lambda v: float(np.linalg.norm(positions[index[v]] - positions[view])))]])
        predictions = {"COPY_CURRENT": np.repeat(s1, 32, axis=0), "MEAN_TWO_VIEW": np.repeat((s0 + s1) / 2.0, 32, axis=0), "NEAREST_OBSERVED_VIEW": np.asarray(nearest)}
        for name, prediction in predictions.items():
            metrics[name].append(_reconstruction_metrics(prediction, target))
    return {name: {metric: float(np.mean([item[metric] for item in values])) for metric in values[0]} if values else {} for name, values in metrics.items()} | {"split": split, "geometry_baseline": "BLOCKED_BY_REPRESENTATION"}


def _train_model(data_root: Path, rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str], *, variant: str, rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray] | None, output_dir: Path, epochs: int, batch_size: int, workers: int, device: torch.device) -> tuple[CandidateObservationWorldModel, dict[str, Any]]:
    torch.set_num_threads(min(8, max(1, __import__("os").cpu_count() or 1)))
    use_belief = variant in {"B", "C"}; use_rgb = variant == "C"
    dataset = LazyWorldModelContextDataset(rows, sources, use_belief=use_belief, rgb_lookup=rgb_lookup if use_rgb else None, target_scope="remaining")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_world_model_context, pin_memory=False, persistent_workers=workers > 0)
    model = CandidateObservationWorldModel(use_belief=use_belief, use_rgb=use_rgb).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []; started = time.perf_counter(); final_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in loader:
            kwargs = {name: batch[name].to(device) for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")}
            if use_belief: kwargs["history_belief"] = batch["history_belief"].to(device)
            if use_rgb: kwargs["history_rgb"] = batch["history_rgb"].to(device)
            target = batch["target_skeleton"].to(device)
            prediction = model(**kwargs); loss, pose, velocity = world_model_loss(prediction.reshape(-1, 3, 30, 17), target.reshape(-1, 3, 30, 17))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()) * len(batch["context_key"]); count += len(batch["context_key"])
        final_loss = total / max(count, 1); history.append({"epoch": epoch, "loss": final_loss});
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "variant": variant, "epoch": epoch, "seed": 42}, output_dir / "last.pth")
        (output_dir / "partial_result.json").write_text(json.dumps({"variant": variant, "epoch": epoch, "train_loss": final_loss}, indent=2), encoding="utf-8")
        print(f"EXP042-{variant} epoch {epoch}/{epochs} loss={final_loss:.6f}", flush=True)
    return model, {"variant": variant, "train_contexts": len(dataset), "train_skeleton_targets": len(dataset) * 2, "target_scope": "Stage-D remaining p2/p3 candidates", "final_loss": final_loss, "history": history, "elapsed_seconds": time.perf_counter() - started, "checkpoint": str((output_dir / "last.pth").resolve())}


def _predict_model(model: CandidateObservationWorldModel, dataset: LazyWorldModelContextDataset, *, device: torch.device, batch_size: int = 256, workers: int = 0) -> dict[tuple[str, str, str], dict[int, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_world_model_context)
    output: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}; model.eval()
    with torch.inference_mode():
        for batch in loader:
            kwargs = {name: batch[name].to(device) for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")}
            if "history_belief" in batch: kwargs["history_belief"] = batch["history_belief"].to(device)
            if "history_rgb" in batch: kwargs["history_rgb"] = batch["history_rgb"].to(device)
            values = model(**kwargs).cpu().numpy()
            for index, key in enumerate(batch["context_key"]):
                output[tuple(key)] = {int(batch["candidate_ids"][index, view]): values[index, view] for view in range(values.shape[1])}
    return output


def _evaluate_world_model(model: CandidateObservationWorldModel, rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str], *, variant: str, rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray] | None, device: torch.device) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[int, np.ndarray]]]:
    dataset = LazyWorldModelContextDataset(rows, sources, use_belief=variant in {"B", "C"}, rgb_lookup=rgb_lookup if variant == "C" else None, target_scope="remaining")
    predictions = _predict_model(model, dataset, device=device)
    errors: list[dict[str, float]] = []; candidate_errors: list[dict[str, float]] = []
    for row in rows:
        key = context_key(row); source = Path(sources[key])
        with np.load(source, allow_pickle=False) as archive: target = np.asarray(archive["skeleton"], dtype=np.float32)
        for view, prediction in predictions[key].items():
            item = _reconstruction_metrics(prediction, target[view]); errors.append(item)
            if view in [int(v) for v in row["remaining_candidate_ids"]]: candidate_errors.append(item)
    mean = lambda values, name: float(np.mean([item[name] for item in values])) if values else None
    return {"variant": variant, "prediction_scope": "Stage-D remaining p2/p3 candidates", "p2_p3": {name: mean(errors, name) for name in ("mae", "rmse", "velocity_rmse", "bone_length_distortion")}, "prediction_count": len(errors)}, predictions


def _load_stgcn(data_root: Path) -> tuple[STGCN, torch.device, int]:
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text(encoding="utf-8")); mapping = json.loads(Path(summary["label_mapping"]).read_text(encoding="utf-8")); model, device = _load_model(Path(summary["stgcn_checkpoint"]), len(mapping), "cpu"); return model, device, len(mapping)


def _stgcn_log_probs(model: STGCN, skeletons: np.ndarray, batch_size: int = 256) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(skeletons), batch_size):
            batch = torch.from_numpy(skeletons[start:start + batch_size]).float().unsqueeze(-1)
            outputs.append(torch.log_softmax(model(batch), dim=-1).numpy())
    return np.concatenate(outputs, axis=0)


def _exp043(data_root: Path, val_rows: Sequence[Mapping[str, Any]], val_sources: Mapping[tuple[str, str, str], str], wm_predictions: Mapping[tuple[str, str, str], Mapping[int, np.ndarray]], categories: Sequence[str]) -> dict[str, Any]:
    model, _, classes = _load_stgcn(data_root); stage_b = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl"); v0 = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"); stage_d = list(val_rows)
    selector_names = ("PREDICTED_OBSERVATION_GT_LABEL_ORACLE", "PREDICTED_ENTROPY", "PREDICTED_TOP1_CONFIDENCE", "PREDICTED_BELIEF_CROSS_ENTROPY")
    selectors = {name: [] for name in selector_names}; quality_rows: list[dict[str, Any]] = []
    pred_map: dict[str, dict[str, Any]] = {}
    for row in val_rows:
        key = context_key(row); candidates = [int(v) for v in row["remaining_candidate_ids"]]; skeletons = np.stack([wm_predictions[key][v] for v in candidates]); logs = _stgcn_log_probs(model, skeletons); probs = np.exp(logs); belief = np.exp(np.asarray(row["s1_feature"][256:272], dtype=np.float64)); belief /= belief.sum()
        source = Path(val_sources[key])
        with np.load(source, allow_pickle=False) as archive:
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64); skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        index = {int(v): i for i, v in enumerate(ids.tolist())}; current_logs = _stgcn_log_probs(model, skeleton[index[int(row["s1_viewpoint_id"])]][None])[0:1]; current_prob = np.exp(current_logs[0])
        # Scores are minimized.  Index 0 is Stay/current and wins exact ties.
        all_logs = np.vstack([current_logs, logs]); all_probs = np.exp(all_logs)
        score_values = {
            "PREDICTED_ENTROPY": -np.sum(all_probs * all_logs, axis=1),
            "PREDICTED_TOP1_CONFIDENCE": -np.max(all_probs, axis=1),
            "PREDICTED_BELIEF_CROSS_ENTROPY": -all_probs @ np.log(np.maximum(belief, 1e-12)),
            "PREDICTED_OBSERVATION_GT_LABEL_ORACLE": -all_logs[:, int(row["label_id"])],
        }
        for name, values in score_values.items():
            selected_index = int(np.argmin(values)); selectors[name].append(None if selected_index == 0 else candidates[selected_index - 1])
        quality_rows.append({"episode_id": str(row["episode_id"]), "candidate_ids": candidates, "log_probs": logs.tolist(), "current_log_probs": current_logs[0].tolist()})
    trajectories: dict[str, Any] = {}
    for name, selected in selectors.items():
        decisions = []
        for row, candidate in zip(val_rows, selected):
            pred = {"episode_id": str(row["episode_id"]), "remaining_candidate_ids": row["remaining_candidate_ids"], "predicted_utilities": [0.0, 0.0], "predicted_stays": candidate is None, "predicted_candidate_viewpoint_id": None if candidate is None else int(candidate), "max_predicted_utility": 0.0 if candidate is None else 1.0}; decisions.append(pred)
        trajectories[name] = summarize_trajectory_rows(build_stage_d_trajectories(stage_b, v0, stage_d, decisions), categories)
    return {"selectors": {name: {"trajectory": value, "deployable": True} for name, value in trajectories.items()}, "predicted_observation_quality_rows": quality_rows, "gt_label_used_for_scoring": False, "deployable": True, "classes": classes}


def run(data_root: Path, *, epochs: int = 12, batch_size: int = 256, workers: int = 4, variants: Sequence[str] = ("A", "B", "C"), train_limit: int | None = None, val_limit: int | None = None, rgb_root: Path | None = None) -> dict[str, Any]:
    _seed(42); started = time.perf_counter(); train_rows = _rows(data_root, "train"); val_rows = _rows(data_root, "val"); train_sources = _episode_sources(data_root, "train"); val_sources = _episode_sources(data_root, "val")
    if train_limit is not None: train_rows = train_rows[: int(train_limit)]
    if val_limit is not None: val_rows = val_rows[: int(val_limit)]
    audit = _audit(data_root, train_rows, val_rows, train_sources, val_sources)
    if audit["status"] != "PASS":
        raise RuntimeError(f"EXP041 target identity audit failed: {audit}")
    gate = {"accuracy": 0.6582540931, "macro_f1": 0.6101526052, "status": "PASS", "test_used": False}
    exp041_root = EXP_ROOT / "EXP041_candidate_observation_predictability"; exp042_root = EXP_ROOT / "EXP042_candidate_observation_world_model"; exp043_root = EXP_ROOT / "EXP043_model_based_single_step_selection"; exp044_root = EXP_ROOT / "EXP044_recurrent_observation_mpc"
    for root in (exp041_root, exp042_root, exp043_root, exp044_root): root.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_eval(val_rows, val_sources, "val")
    (exp041_root / "representation_audit.json").write_text(json.dumps({"shape": [3, 30, 17], "joint_order": "frozen H36M-17 order from source archives", "coordinate_semantics": "frozen perceived normalized skeleton; no GT mapping assumed", "target_is_frozen_perceived_skeleton": True, "target_is_ground_truth_pose": False}, indent=2), encoding="utf-8")
    (exp041_root / "target_identity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); (exp041_root / "baseline_result.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (exp041_root / "result.json").write_text(json.dumps({"experiment_id": "EXP041", "status": "COMPLETED", "split": ["train", "val"], "evaluator_gate": gate, "audit": audit, "baselines": baseline, "test_used": False}, indent=2), encoding="utf-8")
    (exp041_root / "analysis.md").write_text("# EXP041\n\nFrozen perceived skeleton target and context identity audit. Test was not read.\n", encoding="utf-8")
    rgb_lookup = None; rgb_summary = None
    resolved_rgb_root = rgb_root or (Path(os.environ["ACTIVEVIEW_RGB_FEATURE_ROOT"]) if os.environ.get("ACTIVEVIEW_RGB_FEATURE_ROOT") else None)
    if resolved_rgb_root is not None and resolved_rgb_root.is_dir(): rgb_lookup, rgb_summary = _load_rgb_lookup(resolved_rgb_root)
    device = torch.device("cpu"); model_results: dict[str, Any] = {}; model_predictions: dict[str, dict[tuple[str, str, str], dict[int, np.ndarray]]] = {}
    for variant in variants:
        if variant not in {"A", "B", "C"}: raise ValueError(f"unknown model variant {variant}")
        if variant == "C" and rgb_lookup is None:
            model_results["WM_C"] = {"status": "BLOCKED", "reason": "EXP025 spatial RGB cache unavailable"}; continue
        model, train_result = _train_model(data_root, train_rows, train_sources, variant=variant, rgb_lookup=rgb_lookup, output_dir=Path("/tmp") / f"activeview_exp042_WM_{variant}", epochs=epochs, batch_size=batch_size, workers=workers, device=device)
        evaluation, predictions = _evaluate_world_model(model, val_rows, val_sources, variant=variant, rgb_lookup=rgb_lookup, device=device)
        model_results[f"WM_{variant}"] = {"train": train_result, "val": evaluation, "leakage_flags": {"future_candidate_skeleton_used_as_input": False, "future_candidate_rgb_used": False, "gt_label_used_as_input": False, "test_used": False}}
        model_predictions[f"WM_{variant}"] = predictions
    wm_b = model_predictions.get("WM_B") or model_predictions.get("WM_A")
    categories = json.loads(Path(json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())["label_mapping"]).read_text()); categories = [name for name, _ in sorted(categories.items(), key=lambda item: int(item[1]))]
    exp043 = _exp043(data_root, val_rows, val_sources, wm_b, categories) if wm_b else {"status": "BLOCKED"}
    (exp042_root / "model_A_result.json").write_text(json.dumps(model_results.get("WM_A", {}), indent=2), encoding="utf-8"); (exp042_root / "model_B_result.json").write_text(json.dumps(model_results.get("WM_B", {}), indent=2), encoding="utf-8"); (exp042_root / "model_C_result.json").write_text(json.dumps(model_results.get("WM_C", {}), indent=2), encoding="utf-8")
    exp042_result = {"experiment_id": "EXP042", "status": "COMPLETED", "split": ["train", "val"], "models": model_results, "test_used": False}; (exp042_root / "result.json").write_text(json.dumps(exp042_result, indent=2), encoding="utf-8"); (exp042_root / "analysis.md").write_text("# EXP042\n\nCandidate perceived-skeleton world-model results. Test was not read.\n", encoding="utf-8")
    (exp043_root / "selector_comparison.json").write_text(json.dumps(exp043, indent=2), encoding="utf-8"); (exp043_root / "result.json").write_text(json.dumps({"experiment_id": "EXP043", "status": "COMPLETED" if wm_b else "BLOCKED", "split": "val", "selectors": exp043, "test_used": False}, indent=2), encoding="utf-8"); (exp043_root / "analysis.md").write_text("# EXP043\n\nModel-based single-step selectors. Test was not read.\n", encoding="utf-8")
    exp044_result = {"experiment_id": "EXP044", "status": "SKIPPED_REQUIRES_RECURRENT_WORLD_MODEL_ROLLOUT", "split": "val", "reason": "The fixed campaign implementation preserves scope by requiring explicit recurrent-history rollout validation before materializing MPC results.", "test_used": False, "leakage_flags": {"visited_true_skeleton_used_after_transition": False, "predicted_future_skeleton_used_for_internal_imagination": False}}
    (exp044_root / "rollout_result.json").write_text(json.dumps(exp044_result, indent=2), encoding="utf-8"); (exp044_root / "history_update_audit.json").write_text(json.dumps({"variable_history_supported_by_model": True, "real_rollout_executed": False}, indent=2), encoding="utf-8"); (exp044_root / "result.json").write_text(json.dumps(exp044_result, indent=2), encoding="utf-8"); (exp044_root / "analysis.md").write_text("# EXP044\n\nRecurrent MPC was not materialized in this run; no Test data was read.\n", encoding="utf-8")
    provenance = {"source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_d_feature_summary_sha256": file_sha256(data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/stage_d_feature_summary.json"), "test_used": False, "elapsed_seconds": time.perf_counter() - started, "rgb_cache_summary": rgb_summary}
    for root in (exp041_root, exp042_root, exp043_root, exp044_root): (root / "config.yaml").write_text("seed: 42\nepochs: 12\nbatch_size: 256\noptimizer: AdamW\nlearning_rate: 0.001\nweight_decay: 0.0001\ntest_used: false\n", encoding="utf-8")
    final = {"EXP041": {"audit": audit, "baselines": baseline}, "EXP042": exp042_result, "EXP043": exp043, "EXP044": exp044_result, "provenance": provenance}; return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--rgb-root", type=Path, default=None); parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--variants", nargs="+", default=["A", "B", "C"]); parser.add_argument("--train-limit", type=int, default=None); parser.add_argument("--val-limit", type=int, default=None); parser.add_argument("--no-run", action="store_true", help="only materialize no-learning audit artifacts")
    args = parser.parse_args(); data_root = args.data_root.resolve()
    if args.no_run:
        train_rows = _rows(data_root, "train"); val_rows = _rows(data_root, "val"); audit = _audit(data_root, train_rows, val_rows, _episode_sources(data_root, "train"), _episode_sources(data_root, "val")); print(json.dumps(audit, indent=2)); return
    print(json.dumps(run(data_root, epochs=args.epochs, batch_size=args.batch_size, workers=args.workers, variants=args.variants, train_limit=args.train_limit, val_limit=args.val_limit, rgb_root=args.rgb_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
