#!/usr/bin/env python3
"""Dense world-model and recognition-aware observation-model campaign.

Train/Val only.  Future skeletons are targets/evaluation references; they are
never supplied as model inputs.  Test is intentionally not exposed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from activeview.action_recognition.st_gcn_model import STGCN
from activeview.active_view.stage_d_dense_campaign import context_key
from activeview.active_view.stage_d_world_model import (
    CandidateObservationWorldModel,
    LazyWorldModelContextDataset,
    collate_world_model_context,
    world_model_loss,
)
from activeview.core.paths import get_data_root
from activeview.perception.skeleton_definition import get_skeleton_definition
from activeview.scripts.build_stage_b_utility_labels import _load_model
from activeview.scripts.run_stage_d_exp041_044 import (
    EXP_ROOT,
    _baseline_eval,
    _episode_sources,
    _load_rgb_lookup,
    _rows,
)
from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
from activeview.active_view.stage_d_dataset import load_jsonl

CANDIDATE_CHUNK = 16


def _seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def _stgcn(data_root: Path, device: torch.device) -> tuple[STGCN, int]:
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())
    mapping = json.loads(Path(summary["label_mapping"]).read_text())
    model, _ = _load_model(Path(summary["stgcn_checkpoint"]), len(mapping), str(device))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, len(mapping)


def _log_probs(model: STGCN, skeleton: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.log_softmax(model(skeleton.to(device).float().unsqueeze(-1)), dim=-1)


def _corrected_metrics(pred: np.ndarray, target: np.ndarray, edges: Sequence[tuple[int, int]]) -> dict[str, float]:
    if pred.shape != (3, 30, 17) or target.shape != pred.shape:
        raise ValueError("skeleton must have shape [3,30,17]")
    error = pred.astype(np.float64) - target.astype(np.float64)
    velocity_error = np.diff(pred, axis=1) - np.diff(target, axis=1)
    acceleration_error = np.diff(pred, axis=1, n=2) - np.diff(target, axis=1, n=2)
    pred_xyz = np.transpose(pred, (1, 2, 0))
    target_xyz = np.transpose(target, (1, 2, 0))
    pred_bones = np.stack([np.linalg.norm(pred_xyz[:, i] - pred_xyz[:, j], axis=-1) for i, j in edges])
    target_bones = np.stack([np.linalg.norm(target_xyz[:, i] - target_xyz[:, j], axis=-1) for i, j in edges])
    absolute = np.abs(pred_bones - target_bones)
    relative = absolute / np.maximum(target_bones, 1e-8)
    return {
        "coordinate_mae": float(np.mean(np.abs(error))),
        "coordinate_rmse": float(np.sqrt(np.mean(error**2))),
        "velocity_mae": float(np.mean(np.abs(velocity_error))),
        "velocity_rmse": float(np.sqrt(np.mean(velocity_error**2))),
        "acceleration_rmse": float(np.sqrt(np.mean(acceleration_error**2))),
        "bone_length_mae": float(np.mean(absolute)),
        "bone_length_relative_error": float(np.mean(relative)),
    }


def _mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([item[name] for item in values])) for name in values[0]} if values else {}


def _train(
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    sources: Mapping[tuple[str, str, str], str],
    *,
    variant: str,
    rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray] | None,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    micro_batch: int,
    grad_accum: int,
    workers: int,
) -> tuple[CandidateObservationWorldModel, dict[str, Any]]:
    use_belief = variant in {"B", "C", "D", "E", "F"}
    use_rgb = variant in {"C", "D", "E", "F"}
    residual = variant in {"D", "F"}
    recognition = variant in {"E", "F"}
    dataset = LazyWorldModelContextDataset(rows, sources, use_belief=use_belief, rgb_lookup=rgb_lookup if use_rgb else None, target_scope="all", cache_size=len(rows))
    loader = DataLoader(dataset, batch_size=micro_batch, shuffle=True, num_workers=workers, collate_fn=collate_world_model_context, pin_memory=True, persistent_workers=workers > 0)
    model = CandidateObservationWorldModel(use_belief=use_belief, use_rgb=use_rgb, residual=residual).to(device)
    teacher, classes = _stgcn(data_root, device) if recognition else (None, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); running = 0.0; count = 0; optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            kwargs = {name: batch[name].to(device, non_blocking=True) for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")}
            if use_belief: kwargs["history_belief"] = batch["history_belief"].to(device, non_blocking=True)
            if use_rgb: kwargs["history_rgb"] = batch["history_rgb"].to(device, non_blocking=True)
            target = batch["target_skeleton"].to(device, non_blocking=True)
            mask = batch["candidate_mask"].to(device)
            # Dense all-32 targets are preserved, while candidate chunks keep
            # decoder activation memory bounded on the CUDA worker.
            chunk_losses: list[torch.Tensor] = []
            chunk_count = int(kwargs["candidate_descriptor"].shape[1])
            for start in range(0, chunk_count, CANDIDATE_CHUNK):
                stop = min(start + CANDIDATE_CHUNK, chunk_count)
                chunk_kwargs = dict(kwargs); chunk_kwargs["candidate_descriptor"] = kwargs["candidate_descriptor"][:, start:stop]
                prediction = model(**chunk_kwargs)
                valid = mask[:, start:stop].reshape(-1)
                flat_prediction = prediction.reshape(-1, 3, 30, 17)[valid]
                flat_target = target[:, start:stop].reshape(-1, 3, 30, 17)[valid]
                if not valid.any():
                    continue
                pose_loss, _, _ = world_model_loss(flat_prediction, flat_target)
                loss = pose_loss
                if teacher is not None:
                    with torch.no_grad():
                        target_logp = _log_probs(teacher, flat_target, device)
                    pred_logp = _log_probs(teacher, flat_prediction, device)
                    rec_loss = torch.nn.functional.kl_div(pred_logp, target_logp.exp(), reduction="batchmean")
                    loss = loss + 0.10 * rec_loss
                (loss / grad_accum / max(1, math.ceil(chunk_count / CANDIDATE_CHUNK))).backward()
                chunk_losses.append(loss.detach())
            loss = torch.stack(chunk_losses).mean() if chunk_losses else torch.zeros((), device=device)
            if step % grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            running += float(loss) * len(batch["context_key"]); count += len(batch["context_key"])
        result = {"epoch": epoch, "loss": running / max(count, 1)}
        history.append(result)
        torch.save({"model_state_dict": model.state_dict(), "variant": variant, "epoch": epoch, "seed": 42}, output_dir / "last.pth")
        (output_dir / "partial_result.json").write_text(json.dumps(result, indent=2))
        print(f"{variant} epoch {epoch}/{epochs} loss={result['loss']:.6f}", flush=True)
    return model, {"variant": variant, "train_contexts": len(dataset), "train_targets": len(dataset) * 32, "micro_batch_contexts": micro_batch, "gradient_accumulation": grad_accum, "effective_batch_contexts": micro_batch * grad_accum, "residual": residual, "recognition_aware": recognition, "final_loss": history[-1]["loss"], "history": history, "elapsed_seconds": time.perf_counter() - started, "checkpoint": str((output_dir / "last.pth").resolve()), "stgcn_classes": classes}


def _evaluate(
    model: CandidateObservationWorldModel,
    data_root: Path,
    rows: Sequence[Mapping[str, Any]],
    sources: Mapping[tuple[str, str, str], str],
    *,
    variant: str,
    rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray] | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[int, np.ndarray]]]:
    dataset = LazyWorldModelContextDataset(rows, sources, use_belief=variant in {"B", "C", "D", "E", "F"}, rgb_lookup=rgb_lookup if variant in {"C", "D", "E", "F"} else None, target_scope="all")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_world_model_context)
    edges = get_skeleton_definition(backend="h36m_17").edges
    all_metrics: list[dict[str, float]] = []; unseen_metrics: list[dict[str, float]] = []; p23_metrics: list[dict[str, float]] = []
    agreement: list[float] = []; l1_values: list[float] = []; kl_values: list[float] = []; entropy_error: list[float] = []; true_logp: list[float] = []; pred_logp: list[float] = []; pair_true: list[float] = []; pair_pred: list[float] = []
    predictions: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    teacher, classes = _stgcn(data_root, device)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            kwargs = {name: batch[name].to(device) for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")}
            if "history_belief" in batch: kwargs["history_belief"] = batch["history_belief"].to(device)
            if "history_rgb" in batch: kwargs["history_rgb"] = batch["history_rgb"].to(device)
            target = batch["target_skeleton"].numpy(); ids = batch["candidate_ids"].numpy(); mask = batch["candidate_mask"].numpy(); outputs: list[np.ndarray] = []
            for start in range(0, kwargs["candidate_descriptor"].shape[1], CANDIDATE_CHUNK):
                stop = min(start + CANDIDATE_CHUNK, kwargs["candidate_descriptor"].shape[1]); chunk_kwargs = dict(kwargs); chunk_kwargs["candidate_descriptor"] = kwargs["candidate_descriptor"][:, start:stop]
                outputs.append(model(**chunk_kwargs).cpu().numpy())
            output = np.concatenate(outputs, axis=1)
            flat_pred = torch.from_numpy(output[mask]).to(device); flat_target = torch.from_numpy(target[mask]).to(device)
            pred_lp = _log_probs(teacher, flat_pred, device).cpu().numpy(); true_lp = _log_probs(teacher, flat_target, device).cpu().numpy(); cursor = 0
            for index, key_value in enumerate(batch["context_key"]):
                key = tuple(key_value); predictions[key] = {}
                row = rows[len(predictions) - 1]
                valid_count = int(mask[index].sum())
                for position in range(valid_count):
                    view = int(ids[index, position]); prediction = output[index, position]; truth = target[index, position]
                    metric = _corrected_metrics(prediction, truth, edges); all_metrics.append(metric)
                    if view not in {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])}: unseen_metrics.append(metric)
                    if view in {int(v) for v in row["remaining_candidate_ids"]}:
                        p23_metrics.append(metric); predictions[key][view] = prediction
                    pl = pred_lp[cursor]; tl = true_lp[cursor]; cursor += 1
                    pp = np.exp(pl); tp = np.exp(tl); agreement.append(float(np.argmax(pp) == np.argmax(tp))); l1_values.append(float(np.abs(pp - tp).sum())); kl_values.append(float(np.sum(tp * (tl - pl)))); entropy_error.append(float(abs(np.sum(-tp * tl) - np.sum(-pp * pl)))); true_logp.append(float(tl[int(row["label_id"])])); pred_logp.append(float(pl[int(row["label_id"])]))
                if len(row["remaining_candidate_ids"]) == 2 and all(int(v) in predictions[key] for v in row["remaining_candidate_ids"]):
                    p2, p3 = [int(v) for v in row["remaining_candidate_ids"]]; pair_true.append(float(true_logp[-2] - true_logp[-1])); pair_pred.append(float(pred_logp[-2] - pred_logp[-1]))
    def corr(left: Sequence[float], right: Sequence[float]) -> float | None:
        return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1]) if len(left) > 1 and np.std(left) > 0 and np.std(right) > 0 else None
    return {"variant": variant, "train_target_scope": "all-32", "all_32": _mean_metrics(all_metrics), "unobserved_30": _mean_metrics(unseen_metrics), "canonical_p2_p3": _mean_metrics(p23_metrics), "recognition": {"stgcn_class_agreement": float(np.mean(agreement)), "probability_l1": float(np.mean(l1_values)), "kl_true_pred": float(np.mean(kl_values)), "entropy_mae": float(np.mean(entropy_error)), "true_label_logp_mae": float(np.mean(np.abs(np.asarray(true_logp) - np.asarray(pred_logp)))), "true_label_logp_pearson": corr(true_logp, pred_logp), "true_label_logp_spearman": corr(np.argsort(np.argsort(true_logp)), np.argsort(np.argsort(pred_logp))), "p2_p3_delta_pearson": corr(pair_true, pair_pred), "p2_p3_delta_spearman": corr(np.argsort(np.argsort(pair_true)), np.argsort(np.argsort(pair_pred))), "p2_p3_delta_sign_agreement": float(np.mean(np.sign(pair_true) == np.sign(pair_pred))) if pair_true else 0.0, "p2_p3_count": len(pair_true), "classes": classes}, "prediction_count": len(all_metrics), "test_used": False}, predictions


def _predict_model(model: CandidateObservationWorldModel, dataset: LazyWorldModelContextDataset, *, device: torch.device, batch_size: int = 256, workers: int = 0) -> dict[tuple[str, str, str], dict[int, np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_world_model_context)
    output: dict[tuple[str, str, str], dict[int, np.ndarray]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            kwargs = {name: batch[name].to(device) for name in ("history_skeleton", "history_descriptor", "candidate_descriptor")}
            if "history_belief" in batch:
                kwargs["history_belief"] = batch["history_belief"].to(device)
            if "history_rgb" in batch:
                kwargs["history_rgb"] = batch["history_rgb"].to(device)
            values = model(**kwargs).cpu().numpy()
            valid = batch["candidate_mask"].numpy()
            for index, key in enumerate(batch["context_key"]):
                output[tuple(key)] = {int(batch["candidate_ids"][index, view]): values[index, view] for view in range(values.shape[1]) if valid[index, view]}
    return output


def _corrected_baselines(rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str]) -> dict[str, Any]:
    """Compute corrected coordinate/velocity/acceleration/bone baselines."""
    edges = get_skeleton_definition(backend="h36m_17").edges
    values: dict[str, list[dict[str, float]]] = {"COPY_CURRENT": [], "MEAN_TWO_VIEW": [], "NEAREST_OBSERVED_VIEW": []}
    for row in rows:
        with np.load(sources[context_key(row)], allow_pickle=False) as archive:
            skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
            positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
        index = {int(view): i for i, view in enumerate(ids.tolist())}
        s0 = skeleton[index[int(row["s0_viewpoint_id"])]]
        s1 = skeleton[index[int(row["s1_viewpoint_id"])]]
        nearest = []
        for view in range(32):
            nearest_view = min((int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])), key=lambda observed: float(np.linalg.norm(positions[index[observed]] - positions[index[view]])))
            nearest.append(skeleton[index[nearest_view]])
        predictions = {"COPY_CURRENT": np.repeat(s1[None], 32, axis=0), "MEAN_TWO_VIEW": np.repeat(((s0 + s1) / 2.0)[None], 32, axis=0), "NEAREST_OBSERVED_VIEW": np.asarray(nearest)}
        for name, prediction in predictions.items():
            values[name].append(_mean_metrics([_corrected_metrics(prediction[view], skeleton[view], edges) for view in range(32)]))
    return {name: _mean_metrics(metrics) for name, metrics in values.items()} | {"split": "val", "metric_definition": "corrected_axis1_velocity_axis1_acceleration_h36m17_edges"}


def _run_exp043_r1(
    data_root: Path,
    val_rows: Sequence[Mapping[str, Any]],
    val_sources: Mapping[tuple[str, str, str], str],
    predictions_by_variant: Mapping[str, Mapping[tuple[str, str, str], Mapping[int, np.ndarray]]],
    device: torch.device,
) -> dict[str, Any]:
    """Run selector/HAR diagnostics for every completed world model."""
    stage_b = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")
    v0 = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    label_summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text())
    mapping = json.loads(Path(label_summary["label_mapping"]).read_text())
    categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    teacher, classes = _stgcn(data_root, device)
    output: dict[str, Any] = {"experiment_id": "EXP043-R1", "status": "COMPLETED", "split": "val", "models": {}, "test_used": False}
    for variant, prediction_map in predictions_by_variant.items():
        selectors: dict[str, list[dict[str, Any]]] = {name: [] for name in ("PREDICTED_OBSERVATION_GT_LABEL_ORACLE", "PREDICTED_ENTROPY", "PREDICTED_TOP1_CONFIDENCE", "PREDICTED_BELIEF_CROSS_ENTROPY")}
        for row in val_rows:
            key = context_key(row); candidates = [int(v) for v in row["remaining_candidate_ids"]]
            candidate_skeletons = np.stack([prediction_map[key][view] for view in candidates])
            candidate_logs = _log_probs(teacher, torch.from_numpy(candidate_skeletons), device).cpu().numpy()
            source = Path(val_sources[key])
            with np.load(source, allow_pickle=False) as archive:
                ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64); skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
            index = {int(view): pos for pos, view in enumerate(ids.tolist())}
            current = skeleton[index[int(row["s1_viewpoint_id"])]]
            current_logs = _log_probs(teacher, torch.from_numpy(current[None]), device).cpu().numpy()
            all_logs = np.vstack([current_logs, candidate_logs]); all_probs = np.exp(all_logs)
            belief = np.exp(np.asarray(row["s1_feature"][256:272], dtype=np.float64)); belief /= max(float(belief.sum()), 1e-12)
            scores = {
                "PREDICTED_ENTROPY": -np.sum(all_probs * all_logs, axis=1),
                "PREDICTED_TOP1_CONFIDENCE": -np.max(all_probs, axis=1),
                "PREDICTED_BELIEF_CROSS_ENTROPY": -all_probs @ np.log(np.maximum(belief, 1e-12)),
                "PREDICTED_OBSERVATION_GT_LABEL_ORACLE": -all_logs[:, int(row["label_id"])],
            }
            for name, score in scores.items():
                selected = int(np.argmin(score))
                selectors[name].append({"episode_id": str(row["episode_id"]), "candidate": None if selected == 0 else candidates[selected - 1]})
        model_result: dict[str, Any] = {"selectors": {}, "classes": classes}
        for name, selected_rows in selectors.items():
            decisions = []
            for row, selected in zip(val_rows, selected_rows):
                candidate = selected["candidate"]
                decisions.append({"episode_id": str(row["episode_id"]), "remaining_candidate_ids": row["remaining_candidate_ids"], "predicted_utilities": [0.0, 0.0], "predicted_stays": candidate is None, "predicted_candidate_viewpoint_id": candidate, "max_predicted_utility": 0.0 if candidate is None else 1.0})
            trajectory = summarize_trajectory_rows(build_stage_d_trajectories(stage_b, v0, val_rows, decisions), categories)
            model_result["selectors"][name] = {"trajectory": trajectory, "deployable": name != "PREDICTED_OBSERVATION_GT_LABEL_ORACLE"}
        output["models"][variant] = model_result
    out = EXP_ROOT / "EXP043_R1_model_based_single_step"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _run(data_root: Path, device: torch.device, epochs: int, workers: int, variants: Sequence[str] = ("A", "B", "C", "D", "E", "F")) -> dict[str, Any]:
    _seed(); train_rows = _rows(data_root, "train"); val_rows = _rows(data_root, "val"); train_sources = _episode_sources(data_root, "train"); val_sources = _episode_sources(data_root, "val")
    if len(train_rows) != 29133 or len(val_rows) != 9742: raise RuntimeError("canonical Train/Val population mismatch")
    rgb_root = Path(__import__("os").environ["ACTIVEVIEW_RGB_FEATURE_ROOT"]); rgb_lookup, rgb_summary = _load_rgb_lookup(rgb_root)
    models: dict[str, Any] = {}; evaluations: dict[str, Any] = {}; predictions: dict[str, Any] = {}
    for variant in variants:
        micro_batch = 128 if variant in {"E", "F"} else 32
        grad_accum = 2 if variant in {"E", "F"} else 8
        model, train_result = _train(data_root, train_rows, train_sources, variant=variant, rgb_lookup=rgb_lookup, device=device, output_dir=Path("/tmp") / f"activeview_exp042r1_{variant}", epochs=epochs, micro_batch=micro_batch, grad_accum=grad_accum, workers=0 if variant in {"E", "F"} else workers)
        evaluation, model_predictions = _evaluate(model, data_root, val_rows, val_sources, variant=variant, rgb_lookup=rgb_lookup, device=device); models[variant] = {"train": train_result, "val": evaluation}; predictions[variant] = model_predictions
        partial = {"experiment_id": "EXP042-R1_EXP045", "status": "PARTIAL", "completed_variants": list(models), "population": {"train_contexts": len(train_rows), "val_contexts": len(val_rows), "train_dense_targets": len(train_rows) * 32, "val_dense_targets": len(val_rows) * 32}, "models": models, "rgb_cache": rgb_summary, "test_used": False, "training_performed": True}
        partial_root = EXP_ROOT / "EXP042_R1_dense_cross_view_world_model"; partial_root.mkdir(parents=True, exist_ok=True); (partial_root / "partial_result.json").write_text(json.dumps(partial, indent=2))
    corrected = _corrected_baselines(val_rows, val_sources)
    exp043 = _run_exp043_r1(data_root, val_rows, val_sources, predictions, device) if predictions else {"status": "BLOCKED"}
    exp_root = EXP_ROOT / "EXP042_R1_dense_cross_view_world_model"; exp_root.mkdir(parents=True, exist_ok=True)
    (exp_root / "metric_fix_audit.json").write_text(json.dumps(corrected, indent=2), encoding="utf-8")
    exp045_root = EXP_ROOT / "EXP045_recognition_aware_world_model"; exp045_root.mkdir(parents=True, exist_ok=True)
    exp045_models = {key: value for key, value in models.items() if key in {"E", "F"}}
    (exp045_root / "result.json").write_text(json.dumps({"experiment_id": "EXP045", "status": "COMPLETED" if len(exp045_models) == 2 else "PARTIAL", "models": exp045_models, "recognition_loss_weight": 0.10, "teacher": "frozen ST-GCN", "split": ["train", "val"], "test_used": False, "training_performed": True}, indent=2), encoding="utf-8")
    summary = {"experiment_id": "EXP042-R1_EXP045", "status": "COMPLETED", "split": ["train", "val"], "population": {"train_contexts": len(train_rows), "val_contexts": len(val_rows), "train_dense_targets": len(train_rows) * 32, "val_dense_targets": len(val_rows) * 32}, "models": models, "rgb_cache": rgb_summary, "corrected_baselines": corrected, "exp043_r1": exp043, "test_used": False, "training_performed": True}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--variants", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is required for this campaign")
    result = _run(args.data_root.resolve(), device, args.epochs, args.workers, args.variants)
    out = EXP_ROOT / "EXP042_R1_dense_cross_view_world_model"; out.mkdir(parents=True, exist_ok=True); (out / "result.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
