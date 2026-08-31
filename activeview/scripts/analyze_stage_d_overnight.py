#!/usr/bin/env python3
"""Run the EXP032--EXP034 frozen-artifact overnight audit (Train/Val only)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_predictability import oracle_action_index, oracle_margin
from activeview.active_view.utility_label_builder import file_sha256

EXP_ROOT = REPO_ROOT / "experiments" / "stage_d"
EXP032 = EXP_ROOT / "EXP032_privileged_quality_decomposition"
EXP033 = EXP_ROOT / "EXP033_predict_future_view_quality"
EXP034 = EXP_ROOT / "EXP034_utility_stability_audit"
QUALITY_NAMES = ("future_ce", "future_gt_probability", "future_entropy", "future_correct")
MARGIN_THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


def _corr(left: Sequence[float], right: Sequence[float]) -> dict[str, float | None]:
    x = np.asarray(left, dtype=np.float64); y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"pearson": None, "spearman": None}
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return {"pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(np.corrcoef(rx, ry)[0, 1])}


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    error = pred - target
    return {"n": int(len(target)), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), **_corr(pred, target)}


def _r2_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    base = _metrics(pred, target); denominator = float(np.sum((target - target.mean()) ** 2))
    base["r2"] = float(1.0 - np.sum((pred - target) ** 2) / denominator) if denominator else None
    return base


def _split_check(rows: Sequence[Mapping[str, Any]], split: str, name: str) -> None:
    bad = {str(row.get("policy_split", "<missing>")).lower() for row in rows if str(row.get("policy_split", "")).lower() != split}
    if bad:
        raise ValueError(f"{name} contains non-{split} rows: {sorted(bad)}")


def _load_inputs(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cache = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential"
    summary = json.loads((cache / "stage_d_feature_summary.json").read_text(encoding="utf-8"))
    train = load_jsonl(Path(summary["feature_files"]["train"])); val = load_jsonl(Path(summary["feature_files"]["val"]))
    _split_check(train, "train", "Stage-D Train features"); _split_check(val, "val", "Stage-D Val features")
    utility_root = data_root / "datasets/policy_v11_5/stage_b/utility_labels"
    utilities: dict[str, dict[str, Any]] = {}
    utility_counts: dict[str, int] = {}
    for split in ("train", "val"):
        path = utility_root / f"{split}.jsonl"; count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if str(row.get("policy_split", "")).lower() != split:
                    raise ValueError(f"Stage-B {split} utility has invalid policy_split")
                key = str(row["episode_id"])
                if key in utilities:
                    raise ValueError(f"duplicate utility episode IDs: {key}")
                utilities[key] = {"current": row["current"], "candidates": [{"viewpoint_id": int(item["viewpoint_id"]), "logp_true": float(item["logp_true"]), "entropy": float(item["entropy"]), "correct": bool(item["correct"])} for item in row["candidates"]]}
                count += 1
        utility_counts[split] = count
    if len(utilities) != sum(utility_counts.values()):
        raise ValueError("duplicate utility episode IDs")
    return train, val, {"summary": summary, "utilities": utilities, "utility_counts": utility_counts}


def _candidate_quality(utility_row: Mapping[str, Any], candidate_id: int) -> np.ndarray:
    current = utility_row["current"]
    candidate = next(item for item in utility_row["candidates"] if int(item["viewpoint_id"]) == int(candidate_id))
    logp = float(candidate["logp_true"])
    return np.asarray([-logp, np.exp(logp), float(candidate["entropy"]), float(bool(candidate["correct"]))], dtype=np.float32)


def _records(rows: Sequence[Mapping[str, Any]], utilities: Mapping[str, Mapping[str, Any]], *, quality: bool = True) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        utility = utilities.get(str(row["episode_id"]))
        if utility is None:
            raise ValueError(f"missing Stage-B utility for {row['episode_id']}")
        ids = [int(v) for v in row["remaining_candidate_ids"]]
        if not set(ids).issubset({int(v["viewpoint_id"]) for v in utility["candidates"]}):
            raise ValueError(f"candidate ordering mismatch for {row['episode_id']}")
        base = np.asarray([*row["s0_feature"], *row["s1_feature"], *row["delta_semantic"]], dtype=np.float32)
        geometry = np.asarray(row["second_step_candidate_geometry"], dtype=np.float32).reshape(len(ids), -1)
        targets = np.asarray(row["second_step_utility_targets"], dtype=np.float32)
        result.append({"episode_id": str(row["episode_id"]), "scene_id": str(row["scene_id"]), "region": str(row["region"]), "record_id": str(row["record_id"]), "label_id": int(row["label_id"]), "base": base, "geometry": geometry, "ids": ids, "targets": targets, "quality": np.stack([_candidate_quality(utility, item) for item in ids]) if quality else None, "current_quality": utility["current"]})
    return result


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0); std = train.std(axis=0); std[std < 1e-6] = 1.0
    return (train - mean) / std, (values - mean) / std


def _winner_metrics(records: Sequence[Mapping[str, Any]], scores: Sequence[np.ndarray], *, lower_is_better: bool = False) -> dict[str, Any]:
    hits: list[bool] = []; margin_hits: dict[str, list[bool]] = {str(t): [] for t in MARGIN_THRESHOLDS}
    for row, score in zip(records, scores):
        if len(row["ids"]) != 2 or oracle_action_index(row["targets"]) == 0:
            continue
        predicted = int(np.argmin(score) if lower_is_better else np.argmax(score)); oracle = int(np.argmax(row["targets"])); hit = predicted == oracle; hits.append(hit)
        margin = float(oracle_margin(row["targets"])["margin_1"])
        for threshold in MARGIN_THRESHOLDS:
            if margin >= threshold: margin_hits[str(threshold)].append(hit)
    return {"overall": float(np.mean(hits)) if hits else None, "episode_count": len(hits), "high_margin": {key: {"count": len(values), "accuracy": float(np.mean(values)) if values else None} for key, values in margin_hits.items()}}


def _train_mlp(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(train_x.shape[1], 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.SmoothL1Loss(); x = torch.from_numpy(train_x.astype(np.float32)); y = torch.from_numpy(train_y.astype(np.float32)).reshape(-1, 1)
    final_loss = 0.0
    for _ in range(20):
        order = torch.randperm(len(x)); total = 0.0
        for start in range(0, len(x), 512):
            idx = order[start:start + 512]; loss = loss_fn(model(x[idx]), y[idx]); optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.detach()) * len(idx)
        final_loss = total / len(x)
    with torch.inference_mode(): prediction = model(torch.from_numpy(val_x.astype(np.float32))).reshape(-1).numpy()
    return prediction, {"architecture": f"Linear({train_x.shape[1]},128)->GELU->Linear(128,64)->GELU->Linear(64,1)", "epochs": 20, "batch_size": 512, "learning_rate": 1e-3, "loss": "SmoothL1Loss", "train_final_loss": final_loss}


def _train_samples(records: Sequence[Mapping[str, Any]], include_quality: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []; targets: list[float] = []; episode_ids: list[str] = []
    for row in records:
        for index in range(len(row["ids"])):
            item = np.concatenate([row["base"], row["geometry"][index], row["quality"][index] if include_quality else np.empty(0, dtype=np.float32)])
            features.append(item); targets.append(float(row["targets"][index])); episode_ids.append(row["episode_id"])
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), episode_ids


def run(data_root: Path, train_limit: int | None = None, val_limit: int | None = None) -> dict[str, Any]:
    train_rows, val_rows, context = _load_inputs(data_root); utilities = context["utilities"]
    if train_limit is not None: train_rows = train_rows[:train_limit]
    if val_limit is not None: val_rows = val_rows[:val_limit]
    train = _records(train_rows, utilities); val = _records(val_rows, utilities)
    # EXP032-B/C use the already frozen Stage-B ST-GCN diagnostics.  This is
    # deliberately not a newly invented forward pass or a perception rerun.
    quality_values = {name: [float(row["quality"][i, j]) for row in val for i in range(len(row["ids"]))] for j, name in enumerate(QUALITY_NAMES)}
    utility_values = [float(target) for row in val for target in row["targets"]]
    recognition_corr = {name: _corr(values, utility_values) for name, values in quality_values.items()}
    rec_winners = {name: _winner_metrics(val, [[row["quality"][i, j] for i in range(len(row["ids"]))] for row in val], lower_is_better=(name in ("future_ce", "future_entropy"))) for j, name in enumerate(QUALITY_NAMES)}
    deltas_q: dict[str, Any] = {}; delta_u: list[float] = []
    for row in val:
        if len(row["ids"]) == 2:
            delta_u.append(float(row["targets"][1] - row["targets"][0]))
    for j, name in enumerate(QUALITY_NAMES):
        delta_q = [float(row["quality"][1, j] - row["quality"][0, j]) for row in val if len(row["ids"]) == 2]
        deltas_q[name] = {**_corr(delta_q, delta_u), "sign_agreement": float(np.mean(np.sign(delta_q) == np.sign(delta_u))) if delta_q else None}
    exp032 = {
        "experiment_id": "EXP032", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": False, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False,
        "population": {"train": len(train), "val": len(val), "oracle_move_val": sum(len(r["ids"]) == 2 and oracle_action_index(r["targets"]) > 0 for r in val)},
        "method_a_pose_error": {"status": "EXP032_A_STATUS=BLOCKED", "reason": "repository defines H36M-17 estimated order and meter camera coordinates but exposes no canonical GT-to-H36M17 joint mapping/coordinate conversion; MPJPE would require guessing"},
        "method_b_future_recognition": {"status": "PASS_CANONICAL_ARTIFACT", "diagnostics_source": "frozen Stage-B utility labels generated by frozen ST-GCN", "feature_names": QUALITY_NAMES, "feature_correlations": recognition_corr, "oracle_move_winner": rec_winners, "delta_quality_vs_delta_utility": deltas_q, "future_candidate_skeleton_used": True, "future_stgcn_inference_used": True, "deployable": False, "independent_forward_rerun": False, "unavailable_features": ["top1_probability", "top1_top2_margin"]},
        "method_c_pose_to_recognition": {"status": "BLOCKED_DEPENDS_ON_EXP032_A", "reason": "true pose error unavailable"},
        "method_d_32_view_quality_field": {"status": "NOT_RUN", "reason": "no new frozen ST-GCN forward pass was introduced; canonical Stage-B artifacts are candidate scoped"},
        "leakage_flags": {"future_candidate_skeleton_used": True, "oracle_perception_quality_upper_bound": True, "deployable": False, "test_used": False},
    }
    # EXP033 legal-input predictors: candidate geometry + current Stage-D
    # state only. Future quality is a target, never an input.
    tr_x, _, _ = _train_samples(train, include_quality=False); va_x, _, _ = _train_samples(val, include_quality=False); tr_x, va_x = _standardize(tr_x, va_x)
    tr_ce, _, _ = _train_samples(train, include_quality=False); va_ce, _, val_ids = _train_samples(val, include_quality=False)
    train_ce = np.asarray([float(row["quality"][i, 0]) for row in train for i in range(len(row["ids"]))], dtype=np.float32); val_ce = np.asarray([float(row["quality"][i, 0]) for row in val for i in range(len(row["ids"]))], dtype=np.float32)
    train_u = np.asarray([float(row["targets"][i]) for row in train for i in range(len(row["ids"]))], dtype=np.float32); val_u = np.asarray([float(row["targets"][i]) for row in val for i in range(len(row["ids"]))], dtype=np.float32)
    pred_ce, ce_info = _train_mlp(tr_x, train_ce, va_x); pred_u, u_info = _train_mlp(tr_x, train_u, va_x)
    def episode_scores(prediction: np.ndarray) -> list[np.ndarray]:
        output: list[np.ndarray] = []; offset = 0
        for row in val:
            output.append(prediction[offset:offset + len(row["ids"])]); offset += len(row["ids"])
        return output
    ce_scores = episode_scores(pred_ce); u_scores = episode_scores(pred_u)
    exp033 = {"experiment_id": "EXP033", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": True, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "input": {"features": "legal Stage-D current state + candidate geometry", "future_candidate_skeleton_used_at_inference": False, "future_recognition_output_used_at_inference": False, "true_utility_used_as_input": False}, "pose_error_prediction": {"status": "SKIPPED_DEPENDS_ON_EXP032_A"}, "recognition_ce_prediction": {"model": ce_info, "candidate_level": _metrics(pred_ce, val_ce), "oracle_move_winner": _winner_metrics(val, ce_scores, lower_is_better=True)}, "multi_task": {"status": "SKIPPED_DEPENDS_ON_EXP032_A"}, "direct_utility_prediction": {"model": u_info, "candidate_level": _metrics(pred_u, val_u), "oracle_move_winner": _winner_metrics(val, u_scores), "selected_candidate_mean_true_utility": float(np.mean([row["targets"][int(np.argmax(scores))] for row, scores in zip(val, u_scores)]))}, "leakage_flags": {"future_candidate_rgb_used": False, "future_candidate_depth_used": False, "future_candidate_semantic_used": False, "future_candidate_skeleton_used_at_inference": False, "future_recognition_output_used_at_inference": False, "future_candidate_skeleton_used_as_train_target": True, "future_recognition_quality_used_as_train_target": True, "test_used": False}}
    # EXP034 fixed ridge and conditional variance audit.
    def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1], dtype=np.float32), x.T @ y)
    raw_base_tr, _, _ = _train_samples(train, include_quality=False)
    raw_base_va, _, _ = _train_samples(val, include_quality=False)
    base_tr, base_va = _standardize(raw_base_tr, raw_base_va)
    rec_tr, _, _ = _train_samples(train, include_quality=True); rec_va, _, _ = _train_samples(val, include_quality=True); rec_tr, rec_va = _standardize(rec_tr, rec_va)
    beta0 = ridge_fit(np.c_[np.ones(len(base_tr)), base_tr], train_u); beta2 = ridge_fit(np.c_[np.ones(len(rec_tr)), rec_tr], train_u)
    exp034_models = {"model_0_current_legal_base": _r2_metrics(np.c_[np.ones(len(base_va)), base_va] @ beta0, val_u), "model_1_true_pose_error": {"status": "BLOCKED"}, "model_2_future_recognition_quality": _r2_metrics(np.c_[np.ones(len(rec_va)), rec_va] @ beta2, val_u), "model_3_pose_plus_recognition": {"status": "BLOCKED"}}
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"utility": [], "winner": []})
    for row in val:
        keys = {"action_class": str(row["label_id"]), "scene": row["scene_id"], "region": row["region"], "current_correct": str(row["current_quality"]["correct"]), "current_confidence_quartile": str(min(3, int(float(row["current_quality"]["entropy"]) * 4.0)))}
        for name, key in keys.items(): groups[f"{name}={key}"]["utility"].extend(float(v) for v in row["targets"]); groups[f"{name}={key}"]["winner"].append(float(oracle_action_index(row["targets"]) > 0))
    variance = {key: {"n": len(value["utility"]), "mean": float(np.mean(value["utility"])), "std": float(np.std(value["utility"])), "iqr": float(np.percentile(value["utility"], 75) - np.percentile(value["utility"], 25)), "move_fraction": float(np.mean(value["winner"]))} for key, value in groups.items()}
    context_groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(val): context_groups[tuple(np.round(row["base"], 1))].append(oracle_action_index(row["targets"]))
    duplicate = [values for values in context_groups.values() if len(values) > 1]
    switch = [len(set(values)) > 1 for values in duplicate]
    exp034 = {"experiment_id": "EXP034", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": False, "perception_regenerated": False, "habitat_rendering_performed": False, "stgcn_retrained": False, "models": exp034_models, "incremental_delta_r2": {"model_2_minus_model_0": (exp034_models["model_2_future_recognition_quality"].get("r2") - exp034_models["model_0_current_legal_base"].get("r2")) if exp034_models["model_2_future_recognition_quality"].get("r2") is not None and exp034_models["model_0_current_legal_base"].get("r2") is not None else None}, "conditional_utility_variance": variance, "same_context_cross_motion_instability": {"definition": "rounded current legal base context; duplicate groups only", "duplicate_group_count": len(duplicate), "three_way_switch_rate": float(np.mean(switch)) if switch else None, "binary_switch_rate": float(np.mean([len({int(v > 0) for v in values}) > 1 for values in duplicate])) if duplicate else None}, "leakage_flags": {"future_candidate_skeleton_used": False, "future_recognition_quality_used_as_model_input": False, "test_used": False}}
    train_feature_path = Path(context["summary"]["feature_files"]["train"])
    provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(), "stage_d_feature_summary_sha256": file_sha256(train_feature_path.parent.parent / "stage_d_feature_summary.json"), "stage_b_train_utility_sha256": file_sha256(data_root / "datasets/policy_v11_5/stage_b/utility_labels/train.jsonl"), "stage_b_val_utility_sha256": file_sha256(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")}
    exp032["provenance"] = provenance
    exp033["provenance"] = provenance
    exp034["provenance"] = provenance
    results = {"EXP032": exp032, "EXP033": exp033, "EXP034": exp034}
    for directory, result in ((EXP032, exp032), (EXP033, exp033), (EXP034, exp034)):
        directory.mkdir(parents=True, exist_ok=True); (directory / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); (directory / "analysis.md").write_text(f"# {result['experiment_id']} overnight audit\n\n```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```\n", encoding="utf-8"); (directory / "feature_summary.json").write_text(json.dumps({"experiment_id": result["experiment_id"], "train_episodes": len(train), "val_episodes": len(val), "test_used": False}, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=Path("../../data/ActiveView")); parser.add_argument("--train-limit", type=int, default=None); parser.add_argument("--val-limit", type=int, default=None); args = parser.parse_args(); results = run(args.data_root.resolve(), args.train_limit, args.val_limit); print(json.dumps({key: value["status"] for key, value in results.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
