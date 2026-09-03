#!/usr/bin/env python3
"""EXP051-R2: paired real-observation H1/H2 evaluation.

The evaluator remains frozen for Val and accepts ``split="test"`` only for
the explicitly unlocked EXP057 final runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import binom

from activeview.active_view.data import episode_sources as _episode_sources, load_jsonl, load_stage_d_cache as _load_cache, load_stgcn as _load_stgcn, load_wm_e as _load_wm_e, log_probs as _log_probs, rows
from activeview.active_view.geometry import context_key, relative_view_descriptor, load_pairwise_and_azimuths as _load_pairwise_and_azimuths
from activeview.active_view.joint_revision import JointRevision as _JointRevision, select_actions as _joint_select, tie_argmax as _tie_argmax
from activeview.active_view.rollout_support import (
    _candidate_order,
    _joint_choice,
    _load_sources,
    build_rgb_cache,
    VisitedObservationStore,
)

_rows = rows
from activeview.active_view.stage_d_rgb_context import RGBObservationKey, load_dinov2, observation_keys_from_feature_rows
from activeview.active_view.stage_d_rgb_spatial import build_or_load_spatial_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 16
VIEW_COUNT = 32


def _build_history_rgb_cache(rows: Sequence[Mapping[str, Any]], device: torch.device, cache_dir: Path, extra_views: Mapping[str, int | None] | None = None) -> tuple[dict[tuple[str, str, str, int], np.ndarray], dict[str, Any]]:
    keys, _ = observation_keys_from_feature_rows(rows)
    if extra_views:
        expanded = set(keys)
        for row in rows:
            selected = extra_views.get(str(row["episode_id"]))
            if selected is not None:
                expanded.add(RGBObservationKey(str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(selected)))
        keys = sorted(expanded, key=lambda item: item.tuple)
    values, manifest, info = build_or_load_spatial_cache(
        rgb_root=Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train"),
        cache_dir=cache_dir,
        keys=keys,
        model_loader=load_dinov2,
        device=device,
        batch_size=64,
    )
    lookup = {(str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"])): np.asarray(values[index], dtype=np.float32) for index, row in enumerate(manifest)}
    return lookup, {**info, "contexts": len(rows), "views": len(keys), "future_candidate_rgb_used": False}


def _split_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    """Load an explicitly labelled split; Test is only opened by EXP057."""
    if split == "test":
        path = data_root / "datasets/policy_v11_5/stage_d/EXP014_two_step_sequential/features/test.jsonl"
        rows = load_jsonl(path)
        if any(str(row.get("policy_split", "")).lower() != "test" for row in rows):
            raise ValueError(f"explicit policy_split=test required: {path}")
        return rows
    return _rows(data_root, split)


def _split_sources(data_root: Path, split: str) -> dict[tuple[str, str, str], str]:
    if split == "test":
        path = data_root / "datasets/policy_v11_5/episodes/test_episodes.jsonl"
        episodes = load_jsonl(path)
        result: dict[tuple[str, str, str], str] = {}
        for episode in episodes:
            if str(episode.get("policy_split", "")).lower() != "test":
                raise ValueError(f"explicit policy_split=test required: {path}")
            key = context_key(episode)
            source = str(episode["current_view"]["skeleton_source_path"])
            if key in result and Path(result[key]).resolve() != Path(source).resolve():
                raise ValueError(f"source path mismatch for {key}")
            result[key] = source
        return result
    return _episode_sources(data_root, split)


def _macro_f1(pred: Sequence[int], target: Sequence[int]) -> float:
    p, y = np.asarray(pred, dtype=np.int64), np.asarray(target, dtype=np.int64)
    confusion = np.bincount(y * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    values: list[float] = []
    for cls in range(N_CLASSES):
        tp = float(confusion[cls, cls]); pd = float(confusion[:, cls].sum()); rd = float(confusion[cls].sum())
        precision = tp / pd if pd else 0.0; recall = tp / rd if rd else 0.0
        values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def _classification(pred: Sequence[int], target: Sequence[int]) -> dict[str, float]:
    p, y = np.asarray(pred), np.asarray(target)
    return {"accuracy": float(np.mean(p == y)), "macro_f1": _macro_f1(p, y)}


def _corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _fidelity(imagined: list[np.ndarray], truth: list[np.ndarray], labels: Sequence[int]) -> dict[str, float | int | None]:
    if not imagined:
        return {"count": 0, "top1_agreement": None, "true_label_logp_pearson": None, "true_label_logp_spearman": None, "kl_true_to_imagined": None, "probability_l1": None}
    imag = np.concatenate([x.reshape(-1, N_CLASSES) for x in imagined]); real = np.concatenate([x.reshape(-1, N_CLASSES) for x in truth])
    label_array = np.asarray(labels, dtype=np.int64)
    repeat = np.asarray([len(x) for x in imagined], dtype=np.int64)
    repeated_labels = np.repeat(label_array, repeat)
    p_imag, p_real = np.exp(imag), np.exp(real)
    true_imag = imag[np.arange(len(imag)), repeated_labels]; true_real = real[np.arange(len(real)), repeated_labels]
    rank_imag = np.argsort(np.argsort(true_imag)); rank_real = np.argsort(np.argsort(true_real))
    kl = np.sum(p_real * (real - imag), axis=1)
    return {"count": int(len(imag)), "top1_agreement": float(np.mean(np.argmax(imag, axis=1) == np.argmax(real, axis=1))), "true_label_logp_pearson": _corr(true_imag, true_real), "true_label_logp_spearman": _corr(rank_imag, rank_real), "kl_true_to_imagined": float(np.mean(kl)), "probability_l1": float(np.mean(np.sum(np.abs(p_real - p_imag), axis=1)))}


def _paired(pred_h1: Sequence[int], pred_h2: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    p1, p2, y = np.asarray(pred_h1), np.asarray(pred_h2), np.asarray(labels)
    c1, c2 = p1 == y, p2 == y; rescued = int(np.sum(~c1 & c2)); harmful = int(np.sum(c1 & ~c2)); n = rescued + harmful
    p_value = float(1.0 if n == 0 else min(1.0, 2.0 * binom.cdf(min(rescued, harmful), n, 0.5)))
    rng = np.random.default_rng(42); acc_delta = (c2.astype(float) - c1.astype(float)); boot_acc = np.empty(10000); boot_f1 = np.empty(10000)
    for start in range(0, 10000, 500):
        count = min(500, 10000 - start); indices = rng.integers(0, len(y), size=(count, len(y))); boot_acc[start:start + count] = np.mean(acc_delta[indices], axis=1)
        for offset, sample in enumerate(indices): boot_f1[start + offset] = _macro_f1(p2[sample], y[sample]) - _macro_f1(p1[sample], y[sample])
    return {"n": int(len(y)), "rescued": rescued, "harmful": harmful, "net": rescued - harmful, "mcnemar_p": p_value, "delta_accuracy": float(np.mean(acc_delta)), "delta_macro_f1": float(_macro_f1(p2, y) - _macro_f1(p1, y)), "accuracy_ci95": [float(np.quantile(boot_acc, .025)), float(np.quantile(boot_acc, .975))], "macro_f1_ci95": [float(np.quantile(boot_f1, .025)), float(np.quantile(boot_f1, .975))], "bootstrap_replicates": 10000, "seed": 42}


def run_real_observation_evaluation(data_root: Path, checkpoint: Path, device: torch.device, wm_checkpoint: Path | None = None, output_dir: Path | None = None, split: str = "val", rgb_cache_dir: Path | None = None) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if wm_checkpoint is None:
        wm_checkpoint = Path("/home/zxf/WorkSpace/code/data/ActiveView/experiments/stage_d/EXP057_final_method_freeze/runtime/wm_e_frozen.pth")
    rows = _split_rows(data_root, split); sources = _split_sources(data_root, split)
    if split == "val" and len(rows) != 9742: raise RuntimeError("canonical moving Val population mismatch")
    cache = _load_cache(data_root / f"experiments/stage_d/EXP046_counterfactual_recognition_dataset/{split}_cache.npz")
    v0_rows = load_jsonl(data_root / f"experiments/stage_d/EXP014_two_step_sequential/v0_predictions/{split}_predictions.jsonl"); v0 = {str(r["episode_id"]): r for r in v0_rows}
    pairwise, azimuths = _load_pairwise_and_azimuths(data_root, rows, sources)
    orders = {str(r["episode_id"]): ([] if bool(v0[str(r["episode_id"])] ["predicted_stays"]) else _candidate_order(r, int(r["s1_viewpoint_id"]), {int(r["s0_viewpoint_id"]), int(r["s1_viewpoint_id"])}, pairwise[(str(r["scene_id"]), str(r["region"]))], azimuths[(str(r["scene_id"]), str(r["region"]))])) for r in rows}
    joint_payload = torch.load(checkpoint, map_location=device, weights_only=False); joint = _JointRevision().to(device); joint.load_state_dict(joint_payload["model_state_dict"]); joint.eval()
    wm = _load_wm_e(wm_checkpoint, device); stgcn = _load_stgcn(data_root, device)
    initial_selected = _joint_select(joint, cache, rows, orders, "ALL_LEGAL", device); selected_by_episode = {str(r["episode_id"]): v for r, v in zip(rows, initial_selected)}
    if rgb_cache_dir is None:
        rgb_lookup, rgb_audit = build_rgb_cache(data_root, rows, sources, device, Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_allview_exp051_r1"))
    else:
        rgb_lookup, rgb_audit = _build_history_rgb_cache(rows, device, rgb_cache_dir, extra_views=selected_by_episode)
    cache_indices = {str(v): i for i, v in enumerate(cache["episode_ids"].tolist())}
    labels: list[int] = []; moving_labels: list[int] = []; h1_terminal: list[int] = []; h2_terminal: list[int] = []; h1_fused: list[int] = []; h2_fused: list[int] = []; h1_moves: list[int] = []; h2_moves: list[int] = []; h2_action_signatures: list[list[int]] = []
    h0_imag: list[np.ndarray] = []; h0_true: list[np.ndarray] = []; h0_labels: list[int] = []; h1_imag: list[np.ndarray] = []; h1_true: list[np.ndarray] = []; h1_labels: list[int] = []
    for row in rows:
        key = context_key(row); idx = cache_indices[str(row["episode_id"])]
        archive = _load_sources(Path(sources[key]), {"true_logp": cache["true_logp"][idx], "viewpoint_ids": np.arange(VIEW_COUNT)}, key); positions = archive["positions"]; s0, s1 = int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]); label = int(row["label_id"]); labels.append(label)
        store = VisitedObservationStore(Path(sources[key]), archive, rgb_lookup, key); obs0, obs1 = store.reveal(s0), store.reveal(s1); order = list(orders[str(row["episode_id"])])
        selected = selected_by_episode[str(row["episode_id"])]
        if order:
            hs0 = torch.from_numpy(np.stack([obs0.skeleton, obs1.skeleton])[None]).float().to(device)
            hd0 = torch.from_numpy(np.stack([relative_view_descriptor(positions, positions[s1], v) for v in (s0, s1)])[None]).float().to(device)
            hb0 = torch.from_numpy(np.concatenate([obs0.logp, obs1.logp])[None]).float().to(device)
            hr0 = torch.from_numpy(np.stack([obs0.rgb, obs1.rgb])[None]).float().to(device)
            cd0_np = np.stack([relative_view_descriptor(positions, positions[s1], v) for v in order])
            with torch.inference_mode():
                imagined_s0 = wm(hs0, hd0, torch.from_numpy(cd0_np[None]).float().to(device), history_belief=hb0, history_rgb=hr0).cpu().numpy()[0]
            ordered_imag = _log_probs(stgcn, imagined_s0, device)
            ordered_true = archive["logp"][np.asarray(order, dtype=np.int64)]
            h0_imag.append(ordered_imag); h0_true.append(ordered_true); h0_labels.append(label); moving_labels.append(label)
        terminal_h1 = s1 if selected is None else int(selected); h1_moves.append(int(selected is not None)); h1_terminal.append(int(np.argmax(archive["logp"][terminal_h1]))); visited_h1 = [s0, s1] if selected is None else [s0, s1, terminal_h1]; h1_fused.append(int(np.argmax(np.mean(np.exp(archive["logp"][visited_h1]), axis=0))))
        h2_visited = [s0, s1]; h2_current = s1; h2_count = 0
        if selected is not None:
            revealed = store.reveal(int(selected)); h2_visited.append(int(selected)); h2_current = int(selected); h2_count = 1
            remaining = _candidate_order(row, h2_current, set(h2_visited), pairwise[(key[0], key[1])], azimuths[(key[0], key[1])])
            if remaining:
                hs = torch.from_numpy(np.stack([obs1.skeleton, revealed.skeleton])[None]).float().to(device); hd = torch.from_numpy(np.stack([relative_view_descriptor(positions, positions[h2_current], v) for v in (s1, h2_current)])[None]).float().to(device); hb = torch.from_numpy(np.concatenate([obs1.logp, revealed.logp])[None]).float().to(device); hr = torch.from_numpy(np.stack([obs1.rgb, revealed.rgb])[None]).float().to(device); cd_np = np.stack([relative_view_descriptor(positions, positions[h2_current], v) for v in remaining]); cd = torch.from_numpy(cd_np[None]).float().to(device)
                with torch.inference_mode(): imagined_s = wm(hs, hd, cd, history_belief=hb, history_rgb=hr).cpu().numpy()[0]
                imagined_logs = _log_probs(stgcn, imagined_s, device); true_logs = archive["logp"][np.asarray(remaining, dtype=np.int64)]; h1_imag.append(imagined_logs); h1_true.append(true_logs); h1_labels.append(label)
                second_choice = _joint_choice(joint, (obs1.logp, revealed.logp), imagined_logs, cd_np, device)
                if second_choice is not None:
                    v2 = int(remaining[second_choice])
                    # No WM inference follows the terminal second move, so the
                    # archived skeleton/log-probability is sufficient here;
                    # avoid loading an unneeded terminal RGB frame.
                    h2_visited.append(v2); h2_current = v2; h2_count = 2
        h2_moves.append(h2_count); h2_action_signatures.append([int(value) for value in h2_visited[2:]]); h2_terminal.append(int(np.argmax(archive["logp"][h2_current]))); h2_fused.append(int(np.argmax(np.mean(np.exp(archive["logp"][h2_visited]), axis=0))))
    moving_ids = [str(r["episode_id"]) for r in rows]
    by_id = {episode_id: index for index, episode_id in enumerate(moving_ids)}
    full_labels: list[int] = []
    full_h1: list[int] = []
    full_h2: list[int] = []
    full_fused_h1: list[int] = []
    full_fused_h2: list[int] = []
    stay_count = 0
    for v0_row in v0_rows:
        episode_id = str(v0_row["episode_id"])
        full_labels.append(int(v0_row["label_id"]))
        if bool(v0_row["predicted_stays"]):
            stay_count += 1
            prediction = int(v0_row["current_predicted_label_id"])
            full_h1.append(prediction); full_h2.append(prediction)
            full_fused_h1.append(prediction); full_fused_h2.append(prediction)
        else:
            index = by_id[episode_id]
            full_h1.append(int(h1_terminal[index])); full_h2.append(int(h2_terminal[index]))
            full_fused_h1.append(int(h1_fused[index])); full_fused_h2.append(int(h2_fused[index]))
    if split == "val" and (len(full_labels) != 13987 or stay_count != 4245):
        raise RuntimeError(f"canonical full Val population mismatch: {len(full_labels)} rows, {stay_count} v0 stays")
    result = {"experiment_id": "EXP051-R2", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": False, "terminal_recognition_source": "frozen EXP046 true_logp from archived skeleton through frozen ST-GCN", "first_step_policy_source": "frozen EXP050-R1/Stage C-v0", "population": {"full": len(full_labels), "moving_subset": len(rows), "v0_stay": stay_count}, "h1_real": {"full": _classification(full_h1, full_labels), "moving_subset": _classification(h1_terminal, labels)}, "h2_real": {"full": _classification(full_h2, full_labels), "moving_subset": _classification(h2_terminal, labels)}, "fused": {"h1_full": _classification(full_fused_h1, full_labels), "h1_moving_subset": _classification(h1_fused, labels), "h2_full": _classification(full_fused_h2, full_labels), "h2_moving_subset": _classification(h2_fused, labels)}, "paired_moving_subset": _paired(h1_terminal, h2_terminal, labels), "paired_fused_moving_subset": _paired(h1_fused, h2_fused, labels), "history_shift_fidelity": {"h0": _fidelity(h0_imag, h0_true, h0_labels), "h1": _fidelity(h1_imag, h1_true, h1_labels), "recurrent_episode_count": len(h1_imag)}, "motion": {"h1_average_moves": float(np.mean(h1_moves)), "h2_average_moves": float(np.mean(h2_moves))}, "h2_action_signatures_moving": h2_action_signatures, "h2_terminal_predictions_moving": [int(value) for value in h2_terminal], "moving_labels": [int(value) for value in labels], "rgb_history_artifact_audit": rgb_audit, "leakage_flags": {"future_rgb_access_before_execution": False, "future_skeleton_access_before_execution": False, "visited_rgb_revealed_only_after_execution": True, "visited_skeleton_revealed_only_after_execution": True, "habitat_rendering_performed": False, "perception_regenerated": False, "wm_e_frozen": True, "joint_revision_frozen": True, "stgcn_frozen": True}}
    result["split"] = split
    result["test_used"] = split == "test"
    out = output_dir or (REPO_ROOT / "experiments/stage_d/EXP051_R2_real_observation_evaluation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "paired_statistics.json").write_text(json.dumps({"terminal": result["paired_moving_subset"], "fused": result["paired_fused_moving_subset"]}, indent=2) + "\n", encoding="utf-8")
    (out / "history_shift_fidelity.json").write_text(json.dumps(result["history_shift_fidelity"], indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--wm-checkpoint", type=Path, default=Path("/tmp/activeview_exp042r1_E/last.pth")); parser.add_argument("--output-dir", type=Path); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--split", choices=("val", "test"), default="val"); args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    print(json.dumps(run_real_observation_evaluation(args.data_root.resolve(), args.checkpoint.resolve(), device, args.wm_checkpoint.resolve(), args.output_dir.resolve() if args.output_dir else None, split=args.split), indent=2))


if __name__ == "__main__": main()
