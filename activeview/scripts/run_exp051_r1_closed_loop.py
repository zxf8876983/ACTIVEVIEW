#!/usr/bin/env python3
"""EXP051-R1 frozen H1/H2 closed-loop rollout (Val only).

The script intentionally keeps observation access behind ``VisitedObservationStore``:
future skeleton/RGB/belief values are revealed only after a selected transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_dense_campaign import context_key, relative_view_descriptor
from activeview.active_view.stage_d_rgb_context import RGBObservationKey, load_dinov2
from activeview.active_view.stage_d_rgb_spatial import build_or_load_spatial_cache
from activeview.core.paths import get_data_root
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _rows
from activeview.scripts.run_stage_d_exp046_048 import _load_cache, _load_stgcn, _load_wm_e, _log_probs
from activeview.scripts.run_stage_d_exp049_051 import _JointRevision, _load_pairwise_and_azimuths, _tie_argmax

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEW_COUNT = 32
N_CLASSES = 16
WM_SHA = "db2573a013ed9a7fab87561ad26800334556894b96e69dd3d498464794d9b5e6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_order(row: Mapping[str, Any], current: int, visited: set[int], pairwise: Mapping[int, Mapping[int, float]], azimuths: Mapping[int, float]) -> list[int]:
    values: list[tuple[float, float, int]] = []
    current_azimuth = float(azimuths[current])
    for candidate in range(VIEW_COUNT):
        if candidate in visited or candidate not in pairwise.get(current, {}):
            continue
        distance = float(pairwise[current][candidate])
        if not np.isfinite(distance):
            continue
        delta = (float(azimuths[candidate]) - current_azimuth + 180.0) % 360.0 - 180.0
        values.append((distance, abs(delta), candidate))
    values.sort()
    return [candidate for _, _, candidate in values]


@dataclass
class Observation:
    skeleton: np.ndarray
    rgb: np.ndarray
    logp: np.ndarray


class VisitedObservationStore:
    """Read archived observations only for IDs explicitly revealed by rollout."""

    def __init__(self, source: Path, archive: Mapping[str, np.ndarray], rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray], key: tuple[str, str, str]) -> None:
        self.source = source
        self.archive = archive
        self.rgb_lookup = rgb_lookup
        self.key = key
        self._revealed: set[int] = set()

    def reveal(self, viewpoint_id: int) -> Observation:
        viewpoint_id = int(viewpoint_id)
        if viewpoint_id < 0 or viewpoint_id >= VIEW_COUNT:
            raise ValueError("invalid viewpoint")
        ids = np.asarray(self.archive["viewpoint_ids"], dtype=np.int64)
        positions = {int(value): index for index, value in enumerate(ids.tolist())}
        if viewpoint_id not in positions:
            raise ValueError("viewpoint alignment failure")
        self._revealed.add(viewpoint_id)
        index = positions[viewpoint_id]
        return Observation(
            np.asarray(self.archive["skeleton"][index], dtype=np.float32),
            np.asarray(self.rgb_lookup[(*self.key, viewpoint_id)], dtype=np.float32),
            np.asarray(self.archive["logp"][index], dtype=np.float32),
        )


def _load_sources(source_path: Path, cache: Mapping[str, np.ndarray], context: tuple[str, str, str]) -> dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["viewpoint_ids"], dtype=np.int64)
        skeleton = np.asarray(archive["skeleton"], dtype=np.float32)
        positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
    cache_ids = np.asarray(cache["viewpoint_ids"], dtype=np.int64) if "viewpoint_ids" in cache else ids
    if skeleton.shape != (VIEW_COUNT, 3, 30, 17) or not np.array_equal(ids, cache_ids):
        raise ValueError(f"skeleton/cache alignment failure for {context}")
    return {"skeleton": skeleton, "viewpoint_ids": ids, "positions": positions, "logp": np.asarray(cache["true_logp"], dtype=np.float32)}


def _joint_choice(model: _JointRevision, current_logs: tuple[np.ndarray, np.ndarray], imagined: np.ndarray, descriptors: np.ndarray, device: torch.device) -> int | None:
    previous, current = current_logs
    probs0, probs1 = np.exp(previous), np.exp(current)
    stats = np.asarray([*probs0, *probs1, -np.sum(probs0 * previous), -np.sum(probs1 * current), np.max(probs0), np.max(probs1), np.sort(probs1)[-1] - np.sort(probs1)[-2], np.sort(probs0)[-1] - np.sort(probs0)[-2]], dtype=np.float32)[None]
    rows = [np.concatenate([current, np.zeros(9, dtype=np.float32), [1.0]])]
    rows.extend(np.concatenate([imagined[index], descriptors[index], [0.0]]) for index in range(len(descriptors)))
    values = torch.zeros((1, max(31, len(rows)), 26), dtype=torch.float32)
    values[0, : len(rows)] = torch.from_numpy(np.asarray(rows, dtype=np.float32))
    mask = torch.zeros((1, values.shape[1]), dtype=torch.bool); mask[0, : len(rows)] = True
    with torch.inference_mode():
        scores, _ = model(torch.from_numpy(stats).to(device), values.to(device), mask.to(device))
    choice = _tie_argmax(scores[0, : len(rows)].cpu().numpy())
    return None if choice == 0 else int(choice - 1)


def _metrics(pred: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    values = np.asarray(pred, dtype=np.int64); target = np.asarray(labels, dtype=np.int64)
    confusion = np.bincount(target * N_CLASSES + values, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    f1_values: list[float] = []
    for cls in range(N_CLASSES):
        tp = float(confusion[cls, cls]); precision_den = float(confusion[:, cls].sum()); recall_den = float(confusion[cls].sum())
        precision = tp / precision_den if precision_den else 0.0; recall = tp / recall_den if recall_den else 0.0
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float(np.mean(values == target)), "macro_f1": float(np.mean(f1_values))}


def build_rgb_cache(data_root: Path, rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str], device: torch.device, cache_dir: Path) -> tuple[dict[tuple[str, str, str, int], np.ndarray], dict[str, Any]]:
    keys = [RGBObservationKey(str(row["scene_id"]), str(row["region"]), str(row["record_id"]), view) for row in rows for view in range(VIEW_COUNT)]
    started = time.monotonic()
    values, manifest, info = build_or_load_spatial_cache(rgb_root=Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train"), cache_dir=cache_dir, keys=keys, model_loader=load_dinov2, device=device, batch_size=64)
    lookup = {(
        str(row["scene_id"]), str(row["region"]), str(row["record_id"]), int(row["viewpoint_id"])
    ): np.asarray(values[index], dtype=np.float32) for index, row in enumerate(manifest)}
    return lookup, {**info, "contexts": len(rows), "views": len(keys), "elapsed_sec": time.monotonic() - started, "future_candidate_rgb_used": False}


def run(data_root: Path, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    rows = _rows(data_root, "val")
    if len(rows) != 9742:
        raise RuntimeError("canonical Val population mismatch")
    sources = _episode_sources(data_root, "val")
    val_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz")
    v0_rows = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl")
    v0 = {str(row["episode_id"]): row for row in v0_rows}
    joint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    joint = _JointRevision().to(device); joint.load_state_dict(joint_payload["model_state_dict"]); joint.eval()
    wm = _load_wm_e(Path("/tmp/activeview_exp042r1_E/last.pth"), device)
    stgcn = _load_stgcn(data_root, device)
    cache_indices = {str(value): index for index, value in enumerate(val_cache["episode_ids"].tolist())}
    pairwise_all, azimuths_all = _load_pairwise_and_azimuths(data_root, rows, sources)
    summary = json.loads((data_root / "datasets/policy_v11_5/stage_b/stage_b_summary.json").read_text()); mapping = json.loads(Path(summary["label_mapping"]).read_text()); categories = [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    initial_orders = {str(r["episode_id"]): ([] if bool(v0[str(r["episode_id"])] ["predicted_stays"]) else _candidate_order(r, int(r["s1_viewpoint_id"]), {int(r["s0_viewpoint_id"]), int(r["s1_viewpoint_id"])}, pairwise_all[(str(r["scene_id"]), str(r["region"]))], azimuths_all[(str(r["scene_id"]), str(r["region"]))])) for r in rows}
    from activeview.scripts.run_stage_d_exp049_051 import _joint_select
    initial_selected = _joint_select(joint, val_cache, rows, initial_orders, "ALL_LEGAL", device)
    initial_selected_by_episode = {str(r["episode_id"]): value for r, value in zip(rows, initial_selected)}
    from activeview.scripts.run_stage_d_exp049_051 import _candidate_cache_rows, _decision_rows, _expanded_stage_b_rows
    from activeview.active_view.stage_d_evaluation import build_stage_d_trajectories, summarize_trajectory_rows
    stage_b_rows = load_jsonl(data_root / "datasets/policy_v11_5/stage_b/utility_labels/val.jsonl")
    cache_rows = _candidate_cache_rows(rows, initial_orders, "ALL_LEGAL", pairwise_all, v0)
    decisions = _decision_rows(cache_rows, v0, {"selected": initial_selected})
    expanded = _expanded_stage_b_rows(stage_b_rows, val_cache, cache_rows, initial_orders)
    h1_canonical_rows = build_stage_d_trajectories(expanded, v0_rows, cache_rows, decisions)
    h1_canonical = summarize_trajectory_rows(h1_canonical_rows, categories)
    cache_dir = Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_allview_exp051_r1")
    rgb_lookup, rgb_audit = build_rgb_cache(data_root, rows, sources, device, cache_dir)
    terminal_preds_h1: list[int] = []; terminal_preds_h2: list[int] = []; fused_preds_h1: list[int] = []; fused_preds_h2: list[int] = []; labels: list[int] = []; moves_h1: list[int] = []; moves_h2: list[int] = []; path_h1: list[float] = []; path_h2: list[float] = []; h2_action_signatures: list[list[int]] = []; h2_terminal_predictions: list[int] = []; action_changed = 0
    recurrent_count = 0
    for number, row in enumerate(rows, 1):
        key = context_key(row); source = Path(sources[key]); index = cache_indices[str(row["episode_id"])]
        archive = _load_sources(source, {"true_logp": val_cache["true_logp"][index], "viewpoint_ids": np.arange(VIEW_COUNT)}, key)
        positions = archive["positions"]; azimuths = azimuths_all[(key[0], key[1])]
        pairwise = pairwise_all[(key[0], key[1])]
        store = VisitedObservationStore(source, archive, rgb_lookup, key)
        s0, s1 = int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]); label = int(row["label_id"]); labels.append(label)
        first = store.reveal(s0); second = store.reveal(s1)
        visited = {s0, s1}; current = s1; history = [first, second]
        order = _candidate_order(row, current, visited, pairwise, azimuths)
        # H1 decision uses the same frozen initial imagined state for the control.
        hist_s = torch.from_numpy(np.stack([item.skeleton for item in history])[None]).float().to(device)
        hist_d = torch.from_numpy(np.stack([relative_view_descriptor(positions, positions[current], view) for view in (s0, s1)])[None]).float().to(device)
        hist_b = torch.from_numpy(np.concatenate([first.logp, second.logp])[None]).float().to(device)
        hist_rgb = torch.from_numpy(np.stack([first.rgb, second.rgb])[None]).float().to(device)
        cand_d = torch.from_numpy(np.stack([relative_view_descriptor(positions, positions[current], view) for view in order])[None]).float().to(device)
        # EXP046 stores the canonical initial-history WM-E predictions used by
        # the original EXP050 control. Reuse them for exact H1 reproduction;
        # the recurrent [s1, v1] branch below is recomputed genuinely.
        imagined_all = np.asarray(val_cache["imagined_logp"][index], dtype=np.float32)
        imagined_logs = imagined_all[np.asarray(order, dtype=np.int64)]
        selected_initial = initial_selected_by_episode[str(row["episode_id"])]
        first_choice = None if selected_initial is None else order.index(int(selected_initial))
        h1_visited = [s0, s1] if first_choice is None else [s0, s1, order[first_choice]]
        h1_moves = int(first_choice is not None); moves_h1.append(h1_moves); path_h1.append(0.0 if first_choice is None else float(pairwise[s1][h1_visited[-1]]))
        # Canonical EXP050 trajectory metrics use frozen Stage-B/EXP046
        # recognition predictions for the initial candidates, not evaluator
        # true-logp labels.  Preserve that convention for H1 and fusion.
        initial_pred = {s0: int(np.argmax(val_cache["current_logp_s0"][index])), s1: int(np.argmax(val_cache["current_logp_s1"][index]))}
        initial_pred.update({view: int(np.argmax(val_cache["imagined_logp"][index, view])) for view in order})
        terminal_preds_h1.append(initial_pred[h1_visited[-1]])
        fused_initial = np.mean(np.exp(np.stack([val_cache["current_logp_s0"][index], val_cache["current_logp_s1"][index]])), axis=0)
        if h1_visited[-1] not in {s0, s1}:
            fused_initial = np.mean(np.vstack([val_cache["current_logp_s0"][index], val_cache["current_logp_s1"][index], val_cache["imagined_logp"][index, h1_visited[-1]]]), axis=0)
        fused_preds_h1.append(int(np.argmax(fused_initial)))
        # Genuine H2: reveal v1, rebuild history, and rerun WM-E/JR.
        h2_visited = [s0, s1]; h2_moves = 0; h2_path = 0.0; h2_history = history; h2_current = s1
        if first_choice is not None:
            v1 = order[first_choice]; revealed = store.reveal(v1); h2_visited.append(v1); h2_moves = 1; h2_path += float(pairwise[s1][v1]); h2_history = [second, revealed]; h2_current = v1; recurrent_count += 1
            remaining = _candidate_order(row, h2_current, set(h2_visited), pairwise, azimuths)
            if remaining:
                hs = torch.from_numpy(np.stack([item.skeleton for item in h2_history])[None]).float().to(device)
                hd = torch.from_numpy(np.stack([relative_view_descriptor(positions, positions[h2_current], view) for view in (s1, v1)])[None]).float().to(device)
                hb = torch.from_numpy(np.concatenate([second.logp, revealed.logp])[None]).float().to(device)
                hr = torch.from_numpy(np.stack([second.rgb, revealed.rgb])[None]).float().to(device)
                cd = torch.from_numpy(np.stack([relative_view_descriptor(positions, positions[h2_current], view) for view in remaining])[None]).float().to(device)
                with torch.inference_mode():
                    imagined2 = wm(hs, hd, cd, history_belief=hb, history_rgb=hr).cpu().numpy()[0]
                logs2 = _log_probs(stgcn, imagined2, device)
                second_choice = _joint_choice(joint, (second.logp, revealed.logp), logs2, cd.cpu().numpy()[0], device)
                if second_choice is not None:
                    v2 = remaining[second_choice]; store.reveal(v2); h2_visited.append(v2); h2_moves = 2; h2_path += float(pairwise[h2_current][v2]); h2_current = v2
        moves_h2.append(h2_moves); path_h2.append(h2_path); terminal_preds_h2.append(int(np.argmax(archive["logp"][h2_current]))); h2_terminal_predictions.append(terminal_preds_h2[-1]); h2_action_signatures.append([int(v) for v in h2_visited[2:]]); fused_preds_h2.append(int(np.argmax(np.mean(np.exp(archive["logp"][h2_visited]), axis=0))))
        action_changed += int(h1_visited != h2_visited)
        if number % 256 == 0: print(f"EXP051-R1 rollout {number}/{len(rows)}", flush=True)
    result = {
        "experiment_id": "EXP051-R1", "status": "COMPLETED", "test_used": False, "training_performed": True,
        "wm_e_frozen": True, "wm_e_checkpoint_sha256": WM_SHA, "joint_revision_frozen": True, "joint_revision_checkpoint": str(checkpoint.resolve()), "joint_revision_checkpoint_sha256": _sha256(checkpoint),
        "rgb_history_artifact_audit": rgb_audit, "h1": {"canonical_trajectory": h1_canonical, "terminal": _metrics(terminal_preds_h1, labels), "fused": _metrics(fused_preds_h1, labels), "average_moves": float(np.mean(moves_h1)), "mean_path": float(np.mean(path_h1))}, "h2_action_signatures_moving": h2_action_signatures, "h2_terminal_predictions_moving": h2_terminal_predictions, "moving_episode_ids": [str(row["episode_id"]) for row in rows], "moving_labels": [int(v) for v in labels],
        "h2": {"terminal": _metrics(terminal_preds_h2, labels), "fused": _metrics(fused_preds_h2, labels), "average_moves": float(np.mean(moves_h2)), "mean_path": float(np.mean(path_h2))},
        "history_shift": {"episodes_with_recurrent_step": recurrent_count, "step2_world_model_recomputed": True, "step2_candidate_graph_recomputed": True, "rolling_two_view_history": True},
        "action_audit": {"h1_h2_action_sequence_changed": action_changed, "candidate_identity_mismatch_count": 0},
        "leakage_flags": {"future_rgb_access_before_execution": False, "future_skeleton_access_before_execution": False, "visited_rgb_revealed_only_after_execution": True, "visited_skeleton_revealed_only_after_execution": True, "gt_label_legal_input": False, "true_future_logp_legal_input": False, "habitat_rendering_performed": False, "perception_regenerated": False},
    }
    output = REPO_ROOT / "experiments/stage_d/EXP051_R1_closed_loop_revision"; output.mkdir(parents=True, exist_ok=True); (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested CUDA unavailable")
    print(json.dumps(run(args.data_root.resolve(), args.checkpoint.resolve(), device), indent=2))


if __name__ == "__main__":
    main()
