#!/usr/bin/env python3
"""EXP054 H2 ceiling decomposition (Train-matched JR, Val only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_dense_campaign import context_key
from activeview.core.paths import get_data_root
from activeview.active_view.stage_d_dense_campaign import relative_view_descriptor
from activeview.scripts.run_exp051_r1_closed_loop import _candidate_order, _joint_choice
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _rows
from activeview.scripts.run_stage_d_exp046_048 import _load_cache
from activeview.scripts.run_stage_d_exp049_051 import _JointRevision, _joint_select, _load_pairwise_and_azimuths

REPO_ROOT = Path(__file__).resolve().parents[2]
N_CLASSES = 16
VIEW_COUNT = 32
SEED = 42


def _seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _metrics(pred: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    p, y = np.asarray(pred, dtype=np.int64), np.asarray(labels, dtype=np.int64)
    confusion = np.bincount(y * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    f1 = []
    for cls in range(N_CLASSES):
        tp = float(confusion[cls, cls]); pd = float(confusion[:, cls].sum()); rd = float(confusion[cls].sum())
        pr = tp / pd if pd else 0.0; rc = tp / rd if rd else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"accuracy": float(np.mean(p == y)), "macro_f1": float(np.mean(f1))}


class _JointDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, current: np.ndarray, candidates: np.ndarray, mask: np.ndarray, target: np.ndarray, correctness: np.ndarray, labels: np.ndarray) -> None:
        self.current, self.candidates, self.mask = current, candidates, mask; self.target, self.correctness, self.labels = target, correctness, labels

    def __len__(self) -> int: return len(self.target)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"current": torch.from_numpy(self.current[i]), "candidates": torch.from_numpy(self.candidates[i]), "mask": torch.from_numpy(self.mask[i]), "target": torch.tensor(self.target[i]), "correctness": torch.from_numpy(self.correctness[i]), "label": torch.tensor(self.labels[i])}


def _true_joint_examples(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], v1_by_episode: Mapping[str, int | None], sources: Mapping[tuple[str, str, str], str] | None = None, pairwise: Mapping[tuple[str, str], Mapping[int, Mapping[int, float]]] | None = None, azimuths: Mapping[tuple[str, str], Mapping[int, float]] | None = None, reanchor_geometry: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = {str(e): i for i, e in enumerate(cache["episode_ids"].tolist())}; current_values: list[np.ndarray] = []; candidate_values: list[np.ndarray] = []; masks: list[np.ndarray] = []; targets: list[int] = []; correctness: list[np.ndarray] = []; labels: list[int] = []
    for row in rows:
        v1 = v1_by_episode.get(str(row["episode_id"])); if_none = v1 is None
        if if_none: continue
        i = indices[str(row["episode_id"])]
        if reanchor_geometry:
            if sources is None or pairwise is None or azimuths is None:
                raise ValueError("v1-centered geometry requires source, pairwise and azimuth metadata")
            scene_region = (str(row["scene_id"]), str(row["region"]))
            candidates = _candidate_order(row, int(v1), {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]), int(v1)}, pairwise[scene_region], azimuths[scene_region])
            source = Path(sources[context_key(row)])
            with np.load(source, allow_pickle=False) as archive:
                positions = np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)
            descriptors = {candidate: relative_view_descriptor(positions, positions[int(v1)], candidate) for candidate in candidates}
        else:
            candidates = [int(v) for v in orders[str(row["episode_id"])] if int(v) not in {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]), int(v1)}]
            descriptors = {candidate: cache["candidate_descriptor"][i, candidate] for candidate in candidates}
        if not candidates: continue
        previous = cache["true_logp"][i, int(row["s1_viewpoint_id"])]; current = cache["true_logp"][i, int(v1)]; p0, p1 = np.exp(previous), np.exp(current)
        stats = np.asarray([*p0, *p1, -np.sum(p0 * previous), -np.sum(p1 * current), np.max(p0), np.max(p1), np.sort(p1)[-1] - np.sort(p1)[-2], np.sort(p0)[-1] - np.sort(p0)[-2]], dtype=np.float32)
        candidate_rows = [np.concatenate([current, np.zeros(9, dtype=np.float32), [1.0]])]
        candidate_rows.extend(np.concatenate([cache["true_logp"][i, c], np.asarray(descriptors[c], dtype=np.float32), [0.0]]) for c in candidates)
        label = int(row["label_id"]); action_correct = [int(np.argmax(current) == label)] + [int(np.argmax(cache["true_logp"][i, c]) == label) for c in candidates]
        if any(action_correct): target = next(j for j, value in enumerate(action_correct) if value)
        else: target = int(np.argmax([current[label]] + [cache["true_logp"][i, c, label] for c in candidates]))
        padded = np.zeros((31, 26), dtype=np.float32); padded[:len(candidate_rows)] = np.asarray(candidate_rows, dtype=np.float32); corr = np.zeros(31, dtype=np.float32); corr[:len(action_correct)] = np.asarray(action_correct, dtype=np.float32)
        current_values.append(stats); candidate_values.append(padded); masks.append(np.asarray([True] * len(candidate_rows) + [False] * (30 - len(candidates)), dtype=bool)); targets.append(target); correctness.append(corr); labels.append(label)
    return np.asarray(current_values), np.asarray(candidate_values), np.asarray(masks), np.asarray(targets), np.asarray(correctness), np.asarray(labels)


def _fit_true_joint(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], v1_by_episode: Mapping[str, int | None], device: torch.device, output: Path, sources: Mapping[tuple[str, str, str], str] | None = None, pairwise: Mapping[tuple[str, str], Mapping[int, Mapping[int, float]]] | None = None, azimuths: Mapping[tuple[str, str], Mapping[int, float]] | None = None, reanchor_geometry: bool = False) -> tuple[_JointRevision, dict[str, Any]]:
    model = _JointRevision().to(device)
    if output.exists():
        payload = torch.load(output, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        contexts = int(payload.get("contexts", 0))
        if contexts == 0:
            contexts = len(_true_joint_examples(rows, cache, orders, v1_by_episode, sources, pairwise, azimuths, reanchor_geometry)[0])
        return model.eval(), {"contexts": contexts, "epochs": 20, "batch_size": 512, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "checkpoint": str(output.resolve()), "checkpoint_sha256": _sha256(output), "reused": True}
    arrays = _true_joint_examples(rows, cache, orders, v1_by_episode, sources, pairwise, azimuths, reanchor_geometry); dataset = _JointDataset(*arrays); loader = DataLoader(dataset, batch_size=512, shuffle=True); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); history: list[float] = []
    for epoch in range(1, 21):
        model.train(); losses = []
        for batch in loader:
            current = batch["current"].float().to(device)
            candidates = batch["candidates"].float().to(device)
            mask = batch["mask"].bool().to(device)
            target = batch["target"].long().to(device)
            corr = batch["correctness"].float().to(device)
            labels = batch["label"].long().to(device)
            scores, posterior = model(current, candidates, mask); ce = nn.functional.cross_entropy(scores.masked_fill(~mask, -1e9), target); bce = nn.functional.binary_cross_entropy_with_logits(scores, corr, reduction="none").masked_select(mask).mean(); pt = labels.unsqueeze(1).expand(-1, posterior.size(1)); post = nn.functional.cross_entropy(posterior.reshape(-1, N_CLASSES), pt.reshape(-1), reduction="none").reshape_as(scores).masked_select(mask).mean(); loss = ce + 0.25 * bce + 0.05 * post
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        value = float(np.mean(losses)); history.append(value); print(f"EXP054 true-JR epoch {epoch}/20 loss={value:.6f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True); torch.save({"model_state_dict": model.state_dict(), "epoch": 20, "seed": SEED, "input": "true_future_recognition", "contexts": len(dataset), "geometry": "v1_centered" if reanchor_geometry else "s1_centered"}, output)
    return model.eval(), {"contexts": len(dataset), "epochs": 20, "batch_size": 512, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "final_loss": history[-1], "history": history, "checkpoint": str(output.resolve()), "checkpoint_sha256": _sha256(output)}


def _full(moving: Sequence[int], labels: Sequence[int], v0_rows: Sequence[Mapping[str, Any]], moving_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_id = {str(r["episode_id"]): int(p) for r, p in zip(moving_rows, moving)}; pred = [int(r["current_predicted_label_id"]) if bool(r["predicted_stays"]) else by_id[str(r["episode_id"])] for r in v0_rows]; truth = [int(r["label_id"]) for r in v0_rows]; return _metrics(pred, truth)


def run(data_root: Path, device: torch.device, r1: bool = False, output_dir: Path | None = None) -> dict[str, Any]:
    _seed(); train_rows, val_rows = _rows(data_root, "train"), _rows(data_root, "val"); train_sources, val_sources = _episode_sources(data_root, "train"), _episode_sources(data_root, "val"); train_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz"); val_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz")
    joint_payload = torch.load(data_root / "experiments/stage_d/EXP050_joint_rollout_revision/joint_revision_final.pth", map_location=device, weights_only=False); frozen_joint = _JointRevision().to(device); frozen_joint.load_state_dict(joint_payload["model_state_dict"]); frozen_joint.eval()
    train_pair, train_az = _load_pairwise_and_azimuths(data_root, train_rows, train_sources); val_pair, val_az = _load_pairwise_and_azimuths(data_root, val_rows, val_sources); train_v0 = {str(x["episode_id"]): x for x in load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/train_predictions.jsonl")}; val_v0_rows = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"); val_v0 = {str(x["episode_id"]): x for x in val_v0_rows}
    train_orders = {str(r["episode_id"]): ([] if bool(train_v0[str(r["episode_id"])]["predicted_stays"]) else _candidate_order(r, int(r["s1_viewpoint_id"]), {int(r["s0_viewpoint_id"]), int(r["s1_viewpoint_id"])}, train_pair[(str(r["scene_id"]), str(r["region"]))], train_az[(str(r["scene_id"]), str(r["region"]))])) for r in train_rows}; train_selected = _joint_select(frozen_joint, train_cache, train_rows, train_orders, "ALL_LEGAL", device); train_v1 = {str(r["episode_id"]): (None if v is None else int(v)) for r, v in zip(train_rows, train_selected)}
    default_runtime = data_root / "experiments/stage_d/EXP054_R1_h2_ceiling_decomposition/runtime" if r1 else data_root / "experiments/stage_d/EXP054_h2_ceiling_decomposition/runtime"
    checkpoint = default_runtime / ("true_future_matched_joint_revision_r1.pth" if r1 else "true_future_matched_joint_revision.pth")
    model, train_info = _fit_true_joint(train_rows, train_cache, train_orders, train_v1, device, checkpoint, train_sources, train_pair, train_az, r1)
    original_r2 = json.loads(Path("/tmp/activeview_exp052_r2_original3/result.json").read_text(encoding="utf-8")); original_sigs = original_r2["h2_action_signatures_moving"]; original_term = np.asarray(original_r2["h2_terminal_predictions_moving"], dtype=np.int64)
    val_indices = {str(v): i for i, v in enumerate(val_cache["episode_ids"].tolist())}; labels: list[int] = []; frozen_pred: list[int] = []; matched_pred: list[int] = []; oracle_pred: list[int] = []; frozen_sigs: list[list[int]] = []; matched_sigs: list[list[int]] = []; oracle_actions: list[int | None] = []
    for i, row in enumerate(val_rows):
        key = context_key(row); idx = val_indices[str(row["episode_id"])]; label = int(row["label_id"]); labels.append(label); sig = original_sigs[i]; v1 = int(sig[0]) if sig else None
        if v1 is None:
            frozen_pred.append(int(np.argmax(val_cache["true_logp"][idx, int(row["s1_viewpoint_id"])]))); matched_pred.append(frozen_pred[-1]); oracle_pred.append(frozen_pred[-1]); frozen_sigs.append([]); matched_sigs.append([]); oracle_actions.append(None); continue
        rem = _candidate_order(row, v1, {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]), v1}, val_pair[(str(row["scene_id"]), str(row["region"]))], val_az[(str(row["scene_id"]), str(row["region"]))])
        source = Path(val_sources[key]);
        with np.load(source, allow_pickle=False) as z: positions = np.asarray(z["viewpoint_agent_positions"], dtype=np.float32)
        desc = np.stack([relative_view_descriptor(positions, positions[v1], c) for c in rem])
        true_logs = val_cache["true_logp"][idx, np.asarray(rem)]
        choice_f = _joint_choice(frozen_joint, (val_cache["true_logp"][idx, int(row["s1_viewpoint_id"])], val_cache["true_logp"][idx, v1]), true_logs, desc, device); choice_m = _joint_choice(model, (val_cache["true_logp"][idx, int(row["s1_viewpoint_id"])], val_cache["true_logp"][idx, v1]), true_logs, desc, device)
        frozen_v = None if choice_f is None else int(rem[choice_f]); matched_v = None if choice_m is None else int(rem[choice_m]); frozen_sigs.append([v1] + ([] if frozen_v is None else [frozen_v])); matched_sigs.append([v1] + ([] if matched_v is None else [matched_v]))
        frozen_terminal = v1 if frozen_v is None else frozen_v; matched_terminal = v1 if matched_v is None else matched_v; frozen_pred.append(int(np.argmax(val_cache["true_logp"][idx, frozen_terminal]))); matched_pred.append(int(np.argmax(val_cache["true_logp"][idx, matched_terminal])))
        scores = [float(val_cache["true_logp"][idx, v1, label])] + [float(val_cache["true_logp"][idx, c, label]) for c in rem]; oracle_index = int(np.argmax(scores)); oracle_v = None if oracle_index == 0 else int(rem[oracle_index - 1]); oracle_actions.append(oracle_v); oracle_terminal = v1 if oracle_v is None else oracle_v; oracle_pred.append(int(np.argmax(val_cache["true_logp"][idx, oracle_terminal])))
    moving_labels = labels; full_frozen = _full(frozen_pred, labels, val_v0_rows, val_rows); full_matched = _full(matched_pred, labels, val_v0_rows, val_rows); full_oracle = _full(oracle_pred, labels, val_v0_rows, val_rows)
    second_action = lambda signature: None if len(signature) < 2 else int(signature[1])
    changed = [i for i, (a, b) in enumerate(zip(original_sigs, matched_sigs)) if a != b]; old_correct = original_term == np.asarray(labels); matched_correct = np.asarray(matched_pred) == np.asarray(labels); rescued = int(np.sum(~old_correct & matched_correct)); harmful = int(np.sum(old_correct & ~matched_correct)); oracle_match = sum(second_action(s) == a for s, a in zip(original_sigs, oracle_actions)); matched_match = sum(second_action(s) == a for s, a in zip(matched_sigs, oracle_actions))
    out = output_dir or (REPO_ROOT / ("experiments/stage_d/EXP054_R1_h2_ceiling_decomposition" if r1 else "experiments/stage_d/EXP054_h2_ceiling_decomposition")); out.mkdir(parents=True, exist_ok=True); h1 = original_r2["h1_real"]; matched_key = "TRUE_FUTURE_MATCHED_JR_R1" if r1 else "TRUE_FUTURE_MATCHED_JR"; result = {"experiment_id": "EXP054-R1" if r1 else "EXP054", "status": "COMPLETED", "split": "val", "test_used": False, "training_performed": True, "h1_real": h1, "episode_counts": {"moving_val": len(val_rows), "full_val": len(val_v0_rows), "train_matched_contexts": train_info["contexts"]}, "methods": {"CURRENT_WM_JR": {"moving": original_r2["h2_real"]["moving_subset"], "full": original_r2["h2_real"]["full"]}, "TRUE_FUTURE_FROZEN_JR": {"moving": _metrics(frozen_pred, moving_labels), "full": full_frozen}, matched_key: {"moving": _metrics(matched_pred, moving_labels), "full": full_matched}, "TRUE_FUTURE_GT_ORACLE": {"moving": _metrics(oracle_pred, moving_labels), "full": full_oracle}}, "train_matched_joint_revision": train_info, "ceiling": {"world_model_recoverable_headroom": None, "decision_remaining_headroom": None, "total_headroom": None}, "action_audit": {"changed_second_action_count": len(changed), "rescued": rescued, "harmful": harmful, "net": rescued - harmful, "current_oracle_action_agreement": float(np.mean([second_action(s) == a for s, a in zip(original_sigs, oracle_actions)])), "matched_oracle_action_agreement": float(matched_match / len(matched_sigs))}, "provenance": {"frozen_joint_revision": str((data_root / "experiments/stage_d/EXP050_joint_rollout_revision/joint_revision_final.pth").resolve()), "true_future_cache": str((data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz").resolve()), "stgcn_frozen": True, "first_step_frozen": True, "terminal_recognition_real_archived": True, "perception_regenerated": False, "habitat_rendering_performed": False, "test_used": False, "train_geometry": "v1_centered" if r1 else "legacy_s1_centered"}}
    matched_acc = result["methods"][matched_key]["moving"]["accuracy"]; current_acc = result["methods"]["CURRENT_WM_JR"]["moving"]["accuracy"]; oracle_acc = result["methods"]["TRUE_FUTURE_GT_ORACLE"]["moving"]["accuracy"]; result["ceiling"] = {"world_model_recoverable_headroom": matched_acc - current_acc, "decision_remaining_headroom": oracle_acc - matched_acc, "total_headroom": oracle_acc - current_acc, "wm_fraction": (matched_acc - current_acc) / (oracle_acc - current_acc) if oracle_acc != current_acc else None, "decision_fraction": (oracle_acc - matched_acc) / (oracle_acc - current_acc) if oracle_acc != current_acc else None}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n"); (out / "ceiling_decomposition.json").write_text(json.dumps(result["ceiling"], indent=2) + "\n"); (out / "action_audit.json").write_text(json.dumps(result["action_audit"], indent=2) + "\n"); print(json.dumps(result, indent=2)); return result


def _archive_positions(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z: return np.asarray(z["viewpoint_agent_positions"], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--r1", action="store_true"); parser.add_argument("--output-dir", type=Path); args = parser.parse_args(); _seed(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; EXP054 requires GPU")
    run(args.data_root.resolve(), device, r1=args.r1, output_dir=args.output_dir.resolve() if args.output_dir else None)


if __name__ == "__main__": main()
