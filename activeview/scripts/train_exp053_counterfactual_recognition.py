#!/usr/bin/env python3
"""EXP053 direct counterfactual recognition world model (Train/Val only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from activeview.active_view.stage_d_counterfactual_recognition import CounterfactualRecognitionModel
from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_dense_campaign import context_key, relative_view_descriptor
from activeview.active_view.stage_d_rgb_context import RGBObservationKey, load_dinov2
from activeview.active_view.stage_d_rgb_spatial import build_or_load_spatial_cache
from activeview.core.paths import get_data_root
from activeview.scripts.run_exp051_r1_closed_loop import _candidate_order, _joint_choice
from activeview.scripts.run_exp051_r2_real_eval import _classification, _fidelity
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _rows
from activeview.scripts.run_stage_d_exp046_048 import _load_cache, _load_stgcn
from activeview.scripts.run_stage_d_exp049_051 import _JointRevision, _joint_select, _load_pairwise_and_azimuths

REPO_ROOT = Path(__file__).resolve().parents[2]
RGB_ROOT = Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train")
RGB_BASE = Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4")
RGB_VAL = Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_allview_exp051_r1")
RGB_EXP052 = Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_exp052_train")
N_CLASSES = 16
VIEW_COUNT = 32
SEED = 42


def _seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


class RGBIndex:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str, int], np.ndarray] = {}
        self.manifests: list[Path] = []

    def add_cache(self, root: Path) -> None:
        manifest = root / "manifest.jsonl"; emb_path = root / "embeddings.npy"
        if not manifest.is_file() or not emb_path.is_file(): return
        embeddings = np.load(emb_path, mmap_mode="r")
        self.manifests.append(manifest)
        for index, line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
            if not line.strip(): continue
            item = json.loads(line)
            key = (str(item["scene_id"]), str(item["region"]), str(item["record_id"]), int(item["viewpoint_id"]))
            self.values.setdefault(key, np.asarray(embeddings[index], dtype=np.float32))

    def ensure(self, keys: Sequence[tuple[str, str, str, int]], device: torch.device) -> dict[str, Any]:
        missing = sorted(set(keys) - set(self.values))
        if not missing: return {"cache_miss_count": 0, "extracted": False, "extraction_time_sec": 0.0}
        out = Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_exp053_train_history")
        started = time.monotonic()
        values, manifest, info = build_or_load_spatial_cache(
            rgb_root=RGB_ROOT, cache_dir=out,
            keys=[RGBObservationKey(*key) for key in missing],
            model_loader=load_dinov2, device=device, batch_size=32,
        )
        for index, item in enumerate(manifest):
            key = (str(item["scene_id"]), str(item["region"]), str(item["record_id"]), int(item["viewpoint_id"]))
            self.values[key] = np.asarray(values[index], dtype=np.float32)
        return {"cache_miss_count": len(missing), "extracted": True, "extraction_time_sec": time.monotonic() - started, **info}


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {"skeleton": np.asarray(z["skeleton"], dtype=np.float32), "viewpoint_ids": np.asarray(z["viewpoint_ids"], dtype=np.int64), "positions": np.asarray(z["viewpoint_agent_positions"], dtype=np.float32)}


def _view_input(archive: Mapping[str, np.ndarray], cache: Mapping[str, np.ndarray], index: int, key: tuple[str, str, str], rgb: RGBIndex, current: int, view: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = archive["viewpoint_ids"]; by_id = {int(v): i for i, v in enumerate(ids.tolist())}; position = archive["positions"]
    skeleton = archive["skeleton"][by_id[view]]
    compact = np.concatenate([skeleton.mean(axis=1).reshape(-1), skeleton.std(axis=1).reshape(-1)]).astype(np.float32)
    rgb_value = np.asarray(rgb.values[(*key, int(view))], dtype=np.float32).mean(axis=0)
    rec = np.asarray(cache["true_logp"][index, view], dtype=np.float32)
    geom = np.asarray(relative_view_descriptor(position, position[by_id[current]], view), dtype=np.float32)
    return rec, rgb_value, compact, geom


class RecognitionDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(self, contexts: Sequence[dict[str, Any]]) -> None:
        self.contexts = list(contexts)
        self.samples = [(i, tuple(item["history"]), int(candidate)) for i, item in enumerate(self.contexts) for candidate in item["candidate_ids"]]

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        context_index, history, candidate = self.samples[index]; item = self.contexts[context_index]
        history_data = [item["views"][view] for view in history]
        return {
            "history_recognition": np.stack([x[0] for x in history_data]),
            "history_rgb": np.stack([x[1] for x in history_data]),
            "history_skeleton": np.stack([x[2] for x in history_data]),
            "history_geometry": np.stack([x[3] for x in history_data]),
            "candidate_geometry": item["candidate_geometry"][candidate],
            "target": item["target"][candidate],
        }


def _collate(batch: Sequence[Mapping[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    max_h = max(len(x["history_recognition"]) for x in batch); n = len(batch)
    out = {name: torch.zeros((n, max_h, dim), dtype=torch.float32) for name, dim in (("history_recognition", 16), ("history_rgb", 768), ("history_skeleton", 102), ("history_geometry", 9))}
    mask = torch.zeros((n, max_h), dtype=torch.bool); candidate = torch.zeros((n, 9), dtype=torch.float32); target = torch.zeros((n, 16), dtype=torch.float32)
    for i, x in enumerate(batch):
        h = len(x["history_recognition"]); mask[i, :h] = True
        for name in out: out[name][i, :h] = torch.from_numpy(x[name])
        candidate[i] = torch.from_numpy(x["candidate_geometry"]); target[i] = torch.from_numpy(x["target"])
    out.update({"candidate_geometry": candidate, "target": target, "history_mask": mask}); return out


def _build_contexts(rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str], cache: Mapping[str, np.ndarray], rgb: RGBIndex, rollout_v1: Sequence[int | None], *, train: bool) -> tuple[list[dict[str, Any]], list[tuple[str, str, str, int]]]:
    rng = np.random.default_rng(SEED); index_by_id = {str(v): i for i, v in enumerate(cache["episode_ids"].tolist())}; contexts: list[dict[str, Any]] = []; keys: list[tuple[str, str, str, int]] = []
    for row_index, row in enumerate(rows):
        key = context_key(row); idx = index_by_id[str(row["episode_id"])]
        arc = _archive(Path(sources[key])); ids = [int(v) for v in arc["viewpoint_ids"].tolist()]; s0, s1 = int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])
        histories = [(s0, s1)]
        v1 = rollout_v1[row_index] if train else None
        if v1 is not None: histories.append((s0, s1, int(v1)))
        for history in histories:
            current = history[-1]; candidates = [v for v in ids if v not in set(history)]
            if train:
                candidates = rng.choice(candidates, size=min(4, len(candidates)), replace=False).tolist()
            views = {v: _view_input(arc, cache, idx, key, rgb, current, v) for v in history}; keys.extend((*key, int(v)) for v in history)
            descriptor = {v: np.asarray(relative_view_descriptor(arc["positions"], arc["positions"][{int(x): i for i, x in enumerate(arc["viewpoint_ids"].tolist())}[current]], v), dtype=np.float32) for v in candidates}
            target = {v: np.asarray(cache["true_logp"][idx, v], dtype=np.float32) for v in candidates}
            contexts.append({"history": history, "views": views, "candidate_geometry": descriptor, "target": target, "key": key, "candidate_ids": candidates})
    return contexts, keys


def _train(model: CounterfactualRecognitionModel, dataset: RecognitionDataset, device: torch.device, output: Path, epochs: int, batch_size: int) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in loader:
            values = {k: batch[k].to(device) for k in ("history_recognition", "history_rgb", "history_skeleton", "history_geometry", "candidate_geometry", "history_mask")}; target = batch["target"].to(device)
            logits = model(**values); pred_logp = torch.log_softmax(logits, dim=-1); loss = nn.functional.kl_div(pred_logp, target.exp(), reduction="batchmean") + 0.1 * nn.functional.mse_loss(pred_logp, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += float(loss.detach()) * len(target); count += len(target)
        value = total / max(count, 1); history.append({"epoch": epoch, "loss": value}); output.parent.mkdir(parents=True, exist_ok=True); torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "seed": SEED, "architecture": "recognition-128-transformer-2"}, output); print(f"EXP053 epoch {epoch}/{epochs} loss={value:.6f}", flush=True)
    return {"epochs": epochs, "batch_size": batch_size, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "samples": len(dataset), "final_loss": history[-1]["loss"], "history": history}


def _predict(model: CounterfactualRecognitionModel, item: dict[str, Any], candidate_ids: Sequence[int], device: torch.device) -> np.ndarray:
    history = item["history"]; h = len(history); n = len(candidate_ids)
    def stack(name: int) -> np.ndarray: return np.stack([item["views"][v][name] for v in history])
    base = {"history_recognition": np.repeat(stack(0)[None], n, axis=0), "history_rgb": np.repeat(stack(1)[None], n, axis=0), "history_skeleton": np.repeat(stack(2)[None], n, axis=0), "history_geometry": np.repeat(stack(3)[None], n, axis=0), "history_mask": np.ones((n, h), dtype=bool), "candidate_geometry": np.stack([item["candidate_geometry"][v] for v in candidate_ids])}
    with torch.inference_mode(): return torch.log_softmax(model(**{k: torch.from_numpy(v).float().to(device) for k, v in base.items()}), dim=-1).cpu().numpy()


def _metrics(pred: Sequence[int], labels: Sequence[int]) -> dict[str, float]: return _classification(pred, labels)


def _run_val(data_root: Path, model: CounterfactualRecognitionModel, device: torch.device, joint_checkpoint: Path) -> dict[str, Any]:
    rows = _rows(data_root, "val"); sources = _episode_sources(data_root, "val"); cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/val_cache.npz"); index_by_id = {str(v): i for i, v in enumerate(cache["episode_ids"].tolist())}
    pair, az = _load_pairwise_and_azimuths(data_root, rows, sources); v0_rows = load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/val_predictions.jsonl"); v0 = {str(x["episode_id"]): x for x in v0_rows}; orders = {str(r["episode_id"]): ([] if bool(v0[str(r["episode_id"])] ["predicted_stays"]) else _candidate_order(r, int(r["s1_viewpoint_id"]), {int(r["s0_viewpoint_id"]), int(r["s1_viewpoint_id"])}, pair[(str(r["scene_id"]), str(r["region"]))], az[(str(r["scene_id"]), str(r["region"]))])) for r in rows}
    joint_payload = torch.load(joint_checkpoint, map_location=device, weights_only=False); joint = _JointRevision().to(device); joint.load_state_dict(joint_payload["model_state_dict"]); joint.eval()
    rgb = RGBIndex(); rgb.add_cache(RGB_BASE); rgb.add_cache(RGB_VAL); rgb.add_cache(RGB_EXP052)
    h0_imag: list[np.ndarray] = []; h0_true: list[np.ndarray] = []; h0_labels: list[int] = []; h1_imag: list[np.ndarray] = []; h1_true: list[np.ndarray] = []; h1_labels: list[int] = []
    terminal: list[int] = []; fused: list[int] = []; labels: list[int] = []; signatures: list[list[int]] = []; original = json.loads(Path("/tmp/activeview_exp052_r2_original3/result.json").read_text())
    original_signatures = original["h2_action_signatures_moving"]
    for i, row in enumerate(rows):
        key = context_key(row); idx = index_by_id[str(row["episode_id"] )]; arc = _archive(Path(sources[key])); by_id = {int(v): j for j, v in enumerate(arc["viewpoint_ids"].tolist())}; s0, s1 = int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"]); labels.append(int(row["label_id"]))
        h0_candidates = orders[str(row["episode_id"])]; h0_item = {"history": (s0, s1), "views": {v: _view_input(arc, cache, idx, key, rgb, s1, v) for v in (s0, s1)}, "candidate_geometry": {v: relative_view_descriptor(arc["positions"], arc["positions"][by_id[s1]], v) for v in h0_candidates}}
        if h0_candidates:
            logs0 = _predict(model, h0_item, h0_candidates, device); h0_imag.append(logs0); h0_true.append(cache["true_logp"][idx, np.asarray(h0_candidates)]); h0_labels.append(int(row["label_id"]))
            desc0 = np.stack([h0_item["candidate_geometry"][v] for v in h0_candidates]); v1_choice = _joint_choice(joint, (cache["current_logp_s0"][idx], cache["current_logp_s1"][idx]), logs0, desc0, device)
            v1 = None if v1_choice is None else int(h0_candidates[v1_choice])
        else: v1 = None
        history_ids = [s0, s1] if v1 is None else [s0, s1, v1]; current = s1 if v1 is None else v1; second_choice = None
        if v1 is not None:
            rem = _candidate_order(row, current, set(history_ids), pair[(str(row["scene_id"]), str(row["region"]))], az[(str(row["scene_id"]), str(row["region"]))]); h1_item = {"history": tuple(history_ids), "views": {v: _view_input(arc, cache, idx, key, rgb, current, v) for v in history_ids}, "candidate_geometry": {v: relative_view_descriptor(arc["positions"], arc["positions"][by_id[current]], v) for v in rem}}
            if rem:
                logs1 = _predict(model, h1_item, rem, device); second_choice = _joint_choice(joint, (cache["true_logp"][idx, s1], cache["true_logp"][idx, v1]), logs1, np.stack([h1_item["candidate_geometry"][v] for v in rem]), device); current = rem[second_choice] if second_choice is not None else current
        # Fidelity h1 is evaluated on the frozen EXP051-R2 rollout history,
        # independently of any EXP053 action changes.
        original_v1 = int(original_signatures[i][0]) if original_signatures[i] else None
        if original_v1 is not None:
            original_remaining = _candidate_order(row, original_v1, {s0, s1, original_v1}, pair[(str(row["scene_id"]), str(row["region"]))], az[(str(row["scene_id"]), str(row["region"]))])
            if original_remaining:
                original_item = {"history": (s0, s1, original_v1), "views": {v: _view_input(arc, cache, idx, key, rgb, original_v1, v) for v in (s0, s1, original_v1)}, "candidate_geometry": {v: relative_view_descriptor(arc["positions"], arc["positions"][by_id[original_v1]], v) for v in original_remaining}}
                h1_logs = _predict(model, original_item, original_remaining, device)
                h1_imag.append(h1_logs); h1_true.append(cache["true_logp"][idx, np.asarray(original_remaining)]); h1_labels.append(int(row["label_id"]))
        signatures.append([] if v1 is None else [v1] + ([] if second_choice is None else [int(current)])); terminal.append(int(np.argmax(cache["true_logp"][idx, current]))); fused.append(int(np.argmax(np.mean(np.exp(cache["true_logp"][idx, history_ids if second_choice is None else history_ids + [current]]), axis=0))))
        if i and i % 512 == 0: print(f"EXP053 Val {i}/{len(rows)}", flush=True)
    full_labels = []; full_h2 = []; full_fused = []; by_ep = {str(r["episode_id"]): i for i, r in enumerate(rows)}
    for row in v0_rows:
        full_labels.append(int(row["label_id"])); ep = str(row["episode_id"])
        if bool(row["predicted_stays"]): full_h2.append(int(row["current_predicted_label_id"])); full_fused.append(int(row["current_predicted_label_id"]))
        else: full_h2.append(terminal[by_ep[ep]]); full_fused.append(fused[by_ep[ep]])
    return {"h0_fidelity": _fidelity(h0_imag, h0_true, h0_labels), "h1_fidelity": _fidelity(h1_imag, h1_true, h1_labels), "h2_moving": _metrics(terminal, labels), "h2_full": _metrics(full_h2, full_labels), "h2_fused_moving": _metrics(fused, labels), "h2_fused_full": _metrics(full_fused, full_labels), "terminal_predictions": terminal, "action_signatures": signatures, "labels": labels, "original_signatures": original_signatures}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=get_data_root()); parser.add_argument("--device", default="cpu"); parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--batch-size", type=int, default=256); args = parser.parse_args(); _seed(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    data_root = args.data_root.resolve(); train_rows, val_rows = _rows(data_root, "train"), _rows(data_root, "val"); train_sources, val_sources = _episode_sources(data_root, "train"), _episode_sources(data_root, "val"); train_cache = _load_cache(data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz")
    pair, az = _load_pairwise_and_azimuths(data_root, train_rows, train_sources); train_v0 = {str(x["episode_id"]): x for x in load_jsonl(data_root / "experiments/stage_d/EXP014_two_step_sequential/v0_predictions/train_predictions.jsonl")}; train_orders = {str(r["episode_id"]): _candidate_order(r, int(r["s1_viewpoint_id"]), {int(r["s0_viewpoint_id"]), int(r["s1_viewpoint_id"])}, pair[(str(r["scene_id"]), str(r["region"]))], az[(str(r["scene_id"]), str(r["region"]))]) if not bool(train_v0[str(r["episode_id"])]["predicted_stays"]) else [] for r in train_rows}
    joint_payload = torch.load(data_root / "experiments/stage_d/EXP050_joint_rollout_revision/joint_revision_final.pth", map_location=device, weights_only=False); joint = _JointRevision().to(device); joint.load_state_dict(joint_payload["model_state_dict"]); joint.eval(); train_v1 = _joint_select(joint, train_cache, train_rows, train_orders, "ALL_LEGAL", device)
    rgb = RGBIndex(); rgb.add_cache(RGB_BASE); rgb.add_cache(RGB_EXP052); rgb.add_cache(RGB_VAL)
    available_v1 = [v is not None and (*context_key(row), int(v)) in rgb.values for row, v in zip(train_rows, train_v1)]
    # Existing DINO caches are the approved provenance.  Do not fill missing
    # rollout observations with placeholders; keep only complete real-history
    # samples and record the skipped contexts in the result.
    rollout_for_train = [v if keep else None for v, keep in zip(train_v1, available_v1)]
    rgb_info = {"cache_sources": [str(RGB_BASE), str(RGB_EXP052), str(RGB_VAL)], "new_extraction_performed": False, "future_candidate_rgb_used": False}
    contexts, _ = _build_contexts(train_rows, train_sources, train_cache, rgb, rollout_for_train, train=True); dataset = RecognitionDataset(contexts); model = CounterfactualRecognitionModel().to(device); checkpoint = data_root / "experiments/stage_d/EXP053_counterfactual_recognition_world_model/runtime/counterfactual_recognition_model.pth"
    existing_result = REPO_ROOT / "experiments/stage_d/EXP053_counterfactual_recognition_world_model/result.json"
    if checkpoint.is_file() and existing_result.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"]); train_info = dict(json.loads(existing_result.read_text(encoding="utf-8")).get("train", {})); train_info["resumed_from_existing_checkpoint"] = True
    else:
        train_info = _train(model, dataset, device, checkpoint, args.epochs, args.batch_size)
    val_info = _run_val(data_root, model.eval(), device, data_root / "experiments/stage_d/EXP050_joint_rollout_revision/joint_revision_final.pth"); out = REPO_ROOT / "experiments/stage_d/EXP053_counterfactual_recognition_world_model"; out.mkdir(parents=True, exist_ok=True)
    orig = json.loads(Path("/tmp/activeview_exp052_r2_original3/result.json").read_text()); dh = val_info
    changed = [i for i, (a, b) in enumerate(zip(dh["original_signatures"], dh["action_signatures"])) if a != b]
    original_terminal = np.asarray(orig["h2_terminal_predictions_moving"], dtype=np.int64); new_terminal = np.asarray(val_info["terminal_predictions"], dtype=np.int64)
    # The per-episode predictions are retained only for this paired audit.
    if new_terminal.size != len(val_info["labels"]):
        raise RuntimeError("EXP053 terminal prediction alignment failure")
    labels_array = np.asarray(val_info["labels"], dtype=np.int64); old_correct = original_terminal == labels_array; new_correct = new_terminal == labels_array
    rescued = int(np.sum(~old_correct & new_correct)); harmful = int(np.sum(old_correct & ~new_correct))
    result = {"experiment_id": "EXP053", "status": "COMPLETED", "split": ["train", "val"], "test_used": False, "training_performed": True, "train": {**train_info, "contexts": len(train_rows), "canonical_samples": len(train_rows) * 4, "rollout_contexts": int(sum(available_v1)), "rollout_samples": int(sum(available_v1) * 4), "total_samples": len(dataset), "rollout_v1_selected_original": int(sum(v is not None for v in train_v1)), "rollout_v1_rgb_available": int(sum(available_v1)), "rollout_v1_rgb_missing_skipped": int(sum(v is not None for v in train_v1) - sum(available_v1))}, "rgb_history": rgb_info, "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256(checkpoint)}, "h0_fidelity": val_info["h0_fidelity"], "h1_fidelity": val_info["h1_fidelity"], "h2": {k: val_info[k] for k in ("h2_moving", "h2_full", "h2_fused_moving", "h2_fused_full")}, "action_audit": {"changed_trajectories": len(changed), "rescued": rescued, "harmful": harmful, "net": rescued - harmful}, "provenance": {"joint_revision_frozen": True, "stgcn_frozen": True, "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()}}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n"); (out / "fidelity_comparison.json").write_text(json.dumps({"original_wm_e_h0": orig["history_shift_fidelity"]["h0"], "exp053_h0": val_info["h0_fidelity"], "original_wm_e_h1": orig["history_shift_fidelity"]["h1"], "exp053_h1": val_info["h1_fidelity"]}, indent=2) + "\n"); (out / "h2_comparison.json").write_text(json.dumps({"original": {"moving": orig["h2_real"]["moving_subset"], "full": orig["h2_real"]["full"]}, "exp053": {"moving": val_info["h2_moving"], "full": val_info["h2_full"]}}, indent=2) + "\n"); (out / "checkpoint_manifest.json").write_text(json.dumps(result["checkpoint"], indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
