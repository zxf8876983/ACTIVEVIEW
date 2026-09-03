#!/usr/bin/env python3
"""EXP052 diverse-history WM-E training and frozen Val evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from activeview.active_view.stage_d_dataset import load_jsonl
from activeview.active_view.stage_d_dense_campaign import context_key, relative_view_descriptor
from activeview.active_view.stage_d_rgb_context import RGBObservationKey, load_dinov2
from activeview.active_view.stage_d_rgb_spatial import build_or_load_spatial_cache
from activeview.active_view.stage_d_world_model import CandidateObservationWorldModel, world_model_loss
from activeview.scripts.run_stage_d_exp041_044 import _episode_sources, _rows
from activeview.scripts.run_stage_d_exp046_048 import _load_cache, _load_stgcn
from activeview.scripts.run_stage_d_exp042r1_045 import _log_probs as _log_probs_torch
from activeview.scripts.run_exp051_r1_closed_loop import _candidate_order
from activeview.scripts.run_exp051_r2_real_eval import _fidelity

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEW_COUNT = 32
PAIRS_PER_CONTEXT = 4
CANDIDATES_PER_PAIR = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed() -> None:
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


class DiverseHistoryDataset(Dataset[dict[str, Any]]):
    """Lazy samples with four deterministic diverse ordered history pairs."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], sources: Mapping[tuple[str, str, str], str], cache: Mapping[str, np.ndarray], rgb_lookup: Mapping[tuple[str, str, str, int], np.ndarray], *, seed: int = 42) -> None:
        self.rows = list(rows); self.sources = dict(sources); self.cache = cache; self.rgb_lookup = rgb_lookup
        self._archives: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._cache_index = {str(value): index for index, value in enumerate(cache["episode_ids"].tolist())}
        rng = np.random.default_rng(seed)
        self.samples: list[tuple[int, int, int, int]] = []
        self.history_keys: set[tuple[str, str, str, int]] = set()
        for row_index, row in enumerate(self.rows):
            common = context_key(row); s0, s1 = int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])
            pairs = [(s0, s1)]
            all_pairs = [(i, j) for i in range(VIEW_COUNT) for j in range(VIEW_COUNT) if i != j and (i, j) != (s0, s1)]
            for pair_index in rng.choice(len(all_pairs), size=PAIRS_PER_CONTEXT - 1, replace=False): pairs.append(all_pairs[int(pair_index)])
            for vi, vj in pairs:
                candidates = [v for v in range(VIEW_COUNT) if v not in {vi, vj}]
                for candidate in rng.choice(candidates, size=CANDIDATES_PER_PAIR, replace=False).tolist():
                    self.samples.append((row_index, int(vi), int(vj), int(candidate)))
                self.history_keys.update((*common, int(vi)) for _ in [0]); self.history_keys.update((*common, int(vj)) for _ in [0])

    def __len__(self) -> int:
        return len(self.samples)

    def _archive(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path); cached = self._archives.get(key)
        if cached is not None: self._archives.move_to_end(key); return cached
        with np.load(path, allow_pickle=False) as archive:
            value = {"skeleton": np.asarray(archive["skeleton"], dtype=np.float32), "viewpoint_ids": np.asarray(archive["viewpoint_ids"], dtype=np.int64), "positions": np.asarray(archive["viewpoint_agent_positions"], dtype=np.float32)}
        self._archives[key] = value; self._archives.move_to_end(key)
        while len(self._archives) > 8: self._archives.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, vi, vj, candidate = self.samples[index]; row = self.rows[row_index]; key = context_key(row); archive = self._archive(Path(self.sources[key])); ids = archive["viewpoint_ids"]; by_id = {int(view): pos for pos, view in enumerate(ids.tolist())}; positions = archive["positions"]
        if any(view not in by_id for view in (vi, vj, candidate)): raise ValueError(f"viewpoint alignment failure: {key}")
        current = positions[by_id[vj]]
        cache_index = self._cache_index[str(row["episode_id"])]
        history_ids = (vi, vj)
        return {"history_skeleton": torch.from_numpy(np.stack([archive["skeleton"][by_id[v]] for v in history_ids])), "history_descriptor": torch.from_numpy(np.stack([relative_view_descriptor(positions, current, v) for v in history_ids])), "candidate_descriptor": torch.from_numpy(relative_view_descriptor(positions, current, candidate)), "target_skeleton": torch.from_numpy(archive["skeleton"][by_id[candidate]]), "history_belief": torch.from_numpy(np.concatenate([self.cache["true_logp"][cache_index, vi], self.cache["true_logp"][cache_index, vj]]).astype(np.float32)), "history_rgb": torch.from_numpy(np.stack([np.asarray(self.rgb_lookup[(*key, v)], dtype=np.float32) for v in history_ids])), "context_key": key, "history_ids": history_ids, "candidate_id": candidate}


def _collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"history_skeleton": torch.stack([item["history_skeleton"] for item in batch]), "history_descriptor": torch.stack([item["history_descriptor"] for item in batch]), "candidate_descriptor": torch.stack([item["candidate_descriptor"] for item in batch]), "target_skeleton": torch.stack([item["target_skeleton"] for item in batch]), "history_belief": torch.stack([item["history_belief"] for item in batch]), "history_rgb": torch.stack([item["history_rgb"] for item in batch]), "context_key": [item["context_key"] for item in batch]}


def _build_rgb_cache(rows: Sequence[Mapping[str, Any]], dataset: DiverseHistoryDataset, cache_dir: Path, device: torch.device) -> tuple[dict[tuple[str, str, str, int], np.ndarray], dict[str, Any]]:
    keys = [RGBObservationKey(*key) for key in sorted(dataset.history_keys)]
    values, manifest, info = build_or_load_spatial_cache(rgb_root=Path("/home/zxf/MG08/robot/ActiveView/datasets/offline/hm3d-train"), cache_dir=cache_dir, keys=keys, model_loader=load_dinov2, device=device, batch_size=64)
    lookup = {tuple((str(item["scene_id"]), str(item["region"]), str(item["record_id"]), int(item["viewpoint_id"]))): np.asarray(values[index], dtype=np.float32) for index, item in enumerate(manifest)}
    return lookup, {**info, "unique_history_observations": len(keys), "future_candidate_rgb_used": False}


def _train(dataset: DiverseHistoryDataset, stgcn: torch.nn.Module, *, device: torch.device, output: Path, epochs: int, batch_size: int, workers: int) -> dict[str, Any]:
    model = CandidateObservationWorldModel(use_belief=True, use_rgb=True, residual=False).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=_collate, pin_memory=True, persistent_workers=workers > 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float]] = []; started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for batch in loader:
            values = {name: batch[name].to(device, non_blocking=True) for name in ("history_skeleton", "history_descriptor", "candidate_descriptor", "history_belief", "history_rgb")}
            target = batch["target_skeleton"].to(device, non_blocking=True)
            prediction = model(**values)
            pose_loss, _, _ = world_model_loss(prediction, target)
            target_logp = _log_probs_torch(stgcn, target, device)
            pred_logp = _log_probs_torch(stgcn, prediction, device)
            recognition_loss = torch.nn.functional.kl_div(pred_logp, target_logp.exp(), reduction="batchmean")
            loss = pose_loss + 0.10 * recognition_loss
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()) * len(batch["context_key"]); count += len(batch["context_key"])
        epoch_result = {"epoch": epoch, "loss": total / max(count, 1)}; history.append(epoch_result)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "variant": "EXP052-WM-E-DH", "epoch": epoch, "seed": 42}, output)
        print(f"EXP052 epoch {epoch}/{epochs} loss={epoch_result['loss']:.6f}", flush=True)
    return {"samples": len(dataset), "contexts": len(dataset.rows), "samples_per_context": PAIRS_PER_CONTEXT * CANDIDATES_PER_PAIR, "epochs": epochs, "batch_size": batch_size, "optimizer": "Adam", "learning_rate": 1e-3, "recognition_weight": 0.10, "final_loss": history[-1]["loss"], "history": history, "elapsed_seconds": time.monotonic() - started, "checkpoint": str(output.resolve()), "checkpoint_sha256": _sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EXP052 diverse-history WM-E on Train only")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/zxf/MG08/robot/ActiveView/features/dinov2_vitb14_spatial4x4_exp052_train"))
    args = parser.parse_args(); _seed(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    rows = _rows(args.data_root.resolve(), "train"); sources = _episode_sources(args.data_root.resolve(), "train")
    if len(rows) != 29133: raise RuntimeError("canonical Train population mismatch")
    cache_path = args.data_root / "experiments/stage_d/EXP046_counterfactual_recognition_dataset/train_cache.npz"; cache = _load_cache(cache_path)
    dataset = DiverseHistoryDataset(rows, sources, cache, {})
    rgb_lookup, rgb_info = _build_rgb_cache(rows, dataset, args.cache_dir, device); dataset.rgb_lookup = rgb_lookup
    stgcn = _load_stgcn(args.data_root.resolve(), device)
    output = args.data_root / "experiments/stage_d/EXP052_diverse_history_world_model/runtime/wm_e_diverse_history.pth"
    train_info = _train(dataset, stgcn, device=device, output=output, epochs=args.epochs, batch_size=args.batch_size, workers=args.workers)
    result = {"experiment_id": "EXP052", "status": "TRAINED", "split": "train", "test_used": False, "training_performed": True, "wm_e_frozen_control": str(Path("/tmp/activeview_exp042r1_E/last.pth")), "sampling": {"seed": 42, "pairs_per_context": PAIRS_PER_CONTEXT, "candidates_per_pair": CANDIDATES_PER_PAIR, "distinct_ordered_pairs": True}, "rgb_cache": rgb_info, "train": train_info, "leakage_flags": {"future_candidate_rgb_used": False, "true_u2_used_as_input": False, "gt_pose_target": False, "test_accessed": False}}
    out = REPO_ROOT / "experiments/stage_d/EXP052_diverse_history_world_model"; out.mkdir(parents=True, exist_ok=True); (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
