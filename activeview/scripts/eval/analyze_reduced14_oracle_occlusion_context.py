#!/usr/bin/env python3
"""Train/Val diagnostic for oracle candidate-specific scene visibility.

The experiment augments the frozen Action-Discriminative WM-E candidate
descriptor with a small static Habitat ray-cast descriptor.  Only Train and
Val Stage-D rows are loaded; no Test path is present in this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import habitat_sim
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root, get_habitat_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.world_model.model import CandidateObservationWorldModel
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _validate_rows_cache,
)
from activeview.scripts.eval.analyze_reduced14_wm_e import _correlation
from activeview.scripts.train.train_reduced14_action_discriminative_world_model import (
    BELIEF_WEIGHT,
    CONTEXT_BATCH_SIZE,
    DATASET_NAME,
    FEATURE_WEIGHT,
    NUM_CLASSES,
    REC_WEIGHT,
    SEED,
    WM_BATCH_SIZE,
    WM_EPOCHS,
    WM_LR,
    WM_WEIGHT_DECAY,
    WM_WORKERS,
    _build_loader,
    _evaluate_wm,
    _filtered_rgb_lookup,
    _load_history_identity,
    _label_names,
    _run_h1,
    _train_step,
)
from activeview.scripts.train.train_reduced14_world_model import _sources


CHECKPOINT_DIR = "activeview_reduced14_eight_placement_v1"
OLD_CHECKPOINT_NAME = "wm_e_action_discriminative_best.pth"
NEW_CHECKPOINT_NAME = "wm_e_oracle_occlusion_context_best.pth"
OCCLUSION_DIM = 10
SENSOR_HEIGHT_M = 1.10
RAY_EPSILON_M = 0.03
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/oracle_occlusion_context"
CACHE_DIR_NAME = "oracle_occlusion_context_reduced14_eight_placement_v1"

# Eight fixed body landmarks are deliberately action- and recognition-free.
LANDMARK_NAMES = (
    "root",
    "pelvis",
    "torso",
    "head",
    "left_upper_body",
    "right_upper_body",
    "left_lower_body",
    "right_lower_body",
)
LANDMARK_OFFSETS = np.asarray(
    [
        (0.00, 0.10, 0.00),
        (0.00, 0.85, 0.00),
        (0.00, 1.20, 0.00),
        (0.00, 1.70, 0.00),
        (-0.20, 1.30, 0.00),
        (0.20, 1.30, 0.00),
        (-0.15, 0.50, 0.00),
        (0.15, 0.50, 0.00),
    ],
    dtype=np.float32,
)


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_sim(habitat_root: Path, scene_id: str) -> habitat_sim.Simulator:
    scene_dir = habitat_root / "hm3d-train" / scene_id
    glbs = sorted(scene_dir.glob("*.basis.glb"))
    navmeshes = sorted(scene_dir.glob("*.basis.navmesh"))
    if not glbs or not navmeshes:
        raise FileNotFoundError(f"missing HM3D assets for {scene_id}: {scene_dir}")
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glbs[0].resolve())
    backend.enable_physics = True
    agent = habitat_sim.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    sim.pathfinder.load_nav_mesh(str(navmeshes[0].resolve()))
    return sim


def _rotate_y(offsets: np.ndarray, yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(yaw_deg))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    rotation = np.asarray(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float32,
    )
    return offsets @ rotation.T


def _ray_visibility(
    sim: habitat_sim.Simulator,
    origin: np.ndarray,
    endpoint: np.ndarray,
) -> tuple[float, float]:
    delta = endpoint - origin
    endpoint_distance = float(np.linalg.norm(delta))
    if not np.isfinite(endpoint_distance) or endpoint_distance <= 1e-6:
        return 0.0, 0.0
    ray = habitat_sim.geo.Ray(origin.astype(np.float32), (delta / endpoint_distance).astype(np.float32))
    hits = sim.cast_ray(ray).hits
    distances = [
        float(getattr(hit, "ray_distance"))
        for hit in hits
        if np.isfinite(float(getattr(hit, "ray_distance", np.inf)))
    ]
    first = min(distances) if distances else endpoint_distance
    visible = float(first >= endpoint_distance - RAY_EPSILON_M)
    normalized_obstruction = float(np.clip(first / endpoint_distance, 0.0, 1.0))
    return visible, normalized_obstruction


def _descriptor(
    sim: habitat_sim.Simulator,
    camera_position: Sequence[float],
    human_position: Sequence[float],
    human_yaw_deg: float,
) -> np.ndarray:
    camera = np.asarray(camera_position, dtype=np.float32).reshape(3).copy()
    camera[1] += SENSOR_HEIGHT_M
    human = np.asarray(human_position, dtype=np.float32).reshape(3)
    landmarks = human[None, :] + _rotate_y(LANDMARK_OFFSETS, human_yaw_deg)
    values = [_ray_visibility(sim, camera, endpoint) for endpoint in landmarks]
    visibility = np.asarray([value[0] for value in values], dtype=np.float32)
    obstruction = np.asarray([value[1] for value in values], dtype=np.float32)
    return np.concatenate(
        [visibility, np.asarray([visibility.mean(), obstruction.mean()], dtype=np.float32)]
    ).astype(np.float32)


def _candidate_metadata(data_root: Path, scene_id: str) -> dict[str, Any]:
    path = (
        data_root
        / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
        / scene_id
        / "candidate_metadata/manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("version") != "furniture-placement-v2" or int(metadata.get("placements", 0)) != 8:
        raise ValueError(f"candidate metadata is not reduced14 8-placement: {path}")
    if len(metadata.get("placements_data", [])) != 8:
        raise ValueError(f"candidate metadata placement count mismatch: {path}")
    return metadata


def _cache_paths(data_root: Path) -> tuple[Path, Path, Path]:
    cache = data_root / "features" / CACHE_DIR_NAME
    return cache / "descriptors.npz", cache / "manifest.jsonl", cache / "summary.json"


def _decode_cache_key(value: str) -> tuple[str, str, int]:
    scene, placement, viewpoint = value.split("|", 2)
    return scene, placement, int(viewpoint)


def _build_occlusion_cache(
    data_root: Path,
    habitat_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, ...], np.ndarray], dict[str, Any]]:
    descriptor_path, manifest_path, summary_path = _cache_paths(data_root)
    scene_ids = sorted({str(row["scene_id"]) for row in rows})
    if descriptor_path.is_file() and manifest_path.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("version") == "oracle-occlusion-context-v1"
            and summary.get("descriptor_dim") == OCCLUSION_DIM
            and summary.get("scenes") == scene_ids
        ):
            with np.load(descriptor_path, allow_pickle=False) as archive:
                keys = [str(value) for value in archive["keys"].tolist()]
                descriptors = np.asarray(archive["descriptors"], dtype=np.float32)
            if descriptors.shape == (len(keys), OCCLUSION_DIM):
                return {
                    _decode_cache_key(key): descriptors[index]
                    for index, key in enumerate(keys)
                }, summary

    keys: list[str] = []
    descriptors: list[np.ndarray] = []
    started = time.perf_counter()
    for scene_id in scene_ids:
        metadata = _candidate_metadata(data_root, scene_id)
        sim = _scene_sim(habitat_root, scene_id)
        try:
            for placement in metadata["placements_data"]:
                placement_id = str(placement["placement_id"])
                human_position = np.asarray(placement["position"], dtype=np.float32)
                yaw_deg = float(placement["yaw_deg"])
                viewpoints = placement.get("viewpoints", [])
                if len(viewpoints) != 32:
                    raise ValueError(f"{scene_id}/{placement_id} does not contain 32 viewpoints")
                for viewpoint in viewpoints:
                    viewpoint_id = int(viewpoint["viewpoint_id"])
                    camera = viewpoint.get("snapped_position", viewpoint.get("position"))
                    key = f"{scene_id}|{placement_id}|{viewpoint_id}"
                    keys.append(key)
                    descriptors.append(_descriptor(sim, camera, human_position, yaw_deg))
        finally:
            sim.close()
        print(f"occlusion descriptors: {scene_id} ({len(keys)} total)", flush=True)
    descriptor_array = np.asarray(descriptors, dtype=np.float32)
    if descriptor_array.shape != (len(scene_ids) * 8 * 32, OCCLUSION_DIM):
        raise ValueError(f"invalid occlusion cache shape: {descriptor_array.shape}")
    cache_dir = descriptor_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(descriptor_path, keys=np.asarray(keys), descriptors=descriptor_array)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for key, descriptor in zip(keys, descriptor_array):
            scene, placement, viewpoint = key.split("|")
            handle.write(json.dumps({
                "scene_id": scene,
                "placement_id": placement,
                "viewpoint_id": int(viewpoint),
                "frame_index": 15,
                "feature_schema": list(LANDMARK_NAMES) + ["visible_keypoint_ratio", "mean_normalized_obstruction_distance"],
                "static_scene_raycast_only": True,
                "descriptor_dim": OCCLUSION_DIM,
                "descriptor": descriptor.tolist(),
            }) + "\n")
    summary = {
        "version": "oracle-occlusion-context-v1",
        "scenes": scene_ids,
        "scene_count": len(scene_ids),
        "placements_per_scene": 8,
        "viewpoints_per_placement": 32,
        "descriptor_dim": OCCLUSION_DIM,
        "feature_schema": list(LANDMARK_NAMES) + ["visible_keypoint_ratio", "mean_normalized_obstruction_distance"],
        "sensor_height_m": SENSOR_HEIGHT_M,
        "ray_epsilon_m": RAY_EPSILON_M,
        "static_scene_raycast_only": True,
        "train_val_rows_only": True,
        "test_used": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        _decode_cache_key(key): descriptor_array[index]
        for index, key in enumerate(keys)
    }, summary


def _load_action_model(checkpoint: Path, device: torch.device) -> CandidateObservationWorldModel:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if state is None:
        raise ValueError(f"invalid Action-Discriminative WM-E checkpoint: {checkpoint}")
    model = CandidateObservationWorldModel(
        use_belief=True,
        use_rgb=True,
        residual=False,
        num_classes=NUM_CLASSES,
        use_action_discriminative_heads=True,
        candidate_context_dim=OCCLUSION_DIM,
    ).to(device)
    old_weight = state.get("candidate_encoder.0.weight")
    if old_weight is None or tuple(old_weight.shape) != (64, 9):
        raise ValueError("unexpected old candidate encoder shape")
    filtered = {key: value for key, value in state.items() if key != "candidate_encoder.0.weight"}
    loaded = model.load_state_dict(filtered, strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys) != {"candidate_encoder.0.weight"}:
        raise ValueError(f"unexpected Action-Discriminative state mismatch: {loaded}")
    with torch.no_grad():
        model.candidate_encoder[0].weight[:, :9].copy_(old_weight)
        model.candidate_encoder[0].weight[:, 9:].zero_()
    return model


def _rename_occlusion_selectors(h1: dict[str, Any]) -> None:
    selectors = h1["selectors"]
    for old, new in (
        (
            "Action_Discriminative_WM_imagined_Min_entropy_H1",
            "Oracle_Occlusion_Context_WM_imagined_Min_entropy_H1",
        ),
        (
            "Action_Discriminative_WM_imagined_Max_margin_H1",
            "Oracle_Occlusion_Context_WM_imagined_Max_margin_H1",
        ),
    ):
        if old in selectors:
            selectors[new] = selectors.pop(old)
            selectors[new]["selector"] = new


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    wm = result["wm_val"]
    old = result["comparison"]["action_discriminative_wm_e"]
    selectors = result["h1"]["selectors"]
    lines = [
        "# Oracle Candidate Occlusion Context (reduced14, Train/Val)",
        "",
        "Only Train/Val Stage-D contexts were read; Test was not read.",
        "",
        "## Candidate descriptor",
        "",
        f"Static Habitat ray-cast descriptor dimension: {OCCLUSION_DIM}. It contains LOS bits for "
        f"{', '.join(LANDMARK_NAMES)}, visible ratio, and mean normalized obstruction distance.",
        "",
        "## WM-E diagnostics",
        "",
        "| Metric | Action-discriminative WM-E | Oracle-occlusion WM-E |",
        "|---|---:|---:|",
    ]
    metrics = (
        ("recognition_agreement", "recognition agreement"),
        ("pearson", "candidate true-class Pearson"),
        ("spearman", "candidate true-class Spearman"),
        ("feature_cosine_similarity", "feature cosine"),
        ("belief_kl", "belief KL"),
        ("belief_entropy_pearson", "belief entropy Pearson"),
        ("belief_entropy_spearman", "belief entropy Spearman"),
        ("belief_margin_pearson", "belief margin Pearson"),
        ("belief_margin_spearman", "belief margin Spearman"),
        ("top1_positive_hit", "Top-1 positive hit"),
        ("top3_positive_hit", "Top-3 positive hit"),
    )
    for key, label in metrics:
        lines.append(f"| {label} | {old.get(key)} | {wm.get(key)} |")
    lines.extend([
        "",
        "## Imagined H1 identity",
        "",
        "| Selector | Accuracy | Macro-F1 | Mean entropy | s0 correction |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in selectors.items():
        identity = item["identity"]
        lines.append(
            f"| {name} | {identity['accuracy']:.6f} | {identity['macro_f1']:.6f} | "
            f"{item['mean_entropy']:.6f} | {item['s0_error_correction_rate']:.6f} |"
        )
    occlusion_min = selectors.get("Oracle_Occlusion_Context_WM_imagined_Min_entropy_H1")
    occlusion_max = selectors.get("Oracle_Occlusion_Context_WM_imagined_Max_margin_H1")
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"Oracle occlusion context changed candidate recognition agreement from {old.get('recognition_agreement')} "
        f"to {wm.get('recognition_agreement')}, and candidate true-class Spearman from {old.get('spearman')} "
        f"to {wm.get('spearman')}.",
        f"The imagined H1 min-entropy Accuracy is {occlusion_min['identity']['accuracy'] if occlusion_min else None}; "
        f"max-margin Accuracy is {occlusion_max['identity']['accuracy'] if occlusion_max else None}.",
        "This is a privileged static-geometry diagnostic: no action labels, recognition outputs, or real candidate predictions enter the occlusion descriptor.",
        "If candidate fidelity and imagined H1 improve substantially, candidate-specific scene observability is a likely information gap; otherwise static occlusion alone is insufficient.",
        "",
        "Leakage audit: test_used=false; no Test rows/cache were read; formal WM-E, JR, ST-GCN and taxonomy artifacts were not overwritten.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root: Path,
    device: torch.device,
    habitat_root: Path,
    evaluate_only: bool = False,
) -> dict[str, Any]:
    _seed()
    started = time.perf_counter()
    policy_root = data_root / "datasets" / DATASET_NAME
    train_rows = load_jsonl(policy_root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    if any(str(row.get("policy_split", "")).lower() != "train" for row in train_rows):
        raise ValueError("Train split contamination")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows):
        raise ValueError("Val split contamination")
    occlusion_lookup, occlusion_summary = _build_occlusion_cache(
        data_root, habitat_root, [*train_rows, *val_rows]
    )
    rgb_lookup = _filtered_rgb_lookup(data_root, [*train_rows, *val_rows])
    train_loader = _build_loader(
        data_root, train_rows, rgb_lookup, WM_BATCH_SIZE, WM_WORKERS, True, occlusion_lookup
    )
    val_loader = _build_loader(
        data_root, val_rows, rgb_lookup, WM_BATCH_SIZE, 2, False, occlusion_lookup
    )
    checkpoint_root = data_root / "checkpoints" / CHECKPOINT_DIR
    old_checkpoint = checkpoint_root / OLD_CHECKPOINT_NAME
    new_checkpoint = checkpoint_root / NEW_CHECKPOINT_NAME
    model = _load_action_model(old_checkpoint, device)
    teacher_checkpoint = checkpoint_root.parent / "stgcn_reduced14_kneel_babel_diversity_v1" / "stgcn_reduced14_kneel_best.pth"
    teacher, _ = load_checkpoint(teacher_checkpoint, NUM_CLASSES, str(device))
    identity, identity_checkpoint = _load_history_identity(data_root, device)
    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    if evaluate_only:
        if not new_checkpoint.is_file():
            raise FileNotFoundError(f"cannot evaluate without checkpoint: {new_checkpoint}")
        checkpoint_payload = torch.load(new_checkpoint, map_location=device, weights_only=False)
        best_epoch = int(checkpoint_payload.get("epoch", 0))
        val_history = [dict(checkpoint_payload.get("val_metrics", {}))]
        print(f"reusing trained oracle-occlusion checkpoint epoch {best_epoch}", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY)
        best_key = (-np.inf, -np.inf)
        best_epoch = 0
        for epoch in range(1, WM_EPOCHS + 1):
            model.train()
            batches: list[dict[str, float]] = []
            for batch in train_loader:
                loss, stats = _train_step(model, teacher, identity, batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batches.append(stats)
            train_stats = {key: float(np.mean([item[key] for item in batches])) for key in batches[0]}
            train_stats["epoch"] = float(epoch)
            train_history.append(train_stats)
            val_stats = _evaluate_wm(model, teacher, identity, val_loader, device)
            val_stats["epoch"] = epoch
            val_history.append(val_stats)
            key = (
                float(val_stats["belief_entropy_pearson"] or -1.0),
                float(val_stats["belief_margin_spearman"] or -1.0),
            )
            if key > best_key:
                best_key = key
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "state_dict": model.state_dict(),
                        "variant": "E_action_discriminative_oracle_occlusion_context",
                        "epoch": epoch,
                        "seed": SEED,
                        "num_classes": NUM_CLASSES,
                        "candidate_context_dim": OCCLUSION_DIM,
                        "loss_weights": {"recognition": REC_WEIGHT, "feature": FEATURE_WEIGHT, "belief": BELIEF_WEIGHT},
                        "val_metrics": val_stats,
                    },
                    new_checkpoint,
                )
            print(
                f"Oracle-occlusion WM epoch {epoch}/{WM_EPOCHS} loss={train_stats['loss']:.6f} "
                f"val_entropy_r={val_stats['belief_entropy_pearson']}",
                flush=True,
            )
    payload = torch.load(new_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    val_cache = _load_npz(policy_root / "counterfactual_cache/val.npz")
    _validate_rows_cache(val_rows, val_cache, "val")
    names = _label_names(data_root)
    h1_result, _ = _run_h1(
        data_root,
        val_rows,
        val_cache,
        rgb_lookup,
        model,
        device,
        names,
        occlusion_lookup,
    )
    _rename_occlusion_selectors(h1_result)
    prior_path = REPO_ROOT / "experiments/reduced14_eight_placement_v1/action_discriminative_wm_e/result.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    old_wm_val = prior["wm_val"]
    for name, value in prior.get("h1", {}).get("selectors", {}).items():
        h1_result["selectors"].setdefault(name, value)
    final_wm_val = val_history[-1] if evaluate_only else val_history[best_epoch - 1]
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_ORACLE_CANDIDATE_OCCLUSION_CONTEXT",
        "status": "COMPLETED",
        "split": "train_val",
        "test_used": False,
        "population": {
            "train_contexts": len(train_rows),
            "val_moving_contexts": len(val_rows),
            "val_candidate_hypotheses": h1_result["inference"]["candidate_hypotheses"],
        },
        "occlusion_cache": occlusion_summary,
        "training": {
            "epochs": WM_EPOCHS,
            "batch_size": WM_BATCH_SIZE,
            "workers": WM_WORKERS,
            "learning_rate": WM_LR,
            "weight_decay": WM_WEIGHT_DECAY,
            "seed": SEED,
            "best_epoch": best_epoch,
            "final_train_loss": train_history[-1]["loss"] if train_history else None,
            "train_history": train_history,
            "val_history": val_history,
            "elapsed_seconds": time.perf_counter() - training_started,
            "checkpoint": str(new_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(new_checkpoint),
        },
        "wm_val": final_wm_val,
        "comparison": {"action_discriminative_wm_e": old_wm_val},
        "h1": h1_result,
        "artifacts": {
            "old_wm_checkpoint": str(old_checkpoint.resolve()),
            "old_wm_untouched": True,
            "new_wm_checkpoint": str(new_checkpoint.resolve()),
            "stgcn_checkpoint": str(teacher_checkpoint.resolve()),
            "history_identity_checkpoint": str(identity_checkpoint.resolve()),
            "occlusion_cache": str(_cache_paths(data_root)[0].resolve()),
        },
        "protocol": {
            "candidate_descriptor_base_dim": 9,
            "candidate_occlusion_descriptor_dim": OCCLUSION_DIM,
            "candidate_conditioning": "[9-D relative geometry, 10-D static Habitat raycast descriptor]",
            "landmarks": list(LANDMARK_NAMES),
            "static_scene_raycast_only": True,
            "raycast_sensor_height_m": SENSOR_HEIGHT_M,
            "horizon": "single candidate observation diagnostic",
            "teacher": "frozen reduced14 ST-GCN",
            "history_identity": "frozen pretrained History Identity",
            "candidate_budget": "ALL_LEGAL",
        },
        "leakage_flags": {
            "test_used": False,
            "future_candidate_observation_used_as_wm_input": False,
            "gt_action_label_used_in_occlusion_descriptor": False,
            "candidate_recognition_used_in_occlusion_descriptor": False,
            "formal_stgcn_modified": False,
            "formal_jr_modified": False,
            "old_wm_overwritten": False,
            "train_val_rows_only": True,
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--habitat-root", type=Path, default=get_habitat_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate-only", action="store_true", help="reuse the already trained oracle-occlusion checkpoint")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Oracle occlusion WM-E requires CUDA; CPU fallback is disabled")
    run(args.data_root.resolve(), device, args.habitat_root.resolve(), args.evaluate_only)


if __name__ == "__main__":
    main()
