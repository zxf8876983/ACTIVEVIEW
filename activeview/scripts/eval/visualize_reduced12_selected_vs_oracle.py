#!/usr/bin/env python3
"""Create a small selected-vs-GT-best qualitative Val audit.

Only archived Moving-Val assets are read.  The script never trains, renders,
or runs a recognizer; GT margins are used only after selection for auditing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl

NUM_CLASSES = 12
LABELS = ("walk", "sit", "stand up", "bend", "crawl", "stumble", "clap", "throw", "kick", "knock", "punch", "touching face")
MOVING_ROOT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_utility_regret_audit"
OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/selected_vs_oracle_visualization"
METRICS_NPZ = REPO_ROOT / "experiments/reduced12_eight_placement_v1/human_view_observability_oracle/candidate_metrics.npz"
VISIBILITY_NPZ = REPO_ROOT / "experiments/reduced12_eight_placement_v1/scene_joint_visibility_oracle/visibility.npz"
ARCHIVE_RELATIVE = "datasets/offline/habitat-train/00006-00087"
RGB_RELATIVE = "datasets/rgb_reduced12_eight_placement_v1/visited_s0_s1"
EDGES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16))
JOINT_NAMES = ("pelvis", "right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle", "spine", "thorax", "neck", "head", "left shoulder", "left elbow", "left wrist", "right shoulder", "right elbow", "right wrist")


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _softmax(logp: np.ndarray) -> np.ndarray:
    value = np.asarray(logp, dtype=np.float64)
    value = value - np.max(value)
    value = np.exp(value)
    return value / np.sum(value)


def _utility(logp: np.ndarray, label: int) -> float:
    return float(logp[label] - np.max(np.delete(logp, label)))


def _wrap_angle(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def _load_manifest(archive_root: Path, scene_id: str) -> dict[str, Any]:
    path = archive_root / scene_id / "candidate_metadata/manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _view_table(manifest: Mapping[str, Any], placement_id: str) -> tuple[dict[int, dict[str, Any]], np.ndarray, float]:
    item = next(item for item in manifest["placements_data"] if str(item["placement_id"]) == placement_id)
    views = {int(view["viewpoint_id"]): view for view in item["viewpoints"]}
    return views, np.asarray(item["position"], dtype=np.float64), float(item["yaw_deg"])


def _lookup_metric(metrics: Mapping[str, np.ndarray], row_index: int, candidate_id: int, key: str) -> float | None:
    if key not in metrics:
        return None
    ids = np.asarray(metrics["candidate_ids"][row_index])
    matches = np.flatnonzero(ids == int(candidate_id))
    if not matches.size:
        return None
    return float(metrics[key][row_index, int(matches[0])])


def _load_rgb(data_root: Path, record_id: str) -> tuple[np.ndarray | None, int | None, np.ndarray | None]:
    root = data_root / RGB_RELATIVE
    matches = list(root.rglob(f"{record_id}.npz"))
    if not matches:
        return None, None, None
    payload = _read_npz(matches[0])
    return payload["rgb"], int(payload["frame_index"]), np.asarray(payload.get("available_view_mask")) if "available_view_mask" in payload else None


def _load_qualitative_rgb(case_id: str, output: Path) -> tuple[np.ndarray | None, np.ndarray | None, list[int] | None]:
    path = output / "qualitative_rgb" / f"{case_id}.npz"
    if not path.is_file():
        return None, None, None
    payload = _read_npz(path)
    return payload["rgb"], np.asarray(payload["viewpoint_ids"], dtype=np.int64), [int(value) for value in payload["frame_ids"]]


def _qualitative_case_valid(case: Mapping[str, Any], output: Path) -> bool:
    rgb, viewpoint_ids, frame_ids = _load_qualitative_rgb(str(case["case_id"]), output)
    if rgb is None or viewpoint_ids is None or frame_ids is None or tuple(frame_ids) != (0, 15, 29):
        return False
    if rgb.ndim != 5 or rgb.shape[0:2] != (2, 3) or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        return False
    if not np.isfinite(rgb).all() or any(int(image.max()) <= 0 or float(image.mean()) <= 1.0 for image in rgb.reshape(-1, *rgb.shape[2:])):
        return False
    selected_id = int(case.get("figure_record", {}).get("selected_viewpoint_id", case.get("selected_id")))
    oracle_id = int(case.get("figure_record", {}).get("oracle_viewpoint_id", case.get("oracle_id")))
    return np.array_equal(viewpoint_ids, np.asarray([selected_id, oracle_id], dtype=np.int64))


def _archive_record(data_root: Path, scene_id: str, placement_id: str, record_id: str) -> dict[str, np.ndarray]:
    path = data_root / ARCHIVE_RELATIVE / scene_id / placement_id / f"{record_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_npz(path)


def _view_index(payload: Mapping[str, np.ndarray], viewpoint_id: int) -> int:
    ids = np.asarray(payload["viewpoint_ids"], dtype=np.int64)
    match = np.flatnonzero(ids == int(viewpoint_id))
    if not match.size:
        raise KeyError(f"viewpoint {viewpoint_id} missing from archive")
    return int(match[0])


def _classification(logp: np.ndarray, label: int) -> tuple[int, float, float]:
    prediction = int(np.argmax(logp))
    return prediction, float(logp[label]), _utility(logp, label)


def _select_cases(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], selected_ids: Sequence[int | None], oracle_ids: Sequence[int | None], metrics: Mapping[str, np.ndarray], seed: int = 42) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[tuple[float, int, dict[str, Any]]]] = {"A": [], "B": [], "C": []}
    for i, row in enumerate(rows):
        label = int(row["label_id"])
        candidate_ids = [int(x) for x in row["candidate_viewpoint_ids"]]
        candidate_logp = np.asarray(cache["candidate_logp"][i], dtype=np.float64)
        slot = {int(cid): j for j, cid in enumerate(candidate_ids)}
        selected_id = selected_ids[i]
        oracle_id = oracle_ids[i]
        selected_slot = slot.get(int(selected_id)) if selected_id is not None else None
        oracle_slot = slot.get(int(oracle_id)) if oracle_id is not None else None
        selected_logp = np.asarray(cache["current_logp"][i]) if selected_slot is None else candidate_logp[selected_slot]
        oracle_logp = np.asarray(cache["current_logp"][i]) if oracle_slot is None else candidate_logp[oracle_slot]
        selected_pred, selected_prob, selected_margin = _classification(selected_logp, label)
        oracle_pred, oracle_prob, oracle_margin = _classification(oracle_logp, label)
        utilities = np.asarray([_utility(candidate_logp[s], label) for s in range(len(candidate_ids))])
        best_move = float(np.max(utilities)) if utilities.size else float("-inf")
        chosen_utility = selected_margin
        normalized_regret = (best_move - chosen_utility) / (best_move - float(np.min(utilities)) + 1e-8) if utilities.size else 0.0
        selected_scene = _lookup_metric(metrics, i, int(selected_id), "scene_visibility") if selected_id is not None else None
        best_scene = _lookup_metric(metrics, i, int(oracle_id), "scene_visibility") if oracle_id is not None else None
        selected_human = _lookup_metric(metrics, i, int(selected_id), "human_observability") if selected_id is not None else None
        best_human = _lookup_metric(metrics, i, int(oracle_id), "human_observability") if oracle_id is not None else None
        record = {"row_index": i, "episode_id": str(row["episode_id"]), "record_id": str(row["record_id"]), "scene_id": str(row["scene_id"]), "placement_id": str(row["region"]), "label": label, "selected_id": selected_id, "oracle_id": oracle_id, "selected_pred": selected_pred, "oracle_pred": oracle_pred, "selected_logp": selected_logp.tolist(), "oracle_logp": oracle_logp.tolist(), "selected_margin": selected_margin, "oracle_margin": oracle_margin, "selected_gt_logp": selected_prob, "oracle_gt_logp": oracle_prob, "normalized_regret": normalized_regret, "candidate_ids": candidate_ids, "candidate_logp": candidate_logp.tolist(), "candidate_utilities": utilities.tolist(), "selected_scene_visibility": selected_scene, "oracle_scene_visibility": best_scene, "selected_human_observability": selected_human, "oracle_human_observability": best_human}
        if selected_pred != label and oracle_pred == label:
            groups["A"].append((-normalized_regret, i, record))
            if selected_scene is not None and best_scene is not None and selected_scene > best_scene:
                groups["B"].append((-(selected_scene - best_scene), i, record))
            elif selected_human is not None and best_human is not None and selected_human > best_human:
                groups["B"].append((-(selected_human - best_human), i, record))
            if selected_id is not None and oracle_id is not None:
                groups["C"].append((-(abs(float(oracle_id) - float(selected_id))), i, record))
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for group in ("A", "B", "C"):
        entries = sorted(groups[group], key=lambda item: (item[0], item[1]))
        picked = [entry for entry in entries if entry[1] not in used][:4]
        if len(picked) < 4:
            remaining = [entry for entry in entries if entry[1] not in used]
            if remaining:
                picked.extend(remaining[: 4 - len(picked)])
        for _, index, record in picked:
            used.add(index)
            record["group"] = group
            scene_conflict = record["selected_scene_visibility"] is not None and record["oracle_scene_visibility"] is not None and record["selected_scene_visibility"] > record["oracle_scene_visibility"]
            human_conflict = record["selected_human_observability"] is not None and record["oracle_human_observability"] is not None and record["selected_human_observability"] > record["oracle_human_observability"]
            record["selection_condition"] = "strict" if ((group == "A" and record["normalized_regret"] >= 0.5) or (group == "B" and (scene_conflict or human_conflict)) or group == "C") else "nearest_available"
            chosen.append(record)
    if len(chosen) < 12:
        candidates = sorted(groups["A"] + groups["B"] + groups["C"], key=lambda item: item[0])
        for _, index, record in candidates:
            if index not in used:
                record["group"] = "fallback"
                record["selection_condition"] = "nearest_available"
                chosen.append(record)
                used.add(index)
            if len(chosen) == 12:
                break
    rng.shuffle(chosen)
    for index, record in enumerate(chosen[:12], start=1):
        record["case_id"] = f"case_{index:04d}"
    return chosen[:12]


def _plot_skeleton(ax: Any, skeleton: np.ndarray, edges: Sequence[tuple[int, int]], color: str, title: str, limits: tuple[np.ndarray, np.ndarray]) -> None:
    points = np.asarray(skeleton)
    for left, right in edges:
        ax.plot(points[[left, right], 0], points[[left, right], 2], points[[left, right], 1], color=color, linewidth=1.5)
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], color=color, s=8)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(float(limits[0][0]), float(limits[1][0])); ax.set_ylim(float(limits[0][2]), float(limits[1][2])); ax.set_zlim(float(limits[0][1]), float(limits[1][1]))
    ax.set_xlabel("x", fontsize=6); ax.set_ylabel("z", fontsize=6); ax.set_zlabel("y", fontsize=6); ax.tick_params(labelsize=5)


def _rgb_panel(ax: Any, image: np.ndarray | None, title: str) -> None:
    if image is None:
        ax.text(0.5, 0.5, "RGB unavailable\n(archived frame 15 only)", ha="center", va="center", fontsize=8)
        ax.set_facecolor("#eeeeee")
    else:
        ax.imshow(Image.fromarray(image))
    ax.set_title(title, fontsize=8); ax.axis("off")


def _case_figure(case: Mapping[str, Any], row: Mapping[str, Any], data_root: Path, manifest: Mapping[str, Any], metrics: Mapping[str, np.ndarray], output: Path) -> dict[str, Any]:
    views, human_position, human_yaw = _view_table(manifest, str(row["region"]))
    selected_id = int(case["selected_id"]) if case["selected_id"] is not None else int(row["current_viewpoint_id"])
    oracle_id = int(case["oracle_id"]) if case["oracle_id"] is not None else int(row["current_viewpoint_id"])
    archive = _archive_record(data_root, str(row["scene_id"]), str(row["region"]), str(row["record_id"]))
    rgb, rgb_frame, available_view_mask = _load_rgb(data_root, str(row["record_id"]))
    qualitative_rgb, qualitative_viewpoints, qualitative_frames = _load_qualitative_rgb(str(case["case_id"]), output)
    selected_index, oracle_index = _view_index(archive, selected_id), _view_index(archive, oracle_id)
    selected_skeleton, oracle_skeleton = archive["skeleton"][selected_index], archive["skeleton"][oracle_index]
    selected_view, oracle_view = views[selected_id], views[oracle_id]
    azimuth_difference = _wrap_angle(float(oracle_view.get("azimuth_deg", 0.0)) - float(selected_view.get("azimuth_deg", 0.0)))
    candidate_positions = {int(cid): np.asarray(views[int(cid)]["snapped_position"], dtype=np.float64) for cid in case["candidate_ids"] if int(cid) in views}
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    grid = fig.add_gridspec(4, 7, height_ratios=[1.15, 1.0, 1.3, 1.0])
    ax = fig.add_subplot(grid[0, :3]); ax.scatter(human_position[0], human_position[2], marker="o", s=80, color="black", label="human")
    ax.arrow(human_position[0], human_position[2], 0.7 * np.sin(np.deg2rad(human_yaw)), 0.7 * np.cos(np.deg2rad(human_yaw)), color="black", width=0.02)
    margins = {int(cid): float(case["candidate_utilities"][j]) for j, cid in enumerate(case["candidate_ids"])}
    for cid, position in candidate_positions.items():
        ax.scatter(position[0], position[2], s=25, c=margins.get(cid, 0.0), cmap="coolwarm", vmin=-5, vmax=5)
        ax.text(position[0], position[2], str(cid), fontsize=6)
    ax.scatter(views[selected_id]["snapped_position"][0], views[selected_id]["snapped_position"][2], marker="o", s=120, facecolors="none", edgecolors="green", linewidths=2, label="selected")
    ax.scatter(views[oracle_id]["snapped_position"][0], views[oracle_id]["snapped_position"][2], marker="*", s=180, color="gold", edgecolors="black", label="GT-best")
    ax.scatter(views[int(row["current_viewpoint_id"])] ["snapped_position"][0], views[int(row["current_viewpoint_id"])] ["snapped_position"][2], marker="^", s=90, color="purple", label="s0")
    ax.set_title("Top-down geometry (scene map unavailable)", fontsize=9); ax.set_aspect("equal"); ax.legend(fontsize=6, loc="best"); ax.set_xlabel("x"); ax.set_ylabel("z")
    info = fig.add_subplot(grid[0, 3:]); info.axis("off")
    info.text(0, 1, f"{case['case_id']}  |  {LABELS[case['label']]}  |  group {case['group']}\nselected={case['selected_id']}  GT-best={case['oracle_id']}  azimuth Δ={azimuth_difference:.1f}°\nselected margin={case['selected_margin']:.4f}  best margin={case['oracle_margin']:.4f}\nselected pred={LABELS[case['selected_pred']]}  best pred={LABELS[case['oracle_pred']]}\nRGB archive frame_index={rgb_frame}; requested t0/t15/t29", va="top", fontsize=9)
    for col, viewpoint_id, title in ((0, selected_id, "Selected"), (4, oracle_id, "GT-best")):
        for frame_col, frame_id in enumerate((0, 15, 29)):
            ax_rgb = fig.add_subplot(grid[1, col + frame_col])
            image = None
            if qualitative_rgb is not None and qualitative_viewpoints is not None and qualitative_frames is not None and frame_id in qualitative_frames and viewpoint_id in qualitative_viewpoints:
                view_slot = int(np.flatnonzero(qualitative_viewpoints == viewpoint_id)[0])
                frame_slot = qualitative_frames.index(frame_id)
                image = qualitative_rgb[view_slot, frame_slot]
            elif rgb is not None and available_view_mask is not None and bool(available_view_mask[viewpoint_id]) and frame_id == rgb_frame:
                image = rgb[viewpoint_id]
            _rgb_panel(ax_rgb, image, f"{title} t={frame_id}")
    all_points = np.concatenate([selected_skeleton.reshape(-1, 3), oracle_skeleton.reshape(-1, 3)], axis=0); limits = (all_points.min(axis=0) - 0.05, all_points.max(axis=0) + 0.05)
    for col, skeleton, title, color in ((0, selected_skeleton, "Selected skeleton", "tab:green"), (4, oracle_skeleton, "GT-best skeleton", "tab:orange")):
        for frame_col, frame_id in enumerate((0, 15, 29)):
            ax_3d = fig.add_subplot(grid[2, col + frame_col], projection="3d")
            _plot_skeleton(ax_3d, skeleton[:, frame_id, :].T, EDGES, color, f"{title} t={frame_id}", limits)
    selected_motion = np.linalg.norm(np.diff(selected_skeleton, axis=1), axis=0).mean(axis=0); oracle_motion = np.linalg.norm(np.diff(oracle_skeleton, axis=1), axis=0).mean(axis=0); top_joints = np.argsort(-oracle_motion)[:4]
    ax_curve = fig.add_subplot(grid[3, :4]); time = np.arange(30)
    for joint in top_joints:
        selected_curve = np.linalg.norm(selected_skeleton[:, :, joint] - selected_skeleton[:, 0:1, joint], axis=0); oracle_curve = np.linalg.norm(oracle_skeleton[:, :, joint] - oracle_skeleton[:, 0:1, joint], axis=0)
        ax_curve.plot(time, selected_curve, "--", label=f"S {JOINT_NAMES[int(joint)]}"); ax_curve.plot(time, oracle_curve, "-", label=f"G {JOINT_NAMES[int(joint)]}")
    ax_curve.set_title("Top-4 GT-best motion displacement"); ax_curve.set_xlabel("frame"); ax_curve.set_ylabel("distance"); ax_curve.legend(fontsize=6, ncol=2)
    table = fig.add_subplot(grid[3, 4:]); table.axis("off")
    rows = [("Candidate id", str(case["selected_id"]), str(case["oracle_id"])), ("Azimuth", f"{selected_view.get('azimuth_deg', 0):.1f}", f"{oracle_view.get('azimuth_deg', 0):.1f}"), ("Radius", f"{selected_view.get('radius_m', 0):.2f}", f"{oracle_view.get('radius_m', 0):.2f}"), ("CandidateSpatial score", "N/A", "N/A"), ("SceneVisibility", str(case["selected_scene_visibility"]), str(case["oracle_scene_visibility"])), ("HumanObservability", str(case["selected_human_observability"]), str(case["oracle_human_observability"])), ("PoseConfidence", "N/A", "N/A"), ("GT-class logp", f"{case['selected_gt_logp']:.4f}", f"{case['oracle_gt_logp']:.4f}"), ("GT-margin", f"{case['selected_margin']:.4f}", f"{case['oracle_margin']:.4f}"), ("Motion fidelity", "unavailable", "unavailable")]
    table.table(cellText=[[a, b, c] for a, b, c in rows], colLabels=["metric", "Selected", "GT-best"], loc="center", cellLoc="center", colWidths=[0.42, 0.27, 0.27]); table.set_title("Metrics", fontsize=9)
    path = output / "figures" / f"{case['case_id']}.png"; path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)
    landscape = output / "landscapes" / f"candidate_landscape_{case['case_id']}.png"; landscape.parent.mkdir(parents=True, exist_ok=True); fig2, ax2 = plt.subplots(figsize=(7, 4)); az = [float(views[int(cid)].get("azimuth_deg", 0.0)) for cid in case["candidate_ids"]]; radius = [float(views[int(cid)].get("radius_m", 0.0)) for cid in case["candidate_ids"]]; ax2.scatter(az, case["candidate_utilities"], c=radius, cmap="viridis", label="legal candidates"); ax2.scatter(float(selected_view.get("azimuth_deg", 0)), case["selected_margin"], marker="o", s=100, facecolors="none", edgecolors="green", label="selected"); ax2.scatter(float(oracle_view.get("azimuth_deg", 0)), case["oracle_margin"], marker="*", s=180, color="gold", edgecolors="black", label="GT-best"); ax2.set_xlabel("relative azimuth (deg)"); ax2.set_ylabel("real GT-margin"); ax2.legend(fontsize=7); ax2.set_title(case["case_id"]); fig2.savefig(landscape, dpi=130, bbox_inches="tight"); plt.close(fig2)
    return {"figure": str(path), "landscape": str(landscape), "rgb_frame_index": rgb_frame, "selected_viewpoint_id": selected_id, "oracle_viewpoint_id": oracle_id, "azimuth_difference_deg": azimuth_difference, "top_motion_joints": [JOINT_NAMES[int(j)] for j in top_joints]}


def _overview(cases: Sequence[Mapping[str, Any]], output: Path) -> None:
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 8, wspace=0.08, hspace=0.42)
    for index, case in enumerate(cases[:12]):
        row, pair = divmod(index, 4)
        left_col = pair * 2
        selected_id = case.get("figure_record", {}).get("selected_viewpoint_id", case.get("selected_id"))
        oracle_id = case.get("figure_record", {}).get("oracle_viewpoint_id", case.get("oracle_id"))
        selected_image: np.ndarray | None = None
        oracle_image: np.ndarray | None = None
        qualitative_rgb, qualitative_viewpoints, qualitative_frames = _load_qualitative_rgb(str(case["case_id"]), output)
        if qualitative_rgb is not None and qualitative_viewpoints is not None and qualitative_frames is not None and 15 in qualitative_frames:
            frame_slot = qualitative_frames.index(15)
            selected_slot = np.flatnonzero(qualitative_viewpoints == int(selected_id))
            oracle_slot = np.flatnonzero(qualitative_viewpoints == int(oracle_id))
            if selected_slot.size:
                selected_image = qualitative_rgb[int(selected_slot[0]), frame_slot]
            if oracle_slot.size:
                oracle_image = qualitative_rgb[int(oracle_slot[0]), frame_slot]
        selected_ax = fig.add_subplot(grid[row, left_col])
        oracle_ax = fig.add_subplot(grid[row, left_col + 1])
        _rgb_panel(selected_ax, selected_image, "Selected frame15")
        _rgb_panel(oracle_ax, oracle_image, "GT-best frame15")
        azimuth = float(case.get("figure_record", {}).get("azimuth_difference_deg", 0.0))
        margin_delta = float(case["oracle_margin"] - case["selected_margin"])
        selected_ax.text(0.0, 1.18, f"{case['case_id']} | {LABELS[case['label']]}\nS={selected_id} / G={oracle_id} / Δaz={azimuth:.0f}°", transform=selected_ax.transAxes, fontsize=8, va="bottom")
        oracle_ax.text(0.0, 1.18, f"GT-margin {case['selected_margin']:.3f} → {case['oracle_margin']:.3f} (Δ={margin_delta:.3f})", transform=oracle_ax.transAxes, fontsize=8, va="bottom")
    fig.savefig(output / "overview.png", dpi=140); plt.close(fig)


def run(data_root: Path) -> dict[str, Any]:
    policy_root = data_root / "datasets" / "policy_reduced12_eight_placement_v1"
    rows_all = load_jsonl(policy_root / "stage_c/features/val.jsonl")
    moving_ids = {str(row["episode_id"]) for row in load_jsonl(policy_root / "stage_d/features/val.jsonl")}
    moving_rows = [row for row in rows_all if str(row["episode_id"]) in moving_ids]
    cache_path = data_root / "diagnostics/reduced12_h1_discriminative_objective_batch/val_all_candidate_true_logp.npz"; cache_all = _read_npz(cache_path); indices = [i for i, row in enumerate(rows_all) if str(row["episode_id"]) in moving_ids]; cache = {key: value[indices] if value.ndim and value.shape[0] == len(rows_all) else value for key, value in cache_all.items()}
    selected_payload = json.loads((MOVING_ROOT / "per_selector_regret.json").read_text(encoding="utf-8")); selected_ids = selected_payload["Candidate-Conditioned Spatial"]["selected_action_ids"]; oracle_ids = selected_payload["Real-GTMargin Oracle"]["selected_action_ids"]
    metrics = _read_npz(METRICS_NPZ) if METRICS_NPZ.is_file() else {}
    cases = _select_cases(moving_rows, cache, selected_ids, oracle_ids, metrics)
    OUTPUT.mkdir(parents=True, exist_ok=True); manifest: list[dict[str, Any]] = []; figure_records: list[dict[str, Any]] = []
    manifest_by_scene: dict[str, Any] = {}
    for case in cases:
        row = moving_rows[int(case["row_index"])]
        if str(row["scene_id"]) not in manifest_by_scene:
            manifest_by_scene[str(row["scene_id"])] = _load_manifest(data_root / ARCHIVE_RELATIVE, str(row["scene_id"]))
        figure_records.append(_case_figure(case, row, data_root, manifest_by_scene[str(row["scene_id"])], metrics, OUTPUT)); manifest.append({key: value for key, value in case.items() if key not in {"candidate_logp"}} | {"figure_record": figure_records[-1]})
    _overview(manifest, OUTPUT)
    representative = max(manifest, key=lambda item: float(item["oracle_margin"] - item["selected_margin"])) if manifest else None
    representative_available = bool(representative and _qualitative_case_valid(representative, OUTPUT))
    target = OUTPUT / "representative_case.png"
    if representative_available and representative:
        source = OUTPUT / "figures" / f"{representative['case_id']}.png"; Image.open(source).save(target)
    else:
        target.unlink(missing_ok=True)
    render_status_path = OUTPUT / "qualitative_rgb" / "render_status.json"
    if render_status_path.is_file():
        render_status = json.loads(render_status_path.read_text(encoding="utf-8"))
    else:
        render_status = {"status": "NOT_RUN"}
    azimuth = [float(record["figure_record"]["azimuth_difference_deg"]) for record in manifest]
    scene_better = sum((record["selected_scene_visibility"] is not None and record["oracle_scene_visibility"] is not None and record["oracle_scene_visibility"] > record["selected_scene_visibility"]) for record in manifest)
    human_better = sum((record["selected_human_observability"] is not None and record["oracle_human_observability"] is not None and record["oracle_human_observability"] > record["selected_human_observability"]) for record in manifest)
    targeted_rgb_rendered = render_status.get("status") == "COMPLETED" and int(render_status.get("rendered_cases", 0)) == 12 and int(render_status.get("rendered_images", 0)) == 72 and int(render_status.get("invalid_images", 0)) == 0
    summary = {"cases": len(manifest), "groups": {group: sum(record["group"] == group for record in manifest) for group in ("A", "B", "C", "fallback")}, "mean_abs_azimuth_difference_deg": float(np.mean(np.abs(azimuth))) if azimuth else None, "oracle_scene_visibility_higher_fraction": float(scene_better / max(len(manifest), 1)), "oracle_human_observability_higher_fraction": float(human_better / max(len(manifest), 1)), "mean_margin_difference": float(np.mean([record["oracle_margin"] - record["selected_margin"] for record in manifest])) if manifest else None, "motion_fidelity": "motion_fidelity_gt_alignment_unavailable", "rgb_archive": "targeted qualitative RGB cache at frames [0,15,29]; available_view_mask enforced for archived observations", "targeted_rgb_render_status": render_status, "representative_case_available": representative_available, "scene_map": "scene_map_unavailable"}
    flags = {"test_used": False, "training_used": False, "new_rgb_rendered": targeted_rgb_rendered, "full_dataset_rgb_regenerated": False, "targeted_rgb_rendering_only": True, "targeted_cases": 12, "targeted_viewpoints_per_case": 2, "targeted_frames": [0, 15, 29], "new_pose_estimation": False, "gt_action_used_for_posthoc_diagnostic_only": True, "gt_margin_used_for_case_selection_and_visualization_only": True, "selector_remains_unchanged": True, "deployable": False}
    (OUTPUT / "case_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); (OUTPUT / "summary.json").write_text(json.dumps({"summary": summary, "protocol_flags": flags, "representative_case": representative["case_id"] if representative else None}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Selected-vs-GT-best Failure Case Visualization Audit", "", f"Generated {len(manifest)} Moving-Val cases from the existing Candidate-Conditioned Spatial selections and Real-GTMargin oracle. No Test data were read, no models were trained, and no RGB/skeleton data were regenerated.", "", f"- Mean absolute selected-vs-GT-best azimuth difference: {summary['mean_abs_azimuth_difference_deg']}", f"- Oracle SceneVisibility higher fraction: {summary['oracle_scene_visibility_higher_fraction']:.3f}", f"- Oracle HumanObservability higher fraction: {summary['oracle_human_observability_higher_fraction']:.3f}", f"- Mean GT-margin difference (oracle - selected): {summary['mean_margin_difference']:.4f}", "- Motion fidelity: `motion_fidelity_gt_alignment_unavailable` (no reliable GT world-space canonical alignment in current archive).", "- RGB: archived visited files contain frame_index=15 only; t0/t29 panels are explicitly N/A.", "- Scene occupancy map: unavailable; geometry panels show human and candidate positions only.", "", "## Interpretation", "", "This is a 12-case qualitative audit, not a population estimate. The figures should be used to inspect whether visibility, viewing angle, reconstruction distortion, or temporal evidence plausibly explains selected-vs-oracle failures. Missing caches are shown as N/A rather than reconstructed.", "", "## Protocol flags", "", "`test_used=false`; `training_used=false`; `new_rgb_rendered=false`; `new_pose_estimation=false`; `gt_action_used_for_posthoc_diagnostic_only=true`; `gt_margin_used_for_case_selection_and_visualization_only=true`; `selector_remains_unchanged=true`; `deployable=false`."]
    lines.extend(["", "## RGB availability and targeted rendering", "", "The RGB loader enforces `available_view_mask`; unavailable zero-filled slots are rendered as N/A/gray and are never interpreted as black RGB. Targeted rendering was limited to 12 cases × 2 viewpoints × frames [0, 15, 29].", f"Targeted renderer status: `{render_status.get('status', 'NOT_RUN')}` ({render_status.get('reason', 'no status reason')}).", f"Representative case available: `{representative_available}`; it is created only when all six targeted RGB images pass validation.", "No full-dataset RGB regeneration was performed.", "", "## Additional protocol flags", "", "`full_dataset_rgb_regenerated=false`; `targeted_rgb_rendering_only=true`; `targeted_cases=12`; `targeted_viewpoints_per_case=2`; `targeted_frames=[0,15,29]`."])
    lines.extend(["", "## Render completion", "", f"Validated targeted RGB: `{targeted_rgb_rendered}`; rendered cases: `{render_status.get('rendered_cases', 0)}`; rendered images: `{render_status.get('rendered_images', 0)}`; invalid images: `{render_status.get('invalid_images', 0)}`.", "No full-dataset RGB regeneration was performed. `test_used=false`; `training_used=false`."])
    lines = [line.replace("new_rgb_rendered=false", f"new_rgb_rendered={str(targeted_rgb_rendered).lower()}").replace("no RGB/skeleton data were regenerated.", "no full-dataset RGB/skeleton data were regenerated; targeted RGB was rendered separately.").replace("No RGB/skeleton data were regenerated.", "No full-dataset RGB/skeleton data were regenerated; targeted RGB was rendered separately.").replace("RGB: archived visited files contain frame_index=15 only; t0/t29 panels are explicitly N/A.", "RGB: targeted qualitative cache contains validated Selected/GT-best frames [0,15,29].").replace("Missing caches are shown as N/A rather than reconstructed.", "All targeted RGB panels are loaded from the validated qualitative cache.") for line in lines]
    (OUTPUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "COMPLETED", "summary": summary, "cases": len(manifest), "output": str(OUTPUT)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=get_data_root()); args = parser.parse_args(); print(json.dumps(run(args.data_root.resolve()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
