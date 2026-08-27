#!/usr/bin/env python3
"""Visualize the pure-Habitat RGB -> 2D pose -> VideoPose3D pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.dataset.babel_clean_dataset_generator import (
    BabelCleanDatasetGenerator,
    _load_resampled_motion,
    transform_camera_sequence_to_gravity,
)
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition

LOGGER = logging.getLogger(__name__)

COCO17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)


def _load_one_record_per_label(manifest_path: Path) -> Dict[str, Mapping[str, object]]:
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected: Dict[str, Mapping[str, object]] = {}
    for record in sorted(records, key=lambda row: str(row["record_id"])):
        selected.setdefault(str(record["action_label"]), record)
    return selected


def _extract_2d_keypoints(estimator: object, rgb_frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract YOLO26n COCO-17 points for visualization."""
    if not hasattr(estimator, "pose_model"):
        raise TypeError("v11 visualization requires UltralyticsPose3DEstimator")
    results = estimator.pose_model.predict(
        source=list(rgb_frames),
        imgsz=estimator.inference_size,
        device=str(estimator.device),
        verbose=False,
        conf=0.15,
        max_det=10,
    )
    keypoints: List[np.ndarray] = []
    confidences: List[np.ndarray] = []
    previous_box = None
    for result in results:
        points, confidence, previous_box = estimator._select_person(result, previous_box)
        keypoints.append(points)
        confidences.append(confidence)
    return np.asarray(keypoints), np.asarray(confidences)


def _axis_limits(sequence: np.ndarray) -> Tuple[Tuple[float, float], ...]:
    plotted = sequence[:, :, [0, 2, 1]]
    low = plotted.min(axis=(0, 1))
    high = plotted.max(axis=(0, 1))
    center = (low + high) / 2.0
    span = max(float(np.max(high - low)), 1.6) * 1.12
    return tuple((float(c - span / 2.0), float(c + span / 2.0)) for c in center)


def _render_side_by_side(
    rgb_frames: Sequence[np.ndarray],
    keypoints_2d: np.ndarray,
    confidence_2d: np.ndarray,
    pose3d_gravity: np.ndarray,
    label: str,
    record_id: str,
    output_path: Path,
    edges_3d: Sequence[Tuple[int, int]],
    fps: int,
) -> None:
    if pose3d_gravity.ndim != 3 or pose3d_gravity.shape[1:] != (17, 3):
        raise ValueError(f"Unexpected gravity-aligned 3D shape: {pose3d_gravity.shape}")
    num_frames = int(pose3d_gravity.shape[0])
    # VideoPose3D is root-relative, so global camera translation is absent.
    # This constant display-only offset places the lowest estimated joint on
    # a visible ground plane without changing the ST-GCN input tensor.
    pose3d_gravity = np.asarray(pose3d_gravity, dtype=np.float32).copy()
    pose3d_gravity[..., 1] -= float(np.min(pose3d_gravity[..., 1]))
    height, width = rgb_frames[0].shape[:2]
    limits = _axis_limits(pose3d_gravity)
    fig = plt.figure(figsize=(13.4, 6.4), dpi=110)
    ax_rgb = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    image = ax_rgb.imshow(rgb_frames[0])
    ax_rgb.set_xlim(0, width)
    ax_rgb.set_ylim(height, 0)
    ax_rgb.axis("off")
    ax_rgb.set_title("Habitat humanoid RGB + detected 2D pose", fontsize=10, fontweight="bold")

    kp_lines = [
        ax_rgb.plot([], [], color="#00E5FF", linewidth=2.0)[0] for _ in COCO17_EDGES
    ]
    kp_points = ax_rgb.scatter([], [], color="#FFD700", s=18, edgecolors="black", linewidths=0.4)

    line_colors = ["#3498DB" if left in (4, 5, 6, 11, 12, 13) or right in (4, 5, 6, 11, 12, 13)
                   else "#2ECC71" if left in (1, 2, 3, 14, 15, 16) or right in (1, 2, 3, 14, 15, 16)
                   else "#E67E22" if left in (9, 10) or right in (9, 10)
                   else "#9B59B6" for left, right in edges_3d]
    lines_3d = [ax_3d.plot([], [], [], color=color, linewidth=2.4)[0]
                for color in line_colors]
    points_3d = ax_3d.scatter([], [], [], color="#1B4F72", s=28, depthshade=True)
    root_3d = ax_3d.scatter([], [], [], color="#E74C3C", marker="^", s=65, label="pelvis")
    ax_3d.set_xlim(*limits[0])
    ax_3d.set_ylim(*limits[1])
    ax_3d.set_zlim(*limits[2])
    ax_3d.set_box_aspect((1.0, 1.0, 1.0))
    ax_3d.set_xlabel("X", labelpad=4)
    ax_3d.set_ylabel("Z depth", labelpad=4)
    ax_3d.set_zlabel("Y gravity", labelpad=4)
    ax_3d.view_init(elev=16, azim=-62)
    ax_3d.legend(loc="upper right", fontsize=8)

    def update(frame_index: int):
        image.set_data(rgb_frames[frame_index])
        kpts = keypoints_2d[frame_index]
        conf = confidence_2d[frame_index]
        for line, (left, right) in zip(kp_lines, COCO17_EDGES):
            line.set_data(kpts[[left, right], 0], kpts[[left, right], 1])
        kp_points.set_offsets(kpts)

        frame = pose3d_gravity[frame_index]
        plotted = frame[:, [0, 2, 1]]
        for line, (left, right) in zip(lines_3d, edges_3d):
            line.set_data(plotted[[left, right], 0], plotted[[left, right], 1])
            line.set_3d_properties(plotted[[left, right], 2])
        points_3d._offsets3d = (plotted[:, 0], plotted[:, 1], plotted[:, 2])
        root_3d._offsets3d = ([plotted[0, 0]], [plotted[0, 1]], [plotted[0, 2]])
        mean_conf = float(conf.mean())
        fig.suptitle(
            f"ACTIVEVIEW v11.5 | {label} | {record_id} | frame {frame_index + 1}/{num_frames} | "
            f"2D mean confidence {mean_conf:.3f} | gravity-aligned display (Y up)",
            fontsize=11,
            fontweight="bold",
        )
        return [image, *kp_lines, kp_points, *lines_3d, points_3d, root_3d]

    animation = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2200, extra_args=["-pix_fmt", "yuv420p"])
    animation.save(str(output_path), writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--camera-height", type=float, default=1.2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--pose-backend",
        choices=("ultralytics_yolo26n",),
        default="ultralytics_yolo26n",
    )
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional subset of action labels for a short diagnostic smoke test.",
    )
    args = parser.parse_args()

    data_root = get_data_root()
    yolo_weights = args.yolo_weights or data_root / "checkpoints/ultralytics/yolo26n-pose.pt"
    dataset_root = data_root / "datasets" / "stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed"
    manifest = args.manifest or dataset_root / "train.json"
    output_dir = args.output_dir or data_root / "visualizations" / "household_rgb_to_3d_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_one_record_per_label(manifest)
    if args.labels:
        missing = sorted(set(args.labels) - set(selected))
        if missing:
            raise ValueError(f"Labels not present in manifest: {missing}")
        selected = {label: selected[label] for label in args.labels}
    generator = BabelCleanDatasetGenerator(
        output_root=output_dir / "cache",
        image_size=args.image_size,
        target_frames=args.target_frames,
        camera_height=args.camera_height,
        device=args.device,
        pose_backend=args.pose_backend,
        yolo_weights=yolo_weights,
    )
    skeleton_def = get_skeleton_definition(backend="h36m_17")
    diagnostic_manifest: Dict[str, object] = {}
    sim, human = generator._make_sim()
    try:
        for label, record in sorted(selected.items()):
            normalized_motion = _load_resampled_motion(record, args.target_frames)
            converted = generator.converter.convert(normalized_motion)
            joints = np.asarray(converted["pose_motion"]["joints_array"], dtype=np.float32)
            roots = np.asarray(converted["pose_motion"]["transform_array"], dtype=np.float32)
            rgb_frames, camera_to_world = generator._render_rgb(sim, human, joints, roots, record)
            pose3d_camera, confidence3d = generator.estimator.estimate_sequence(rgb_frames)
            pose3d = transform_camera_sequence_to_gravity(pose3d_camera, camera_to_world)
            keypoints_2d, confidence2d = _extract_2d_keypoints(generator.estimator, rgb_frames)
            if pose3d.shape != (args.target_frames, 17, 3):
                raise ValueError(f"Unexpected 3D sequence shape for {label}: {pose3d.shape}")
            record_id = str(record["record_id"])
            output_path = output_dir / f"{label.replace('/', '_').replace(' ', '_')}_{record_id}.mp4"
            _render_side_by_side(
                rgb_frames,
                keypoints_2d,
                confidence2d,
                pose3d,
                label,
                record_id,
                output_path,
                skeleton_def.edges,
                args.fps,
            )
            diagnostic_manifest[label] = {
                "record_id": record_id,
                "source_path": record["source_path"],
                "action_label": label,
                "video_path": str(output_path),
                "rgb_frames": args.target_frames,
                "rgb_resolution": [args.image_size, args.image_size],
                "camera_height_m": float(args.camera_height),
                "diagnostic_floor_y_m": 0.0,
                "human_grounding": "geometry-aware_min_visual_geometry_to_floor",
                "pose3d_coordinate_system": "VideoPose3D_H36M_camera_to_Habitat_gravity; display_grounded_only",
                "mean_2d_confidence": float(confidence2d.mean()),
                "mean_3d_confidence": float(confidence3d.mean()),
                "gt_skeleton_used_for_video": False,
            }
            LOGGER.info("Rendered RGB-to-3D diagnostic for %s -> %s", label, output_path)
    finally:
        sim.close()

    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(diagnostic_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Rendered %d RGB-to-3D diagnostics", len(diagnostic_manifest))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
