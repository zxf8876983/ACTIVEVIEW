"""Shared Habitat articulated-humanoid to H36M-17 forward-kinematics helper."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from activeview.data.motion.babel_clean_dataset_generator import (
    apply_humanoid_pose,
    precompute_grounding_offsets,
)


# This ordering is the canonical ordering used by the scene-visibility
# diagnostics and by the reduced12 recognizer preprocessing.
H36M17_LINKS = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine1", "spine3", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)


def extract_h36m17_from_humanoid(
    human: Any,
    converted_motion: Mapping[str, Any],
    placement: Mapping[str, Any],
    yaw_deg: float,
    frame_ids: Sequence[int],
    *,
    floor_y: Optional[float] = None,
) -> np.ndarray:
    """Extract world-space H36M-17 link positions for arbitrary motion frames.

    ``converted_motion`` is the exact output of :class:`MotionConverter`.
    Grounding offsets are precomputed once for the complete 30-frame motion,
    while ``frame_ids`` controls the returned subset.  Pelvis follows the
    existing renderer convention and is the articulated object's world
    translation; all other joints are read from link scene nodes.
    """
    pose_motion = converted_motion.get("pose_motion")
    if not isinstance(pose_motion, Mapping):
        raise ValueError("converted_motion must contain a pose_motion mapping")
    joints = np.asarray(pose_motion.get("joints_array"), dtype=np.float32)
    roots = np.asarray(pose_motion.get("transform_array"), dtype=np.float32)
    if joints.ndim != 2 or roots.ndim != 3 or roots.shape[0] != joints.shape[0]:
        raise ValueError("converted motion arrays have incompatible shapes")
    if roots.shape[1:] != (4, 4) or not np.isfinite(joints).all() or not np.isfinite(roots).all():
        raise ValueError("converted motion arrays must be finite with transform shape (N,4,4)")
    ids = [int(value) for value in frame_ids]
    if not ids or any(value < 0 or value >= joints.shape[0] for value in ids):
        raise ValueError("frame_ids must be non-empty and within converted motion length")
    base = np.asarray(placement.get("position"), dtype=np.float32)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("placement position must be a finite 3-vector")
    ground = float(base[1]) if floor_y is None else float(floor_y)
    offsets, _ = precompute_grounding_offsets(
        human, joints, roots, scene_yaw_deg=float(yaw_deg)
    )
    names = {str(human.get_link_name(index)): index for index in range(int(human.num_links))}
    required = [name for name in H36M17_LINKS if name != "pelvis"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"humanoid is missing H36M17 links: {missing}")
    output = np.empty((len(ids), len(H36M17_LINKS), 3), dtype=np.float32)
    for out_index, frame_id in enumerate(ids):
        apply_humanoid_pose(
            human,
            joints[frame_id],
            roots[frame_id],
            base_position=base,
            scene_yaw_deg=float(yaw_deg),
            floor_y=ground,
            grounding_offset=float(offsets[frame_id]),
        )
        values = [np.asarray(human.translation, dtype=np.float32)]
        for name in H36M17_LINKS[1:]:
            node = human.get_link_scene_node(names[name])
            values.append(np.asarray(node.absolute_translation, dtype=np.float32))
        output[out_index] = np.asarray(values, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("extracted H36M-17 joints are not finite")
    return output

