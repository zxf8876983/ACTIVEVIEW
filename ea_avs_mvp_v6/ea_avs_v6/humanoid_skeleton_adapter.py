"""
Humanoid 骨架适配器 —— humanoid_skeleton_adapter.py
=====================================================

功能：
    将 Humanoid 实际关节/链接状态转换为 evaluator 能使用的关键点字典。
"""

from typing import Dict, Optional
import numpy as np

LINK_TO_KEYPOINT = {
    "head": "head",
    "neck": "neck",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
}

ALL_KEYPOINTS = (
    "head", "neck", "pelvis",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


def _link_world_position(humanoid_manager, link_name: str):
    ao = humanoid_manager.sim_obj
    try:
        link_id = ao.get_link_id_from_name(link_name)
    except Exception:
        return None
    if link_id is None or int(link_id) < 0:
        return None
    try:
        node = ao.get_link_scene_node(link_id)
    except Exception:
        return None
    pos = np.array(node.transformation.translation, dtype=np.float64).copy()
    return pos


def transform_legacy_local_skeleton_to_world(
    local_skeleton: Dict[str, np.ndarray],
    human_base_pos: np.ndarray,
    human_yaw: float,
) -> Dict[str, np.ndarray]:
    cos_y, sin_y = np.cos(human_yaw), np.sin(human_yaw)
    base = np.asarray(human_base_pos, dtype=np.float64)
    world = {}
    for kp, local in local_skeleton.items():
        l = np.asarray(local, dtype=np.float64)
        x = l[0] * cos_y + l[2] * sin_y
        y = l[1]
        z = -l[0] * sin_y + l[2] * cos_y
        world[kp] = base + np.array([x, y, z])
    return world


def get_humanoid_gt_skeleton(
    humanoid_manager,
    fallback_skeleton: Optional[Dict[str, np.ndarray]] = None,
    human_base_pos: Optional[np.ndarray] = None,
    human_yaw: float = 0.0,
    strict: bool = True,
) -> dict:
    skeleton: Dict[str, np.ndarray] = {}
    per_source: Dict[str, str] = {}
    keypoint_meta: Dict[str, dict] = {}

    link_meta = humanoid_manager.get_humanoid_link_metadata()

    def _object_id_for(link_name):
        m = link_meta.get(link_name)
        return m["object_id"] if m else None

    for link_name, kp in LINK_TO_KEYPOINT.items():
        if kp == "pelvis":
            continue
        pos = _link_world_position(humanoid_manager, link_name)
        if pos is not None:
            skeleton[kp] = pos
            per_source[kp] = "link"
            m = link_meta.get(link_name, {})
            keypoint_meta[kp] = {
                "source": "link",
                "link_name": link_name,
                "link_id": m.get("link_id"),
                "link_object_id": m.get("object_id"),
                "target_link_object_ids": [m["object_id"]] if m.get("object_id") is not None else [],
            }

    if "left_hip" in skeleton and "right_hip" in skeleton:
        skeleton["pelvis"] = 0.5 * (skeleton["left_hip"] + skeleton["right_hip"])
        per_source["pelvis"] = "link_derived"
        hip_oids = [oid for oid in (
            _object_id_for("left_hip"), _object_id_for("right_hip")) if oid is not None]
        pelvis_oids = hip_oids
        for extra in ("pelvis", "spine1", "root"):
            oid = _object_id_for(extra)
            if oid is not None and oid not in pelvis_oids:
                pelvis_oids.append(oid)
        keypoint_meta["pelvis"] = {
            "source": "link_derived",
            "link_name": None,
            "link_id": None,
            "link_object_id": None,
            "target_link_object_ids": pelvis_oids,
        }

    direct_link_count = sum(1 for s in per_source.values() if s == "link")
    link_derived_count = sum(1 for s in per_source.values() if s == "link_derived")

    if fallback_skeleton is not None and (not strict or direct_link_count + link_derived_count < 15):
        legacy_world = fallback_skeleton
        if human_base_pos is not None:
            legacy_world = transform_legacy_local_skeleton_to_world(
                fallback_skeleton, human_base_pos, human_yaw)
        for kp in ALL_KEYPOINTS:
            if kp not in skeleton and kp in legacy_world:
                skeleton[kp] = np.asarray(legacy_world[kp], dtype=np.float64)
                per_source[kp] = "legacy"
                keypoint_meta[kp] = {
                    "source": "legacy",
                    "link_name": None,
                    "link_id": None,
                    "link_object_id": None,
                    "target_link_object_ids": [],
                }

    total = len(skeleton)
    fallback_count = sum(1 for s in per_source.values() if s == "legacy")
    if total == 0:
        source = "legacy_fallback"
    elif fallback_count == 0 and (direct_link_count + link_derived_count) == total:
        source = "habitat_humanoid_gt"
    else:
        source = "mixed"

    result = {
        "skeleton": skeleton,
        "keypoint_meta": keypoint_meta,
        "source": source,
        "per_keypoint_source": per_source,
        "keypoint_count": total,
        "direct_link_count": direct_link_count,
        "link_derived_count": link_derived_count,
        "fallback_count": fallback_count,
        "missing_keypoints": [k for k in ALL_KEYPOINTS if k not in skeleton],
    }

    if strict:
        from .humanoid_validation import validate_gt_skeleton
        validate_gt_skeleton(result, strict=True)

    return result
