"""
Humanoid 验证模块 —— humanoid_validation.py
=============================================

功能：
    统一 Humanoid 渲染与骨架验证逻辑，供 smoke test / debug RGB-D / 主实验复用。
"""

from typing import Dict, List, Optional
import numpy as np


def compute_humanoid_semantic_stats(
    semantic: Optional[np.ndarray],
    humanoid_semantic_ids: List[int],
    semantic_assignment_ok: bool = False,
) -> dict:
    sensor_available = semantic is not None
    if not sensor_available or len(humanoid_semantic_ids) == 0:
        return {
            "semantic_sensor_available": sensor_available,
            "semantic_assignment_ok": bool(semantic_assignment_ok),
            "semantic_available": False,
            "humanoid_semantic_visible": False,
            "humanoid_pixel_count": 0, "humanoid_pixel_ratio": 0.0,
            "bbox": None,
        }
    sem = np.asarray(semantic)
    if sem.ndim == 3:
        sem = sem[..., 0]
    mask = np.isin(sem, humanoid_semantic_ids)
    count = int(mask.sum())
    height, width = sem.shape
    ratio = count / float(height * width) if height * width > 0 else 0.0
    bbox = None
    if count > 0:
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "semantic_sensor_available": True,
        "semantic_assignment_ok": bool(semantic_assignment_ok),
        "semantic_available": bool(semantic_assignment_ok),
        "humanoid_semantic_visible": count > 0,
        "humanoid_pixel_count": count,
        "humanoid_pixel_ratio": ratio,
        "bbox": bbox,
    }


def compute_humanoid_depth_stats(
    depth: Optional[np.ndarray],
    humanoid_mask: Optional[np.ndarray],
) -> dict:
    if depth is None:
        return {"depth_available": False, "humanoid_pixel_count": 0,
                "humanoid_depth_valid_pixel_count": 0,
                "humanoid_depth_valid_ratio": 0.0,
                "humanoid_depth_median": None,
                "humanoid_depth_min": None, "humanoid_depth_max": None}
    dep = np.asarray(depth)
    if dep.ndim == 3:
        dep = dep[..., 0]

    if humanoid_mask is not None:
        mask = np.asarray(humanoid_mask, dtype=bool)
        count = int(mask.sum())
        if count == 0:
            return {"depth_available": True, "humanoid_pixel_count": 0,
                    "humanoid_depth_valid_pixel_count": 0,
                    "humanoid_depth_valid_ratio": 0.0,
                    "humanoid_depth_median": None,
                    "humanoid_depth_min": None, "humanoid_depth_max": None}
        region = dep[mask]
    else:
        region = dep
        count = int((dep > 0).sum())

    valid = region[region > 0]
    valid_count = int(valid.size)
    ratio = valid_count / float(count) if count > 0 else 0.0
    return {
        "depth_available": True,
        "humanoid_pixel_count": count,
        "humanoid_depth_valid_pixel_count": valid_count,
        "humanoid_depth_valid_ratio": ratio,
        "humanoid_depth_median": float(np.median(valid)) if valid.size else None,
        "humanoid_depth_min": float(valid.min()) if valid.size else None,
        "humanoid_depth_max": float(valid.max()) if valid.size else None,
    }


def compute_gt_anchored_depth_proxy(
    depth: Optional[np.ndarray],
    width: int,
    height: int,
    hfov_deg: float,
    view_pos: np.ndarray,
    view_yaw: float,
    camera_height: float,
    human_keypoints: Dict[str, np.ndarray],
    pad: int = 10,
    depth_match_tolerance: float = 0.5,
) -> Optional[dict]:
    if depth is None or not human_keypoints:
        return None
    dep = np.asarray(depth)
    if dep.ndim == 3:
        dep = dep[..., 0]
    h, w = dep.shape

    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    cam_pos = np.array(view_pos, dtype=np.float64) + np.array(
        [0.0, camera_height, 0.0])
    th = view_yaw
    right = np.array([-np.cos(th), 0.0, np.sin(th)])
    forward = np.array([np.sin(th), 0.0, np.cos(th)])

    candidate_region = np.zeros((h, w), dtype=bool)
    per_kp_z = []

    for kp in human_keypoints.values():
        v = np.array(kp, dtype=np.float64) - cam_pos
        z_cam = float(v[0] * forward[0] + v[2] * forward[2])
        if z_cam <= 0.05:
            continue
        x_cam = float(v[0] * right[0] + v[2] * right[2])
        y_cam = float(v[1])
        u = fx * x_cam / z_cam + width / 2.0
        v_px = height / 2.0 - fx * y_cam / z_cam
        ui, vi = int(round(u)), int(round(v_px))
        lo_x, hi_x = max(0, ui - pad), min(w - 1, ui + pad)
        lo_y, hi_y = max(0, vi - pad), min(h - 1, vi + pad)
        if lo_x > hi_x or lo_y > hi_y:
            continue
        candidate_region[lo_y:hi_y + 1, lo_x:hi_x + 1] = True
        per_kp_z.append((lo_y, hi_y, lo_x, hi_x, z_cam))

    depth_match = np.zeros((h, w), dtype=bool)
    for (lo_y, hi_y, lo_x, hi_x, z_cam) in per_kp_z:
        region = dep[lo_y:hi_y + 1, lo_x:hi_x + 1]
        match = (region > 0) & (np.abs(region - z_cam) < depth_match_tolerance)
        depth_match[lo_y:hi_y + 1, lo_x:hi_x + 1] |= match

    candidate_count = int(candidate_region.sum())
    depth_match_count = int(depth_match.sum())
    ratio = (depth_match_count / candidate_count
             if candidate_count > 0 else 0.0)
    return {
        "candidate_region_mask": candidate_region,
        "depth_match_mask": depth_match,
        "candidate_region_pixel_count": candidate_count,
        "depth_match_pixel_count": depth_match_count,
        "depth_match_ratio": ratio,
    }


def validate_humanoid_render(
    rgb_ok: bool,
    depth_ok: bool,
    humanoid_mask: Optional[np.ndarray],
    min_pixel_count: int,
    min_depth_valid_ratio: float,
    depth: Optional[np.ndarray] = None,
) -> dict:
    if humanoid_mask is not None:
        pixel_count = int(np.asarray(humanoid_mask, dtype=bool).sum())
        dstats = compute_humanoid_depth_stats(depth, humanoid_mask) \
            if depth is not None else {}
        depth_valid_ratio = dstats.get("humanoid_depth_valid_ratio", 0.0)
    else:
        pixel_count = 0
        depth_valid_ratio = 0.0

    visible = bool(pixel_count >= min_pixel_count)
    render_success = bool(
        rgb_ok and depth_ok and visible
        and depth_valid_ratio >= min_depth_valid_ratio
    )
    return {
        "rgb_render_success": bool(rgb_ok),
        "depth_render_success": bool(depth_ok),
        "humanoid_visible": visible,
        "humanoid_pixel_count": pixel_count,
        "humanoid_depth_valid_ratio": depth_valid_ratio,
        "humanoid_render_success": render_success,
    }


def compute_humanoid_render_stats(
    obs,
    config: dict,
    view_pos,
    view_yaw,
    human_skeleton,
    humanoid_semantic_ids,
    semantic_assignment_ok: bool = False,
    semantic_assignment_count: int = 0,
    camera_cfg=None,
) -> dict:
    camera_cfg = camera_cfg or config.get("camera", {})
    val_cfg = config.get("humanoid_validation", {})
    min_pixels = val_cfg.get("min_pixel_count", 100)
    min_depth_ratio = val_cfg.get("min_depth_valid_ratio", 0.5)
    depth_tol = val_cfg.get("depth_match_tolerance", 0.5)

    rgb_ok = obs is not None and obs.get("rgb") is not None
    depth_ok = obs is not None and obs.get("depth") is not None
    semantic = obs.get("semantic") if obs is not None else None

    out = {
        "humanoid_validation_source": "unavailable",
        "humanoid_render_success": False,
        "semantic_sensor_available": semantic is not None,
        "semantic_assignment_ok": bool(semantic_assignment_ok),
        "semantic_assignment_count": int(semantic_assignment_count),
        "humanoid_depth_valid_ratio": 0.0,
        "humanoid_semantic_visible": False,
        "humanoid_semantic_pixel_count": 0,
        "humanoid_semantic_pixel_ratio": 0.0,
        "humanoid_semantic_bbox": None,
        "humanoid_proxy_visible": False,
        "humanoid_proxy_pixel_count": 0,
        "humanoid_proxy_candidate_region_count": 0,
        "humanoid_proxy_match_ratio": 0.0,
        "humanoid_proxy_bbox": None,
    }

    sem_stats = compute_humanoid_semantic_stats(
        semantic, humanoid_semantic_ids, semantic_assignment_ok)
    if sem_stats["semantic_available"]:
        sc = sem_stats["humanoid_pixel_count"]
        sbbox = sem_stats["bbox"]
        sem_mask = None
        if semantic is not None and humanoid_semantic_ids:
            sem_arr = np.asarray(semantic)
            if sem_arr.ndim == 3:
                sem_arr = sem_arr[..., 0]
            sem_mask = np.isin(sem_arr, list(humanoid_semantic_ids))
        sem_depth_stats = compute_humanoid_depth_stats(
            obs["depth"] if depth_ok else None, sem_mask)
        depth_valid_ratio = sem_depth_stats.get("humanoid_depth_valid_ratio", 0.0)
        out.update({
            "humanoid_validation_source": "semantic",
            "humanoid_semantic_visible": sem_stats["humanoid_semantic_visible"],
            "humanoid_semantic_pixel_count": sc,
            "humanoid_semantic_pixel_ratio": sem_stats["humanoid_pixel_ratio"],
            "humanoid_semantic_bbox": sbbox,
            "humanoid_depth_valid_ratio": depth_valid_ratio,
        })
        out["humanoid_render_success"] = bool(
            rgb_ok and depth_ok and sc >= min_pixels
            and depth_valid_ratio >= min_depth_ratio)
        return out

    if depth_ok and human_skeleton:
        proxy = compute_gt_anchored_depth_proxy(
            obs["depth"], camera_cfg["width"], camera_cfg["height"],
            camera_cfg["hfov_deg"], view_pos, view_yaw,
            camera_cfg["camera_height"], human_skeleton,
            pad=8, depth_match_tolerance=depth_tol)
        if proxy is not None:
            match_cnt = proxy["depth_match_pixel_count"]
            cand_cnt = proxy["candidate_region_pixel_count"]
            match_ratio = proxy["depth_match_ratio"]
            bbox = None
            dm = proxy["depth_match_mask"]
            if match_cnt > 0:
                ys, xs = np.where(dm)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            out.update({
                "humanoid_validation_source": "gt_depth_proxy",
                "humanoid_proxy_visible": bool(match_cnt >= min_pixels),
                "humanoid_proxy_pixel_count": match_cnt,
                "humanoid_proxy_candidate_region_count": cand_cnt,
                "humanoid_proxy_match_ratio": match_ratio,
                "humanoid_proxy_bbox": bbox,
            })
            out["humanoid_render_success"] = bool(
                rgb_ok and depth_ok and match_cnt >= min_pixels
                and match_ratio >= min_depth_ratio)
        return out

    return out


def validate_gt_skeleton(gt_result: dict, strict: bool = True) -> dict:
    source = gt_result.get("source", "")
    skeleton = gt_result.get("skeleton", {})
    per_source = gt_result.get("per_keypoint_source", {})
    keypoint_count = len(skeleton)
    direct_link_count = gt_result.get("direct_link_count",
        sum(1 for s in per_source.values() if s == "link"))
    link_derived_count = gt_result.get("link_derived_count",
        sum(1 for s in per_source.values() if s == "link_derived"))
    fallback_count = sum(1 for s in per_source.values() if s == "legacy")
    missing = [
        k for k in (
            "head", "neck", "pelvis",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        )
        if k not in skeleton
    ]

    derived_total = direct_link_count + link_derived_count
    valid = (source == "habitat_humanoid_gt"
             and derived_total == 15 and fallback_count == 0 and not missing)
    reason = None
    if source != "habitat_humanoid_gt":
        reason = f"gt_source={source}"
    elif derived_total != 15:
        reason = f"derived_total={derived_total}/15 (direct={direct_link_count}, derived={link_derived_count})"
    elif fallback_count > 0:
        reason = f"fallback_count={fallback_count}"
    elif missing:
        reason = f"missing={missing}"

    if strict and not valid:
        raise RuntimeError(f"invalid_humanoid_gt_skeleton: {reason}（strict_gt_skeleton=True）")

    return {
        "valid": valid, "reason": reason,
        "source": source, "keypoint_count": keypoint_count,
        "direct_link_count": direct_link_count,
        "link_derived_count": link_derived_count,
        "fallback_count": fallback_count,
        "missing_keypoints": missing,
    }
