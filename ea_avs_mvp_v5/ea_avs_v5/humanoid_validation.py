"""
Humanoid 验证模块 —— humanoid_validation.py
=============================================

功能：
    统一 Humanoid 渲染与骨架验证逻辑，供 smoke test / debug RGB-D / 主实验复用。

背景（已验证的 Habitat 实际行为）：
    1. 当前 apartment/skokloster/van-gogh 场景均无 semantic 标注，且 Humanoid
       articulated object（URDF/GLB）不携带 semantic class，因此 semantic sensor
       输出恒为 0，无法用 semantic 区分 Humanoid 像素。
    2. 因此本模块默认采用 **GT 锚定 depth 验证**作为 Humanoid 可见性的可靠
       fallback：结合已知 Humanoid GT 骨架与渲染深度，判断人体表面是否存在。
    3. 保留 compute_humanoid_semantic_stats 供将来有 semantic 数据的场景使用，
       绝不将其接入 NBV 在线策略输入。

本模块约束：
    semantic / humanoid mask / pixel count 只能用于验收和 debug，
    绝不能进入 PredictiveEvaluator / OursPolicy / Q_pred。
"""

from typing import Dict, List, Optional

import numpy as np


def compute_humanoid_semantic_stats(
    semantic: Optional[np.ndarray],
    humanoid_semantic_ids: List[int],
) -> dict:
    """在 semantic 图像可用时统计 Humanoid 像素（调试用）。

    参数：
        semantic: (H,W) uint32 语义图；None 或全 0 时无法判断。
        humanoid_semantic_ids: Humanoid 相关 semantic id 集合（不假设单值）。

    返回：
        {
            "semantic_available": bool,
            "humanoid_visible": bool,
            "humanoid_pixel_count": int,
            "humanoid_pixel_ratio": float,
            "bbox": [x1,y1,x2,y2] | None,
        }
    """
    if semantic is None or len(humanoid_semantic_ids) == 0:
        return {
            "semantic_available": False, "humanoid_visible": False,
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
        "semantic_available": count > 0,
        "humanoid_visible": count > 0,
        "humanoid_pixel_count": count,
        "humanoid_pixel_ratio": ratio,
        "bbox": bbox,
    }


def compute_humanoid_depth_stats(
    depth: Optional[np.ndarray],
    humanoid_mask: Optional[np.ndarray],
) -> dict:
    """统计 Humanoid 区域的 depth 有效性。

    参数：
        depth: (H,W) float32 公制深度。
        humanoid_mask: 布尔掩码（semantic 或 GT 锚定）；None 时用深度非 0 全图。

    返回：
        {
            "depth_available": bool,
            "humanoid_pixel_count": int,
            "humanoid_depth_valid_pixel_count": int,
            "humanoid_depth_valid_ratio": float,  # valid / humanoid_pixel_count
            "humanoid_depth_median": float|None,
            "humanoid_depth_min": float|None,
            "humanoid_depth_max": float|None,
        }
    """
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
        # 无 mask 时退化为统计全图有效深度（仅用于辅助）
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


def compute_gt_anchored_humanoid_mask(
    depth: Optional[np.ndarray],
    width: int,
    height: int,
    hfov_deg: float,
    view_pos: np.ndarray,
    view_yaw: float,
    camera_height: float,
    human_keypoints: Dict[str, np.ndarray],
    pad: int = 10,
) -> Optional[np.ndarray]:
    """基于 GT 骨架投影的 Humanoid 区域掩码（semantic 不可用时的 fallback）。

    将 Humanoid GT 关键点投影到相机图像，在关键点周围 pad 邻域标记为
    Humanoid 区域，再结合 depth：区域内存在有效 metric depth ≈ 关键点深度
    的像素才视为人体表面（区分环境背景）。

    参数：
        depth: (H,W) 渲染深度。
        width/height/hfov_deg: 相机参数。
        view_pos: 机器人基座位置。
        view_yaw: 相机朝向（模型约定 +Z）。
        camera_height: 相机高度。
        human_keypoints: Humanoid GT 世界坐标关键点。
        pad: 投影点周围像素宽带的半径。

    返回：
        (H,W) bool 掩码；投影退化或 depth 不可用时返回 None。
    """
    if depth is None or not human_keypoints:
        return None
    dep = np.asarray(depth)
    if dep.ndim == 3:
        dep = dep[..., 0]
    h, w = dep.shape

    # 复用与 true_evaluator._depth_occlusion 相同的投影约定
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    cam_pos = np.array(view_pos, dtype=np.float64) + np.array(
        [0.0, camera_height, 0.0])
    th = view_yaw
    right = np.array([-np.cos(th), 0.0, np.sin(th)])
    forward = np.array([np.sin(th), 0.0, np.cos(th)])

    mask = np.zeros((h, w), dtype=bool)
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
        # 区域内存在有效深度且落在关键点 z_cam 一个容差范围内（人体表面）
        region = dep[lo_y:hi_y + 1, lo_x:hi_x + 1]
        mask[lo_y:hi_y + 1, lo_x:hi_x + 1] |= (region > 0) & (
            np.abs(region - z_cam) < 0.5)
    return mask


def validate_humanoid_render(
    rgb_ok: bool,
    depth_ok: bool,
    humanoid_mask: Optional[np.ndarray],
    min_pixel_count: int,
    min_depth_valid_ratio: float,
    depth: Optional[np.ndarray] = None,
) -> dict:
    """计算 humanoid_render_success（真实测量，不硬编码）。

    定义：
        humanoid_render_success = rgb_ok AND depth_ok AND humanoid_visible
                                 AND pixel_count >= min_pixel_count
                                 AND depth_valid_ratio >= min_depth_valid_ratio

    参数：
        rgb_ok: 是否成功渲染到 RGB。
        depth_ok: 是否成功渲染到 depth。
        humanoid_mask: Humanoid 像素掩码（semantic 或 GT 锚定）。
        min_pixel_count / min_depth_valid_ratio: 阈值。
        depth: 可选，用于计算 depth valid ratio。

    返回：
        {
            "rgb_render_success": bool,
            "depth_render_success": bool,
            "humanoid_visible": bool,
            "humanoid_pixel_count": int,
            "humanoid_depth_valid_ratio": float,
            "humanoid_render_success": bool,
        }
    """
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


def validate_gt_skeleton(gt_result: dict, strict: bool = True) -> dict:
    """校验 GT skeleton 完整性。

    参数：
        gt_result: get_humanoid_gt_skeleton 返回的字典。
        strict: 严格模式。True 时若来源非 habitat_humanoid_gt 或 link 数 <15
            则抛 RuntimeError；False（debug 模式）返回校验结果不抛错。

    返回：
        {
            "valid": bool,
            "reason": str|None,
            "source": str,
            "keypoint_count": int,
            "link_count": int,
            "fallback_count": int,
            "missing_keypoints": list,
        }

    抛出异常：
        RuntimeError: strict=True 且不满足 15/15 Humanoid links。
    """
    source = gt_result.get("source", "")
    skeleton = gt_result.get("skeleton", {})
    per_source = gt_result.get("per_keypoint_source", {})
    keypoint_count = len(skeleton)
    link_count = sum(1 for s in per_source.values() if s != "legacy")
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

    valid = (source == "habitat_humanoid_gt"
             and link_count == 15 and fallback_count == 0 and not missing)
    reason = None
    if source != "habitat_humanoid_gt":
        reason = f"gt_source={source}"
    elif link_count != 15:
        reason = f"link_count={link_count}/15"
    elif fallback_count > 0:
        reason = f"fallback_count={fallback_count}"
    elif missing:
        reason = f"missing={missing}"

    if strict and not valid:
        raise RuntimeError(
            f"invalid_humanoid_gt_skeleton: {reason}（strict_gt_skeleton=True）")

    return {
        "valid": valid, "reason": reason,
        "source": source, "keypoint_count": keypoint_count,
        "link_count": link_count, "fallback_count": fallback_count,
        "missing_keypoints": missing,
    }