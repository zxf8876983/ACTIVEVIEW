"""
可视化与调试输出模块 —— visualization.py
=========================================

功能：
    保存 RGB/Depth 观测图、2D 姿态检测骨架叠加图、3D 估计骨架 vs GT 对比图及候选视角分布图。
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw


def save_rgb_image(rgb: np.ndarray, path: str):
    """保存 RGB 图像到指定路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = Image.fromarray(rgb.astype(np.uint8))
    img.save(path)


def save_depth_image(depth: np.ndarray, path: str, max_depth: float = 6.0):
    """保存伪彩色 / 灰度深度图。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if depth.ndim == 3:
        depth = depth[..., 0]
    clipped = np.clip(depth, 0.0, max_depth) / max_depth
    norm = (clipped * 255.0).astype(np.uint8)
    img = Image.fromarray(norm)
    img.save(path)


def save_pose_overlay_image(
    rgb: np.ndarray,
    keypoints_2d: dict,
    path: str,
    bbox_xyxy: Optional[tuple] = None,
):
    """在 RGB 图像上叠加绘制 2D 关键点骨架与检测框。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 骨架连线对定义 (EA-AVS 15)
    bones = [
        ("head", "neck"),
        ("neck", "left_shoulder"),
        ("neck", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("neck", "pelvis"),
        ("pelvis", "left_hip"),
        ("pelvis", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    # 绘制连接线
    for k1, k2 in bones:
        kp1 = keypoints_2d.get(k1)
        kp2 = keypoints_2d.get(k2)
        if kp1 and kp2:
            u1, v1 = (kp1.u, kp1.v) if hasattr(kp1, "u") else (kp1[0], kp1[1])
            u2, v2 = (kp2.u, kp2.v) if hasattr(kp2, "u") else (kp2[0], kp2[1])
            det1 = kp1.detected if hasattr(kp1, "detected") else True
            det2 = kp2.detected if hasattr(kp2, "detected") else True
            if det1 and det2:
                draw.line([(u1, v1), (u2, v2)], fill=(0, 255, 0), width=3)

    # 绘制关键点圆圈
    for name, kp in keypoints_2d.items():
        u, v = (kp.u, kp.v) if hasattr(kp, "u") else (kp[0], kp[1])
        det = kp.detected if hasattr(kp, "detected") else True
        if det:
            color = (255, 0, 0) if name in ("head", "neck", "pelvis") else (0, 128, 255)
            r = 4
            draw.ellipse([(u - r, v - r), (u + r, v + r)], fill=color, outline=(255, 255, 255))

    # 绘制 BBox
    if bbox_xyxy is not None:
        draw.rectangle(list(bbox_xyxy), outline=(255, 255, 0), width=2)

    img.save(path)


def save_candidate_debug_json(data: dict, path: str):
    """保存候选视角与决策调试 JSON 文件。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
