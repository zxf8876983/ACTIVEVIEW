"""
可视化与调试输出模块 —— visualization.py
=========================================

功能：
    保存渲染图像和候选点调试 JSON。
"""

import json
import os
import numpy as np
from PIL import Image


def save_rgb_image(rgb: np.ndarray, path: str):
    """保存 RGB 图像为 PNG。

    参数：
        rgb: shape=(H,W,3) 或 (H,W,4) 的 uint8 数组。
        path: 输出文件路径。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    img = Image.fromarray(rgb)
    img.save(path)


def save_candidate_debug_json(candidates: list, path: str):
    """保存候选点调试信息为 JSON。

    每个候选点包含：位置、朝向、距离、有效性、pred_score、true_score。

    参数：
        candidates: CandidateView 列表。
        path: 输出 JSON 文件路径。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _candidate_to_dict(c):
        d = {
            "candidate_id": c.candidate_id,
            "position": c.position.tolist() if hasattr(c.position, "tolist") else list(c.position),
            "yaw": float(c.yaw),
            "geodesic_distance": float(c.geodesic_distance),
            "euclidean_distance_to_human": float(c.euclidean_distance_to_human),
            "is_valid": c.is_valid,
            "invalid_reason": c.invalid_reason,
            "selected_by": c.selected_by,
        }
        if c.pred_score:
            d["pred_score"] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in c.pred_score.items()
                if not isinstance(v, list)
            }
        if c.true_score:
            d["true_score"] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in c.true_score.items()
                if not isinstance(v, list)
            }
        return d

    data = [_candidate_to_dict(c) for c in candidates]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
