"""
可视化与调试输出模块 —— visualization.py
=========================================

功能：
    保存渲染图像和候选点调试 JSON。
    从 v3.0 迁移，v4.0 的 debug JSON 新增遮挡感知字段。

v4.0 更新：
    - debug JSON 的 pred_score / true_score 内含遮挡指标：
      S_action_occ_pred, occlusion_rate_pred, occluded_keypoints_pred,
      visible_keypoints_occ_pred 等
    - Oracle 的 true_score 单独保存，并标注为 offline evaluation
"""

import json
import math
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
    Image.fromarray(rgb).save(path)


def _score_to_serializable(score: dict) -> dict:
    """将得分字典转换为可 JSON 序列化的字典。

    - numpy 标量转 Python 标量
    - 非有限浮点（inf/nan）转为 None，保证输出为标准 JSON
    - 列表保持为列表（如 visible_keypoints_occ_pred）
    - 嵌套字典（occlusion_result）递归转换
    """
    if not isinstance(score, dict):
        return {}
    out = {}
    for k, v in score.items():
        if isinstance(v, dict):
            out[k] = _score_to_serializable(v)
        elif isinstance(v, (np.floating, float)):
            val = float(v)
            out[k] = val if math.isfinite(val) else None
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif isinstance(v, list):
            items = []
            for item in v:
                if isinstance(item, (np.floating, float)):
                    fitem = float(item)
                    items.append(fitem if math.isfinite(fitem) else None)
                else:
                    items.append(item)
            out[k] = items
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _candidate_to_dict(c, candidate_attrs) -> dict:
    """将 CandidateView 转换为字典。"""
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
    d.update(candidate_attrs)
    if c.pred_score:
        d["pred_score"] = _score_to_serializable(c.pred_score)
    if c.true_score:
        # true_score 为离线评估结果，标注来源
        d["true_score"] = _score_to_serializable(c.true_score)
    return d


def save_candidate_debug_json(
    candidates: list,
    path: str,
    episode_info: dict = None,
    oracle_eval: dict = None,
):
    """保存候选点调试信息为 JSON。

    参数：
        candidates: CandidateView 列表。
        path: 输出 JSON 文件路径。
        episode_info: 可选的 episode 级别信息字典。
        oracle_eval: 可选的 Oracle 离线评估结果字典（单独标注）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {
        "episode_info": episode_info or {},
        "candidates": [_candidate_to_dict(c, {}) for c in candidates],
    }
    if oracle_eval is not None:
        data["oracle_eval"] = {
            **oracle_eval,
            "note": "offline evaluation (upper bound, not deployable)",
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)