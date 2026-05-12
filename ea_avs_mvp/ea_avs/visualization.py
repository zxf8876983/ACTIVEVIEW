"""
可视化与调试输出模块 —— visualization.py
=========================================

功能：
    负责保存渲染图像和调试数据文件。

MVP0.1 的输出要求：
    - 每个 episode 的每种策略选择后的 RGB 图像（PNG 格式）
    - 每个 episode 的候选视角调试信息（JSON 格式）
    
输出文件命名规则：
    - 图像：images/ep_{episode_id:03d}_{policy_name}.png
    - 候选点调试：debug/ep_{episode_id:03d}_candidates.json

图像保存说明（每个成功 episode 至少 4 张）：
    - ep_000_fixed.png —— Fixed 策略选中的视角
    - ep_000_random.png —— Random 策略选中的视角
    - ep_000_nearest.png —— Nearest 策略选中的视角
    - ep_000_ours.png —— Ours 策略选中的视角
    即使 Fixed 和初始视角相同，仍然保存独立的 fixed 图片，
    方便后续自动对比。
"""

import json
import os
from typing import Dict, List

import numpy as np
from PIL import Image


def save_rgb_image(rgb: np.ndarray, path: str):
    """
    将 RGB 图像保存为 PNG 文件。

    参数：
        rgb: RGB 图像数组，shape=(H, W, 3) 或 (H, W, 4)。
             Habitat 的 color_sensor 输出是 4 通道 RGBA，
             如果是 4 通道会自动去掉 Alpha 通道。
        path: 输出文件路径，如 "outputs/images/ep_000_fixed.png"。

    说明：
        - 自动创建父目录（如果不存在）
        - 使用 PIL (Pillow) 库保存图像
        - 保存为 PNG 格式（无损压缩）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 如果是 RGBA 4 通道，去掉 Alpha 通道转成 RGB
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    img = Image.fromarray(rgb)
    img.save(path)


def draw_skeleton_projection_placeholder(
    rgb: np.ndarray, score: dict
) -> np.ndarray:
    """
    骨架投影绘制的占位函数。
    
    MVP0.1 暂不需要在图像上绘制真实的骨架 2D 投影。
    此函数作为占位符，直接返回原始 RGB 图像。
    
    后续版本（MVP0.3+）会在此函数中实现：
        - 将 3D 骨架关键点投影到 2D 图像平面
        - 绘制关键点和骨骼连线
        - 用颜色标记可见/不可见关键点

    参数：
        rgb: RGB 图像数组。
        score: 视角评分字典（包含 visible_keypoints 等信息）。

    返回：
        与原图像相同的 RGB 数组（不做任何修改）。
    """
    return rgb


def save_candidate_debug_json(
    candidates: list, path: str
):
    """
    将候选视角的详细信息保存为 JSON 文件（用于调试分析）。

    保存的信息包括：
        - 每个候选点的位置、朝向
        - 测地距离和欧氏距离
        - 是否有效及失效原因
        - 评分结果（S_kp, S_center, S_dist, C_move, Q）
        - 可见/不可见关键点列表

    参数：
        candidates: CandidateView 对象列表（包含有效和无效点）。
        path: 输出 JSON 文件路径。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _candidate_to_dict(c):
        """将 CandidateView 对象转换为可 JSON 序列化的字典。"""
        d = {
            "candidate_id": c.candidate_id,
            "position": c.position.tolist() if hasattr(c.position, "tolist") else list(c.position),
            "yaw": float(c.yaw),
            "geodesic_distance": float(c.geodesic_distance),
            "euclidean_distance_to_human": float(c.euclidean_distance_to_human),
            "is_valid": c.is_valid,
            "invalid_reason": c.invalid_reason,
        }
        if c.score:
            # 将评分中的数值转换为 float（去除 numpy 类型）
            d["score"] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in c.score.items()
                if not isinstance(v, list)
            }
            # 保存关键点列表（visible/invisible）作为单独的顶级字段
            for k in ["visible_keypoints", "invisible_keypoints"]:
                if k in c.score:
                    d[k] = c.score[k]
        return d

    data = [_candidate_to_dict(c) for c in candidates]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
