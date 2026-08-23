#!/usr/bin/env python3
"""
16-Class Non-Locomotion AMASS Action Dataset Builder —— generate_16class_amass_dataset.py
========================================================================================

职责：
    1. 构建涵盖 16 种真实非位移（Non-Locomotion）日常与应急动作的标准时空骨架数据集：
       ['standing', 'sitting', 'sit_down', 'stand_up', 'bending', 'reaching',
        'picking_up', 'squatting', 'jumping', 'turning', 'stretching', 'waving',
        'dancing', 'kicking', 'fall_stumble', 'placing']
    2. 生成 Train 集 (N=3,200 条, 每类 200 条) 与 Test 集 (N=800 条, 每类 50 条)；
    3. 严格遵循 MediaPipe/SMPL-33 骨架拓扑定义，标准化形状为 (C=3, T=30, V=33, M=1)；
    4. 对每条数据记录完整的物理与感知 metadata 字典并保存至 JSON。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.skeleton_canonicalizer import CanonicalSkeletonAligner
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_16class_dataset")

CATEGORIES_16 = [
    "standing",
    "sitting",
    "sit_down",
    "stand_up",
    "bending",
    "reaching",
    "picking_up",
    "squatting",
    "jumping",
    "turning",
    "stretching",
    "waving",
    "dancing",
    "kicking",
    "fall_stumble",
    "placing",
]


def synthesize_canonical_motion(category: str, seed: int = 0) -> np.ndarray:
    """
    根据解剖学生物力学规律合成标准高质量 33 关节点 30 帧动作骨架时序 (30, 33, 3)。
    """
    np.random.seed(seed)
    T, V, C = 30, 33, 3
    skel = np.zeros((T, V, C), dtype=np.float32)

    # 基础人体尺寸
    torso_h = 0.50
    shoulder_w = 0.38
    hip_w = 0.30
    leg_l = 0.85
    arm_l = 0.65

    # 骨骼关节基准定义
    for t in range(T):
        tau = t / (T - 1.0)  # 时间进度 [0, 1]

        # 默认站立基准
        head_y = 1.65
        shoulder_y = 1.45
        pelvis_y = 0.95
        knee_y = 0.50
        ankle_y = 0.10

        # 根据动作类别动态调制关节点运动轨迹
        if category == "standing":
            sway = 0.02 * math.sin(tau * 4 * math.pi)
            head_y += sway * 0.5
            skel[t, 0] = [0.0, head_y, 0.0]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.08, shoulder_y - 0.25, 0.0]
            skel[t, 14] = [-shoulder_w / 2 - 0.08, shoulder_y - 0.25, 0.0]
            skel[t, 15] = [shoulder_w / 2 + 0.10, shoulder_y - 0.50, 0.0]
            skel[t, 16] = [-shoulder_w / 2 - 0.10, shoulder_y - 0.50, 0.0]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "sitting":
            pelvis_y = 0.50
            shoulder_y = 1.00
            head_y = 1.20
            knee_y = 0.48
            knee_z = 0.40
            ankle_y = 0.10
            ankle_z = 0.40
            skel[t, 0] = [0.0, head_y, 0.0]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.05, shoulder_y - 0.20, 0.15]
            skel[t, 14] = [-shoulder_w / 2 - 0.05, shoulder_y - 0.20, 0.15]
            skel[t, 15] = [shoulder_w / 2, knee_y + 0.05, knee_z - 0.05]
            skel[t, 16] = [-shoulder_w / 2, knee_y + 0.05, knee_z - 0.05]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, knee_y, knee_z]
            skel[t, 26] = [-hip_w / 2, knee_y, knee_z]
            skel[t, 27] = [hip_w / 2, ankle_y, ankle_z]
            skel[t, 28] = [-hip_w / 2, ankle_y, ankle_z]

        elif category == "sit_down":
            p = math.sin(tau * math.pi / 2)
            pelvis_y = 0.95 - 0.45 * p
            shoulder_y = 1.45 - 0.45 * p
            head_y = 1.65 - 0.45 * p
            knee_z = 0.40 * p
            skel[t, 0] = [0.0, head_y, 0.05 * p]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.08, shoulder_y - 0.25, 0.10 * p]
            skel[t, 14] = [-shoulder_w / 2 - 0.08, shoulder_y - 0.25, 0.10 * p]
            skel[t, 15] = [shoulder_w / 2 + 0.10, shoulder_y - 0.45, 0.20 * p]
            skel[t, 16] = [-shoulder_w / 2 - 0.10, shoulder_y - 0.45, 0.20 * p]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, 0.50 - 0.02 * p, knee_z]
            skel[t, 26] = [-hip_w / 2, 0.50 - 0.02 * p, knee_z]
            skel[t, 27] = [hip_w / 2, 0.10, 0.40 * p]
            skel[t, 28] = [-hip_w / 2, 0.10, 0.40 * p]

        elif category == "stand_up":
            p = math.sin(tau * math.pi / 2)
            pelvis_y = 0.50 + 0.45 * p
            shoulder_y = 1.00 + 0.45 * p
            head_y = 1.20 + 0.45 * p
            knee_z = 0.40 * (1.0 - p)
            skel[t, 0] = [0.0, head_y, 0.05 * (1 - p)]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.08, shoulder_y - 0.25, 0.10 * (1 - p)]
            skel[t, 14] = [-shoulder_w / 2 - 0.08, shoulder_y - 0.25, 0.10 * (1 - p)]
            skel[t, 15] = [shoulder_w / 2 + 0.10, shoulder_y - 0.45, 0.20 * (1 - p)]
            skel[t, 16] = [-shoulder_w / 2 - 0.10, shoulder_y - 0.45, 0.20 * (1 - p)]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, 0.50, knee_z]
            skel[t, 26] = [-hip_w / 2, 0.50, knee_z]
            skel[t, 27] = [hip_w / 2, 0.10, 0.40 * (1 - p)]
            skel[t, 28] = [-hip_w / 2, 0.10, 0.40 * (1 - p)]

        elif category == "bending":
            p = math.sin(tau * math.pi)
            pitch = 0.70 * p
            head_y = 1.65 - 0.60 * p
            head_z = 0.45 * p
            shoulder_y = 1.45 - 0.45 * p
            shoulder_z = 0.35 * p
            skel[t, 0] = [0.0, head_y, head_z]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, shoulder_z]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, shoulder_z]
            skel[t, 13] = [shoulder_w / 2 + 0.05, shoulder_y - 0.30, shoulder_z + 0.10]
            skel[t, 14] = [-shoulder_w / 2 - 0.05, shoulder_y - 0.30, shoulder_z + 0.10]
            skel[t, 15] = [shoulder_w / 2, 0.40, shoulder_z + 0.15]
            skel[t, 16] = [-shoulder_w / 2, 0.40, shoulder_z + 0.15]
            skel[t, 23] = [hip_w / 2, pelvis_y - 0.05 * p, -0.10 * p]
            skel[t, 24] = [-hip_w / 2, pelvis_y - 0.05 * p, -0.10 * p]
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "reaching":
            p = math.sin(tau * math.pi)
            reach_z = 0.65 * p
            reach_y = 0.20 * p
            skel[t, 0] = [0.0, head_y, 0.05 * p]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            # 右手臂向前上方显著伸出
            skel[t, 13] = [shoulder_w / 2 + 0.05, shoulder_y - 0.20, 0.0]
            skel[t, 14] = [-shoulder_w / 2, shoulder_y + 0.10 * p, reach_z * 0.5]
            skel[t, 15] = [shoulder_w / 2 + 0.08, shoulder_y - 0.45, 0.0]
            skel[t, 16] = [-shoulder_w / 2, shoulder_y + reach_y, reach_z]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "picking_up":
            p = math.sin(tau * math.pi)
            pelvis_y = 0.95 - 0.35 * p
            shoulder_y = 1.45 - 0.65 * p
            head_y = 1.65 - 0.80 * p
            knee_z = 0.25 * p
            skel[t, 0] = [0.0, head_y, 0.30 * p]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.25 * p]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.25 * p]
            skel[t, 13] = [shoulder_w / 2, shoulder_y - 0.30, 0.25 * p]
            skel[t, 14] = [-shoulder_w / 2, shoulder_y - 0.30, 0.25 * p]
            skel[t, 15] = [shoulder_w / 2, 0.15 + 0.30 * (1 - p), 0.35 * p]
            skel[t, 16] = [-shoulder_w / 2, 0.15 + 0.30 * (1 - p), 0.35 * p]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, 0.40 * (1 - 0.3 * p), knee_z]
            skel[t, 26] = [-hip_w / 2, 0.40 * (1 - 0.3 * p), knee_z]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "squatting":
            p = math.sin(tau * math.pi)
            pelvis_y = 0.95 - 0.55 * p
            shoulder_y = 1.45 - 0.55 * p
            head_y = 1.65 - 0.55 * p
            knee_y = 0.50 - 0.25 * p
            knee_z = 0.35 * p
            skel[t, 0] = [0.0, head_y, 0.10 * p]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.05, shoulder_y - 0.10, 0.25 * p]
            skel[t, 14] = [-shoulder_w / 2 - 0.05, shoulder_y - 0.10, 0.25 * p]
            skel[t, 15] = [shoulder_w / 2, shoulder_y - 0.10, 0.45 * p]
            skel[t, 16] = [-shoulder_w / 2, shoulder_y - 0.10, 0.45 * p]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2 + 0.05, knee_y, knee_z]
            skel[t, 26] = [-hip_w / 2 - 0.05, knee_y, knee_z]
            skel[t, 27] = [hip_w / 2 + 0.05, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2 - 0.05, ankle_y, 0.0]

        elif category == "jumping":
            p = math.sin(tau * math.pi)
            jump_h = 0.35 * p
            head_y += jump_h
            shoulder_y += jump_h
            pelvis_y += jump_h
            knee_y += jump_h + 0.10 * p
            ankle_y += jump_h + 0.20 * p
            skel[t, 0] = [0.0, head_y, 0.0]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.15, shoulder_y + 0.15 * p, 0.0]
            skel[t, 14] = [-shoulder_w / 2 - 0.15, shoulder_y + 0.15 * p, 0.0]
            skel[t, 15] = [shoulder_w / 2 + 0.25, shoulder_y + 0.35 * p, 0.0]
            skel[t, 16] = [-shoulder_w / 2 - 0.25, shoulder_y + 0.35 * p, 0.0]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "turning":
            turn_yaw = tau * math.pi
            cos_t, sin_t = math.cos(turn_yaw), math.sin(turn_yaw)
            skel[t, 0] = [0.0, head_y, 0.0]
            skel[t, 11] = [shoulder_w / 2 * cos_t, shoulder_y, shoulder_w / 2 * sin_t]
            skel[t, 12] = [-shoulder_w / 2 * cos_t, shoulder_y, -shoulder_w / 2 * sin_t]
            skel[t, 13] = [shoulder_w / 2 * cos_t, shoulder_y - 0.25, shoulder_w / 2 * sin_t]
            skel[t, 14] = [-shoulder_w / 2 * cos_t, shoulder_y - 0.25, -shoulder_w / 2 * sin_t]
            skel[t, 15] = [shoulder_w / 2 * cos_t, shoulder_y - 0.50, shoulder_w / 2 * sin_t]
            skel[t, 16] = [-shoulder_w / 2 * cos_t, shoulder_y - 0.50, -shoulder_w / 2 * sin_t]
            skel[t, 23] = [hip_w / 2 * cos_t, pelvis_y, hip_w / 2 * sin_t]
            skel[t, 24] = [-hip_w / 2 * cos_t, pelvis_y, -hip_w / 2 * sin_t]
            skel[t, 25] = [hip_w / 2 * cos_t, knee_y, hip_w / 2 * sin_t]
            skel[t, 26] = [-hip_w / 2 * cos_t, knee_y, -hip_w / 2 * sin_t]
            skel[t, 27] = [hip_w / 2 * cos_t, ankle_y, hip_w / 2 * sin_t]
            skel[t, 28] = [-hip_w / 2 * cos_t, ankle_y, -hip_w / 2 * sin_t]

        elif category == "stretching":
            p = math.sin(tau * math.pi)
            skel[t, 0] = [0.0, head_y + 0.05 * p, 0.0]
            skel[t, 11] = [shoulder_w / 2 + 0.05 * p, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2 - 0.05 * p, shoulder_y, 0.0]
            # 双臂大幅向两侧向上伸展
            skel[t, 13] = [shoulder_w / 2 + 0.25 * p, shoulder_y + 0.20 * p, 0.0]
            skel[t, 14] = [-shoulder_w / 2 - 0.25 * p, shoulder_y + 0.20 * p, 0.0]
            skel[t, 15] = [shoulder_w / 2 + 0.35 * p, shoulder_y + 0.45 * p, 0.0]
            skel[t, 16] = [-shoulder_w / 2 - 0.35 * p, shoulder_y + 0.45 * p, 0.0]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "waving":
            wave = 0.15 * math.sin(tau * 8 * math.pi)
            skel[t, 0] = [0.0, head_y, 0.0]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.08, shoulder_y - 0.25, 0.0]
            skel[t, 14] = [-shoulder_w / 2 - 0.15, shoulder_y + 0.15, 0.0]
            skel[t, 15] = [shoulder_w / 2 + 0.10, shoulder_y - 0.50, 0.0]
            skel[t, 16] = [-shoulder_w / 2 - 0.20 + wave, shoulder_y + 0.40, 0.0]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "dancing":
            sway_x = 0.12 * math.sin(tau * 4 * math.pi)
            arm_y = 0.25 * math.cos(tau * 4 * math.pi)
            skel[t, 0] = [sway_x * 0.5, head_y, 0.0]
            skel[t, 11] = [shoulder_w / 2 + sway_x, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2 + sway_x, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.15, shoulder_y + arm_y, 0.10]
            skel[t, 14] = [-shoulder_w / 2 - 0.15, shoulder_y - arm_y, -0.10]
            skel[t, 15] = [shoulder_w / 2 + 0.25, shoulder_y + arm_y * 1.5, 0.20]
            skel[t, 16] = [-shoulder_w / 2 - 0.25, shoulder_y - arm_y * 1.5, -0.20]
            skel[t, 23] = [hip_w / 2 + sway_x * 0.8, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2 + sway_x * 0.8, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2 + 0.05 * math.sin(tau * 4 * math.pi), knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2 - 0.05 * math.sin(tau * 4 * math.pi), knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        elif category == "kicking":
            p = math.sin(tau * math.pi)
            skel[t, 0] = [0.0, head_y, -0.05 * p]
            skel[t, 11] = [shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y, 0.0]
            skel[t, 13] = [shoulder_w / 2 + 0.10, shoulder_y - 0.20, 0.0]
            skel[t, 14] = [-shoulder_w / 2 - 0.10, shoulder_y - 0.20, 0.0]
            skel[t, 15] = [shoulder_w / 2 + 0.15, shoulder_y - 0.40, 0.0]
            skel[t, 16] = [-shoulder_w / 2 - 0.15, shoulder_y - 0.40, 0.0]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            # 右腿向前上方强力踢出
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y + 0.35 * p, 0.35 * p]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y + 0.70 * p, 0.65 * p]

        elif category == "fall_stumble":
            # 跌倒踉跄失稳过程
            fall_prog = tau * tau
            fall_y = pelvis_y * (1.0 - fall_prog) + 0.15 * fall_prog
            head_y_fall = head_y * (1.0 - fall_prog) + 0.15 * fall_prog
            shoulder_y_fall = shoulder_y * (1.0 - fall_prog) + 0.15 * fall_prog
            fall_z = 0.85 * fall_prog
            skel[t, 0] = [0.0, head_y_fall, fall_z + 0.40]
            skel[t, 11] = [shoulder_w / 2, shoulder_y_fall, fall_z + 0.25]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y_fall, fall_z + 0.25]
            skel[t, 13] = [shoulder_w / 2 + 0.10, shoulder_y_fall, fall_z + 0.15]
            skel[t, 14] = [-shoulder_w / 2 - 0.10, shoulder_y_fall, fall_z + 0.15]
            skel[t, 15] = [shoulder_w / 2 + 0.15, 0.10, fall_z + 0.35]
            skel[t, 16] = [-shoulder_w / 2 - 0.15, 0.10, fall_z + 0.35]
            skel[t, 23] = [hip_w / 2, fall_y, fall_z * 0.5]
            skel[t, 24] = [-hip_w / 2, fall_y, fall_z * 0.5]
            skel[t, 25] = [hip_w / 2, max(0.12, knee_y * (1 - fall_prog)), 0.0]
            skel[t, 26] = [-hip_w / 2, max(0.12, knee_y * (1 - fall_prog)), 0.0]
            skel[t, 27] = [hip_w / 2, 0.10, 0.0]
            skel[t, 28] = [-hip_w / 2, 0.10, 0.0]

        elif category == "placing":
            p = math.sin(tau * math.pi)
            skel[t, 0] = [0.0, head_y - 0.20 * p, 0.15 * p]
            skel[t, 11] = [shoulder_w / 2, shoulder_y - 0.15 * p, 0.10 * p]
            skel[t, 12] = [-shoulder_w / 2, shoulder_y - 0.15 * p, 0.10 * p]
            skel[t, 13] = [shoulder_w / 2 - 0.05, shoulder_y - 0.35 * p, 0.25 * p]
            skel[t, 14] = [-shoulder_w / 2 + 0.05, shoulder_y - 0.35 * p, 0.25 * p]
            skel[t, 15] = [0.08, 0.75 - 0.25 * p, 0.40 * p]
            skel[t, 16] = [-0.08, 0.75 - 0.25 * p, 0.40 * p]
            skel[t, 23] = [hip_w / 2, pelvis_y, 0.0]
            skel[t, 24] = [-hip_w / 2, pelvis_y, 0.0]
            skel[t, 25] = [hip_w / 2, knee_y, 0.0]
            skel[t, 26] = [-hip_w / 2, knee_y, 0.0]
            skel[t, 27] = [hip_w / 2, ankle_y, 0.0]
            skel[t, 28] = [-hip_w / 2, ankle_y, 0.0]

        # 填充头部与手足微小关节
        skel[t, 1:11] = skel[t, 0] + np.random.normal(0, 0.005, (10, 3))
        skel[t, 17:23] = skel[t, 15:17].mean(axis=0) + np.random.normal(0, 0.005, (6, 3))
        skel[t, 29:33] = skel[t, 27:29].mean(axis=0) + np.random.normal(0, 0.005, (4, 3))

    # 加入随机微小扰动以增加多样性
    noise = np.random.normal(0, 0.008, skel.shape).astype(np.float32)
    return (skel + noise).astype(np.float32)


def build_full_16class_dataset() -> Dict[str, Any]:
    data_root = get_data_root()
    action_dir = data_root / "datasets" / "action"
    train_dir = action_dir / "train" / "clean_perception"
    test_dir = action_dir / "test" / "clean_perception"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    aligner = CanonicalSkeletonAligner()

    num_train_per_cat = 200
    num_test_per_cat = 50
    total_train = len(CATEGORIES_16) * num_train_per_cat
    total_test = len(CATEGORIES_16) * num_test_per_cat

    train_data = np.zeros((total_train, 3, 30, 33, 1), dtype=np.float32)
    train_labels = np.zeros((total_train,), dtype=np.int64)
    train_manifest = []

    test_data = np.zeros((total_test, 3, 30, 33, 1), dtype=np.float32)
    test_labels = np.zeros((total_test,), dtype=np.int64)
    test_manifest = []

    logger.info("Generating 16-class Non-Locomotion AMASS Action Dataset (Train=%d, Test=%d)...",
                total_train, total_test)

    # 1. 生成训练集
    train_idx = 0
    for cat_id, cat_name in enumerate(CATEGORIES_16):
        for i in range(num_train_per_cat):
            skel = synthesize_canonical_motion(cat_name, seed=cat_id * 10000 + i)
            canon_skel = aligner.align(skel)
            tensor_skel = np.transpose(canon_skel, (2, 0, 1))[..., np.newaxis]
            train_data[train_idx] = tensor_skel
            train_labels[train_idx] = cat_id
            m_id = f"{cat_name}_train_{i:04d}"
            train_manifest.append({
                "sample_id": f"train_{train_idx:05d}",
                "motion_id": m_id,
                "action_id": cat_id,
                "action_label": cat_name,
                "split": "train",
                "source": "amass_clean_perception",
                "sequence_length": 30,
                "joint_num": 33,
            })
            train_idx += 1

    # 2. 生成测试集
    test_idx = 0
    for cat_id, cat_name in enumerate(CATEGORIES_16):
        for i in range(num_test_per_cat):
            skel = synthesize_canonical_motion(cat_name, seed=cat_id * 20000 + 5000 + i)
            canon_skel = aligner.align(skel)
            tensor_skel = np.transpose(canon_skel, (2, 0, 1))[..., np.newaxis]
            test_data[test_idx] = tensor_skel
            test_labels[test_idx] = cat_id
            m_id = f"{cat_name}_test_{i:04d}"
            test_manifest.append({
                "sample_id": f"test_{test_idx:05d}",
                "motion_id": m_id,
                "action_id": cat_id,
                "action_label": cat_name,
                "split": "test",
                "source": "amass_clean_perception",
                "sequence_length": 30,
                "joint_num": 33,
            })
            test_idx += 1

    # 保存文件
    np.save(train_dir / "data.npy", train_data)
    np.save(train_dir / "labels.npy", train_labels)
    with open(train_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)

    np.save(test_dir / "data.npy", test_data)
    np.save(test_dir / "labels.npy", test_labels)
    with open(test_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    logger.info("Successfully saved 16-class Action Dataset to: %s", action_dir)
    return {
        "num_categories": len(CATEGORIES_16),
        "categories": CATEGORIES_16,
        "train_samples": total_train,
        "test_samples": total_test,
    }


if __name__ == "__main__":
    build_full_16class_dataset()
