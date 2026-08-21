"""
人体姿态特征编码器 —— pose_encoder.py
======================================

职责：
    将人体的 16 个骨骼关键点 3D 相对坐标 (48 维) 与人体偏航朝向角 (1 维) 编码为低维潜在表示 (32 维)。
"""

import math
from typing import Dict, List, Union
import numpy as np
import torch
import torch.nn as nn


def extract_pose_vector(
    human_joints_3d: Dict[str, List[float]],
    human_yaw_deg: float = 0.0,
) -> np.ndarray:
    """将关节字典转换为标准化的 49 维向量 (16 关节相对 Pelvis 坐标 + 偏航角)。"""
    pelvis = human_joints_3d.get("pelvis", [0.0, 0.0, 0.0])
    px, py, pz = float(pelvis[0]), float(pelvis[1]), float(pelvis[2])

    # 确定性的标准关节顺序
    joint_order = [
        "pelvis", "head", "neck", "spine", "chest",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle"
    ]

    coords = []
    for j_name in joint_order:
        pos = human_joints_3d.get(j_name)
        if pos is not None and len(pos) >= 3:
            coords.extend([float(pos[0]) - px, float(pos[1]) - py, float(pos[2]) - pz])
        else:
            coords.extend([0.0, 0.0, 0.0])

    # 截断或填充至 48 维
    coords = coords[:48]
    if len(coords) < 48:
        coords.extend([0.0] * (48 - len(coords)))

    # 添加归一化朝向角 [-1.0, 1.0]
    norm_yaw = float(human_yaw_deg % 360.0) / 180.0 - 1.0
    coords.append(norm_yaw)

    return np.array(coords, dtype=np.float32)


class HumanPoseEncoder(nn.Module):
    """人体骨骼姿态 MLP 编码器。"""

    def __init__(
        self,
        input_dim: int = 49,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pose_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_input: Tensor of shape (B, 49) or (B, N, 49)
        Returns:
            Tensor of shape (B, 32) or (B, N, 32)
        """
        return self.net(pose_input)
