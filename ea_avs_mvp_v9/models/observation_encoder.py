"""
当前感知状态编码器 —— observation_encoder.py
============================================

职责：
    1. 提取当前观测状态 (Estimated Joints + Joint Confidences + Body Part Confidences, 71d)；
    2. 将 71 维不完整感知状态映射为 32 维稠密特征向量 (Observation Embedding)。

# GT is only used for supervision/evaluation.
# It must never enter model forward pass.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from ea_avs_mvp_v9.core.types import ObservationState

ALL_16_JOINTS = [
    "pelvis", "head", "neck", "spine", "chest",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle",
]

ALL_7_BODY_PARTS = [
    "head", "torso", "pelvis", "left_hand", "right_hand", "left_leg", "right_leg"
]


def extract_observation_vector(obs: ObservationState) -> np.ndarray:
    """
    将 ObservationState 转换为 71 维归一化感知特征向量:
      - 48 维: 16 关节估计坐标 (相对 estimated pelvis)
      - 16 维: 16 关节点估计置信度
      -  7 维:  7 大解剖部位可见置信度
    """
    pelvis = obs.estimated_joints_3d.get("pelvis", [0.0, 0.0, 0.0])
    px, py, pz = pelvis[0], pelvis[1], pelvis[2]

    coords_rel = []
    conf_list = []

    for j_name in ALL_16_JOINTS:
        if j_name in obs.estimated_joints_3d:
            jx, jy, jz = obs.estimated_joints_3d[j_name]
            coords_rel.extend([jx - px, jy - py, jz - pz])
        else:
            coords_rel.extend([0.0, 0.0, 0.0])

        c = obs.joint_confidences.get(j_name, 0.5)
        conf_list.append(c)

    part_list = [obs.body_part_confidences.get(p, 0.5) for p in ALL_7_BODY_PARTS]

    feat = np.array(coords_rel + conf_list + part_list, dtype=np.float32)
    return feat


class ObservationEncoder(nn.Module):
    """当前人体观测感知质量编码器 (71d -> 64d -> 32d)。"""

    def __init__(
        self,
        input_dim: int = 71,
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        # GT is only used for supervision/evaluation.
        # It must never enter model forward pass.
        
        Args:
            x: (B, 71) 感知特征向量
        Returns:
            out: (B, 32) 感知嵌入向量
        """
        return self.net(x)
