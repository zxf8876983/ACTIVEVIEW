"""
候选视点几何与观测特征编码器 —— view_encoder.py
=================================================

职责：
    将单个候选视点的几何特征 (距离、角度 sin/cos、姿态覆盖率、损失率、投影面积、5 身体分区覆盖率) 编码为 32 维潜在表示。
"""

import math
from typing import Dict, Union
import numpy as np
import torch
import torch.nn as nn

from ea_avs_mvp_v9.core.types import ViewFeature


def extract_view_vector(
    feature: Union[ViewFeature, Dict],
    max_distance: float = 4.5,
) -> np.ndarray:
    """将 ViewFeature 转换为 11 维标准化特征向量。"""
    if isinstance(feature, ViewFeature):
        dist = feature.distance
        angle_deg = feature.viewing_angle_deg
        pose_cov = feature.pose_coverage
        vis_loss = feature.visibility_loss_ratio
        proj_area = feature.projected_area_ratio
        regions = feature.region_coverages
    else:
        dist = float(feature.get("distance", 2.0))
        angle_deg = float(feature.get("viewing_angle_deg", 0.0))
        pose_cov = float(feature.get("pose_coverage", 1.0))
        vis_loss = float(feature.get("visibility_loss_ratio", 0.0))
        proj_area = float(feature.get("projected_area_ratio", 0.05))
        regions = feature.get("region_coverages", {})

    angle_rad = math.radians(angle_deg)
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)

    norm_dist = min(1.0, dist / max(1e-4, max_distance))

    head_cov = float(regions.get("head", pose_cov))
    torso_cov = float(regions.get("torso", pose_cov))
    pelvis_cov = float(regions.get("pelvis", pose_cov))
    upper_cov = float(regions.get("upper_body", pose_cov))
    lower_cov = float(regions.get("lower_body", pose_cov))

    vec = [
        norm_dist,
        sin_a,
        cos_a,
        pose_cov,
        vis_loss,
        proj_area,
        head_cov,
        torso_cov,
        pelvis_cov,
        upper_cov,
        lower_cov,
    ]
    return np.array(vec, dtype=np.float32)


class ViewFeatureEncoder(nn.Module):
    """候选视点多维特征 MLP 编码器。"""

    def __init__(
        self,
        input_dim: int = 11,
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

    def forward(self, view_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_input: Tensor of shape (B, 11) or (B, N, 11)
        Returns:
            Tensor of shape (B, 32) or (B, N, 32)
        """
        return self.net(view_input)
