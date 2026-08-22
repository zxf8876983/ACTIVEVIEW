"""
ST-GCN 时空图卷积动作识别网络 —— st_gcn_model.py
=================================================

参考论文：
    Yan et al. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" (AAAI 2018).

特性：
    1. 输入张量形状：(N, C_in, T, V, M)，其中 N=batch, C_in=3, T=frames, V=33, M=1;
    2. 基于 Graph 模块动态加载 33 关节空间划分邻接矩阵 A (K, V, V);
    3. 包含多层时空图卷积残差块 (Spatial GCN + Temporal Conv2d + Residual + Dropout);
    4. 输出：(N, num_classes) 未归一化预测 logits;
    5. 完全自适应任何骨架拓扑定义。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ea_avs_mvp_v10.action_recognition.graph import Graph
from ea_avs_mvp_v10.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


class ConvTemporalGraphical(nn.Module):
    """空间图卷积层 (Spatial Graph Convolution Layer)。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        # A: (K, V, V)
        assert A.size(0) == self.kernel_size

        x = self.conv(x) # (N, out_channels * K, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v) # (N, K, out_channels, T, V)
        # 矩阵乘法沿节点维度融合: (N, K, out_channels, T, V) * (K, V, V) -> (N, out_channels, T, V)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous()


class STGCNBlock(nn.Module):
    """时空图卷积残差块 (Spatial-Temporal GCN Block)。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCN(nn.Module):
    """完整的 ST-GCN 动作分类神经网络。"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        graph_strategy: str = "spatial",
        edge_importance_weighting: bool = True,
        skel_def: Optional[SkeletonDefinition] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ):
        super().__init__()
        self.skel_def = skel_def or get_skeleton_definition()
        self.graph = Graph(strategy=graph_strategy, skel_def=self.skel_def)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # 批归一化输入骨架
        self.data_bn = nn.BatchNorm1d(in_channels * self.skel_def.joint_num)

        # 9 层时空图卷积残差层
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, dropout=dropout),
            STGCNBlock(64, 64, kernel_size, 1, dropout=dropout),
            STGCNBlock(64, 64, kernel_size, 1, dropout=dropout),
            STGCNBlock(64, 128, kernel_size, 2, dropout=dropout),
            STGCNBlock(128, 128, kernel_size, 1, dropout=dropout),
            STGCNBlock(128, 128, kernel_size, 1, dropout=dropout),
            STGCNBlock(128, 256, kernel_size, 2, dropout=dropout),
            STGCNBlock(256, 256, kernel_size, 1, dropout=dropout),
            STGCNBlock(256, 256, kernel_size, 1, dropout=dropout),
        ))

        # 可学习的自适应边重要性权重矩阵 (Learnable Edge Importance)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 全局分类头
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状支持: (N, C, T, V, M) 或 (N, C, T, V)
        if x.dim() == 4:
            x = x.unsqueeze(-1) # (N, C, T, V, 1)

        N, C, T, V, M = x.size()
        assert V == self.skel_def.joint_num, f"Input joint dimension V={V} does not match configured {self.skel_def.joint_num}"
        assert C == 3, f"Input feature dimension C={C} (expected 3 for XYZ)"

        # 对单人/多人通道进行折叠处理
        x = x.permute(0, 4, 3, 1, 2).contiguous() # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous() # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        # 通过 9 个 ST-GCN 块
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        # 全局时空平均池化 (Global Average Pooling)
        x = F.avg_pool2d(x, x.size()[2:]) # (N * M, 256, 1, 1)
        x = x.view(N, M, -1).mean(dim=1)  # (N, 256)

        # 线性分类输出 logits
        logits = self.fc(x) # (N, num_classes)
        return logits
