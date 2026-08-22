"""
ST-GCN 人体骨架时空图拓扑与邻接矩阵构建器 —— graph.py
===================================================

职责：
    1. 动态加载 `configs/skeleton_definition.json` 中的 33 关节与骨骼边；
    2. 计算无向图的最短跳数距离矩阵 (Shortest Path Distance Matrix)；
    3. 支持 Spatial Partitioning (空间划分策略) 生成 $K=3$ 个物理邻接矩阵：
       - $A_0$: 自环 (Self-loop / Root node itself)
       - $A_1$: 向心节点 (Centripetal nodes: 距离根节点更近的邻居)
       - $A_2$: 离心节点 (Centrifugal nodes: 距离根节点更远的邻居)
    4. 执行行归一化度矩阵运算 $D^{-1} A$，保证图卷积数值稳定性；
    5. 杜绝硬编码，关节数量 $V$ 与拓扑完全自适应。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v10.perception.skeleton_definition import SkeletonDefinition, get_skeleton_definition

logger = logging.getLogger(__name__)


def build_hop_distance_matrix(num_node: int, edges: List[Tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    """计算节点间跳数距离矩阵。"""
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        if i < num_node and j < num_node:
            A[i, j] = 1
            A[j, i] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    """对有向图/邻接矩阵执行行归一化 D^{-1} A。"""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


class Graph:
    """ST-GCN 骨架图拓扑结构类。"""

    def __init__(
        self,
        strategy: str = "spatial",
        max_hop: int = 1,
        dilation: int = 1,
        skel_def: Optional[SkeletonDefinition] = None,
    ):
        self.skel_def = skel_def or get_skeleton_definition()
        self.num_node = self.skel_def.joint_num
        self.edges = self.skel_def.edges
        self.root_indices = self.skel_def.root_joints
        self.strategy = strategy
        self.max_hop = max_hop
        self.dilation = dilation

        self.hop_dis = build_hop_distance_matrix(self.num_node, self.edges, max_hop=max_hop)
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self) -> np.ndarray:
        """根据策略构建空间邻接矩阵张量 (K, V, V)。"""
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if self.strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif self.strategy == "distance":
            num_subsets = len(valid_hop)
            A = np.zeros((num_subsets, self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif self.strategy == "spatial":
            # 标准空间划分策略 (Spatial Partitioning Strategy: K=3)
            # 1. a_root: 自环 (hop = 0)
            # 2. a_close: 向心邻居 (hop = 1 且更靠近根节点)
            # 3. a_further: 离心邻居 (hop = 1 且距离根节点更远或相等)
            root_dist = np.min(self.hop_dis[:, self.root_indices], axis=1)

            a_root = np.zeros((self.num_node, self.num_node))
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))

            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[i, j] == 0:
                        a_root[j, i] = normalize_adjacency[j, i]
                    elif self.hop_dis[i, j] == 1:
                        if root_dist[j] < root_dist[i]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]

            A = np.stack([a_root, a_close, a_further])
            self.A = A
        else:
            raise ValueError(f"Unknown graph partitioning strategy: {self.strategy}")

        return self.A
