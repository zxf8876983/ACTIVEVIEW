"""文件用途：暴露 ST-GCN 图结构与模型。

主要输入：H36M-17 骨架序列和图拓扑配置。
主要输出：Graph、STGCN 与 STGCNBlock。
项目角色：冻结动作识别 backbone 的正式实现。
"""

from .graph import Graph
from .model import STGCN, STGCNBlock, load_checkpoint

__all__ = ["Graph", "STGCN", "STGCNBlock", "load_checkpoint"]
