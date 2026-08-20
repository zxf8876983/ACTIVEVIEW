"""
EA-AVS-MVP v7.0 基础评测指标工具 (仅用于验证生成数据集的基本完整性)
"""

from .basic_metrics import compute_episode_statistics

__all__ = [
    "compute_episode_statistics",
]
