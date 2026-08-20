"""
v8 视点评估、质量打分与基线策略包
"""

from .baseline_strategies import (
    select_geometry_best_view,
    select_nearest_view,
    select_random_view,
    select_view,
)
from .view_metrics import summarize_viewpoint_qualities
from .view_quality import ViewQualityEvaluator, compute_view_quality_score

__all__ = [
    "summarize_viewpoint_qualities",
    "ViewQualityEvaluator",
    "compute_view_quality_score",
    "select_view",
    "select_random_view",
    "select_nearest_view",
    "select_geometry_best_view",
]
