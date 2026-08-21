"""
v9 可视化与报表输出模块
"""

from .action_view_plotter import format_comparison_table
from .action_comparison_plotter import plot_action_comparison_figure

__all__ = [
    "format_comparison_table",
    "plot_action_comparison_figure",
]
