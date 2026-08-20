"""
动作感知视点可视化与报表工具 —— action_view_plotter.py
===================================================
"""

from typing import Any, Dict


def format_comparison_table(report: Dict[str, Any]) -> str:
    """生成格式化的 Baseline 对比文本表格。"""
    lines = []
    lines.append("=" * 78)
    lines.append(f"  ACTIVEVIEW v9.0 Action-conditioned View Selection Comparison")
    lines.append(f"  Action Target: {report.get('action_name', 'unknown').upper()}")
    lines.append("=" * 78)
    lines.append(f"{'Strategy':<20} | {'Selected View':<16} | {'Dist(m)':<8} | {'Angle(deg)':<10} | {'Score Q(v|a)':<12}")
    lines.append("-" * 78)

    strategies = report.get("strategies", {})
    for name, data in strategies.items():
        v_id = data.get("selected_view_id", "N/A")
        dist = f"{data.get('distance', 0.0):.2f}"
        ang = f"{data.get('viewing_angle_deg', 0.0):.1f}"
        score = f"{data.get('action_total_score', 0.0):.3f}"
        lines.append(f"{name:<20} | {v_id:<16} | {dist:<8} | {ang:<10} | {score:<12}")

    lines.append("-" * 78)
    gain = report.get("action_conditioned_gain_over_v8", 0.0)
    shifted = report.get("preferred_viewpoint_shifted", False)
    lines.append(f"  Gain over v8 Geometry Baseline: {gain:+.3f} (Viewpoint Shifted: {shifted})")
    lines.append("=" * 78)

    return "\n".join(lines)
