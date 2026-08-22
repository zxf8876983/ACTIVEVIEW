#!/usr/bin/env python3
"""
Top-Level Tool Shortcut: Closed-Loop Active Perception Visualizer (v11.4)
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tools.visualization.closed_loop_visualizer import ClosedLoopVisualizer

if __name__ == "__main__":
    visualizer = ClosedLoopVisualizer()
    visualizer.plot_closed_loop_evaluation()
