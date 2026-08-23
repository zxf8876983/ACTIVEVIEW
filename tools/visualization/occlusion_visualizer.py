#!/usr/bin/env python3
"""
Top-Level Tool Shortcut: Occlusion Analysis Visualizer (v11.5)
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tools.visualization.occlusion_visualizer import OcclusionVisualizer

if __name__ == "__main__":
    visualizer = OcclusionVisualizer()
    visualizer.plot_occlusion_analysis()
