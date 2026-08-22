#!/usr/bin/env python3
"""
Top-Level Tool Alias: Viewpoint Visualizer
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tools.visualization.viewpoint_visualizer import main, visualize_candidate_viewpoints

if __name__ == "__main__":
    main()
