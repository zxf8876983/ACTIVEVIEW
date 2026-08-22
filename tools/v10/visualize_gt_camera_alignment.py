#!/usr/bin/env python3
"""
Alias for tools/visualize_gt_camera_alignment.py located in tools/v10/.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.visualize_gt_camera_alignment import main

if __name__ == "__main__":
    main()
