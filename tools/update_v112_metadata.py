#!/usr/bin/env python3
"""
Top-Level Tool Shortcut: v11.2.1 Metadata Enhancement Tool
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tools.update_v112_metadata import main

if __name__ == "__main__":
    main()
