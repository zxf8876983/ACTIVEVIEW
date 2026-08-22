#!/usr/bin/env python3
"""
Top-Level Tool Shortcut: Check Habitat Scene Assets
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tools.check_habitat_scenes import main

if __name__ == "__main__":
    main()
