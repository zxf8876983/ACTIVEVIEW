#!/usr/bin/env python3
"""
Top-Level ACTIVEVIEW v11.0 Runner Shortcut
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.scripts.run_v11 import main

if __name__ == "__main__":
    main()
