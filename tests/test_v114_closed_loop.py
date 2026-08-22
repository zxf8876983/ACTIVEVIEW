#!/usr/bin/env python3
"""
Top-Level Unit Test Alias: test_v114_closed_loop.py
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tests.unit.test_v114_closed_loop import TestV114ClosedLoop

if __name__ == "__main__":
    unittest.main()
