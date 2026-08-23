#!/usr/bin/env python3
"""
Top-Level Test Shortcut: test_amass_full_dataset.py (v11.5)
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tests.unit.test_amass_full_dataset import TestAmassFullDataset

if __name__ == "__main__":
    unittest.main()
