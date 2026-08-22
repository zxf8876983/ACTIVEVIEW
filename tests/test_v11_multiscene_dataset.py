#!/usr/bin/env python3
"""
Top-Level Tool Shortcut: Test Multi-Scene Active View Dataset & Predictor
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tests.unit.test_v11_multiscene_dataset import TestMultiSceneDataset

if __name__ == "__main__":
    unittest.main()
