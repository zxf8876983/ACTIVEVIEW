"""
Root Test Shortcut: v11.3 Utility Predictor Test
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tests.unit.test_v113_utility_predictor import TestV113UtilityPredictor

if __name__ == "__main__":
    unittest.main()
