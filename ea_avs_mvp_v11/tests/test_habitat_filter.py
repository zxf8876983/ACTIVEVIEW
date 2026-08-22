"""
HabitatViewFilter 单元测试入口 —— ea_avs_mvp_v11/tests/test_habitat_filter.py
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.tests.test_habitat_filter import TestHabitatViewFilter

if __name__ == "__main__":
    unittest.main()
