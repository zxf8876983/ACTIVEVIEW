"""
Root Test Shortcut: v11.2.1 Metadata Test
"""
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.tests.unit.test_v1121_metadata import TestV1121MetadataEnhancement

if __name__ == "__main__":
    unittest.main()
