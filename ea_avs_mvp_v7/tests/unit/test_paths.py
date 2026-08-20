"""
Paths 路径解析模块单元测试 —— test_paths.py
===========================================
"""

import os
import unittest
from pathlib import Path

from ea_avs_mvp_v7.core.paths import (
    from_relative_data_path,
    get_data_root,
    get_repo_root,
    to_relative_data_path,
)


class TestPaths(unittest.TestCase):
    """路径管理与相对性解析测试。"""

    def test_repo_root_exists(self):
        repo_root = get_repo_root()
        self.assertTrue(repo_root.exists())
        self.assertTrue((repo_root / "ea_avs_mvp_v7").exists())

    def test_data_root_resolution(self):
        data_root = get_data_root()
        self.assertTrue(data_root.is_absolute())
        self.assertTrue(data_root.exists())

    def test_relative_path_conversions(self):
        data_root = get_data_root()
        test_file = data_root / "runs" / "test_ep" / "frame_0.png"
        rel_path = to_relative_data_path(test_file)
        self.assertEqual(rel_path, "runs/test_ep/frame_0.png")

        reconstructed = from_relative_data_path(rel_path)
        self.assertEqual(reconstructed.resolve(), test_file.resolve())


if __name__ == "__main__":
    unittest.main()
