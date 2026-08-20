"""
Paths 模块单元测试 —— test_paths.py
===================================
"""

import os
import unittest
from pathlib import Path

from ea_avs_mvp_v7.core.paths import (
    get_repo_root,
    get_data_root,
    get_assets_dir,
    get_runs_dir,
    get_datasets_dir,
    to_relative_data_path,
    from_relative_data_path,
)


class TestPaths(unittest.TestCase):
    """路径管理与相对性解析测试。"""

    def test_repo_root_exists(self):
        repo_root = get_repo_root()
        self.assertTrue(repo_root.exists())
        self.assertTrue((repo_root / "AGENTS.md").exists())

    def test_data_root_resolution(self):
        data_root = get_data_root()
        self.assertTrue(data_root.exists())
        self.assertTrue(data_root.is_absolute())

    def test_subdirectories_creation(self):
        assets_p = get_assets_dir("test_sub")
        self.assertTrue(assets_p.exists())
        runs_p = get_runs_dir("test_sub")
        self.assertTrue(runs_p.exists())
        ds_p = get_datasets_dir("test_sub")
        self.assertTrue(ds_p.exists())

    def test_relative_path_roundtrip(self):
        data_root = get_data_root()
        sample_abs = data_root / "assets" / "sample.txt"
        rel = to_relative_data_path(sample_abs)
        self.assertFalse(rel.startswith("/"))
        self.assertFalse("/home/" in rel)
        self.assertEqual(rel, "assets/sample.txt")

        restored = from_relative_data_path(rel)
        self.assertEqual(restored.resolve(), sample_abs.resolve())


if __name__ == "__main__":
    unittest.main()
