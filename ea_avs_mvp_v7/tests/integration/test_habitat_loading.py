"""
Habitat 场景集成测试 —— test_habitat_loading.py
===============================================
"""

import unittest

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.environment.scene_manager import SceneManager


class TestHabitatLoading(unittest.TestCase):
    """Habitat 室内场景与 NavMesh 定位集成测试。"""

    def test_scene_asset_resolution(self):
        cfg = load_v7_config()
        scene_mgr = SceneManager(cfg.habitat)
        scene_path = scene_mgr.scene_path
        navmesh_path = scene_mgr.navmesh_path

        self.assertTrue(scene_path.exists())
        self.assertEqual(scene_path.suffix, ".glb")
        if navmesh_path:
            self.assertTrue(navmesh_path.exists())


if __name__ == "__main__":
    unittest.main()
