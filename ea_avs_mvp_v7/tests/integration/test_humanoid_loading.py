"""
Humanoid 加载集成测试 —— test_humanoid_loading.py
=================================================
"""

import unittest

from ea_avs_mvp_v7.core.config import load_v7_config
from ea_avs_mvp_v7.human.humanoid_agent import resolve_humanoid_urdf_path


class TestHumanoidLoading(unittest.TestCase):
    """Humanoid URDF 资产与路径解析集成测试。"""

    def test_humanoid_urdf_resolution(self):
        cfg = load_v7_config()
        urdf_p, motion_p = resolve_humanoid_urdf_path(cfg.humanoid)

        self.assertTrue(urdf_p.exists())
        self.assertEqual(urdf_p.name, "neutral_0.urdf")


if __name__ == "__main__":
    unittest.main()
