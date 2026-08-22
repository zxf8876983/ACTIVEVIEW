"""
CandidateViewGenerator 单元测试 —— test_candidate_generator.py
=============================================================
"""

import math
import sys
import unittest
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint


class TestCandidateViewGenerator(unittest.TestCase):
    """测试极坐标候选视点生成器功能与数据结构。"""

    def setUp(self):
        self.generator = CandidateViewGenerator()
        self.human_pos = [1.0, 0.0, 2.0]
        self.robot_pos = [3.0, 0.0, 4.0]

    def test_default_generation_count(self):
        """测试默认参数 (8 角度 x 4 距离) 必须精准生成 32 个候选视点。"""
        candidates = self.generator.generate(self.human_pos)
        self.assertEqual(len(candidates), 32)

    def test_viewpoint_attributes_completeness(self):
        """测试生成的每一个视点均包含完整合规的物理姿态与元数据。"""
        candidates = self.generator.generate(self.human_pos, robot_current_position=self.robot_pos)
        for vp in candidates:
            self.assertIsInstance(vp, Viewpoint)
            self.assertIsInstance(vp.id, int)
            self.assertEqual(len(vp.position), 3)
            self.assertEqual(len(vp.rotation), 2)
            self.assertIsInstance(vp.yaw, float)
            self.assertIsInstance(vp.pitch, float)
            self.assertIsInstance(vp.distance, float)
            self.assertIsInstance(vp.angle, float)
            self.assertIsInstance(vp.camera_height, float)
            self.assertIsInstance(vp.navigation_cost, float)
            self.assertTrue(vp.is_navigable)
            self.assertTrue(vp.is_reachable)
            self.assertTrue(vp.is_visible)

            # 校验导出的字典结构
            d = vp.to_dict()
            self.assertIn("id", d)
            self.assertIn("position", d)
            self.assertIn("camera_position", d)
            self.assertIn("yaw", d)
            self.assertIn("distance", d)
            self.assertIn("navigation_cost", d)

    def test_yaw_facing_human(self):
        """测试机器人相机朝向人体 (Face Human Yaw) 的计算几何正确性。"""
        hx, hy, hz = self.human_pos
        candidates = self.generator.generate(self.human_pos)
        for vp in candidates:
            vx, vy, vz = vp.position
            dx = hx - vx
            dz = hz - vz
            expected_yaw = (math.degrees(math.atan2(dx, dz)) + 360.0) % 360.0
            diff = abs((vp.yaw - expected_yaw + 180.0) % 360.0 - 180.0)
            self.assertLess(diff, 1e-3)

    def test_custom_sampling_configuration(self):
        """测试自定义角度与距离配置。"""
        custom_gen = CandidateViewGenerator(
            angles=[0.0, 90.0, 180.0, 270.0],
            distances=[1.5, 3.0],
            camera_height=1.40,
            camera_pitch=-5.0,
        )
        candidates = custom_gen.generate(self.human_pos)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(candidates[0].camera_height, 1.40)
        self.assertEqual(candidates[0].pitch, -5.0)


if __name__ == "__main__":
    unittest.main()
