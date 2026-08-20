"""
Humanoid State 模块单元测试 —— test_humanoid_state.py
=====================================================
"""

import unittest

from ea_avs_mvp_v7.human.action_state import ActionState
from ea_avs_mvp_v7.human.human_state import HumanState
from ea_avs_mvp_v7.human.humanoid_agent import KEYPOINT_LINK_MAP, resolve_humanoid_urdf_path


class TestHumanoidState(unittest.TestCase):
    """Humanoid 状态与 3D 骨架真值测试。"""

    def test_keypoint_link_map_completeness(self):
        """验证标准 16 关键点映射完整性。"""
        self.assertGreaterEqual(len(KEYPOINT_LINK_MAP), 15)
        for expected in ("pelvis", "head", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            self.assertIn(expected, KEYPOINT_LINK_MAP)

    def test_human_state_full_fields(self):
        """验证 HumanState 的所有必需字段。"""
        gt_joints = {
            "pelvis": [0.0, 0.9, 0.0],
            "head": [0.0, 1.7, 0.0],
            "neck": [0.0, 1.5, 0.0],
            "left_shoulder": [-0.2, 1.4, 0.0],
            "right_shoulder": [0.2, 1.4, 0.0],
        }
        hs = HumanState(
            position=[0.0, 0.1, 0.0],
            orientation_yaw_rad=0.0,
            frame_id=0,
            timestamp=0.0,
            action_class="fall_related",
            action_label="fall down",
            joint_positions_3d_world=gt_joints,
        )
        self.assertEqual(hs.num_gt_joints, 5)
        d = hs.to_dict()
        self.assertEqual(d["action_class"], "fall_related")
        self.assertEqual(d["position"], [0.0, 0.1, 0.0])
        self.assertEqual(d["joint_positions_3d_world"]["head"], [0.0, 1.7, 0.0])

    def test_urdf_asset_resolution(self):
        """验证 Humanoid URDF 资产解析路径。"""
        urdf_p, motion_p = resolve_humanoid_urdf_path({"avatar_name": "neutral_0"})
        self.assertTrue(urdf_p.exists())
        self.assertEqual(urdf_p.name, "neutral_0.urdf")


if __name__ == "__main__":
    unittest.main()
