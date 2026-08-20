"""
Humanoid Agent 单元测试 —— test_humanoid_agent.py
=================================================
"""

import unittest

from ea_avs_mvp_v7.human.action_state import ActionState
from ea_avs_mvp_v7.human.human_state import HumanState
from ea_avs_mvp_v7.human.humanoid_agent import (
    KEYPOINT_LINK_MAP,
    resolve_humanoid_urdf_path,
)


class TestHumanoidAgent(unittest.TestCase):
    """Humanoid 代理与 3D 关节真值提取测试。"""

    def test_keypoint_map_coverage(self):
        """验证 16 核心关节定义完备性。"""
        self.assertGreaterEqual(len(KEYPOINT_LINK_MAP), 15)
        for expected in ("pelvis", "head", "neck", "left_shoulder", "right_shoulder", "left_knee", "right_knee"):
            self.assertIn(expected, KEYPOINT_LINK_MAP)

    def test_human_state_integrity(self):
        """验证 HumanState 数据结构完整性。"""
        joints = {
            "pelvis": [0.0, 0.85, 0.0],
            "head": [0.0, 1.68, 0.0],
            "neck": [0.0, 1.48, 0.0],
            "left_shoulder": [-0.18, 1.38, 0.0],
            "right_shoulder": [0.18, 1.38, 0.0],
        }
        hs = HumanState(
            position=[0.0, 0.1, 0.0],
            orientation_yaw_rad=0.0,
            frame_id=0,
            timestamp=0.0,
            action_class="fall_related",
            action_label="fall down",
            joint_positions_3d_world=joints,
        )
        self.assertEqual(hs.num_gt_joints, 5)
        d = hs.to_dict()
        self.assertEqual(d["action_class"], "fall_related")
        self.assertEqual(d["position"], [0.0, 0.1, 0.0])
        self.assertIn("pelvis", d["joint_positions_3d_world"])

    def test_urdf_path_resolution(self):
        """验证 Humanoid URDF 与运动数据路径可解析性。"""
        urdf_p, motion_p = resolve_humanoid_urdf_path({"avatar_name": "neutral_0"})
        self.assertTrue(urdf_p.exists())
        self.assertEqual(urdf_p.name, "neutral_0.urdf")


if __name__ == "__main__":
    unittest.main()
