"""
Human 模块单元测试 —— test_human.py
===================================
"""

import unittest

from ea_avs_mvp_v7.human.action_state import ActionState
from ea_avs_mvp_v7.human.human_state import HumanState
from ea_avs_mvp_v7.human.humanoid_agent import resolve_humanoid_urdf_path


class TestHuman(unittest.TestCase):
    """Humanoid 状态与动作标注测试。"""

    def test_action_state_serialization(self):
        act = ActionState(
            action_class="fall_related",
            raw_label="tripping over",
            proc_label="fall down",
            babel_sid=3522,
            start_frame=0,
            end_frame=120,
        )
        d = act.to_dict()
        self.assertEqual(d["action_class"], "fall_related")
        self.assertEqual(d["babel_sid"], 3522)
        self.assertEqual(d["start_frame"], 0)

    def test_human_state_serialization(self):
        hs = HumanState(
            position=[1.0, 0.0, -1.0],
            orientation_yaw_rad=1.57,
            frame_id=5,
            timestamp=5.0 / 30.0,
            action_class="standing",
            action_label="stand naturally",
            joint_positions_3d_world={
                "pelvis": [1.0, 0.9, -1.0],
                "head": [1.0, 1.7, -1.0],
            },
        )
        d = hs.to_dict()
        self.assertEqual(d["frame_id"], 5)
        self.assertEqual(len(d["joint_positions_3d_world"]), 2)
        self.assertAlmostEqual(d["position"][0], 1.0)

    def test_resolve_humanoid_urdf(self):
        urdf_path, motion_path = resolve_humanoid_urdf_path({"avatar_name": "neutral_0"})
        self.assertTrue(urdf_path.exists())
        self.assertEqual(urdf_path.name, "neutral_0.urdf")


if __name__ == "__main__":
    unittest.main()
