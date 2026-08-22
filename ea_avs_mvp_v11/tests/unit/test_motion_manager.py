"""
Unit tests for v10 MotionManager.
"""

import unittest
from pathlib import Path

from ea_avs_mvp_v11.core.types import ActionClassV10
from ea_avs_mvp_v11.motion.motion_manager import MotionManager


class TestV10MotionManager(unittest.TestCase):
    def setUp(self):
        self.mgr = MotionManager()

    def test_scan_and_index_motions(self):
        motions = self.mgr.list_available_motions()
        self.assertGreater(len(motions), 0)

        # 校验 6 大动作类别均能成功检索到对应资产
        for act_class in ActionClassV10:
            m_list = self.mgr.get_motions_by_class(act_class)
            self.assertGreater(
                len(m_list), 0, f"No motions found for action class: {act_class.value}"
            )

    def test_instantiate_motion_player(self):
        motions = self.mgr.list_available_motions()
        first_m = motions[0]
        player = self.mgr.get_motion_player(first_m)
        self.assertGreater(player.total_frames, 0)
        self.assertEqual(player.current_frame, 0)

        # 测试步进
        joints, root_trans = player.step(advance=True)
        self.assertEqual(joints.shape, (216,))
        self.assertEqual(root_trans.shape, (4, 4))
        self.assertEqual(player.current_frame, 1)


if __name__ == "__main__":
    unittest.main()
