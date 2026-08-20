"""
Humanoid Playback 单元测试 —— test_humanoid_playback.py
======================================================
"""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.human.human_state import HumanState
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer


class TestHumanoidPlayback(unittest.TestCase):
    """动作回放控制器与人体状态测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.pkl_path = Path(self.tmp_dir) / "playback_test.pkl"

        num_frames = 20
        joints = np.zeros((num_frames, 216), dtype=np.float32)
        joints[:, 3::4] = 1.0
        transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_frames, axis=0)

        with open(self.pkl_path, "wb") as f:
            pickle.dump({
                "pose_motion": {
                    "joints_array": joints,
                    "transform_array": transforms,
                    "fps": 30.0,
                },
                "metadata": {
                    "target_class": "fall_related",
                    "proc_label": "fall to the ground",
                    "babel_sid": 3522,
                }
            }, f)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_motion_player_stepping_and_seeking(self):
        """测试播放器单步前进与跳帧。"""
        player = MotionPlayer(self.pkl_path, playback_fps=30.0)
        self.assertEqual(player.total_frames, 20)
        self.assertEqual(player.current_frame, 0)

        j, t = player.step(advance=True)
        self.assertEqual(player.current_frame, 1)
        self.assertEqual(j.shape, (216,))
        self.assertEqual(t.shape, (4, 4))

        new_frame_idx = player.seek(10)
        self.assertEqual(new_frame_idx, 10)
        self.assertEqual(player.current_frame, 10)
        pose_info = player.get_current_pose()
        self.assertAlmostEqual(pose_info["timestamp"], 10.0 / 30.0)

    def test_human_state_generation(self):
        """测试 HumanState 对象构建与序列化。"""
        state = HumanState(
            position=[0.0, 0.1, 0.0],
            orientation_yaw_rad=0.0,
            frame_id=10,
            timestamp=0.333,
            action_class="fall_related",
            action_label="fall to the ground",
            joint_positions_3d_world={"pelvis": [0.0, 0.9, 0.0]},
        )
        self.assertEqual(state.frame_id, 10)
        self.assertEqual(state.action_class, "fall_related")
        self.assertEqual(state.num_gt_joints, 1)


if __name__ == "__main__":
    unittest.main()
