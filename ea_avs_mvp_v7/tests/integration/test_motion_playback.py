"""
Motion Playback 集成测试 —— test_motion_playback.py
===================================================
"""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.motion.motion_player import MotionPlayer


class TestMotionPlayback(unittest.TestCase):
    """动作文件回放控制器集成测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.pkl_path = Path(self.tmp_dir) / "playback_int_test.pkl"

        num_frames = 10
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

    def test_motion_playback_lifecycle(self):
        player = MotionPlayer(self.pkl_path)
        self.assertEqual(player.total_frames, 10)
        self.assertEqual(player.action_class, "fall_related")

        # 步进测试
        for i in range(10):
            self.assertEqual(player.current_frame, i)
            player.step(advance=True)

        self.assertTrue(player.is_finished())
        player.reset()
        self.assertEqual(player.current_frame, 0)


if __name__ == "__main__":
    unittest.main()
