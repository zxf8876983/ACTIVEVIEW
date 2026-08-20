"""
Motion 模块单元测试 —— test_motion.py
=====================================
"""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.motion.amass_loader import AMASSLoader, NormalizedMotion, SMPLX_JOINT_DIM
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer


class TestMotion(unittest.TestCase):
    """动作加载与播放器测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_schema_a_loading(self):
        num_frames = 15
        npz_file = Path(self.tmp_dir) / "motion_schema_a.npz"
        np.savez(
            npz_file,
            trans=np.ones((num_frames, 3), dtype=np.float32),
            root_orient=np.full((num_frames, 3), 0.5, dtype=np.float32),
            poses=np.zeros((num_frames, 156), dtype=np.float32),
            mocap_framerate=np.array([60.0]),
        )

        motion = AMASSLoader.load(npz_file, start_frame=2, end_frame=10)
        self.assertIsInstance(motion, NormalizedMotion)
        self.assertEqual(motion.num_frames, 9)
        self.assertEqual(motion.fps, 60.0)
        self.assertEqual(motion.translation.shape, (9, 3))
        self.assertEqual(motion.root_rotation.shape, (9, 3))
        self.assertEqual(motion.body_pose.shape, (9, SMPLX_JOINT_DIM))
        self.assertAlmostEqual(motion.root_rotation[0, 0], 0.5)

    def test_schema_b_loading(self):
        num_frames = 20
        npz_file = Path(self.tmp_dir) / "motion_schema_b.npz"
        poses = np.zeros((num_frames, 156), dtype=np.float32)
        poses[:, :3] = 0.8  # Schema B: root orient in poses[:, :3]
        np.savez(
            npz_file,
            trans=np.zeros((num_frames, 3), dtype=np.float32),
            poses=poses,
            fps=30.0,
        )

        motion = AMASSLoader.load(npz_file)
        self.assertEqual(motion.num_frames, 20)
        self.assertAlmostEqual(motion.root_rotation[0, 0], 0.8)
        self.assertEqual(motion.body_pose.shape, (20, SMPLX_JOINT_DIM))

    def test_motion_player(self):
        num_frames = 10
        joints = np.zeros((num_frames, 216), dtype=np.float32)
        transforms = np.zeros((num_frames, 4, 4), dtype=np.float32)
        transforms[:, 0, 0] = 1.0
        transforms[:, 1, 1] = 1.0
        transforms[:, 2, 2] = 1.0
        transforms[:, 3, 3] = 1.0

        pkl_path = Path(self.tmp_dir) / "test_player.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "pose_motion": {
                    "joints_array": joints,
                    "transform_array": transforms,
                    "fps": 30.0,
                },
                "metadata": {
                    "target_class": "fall_related",
                    "proc_label": "fall down",
                    "babel_sid": 794,
                }
            }, f)

        player = MotionPlayer(pkl_path)
        self.assertEqual(player.total_frames, 10)
        self.assertEqual(player.action_class, "fall_related")
        self.assertEqual(player.action_label, "fall down")
        self.assertFalse(player.is_finished())

        j0, t0 = player.step(advance=True)
        self.assertEqual(j0.shape, (216,))
        self.assertEqual(t0.shape, (4, 4))
        self.assertEqual(player.current_frame, 1)

        player.seek(8)
        self.assertEqual(player.current_frame, 8)
        player.reset()
        self.assertEqual(player.current_frame, 0)


if __name__ == "__main__":
    unittest.main()
