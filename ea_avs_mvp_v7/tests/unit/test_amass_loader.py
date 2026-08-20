"""
AMASS Loader 单元测试 —— test_amass_loader.py
=============================================
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.motion.amass_loader import AMASSLoader, NormalizedMotion, SMPLX_JOINT_DIM


class TestAMASSLoader(unittest.TestCase):
    """AMASS 动作加载与 Schema A/B 兼容性测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_schema_a_loading(self):
        """测试显式 root_orient 的 Schema A 格式。"""
        num_frames = 15
        npz_file = Path(self.tmp_dir) / "motion_a.npz"
        np.savez(
            npz_file,
            trans=np.ones((num_frames, 3), dtype=np.float32),
            root_orient=np.full((num_frames, 3), 0.5, dtype=np.float32),
            poses=np.zeros((num_frames, 156), dtype=np.float32),
            fps=60.0,
        )

        motion = AMASSLoader.load(npz_file, start_frame=2, end_frame=10)
        self.assertIsInstance(motion, NormalizedMotion)
        self.assertEqual(motion.num_frames, 9)
        self.assertEqual(motion.fps, 60.0)
        self.assertEqual(motion.body_pose.shape, (9, SMPLX_JOINT_DIM))
        self.assertAlmostEqual(float(motion.root_rotation[0, 0]), 0.5)

    def test_schema_b_loading(self):
        """测试 root_orient 位于 poses 前 3 维的 Schema B 格式。"""
        num_frames = 12
        npz_file = Path(self.tmp_dir) / "motion_b.npz"
        poses = np.zeros((num_frames, 156), dtype=np.float32)
        poses[:, :3] = 0.8
        np.savez(
            npz_file,
            trans=np.zeros((num_frames, 3), dtype=np.float32),
            poses=poses,
            mocap_framerate=np.array([120.0]),
        )

        motion = AMASSLoader.load(npz_file)
        self.assertEqual(motion.num_frames, 12)
        self.assertEqual(motion.fps, 120.0)
        self.assertAlmostEqual(float(motion.root_rotation[0, 0]), 0.8)
        self.assertEqual(motion.body_pose.shape, (12, SMPLX_JOINT_DIM))

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            AMASSLoader.load("/nonexistent/file.npz")


if __name__ == "__main__":
    unittest.main()
