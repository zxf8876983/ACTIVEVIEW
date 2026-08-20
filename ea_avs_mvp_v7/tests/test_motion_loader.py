"""
AMASS Loader 模块单元测试 —— test_motion_loader.py
===================================================
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.motion.amass_loader import AMASSLoader, NormalizedMotion, SMPLX_JOINT_DIM


class TestMotionLoader(unittest.TestCase):
    """AMASS 动作读取与标准化测试 (Schema A / Schema B)。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_schema_a_reading(self):
        """测试显式 root_orient 的 Schema A 格式读取。"""
        num_frames = 20
        npz_file = Path(self.tmp_dir) / "motion_a.npz"
        np.savez(
            npz_file,
            trans=np.ones((num_frames, 3), dtype=np.float32),
            root_orient=np.full((num_frames, 3), 0.25, dtype=np.float32),
            poses=np.zeros((num_frames, 156), dtype=np.float32),
            fps=60.0,
        )

        motion = AMASSLoader.load(npz_file, start_frame=5, end_frame=15)
        self.assertIsInstance(motion, NormalizedMotion)
        self.assertEqual(motion.num_frames, 11)
        self.assertEqual(motion.fps, 60.0)
        self.assertEqual(motion.translation.shape, (11, 3))
        self.assertEqual(motion.root_rotation.shape, (11, 3))
        self.assertEqual(motion.body_pose.shape, (11, SMPLX_JOINT_DIM))
        self.assertAlmostEqual(float(motion.root_rotation[0, 0]), 0.25)

    def test_schema_b_reading(self):
        """测试 root_orient 包含在 poses[:, :3] 的 Schema B 格式读取。"""
        num_frames = 10
        npz_file = Path(self.tmp_dir) / "motion_b.npz"
        poses = np.zeros((num_frames, 156), dtype=np.float32)
        poses[:, :3] = 0.75
        np.savez(
            npz_file,
            trans=np.zeros((num_frames, 3), dtype=np.float32),
            poses=poses,
            mocap_framerate=np.array([30.0]),
        )

        motion = AMASSLoader.load(npz_file)
        self.assertEqual(motion.num_frames, 10)
        self.assertEqual(motion.fps, 30.0)
        self.assertAlmostEqual(float(motion.root_rotation[0, 0]), 0.75)
        self.assertEqual(motion.body_pose.shape, (10, SMPLX_JOINT_DIM))

    def test_missing_file_raises_error(self):
        """测试文件缺失时抛出 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            AMASSLoader.load("/nonexistent/path/to/motion.npz")


if __name__ == "__main__":
    unittest.main()
