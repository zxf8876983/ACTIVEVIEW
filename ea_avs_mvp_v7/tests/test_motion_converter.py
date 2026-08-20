"""
Motion Converter 模块单元测试 —— test_motion_converter.py
=========================================================
"""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.motion.joint_mapping import (
    SMPLX_JOINT_NAMES,
    SMPLX_RODRIGUES_DIM,
    HABITAT_HUMANOID_QUAT_DIM,
    get_joint_index,
    get_joint_slice,
    validate_motion_quaternions,
)
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer


class TestMotionConverter(unittest.TestCase):
    """关节映射与四元数转换校验测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_joint_hierarchy_and_slices(self):
        """测试 SMPL-X 关节序号与 162 维切片。"""
        self.assertEqual(len(SMPLX_JOINT_NAMES), 55)
        self.assertEqual(SMPLX_RODRIGUES_DIM, 162)
        self.assertEqual(HABITAT_HUMANOID_QUAT_DIM, 216)

        idx_hip = get_joint_index("left_hip")
        self.assertEqual(idx_hip, 1)
        slice_hip = get_joint_slice(idx_hip)
        self.assertEqual(slice_hip, slice(0, 3))

        idx_thumb = get_joint_index("right_thumb3")
        self.assertEqual(idx_thumb, 54)
        slice_thumb = get_joint_slice(idx_thumb)
        self.assertEqual(slice_thumb, slice(159, 162))

    def test_quaternion_validation(self):
        """测试 216 维四元数模长校验器。"""
        num_frames = 5
        # 构造有效单位四元数 [0, 0, 0, 1]
        valid_quats = np.zeros((num_frames, HABITAT_HUMANOID_QUAT_DIM), dtype=np.float32)
        valid_quats[:, 3::4] = 1.0
        self.assertTrue(validate_motion_quaternions(valid_quats))

        # 构造错误维度
        with self.assertRaises(ValueError):
            validate_motion_quaternions(np.zeros((num_frames, 100), dtype=np.float32))

    def test_motion_player_playback(self):
        """测试转换后 PKL 的 MotionPlayer 回放逻辑。"""
        num_frames = 10
        joints = np.zeros((num_frames, 216), dtype=np.float32)
        joints[:, 3::4] = 1.0
        transforms = np.zeros((num_frames, 4, 4), dtype=np.float32)
        transforms[:, 0, 0] = 1.0
        transforms[:, 1, 1] = 1.0
        transforms[:, 2, 2] = 1.0
        transforms[:, 3, 3] = 1.0

        pkl_path = Path(self.tmp_dir) / "test_motion.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "pose_motion": {
                    "joints_array": joints,
                    "transform_array": transforms,
                    "fps": 30.0,
                },
                "metadata": {
                    "target_class": "fall_related",
                    "proc_label": "fall to ground",
                    "babel_sid": 3522,
                }
            }, f)

        player = MotionPlayer(pkl_path)
        self.assertEqual(player.total_frames, 10)
        self.assertEqual(player.action_class, "fall_related")
        self.assertAlmostEqual(player.duration_seconds, 10.0 / 30.0)

        j, t = player.step(advance=True)
        self.assertEqual(j.shape, (216,))
        self.assertEqual(t.shape, (4, 4))
        self.assertEqual(player.current_frame, 1)


if __name__ == "__main__":
    unittest.main()
