"""
Motion Converter 单元测试 —— test_motion_converter.py
=====================================================
"""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.motion.joint_mapping import (
    HABITAT_HUMANOID_QUAT_DIM,
    SMPLX_JOINT_NAMES,
    SMPLX_RODRIGUES_DIM,
    get_joint_index,
    get_joint_slice,
    validate_habitat_motion_dict,
    validate_motion_quaternions,
)
from ea_avs_mvp_v7.motion.motion_player import MotionPlayer


class TestMotionConverter(unittest.TestCase):
    """关节映射与动作格式转换单元测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_smplx_joint_names_and_indices(self):
        self.assertEqual(len(SMPLX_JOINT_NAMES), 55)
        self.assertEqual(SMPLX_RODRIGUES_DIM, 162)
        self.assertEqual(HABITAT_HUMANOID_QUAT_DIM, 216)

        self.assertEqual(get_joint_index("left_hip"), 1)
        self.assertEqual(get_joint_slice(1), slice(0, 3))

    def test_validate_habitat_motion_dict(self):
        num_frames = 10
        joints = np.zeros((num_frames, HABITAT_HUMANOID_QUAT_DIM), dtype=np.float32)
        joints[:, 3::4] = 1.0
        transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_frames, axis=0)

        data = {
            "pose_motion": {
                "joints_array": joints,
                "transform_array": transforms,
                "fps": 30.0,
            },
            "metadata": {"babel_sid": 123},
        }

        stats = validate_habitat_motion_dict(data)
        self.assertEqual(stats["num_frames"], 10)
        self.assertEqual(stats["joints_dim"], 216)


if __name__ == "__main__":
    unittest.main()
