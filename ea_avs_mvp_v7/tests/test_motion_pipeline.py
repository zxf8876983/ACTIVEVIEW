"""
Motion Pipeline 转换与维度测试 —— test_motion_pipeline.py
=========================================================
"""

import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ea_avs_mvp_v7.motion.amass_loader import AMASSLoader, NormalizedMotion
from ea_avs_mvp_v7.motion.joint_mapping import (
    HABITAT_HUMANOID_QUAT_DIM,
    SMPLX_RODRIGUES_DIM,
    validate_habitat_motion_dict,
    validate_motion_quaternions,
)


class TestMotionPipeline(unittest.TestCase):
    """AMASS 动作读取、162 维规范化与 Habitat 格式转换校验测试。"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_normalized_motion_creation(self):
        """测试 NormalizedMotion 数据结构与切片校验。"""
        num_frames = 15
        motion = NormalizedMotion(
            num_frames=num_frames,
            translation=np.zeros((num_frames, 3), dtype=np.float32),
            root_rotation=np.zeros((num_frames, 3), dtype=np.float32),
            body_pose=np.zeros((num_frames, SMPLX_RODRIGUES_DIM), dtype=np.float32),
            fps=30.0,
            metadata={"action": "standing"},
        )
        self.assertEqual(motion.num_frames, num_frames)
        self.assertEqual(motion.body_pose.shape[1], 162)
        self.assertEqual(motion.fps, 30.0)

    def test_habitat_motion_dict_validation(self):
        """测试 Habitat 动作字典的格式与四元数验证。"""
        num_frames = 8
        joints = np.zeros((num_frames, HABITAT_HUMANOID_QUAT_DIM), dtype=np.float32)
        joints[:, 3::4] = 1.0  # 单位四元数
        transforms = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_frames, axis=0)

        motion_dict = {
            "pose_motion": {
                "joints_array": joints,
                "transform_array": transforms,
                "fps": 30.0,
            },
            "metadata": {"babel_sid": 100},
        }

        stats = validate_habitat_motion_dict(motion_dict)
        self.assertEqual(stats["num_frames"], 8)
        self.assertEqual(stats["joints_dim"], 216)
        self.assertEqual(stats["transforms_shape"], [8, 4, 4])


if __name__ == "__main__":
    unittest.main()
