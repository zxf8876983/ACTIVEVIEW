"""
Unit Tests for Motion Processing Pipeline
"""

import unittest
import numpy as np

from ea_avs_mvp_v7.motion.joint_mapping import (
    HABITAT_HUMANOID_QUAT_DIM,
    validate_motion_quaternions,
)
from ea_avs_mvp_v7.motion.amass_loader import NormalizedMotion, SMPLX_JOINT_DIM
from ea_avs_mvp_v7.human.action_metrics import compute_action_motion_metrics


class TestMotionPipeline(unittest.TestCase):
    def test_motion_quaternion_validation(self):
        valid_quats = np.zeros((5, HABITAT_HUMANOID_QUAT_DIM), dtype=np.float32)
        valid_quats[:, 3::4] = 1.0  # Unit quaternions [0, 0, 0, 1]
        # Should not raise
        validate_motion_quaternions(valid_quats)

    def test_normalized_motion_structure(self):
        num_frames = 10
        motion = NormalizedMotion(
            num_frames=num_frames,
            fps=30.0,
            translation=np.zeros((num_frames, 3), dtype=np.float32),
            root_rotation=np.zeros((num_frames, 3), dtype=np.float32),
            body_pose=np.zeros((num_frames, SMPLX_JOINT_DIM), dtype=np.float32),
            metadata={"action_class": "standing", "action_label": "stand"},
        )
        self.assertEqual(motion.num_frames, 10)
        self.assertEqual(motion.metadata["action_class"], "standing")
        self.assertEqual(motion.body_pose.shape, (10, 162))

    def test_motion_metrics_dynamic_flag(self):
        # Static standing sequence
        static_seq = [
            {"pelvis": [1.5, -0.80, 4.0], "head": [1.5, -0.05, 4.0]}
            for _ in range(10)
        ]
        times = [i * 0.1 for i in range(10)]
        m_static = compute_action_motion_metrics(static_seq, times)
        self.assertFalse(m_static.dynamic_motion)
        self.assertAlmostEqual(m_static.height_change, 0.0, places=3)

        # Dynamic falling sequence
        falling_seq = [
            {"pelvis": [1.5, -0.80 - i * 0.08, 4.0], "head": [1.5, -0.05 - i * 0.12, 4.0]}
            for i in range(10)
        ]
        m_fall = compute_action_motion_metrics(falling_seq, times)
        self.assertTrue(m_fall.dynamic_motion)
        self.assertGreater(m_fall.height_change, 0.5)


if __name__ == "__main__":
    unittest.main()
