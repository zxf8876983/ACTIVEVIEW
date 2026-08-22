"""
Unit test for Pose3DEstimator interface and MediaPipe/Mock implementations.
"""

import unittest
import numpy as np
from PIL import Image

from ea_avs_mvp_v10.perception.pose3d_estimator import (
    Mock3DPoseEstimator,
    Pose3DEstimationResult,
    create_pose3d_estimator,
)
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition


class TestPose3DEstimator(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()
        self.mock_estimator = Mock3DPoseEstimator(skel_def=self.skel_def)

    def test_mock_estimator_frame(self):
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.mock_estimator.estimate_frame(dummy_rgb)

        self.assertIsInstance(res, Pose3DEstimationResult)
        self.assertEqual(res.joints.shape, (33, 3))
        self.assertEqual(len(res.joint_names), 33)
        self.assertEqual(res.confidence.shape, (33,))
        self.assertEqual(res.coordinate_system, "camera_frame_right_hand")
        self.assertEqual(res.estimator, "mock_33")

    def test_mock_estimator_sequence(self):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        skeletons, confs = self.mock_estimator.estimate_sequence(frames)

        self.assertEqual(skeletons.shape, (5, 33, 3))
        self.assertEqual(confs.shape, (5, 33))

    def test_factory_method(self):
        est = create_pose3d_estimator("mock", skel_def=self.skel_def)
        self.assertIsInstance(est, Mock3DPoseEstimator)


if __name__ == "__main__":
    unittest.main()
