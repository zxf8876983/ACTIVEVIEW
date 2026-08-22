"""
Unit tests for 2D pose estimation wrappers.
"""

import unittest
import numpy as np
from PIL import Image

from ea_avs_mvp_v11.perception.pose_estimator import (
    COCO_KEYPOINTS,
    COCO_SKELETON_PAIRS,
    MockPoseEstimator,
    Pose2DResult,
)


class TestPoseEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = MockPoseEstimator(default_confidence=0.9)
        self.dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_mock_estimator_shapes_and_values(self):
        res = self.estimator.estimate_pose2d(self.dummy_img)
        self.assertIsInstance(res, Pose2DResult)
        self.assertEqual(res.keypoints.shape, (17, 2))
        self.assertEqual(res.confidence.shape, (17,))
        self.assertEqual(len(res.joint_names), 17)
        self.assertAlmostEqual(res.confidence[0], 0.9)
        self.assertGreater(res.person_score, 0.5)

    def test_pose2d_serialization(self):
        res = self.estimator.estimate_pose2d(self.dummy_img)
        d = res.to_dict()
        self.assertIn("keypoints", d)
        self.assertIn("confidence", d)
        self.assertIn("joint_names", d)

        reloaded = Pose2DResult.from_dict(d)
        np.testing.assert_allclose(res.keypoints, reloaded.keypoints)
        np.testing.assert_allclose(res.confidence, reloaded.confidence)
        self.assertEqual(res.joint_names, reloaded.joint_names)


if __name__ == "__main__":
    unittest.main()
