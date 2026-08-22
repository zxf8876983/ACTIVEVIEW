"""
Unit tests for RGB-D Skeleton Extractor, Normalizer, and Adapters.
"""

import unittest
import numpy as np

from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v10.perception.coordinate_validator import CoordinateValidator
from ea_avs_mvp_v10.perception.rgbd_skeleton_extractor import (
    MEDIAPIPE_33_KEYPOINTS,
    MEDIAPIPE_33_SKELETON_PAIRS,
    MockRGBDSkeletonExtractor,
    RGBDSkeletonExtractor,
)
from ea_avs_mvp_v10.perception.skeleton_adapter import (
    MediaPipe33ToCOCO17Adapter,
    MediaPipe33ToNTU25Adapter,
)
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer


class TestPosePipeline(unittest.TestCase):
    def setUp(self):
        self.extractor = MockRGBDSkeletonExtractor(default_confidence=0.95)
        self.normalizer = SkeletonNormalizer()
        self.validator = CoordinateValidator()
        self.coco_adapter = MediaPipe33ToCOCO17Adapter()
        self.ntu_adapter = MediaPipe33ToNTU25Adapter()

        self.intrinsics = CameraIntrinsics(width=640, height=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0)
        self.cam_pose = CameraPose(
            position=[0.0, 0.0, 0.0],
            rotation_quat=[0.0, 0.0, 0.0, 1.0],
            yaw_deg=0.0,
            intrinsics=self.intrinsics,
            matrix_4x4=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

        self.dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_depth = np.ones((480, 640), dtype=np.float32) * 2.0

    def test_rgbd_extractor_mock(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        self.assertEqual(skel.joint_format, "MediaPipe33")
        self.assertEqual(skel.joints_3d_camera.shape, (33, 3))
        self.assertEqual(len(skel.perception_confidence), 33)
        self.assertTrue(np.all(skel.perception_confidence >= 0.9))

    def test_normalization(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        norm_skel = self.normalizer.normalize(skel)

        self.assertIsNotNone(norm_skel.joints_3d_normalized)
        self.assertEqual(norm_skel.joints_3d_normalized.shape, (33, 3))

        # 骨盆中心应在原点 (0, 0, 0)
        hip_center = (norm_skel.joints_3d_normalized[23] + norm_skel.joints_3d_normalized[24]) / 2.0
        np.testing.assert_allclose(hip_center, [0.0, 0.0, 0.0], atol=1e-5)

    def test_adapter_to_coco17(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        coco_skel = self.coco_adapter.adapt(skel)

        self.assertEqual(coco_skel.joint_format, "COCO17")
        self.assertEqual(coco_skel.joints_3d_camera.shape, (17, 3))
        self.assertEqual(len(coco_skel.perception_confidence), 17)

    def test_adapter_to_ntu25(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        ntu_skel = self.ntu_adapter.adapt(skel)

        self.assertEqual(ntu_skel.joint_format, "NTU25")
        self.assertEqual(ntu_skel.joints_3d_camera.shape, (25, 3))

    def test_coordinate_validator(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        val_res = self.validator.validate(skel)
        self.assertEqual(val_res.status, "VALID")
        self.assertTrue(val_res.depth_valid)
        self.assertTrue(val_res.height_valid)


if __name__ == "__main__":
    unittest.main()
