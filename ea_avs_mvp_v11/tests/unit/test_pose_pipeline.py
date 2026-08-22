"""
Unit tests for RGB-D Skeleton Extractor, Normalizer, and Adapters.
"""

import unittest
import numpy as np

from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v11.perception.coordinate_validator import CoordinateValidator
from ea_avs_mvp_v11.perception.rgbd_skeleton_extractor import (
    MockRGBDSkeletonExtractor,
    RGBDSkeletonExtractor,
)
from ea_avs_mvp_v11.perception.skeleton_adapter import (
    MediaPipe33ToCOCO17Adapter,
    MediaPipe33ToNTU25Adapter,
)
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer


class TestPosePipeline(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()
        self.extractor = MockRGBDSkeletonExtractor(default_confidence=0.95, skel_def=self.skel_def)
        self.normalizer = SkeletonNormalizer(skel_def=self.skel_def)
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
        self.assertEqual(skel.joint_format, "mediapipe_33")
        self.assertEqual(skel.joints_3d_camera.shape, (33, 3))
        self.assertEqual(len(skel.perception_confidence), 33)
        self.assertTrue(np.all(skel.perception_confidence >= 0.9))

    def test_normalization(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        norm_skel = self.normalizer.normalize(skel)

        self.assertIsNotNone(norm_skel.joints_3d_normalized)
        self.assertEqual(norm_skel.joints_3d_normalized.shape, (33, 3))
        # 验证根节点 (hip_center) 归一化后接近 (0,0,0)
        hip_center = np.mean(norm_skel.joints_3d_normalized[[23, 24]], axis=0)
        self.assertTrue(np.allclose(hip_center, [0.0, 0.0, 0.0], atol=1e-3))

    def test_adapter_coco17(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        coco_skel = self.coco_adapter.adapt(skel)

        self.assertEqual(coco_skel.joint_format, "COCO17")
        self.assertEqual(coco_skel.joints_3d_camera.shape, (17, 3))
        self.assertEqual(len(coco_skel.perception_confidence), 17)

    def test_adapter_ntu25(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        ntu_skel = self.ntu_adapter.adapt(skel)

        self.assertEqual(ntu_skel.joint_format, "NTU25")
        self.assertEqual(ntu_skel.joints_3d_camera.shape, (25, 3))
        self.assertEqual(len(ntu_skel.perception_confidence), 25)

    def test_validator(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        res = self.validator.validate(skel)
        self.assertTrue(res.is_valid)


if __name__ == "__main__":
    unittest.main()
