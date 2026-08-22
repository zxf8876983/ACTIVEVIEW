"""
Unit tests for coordinate system, normalization, and validation (MediaPipe-33 / COCO-17).
"""

import unittest
import numpy as np

from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v11.perception.coordinate_validator import CoordinateValidator, ValidationResult
from ea_avs_mvp_v11.perception.rgbd_skeleton_extractor import MockRGBDSkeletonExtractor
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer


class TestCoordinateSystem(unittest.TestCase):
    def setUp(self):
        self.extractor = MockRGBDSkeletonExtractor(default_confidence=0.95)
        self.normalizer = SkeletonNormalizer()
        self.validator = CoordinateValidator()

        self.intrinsics = CameraIntrinsics(width=640, height=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0)
        self.cam_pose = CameraPose(
            position=[1.5, -0.4, 4.0],
            rotation_quat=[0.0, 0.0, 0.0, 1.0],
            yaw_deg=0.0,
            intrinsics=self.intrinsics,
            matrix_4x4=[
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 1.0, 0.0, -0.4],
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

        self.dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_depth = np.ones((480, 640), dtype=np.float32) * 2.0

    def test_camera_to_world_transform(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        # 验证世界坐标变换: X_w = X_c + 1.5, Y_w = Y_c - 0.4, Z_w = Z_c + 4.0
        np.testing.assert_allclose(
            skel.joints_3d_world[:, 0], skel.joints_3d_camera[:, 0] + 1.5, atol=1e-4
        )
        np.testing.assert_allclose(
            skel.joints_3d_world[:, 1], skel.joints_3d_camera[:, 1] - 0.4, atol=1e-4
        )
        np.testing.assert_allclose(
            skel.joints_3d_world[:, 2], skel.joints_3d_camera[:, 2] + 4.0, atol=1e-4
        )

    def test_normalization_and_validation(self):
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        norm_skel = self.normalizer.normalize(skel)

        # 校验合法骨架
        val_res = self.validator.validate(norm_skel)
        self.assertIn(val_res.status, ["VALID", "WARNING"])
        self.assertTrue(val_res.depth_valid)

    def test_abnormal_coordinate_validation(self):
        # 构造异常过远骨架
        skel = self.extractor.extract(self.dummy_rgb, self.dummy_depth, self.cam_pose)
        skel.joints_3d_camera[:, 2] = 15.0  # 超出 8.0m 阈值
        val_res = self.validator.validate(skel)
        self.assertEqual(val_res.status, "INVALID")
        self.assertFalse(val_res.depth_valid)


if __name__ == "__main__":
    unittest.main()
