"""
Unit tests for coordinate system freeze, depth back-projection, normalization, and validation.
"""

import unittest
import numpy as np

from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v10.perception.coordinate_validator import CoordinateValidator, ValidationResult
from ea_avs_mvp_v10.perception.depth_projection import DepthProjector
from ea_avs_mvp_v10.perception.pose_estimator import MockPoseEstimator
from ea_avs_mvp_v10.perception.skeleton_converter import SkeletonConverter
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer


class TestCoordinateSystem(unittest.TestCase):
    def setUp(self):
        self.mock_estimator = MockPoseEstimator(default_confidence=0.95)
        self.depth_projector = DepthProjector(patch_radius=2, min_depth=0.1, max_depth=10.0)
        self.skeleton_converter = SkeletonConverter(uncertainty_conf_thresh=0.35)
        self.skeleton_normalizer = SkeletonNormalizer()
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
        self.dummy_depth = np.ones((480, 640), dtype=np.float32) * 2.0  # 2.0m depth

    def test_camera_coordinate_back_projection(self):
        # 验证逆投影几何公式:
        # 中心像素 (320, 240) 深度 2.0m -> X=0, Y=0, Z=2.0
        kpts_2d = np.array([[320.0, 240.0], [480.0, 120.0]], dtype=np.float32)
        depth_map = np.ones((480, 640), dtype=np.float32) * 2.0

        res = self.depth_projector.project_2d_to_3d(kpts_2d, depth_map, self.intrinsics)
        # 中心点
        np.testing.assert_allclose(res.joints_3d_cam[0], [0.0, 0.0, 2.0], atol=1e-5)
        # 右上方点 (u=480 > cx, v=120 < cy)
        # X = (480-320)*2/320 = 1.0 (右)
        # Y = (120-240)*2/320 = -0.75 (上/下几何按标准针孔定义)
        # Z = 2.0
        self.assertAlmostEqual(res.joints_3d_cam[1, 0], 1.0, places=4)
        self.assertAlmostEqual(res.joints_3d_cam[1, 2], 2.0, places=4)

    def test_camera_to_world_transform(self):
        pose2d = self.mock_estimator.estimate_pose2d(self.dummy_rgb)
        depth_res = self.depth_projector.project_2d_to_3d(
            pose2d.keypoints, self.dummy_depth, self.intrinsics, self.cam_pose
        )
        # 验证世界坐标偏移: X_w = X_c + 1.5, Y_w = Y_c - 0.4, Z_w = Z_c + 4.0
        np.testing.assert_allclose(
            depth_res.joints_3d_world[:, 0], depth_res.joints_3d_cam[:, 0] + 1.5, atol=1e-4
        )
        np.testing.assert_allclose(
            depth_res.joints_3d_world[:, 1], depth_res.joints_3d_cam[:, 1] - 0.4, atol=1e-4
        )
        np.testing.assert_allclose(
            depth_res.joints_3d_world[:, 2], depth_res.joints_3d_cam[:, 2] + 4.0, atol=1e-4
        )

    def test_normalization_and_validation(self):
        pose2d = self.mock_estimator.estimate_pose2d(self.dummy_rgb)
        depth_res = self.depth_projector.project_2d_to_3d(
            pose2d.keypoints, self.dummy_depth, self.intrinsics, self.cam_pose
        )
        skel = self.skeleton_converter.convert_and_fuse(pose2d, depth_res)
        norm_skel = self.skeleton_normalizer.normalize(skel)

        # 校验合法骨架
        val_res = self.validator.validate(norm_skel)
        self.assertIn(val_res.status, ["VALID", "WARNING"])
        self.assertTrue(val_res.depth_valid)

    def test_abnormal_coordinate_validation(self):
        # 构造异常过近/过远的骨架
        pose2d = self.mock_estimator.estimate_pose2d(self.dummy_rgb)
        depth_res = self.depth_projector.project_2d_to_3d(
            pose2d.keypoints, self.dummy_depth, self.intrinsics, self.cam_pose
        )
        skel = self.skeleton_converter.convert_and_fuse(pose2d, depth_res)

        # 强制将深度修改为 15.0m (超出 8.0m 阈值)
        skel.joints_3d_camera[:, 2] = 15.0
        val_res = self.validator.validate(skel)
        self.assertEqual(val_res.status, "INVALID")
        self.assertFalse(val_res.depth_valid)


if __name__ == "__main__":
    unittest.main()
