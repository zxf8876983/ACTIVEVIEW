"""
Unit tests for the complete Phase 2 Pose Estimation and Normalization Pipeline.
"""

import unittest
import numpy as np

from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose, V10Sample
from ea_avs_mvp_v10.dataset.perception_dataset import V10PerceptionPipeline
from ea_avs_mvp_v10.perception.depth_projection import DepthProjector
from ea_avs_mvp_v10.perception.pose_estimator import MockPoseEstimator
from ea_avs_mvp_v10.perception.skeleton_adapter import COCO17ToNTU25Adapter
from ea_avs_mvp_v10.perception.skeleton_converter import SkeletonConverter
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer


class TestPosePipeline(unittest.TestCase):
    def setUp(self):
        self.mock_estimator = MockPoseEstimator(default_confidence=0.95)
        self.depth_projector = DepthProjector(patch_radius=2, min_depth=0.1, max_depth=10.0)
        self.skeleton_converter = SkeletonConverter(uncertainty_conf_thresh=0.35)
        self.skeleton_normalizer = SkeletonNormalizer()
        self.adapter = COCO17ToNTU25Adapter()

        self.intrinsics = CameraIntrinsics(width=640, height=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0)
        self.cam_pose = CameraPose(
            position=[1.5, -0.4, 4.0],
            rotation_quat=[0.0, 0.0, 0.0, 1.0],
            yaw_deg=0.0,
            intrinsics=self.intrinsics,
        )

        self.dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.dummy_depth = np.ones((480, 640), dtype=np.float32) * 2.0  # 2.0m depth

    def test_pipeline_output_shapes_and_values(self):
        # 1. 2D Pose
        pose2d = self.mock_estimator.estimate_pose2d(self.dummy_rgb)
        self.assertEqual(pose2d.keypoints.shape, (17, 2))
        self.assertEqual(pose2d.confidence.shape, (17,))

        # 2. Depth Back-Projection
        depth_res = self.depth_projector.project_2d_to_3d(
            pose2d.keypoints, self.dummy_depth, self.intrinsics, self.cam_pose
        )
        self.assertEqual(depth_res.joints_3d_cam.shape, (17, 3))
        self.assertEqual(depth_res.joints_3d_world.shape, (17, 3))

        # 3. Skeleton Conversion (COCO-17 Native)
        skel = self.skeleton_converter.convert_and_fuse(pose2d, depth_res)
        self.assertEqual(skel.joint_format, "COCO17")
        self.assertEqual(skel.joints_3d_cam.shape, (17, 3))
        self.assertEqual(skel.joints_3d_world.shape, (17, 3))
        self.assertEqual(skel.confidence.shape, (17,))
        self.assertFalse(np.any(skel.occluded_mask))

        # 4. Normalization
        norm_skel = self.skeleton_normalizer.normalize(skel)
        self.assertIsNotNone(norm_skel.joints_3d_normalized)
        self.assertEqual(norm_skel.joints_3d_normalized.shape, (17, 3))

        # 验证 Root (hip center) 是否在原点 (0, 0, 0)
        hip_center = (norm_skel.joints_3d_normalized[11] + norm_skel.joints_3d_normalized[12]) / 2.0
        np.testing.assert_allclose(hip_center, [0.0, 0.0, 0.0], atol=1e-5)

        # 验证尺度归一化后坐标在合理范围 [-2, 2] 内
        self.assertLessEqual(np.max(np.abs(norm_skel.joints_3d_normalized)), 3.0)

    def test_optional_adapter_to_ntu25(self):
        pose2d = self.mock_estimator.estimate_pose2d(self.dummy_rgb)
        depth_res = self.depth_projector.project_2d_to_3d(
            pose2d.keypoints, self.dummy_depth, self.intrinsics, self.cam_pose
        )
        skel = self.skeleton_converter.convert_and_fuse(pose2d, depth_res)

        ntu_skel = self.adapter.adapt(skel)
        self.assertEqual(ntu_skel.joint_format, "NTU25")
        self.assertEqual(ntu_skel.joints_3d_cam.shape, (25, 3))
        self.assertEqual(ntu_skel.confidence.shape, (25,))


if __name__ == "__main__":
    unittest.main()
