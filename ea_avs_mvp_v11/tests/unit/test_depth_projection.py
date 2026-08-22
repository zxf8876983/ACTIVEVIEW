"""
Unit tests for depth back projection.
"""

import unittest
import numpy as np

from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v11.perception.depth_projection import DepthProjectionResult, DepthProjector


class TestDepthProjection(unittest.TestCase):
    def setUp(self):
        self.projector = DepthProjector(patch_radius=2, min_depth=0.1, max_depth=10.0)
        self.intrinsics = CameraIntrinsics(width=640, height=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0)

    def test_back_projection_geometry(self):
        # 构造一个中心点 (320, 240)，深度为 2.0m 的测试用例
        kpts_2d = np.array([[320.0, 240.0]], dtype=np.float32)
        depth_map = np.ones((480, 640), dtype=np.float32) * 2.0

        res = self.projector.project_2d_to_3d(kpts_2d, depth_map, self.intrinsics)
        self.assertIsInstance(res, DepthProjectionResult)
        self.assertEqual(res.joints_3d_cam.shape, (1, 3))
        # (320-320)*2/320 = 0.0, (240-240)*2/320 = 0.0, Z = 2.0
        np.testing.assert_allclose(res.joints_3d_cam[0], [0.0, 0.0, 2.0], atol=1e-5)
        self.assertTrue(res.valid_mask[0])
        self.assertGreater(res.depth_confidence[0], 0.8)

    def test_off_center_back_projection(self):
        # 偏离中心: u=480, v=120, Z=2.0
        # X = (480-320)*2.0/320 = 1.0m
        # Y = (120-240)*2.0/320 = -0.75m
        # Z = 2.0m
        kpts_2d = np.array([[480.0, 120.0]], dtype=np.float32)
        depth_map = np.ones((480, 640), dtype=np.float32) * 2.0

        res = self.projector.project_2d_to_3d(kpts_2d, depth_map, self.intrinsics)
        np.testing.assert_allclose(res.joints_3d_cam[0], [1.0, -0.75, 2.0], atol=1e-5)

    def test_invalid_depth_handling(self):
        # 深度为 0 或超出范围
        kpts_2d = np.array([[100.0, 100.0]], dtype=np.float32)
        depth_map = np.zeros((480, 640), dtype=np.float32)

        res = self.projector.project_2d_to_3d(kpts_2d, depth_map, self.intrinsics)
        self.assertFalse(res.valid_mask[0])
        self.assertEqual(res.depth_confidence[0], 0.0)
        np.testing.assert_allclose(res.joints_3d_cam[0], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
