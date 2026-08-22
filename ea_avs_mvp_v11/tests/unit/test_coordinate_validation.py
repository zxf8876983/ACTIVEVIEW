"""
Unit test for Camera Coordinate System convention and vertical kinematic ordering.
Verifies +X: right, +Y: up, +Z: forward and head_y > hip_y > ankle_y for upright standing pose.
"""

import unittest
import numpy as np

from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose
from ea_avs_mvp_v11.perception.coordinate_validator import CoordinateValidator
from ea_avs_mvp_v11.perception.rgbd_skeleton_extractor import MockRGBDSkeletonExtractor
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition


class TestCoordinateValidation(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()
        self.extractor = MockRGBDSkeletonExtractor(default_confidence=0.95, skel_def=self.skel_def)
        self.validator = CoordinateValidator()

        self.intrinsics = CameraIntrinsics(640, 480, 320.0, 320.0, 320.0, 240.0)
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

    def test_coordinate_system_metadata(self):
        # 验证定义中的坐标系规范
        coord = self.skel_def.coordinate_system
        self.assertEqual(coord["name"], "camera_frame_right_hand")
        self.assertIn("right", coord["x_axis"].lower())
        self.assertIn("up", coord["y_axis"].lower())
        self.assertIn("forward", coord["z_axis"].lower())
        self.assertEqual(coord["unit"], "meter")

    def test_vertical_kinematics_head_hip_ankle(self):
        # 验证提取的 3D 骨架中头部在上方、髋部居中、脚踝在下方
        dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_depth = np.ones((480, 640), dtype=np.float32) * 2.0
        skel = self.extractor.extract(dummy_rgb, dummy_depth, self.cam_pose)

        j_cam = skel.joints_3d_camera
        head_y = j_cam[0, 1]                    # nose / head
        hip_y = (j_cam[23, 1] + j_cam[24, 1]) / 2.0  # hip center
        ankle_y = (j_cam[27, 1] + j_cam[28, 1]) / 2.0 # ankle center

        self.assertGreater(head_y, hip_y, f"Head Y ({head_y:.2f}) must be higher than Hip Y ({hip_y:.2f})")
        self.assertGreater(hip_y, ankle_y, f"Hip Y ({hip_y:.2f}) must be higher than Ankle Y ({ankle_y:.2f})")

        # 检查 Validator 校验通过
        res = self.validator.validate(skel)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.depth_valid)
        self.assertTrue(res.height_valid)
        self.assertTrue(res.kinematics_valid)


if __name__ == "__main__":
    unittest.main()
