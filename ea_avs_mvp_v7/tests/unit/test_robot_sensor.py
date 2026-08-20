"""
Robot & Sensor 单元测试 —— test_robot_sensor.py
===============================================
"""

import unittest
import numpy as np

from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor


class TestRobotSensor(unittest.TestCase):
    """机器人相机内参与几何测试。"""

    def test_camera_intrinsics_calculation(self):
        cfg = {
            "width": 640,
            "height": 480,
            "hfov_deg": 90.0,
            "camera_height": 1.2,
        }
        intrinsics = RGBDSensor.compute_intrinsics_from_config(cfg)
        self.assertEqual(intrinsics["width"], 640)
        self.assertEqual(intrinsics["height"], 480)
        self.assertAlmostEqual(intrinsics["cx"], 320.0)
        self.assertAlmostEqual(intrinsics["cy"], 240.0)
        self.assertAlmostEqual(intrinsics["fx"], 320.0)
        self.assertAlmostEqual(intrinsics["fy"], 320.0)


if __name__ == "__main__":
    unittest.main()
