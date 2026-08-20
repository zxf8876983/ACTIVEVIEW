"""
Robot 模块单元测试 —— test_robot.py
===================================
"""

import math
import unittest

from ea_avs_mvp_v7.robot.robot_agent import RobotAgent
from ea_avs_mvp_v7.robot.rgbd_sensor import RGBDSensor


class TestRobot(unittest.TestCase):
    """RobotAgent 与 RGBDSensor 单元测试。"""

    def test_sensor_intrinsics_calculation(self):
        sensor_cfg = {
            "width": 640,
            "height": 480,
            "hfov_deg": 90.0,
        }
        sensor = RGBDSensor(sim=None, sensor_cfg=sensor_cfg)
        intrinsics = sensor.intrinsics

        # 对于 90 度 HFOV，fx = width / (2 * tan(45°)) = 640 / 2 = 320.0
        self.assertAlmostEqual(intrinsics["fx"], 320.0)
        self.assertAlmostEqual(intrinsics["fy"], 320.0)
        self.assertAlmostEqual(intrinsics["cx"], 320.0)
        self.assertAlmostEqual(intrinsics["cy"], 240.0)
        self.assertEqual(intrinsics["width"], 640.0)
        self.assertEqual(intrinsics["height"], 480.0)

    def test_robot_agent_state(self):
        robot = RobotAgent(sim=None)
        robot.set_pose([1.5, 0.0, 3.0], yaw_deg=90.0)
        self.assertEqual(list(robot.position), [1.5, 0.0, 3.0])
        self.assertEqual(robot.yaw_deg, 90.0)


if __name__ == "__main__":
    unittest.main()
