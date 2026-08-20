"""
Action Metrics 单元测试 —— test_action_metrics.py
=================================================
"""

import unittest
import numpy as np

from ea_avs_mvp_v7.human.action_metrics import ActionMotionMetrics, compute_action_motion_metrics


class TestActionMetrics(unittest.TestCase):
    """动力学运动评价与静止/跌倒区分测试。"""

    def test_static_standing_action_metrics(self):
        """测试静态站立：极小位移与倾角，判定为 static (dynamic_motion=False)。"""
        frames_gt = []
        for _ in range(10):
            frames_gt.append({
                "pelvis": [0.0, 0.9, 0.0],
                "head": [0.0, 1.7, 0.0],
                "neck": [0.0, 1.5, 0.0],
                "left_hip": [-0.1, 0.9, 0.0],
                "right_hip": [0.1, 0.9, 0.0],
            })

        metrics = compute_action_motion_metrics(frames_gt)
        self.assertIsInstance(metrics, ActionMotionMetrics)
        self.assertAlmostEqual(metrics.height_change, 0.0, places=3)
        self.assertAlmostEqual(metrics.pelvis_velocity, 0.0, places=3)
        self.assertAlmostEqual(metrics.joint_motion_energy, 0.0, places=3)
        self.assertAlmostEqual(metrics.torso_angle_change, 0.0, places=1)
        self.assertFalse(metrics.dynamic_motion)

    def test_dynamic_fall_action_metrics(self):
        """测试跌倒动作：高度急剧下降、躯干倾斜，判定为 dynamic (dynamic_motion=True)。"""
        frames_gt = []
        # 10 帧由站立到平躺在地面
        for i in range(10):
            alpha = i / 9.0
            pelvis_y = 0.9 * (1.0 - alpha) + 0.1 * alpha
            neck_y = 1.5 * (1.0 - alpha) + 0.1 * alpha
            neck_z = 0.6 * alpha
            frames_gt.append({
                "pelvis": [0.0, float(pelvis_y), 0.0],
                "neck": [0.0, float(neck_y), float(neck_z)],
                "left_hip": [-0.1, float(pelvis_y), 0.0],
                "right_hip": [0.1, float(pelvis_y), 0.0],
            })

        timestamps = [i * 0.1 for i in range(10)]
        metrics = compute_action_motion_metrics(frames_gt, timestamps)

        self.assertGreater(metrics.height_change, 0.5)
        self.assertGreater(metrics.torso_angle_change, 40.0)
        self.assertGreater(metrics.pelvis_velocity, 0.5)
        self.assertGreater(metrics.joint_motion_energy, 0.5)
        self.assertTrue(metrics.dynamic_motion)


if __name__ == "__main__":
    unittest.main()
