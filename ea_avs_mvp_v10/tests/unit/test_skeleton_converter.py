"""
Unit tests for COCO-17 skeleton converter and confidence fusion.
"""

import unittest
import numpy as np

from ea_avs_mvp_v10.perception.depth_projection import DepthProjectionResult
from ea_avs_mvp_v10.perception.pose_estimator import COCO_KEYPOINTS, Pose2DResult
from ea_avs_mvp_v10.perception.skeleton_converter import (
    COCO_BODY_PART_GROUPS,
    EstimatedSkeleton3D,
    SkeletonConverter,
)


class TestSkeletonConverter(unittest.TestCase):
    def setUp(self):
        self.converter = SkeletonConverter(uncertainty_conf_thresh=0.35)

    def test_convert_and_fuse_structure(self):
        # 17 个 COCO 关键点
        kpts_2d = np.ones((17, 2), dtype=np.float32) * 100.0
        conf_2d = np.ones(17, dtype=np.float32) * 0.9
        pose2d = Pose2DResult(keypoints=kpts_2d, confidence=conf_2d)

        joints_cam = np.ones((17, 3), dtype=np.float32) * 2.0
        joints_world = joints_cam.copy()
        depth_vals = np.ones(17, dtype=np.float32) * 2.0
        depth_conf = np.ones(17, dtype=np.float32) * 0.8
        valid_mask = np.ones(17, dtype=bool)

        depth_res = DepthProjectionResult(
            joints_3d_cam=joints_cam,
            joints_3d_world=joints_world,
            depth_values=depth_vals,
            depth_confidence=depth_conf,
            valid_mask=valid_mask,
        )

        skel = self.converter.convert_and_fuse(pose2d, depth_res)
        self.assertIsInstance(skel, EstimatedSkeleton3D)
        self.assertEqual(skel.joint_format, "COCO17")
        self.assertEqual(skel.joints_3d_cam.shape, (17, 3))
        self.assertEqual(skel.confidence.shape, (17,))
        self.assertEqual(len(skel.joint_names), 17)

        # 检查复合置信度: 0.9 * 0.8 = 0.72
        self.assertAlmostEqual(skel.confidence[0], 0.72, places=4)
        self.assertFalse(skel.occluded_mask[0])

        # 检查部位平均置信度
        for part in COCO_BODY_PART_GROUPS.keys():
            self.assertIn(part, skel.part_confidence)
            self.assertGreater(skel.part_confidence[part], 0.5)

    def test_uncertainty_flagging(self):
        # 模拟低置信度关键点
        kpts_2d = np.ones((17, 2), dtype=np.float32) * 100.0
        conf_2d = np.zeros(17, dtype=np.float32)  # 0 置信度
        pose2d = Pose2DResult(keypoints=kpts_2d, confidence=conf_2d)

        depth_res = DepthProjectionResult(
            joints_3d_cam=np.zeros((17, 3), dtype=np.float32),
            joints_3d_world=np.zeros((17, 3), dtype=np.float32),
            depth_values=np.zeros(17, dtype=np.float32),
            depth_confidence=np.zeros(17, dtype=np.float32),
            valid_mask=np.zeros(17, dtype=bool),
        )

        skel = self.converter.convert_and_fuse(pose2d, depth_res)
        self.assertTrue(np.all(skel.occluded_mask))
        self.assertEqual(skel.part_confidence["head"], 0.0)


if __name__ == "__main__":
    unittest.main()
