"""
Unit test for Root Normalization precision and stability.
Verifies that root position after normalization is strictly (0, 0, 0) with error < 1e-6.
"""

import unittest
import numpy as np

from ea_avs_mvp_v11.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer


class TestRootNormalization(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()
        self.normalizer = SkeletonNormalizer(skel_def=self.skel_def)

    def test_root_is_hip_center_and_zero_origin(self):
        # 创建一个随机偏移且具有 33 关键点的骨架
        num_joints = 33
        raw_joints = np.random.uniform(-2.0, 2.0, size=(num_joints, 3)).astype(np.float32)
        # 显式设定左右髋部位置
        raw_joints[23] = np.array([0.15, -0.40, 1.80], dtype=np.float32)  # left_hip
        raw_joints[24] = np.array([-0.15, -0.40, 1.90], dtype=np.float32) # right_hip
        raw_joints[11] = np.array([0.20, 0.20, 1.80], dtype=np.float32)   # left_shoulder
        raw_joints[12] = np.array([-0.20, 0.20, 1.90], dtype=np.float32)  # right_shoulder

        confs = np.full(num_joints, 0.95, dtype=np.float32)

        skel = EstimatedSkeleton3D(
            joint_format="mediapipe_33",
            joints_3d_camera=raw_joints,
            joints_3d_world=raw_joints,
            joints_2d=np.zeros((num_joints, 2), dtype=np.float32),
            perception_confidence=confs,
            uncertainty_mask=np.zeros(num_joints, dtype=bool),
            part_confidence={"torso": 0.95},
            joint_names=self.skel_def.joint_names,
        )

        norm_skel = self.normalizer.normalize(skel)

        # 检查 normalized joints 存在
        self.assertIsNotNone(norm_skel.joints_3d_normalized)
        norm_j = norm_skel.joints_3d_normalized

        # 验证根节点 (left_hip + right_hip) / 2 在归一化后严格对齐原点 (0, 0, 0)
        norm_root = (norm_j[23] + norm_j[24]) / 2.0
        root_error = float(np.linalg.norm(norm_root))

        self.assertLess(root_error, 1e-6, f"Root position {norm_root} error {root_error} exceeds 1e-6")

    def test_prohibit_single_hip_as_root(self):
        # 确认左髋和右髋在归一化后关于原点对称
        raw_joints = np.zeros((33, 3), dtype=np.float32)
        raw_joints[23] = np.array([0.10, -0.30, 2.00], dtype=np.float32)  # left_hip
        raw_joints[24] = np.array([-0.10, -0.30, 2.00], dtype=np.float32) # right_hip
        raw_joints[11] = np.array([0.18, 0.25, 2.00], dtype=np.float32)   # left_shoulder
        raw_joints[12] = np.array([-0.18, 0.25, 2.00], dtype=np.float32)  # right_shoulder

        skel = EstimatedSkeleton3D(
            joint_format="mediapipe_33",
            joints_3d_camera=raw_joints,
            joints_3d_world=raw_joints,
            joints_2d=np.zeros((33, 2), dtype=np.float32),
            perception_confidence=np.ones(33, dtype=np.float32),
            uncertainty_mask=np.zeros(33, dtype=bool),
            part_confidence={"torso": 1.0},
            joint_names=self.skel_def.joint_names,
        )

        norm_skel = self.normalizer.normalize(skel)
        norm_j = norm_skel.joints_3d_normalized

        # 左髋和右髋向量之和应为 0
        hip_sum = norm_j[23] + norm_j[24]
        self.assertLess(float(np.linalg.norm(hip_sum)), 1e-6)


if __name__ == "__main__":
    unittest.main()
