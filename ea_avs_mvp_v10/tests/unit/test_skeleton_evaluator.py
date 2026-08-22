"""
Unit test for Offline GT Skeleton Evaluator & Alignment.
Verifies MPJPE, PA-MPJPE, PCK@threshold calculation, and Procrustes analysis.
"""

import unittest
import numpy as np

from ea_avs_mvp_v10.evaluation.skeleton_alignment import (
    compute_procrustes_alignment,
    extract_aligned_joint_pairs,
    transform_gt_to_camera_frame,
)
from ea_avs_mvp_v10.evaluation.skeleton_evaluator import EvaluationMetrics, SkeletonEvaluator
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition


class TestSkeletonEvaluator(unittest.TestCase):
    def setUp(self):
        self.skel_def = get_skeleton_definition()
        self.evaluator = SkeletonEvaluator(skel_def=self.skel_def)

        # 构造合成的 GT 字典与估计 3D 骨架
        self.gt_dict = {
            "head": [0.0, 1.6, 2.0],
            "neck": [0.0, 1.4, 2.0],
            "left_shoulder": [0.2, 1.3, 2.0],
            "right_shoulder": [-0.2, 1.3, 2.0],
            "left_elbow": [0.25, 1.0, 2.0],
            "right_elbow": [-0.25, 1.0, 2.0],
            "left_wrist": [0.25, 0.7, 2.0],
            "right_wrist": [-0.25, 0.7, 2.0],
            "left_hip": [0.1, 0.8, 2.0],
            "right_hip": [-0.1, 0.8, 2.0],
            "left_knee": [0.1, 0.4, 2.0],
            "right_knee": [-0.1, 0.4, 2.0],
            "left_ankle": [0.1, 0.05, 2.0],
            "right_ankle": [-0.1, 0.05, 2.0],
            "pelvis": [0.0, 0.8, 2.0],
            "spine3": [0.0, 1.1, 2.0],
        }

        # 构造估计骨架 (带有小幅度高斯噪声，模拟现实感知误差 ~5cm)
        num_joints = 33
        est_cam = np.zeros((num_joints, 3), dtype=np.float32)
        est_cam[:, 2] = 2.0

        est_cam[0] = [0.02, 0.82, 2.0]     # head (rel to pelvis: (0.02, 0.82, 0.0))
        est_cam[11] = [0.21, 0.51, 2.0]   # left_shoulder
        est_cam[12] = [-0.19, 0.51, 2.0]  # right_shoulder
        est_cam[13] = [0.26, 0.21, 2.0]   # left_elbow
        est_cam[14] = [-0.24, 0.21, 2.0]  # right_elbow
        est_cam[15] = [0.26, -0.09, 2.0]  # left_wrist
        est_cam[16] = [-0.24, -0.09, 2.0] # right_wrist
        est_cam[23] = [0.10, 0.00, 2.0]   # left_hip
        est_cam[24] = [-0.10, 0.00, 2.0]  # right_hip
        est_cam[25] = [0.11, -0.39, 2.0]  # left_knee
        est_cam[26] = [-0.09, -0.39, 2.0] # right_knee
        est_cam[27] = [0.11, -0.74, 2.0]  # left_ankle
        est_cam[28] = [-0.09, -0.74, 2.0] # right_ankle

        self.skel = EstimatedSkeleton3D(
            joint_format="mediapipe_33",
            joints_3d_camera=est_cam,
            joints_3d_world=est_cam,
            joints_2d=np.zeros((num_joints, 2), dtype=np.float32),
            perception_confidence=np.ones(num_joints, dtype=np.float32),
            uncertainty_mask=np.zeros(num_joints, dtype=bool),
            part_confidence={"torso": 1.0},
            joint_names=self.skel_def.joint_names,
        )

        self.c2w_identity = np.eye(4, dtype=np.float32)

    def test_procrustes_alignment(self):
        pts1 = np.random.uniform(-1, 1, size=(10, 3)).astype(np.float32)
        # 旋转并平移
        theta = np.pi / 4.0
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        t_true = np.array([0.5, -0.2, 1.0], dtype=np.float32)
        pts2 = (pts1 @ R_true.T) + t_true

        aligned, R_est, s, t_est = compute_procrustes_alignment(pts1, pts2)
        err = np.linalg.norm(aligned - pts2)
        self.assertLess(err, 1e-4)

    def test_evaluate_sample(self):
        metrics = self.evaluator.evaluate_sample(
            estimated_skeleton=self.skel,
            gt_joints_dict=self.gt_dict,
            camera_matrix_4x4=self.c2w_identity,
            sample_id="test_001",
            action_label="standing",
        )

        self.assertEqual(metrics.sample_id, "test_001")
        self.assertGreater(metrics.num_evaluated_joints, 10)
        self.assertGreater(metrics.abs_mpjpe_mm, 0.0)
        self.assertGreater(metrics.mpjpe_mm, 0.0)
        self.assertLess(metrics.mpjpe_mm, 150.0) # 合成误差应在 150mm 内
        self.assertGreaterEqual(metrics.pck_5cm, 0.0)
        self.assertGreaterEqual(metrics.pck_10cm, 0.5)


if __name__ == "__main__":
    unittest.main()
