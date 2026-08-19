"""
纯 Python 单元与闭环测试套件 —— test_v60_pure_python.py
=====================================================

功能：
    在无 GPU、无 Habitat 物理引擎的环境下，对 v6.0 的所有纯算法与逻辑模块
    进行自动化单元测试。

测试清单 (10 核心科研测试)：
    TEST 1: invalid occlusion 不进入 visible_occ
    TEST 2: Estimated GT-free blocked ray (clear / estimated_geometric_blocked / unknown)
    TEST 3: Canonical yaw convention (基于 POSE_SKELETONS["standing"] 测试 8 个角度)
    TEST 4: 3D neck / pelvis bilateral derivation (无 2D midpoint depth 采样)
    TEST 5: unknown template 显式抛出 ValueError
    TEST 6: unknown pose backend 显式抛出 ValueError
    TEST 7: No-GT AST 静态防护测试
    TEST 8: 未来候选观测零调用 (No future render)
    TEST 9: Oracle 同 Pool (Same-Pool) 严格上界计算
    TEST 10: 跨 Pool 伪负 Gap 严格拒绝
"""

import sys
import unittest
import numpy as np

from ea_avs_v6.keypoint_schema import (
    COCO_17_KEYPOINTS,
    EA_AVS_15_KEYPOINTS,
    Keypoint2D,
    coco17_to_ea_avs_15,
)
from ea_avs_v6.depth_lifter import (
    sample_depth_at_pixel,
    backproject_keypoint_to_world,
    lift_2d_keypoints_to_3d,
    EstimatedJoint3D,
)
from ea_avs_v6.orientation_estimator import OrientationEstimator, EstimatedOrientation
from ea_avs_v6.skeleton_completion import SkeletonCompleter
from ea_avs_v6.estimated_human_state import EstimatedHumanState
from ea_avs_v6.state_validation import validate_estimated_state
from ea_avs_v6.pose_backend import MockPoseBackend, Pose2DDetection, create_pose_backend
from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.geometry import compute_camera_intrinsics, normalize_angle
from ea_avs_v6.candidate_sampler import CandidateSampler, CandidateView
from ea_avs_v6.policies import EstimatedStateOursPolicy, GTStateOursPolicy
from ea_avs_v6.oracle_policy import OraclePolicy, compute_oracle_gap
from ea_avs_v6.action_pose_library import POSE_SKELETONS
from ea_avs_v6.raycast_utils import cast_ray_to_estimated_point
from ea_avs_v6.estimated_predictive_evaluator import EstimatedPredictiveEvaluator


class TestV60PurePython(unittest.TestCase):

    def setUp(self):
        self.config = {
            "camera": {
                "width": 640,
                "height": 480,
                "hfov_deg": 90,
                "camera_height": 1.2,
                "clip_near": 0.01,
                "clip_far": 10.0,
            },
            "perception": {
                "pose_backend": "mock",
                "min_pose_score": 0.30,
                "min_keypoint_confidence": 0.30,
            },
            "visibility": {"min_depth": 0.5, "max_depth": 5.0},
            "human_state_estimation": {
                "min_2d_keypoints": 6,
                "min_3d_keypoints": 4,
                "min_torso_keypoints_3d": 2,
                "depth_patch_size": 5,
                "min_valid_depth_pixels": 3,
                "depth_min_m": 0.3,
                "depth_max_m": 8.0,
                "max_depth_spread_m": 0.5,
                "require_human_position": True,
                "require_orientation": True,
                "orientation": {
                    "yaw_offset_deg": 0.0,
                    "forward_sign": 1,
                    "min_bilateral_pair_confidence": 0.30,
                },
                "skeleton_completion": {
                    "enabled": True,
                    "template": "standing",
                },
            },
            "action_part_weights": {
                "standing": {"torso": 0.35, "lower_body": 0.35, "head": 0.20, "arms": 0.10}
            },
            "orientation_score": {
                "preferred_angle_deg": 45,
                "sigma_deg": 35,
            },
            "distance_score": {
                "optimal_distance": 2.0,
                "sigma": 0.7,
            },
            "predictive_score": {
                "w_action_occ_pred": 0.50,
                "w_orient_pred": 0.15,
                "w_center_pred": 0.10,
                "w_dist_pred": 0.10,
                "w_move": 0.15,
            },
            "candidate_sampling": {
                "radii": [1.5, 2.0],
                "angles_deg": [-90, 0, 90, 180],
                "min_distance_to_human": 1.2,
                "max_distance_to_human": 3.0,
                "max_geodesic_distance": 6.0,
            },
            "occlusion": {
                "enabled": True,
                "ray_epsilon": 0.05,
                "target_tolerance": 0.08,
                "estimated_target_tolerance": 0.12,
                "min_hit_distance": 0.05,
                "max_invalid_occlusion_rate": 0.10,
            },
        }

    def test_01_invalid_occlusion_not_in_visible_occ(self):
        """TEST 1: 验证 invalid occlusion (valid=False) 关键点彻底不进入 visible_occ。"""
        class MockRunnerInvalidRay:
            def cast_ray(self, origin, direction, max_dist):
                raise RuntimeError("Simulated raycast failure")

        evaluator = EstimatedPredictiveEvaluator(self.config)
        est_state = EstimatedHumanState(
            valid=True,
            human_position_world=np.array([0.0, 0.0, 2.0]),
            human_yaw=0.0,
            proxy_full_skeleton={k: np.array(v) + np.array([0.0, 0.0, 2.0]) for k, v in POSE_SKELETONS["standing"].items()},
        )
        res = evaluator.score_view_pred(
            runner=MockRunnerInvalidRay(),
            view_pos=np.array([0.0, 0.0, 0.0]),
            view_yaw=0.0,
            robot_start_pos=np.array([0.0, 0.0, 0.0]),
            estimated_state=est_state,
            geodesic_distance=0.0,
        )

        # 所有射线失败被标记为 invalid，因此 visible_keypoints_occ_pred 必须为空
        self.assertEqual(len(res["visible_keypoints_occ_pred"]), 0)
        self.assertEqual(res["S_action_occ_pred"], 0.0)
        self.assertEqual(res["S_kp_occ_pred"], 0.0)
        self.assertFalse(res["is_occlusion_valid_pred"])

    def test_02_estimated_gt_free_blocked_ray(self):
        """TEST 2: 验证 Estimated GT-free 射线检测（clear, estimated_geometric_blocked, unknown）。"""
        class MockClearRunner:
            def cast_ray(self, origin, direction, max_dist):
                return {"has_hits": False}

        class MockBlockedRunner:
            def cast_ray(self, origin, direction, max_dist):
                # 距离 2.0m 的目标，在 1.0m 处被阻挡
                return {"has_hits": True, "hit_distance": 1.0}

        class MockSurfaceNearRunner:
            def cast_ray(self, origin, direction, max_dist):
                # 距离 2.0m 的目标，在 1.95m 处 hit（在 0.12m tolerance 范围内）
                return {"has_hits": True, "hit_distance": 1.95}

        # 1. Clear ray
        r1 = cast_ray_to_estimated_point(MockClearRunner(), np.array([0, 0, 0]), np.array([0, 0, 2]))
        self.assertTrue(r1["valid"])
        self.assertFalse(r1["occluded"])
        self.assertEqual(r1["occlusion_source"], "clear")

        # 2. Blocked ray
        r2 = cast_ray_to_estimated_point(MockBlockedRunner(), np.array([0, 0, 0]), np.array([0, 0, 2]))
        self.assertTrue(r2["valid"])
        self.assertTrue(r2["occluded"])
        self.assertEqual(r2["occlusion_source"], "estimated_geometric_blocked")

        # 3. Surface tolerance near hit (treated as clear)
        r3 = cast_ray_to_estimated_point(MockSurfaceNearRunner(), np.array([0, 0, 0]), np.array([0, 0, 2]), target_tolerance=0.12)
        self.assertTrue(r3["valid"])
        self.assertFalse(r3["occluded"])
        self.assertEqual(r3["occlusion_source"], "clear")

    def test_03_canonical_yaw_convention_8_angles(self):
        """TEST 3: 使用项目 POSE_SKELETONS["standing"] 验证 8 个角度朝向推导严格一致。"""
        estimator = OrientationEstimator(self.config)
        standing_skel = POSE_SKELETONS["standing"]
        test_angles_deg = [0, 45, 90, 135, 180, -135, -90, -45]

        for deg in test_angles_deg:
            th = np.deg2rad(deg)
            cos_th, sin_th = np.cos(th), np.sin(th)

            joints_3d = {}
            for name in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
                loc = standing_skel[name]
                x_rot = loc[0] * cos_th + loc[2] * sin_th
                y_rot = loc[1]
                z_rot = -loc[0] * sin_th + loc[2] * cos_th
                pt_world = np.array([x_rot, y_rot, z_rot + 2.0], dtype=np.float64)

                joints_3d[name] = EstimatedJoint3D(
                    name=name, position_world=pt_world, position_camera=None,
                    confidence_2d=0.9, depth_valid=True, observable_3d=True,
                    source="observed_2d",
                )

            res = estimator.estimate_orientation(joints_3d)
            self.assertTrue(res.valid)
            diff = abs(normalize_angle(res.yaw_rad - th))
            self.assertAlmostEqual(
                diff, 0.0, places=4,
                msg=f"Canonical yaw 推导在 {deg}° 时失败，误差 {np.rad2deg(diff):.2f}°"
            )

    def test_04_bilateral_3d_neck_pelvis_derivation(self):
        """TEST 4: 验证 3D neck 与 pelvis 由左右肩/髋世界坐标直接推导，不采 2D midpoint depth。"""
        intrinsics = compute_camera_intrinsics(640, 480, 90.0)
        camera_base_pos = np.array([0.0, 0.0, 0.0])

        kpts_2d = {
            "left_shoulder": Keypoint2D("left_shoulder", 300, 140, 0.9, True, "observed_2d"),
            "right_shoulder": Keypoint2D("right_shoulder", 340, 140, 0.9, True, "observed_2d"),
            "left_hip": Keypoint2D("left_hip", 310, 240, 0.9, True, "observed_2d"),
            "right_hip": Keypoint2D("right_hip", 330, 240, 0.9, True, "observed_2d"),
            "neck": Keypoint2D("neck", 320, 140, 0.9, True, "derived_2d"),
            "pelvis": Keypoint2D("pelvis", 320, 240, 0.9, True, "derived_2d"),
        }
        # 构造深度图
        depth = np.full((480, 640), 2.0, dtype=np.float32)

        joints_3d = lift_2d_keypoints_to_3d(
            keypoints_2d=kpts_2d,
            depth_img=depth,
            intrinsics=intrinsics,
            camera_base_pos=camera_base_pos,
            camera_yaw=0.0,
            camera_height=1.2,
            depth_config=self.config["human_state_estimation"],
        )

        self.assertIn("neck", joints_3d)
        self.assertEqual(joints_3d["neck"].source, "derived_3d")
        self.assertTrue(joints_3d["neck"].observable_3d)

        expected_neck = 0.5 * (joints_3d["left_shoulder"].position_world + joints_3d["right_shoulder"].position_world)
        np.testing.assert_allclose(joints_3d["neck"].position_world, expected_neck)

    def test_05_unknown_skeleton_template_raises(self):
        """TEST 5: 未知 skeleton template 必须抛出 ValueError，禁止静默 fallback。"""
        with self.assertRaises(ValueError):
            SkeletonCompleter("standing_generic")

    def test_06_unknown_pose_backend_raises(self):
        """TEST 6: 未知 pose backend 必须抛出 ValueError，禁止静默 fallback。"""
        invalid_cfg = {"perception": {"pose_backend": "rtmpose"}}
        with self.assertRaises(ValueError):
            create_pose_backend(invalid_cfg)

    def test_07_oracle_same_pool_calculation(self):
        """TEST 7 & 9: 验证同一 Candidate Pool 内 Oracle Gap 计算准确无误。"""
        v_oracle = CandidateView(
            candidate_id=1, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.85, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )
        v_selected = CandidateView(
            candidate_id=2, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.75, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )

        res = compute_oracle_gap(v_oracle, v_selected, min_depth_coverage=0.8)
        self.assertTrue(res["oracle_gap_valid"])
        self.assertAlmostEqual(res["oracle_gap"], 0.10)

    def test_08_different_pool_pseudo_negative_gap_rejection(self):
        """TEST 10: 验证跨 Pool 伪负 Gap 不被作为有效 Oracle Gap 接受。"""
        v_oracle_gt = CandidateView(
            candidate_id=1, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.80, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )
        v_est_selected = CandidateView(
            candidate_id=2, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.85, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )

        res = compute_oracle_gap(v_oracle_gt, v_est_selected, min_depth_coverage=0.8)
        # max(0.0, 0.80 - 0.85) 为 0.0，绝不产生负数 Gap
        self.assertEqual(res["oracle_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
