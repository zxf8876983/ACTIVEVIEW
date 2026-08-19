"""
纯 Python 单元与闭环测试脚本 —— test_v60_pure_python.py
=====================================================

功能：
    在无需 GPU、无需 Habitat 物理引擎的环境下，对 v6.0 的所有纯算法与逻辑模块
    进行完整自动化单元测试。

测试覆盖：
    1. Keypoint Schema 映射（COCO-17 -> EA-AVS 15，颈部与骨盆中点推导）
    2. Depth Lifter 逆投影与局部深度采样数学正确性
    3. Orientation Estimator 双侧对称关节朝向估计与角度归一化
    4. Skeleton Completion 模板缩放、旋转、平移与观测关节替换
    5. EstimatedHumanState 数据结构完整性与验证器（State Validator）
    6. HumanStateEstimator 闭环推理（使用 Mock 2D 后端）
    7. Candidate Sampler 基于估计位置采样与测地过滤逻辑
    8. EstimatedPredictiveEvaluator 与策略选择（支持 Stay 及失效安全兜底）
    9. Oracle Policy 与 Oracle Gap 严格同口径计算
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
from ea_avs_v6.pose_backend import MockPoseBackend, Pose2DDetection
from ea_avs_v6.human_state_estimator import HumanStateEstimator
from ea_avs_v6.geometry import compute_camera_intrinsics, normalize_angle
from ea_avs_v6.candidate_sampler import CandidateView
from ea_avs_v6.policies import EstimatedStateOursPolicy, GTStateOursPolicy
from ea_avs_v6.oracle_policy import OraclePolicy, compute_oracle_gap


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
            "human_state_estimation": {
                "min_2d_keypoints": 6,
                "min_3d_keypoints": 4,
                "min_torso_keypoints_3d": 2,
                "depth_patch_size": 5,
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
                "min_hit_distance": 0.05,
                "max_raycast_error_rate": 0.10,
            },
        }

    def test_01_keypoint_schema_mapping(self):
        """测试 COCO-17 到 EA-AVS 15 的转换与中点推导。"""
        coco_kpts = {
            "nose": (320.0, 100.0, 0.9),
            "left_shoulder": (300.0, 140.0, 0.85),
            "right_shoulder": (340.0, 140.0, 0.85),
            "left_hip": (310.0, 240.0, 0.8),
            "right_hip": (330.0, 240.0, 0.8),
            "left_knee": (310.0, 340.0, 0.75),
            "right_knee": (330.0, 340.0, 0.75),
            "left_ankle": (310.0, 440.0, 0.7),
            "right_ankle": (330.0, 440.0, 0.7),
        }
        res = coco17_to_ea_avs_15(coco_kpts, 640, 480, min_confidence=0.30)
        self.assertEqual(len(res), 15)

        # 检查 neck 推导
        neck = res["neck"]
        self.assertTrue(neck.detected)
        self.assertEqual(neck.source, "derived_2d")
        self.assertAlmostEqual(neck.u, 320.0)
        self.assertAlmostEqual(neck.v, 140.0)

        # 检查 pelvis 推导
        pelvis = res["pelvis"]
        self.assertTrue(pelvis.detected)
        self.assertEqual(pelvis.source, "derived_2d")
        self.assertAlmostEqual(pelvis.u, 320.0)
        self.assertAlmostEqual(pelvis.v, 240.0)

        # 检查 head 映射
        self.assertTrue(res["head"].detected)
        self.assertEqual(res["head"].source, "observed_2d")

    def test_02_depth_lifter_math(self):
        """测试 2D 到 3D 逆投影数学一致性。"""
        intrinsics = compute_camera_intrinsics(640, 480, 90.0)
        camera_base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        camera_yaw = 0.0  # +Z forward
        camera_height = 1.2

        # 图像中心点 (320, 240)，深度 2.0m
        u, v = 320.0, 240.0
        depth = 2.0
        pos_cam, pos_world = backproject_keypoint_to_world(
            u, v, depth, intrinsics, camera_base_pos, camera_yaw, camera_height
        )

        # 相机坐标系中应为 (0, 0, 2)
        self.assertAlmostEqual(pos_cam[0], 0.0, places=4)
        self.assertAlmostEqual(pos_cam[1], 0.0, places=4)
        self.assertAlmostEqual(pos_cam[2], 2.0, places=4)

        # 世界坐标系中：base=(0,0,0), cam=(0,1.2,0), forward=(0,0,1) -> (0, 1.2, 2.0)
        self.assertAlmostEqual(pos_world[0], 0.0, places=4)
        self.assertAlmostEqual(pos_world[1], 1.2, places=4)
        self.assertAlmostEqual(pos_world[2], 2.0, places=4)

    def test_03_orientation_estimator(self):
        """测试基于左右肩和左右髋的水平朝向角估计。"""
        estimator = OrientationEstimator(self.config)

        # 人体位于 (0, 0, 2)，面向 +Z (yaw=0)
        # 视觉提升坐标系下：解剖左肩在 +X (x=0.22, y=1.35, z=2.0)
        # 解剖右肩在 -X (x=-0.22, y=1.35, z=2.0)
        joints_3d = {
            "left_shoulder": EstimatedJoint3D("left_shoulder", np.array([0.22, 1.35, 2.0]), None, 0.9, True, True, "observed_2d"),
            "right_shoulder": EstimatedJoint3D("right_shoulder", np.array([-0.22, 1.35, 2.0]), None, 0.9, True, True, "observed_2d"),
            "left_hip": EstimatedJoint3D("left_hip", np.array([0.16, 0.90, 2.0]), None, 0.9, True, True, "observed_2d"),
            "right_hip": EstimatedJoint3D("right_hip", np.array([-0.16, 0.90, 2.0]), None, 0.9, True, True, "observed_2d"),
        }

        res = estimator.estimate_orientation(joints_3d)
        self.assertTrue(res.valid)
        self.assertEqual(res.source, "shoulders_and_hips")
        # 验证推导出的 yaw 接近 0
        self.assertAlmostEqual(res.yaw_rad, 0.0, places=3)

    def test_04_skeleton_completion(self):
        """测试使用标准模板补全缺失关节。"""
        completer = SkeletonCompleter(template_name="standing")
        human_pos = np.array([0.0, 0.0, 2.0], dtype=np.float64)
        human_yaw = 0.0

        # 仅提供头部和颈部
        joints_3d = {
            "head": EstimatedJoint3D("head", np.array([0.0, 1.60, 2.0]), None, 0.9, True, True, "observed_2d"),
            "neck": EstimatedJoint3D("neck", np.array([0.0, 1.40, 2.0]), None, 0.9, True, True, "derived_2d"),
        }

        proxy_skel, obs_skel, updated_joints = completer.complete_skeleton(
            joints_3d, human_pos, human_yaw, body_scale=1.0
        )

        self.assertEqual(len(proxy_skel), 15)
        self.assertEqual(len(obs_skel), 2)
        # 检查未观测关节（如 left_ankle）是否由模板准确补全
        self.assertIn("left_ankle", proxy_skel)
        self.assertEqual(updated_joints["left_ankle"].source, "template_completion")
        # left_ankle standing 模板高度为 0.10
        self.assertAlmostEqual(proxy_skel["left_ankle"][1], 0.10, places=2)

    def test_05_state_validation_and_stay_fallback(self):
        """测试状态验证逻辑与无效状态下的安全原地停留 (Stay)。"""
        invalid_state = EstimatedHumanState(
            valid=False,
            human_position_world=None,
            human_yaw=None,
            pose_detection_score=0.1,
            failure_reason="insufficient_keypoints",
        )

        is_valid, reason = validate_estimated_state(invalid_state, self.config)
        self.assertFalse(is_valid)

        policy = EstimatedStateOursPolicy()
        current_view = CandidateView(
            candidate_id=0,
            position=np.array([0, 0, 0]),
            yaw=0.0,
            geodesic_distance=0.0,
            euclidean_distance_to_human=2.0,
            is_valid=True,
            pred_score={"Q_pred": 0.5, "is_occlusion_valid_pred": True},
        )
        cand1 = CandidateView(
            candidate_id=1,
            position=np.array([1, 0, 1]),
            yaw=0.0,
            geodesic_distance=1.5,
            euclidean_distance_to_human=2.0,
            is_valid=True,
            pred_score={"Q_pred": 0.9, "is_occlusion_valid_pred": True},
        )

        # 状态无效时，必须退回当前视角
        selected = policy.select(current_view, [cand1], is_state_valid=False)
        self.assertEqual(selected.candidate_id, 0)
        self.assertTrue(policy.last_selection_stats["fell_back_to_current"])
        self.assertEqual(policy.last_selection_stats["fallback_reason"], "perception_invalid_fallback")

    def test_06_human_state_estimator_with_mock(self):
        """测试完整 HumanStateEstimator 闭环（使用 Mock 2D 后端）。"""
        kpts_2d = {
            "head": Keypoint2D("head", 320, 100, 0.9, True, "observed_2d"),
            "neck": Keypoint2D("neck", 320, 140, 0.9, True, "derived_2d"),
            "pelvis": Keypoint2D("pelvis", 320, 240, 0.9, True, "derived_2d"),
            "left_shoulder": Keypoint2D("left_shoulder", 300, 140, 0.9, True, "observed_2d"),
            "right_shoulder": Keypoint2D("right_shoulder", 340, 140, 0.9, True, "observed_2d"),
            "left_hip": Keypoint2D("left_hip", 310, 240, 0.9, True, "observed_2d"),
            "right_hip": Keypoint2D("right_hip", 330, 240, 0.9, True, "observed_2d"),
            "left_knee": Keypoint2D("left_knee", 310, 340, 0.8, True, "observed_2d"),
            "right_knee": Keypoint2D("right_knee", 330, 340, 0.8, True, "observed_2d"),
            "left_ankle": Keypoint2D("left_ankle", 310, 440, 0.8, True, "observed_2d"),
            "right_ankle": Keypoint2D("right_ankle", 330, 440, 0.8, True, "observed_2d"),
            "left_elbow": Keypoint2D("left_elbow", 280, 180, 0.8, True, "observed_2d"),
            "right_elbow": Keypoint2D("right_elbow", 360, 180, 0.8, True, "observed_2d"),
            "left_wrist": Keypoint2D("left_wrist", 280, 220, 0.8, True, "observed_2d"),
            "right_wrist": Keypoint2D("right_wrist", 360, 220, 0.8, True, "observed_2d"),
        }
        mock_det = Pose2DDetection(
            keypoints=kpts_2d,
            bbox_xyxy=(260, 80, 380, 460),
            score=0.95,
            backend_name="mock_pose_backend",
        )
        mock_backend = MockPoseBackend([mock_det])

        estimator = HumanStateEstimator(self.config, pose_backend=mock_backend)

        # 构造虚拟 RGB 与平坦 2.0m 深度图
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        camera_state = {
            "position": np.array([0.0, 0.0, 0.0]),
            "yaw": 0.0,
            "camera_height": 1.2,
            "intrinsics": compute_camera_intrinsics(640, 480, 90.0),
        }

        state = estimator.estimate(rgb, depth, camera_state)

        self.assertTrue(state.valid)
        self.assertIsNotNone(state.human_position_world)
        self.assertIsNotNone(state.human_yaw)
        self.assertEqual(len(state.observable_3d_keypoints), 15)
        self.assertEqual(len(state.proxy_full_skeleton), 15)

    def test_07_oracle_gap_computation(self):
        """测试 Oracle Gap 的严格同口径与覆盖过滤。"""
        v_oracle = CandidateView(
            candidate_id=1, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.85, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )
        v_ours = CandidateView(
            candidate_id=2, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.75, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )
        v_curr = CandidateView(
            candidate_id=0, position=np.zeros(3), yaw=0.0, geodesic_distance=0.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.50, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )

        res = compute_oracle_gap(v_oracle, v_ours, v_curr, min_depth_coverage=0.8)
        self.assertTrue(res["oracle_gap_valid"])
        self.assertAlmostEqual(res["oracle_gap"], 0.10)

        # 测试低覆盖候选过滤
        v_ours_low_cov = CandidateView(
            candidate_id=3, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True,
            true_score={"Q_true": 0.75, "true_evaluation_source": "depth", "depth_coverage_true": 0.4}
        )
        res2 = compute_oracle_gap(v_oracle, v_ours_low_cov, v_curr, min_depth_coverage=0.8)
        self.assertFalse(res2["oracle_gap_valid"])
        self.assertEqual(res2["oracle_gap_reason"], "ours_low_depth_coverage")


if __name__ == "__main__":
    unittest.main()
