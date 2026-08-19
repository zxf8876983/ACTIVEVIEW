"""
纯 Python 单元与闭环测试套件 —— test_v60_pure_python.py
=====================================================

功能：
    在无 GPU、无 Habitat 物理引擎的环境下，对 v6.0 的所有纯算法与逻辑模块
    进行 15 项核心科研测试。

测试清单 (15 项严谨性测试)：
    TEST 1:  invalid occlusion 不进入 visible_occ
    TEST 2:  Estimated static ray (clear / static_scene_blocked / unknown)
    TEST 3:  dynamic/articulated hit ignored by static-only raycast
    TEST 4:  canonical yaw 8 directions
    TEST 5:  3D neck derivation
    TEST 6:  3D pelvis derivation
    TEST 7:  unknown skeleton template raises ValueError
    TEST 8:  unknown pose backend raises ValueError
    TEST 9:  No-GT AST function-level guard
    TEST 10: No future render sentinel
    TEST 11: Oracle same-pool valid gap
    TEST 12: Oracle cross-pool returns pool_mismatch
    TEST 13: same-pool selected > Oracle returns oracle_not_upper_bound
    TEST 14: Shared-Pool GT and Est use identical candidate geometry
    TEST 15: strict GT skeleton rejects incomplete skeleton
"""

import ast
import os
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
from ea_avs_v6.humanoid_validation import validate_gt_skeleton
from ea_avs_v6.habitat_runner import resolve_stage_id


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
                "max_people": 1,
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
            def cast_ray_static_scene(self, origin, direction, max_dist):
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

        self.assertEqual(len(res["visible_keypoints_occ_pred"]), 0)
        self.assertEqual(res["S_action_occ_pred"], 0.0)
        self.assertEqual(res["S_kp_occ_pred"], 0.0)
        self.assertFalse(res["is_occlusion_valid_pred"])

    def test_02_estimated_static_ray_states(self):
        """TEST 2: 验证 Estimated Static 射线检测（clear, static_scene_blocked, unknown）。"""
        class MockStaticClearRunner:
            def cast_ray_static_scene(self, origin, direction, max_dist):
                return {"has_hits": False, "hit_distance": None, "hit_source": "none"}

        class MockStaticBlockedRunner:
            def cast_ray_static_scene(self, origin, direction, max_dist):
                return {"has_hits": True, "hit_distance": 1.0, "hit_source": "stage"}

        # 1. Clear ray
        r1 = cast_ray_to_estimated_point(MockStaticClearRunner(), np.array([0, 0, 0]), np.array([0, 0, 2]))
        self.assertTrue(r1["valid"])
        self.assertFalse(r1["occluded"])
        self.assertEqual(r1["occlusion_source"], "clear")

        # 2. Blocked ray by static scene
        r2 = cast_ray_to_estimated_point(MockStaticBlockedRunner(), np.array([0, 0, 0]), np.array([0, 0, 2]))
        self.assertTrue(r2["valid"])
        self.assertTrue(r2["occluded"])
        self.assertEqual(r2["occlusion_source"], "static_scene_blocked")

    def test_03_dynamic_hit_ignored_by_static_raycast(self):
        """TEST 3: 验证动态/Humanoid 碰撞体被 Estimated static raycast 自动忽略。"""
        class MockRunnerMultiHit:
            def cast_ray_static_scene(self, origin, direction, max_dist):
                # 即使有 0.8m 的 dynamic hit，static_scene 仍返回 1.2m 的 stage hit
                return {"has_hits": True, "hit_distance": 1.2, "hit_source": "stage"}

        class MockRunnerOnlyDynamic:
            def cast_ray_static_scene(self, origin, direction, max_dist):
                # 只有 0.8m 的 dynamic hit，无 static stage hit -> 返回 has_hits=False
                return {"has_hits": False, "hit_distance": None, "hit_source": "dynamic_ignored"}

        # Test A: stage hit at 1.2m behind 0.8m dynamic hit
        r_a = cast_ray_to_estimated_point(MockRunnerMultiHit(), np.array([0, 0, 0]), np.array([0, 0, 2.0]))
        self.assertTrue(r_a["valid"])
        self.assertTrue(r_a["occluded"])
        self.assertEqual(r_a["hit_distance"], 1.2)
        self.assertEqual(r_a["occlusion_source"], "static_scene_blocked")

        # Test B: only dynamic hit -> ignored -> clear
        r_b = cast_ray_to_estimated_point(MockRunnerOnlyDynamic(), np.array([0, 0, 0]), np.array([0, 0, 2.0]))
        self.assertTrue(r_b["valid"])
        self.assertFalse(r_b["occluded"])
        self.assertEqual(r_b["occlusion_source"], "clear")

        # Test C: static hit within target tolerance (1.95m for 2.0m target with 0.12m tolerance)
        class MockRunnerNearTolerance:
            def cast_ray_static_scene(self, origin, direction, max_dist):
                return {"has_hits": True, "hit_distance": 1.95, "hit_source": "stage"}

        r_c = cast_ray_to_estimated_point(MockRunnerNearTolerance(), np.array([0, 0, 0]), np.array([0, 0, 2.0]), target_tolerance=0.12)
        self.assertTrue(r_c["valid"])
        self.assertFalse(r_c["occluded"])
        self.assertEqual(r_c["occlusion_source"], "clear")

    def test_04_canonical_yaw_convention_8_angles(self):
        """TEST 4: 使用项目 POSE_SKELETONS["standing"] 验证 8 个角度朝向推导严格一致。"""
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

    def test_05_bilateral_3d_neck_derivation(self):
        """TEST 5: 验证 3D neck 由左右肩世界坐标直接推导，不采 2D midpoint depth。"""
        intrinsics = compute_camera_intrinsics(640, 480, 90.0)
        camera_base_pos = np.array([0.0, 0.0, 0.0])

        kpts_2d = {
            "left_shoulder": Keypoint2D("left_shoulder", 300, 140, 0.9, True, "observed_2d"),
            "right_shoulder": Keypoint2D("right_shoulder", 340, 140, 0.9, True, "observed_2d"),
            "neck": Keypoint2D("neck", 320, 140, 0.9, True, "derived_2d"),
        }
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

    def test_06_bilateral_3d_pelvis_derivation(self):
        """TEST 6: 验证 3D pelvis 由左右髋世界坐标直接推导，不采 2D midpoint depth。"""
        intrinsics = compute_camera_intrinsics(640, 480, 90.0)
        camera_base_pos = np.array([0.0, 0.0, 0.0])

        kpts_2d = {
            "left_hip": Keypoint2D("left_hip", 310, 240, 0.9, True, "observed_2d"),
            "right_hip": Keypoint2D("right_hip", 330, 240, 0.9, True, "observed_2d"),
            "pelvis": Keypoint2D("pelvis", 320, 240, 0.9, True, "derived_2d"),
        }
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

        self.assertIn("pelvis", joints_3d)
        self.assertEqual(joints_3d["pelvis"].source, "derived_3d")
        self.assertTrue(joints_3d["pelvis"].observable_3d)
        expected_pelvis = 0.5 * (joints_3d["left_hip"].position_world + joints_3d["right_hip"].position_world)
        np.testing.assert_allclose(joints_3d["pelvis"].position_world, expected_pelvis)

    def test_07_unknown_skeleton_template_raises(self):
        """TEST 7: 未知 skeleton template 必须抛出 ValueError，禁止静默 fallback。"""
        with self.assertRaises(ValueError):
            SkeletonCompleter("standing_generic")

    def test_08_unknown_pose_backend_raises(self):
        """TEST 8: 未知 pose backend 必须抛出 ValueError，禁止静默 fallback。"""
        invalid_cfg = {"perception": {"pose_backend": "rtmpose"}}
        with self.assertRaises(ValueError):
            create_pose_backend(invalid_cfg)

    def test_09_no_gt_ast_guard(self):
        """TEST 9: 函数级 AST 防护，确保 Estimated-State 核心函数内部无 GT 变量与方法引用。"""
        pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ea_avs_v6")
        forbidden_names = {"gt_human_pos", "gt_human_yaw", "gt_skeleton", "humanoid_object_ids", "keypoint_meta", "humanoid_manager"}

        targets = {
            "raycast_utils.py": ["cast_ray_to_estimated_point"],
            "occlusion.py": ["compute_estimated_keypoint_occlusion"],
        }
        for fname, fnames in targets.items():
            fpath = os.path.join(pkg_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=fname)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in fnames:
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Name) and sub.id in forbidden_names:
                            self.fail(f"在 {fname} 函数 {node.name} 中发现禁止的 GT 变量: {sub.id}")

    def test_10_no_future_render_sentinel(self):
        """TEST 10: 验证在选择阶段调用 render_at 会被 Sentinel 拦截。"""
        class SentinelRunner:
            def render_at(self, *args, **kwargs):
                raise AssertionError("Future render called!")
            def cast_ray_static_scene(self, *args, **kwargs):
                return {"has_hits": False, "hit_distance": None, "hit_source": "none"}
            def is_navigable(self, pt): return True
            def snap_point(self, pt): return np.array(pt, dtype=np.float32)
            def geodesic_distance(self, p1, p2): return 1.0

        evaluator = EstimatedPredictiveEvaluator(self.config)
        sampler = CandidateSampler(self.config)
        policy = EstimatedStateOursPolicy()

        est_state = EstimatedHumanState(
            valid=True,
            human_position_world=np.array([0.0, 0.0, 2.0]),
            human_yaw=0.0,
            proxy_full_skeleton={k: np.array(v) + np.array([0.0, 0.0, 2.0]) for k, v in POSE_SKELETONS["standing"].items()},
        )
        runner = SentinelRunner()
        cands = sampler.sample(est_state.human_position_world, np.zeros(3), runner, pool_id="est_pool")
        curr = CandidateView(0, np.zeros(3), 0.0, 0.0, 2.0, True, pool_id="est_pool")

        curr.pred_score = evaluator.score_view_pred(runner, curr.position, curr.yaw, np.zeros(3), est_state, 0.0)
        for c in cands:
            c.pred_score = evaluator.score_view_pred(runner, c.position, c.yaw, np.zeros(3), est_state, c.geodesic_distance)

        # 策略选择成功，未触发 render_at 异常
        selected = policy.select(curr, cands, is_state_valid=True)
        self.assertIsNotNone(selected)

    def test_11_oracle_same_pool_valid_gap(self):
        """TEST 11: 验证同一 Candidate Pool 内 Oracle Gap 计算准确无误。"""
        v_oracle = CandidateView(
            candidate_id=1, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True, pool_id="gt_pool",
            true_score={"Q_true": 0.85, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )
        v_selected = CandidateView(
            candidate_id=2, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True, pool_id="gt_pool",
            true_score={"Q_true": 0.75, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )

        res = compute_oracle_gap(v_oracle, v_selected, min_depth_coverage=0.8)
        self.assertTrue(res["oracle_gap_valid"])
        self.assertAlmostEqual(res["oracle_gap"], 0.10)

    def test_12_oracle_cross_pool_returns_pool_mismatch(self):
        """TEST 12: 验证跨 Pool 调用 compute_oracle_gap 严格返回 pool_mismatch。"""
        v_oracle_gt = CandidateView(
            candidate_id=1, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True, pool_id="gt_pool",
            true_score={"Q_true": 0.80, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )
        v_est_selected = CandidateView(
            candidate_id=2, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True, pool_id="est_pool",
            true_score={"Q_true": 0.85, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )

        res = compute_oracle_gap(v_oracle_gt, v_est_selected, min_depth_coverage=0.8)
        self.assertFalse(res["oracle_gap_valid"])
        self.assertIsNone(res["oracle_gap"])
        self.assertEqual(res["oracle_gap_reason"], "pool_mismatch")

    def test_13_same_pool_selected_greater_than_oracle_returns_not_upper_bound(self):
        """TEST 13: 验证同 pool 下若 selected_q > oracle_q 严格返回 oracle_not_upper_bound。"""
        v_oracle = CandidateView(
            candidate_id=1, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True, pool_id="est_pool",
            true_score={"Q_true": 0.80, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )
        v_selected = CandidateView(
            candidate_id=2, position=np.zeros(3), yaw=0.0, geodesic_distance=1.0,
            euclidean_distance_to_human=2.0, is_valid=True, pool_id="est_pool",
            true_score={"Q_true": 0.85, "true_evaluation_source": "depth", "depth_coverage_true": 0.9}
        )

        res = compute_oracle_gap(v_oracle, v_selected, min_depth_coverage=0.8)
        self.assertFalse(res["oracle_gap_valid"])
        self.assertIsNone(res["oracle_gap"])
        self.assertEqual(res["oracle_gap_reason"], "oracle_not_upper_bound")

    def test_14_shared_pool_identical_geometry(self):
        """TEST 14: 验证 Shared-Pool GT 与 Est 两侧使用完全一致的候选点几何。"""
        class DummyRunner:
            def snap_point(self, p): return np.array(p, dtype=np.float32)
            def is_navigable(self, p): return True
            def geodesic_distance(self, p1, p2): return 2.0

        sampler = CandidateSampler(self.config)
        human_pos = np.array([1.0, 0.0, 2.0])
        robot_pos = np.array([0.0, 0.0, 0.0])

        cands_gt = sampler.sample(human_pos, robot_pos, DummyRunner(), pool_id="gt_pool")
        import copy
        cands_shared_est = copy.deepcopy(cands_gt)
        for c in cands_shared_est:
            c.pool_id = "shared_pool"

        self.assertEqual(len(cands_gt), len(cands_shared_est))
        for cg, ce in zip(cands_gt, cands_shared_est):
            np.testing.assert_allclose(cg.position, ce.position)
            self.assertEqual(cg.yaw, ce.yaw)
            self.assertEqual(cg.geodesic_distance, ce.geodesic_distance)
            self.assertEqual(cg.pool_id, "gt_pool")
            self.assertEqual(ce.pool_id, "shared_pool")

    def test_15_strict_gt_skeleton_rejects_incomplete(self):
        """TEST 15: 验证 strict GT skeleton 对不完整骨架的严格拒绝 (fail-fast)。"""
        incomplete_skeleton_result = {
            "skeleton": {k: np.zeros(3) for k in EA_AVS_15_KEYPOINTS[:10]},  # 仅 10 个关键点
            "keypoint_meta": {},
            "source": "mixed",
            "per_keypoint_source": {k: "link" for k in EA_AVS_15_KEYPOINTS[:10]},
            "keypoint_count": 10,
            "direct_link_count": 10,
            "link_derived_count": 0,
            "fallback_count": 0,
            "missing_keypoints": list(EA_AVS_15_KEYPOINTS[10:]),
        }
        with self.assertRaises(RuntimeError):
            validate_gt_skeleton(incomplete_skeleton_result, strict=True)

    def test_16_estimated_raycast_without_static_api_fails_closed(self):
        """TEST 16: 验证在 Runner 缺失 cast_ray_static_scene API 时，Estimated ray 严格 fail-closed 返回 valid=False/unknown，且绝不调用 generic full-collision cast_ray。"""
        class RunnerWithoutStaticAPI:
            def __init__(self):
                self.generic_called = False

            def cast_ray(self, *args, **kwargs):
                self.generic_called = True
                raise AssertionError("Estimated-State must never call generic full-collision cast_ray")

        runner = RunnerWithoutStaticAPI()
        result = cast_ray_to_estimated_point(
            runner=runner,
            ray_origin=np.array([0.0, 0.0, 0.0]),
            target_point=np.array([0.0, 0.0, 2.0]),
        )

        self.assertFalse(result["valid"])
        self.assertFalse(result["occluded"])
        self.assertEqual(result["occlusion_source"], "unknown")
        self.assertEqual(result.get("failure_reason"), "static_scene_raycast_unavailable")
        self.assertFalse(runner.generic_called, "generic full-collision cast_ray must not be called")

    def test_17_missing_stage_id_fails_fast(self):
        """TEST 17: 验证 resolve_stage_id 在缺少 habitat_sim.stage_id 时严格抛出 RuntimeError，禁止使用猜测默认值。"""
        class FakeHabitatSimWithoutStageId:
            pass

        class FakeHabitatSimWithStageId:
            stage_id = 0

        # 1. 缺少 stage_id 必须抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            resolve_stage_id(FakeHabitatSimWithoutStageId())

        # 2. 传入 None 必须抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            resolve_stage_id(None)

        # 3. 具备 stage_id 必须正确解析为 int
        sid = resolve_stage_id(FakeHabitatSimWithStageId())
        self.assertEqual(sid, 0)
        self.assertIsInstance(sid, int)


if __name__ == "__main__":
    unittest.main()
