"""
主动视角选择闭环仿真评测引擎 —— closed_loop_episode.py (v11.4)
============================================================

职责：
    1. 编排完整的单步/One-shot 主动感知闭环流程：
       Initial State -> Initial Observation -> Candidate Generation & Filter ->
       Utility Prediction & Policy Selection -> Habitat Navigation ->
       Post-Navigation Observation -> Closed-Loop Evaluation Metrics;
    2. 支持在完全相同初始状态与候选池下严格评测 4 大基准策略：
       - Random View
       - Nearest View (Distance Greedy)
       - Fixed Front View (Fixed 0 deg heuristic)
       - Utility Predictor (Ours)
       - Oracle (Theoretical Upper Bound)
    3. 计算闭环核心科研评价指标：
       - Entropy Reduction (H_initial - H_after)
       - Accuracy Improvement (Acc_after - Acc_initial)
       - Navigation Efficiency (Gain / Distance)
       - Navigation Success Rate
    4. 导出结构化 Episode 记录字典。
"""

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v11.active_view.action_registry import ActionRegistry, DEFAULT_ACTION_CATEGORIES
from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.habitat_filter import HabitatViewFilter
from ea_avs_mvp_v11.active_view.human_placement_generator import HumanPlacementGenerator
from ea_avs_mvp_v11.active_view.navigation_controller import NavigationController, NavigationTrajectory
from ea_avs_mvp_v11.active_view.occlusion_analyzer import OcclusionAnalyzer
from ea_avs_mvp_v11.active_view.robot_start_sampler import RobotStartSampler
from ea_avs_mvp_v11.active_view.scene_manager import SceneManager
from ea_avs_mvp_v11.active_view.utility_predictor import ViewpointUtilityPredictor
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.core.paths import get_data_root

logger = logging.getLogger("closed_loop_episode")

ACTION_CLASSES = list(DEFAULT_ACTION_CATEGORIES)



@dataclass
class PolicyExecutionResult:
    """单一策略在单 Episode 中的闭环执行结果。"""
    policy_name: str
    selected_viewpoint: Dict[str, Any]
    selected_viewpoint_id: int
    trajectory: Dict[str, Any]
    navigation_distance: float
    navigation_steps: int
    navigation_success: bool
    entropy_after: float
    confidence_after: float
    predicted_action_id: int
    predicted_action_label: str
    is_correct_after: bool
    entropy_reduction: float
    accuracy_improved: bool
    navigation_efficiency: float
    oracle_gap: float
    occlusion_ratio: float = 0.0
    occlusion_level: str = "Easy"


@dataclass
class ClosedLoopEpisodeResult:
    """完整闭环 Episode 评估记录实体。"""
    episode_id: str
    scene_id: str
    action_id: int
    action_label: str
    motion_id: str
    human_placement: Dict[str, Any]
    robot_initial_viewpoint: Dict[str, Any]
    initial_observation: Dict[str, Any]
    candidate_pool_stats: Dict[str, int]
    occlusion_level: str = "Medium"
    policy_results: Dict[str, PolicyExecutionResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "action_id": self.action_id,
            "action_label": self.action_label,
            "motion_id": self.motion_id,
            "human_placement": self.human_placement,
            "robot_initial_viewpoint": self.robot_initial_viewpoint,
            "initial_observation": self.initial_observation,
            "candidate_pool_stats": self.candidate_pool_stats,
            "occlusion_level": self.occlusion_level,
            "policy_results": {k: v.__dict__ for k, v in self.policy_results.items()},
        }


class ClosedLoopActivePerceptionRunner:
    """主动视角选择闭环仿真评测引擎。"""

    def __init__(
        self,
        predictor_model_path: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ):
        self.data_root = Path(data_root) if data_root else get_data_root()
        self.seed = seed

        self.scene_mgr = SceneManager()
        self.human_gen = HumanPlacementGenerator(scene_manager=self.scene_mgr, seed=seed)
        self.robot_sampler = RobotStartSampler(scene_manager=self.scene_mgr, seed=seed)
        self.candidate_gen = CandidateViewGenerator()
        self.view_filter = HabitatViewFilter()
        self.nav_controller = NavigationController()
        self.occlusion_analyzer = OcclusionAnalyzer(data_root=self.data_root)
        self.action_registry = ActionRegistry(data_root=self.data_root, exclude_locomotion=True)

        # 加载训练好的 ST-GCN 分类器
        stgcn_ckpt = self.data_root / "checkpoints" / "v11_st_gcn" / "best_st_gcn_model.pth"
        if not stgcn_ckpt.exists():
            stgcn_ckpt = self.data_root / "checkpoints" / "v10_st_gcn" / "best_st_gcn_model.pth"

        if stgcn_ckpt.exists():
            self.classifier = ActionClassifier(checkpoint_path=stgcn_ckpt)
        else:
            self.classifier = ActionClassifier()

        # 加载训练好的 Utility Predictor
        if predictor_model_path is None:
            predictor_model_path = self.data_root / "checkpoints" / "v11_utility_multiscene" / "utility_predictor_best.pth"
            if not predictor_model_path.exists():
                predictor_model_path = self.data_root / "checkpoints" / "v11_utility" / "utility_predictor_best.pth"

        self.predictor = ViewpointUtilityPredictor(model_path=predictor_model_path, in_dim=11)

    def _get_skeleton(self, action_id: int, instance_idx: int) -> np.ndarray:
        """提取动作序列。"""
        return self.action_registry.get_skeleton_sequence(action_id=action_id, instance_idx=instance_idx)


    def _evaluate_viewpoint_perception(
        self,
        base_skel: np.ndarray,
        action_id: int,
        angle_deg: float,
        distance_m: float,
        placement_difficulty: float = 0.5,
        scene_id: Optional[str] = None,
    ) -> Tuple[float, float, int, str, bool, Dict[str, Any]]:
        """
        在指定角度、视距与室内遮挡下进行动作识别与物理不确定度推断。
        """
        ang_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)

        joints_3d = np.zeros_like(base_skel)
        joints_3d[:, :, 0] = base_skel[:, :, 0] * cos_a - base_skel[:, :, 2] * sin_a
        joints_3d[:, :, 1] = base_skel[:, :, 1]
        joints_3d[:, :, 2] = base_skel[:, :, 0] * sin_a + base_skel[:, :, 2] * cos_a

        # 严密计算遮挡指标
        occ_res = self.occlusion_analyzer.analyze_viewpoint_occlusion(
            angle_deg=angle_deg,
            distance_m=distance_m,
            scene_id=scene_id,
            placement_difficulty=placement_difficulty,
        )

        # 物理温度衰减建模 (结合方位角自遮挡与家具遮挡)
        gamma_theta = 1.0 + 1.2 * (1.0 - math.cos(ang_rad))
        gamma_r = 1.0 + 0.6 * max(0.0, (distance_m - 1.5) / 1.5)
        occ_penalty = 1.8 * occ_res.occlusion_ratio
        temperature = float(gamma_theta * gamma_r + occ_penalty)

        pred = self.classifier.predict_sequence(joints_3d, is_normalized=True)

        raw_probs = np.array(pred.raw_probabilities, dtype=np.float32)
        logits = np.log(np.clip(raw_probs, 1e-7, 1.0))
        scaled_logits = logits / max(temperature, 0.5)
        calib_probs = np.exp(scaled_logits - np.max(scaled_logits))
        calib_probs /= np.sum(calib_probs)

        entropy = float(-np.sum(calib_probs * np.log(np.clip(calib_probs, 1e-8, 1.0))))
        conf = float(np.max(calib_probs))
        pred_class = int(np.argmax(calib_probs))
        pred_label = ACTION_CLASSES[pred_class] if pred_class < len(ACTION_CLASSES) else f"class_{pred_class}"
        is_correct = bool(pred_class == action_id)

        return entropy, conf, pred_class, pred_label, is_correct, occ_res.to_dict()

    def run_single_episode(
        self,
        episode_idx: int,
        scene_id: str,
        action_id: int,
        instance_idx: int,
    ) -> ClosedLoopEpisodeResult:
        """
        运行单个完整主动感知闭环 Episode。
        """
        action_label = ACTION_CLASSES[action_id]
        motion_id = f"{action_label}_inst_{instance_idx:04d}"
        ep_id = f"closed_loop_ep_{episode_idx:05d}"

        # 1. 采样初始人体与机器人位姿
        human_placement = self.human_gen.sample_human_placement(scene_id=scene_id)
        hx, hy, hz = human_placement["human_position"]

        robot_init_vp = self.robot_sampler.sample_robot_start(human_placement=human_placement, min_dist=3.0, max_dist=7.0)
        rx, ry, rz = robot_init_vp["position"]
        init_dist = robot_init_vp["distance_to_human"]
        init_ang = robot_init_vp["angle_to_human"]

        base_skel = self._get_skeleton(action_id, instance_idx)

        # 2. 计算初始观察 (Initial Observation O_0)
        h_init, conf_init, pred_class_init, pred_lbl_init, is_corr_init, init_occ = self._evaluate_viewpoint_perception(
            base_skel=base_skel,
            action_id=action_id,
            angle_deg=init_ang,
            distance_m=init_dist,
            placement_difficulty=human_placement.get("placement_difficulty", 0.5),
            scene_id=scene_id,
        )
        init_obs_dict = {
            "entropy": round(h_init, 4),
            "confidence": round(conf_init, 4),
            "predicted_action_id": pred_class_init,
            "predicted_action_label": pred_lbl_init,
            "is_correct": is_corr_init,
            "distance": round(init_dist, 4),
            "angle": round(init_ang, 2),
            "occlusion_ratio": init_occ["occlusion_ratio"],
            "occlusion_level": init_occ["occlusion_level"],
        }

        # 3. 候选视点生成与过滤
        raw_candidates = self.candidate_gen.generate(
            human_position=[hx, hy, hz],
            robot_current_position=[rx, ry, rz],
        )
        feasible_viewpoints = self.view_filter.filter_viewpoints(
            candidates=raw_candidates,
            human_position=[hx, hy, hz],
            robot_current_position=[rx, ry, rz],
        )

        # 计算各候选视点的真实后验与特征向量
        candidate_features = []
        candidate_evals = []
        for vp in feasible_viewpoints:
            ent, conf, p_cls, p_lbl, corr, cand_occ = self._evaluate_viewpoint_perception(
                base_skel=base_skel,
                action_id=action_id,
                angle_deg=vp.angle,
                distance_m=vp.distance,
                placement_difficulty=human_placement.get("placement_difficulty", 0.5),
                scene_id=scene_id,
            )
            candidate_evals.append({
                "entropy": ent,
                "confidence": conf,
                "predicted_action_id": p_cls,
                "predicted_action_label": p_lbl,
                "is_correct": corr,
                "occlusion_ratio": cand_occ["occlusion_ratio"],
                "occlusion_level": cand_occ["occlusion_level"],
            })
            feat = self.predictor.construct_features(robot_init_vp, vp)
            candidate_features.append(feat)

        # 4. 执行多策略选择
        # (A) Oracle: 全局最优（最低熵）
        oracle_idx = int(np.argmin([e["entropy"] for e in candidate_evals]))
        h_oracle = candidate_evals[oracle_idx]["entropy"]

        # (B) Random View
        rand_idx = int(self.seed + episode_idx) % len(feasible_viewpoints)

        # (C) Nearest View (距离最近)
        dists = [vp.navigation_cost for vp in feasible_viewpoints]
        nearest_idx = int(np.argmin(dists))

        # (D) Fixed Front View (接近 0 deg, 2.0m 的固定正面视点)
        front_diffs = [abs(vp.angle) + abs(vp.distance - 2.0) for vp in feasible_viewpoints]
        fixed_front_idx = int(np.argmin(front_diffs))

        # (E) Utility Predictor (Ours)
        pred_scores = [self.predictor.predict_single(f) for f in candidate_features]
        ours_idx = int(np.argmax(pred_scores))

        policy_indices = {
            "random": rand_idx,
            "nearest": nearest_idx,
            "fixed_front": fixed_front_idx,
            "utility_predictor": ours_idx,
            "oracle": oracle_idx,
        }

        episode_result = ClosedLoopEpisodeResult(
            episode_id=ep_id,
            scene_id=scene_id,
            action_id=action_id,
            action_label=action_label,
            motion_id=motion_id,
            human_placement=human_placement,
            robot_initial_viewpoint=robot_init_vp,
            initial_observation=init_obs_dict,
            candidate_pool_stats={"raw": len(raw_candidates), "feasible": len(feasible_viewpoints)},
            occlusion_level=init_occ["occlusion_level"],
        )

        for p_name, p_idx in policy_indices.items():
            vp_selected = feasible_viewpoints[p_idx]
            eval_after = candidate_evals[p_idx]

            # 5. Habitat 导航轨迹规划与执行
            traj = self.nav_controller.plan_trajectory(
                start_position=[rx, ry, rz],
                target_position=vp_selected.position,
                human_position=[hx, hy, hz],
            )

            h_after = eval_after["entropy"]
            delta_h = float(h_init - h_after)
            acc_improved = bool(eval_after["is_correct"] and not is_corr_init)
            efficiency = delta_h / max(traj.total_distance, 0.1)

            res = PolicyExecutionResult(
                policy_name=p_name,
                selected_viewpoint=vp_selected.to_dict(),
                selected_viewpoint_id=int(vp_selected.id),
                trajectory=traj.to_dict(),
                navigation_distance=round(traj.total_distance, 4),
                navigation_steps=int(traj.num_steps),
                navigation_success=bool(traj.is_success),
                entropy_after=round(h_after, 4),
                confidence_after=round(eval_after["confidence"], 4),
                predicted_action_id=int(eval_after["predicted_action_id"]),
                predicted_action_label=str(eval_after["predicted_action_label"]),
                is_correct_after=bool(eval_after["is_correct"]),
                entropy_reduction=round(delta_h, 4),
                accuracy_improved=acc_improved,
                navigation_efficiency=round(efficiency, 4),
                oracle_gap=round(max(0.0, h_after - h_oracle), 4),
            )
            episode_result.policy_results[p_name] = res

        return episode_result
