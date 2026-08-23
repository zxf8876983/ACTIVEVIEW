"""
闭环主动感知 Episode 执行引擎 —— closed_loop_episode.py (v11.4.1)
=============================================================

职责：
    1. 严格实现 Embodied Visual Active Perception 数据流：
       Human Placement -> Habitat RGB-D Observation -> Pose Estimator -> Estimated Skeleton -> Canonical Alignment -> ST-GCN；
    2. 彻底删除任何 Ground-Truth 骨架直通通道与人为温度缩放欺骗；
    3. 全流程使用真实分类熵与真实估计姿态完成 5 大策略对比 (Random, Nearest, Fixed Front, Oracle, Ours)；
    4. 输出严格符合 v11.4.1 规范的 ClosedLoopEpisodeResult 结构化指标实体。
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v11.active_view.action_registry import ActionRegistry, DEFAULT_ACTION_CATEGORIES
from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.habitat_filter import HabitatViewFilter
from ea_avs_mvp_v11.active_view.human_placement_generator import HumanPlacementGenerator
from ea_avs_mvp_v11.active_view.perception_pipeline import HabitatPerceptionPipeline, get_perception_pipeline
from ea_avs_mvp_v11.active_view.robot_start_sampler import RobotStartSampler
from ea_avs_mvp_v11.active_view.scene_manager import SceneManager
from ea_avs_mvp_v11.active_view.skeleton_canonicalizer import CanonicalSkeletonAligner
from ea_avs_mvp_v11.active_view.utility_predictor import ViewpointUtilityPredictor
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.core.paths import get_data_root



logger = logging.getLogger("closed_loop_episode")

ACTION_CLASSES = list(DEFAULT_ACTION_CATEGORIES)


@dataclass
class PolicyEvaluationResult:
    """策略闭环评估指标。"""
    policy_name: str
    selected_viewpoint_id: Any
    selected_viewpoint: Dict[str, Any]
    entropy_after: float
    entropy_reduction: float
    confidence_after: float
    confidence_improvement: float
    is_correct_after: bool
    accuracy_improved: bool
    navigation_distance: float
    navigation_steps: int
    navigation_efficiency: float
    navigation_success: bool
    oracle_gap: float
    action_probabilities: List[float] = field(default_factory=list)
    predicted_action: str = ""
    trajectory: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClosedLoopEpisodeResult:
    """闭环 Episode 评测指标结果容器。"""
    episode_id: str
    scene_id: str
    action_id: int
    action_label: str
    initial_observation: Dict[str, Any]
    selected_viewpoint_before: Dict[str, Any]
    candidate_evaluations: List[Dict[str, Any]]
    strategy_results: Dict[str, Dict[str, Any]]
    policy_results: Dict[str, PolicyEvaluationResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def robot_initial_viewpoint(self) -> Dict[str, Any]:
        return self.selected_viewpoint_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "action_id": self.action_id,
            "action_label": self.action_label,
            "initial_observation": self.initial_observation,
            "selected_viewpoint_before": self.selected_viewpoint_before,
            "robot_initial_viewpoint": self.selected_viewpoint_before,
            "candidate_evaluations": self.candidate_evaluations,
            "strategy_results": self.strategy_results,
            "policy_results": {k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in self.policy_results.items()},
            "metadata": self.metadata,
        }





class ClosedLoopActivePerceptionEngine:
    """闭环主动感知实验执行引擎。"""

    def __init__(
        self,
        classifier: Optional[ActionClassifier] = None,
        predictor_model_path: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ):
        self.data_root = Path(data_root) if data_root else get_data_root()
        self.scene_mgr = SceneManager()
        self.human_gen = HumanPlacementGenerator(scene_manager=self.scene_mgr, seed=seed)
        self.robot_sampler = RobotStartSampler(scene_manager=self.scene_mgr, seed=seed)
        self.candidate_gen = CandidateViewGenerator()
        self.view_filter = HabitatViewFilter()

        self.action_registry = ActionRegistry(data_root=self.data_root)
        self.canonical_aligner = CanonicalSkeletonAligner()
        self.perception_pipeline = HabitatPerceptionPipeline(data_root=self.data_root)

        # 加载 ST-GCN 分类器
        if classifier is not None:
            self.classifier = classifier
        else:
            stgcn_ckpt = self.data_root / "checkpoints" / "v11_st_gcn" / "best_st_gcn_model.pth"
            self.classifier = ActionClassifier(checkpoint_path=stgcn_ckpt if stgcn_ckpt.exists() else None)

        # 加载训练好的 Utility Predictor
        if predictor_model_path is None:
            predictor_model_path = self.data_root / "checkpoints" / "v11_utility_multiscene" / "utility_predictor_best.pth"
            if not predictor_model_path.exists():
                predictor_model_path = self.data_root / "checkpoints" / "v11_utility" / "utility_predictor_best.pth"

        self.predictor = ViewpointUtilityPredictor(model_path=predictor_model_path, in_dim=11)

    def _observe_and_infer(
        self,
        scene_id: str,
        human_placement: Dict[str, Any],
        robot_viewpoint: Dict[str, Any],
        base_motion_seq: np.ndarray,
        action_id: int,
    ) -> Tuple[float, float, int, str, bool, Dict[str, Any]]:
        """
        在指定视点执行真实传感器感知与动作识别推断：
        Robot Viewpoint -> Habitat Sensor -> RGB/Depth -> Pose Estimator -> Estimated Skeleton -> Canonical Alignment -> ST-GCN.
        """
        # 1. 传感器渲染与姿态估计
        obs = self.perception_pipeline.observe(
            scene_id=scene_id,
            human_state=human_placement,
            robot_viewpoint=robot_viewpoint,
            base_motion_seq=base_motion_seq,
        )
        assert obs["skeleton_source"] == "estimated", "Strict Violation: GT skeleton bypass is forbidden!"

        est_skel = obs["skeleton"]

        # 2. 人体坐标系对齐 (Canonical Alignment)
        canon_skel = self.canonical_aligner.align(est_skel)

        # 3. ST-GCN 动作识别
        pred = self.classifier.predict_sequence(canon_skel, is_normalized=True, apply_canonical=False)

        entropy = float(pred.entropy)
        conf = float(pred.top1_confidence)
        pred_class = int(pred.predicted_class)
        pred_label = pred.predicted_label
        is_correct = bool(pred_class == action_id)

        return entropy, conf, pred_class, pred_label, is_correct, obs

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
        action_label = ACTION_CLASSES[action_id] if action_id < len(ACTION_CLASSES) else f"class_{action_id}"
        ep_id = f"ep_{episode_idx:04d}_{scene_id}_{action_label}"

        # 1. 采样初始人体与机器人位姿
        human_placement = self.human_gen.sample_human_placement(scene_id=scene_id)
        hx, hy, hz = human_placement["human_position"]

        robot_init_vp = self.robot_sampler.sample_robot_start(human_placement=human_placement, min_dist=3.0, max_dist=7.0)
        rx, ry, rz = robot_init_vp["position"]
        init_dist = robot_init_vp["distance_to_human"]
        init_ang = robot_init_vp["angle_to_human"]

        base_motion_seq = self.action_registry.get_skeleton_sequence(action_id, instance_idx)

        # 2. 计算初始观察 (Initial Observation O_0)
        h_init, conf_init, pred_class_init, pred_lbl_init, is_corr_init, init_obs = self._observe_and_infer(
            scene_id=scene_id,
            human_placement=human_placement,
            robot_viewpoint=robot_init_vp,
            base_motion_seq=base_motion_seq,
            action_id=action_id,
        )
        init_obs_dict = {
            "entropy": round(h_init, 4),
            "confidence": round(conf_init, 4),
            "predicted_action_id": pred_class_init,
            "predicted_action_label": pred_lbl_init,
            "is_correct": is_corr_init,
            "distance": round(init_dist, 4),
            "angle": round(init_ang, 2),
            "visible_ratio": round(init_obs["visible_ratio"], 4),
            "occlusion_ratio": round(init_obs["occlusion_ratio"], 4),
            "occlusion_level": init_obs["occlusion_level"],
            "skeleton_source": "estimated",
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
            vp_dict = {
                "position": vp.position,
                "angle": vp.angle,
                "distance": vp.distance,
                "yaw": vp.yaw,
            }

            ent, conf, p_cls, p_lbl, corr, cand_obs = self._observe_and_infer(
                scene_id=scene_id,
                human_placement=human_placement,
                robot_viewpoint=vp_dict,
                base_motion_seq=base_motion_seq,
                action_id=action_id,
            )
            nav_cost_val = getattr(vp, "navigation_cost", getattr(vp, "nav_cost", 1.0))
            candidate_evals.append({
                "viewpoint_id": vp.id,
                "angle": round(vp.angle, 2),
                "distance": round(vp.distance, 4),

                "entropy": round(ent, 4),
                "confidence": round(conf, 4),
                "predicted_action_id": p_cls,
                "predicted_action_label": p_lbl,
                "is_correct": corr,
                "nav_cost": round(nav_cost_val, 4),
                "visible_ratio": round(cand_obs["visible_ratio"], 4),
                "occlusion_ratio": round(cand_obs["occlusion_ratio"], 4),
                "occlusion_level": cand_obs["occlusion_level"],
                "skeleton_source": "estimated",
            })

            # 11 维特征提取
            ang_rad = math.radians(vp.angle)
            feat_11d = [
                h_init,
                conf_init,
                init_dist,
                math.cos(math.radians(init_ang)),
                math.sin(math.radians(init_ang)),
                vp.distance,
                math.cos(ang_rad),
                math.sin(ang_rad),
                nav_cost_val,
                cand_obs["occlusion_ratio"],
                cand_obs["visible_ratio"],
            ]
            candidate_features.append(feat_11d)

        # 4. 执行 5 种视点选择决策策略
        strategies: Dict[str, Dict[str, Any]] = {}

        # 策略 A: Random Baseline
        rand_idx = int(np.random.randint(0, len(feasible_viewpoints)))
        strategies["Random"] = self._format_strategy_result(
            feasible_viewpoints[rand_idx], candidate_evals[rand_idx], h_init, is_corr_init
        )

        # 策略 B: Nearest Baseline (贪心最近视点)
        nearest_idx = int(np.argmin([getattr(vp, "navigation_cost", getattr(vp, "nav_cost", 1.0)) for vp in feasible_viewpoints]))
        strategies["Nearest"] = self._format_strategy_result(
            feasible_viewpoints[nearest_idx], candidate_evals[nearest_idx], h_init, is_corr_init
        )

        # 策略 C: Fixed Front (固定人体正前方 0 度视点)
        front_idx = int(np.argmin([abs(vp.angle - 0.0) for vp in feasible_viewpoints]))
        strategies["Fixed Front"] = self._format_strategy_result(
            feasible_viewpoints[front_idx], candidate_evals[front_idx], h_init, is_corr_init
        )

        # 策略 D: Utility Predictor (Ours)
        if len(candidate_features) > 0:
            pred_utilities = self.predictor.predict(candidate_features)
            best_util_idx = int(np.argmax(pred_utilities))
            strategies["Utility Predictor (Ours)"] = self._format_strategy_result(
                feasible_viewpoints[best_util_idx], candidate_evals[best_util_idx], h_init, is_corr_init,
                predicted_utility=float(pred_utilities[best_util_idx])
            )
        else:
            strategies["Utility Predictor (Ours)"] = strategies["Fixed Front"]

        # 策略 E: Oracle (理论上界: 真实识别准确且后验熵最低的视点)
        # 排序准则: 准确性 > 最低后验熵 > 最小导航代价
        oracle_idx = 0
        best_score = -9999.0
        for idx, ev in enumerate(candidate_evals):
            corr_score = 100.0 if ev["is_correct"] else 0.0
            ent_score = -ev["entropy"] * 10.0
            nav_penalty = -getattr(feasible_viewpoints[idx], "navigation_cost", getattr(feasible_viewpoints[idx], "nav_cost", 1.0)) * 0.1
            total_sc = corr_score + ent_score + nav_penalty
            if total_sc > best_score:
                best_score = total_sc
                oracle_idx = idx

        strategies["Oracle"] = self._format_strategy_result(
            feasible_viewpoints[oracle_idx], candidate_evals[oracle_idx], h_init, is_corr_init
        )

        # 构建统一的 policy_results 实体字典
        pol_mapping = {
            "random": strategies["Random"],
            "nearest": strategies["Nearest"],
            "fixed_front": strategies["Fixed Front"],
            "utility_predictor": strategies["Utility Predictor (Ours)"],
            "oracle": strategies["Oracle"],
        }
        policy_results = {}
        oracle_h = strategies["Oracle"]["entropy_after"]
        for p_key, st_dict in pol_mapping.items():
            h_after = st_dict["entropy_after"]
            policy_results[p_key] = PolicyEvaluationResult(
                policy_name=p_key,
                selected_viewpoint_id=st_dict["selected_viewpoint_id"],
                selected_viewpoint=st_dict,
                entropy_after=h_after,
                entropy_reduction=st_dict["delta_entropy"],
                confidence_after=st_dict["confidence_after"],
                confidence_improvement=st_dict["confidence_after"] - conf_init,
                is_correct_after=st_dict["is_correct_after"],
                accuracy_improved=bool(not is_corr_init and st_dict["is_correct_after"]),
                navigation_distance=st_dict["nav_cost"],
                navigation_steps=max(1, int(st_dict["nav_cost"] / 0.25)),
                navigation_efficiency=st_dict["navigation_efficiency"],
                navigation_success=True,
                oracle_gap=max(0.0, h_after - oracle_h),
                predicted_action=st_dict.get("predicted_action_label", ""),
                trajectory={
                    "start_position": robot_init_vp["position"],
                    "goal_position": robot_init_vp["position"],
                },
            )

        return ClosedLoopEpisodeResult(
            episode_id=ep_id,
            scene_id=scene_id,
            action_id=action_id,
            action_label=action_label,
            initial_observation=init_obs_dict,
            selected_viewpoint_before=robot_init_vp,
            candidate_evaluations=candidate_evals,
            strategy_results=strategies,
            policy_results=policy_results,
            metadata={
                "num_candidates": len(feasible_viewpoints),
                "placement_difficulty": human_placement.get("placement_difficulty", 0.5),
                "is_hard_occlusion": bool(init_obs_dict["occlusion_ratio"] >= 0.40),
                "skeleton_source": "estimated",
            },
        )

    def _format_strategy_result(
        self,
        vp: Viewpoint,
        cand_eval: Dict[str, Any],
        h_init: float,
        is_corr_init: bool,
        predicted_utility: Optional[float] = None,
    ) -> Dict[str, Any]:
        h_after = cand_eval["entropy"]
        delta_h = round(float(h_init - h_after), 4)
        nav_cost_val = getattr(vp, "navigation_cost", getattr(vp, "nav_cost", 1.0))
        nav_dist = max(nav_cost_val, 0.1)
        eff = round(float(delta_h / nav_dist), 4)



        return {
            "selected_viewpoint_id": vp.id,
            "angle": round(vp.angle, 2),
            "distance": round(vp.distance, 4),
            "nav_cost": round(nav_cost_val, 4),

            "entropy_before": round(h_init, 4),
            "entropy_after": round(h_after, 4),
            "delta_entropy": delta_h,
            "confidence_after": cand_eval["confidence"],
            "accuracy_before": 1.0 if is_corr_init else 0.0,
            "accuracy_after": 1.0 if cand_eval["is_correct"] else 0.0,
            "is_correct_after": cand_eval["is_correct"],
            "predicted_action_id": cand_eval["predicted_action_id"],
            "predicted_action_label": cand_eval["predicted_action_label"],
            "visible_ratio": cand_eval["visible_ratio"],
            "occlusion_ratio": cand_eval["occlusion_ratio"],
            "occlusion_level": cand_eval["occlusion_level"],
            "navigation_efficiency": eff,
            "predicted_utility": round(predicted_utility, 4) if predicted_utility is not None else None,
            "skeleton_source": "estimated",
        }


# 别名兼容
ClosedLoopActivePerceptionRunner = ClosedLoopActivePerceptionEngine

