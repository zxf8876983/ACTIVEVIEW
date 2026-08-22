#!/usr/bin/env python3
"""
ACTIVEVIEW v11.0 统一主运行与验证入口 —— run_v11.py
===================================================

职责：
    1. 验证 ea_avs_mvp_v11 独立闭环系统：
       - 加载 v11 Pose3DEstimator (MediaPipe BlazePose 3D)
       - 加载 v11 ST-GCN ActionClassifier
       - 加载 v11 Active View CandidateViewGenerator & HabitatViewFilter
    2. 执行端到端主动感知闭环流程：
       RGB Observation (T, H, W, 3)
           ↓
       3D Pose Estimator (30, 33, 3)
           ↓
       Skeleton Normalizer (Root-centered, Torso-scaled)
           ↓
       ST-GCN Action Classifier
           ↓
       Action Probability & Uncertainty Entropy H(p)
           ↓
       Candidate View Generation (32 Viewpoints)
           ↓
       Habitat Feasibility Filtering (NavMesh / Reachability / Visibility)
           ↓
       Feasible Viewpoint Candidates
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.habitat_filter import HabitatViewFilter
from ea_avs_mvp_v11.active_view.visibility_checker import VisibilityChecker
from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.perception.pose3d_estimator import create_pose3d_estimator
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v11")


def run_activeview_v11_pipeline(
    human_pos: List[float] = [0.0, 0.0, 0.0],
    robot_pos: List[float] = [2.0, 0.0, 3.5],
    action_type: str = "walking",
) -> Dict[str, Any]:
    """执行 v11 完整端到端感知与主动候选视点生成流程。"""
    logger.info("=================================================================")
    logger.info("  ACTIVEVIEW v11.0: Autonomous Active Perception Pipeline Demo  ")
    logger.info("=================================================================")

    # 1. 加载 v11 Pose3DEstimator
    logger.info("Step 1: Initializing ea_avs_mvp_v11.perception.MediaPipe3DPoseEstimator...")
    pose_estimator = create_pose3d_estimator("mediapipe")

    # 2. 加载 v11 ST-GCN ActionClassifier
    logger.info("Step 2: Initializing ea_avs_mvp_v11.action_recognition.ActionClassifier...")
    data_root = get_data_root()
    ckpt_path = data_root / "checkpoints" / "v10_st_gcn" / "best_st_gcn_model.pth"
    action_classes = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]
    classifier = ActionClassifier(
        checkpoint_path=ckpt_path if ckpt_path.exists() else None,
        action_classes=action_classes,
    )

    # 3. 模拟获取当前 RGB 时序观测并提取 3D 骨架 (30, 33, 3)
    logger.info("Step 3: Processing RGB video observation for action: '%s'...", action_type)
    # 创建合成测试时序 (T=30, H=256, W=256, 3)
    T = 30
    synth_rgb_frames = [np.full((256, 256, 3), 245, dtype=np.uint8) for _ in range(T)]
    joints_3d, confidences = pose_estimator.estimate_sequence(synth_rgb_frames)
    logger.info("  Extracted 3D Skeleton Sequence: shape=%s, mean_confidence=%.3f",
                joints_3d.shape, float(np.mean(confidences)))

    # 4. 骨架时空归一化与 ST-GCN 动作分类
    logger.info("Step 4: Normalizing skeleton and evaluating ST-GCN Action Classifier...")
    prediction = classifier.predict_sequence(joints_3d, is_normalized=False)

    logger.info("  Current Recognition Result:")
    logger.info("    Predicted Action:    %s (True: %s)", prediction.predicted_label, action_type)
    logger.info("    Top-1 Confidence:    %.4f", prediction.top1_confidence)
    logger.info("    Shannon Entropy H(p):%.4f (Normalized Uncertainty: %.4f)",
                prediction.entropy, prediction.normalized_entropy)

    # 5. 生成 32 个候选主动观察视点
    logger.info("Step 5: Generating candidate viewpoints around human at %s...", human_pos)
    generator = CandidateViewGenerator()
    raw_candidates = generator.generate(human_position=human_pos, robot_current_position=robot_pos)

    # 6. Habitat 物理与视线可行性三阶段过滤
    logger.info("Step 6: Applying Habitat 3-stage feasibility filtering...")
    obstacles = [{"name": "Indoor_Wall", "min": [0.8, -0.5, 0.5], "max": [2.2, 2.5, 1.8]}]
    forbidden_boxes = [{"name": "Forbidden_Zone", "min": [0.7, -1.0, 0.4], "max": [2.3, 1.0, 1.9]}]
    vis_checker = VisibilityChecker(obstacles=obstacles)
    habitat_filter = HabitatViewFilter(
        visibility_checker=vis_checker,
        nav_bounds={"min": [-6, -2, -6], "max": [6, 2, 6], "forbidden_boxes": forbidden_boxes},
    )
    feasible_viewpoints = habitat_filter.filter_viewpoints(
        candidates=raw_candidates,
        human_position=human_pos,
        robot_current_position=robot_pos,
    )

    logger.info("=================================================================")
    logger.info("  Pipeline Completed Successfully! Feasible Candidates: %d / %d  ",
                len(feasible_viewpoints), len(raw_candidates))
    logger.info("=================================================================")

    return {
        "action_type": action_type,
        "predicted_class": prediction.predicted_class,
        "entropy": prediction.entropy,
        "raw_candidate_count": len(raw_candidates),
        "feasible_candidate_count": len(feasible_viewpoints),
        "feasible_viewpoints": [v.to_dict() for v in feasible_viewpoints],
    }


def main():
    parser = argparse.ArgumentParser(description="Run ACTIVEVIEW v11.0 End-to-End Pipeline")
    parser.add_argument("--action", type=str, default="walking", help="Action type to evaluate")
    parser.add_argument("--human_x", type=float, default=0.0)
    parser.add_argument("--human_z", type=float, default=0.0)
    parser.add_argument("--robot_x", type=float, default=2.0)
    parser.add_argument("--robot_z", type=float, default=3.5)
    args = parser.parse_args()

    run_activeview_v11_pipeline(
        human_pos=[args.human_x, 0.0, args.human_z],
        robot_pos=[args.robot_x, 0.0, args.robot_z],
        action_type=args.action,
    )


if __name__ == "__main__":
    main()
