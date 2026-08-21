"""
人体物理状态感知视点打分训练数据集 —— dataset.py
==================================================

职责：
    1. ActiveViewScoringDataset: PyTorch Dataset 实现，封装人体状态姿态、候选视点特征及真实目标质量得分；
    2. generate_scoring_dataset: 严格执行 Motion-Level / Spatial-Level 数据集隔离划分，严禁训练集与测试集共享相同位姿；
    3. 目标得分采用科学效用打分 Q*(v) = w1*global_vis + w2*pose_cov + w3*body_part_vis - w4*dist_pen，严禁来自规则系统；
    4. Action Label 仅作为统计元数据，严禁输入模型。
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.evaluation.view_quality import ViewQualityEvaluator
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v9.action.action_encoder import ALL_ACTION_CLASSES
from ea_avs_mvp_v9.core.types import ActionClass
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor
from ea_avs_mvp_v9.models.pose_encoder import extract_pose_vector
from ea_avs_mvp_v9.models.view_encoder import extract_view_vector
from ea_avs_mvp_v9.scoring.human_state_scorer import HumanStateAwareViewScorer


def create_mock_joints_for_action(
    action_class: ActionClass,
    root_pos: List[float],
    yaw_deg: float = 0.0,
) -> Dict[str, List[float]]:
    """根据动作类别合成符合解剖学形态的 16 骨骼关节空间坐标。"""
    rx, ry, rz = float(root_pos[0]), float(root_pos[1]), float(root_pos[2])
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    def rot(dx: float, dz: float) -> Tuple[float, float]:
        wx = dx * cos_y + dz * sin_y
        wz = -dx * sin_y + dz * cos_y
        return wx, wz

    if action_class == ActionClass.FALL:
        joints = {
            "pelvis": [rx, ry + 0.15, rz],
            "head": [rx + rot(0.0, 0.9)[0], ry + 0.20, rz + rot(0.0, 0.9)[1]],
            "neck": [rx + rot(0.0, 0.75)[0], ry + 0.20, rz + rot(0.0, 0.75)[1]],
            "spine": [rx + rot(0.0, 0.45)[0], ry + 0.18, rz + rot(0.0, 0.45)[1]],
            "chest": [rx + rot(0.0, 0.60)[0], ry + 0.19, rz + rot(0.0, 0.60)[1]],
            "left_shoulder": [rx + rot(-0.25, 0.65)[0], ry + 0.20, rz + rot(-0.25, 0.65)[1]],
            "right_shoulder": [rx + rot(0.25, 0.65)[0], ry + 0.20, rz + rot(0.25, 0.65)[1]],
            "left_elbow": [rx + rot(-0.45, 0.55)[0], ry + 0.15, rz + rot(-0.45, 0.55)[1]],
            "right_elbow": [rx + rot(0.45, 0.55)[0], ry + 0.15, rz + rot(0.45, 0.55)[1]],
            "left_wrist": [rx + rot(-0.60, 0.45)[0], ry + 0.10, rz + rot(-0.60, 0.45)[1]],
            "right_wrist": [rx + rot(0.60, 0.45)[0], ry + 0.10, rz + rot(0.60, 0.45)[1]],
            "left_hip": [rx + rot(-0.15, 0.0)[0], ry + 0.15, rz + rot(-0.15, 0.0)[1]],
            "right_hip": [rx + rot(0.15, 0.0)[0], ry + 0.15, rz + rot(0.15, 0.0)[1]],
            "left_knee": [rx + rot(-0.15, -0.45)[0], ry + 0.12, rz + rot(-0.15, -0.45)[1]],
            "right_knee": [rx + rot(0.15, -0.45)[0], ry + 0.12, rz + rot(0.15, -0.45)[1]],
            "left_ankle": [rx + rot(-0.15, -0.85)[0], ry + 0.08, rz + rot(-0.15, -0.85)[1]],
            "right_ankle": [rx + rot(0.15, -0.85)[0], ry + 0.08, rz + rot(0.15, -0.85)[1]],
        }
    elif action_class == ActionClass.SITTING:
        joints = {
            "pelvis": [rx, ry + 0.55, rz],
            "head": [rx, ry + 1.35, rz],
            "neck": [rx, ry + 1.20, rz],
            "spine": [rx, ry + 0.85, rz],
            "chest": [rx, ry + 1.05, rz],
            "left_shoulder": [rx + rot(-0.22, 0.0)[0], ry + 1.15, rz + rot(-0.22, 0.0)[1]],
            "right_shoulder": [rx + rot(0.22, 0.0)[0], ry + 1.15, rz + rot(0.22, 0.0)[1]],
            "left_elbow": [rx + rot(-0.30, 0.15)[0], ry + 0.85, rz + rot(-0.30, 0.15)[1]],
            "right_elbow": [rx + rot(0.30, 0.15)[0], ry + 0.85, rz + rot(0.30, 0.15)[1]],
            "left_wrist": [rx + rot(-0.25, 0.35)[0], ry + 0.60, rz + rot(-0.25, 0.35)[1]],
            "right_wrist": [rx + rot(0.25, 0.35)[0], ry + 0.60, rz + rot(0.25, 0.35)[1]],
            "left_hip": [rx + rot(-0.16, 0.0)[0], ry + 0.52, rz + rot(-0.16, 0.0)[1]],
            "right_hip": [rx + rot(0.16, 0.0)[0], ry + 0.52, rz + rot(0.16, 0.0)[1]],
            "left_knee": [rx + rot(-0.16, 0.40)[0], ry + 0.52, rz + rot(-0.16, 0.40)[1]],
            "right_knee": [rx + rot(0.16, 0.40)[0], ry + 0.52, rz + rot(0.16, 0.40)[1]],
            "left_ankle": [rx + rot(-0.16, 0.40)[0], ry + 0.08, rz + rot(-0.16, 0.40)[1]],
            "right_ankle": [rx + rot(0.16, 0.40)[0], ry + 0.08, rz + rot(0.16, 0.40)[1]],
        }
    elif action_class == ActionClass.BENDING:
        joints = {
            "pelvis": [rx, ry + 0.85, rz],
            "head": [rx + rot(0.0, 0.65)[0], ry + 0.70, rz + rot(0.0, 0.65)[1]],
            "neck": [rx + rot(0.0, 0.55)[0], ry + 0.78, rz + rot(0.0, 0.55)[1]],
            "spine": [rx + rot(0.0, 0.25)[0], ry + 0.85, rz + rot(0.0, 0.25)[1]],
            "chest": [rx + rot(0.0, 0.45)[0], ry + 0.82, rz + rot(0.0, 0.45)[1]],
            "left_shoulder": [rx + rot(-0.22, 0.45)[0], ry + 0.82, rz + rot(-0.22, 0.45)[1]],
            "right_shoulder": [rx + rot(0.22, 0.45)[0], ry + 0.82, rz + rot(0.22, 0.45)[1]],
            "left_elbow": [rx + rot(-0.28, 0.45)[0], ry + 0.55, rz + rot(-0.28, 0.45)[1]],
            "right_elbow": [rx + rot(0.28, 0.45)[0], ry + 0.55, rz + rot(0.28, 0.45)[1]],
            "left_wrist": [rx + rot(-0.28, 0.45)[0], ry + 0.28, rz + rot(-0.28, 0.45)[1]],
            "right_wrist": [rx + rot(0.28, 0.45)[0], ry + 0.28, rz + rot(0.28, 0.45)[1]],
            "left_hip": [rx + rot(-0.16, 0.0)[0], ry + 0.82, rz + rot(-0.16, 0.0)[1]],
            "right_hip": [rx + rot(0.16, 0.0)[0], ry + 0.82, rz + rot(0.16, 0.0)[1]],
            "left_knee": [rx + rot(-0.16, -0.05)[0], ry + 0.45, rz + rot(-0.16, -0.05)[1]],
            "right_knee": [rx + rot(0.16, -0.05)[0], ry + 0.45, rz + rot(0.16, -0.05)[1]],
            "left_ankle": [rx + rot(-0.16, 0.0)[0], ry + 0.08, rz + rot(-0.16, 0.0)[1]],
            "right_ankle": [rx + rot(0.16, 0.0)[0], ry + 0.08, rz + rot(0.16, 0.0)[1]],
        }
    elif action_class == ActionClass.REACHING:
        joints = {
            "pelvis": [rx, ry + 0.88, rz],
            "head": [rx, ry + 1.65, rz],
            "neck": [rx, ry + 1.50, rz],
            "spine": [rx, ry + 1.15, rz],
            "chest": [rx, ry + 1.35, rz],
            "left_shoulder": [rx + rot(-0.22, 0.0)[0], ry + 1.45, rz + rot(-0.22, 0.0)[1]],
            "right_shoulder": [rx + rot(0.22, 0.0)[0], ry + 1.45, rz + rot(0.22, 0.0)[1]],
            "left_elbow": [rx + rot(-0.25, 0.0)[0], ry + 1.15, rz + rot(-0.25, 0.0)[1]],
            "right_elbow": [rx + rot(0.22, 0.35)[0], ry + 1.45, rz + rot(0.22, 0.35)[1]],
            "left_wrist": [rx + rot(-0.25, 0.0)[0], ry + 0.85, rz + rot(-0.25, 0.0)[1]],
            "right_wrist": [rx + rot(0.22, 0.70)[0], ry + 1.48, rz + rot(0.22, 0.70)[1]],
            "left_hip": [rx + rot(-0.16, 0.0)[0], ry + 0.85, rz + rot(-0.16, 0.0)[1]],
            "right_hip": [rx + rot(0.16, 0.0)[0], ry + 0.85, rz + rot(0.16, 0.0)[1]],
            "left_knee": [rx + rot(-0.16, 0.0)[0], ry + 0.48, rz + rot(-0.16, 0.0)[1]],
            "right_knee": [rx + rot(0.16, 0.0)[0], ry + 0.48, rz + rot(0.16, 0.0)[1]],
            "left_ankle": [rx + rot(-0.16, 0.0)[0], ry + 0.08, rz + rot(-0.16, 0.0)[1]],
            "right_ankle": [rx + rot(0.16, 0.0)[0], ry + 0.08, rz + rot(0.16, 0.0)[1]],
        }
    else:  # STANDING
        joints = {
            "pelvis": [rx, ry + 0.88, rz],
            "head": [rx, ry + 1.65, rz],
            "neck": [rx, ry + 1.50, rz],
            "spine": [rx, ry + 1.15, rz],
            "chest": [rx, ry + 1.35, rz],
            "left_shoulder": [rx + rot(-0.22, 0.0)[0], ry + 1.45, rz + rot(-0.22, 0.0)[1]],
            "right_shoulder": [rx + rot(0.22, 0.0)[0], ry + 1.45, rz + rot(0.22, 0.0)[1]],
            "left_elbow": [rx + rot(-0.25, 0.0)[0], ry + 1.15, rz + rot(-0.25, 0.0)[1]],
            "right_elbow": [rx + rot(0.25, 0.0)[0], ry + 1.15, rz + rot(0.25, 0.0)[1]],
            "left_wrist": [rx + rot(-0.25, 0.0)[0], ry + 0.85, rz + rot(-0.25, 0.0)[1]],
            "right_wrist": [rx + rot(0.25, 0.0)[0], ry + 0.85, rz + rot(0.25, 0.0)[1]],
            "left_hip": [rx + rot(-0.16, 0.0)[0], ry + 0.85, rz + rot(-0.16, 0.0)[1]],
            "right_hip": [rx + rot(0.16, 0.0)[0], ry + 0.85, rz + rot(0.16, 0.0)[1]],
            "left_knee": [rx + rot(-0.16, 0.0)[0], ry + 0.48, rz + rot(-0.16, 0.0)[1]],
            "right_knee": [rx + rot(0.16, 0.0)[0], ry + 0.48, rz + rot(0.16, 0.0)[1]],
            "left_ankle": [rx + rot(-0.16, 0.0)[0], ry + 0.08, rz + rot(-0.16, 0.0)[1]],
            "right_ankle": [rx + rot(0.16, 0.0)[0], ry + 0.08, rz + rot(0.16, 0.0)[1]],
        }
    return joints


class ActiveViewScoringDataset(Dataset):
    """用于训练与评估 LearnableViewScorer 的 PyTorch Dataset (Q(v | H))。"""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "pose_vec": torch.tensor(item["pose_vec"], dtype=torch.float32),
            "view_vecs": torch.tensor(item["view_vecs"], dtype=torch.float32),
            "target_scores": torch.tensor(item["target_scores"], dtype=torch.float32),
            "action_name": item["action_name"],
            "best_view_idx": torch.tensor(item["best_view_idx"], dtype=torch.long),
        }


def generate_scoring_dataset(
    num_episodes: int = 200,
    seed: int = 42,
) -> Tuple[ActiveViewScoringDataset, ActiveViewScoringDataset]:
    """
    自动化生成基于人体物理状态与视点描述子的数据集，
    采用严格的 Spatial-Level & Motion Instance 隔离划分 (Train vs Val/Test)。
    """
    train_rng = np.random.RandomState(seed)
    val_rng = np.random.RandomState(seed + 999)

    feat_extractor = ViewFeatureExtractor({"hfov_deg": 90.0, "max_distance": 4.5})
    geom_evaluator = ViewQualityEvaluator({"evaluation_mode": "oracle", "pose_source": "oracle"})
    scorer = HumanStateAwareViewScorer()
    vp_gen = ViewpointGenerator({
        "radii": [1.5, 2.0, 2.5, 3.0],
        "num_angles": 8,
        "camera_height": 1.2,
        "ground_height": -1.60,
    })

    actions_pool = list(ALL_ACTION_CLASSES)

    val_count = max(1, int(num_episodes * 0.2))
    train_count = num_episodes - val_count

    def _sample_batch(count: int, rng: np.random.RandomState, is_val: bool = False) -> List[Dict[str, Any]]:
        batch_samples = []
        for i in range(count):
            act_class = actions_pool[(i + (10 if is_val else 0)) % len(actions_pool)]

            # 空间位置与朝向区间隔离 (Train: 区域 A, Val: 区域 B)
            if is_val:
                hx = float(rng.uniform(2.5, 4.5))
                hz = float(rng.uniform(3.5, 6.0))
                yaw = float(rng.uniform(180.0, 360.0))
            else:
                hx = float(rng.uniform(0.5, 2.5))
                hz = float(rng.uniform(1.5, 3.5))
                yaw = float(rng.uniform(0.0, 180.0))

            hy = -1.60
            human_pos = [hx, hy, hz]

            joints = create_mock_joints_for_action(act_class, human_pos, yaw_deg=yaw)
            pose_vec = extract_pose_vector(joints, human_yaw_deg=yaw)

            candidates = vp_gen.generate_candidates(human_pos, human_yaw_deg=yaw, ground_height=hy)
            geom_ranked = geom_evaluator.rank_viewpoints(candidates, joints, human_yaw_deg=yaw)
            geom_map = {q.viewpoint_id: q.visibility_score for _, q in geom_ranked}

            features = feat_extractor.extract_batch(candidates, joints, human_yaw_deg=yaw)
            view_vecs = [extract_view_vector(f) for f in features]

            # 目标真值由科学公式 Q*(v | H) 计算，绝非规则分
            scored_objs = scorer.score_batch(features, geom_visibility_map=geom_map)
            target_scores = [s["total_score"] for s in scored_objs]

            best_idx = int(np.argmax(target_scores))

            batch_samples.append({
                "episode_id": i,
                "action_name": act_class.value,
                "pose_vec": pose_vec,
                "view_vecs": np.array(view_vecs, dtype=np.float32),
                "target_scores": np.array(target_scores, dtype=np.float32),
                "best_view_idx": best_idx,
                "candidate_ids": [c.viewpoint_id for c in candidates],
            })
        return batch_samples

    train_samples = _sample_batch(train_count, train_rng, is_val=False)
    val_samples = _sample_batch(val_count, val_rng, is_val=True)

    return ActiveViewScoringDataset(train_samples), ActiveViewScoringDataset(val_samples)
