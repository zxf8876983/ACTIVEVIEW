"""
感知质量驱动的信息增益训练数据集 —— dataset.py
==================================================

职责：
    1. ActiveViewScoringDataset: PyTorch Dataset 实现，封装当前不完整感知状态 (71d)、候选视点描述子 (13d) 及真实信息增益目标 Gain(v)；
    2. generate_scoring_dataset: 采用严谨的感知退化模拟与多场景、空间与实例正交隔离划分 (Train vs Val)；
    3. 目标得分严格由视角迁移前后的观测感知质量增量计算：
       Gain(v) = ObservationQuality_after(v) - ObservationQuality_before(v_curr)
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v8.viewpoint.viewpoint_generator import ViewpointGenerator
from ea_avs_mvp_v9.action.action_encoder import ALL_ACTION_CLASSES
from ea_avs_mvp_v9.core.types import ActionClass, ObservationState
from ea_avs_mvp_v9.features.observation_simulator import ObservationSimulator, compute_observation_quality
from ea_avs_mvp_v9.features.view_feature_extractor import ViewFeatureExtractor
from ea_avs_mvp_v9.models.observation_encoder import extract_observation_vector
from ea_avs_mvp_v9.models.view_encoder import extract_view_vector


def create_mock_joints_for_action(
    action_class: ActionClass,
    root_pos: List[float],
    yaw_deg: float = 0.0,
) -> Dict[str, List[float]]:
    """根据动作类别合成符合解剖学形态的 16 骨骼关节空间坐标 (仅供仿真与生成监督标签使用)。"""
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
    """用于训练与评估 PerceptionAwareViewScorer 的 PyTorch Dataset。"""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "obs_vec": torch.tensor(item["obs_vec"], dtype=torch.float32),
            "pose_vec": torch.tensor(item["obs_vec"], dtype=torch.float32),  # 别名兼容
            "view_vecs": torch.tensor(item["view_vecs"], dtype=torch.float32),
            "target_scores": torch.tensor(item["target_scores"], dtype=torch.float32),
            "action_name": item.get("action_name", "unknown"),
            "best_view_idx": torch.tensor(item["best_view_idx"], dtype=torch.long),
        }


def generate_scoring_dataset(
    num_episodes: int = 200,
    seed: int = 42,
) -> Tuple[ActiveViewScoringDataset, ActiveViewScoringDataset]:
    """
    自动化生成基于当前感知状态与信息增益的数据集 (严格空间与实例隔离)。
    """
    train_rng = np.random.RandomState(seed)
    val_rng = np.random.RandomState(seed + 999)

    obs_sim = ObservationSimulator()
    feat_extractor = ViewFeatureExtractor({"hfov_deg": 90.0, "max_distance": 4.5})
    vp_gen = ViewpointGenerator({
        "radii": [1.5, 2.0, 2.5, 3.0],
        "num_angles": 8,
        "camera_height": 1.2,
        "ground_height": -1.60,
    })

    actions_pool = list(ALL_ACTION_CLASSES)
    degradation_modes = ["self_occlusion", "furniture_occlusion", "heavy_noise", "missing_keypoints", "clean"]

    val_count = max(1, int(num_episodes * 0.2))
    train_count = num_episodes - val_count

    def _sample_batch(count: int, rng: np.random.RandomState, is_val: bool = False) -> List[Dict[str, Any]]:
        batch_samples = []
        for i in range(count):
            act_class = actions_pool[(i + (10 if is_val else 0)) % len(actions_pool)]
            deg_mode = degradation_modes[(i + (3 if is_val else 0)) % len(degradation_modes)]

            # 空间位置与朝向区间隔离
            if is_val:
                hx = float(rng.uniform(2.5, 5.0))
                hz = float(rng.uniform(3.5, 6.5))
                yaw = float(rng.uniform(180.0, 360.0))
                curr_cam_offset = [float(rng.uniform(-2.2, 2.2)), 1.20, float(rng.uniform(1.5, 3.2))]
            else:
                hx = float(rng.uniform(0.5, 2.5))
                hz = float(rng.uniform(1.5, 3.5))
                yaw = float(rng.uniform(0.0, 180.0))
                curr_cam_offset = [float(rng.uniform(-2.2, 2.2)), 1.20, float(rng.uniform(1.5, 3.2))]

            hy = -1.60
            human_pos = [hx, hy, hz]
            curr_cam_pos = [hx + curr_cam_offset[0], hy + curr_cam_offset[1], hz + curr_cam_offset[2]]

            joints = create_mock_joints_for_action(act_class, human_pos, yaw_deg=yaw)

            # 模拟当前视点下的不完整感知状态
            obs_curr = obs_sim.simulate_observation(
                gt_joints=joints,
                camera_pos=curr_cam_pos,
                human_pos=human_pos,
                human_yaw_deg=yaw,
                degradation_mode=deg_mode,
                rng=rng,
            )
            obs_vec = extract_observation_vector(obs_curr)
            curr_dist = math.dist([curr_cam_pos[0], curr_cam_pos[2]], [hx, hz])
            q_curr = compute_observation_quality(obs_curr, dist=curr_dist)

            # 生成候选视角
            candidates = vp_gen.generate_candidates(human_pos, human_yaw_deg=yaw, ground_height=hy)
            features = feat_extractor.extract_batch(candidates, obs_curr.estimated_joints_3d, human_yaw_deg=yaw)
            view_vecs = [extract_view_vector(f) for f in features]

            # 计算每个候选视角迁移带来的真实信息增益 Gain(v) = Quality_after(v) - Quality_before
            target_gains = []
            for c_vp in candidates:
                # 候选视点观测同样通过 ObservationSimulator 模拟 (绝非直接读取 GT)
                obs_cand = obs_sim.simulate_observation(
                    gt_joints=joints,
                    camera_pos=c_vp.position,
                    human_pos=human_pos,
                    human_yaw_deg=yaw,
                    degradation_mode="auto",
                    rng=rng,
                )
                q_after = compute_observation_quality(obs_cand, dist=c_vp.radius)
                gain = float(np.clip(q_after - q_curr + 0.20, 0.0, 1.0))
                target_gains.append(gain)

            best_idx = int(np.argmax(target_gains))

            batch_samples.append({
                "episode_id": i,
                "action_name": act_class.value,
                "degradation_mode": deg_mode,
                "obs_vec": obs_vec,
                "view_vecs": np.array(view_vecs, dtype=np.float32),
                "target_scores": np.array(target_gains, dtype=np.float32),
                "best_view_idx": best_idx,
                "candidate_ids": [c.viewpoint_id for c in candidates],
            })
        return batch_samples

    train_samples = _sample_batch(train_count, train_rng, is_val=False)
    val_samples = _sample_batch(val_count, val_rng, is_val=True)

    return ActiveViewScoringDataset(train_samples), ActiveViewScoringDataset(val_samples)
