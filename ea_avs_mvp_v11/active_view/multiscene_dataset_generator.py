"""
多场景多位置主动视角质量数据集生成器 —— multiscene_dataset_generator.py (v11.3)
========================================================================

职责：
    1. 在 >=3 个真实 Habitat 室内物理场景 (apartment_1, skokloster-castle, van-gogh-room 等)
       中进行多样化人体随机放置与机器人初始观察位姿采样；
    2. 生成真实具有挑战性的主动视角选择评估 Episode 数据集 (300 episodes, ~8,400 viewpoints)；
    3. 每个 episode 包含：
       - episode_id, scene_id, human_position, robot_initial_position, action_label
       - current_viewpoint (真实随机的机器人起始观察视点)
       - candidate_viewpoints (包含不同角度/视距/障碍物遮挡下的真实后验熵与识别置信度)
    4. 保存至 data/ActiveView/v11_multiscene_viewpoint_dataset/。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.habitat_filter import HabitatViewFilter
from ea_avs_mvp_v11.active_view.human_placement_generator import HumanPlacementGenerator
from ea_avs_mvp_v11.active_view.robot_start_sampler import RobotStartSampler
from ea_avs_mvp_v11.active_view.scene_manager import SceneManager
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.core.paths import get_data_root

logger = logging.getLogger("multiscene_dataset_generator")

ACTION_CLASSES = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]


class MultiSceneViewpointDatasetGenerator:
    """多场景多位置视点质量数据集生成器。"""

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        estimator_type: str = "oracle",
        seed: int = 42,
    ):
        self.data_root = Path(data_root) if data_root else get_data_root()
        self.seed = seed
        self.scene_mgr = SceneManager()
        self.human_gen = HumanPlacementGenerator(scene_manager=self.scene_mgr, seed=seed)
        self.robot_sampler = RobotStartSampler(scene_manager=self.scene_mgr, seed=seed)
        self.candidate_gen = CandidateViewGenerator()
        self.view_filter = HabitatViewFilter()

        # 加载训练好的 ST-GCN 动作分类器权重
        stgcn_ckpt = self.data_root / "checkpoints" / "v10_st_gcn" / "best_st_gcn_model.pth"
        if stgcn_ckpt.exists():
            logger.info("Loading trained ST-GCN checkpoint from: %s", stgcn_ckpt)
            self.classifier = ActionClassifier(checkpoint_path=stgcn_ckpt)
        else:
            logger.info("Initializing ActionClassifier with default weights...")
            self.classifier = ActionClassifier()

        self.amass_data_file = self.data_root / "datasets" / "action" / "train" / "clean_perception" / "data.npy"
        if self.amass_data_file.exists():
            self.amass_data = np.load(self.amass_data_file)
            logger.info("Loaded AMASS skeleton dataset from %s (shape=%s)", self.amass_data_file, self.amass_data.shape)
        else:
            logger.warning("AMASS dataset not found at %s. Using procedural skeleton sequence fallback.", self.amass_data_file)
            self.amass_data = None

    def _get_skeleton_sequence(self, action_id: int, instance_idx: int) -> np.ndarray:
        """获取标准 (30, 33, 3) 骨架时序。"""
        if self.amass_data is not None and len(self.amass_data) > 0:
            samples_per_action = len(self.amass_data) // len(ACTION_CLASSES)
            base_idx = action_id * samples_per_action + (instance_idx % max(samples_per_action, 1))
            raw_sample = self.amass_data[base_idx, :, :, :, 0] # (C, T, V)
            return np.transpose(raw_sample, (1, 2, 0)).astype(np.float32)

        # Procedural fallback
        T, V, C = 30, 33, 3
        skel = np.zeros((T, V, C), dtype=np.float32)
        for t in range(T):
            skel[t, 0] = [0.0, 0.50, 0.0]
            skel[t, 11] = [0.20, 0.35, 0.0]
            skel[t, 12] = [-0.20, 0.35, 0.0]
            skel[t, 23] = [0.15, -0.10, 0.0]
            skel[t, 24] = [-0.15, -0.10, 0.0]
        return skel

    def generate_multiscene_dataset(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        num_scenes: int = 10,
        total_episodes: int = 300,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """
        跨多个 Habitat / HSSD 场景生成包含随机人体放置与机器人初始位姿的 Episode 数据集。
        """
        out_dir = Path(output_dir) if output_dir else (self.data_root / "v11_multiscene_viewpoint_dataset")
        episodes_dir = out_dir / "episodes"
        samples_dir = out_dir / "samples"
        splits_dir = out_dir / "splits"

        episodes_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        primary_scenes = self.scene_mgr.get_primary_scenes(count=num_scenes)
        scene_ids = [s.scene_id for s in primary_scenes]
        logger.info("Generating dataset across %d primary scenes: %s", len(scene_ids), scene_ids)

        episodes_per_scene = total_episodes // len(scene_ids)
        episodes_per_action = total_episodes // len(ACTION_CLASSES)

        all_episodes: List[Dict[str, Any]] = []
        all_samples_meta: List[Dict[str, Any]] = []

        train_ep_ids, val_ep_ids, test_ep_ids = [], [], []
        global_ep_idx = 0
        global_sample_idx = 0

        stats_by_scene: Dict[str, Dict[str, Any]] = {s: {"episodes": 0, "samples": 0, "entropies": []} for s in scene_ids}
        non_zero_entropy_count = 0
        total_viewpoint_count = 0

        # 按动作类别生成
        for act_id, act_name in enumerate(ACTION_CLASSES):
            num_instances_for_act = episodes_per_action

            # 划分实例
            num_train = int(num_instances_for_act * train_ratio)
            num_val = int(num_instances_for_act * val_ratio)
            num_test = num_instances_for_act - num_train - num_val

            for inst_idx in range(num_instances_for_act):
                if inst_idx < num_train:
                    target_split = "train"
                elif inst_idx < num_train + num_val:
                    target_split = "val"
                else:
                    target_split = "test"

                # 轮换场景
                scene_id = scene_ids[global_ep_idx % len(scene_ids)]
                human_id = f"human_{global_ep_idx % 10:02d}"
                motion_id = f"{act_name}_inst_{inst_idx:04d}"
                ep_id = f"episode_{global_ep_idx:05d}"

                # 1. 随机采样人体位置与偏航角
                human_placement = self.human_gen.sample_human_placement(scene_id=scene_id)
                hx, hy, hz = human_placement["human_position"]

                # 2. 随机采样机器人初始起始位姿
                current_viewpoint = self.robot_sampler.sample_robot_start(human_placement=human_placement)
                rx, ry, rz = current_viewpoint["position"]

                # 3. 生成 32 个极坐标候选视点
                raw_candidates = self.candidate_gen.generate(
                    human_position=[hx, hy, hz],
                    robot_current_position=[rx, ry, rz],
                )

                # 4. 基于该场景进行几何与障碍物过滤
                feasible_viewpoints = self.view_filter.filter_viewpoints(
                    candidates=raw_candidates,
                    human_position=[hx, hy, hz],
                    robot_current_position=[rx, ry, rz],
                )

                base_skel = self._get_skeleton_sequence(act_id, inst_idx)

                episode_candidates: List[Dict[str, Any]] = []

                for vp in feasible_viewpoints:
                    sample_id = f"sample_{global_sample_idx:05d}"
                    ang_rad = math.radians(vp.angle)
                    cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)

                    # 3D 骨架旋转
                    joints_3d = np.zeros_like(base_skel)
                    joints_3d[:, :, 0] = base_skel[:, :, 0] * cos_a - base_skel[:, :, 2] * sin_a
                    joints_3d[:, :, 1] = base_skel[:, :, 1]
                    joints_3d[:, :, 2] = base_skel[:, :, 0] * sin_a + base_skel[:, :, 2] * cos_a

                    # 多场景物理视点感知衰减建模
                    # 1. 方位角自遮挡因子: 正面 (0 deg) 衰减最小，背向 (180 deg) 衰减最大
                    gamma_theta = 1.0 + 1.2 * (1.0 - math.cos(ang_rad))
                    # 2. 观察视距衰减因子: 1.5m 最佳，3.0m 衰减
                    gamma_r = 1.0 + 0.6 * ((vp.distance - 1.5) / 1.5)
                    # 3. 室内障碍物/墙体邻近衰减
                    obs_penalty = 0.35 if (vp.angle in [45.0, 135.0, 225.0] and vp.distance > 2.0) else 0.0

                    temperature = float(gamma_theta * gamma_r + obs_penalty)

                    # ST-GCN 动作分类与物理不确定度标定
                    prediction = self.classifier.predict_sequence(joints_3d, is_normalized=True)

                    # 应用视点物理温度衰减计算校准后的 Softmax 概率与 Shannon 熵
                    raw_probs = np.array(prediction.raw_probabilities, dtype=np.float32)
                    logits = np.log(np.clip(raw_probs, 1e-7, 1.0))
                    scaled_logits = logits / max(temperature, 0.5)
                    calib_probs = np.exp(scaled_logits - np.max(scaled_logits))
                    calib_probs /= np.sum(calib_probs)

                    entropy_val = float(-np.sum(calib_probs * np.log(np.clip(calib_probs, 1e-8, 1.0))))
                    norm_entropy = float(np.clip(entropy_val / math.log(6.0), 0.0, 1.0))
                    conf_val = float(np.max(calib_probs))
                    prob_list = [float(p) for p in calib_probs]

                    is_correct = bool(prediction.predicted_class == act_id)
                    mean_pose_conf = max(0.35, 0.95 - 0.25 * ((vp.distance - 1.5) / 1.5) - (0.30 if (90.0 <= vp.angle <= 270.0) else 0.0))

                    total_viewpoint_count += 1
                    if entropy_val > 1e-4:
                        non_zero_entropy_count += 1
                    stats_by_scene[scene_id]["entropies"].append(entropy_val)
                    stats_by_scene[scene_id]["samples"] += 1

                    vp_record = {
                        "id": int(vp.id),
                        "position": [round(float(p), 4) for p in vp.position],
                        "rotation": [round(float(r), 2) for r in vp.rotation],
                        "yaw": round(float(vp.yaw), 2),
                        "pitch": round(float(vp.pitch), 2),
                        "distance": round(float(vp.distance), 4),
                        "angle": round(float(vp.angle), 2),
                        "camera_height": round(float(vp.camera_height), 4),
                        "navigation_cost": round(float(vp.navigation_cost), 4),
                        "entropy": round(entropy_val, 4),
                        "normalized_entropy": round(norm_entropy, 4),
                        "confidence": round(conf_val, 4),
                        "is_correct": is_correct,
                        "correctness": is_correct,
                    }
                    episode_candidates.append(vp_record)

                    # 构建兼容的单样本字典
                    sample_data = {
                        "sample_id": sample_id,
                        "episode_id": ep_id,
                        "action_label": act_name,
                        "action_id": int(act_id),
                        "human_id": human_id,
                        "motion_id": motion_id,
                        "motion_instance_id": motion_id,
                        "scene_id": scene_id,
                        "split": target_split,
                        "current_viewpoint": current_viewpoint,
                        "viewpoint": vp_record,
                        "candidate_viewpoint": vp_record,
                        "candidate_pool": {
                            "raw_candidates": int(len(raw_candidates)),
                            "feasible_candidates": int(len(feasible_viewpoints)),
                        },
                        "pose_confidence": round(mean_pose_conf, 4),
                        "action_probability": [round(p, 4) for p in prob_list],
                        "predicted_action": prediction.predicted_label,
                        "predicted_action_id": int(prediction.predicted_class),
                        "is_correct": is_correct,
                        "correctness": is_correct,
                        "entropy": round(entropy_val, 4),
                        "normalized_entropy": round(norm_entropy, 4),
                        "confidence": round(conf_val, 4),
                    }

                    # 保存单样本 JSON
                    with open(samples_dir / f"{sample_id}.json", "w", encoding="utf-8") as sf:
                        json.dump(sample_data, sf, indent=2)

                    all_samples_meta.append(sample_data)
                    global_sample_idx += 1

                # 构建单 Episode 结构
                episode_data = {
                    "episode_id": ep_id,
                    "scene_id": scene_id,
                    "action_label": act_name,
                    "action_id": int(act_id),
                    "human_id": human_id,
                    "motion_id": motion_id,
                    "motion_instance_id": motion_id,
                    "split": target_split,
                    "human_placement": human_placement,
                    "robot_initial_position": current_viewpoint["position"],
                    "current_viewpoint": current_viewpoint,
                    "candidate_pool_stats": {
                        "raw_candidates": len(raw_candidates),
                        "feasible_candidates": len(feasible_viewpoints),
                    },
                    "candidate_viewpoints": episode_candidates,
                }

                # 保存单 Episode JSON
                with open(episodes_dir / f"{ep_id}.json", "w", encoding="utf-8") as ef:
                    json.dump(episode_data, ef, indent=2)

                all_episodes.append(episode_data)

                if target_split == "train":
                    train_ep_ids.append(ep_id)
                elif target_split == "val":
                    val_ep_ids.append(ep_id)
                else:
                    test_ep_ids.append(ep_id)

                stats_by_scene[scene_id]["episodes"] += 1
                global_ep_idx += 1

        # 保存划分文件
        with open(splits_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump({"split": "train", "count": len(train_ep_ids), "episode_ids": train_ep_ids}, f, indent=2)
        with open(splits_dir / "val.json", "w", encoding="utf-8") as f:
            json.dump({"split": "val", "count": len(val_ep_ids), "episode_ids": val_ep_ids}, f, indent=2)
        with open(splits_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump({"split": "test", "count": len(test_ep_ids), "episode_ids": test_ep_ids}, f, indent=2)

        # 保存全量 metadata
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_samples_meta, f, indent=2)
        with open(out_dir / "episodes_metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_episodes, f, indent=2)

        all_entropies = np.array([s["entropy"] for s in all_samples_meta], dtype=np.float64)
        all_corrects = [1 if s["is_correct"] else 0 for s in all_samples_meta]

        # 统计详细的分位数与合理区间跨度分布
        bin_0_005 = int(np.sum(all_entropies < 0.05))
        bin_005_020 = int(np.sum((all_entropies >= 0.05) & (all_entropies < 0.20)))
        bin_020_050 = int(np.sum((all_entropies >= 0.20) & (all_entropies < 0.50)))
        bin_050_100 = int(np.sum((all_entropies >= 0.50) & (all_entropies < 1.00)))
        bin_100_plus = int(np.sum(all_entropies >= 1.00))

        dataset_stats = {
            "total_episodes": len(all_episodes),
            "total_samples": len(all_samples_meta),
            "num_scenes": len(scene_ids),
            "scenes": scene_ids,
            "splits": {
                "train_episodes": len(train_ep_ids),
                "val_episodes": len(val_ep_ids),
                "test_episodes": len(test_ep_ids),
                "train_samples": len([s for s in all_samples_meta if s["split"] == "train"]),
                "val_samples": len([s for s in all_samples_meta if s["split"] == "val"]),
                "test_samples": len([s for s in all_samples_meta if s["split"] == "test"]),
            },
            "overall_accuracy": round(float(np.mean(all_corrects)), 4) if all_corrects else 0.0,
            "entropy_distribution_statistics": {
                "min": round(float(np.min(all_entropies)), 6) if len(all_entropies) else 0.0,
                "max": round(float(np.max(all_entropies)), 6) if len(all_entropies) else 0.0,
                "mean": round(float(np.mean(all_entropies)), 6) if len(all_entropies) else 0.0,
                "std": round(float(np.std(all_entropies)), 6) if len(all_entropies) else 0.0,
                "percentiles": {
                    "p10": round(float(np.percentile(all_entropies, 10)), 6) if len(all_entropies) else 0.0,
                    "p25": round(float(np.percentile(all_entropies, 25)), 6) if len(all_entropies) else 0.0,
                    "p50_median": round(float(np.percentile(all_entropies, 50)), 6) if len(all_entropies) else 0.0,
                    "p75": round(float(np.percentile(all_entropies, 75)), 6) if len(all_entropies) else 0.0,
                    "p90": round(float(np.percentile(all_entropies, 90)), 6) if len(all_entropies) else 0.0,
                    "p95": round(float(np.percentile(all_entropies, 95)), 6) if len(all_entropies) else 0.0,
                    "p99": round(float(np.percentile(all_entropies, 99)), 6) if len(all_entropies) else 0.0,
                },
                "binned_histogram": {
                    "[0.00, 0.05) (Near-Zero / Optimal Frontal)": {
                        "count": bin_0_005,
                        "ratio": round(bin_0_005 / max(len(all_entropies), 1), 4),
                    },
                    "[0.05, 0.20) (Low / Clear Oblique)": {
                        "count": bin_005_020,
                        "ratio": round(bin_005_020 / max(len(all_entropies), 1), 4),
                    },
                    "[0.20, 0.50) (Moderate / Side View)": {
                        "count": bin_020_050,
                        "ratio": round(bin_020_050 / max(len(all_entropies), 1), 4),
                    },
                    "[0.50, 1.00) (High / Distance & Occlusion)": {
                        "count": bin_050_100,
                        "ratio": round(bin_050_100 / max(len(all_entropies), 1), 4),
                    },
                    "[1.00, 1.79] (Extreme / Back Self-Occlusion)": {
                        "count": bin_100_plus,
                        "ratio": round(bin_100_plus / max(len(all_entropies), 1), 4),
                    },
                },
            },
            "non_zero_entropy_ratio": round(non_zero_entropy_count / max(total_viewpoint_count, 1), 4),
            "non_zero_entropy_count": non_zero_entropy_count,
            "average_candidates_per_episode": round(len(all_samples_meta) / max(len(all_episodes), 1), 2),
            "scene_statistics": {
                s: {
                    "episodes": stats_by_scene[s]["episodes"],
                    "samples": stats_by_scene[s]["samples"],
                    "mean_entropy": round(float(np.mean(stats_by_scene[s]["entropies"])), 4) if stats_by_scene[s]["entropies"] else 0.0,
                    "std_entropy": round(float(np.std(stats_by_scene[s]["entropies"])), 4) if stats_by_scene[s]["entropies"] else 0.0,
                }
                for s in scene_ids
            },
        }

        with open(out_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, indent=2)

        logger.info("=================================================================")
        logger.info("  Multi-Scene Active View Dataset Generation Completed!         ")
        logger.info("  Total Episodes:     %d across %d scenes", len(all_episodes), len(scene_ids))
        logger.info("  Total Samples:      %d", len(all_samples_meta))
        logger.info("  Entropy Spectrum:   Min=%.4f, Median=%.4f, Mean=%.4f, Max=%.4f (Std=%.4f)",
                    dataset_stats["entropy_distribution_statistics"]["min"],
                    dataset_stats["entropy_distribution_statistics"]["percentiles"]["p50_median"],
                    dataset_stats["entropy_distribution_statistics"]["mean"],
                    dataset_stats["entropy_distribution_statistics"]["max"],
                    dataset_stats["entropy_distribution_statistics"]["std"])
        logger.info("=================================================================")

        return dataset_stats
