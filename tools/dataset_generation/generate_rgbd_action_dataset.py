#!/usr/bin/env python3
"""
Generate Real RGB-D Action Dataset —— generate_rgbd_action_dataset.py (v11.4.2)
=============================================================================

职责：
    1. 遍历 16 类 AMASS 动作，通过 HabitatPerceptionPipeline 渲染真实 RGB-D 传感器画面；
    2. 使用 RGBDPoseEstimator 提取 Estimated 3D 骨架 (30, 33, 3)；
    3. 严格执行 Instance-Level 数据集划分 (Train IDs ∩ Test IDs = ∅)；
    4. 持久化 rgb/, depth/, skeleton/, label.json 并生成 split_statistics.json 与 occlusion_statistics.json。
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.active_view.action_registry import DEFAULT_ACTION_CATEGORIES
from ea_avs_mvp_v11.active_view.perception_pipeline import HabitatPerceptionPipeline
from ea_avs_mvp_v11.core.paths import get_data_root
from tools.dataset_generation.generate_16class_amass_dataset import synthesize_canonical_motion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_rgbd_action_dataset")


def generate_rgbd_action_dataset(
    num_train_per_class: int = 150,
    num_test_per_class: int = 40,
) -> Dict[str, Any]:
    data_root = get_data_root()
    action_root = data_root / "datasets" / "action"
    
    pipeline = HabitatPerceptionPipeline(data_root=data_root)

    scenes = ["apartment_1", "skokloster-castle", "van-gogh-room", "104862501_172226556", "106366386_174226770"]

    train_dir = action_root / "train"
    test_dir = action_root / "test"
    
    for d in [train_dir, test_dir]:
        (d / "rgb").mkdir(parents=True, exist_ok=True)
        (d / "depth").mkdir(parents=True, exist_ok=True)
        (d / "skeleton").mkdir(parents=True, exist_ok=True)

    train_records = []
    test_records = []
    train_motion_ids = set()
    test_motion_ids = set()

    occlusion_counts = {"easy": 0, "medium": 0, "hard": 0}

    logger.info("Generating RGB-D Estimated Action Dataset across %d categories...", len(DEFAULT_ACTION_CATEGORIES))

    for class_id, action_name in enumerate(DEFAULT_ACTION_CATEGORIES):
        # 1. 生成训练集样本 (施加随机视角 0~360 度与不同视距)
        for i in range(num_train_per_class):
            motion_id = f"train_{action_name}_{i:04d}"
            train_motion_ids.add(motion_id)

            seed = class_id * 10000 + i
            np.random.seed(seed)
            angle = float(np.random.uniform(0.0, 360.0))
            distance = float(np.random.uniform(1.5, 3.5))
            scene = scenes[i % len(scenes)]
            placement_diff = float(np.random.uniform(0.1, 0.7))

            base_m = synthesize_canonical_motion(action_name, seed=seed)

            human_state = {"position": [0, 0, 0], "placement_difficulty": placement_diff}
            robot_vp = {"angle": angle, "distance": distance, "position": [0, 1.2, distance]}

            obs = pipeline.observe(
                scene_id=scene,
                human_state=human_state,
                robot_viewpoint=robot_vp,
                base_motion_seq=base_m,
                action_id=class_id,
            )

            # 保存媒体与骨架
            rgb_file = f"{motion_id}_rgb.png"
            depth_file = f"{motion_id}_depth.npy"
            skel_file = f"{motion_id}_skel.npy"

            Image.fromarray(obs["rgb"]).save(train_dir / "rgb" / rgb_file)
            np.save(train_dir / "depth" / depth_file, obs["depth"])
            np.save(train_dir / "skeleton" / skel_file, obs["skeleton"])

            vis_ratio = obs["visible_ratio"]
            if vis_ratio >= 0.80:
                occlusion_counts["easy"] += 1
            elif vis_ratio >= 0.50:
                occlusion_counts["medium"] += 1
            else:
                occlusion_counts["hard"] += 1

            train_records.append({
                "motion_id": motion_id,
                "action_id": class_id,
                "action_label": action_name,
                "scene_id": scene,
                "rgb_path": f"rgb/{rgb_file}",
                "depth_path": f"depth/{depth_file}",
                "skeleton_path": f"skeleton/{skel_file}",
                "visible_ratio": round(vis_ratio, 4),
                "occlusion_ratio": round(obs["occlusion_ratio"], 4),
                "occlusion_level": obs["occlusion_level"],
                "missing_joints": obs["missing_joints"],
                "skeleton_source": "estimated",
                "camera_pose": obs["camera_pose"],
            })

        # 2. 生成测试集样本 (严格独立 seed 与实例 ID)
        for j in range(num_test_per_class):
            motion_id = f"test_{action_name}_{j:04d}"
            test_motion_ids.add(motion_id)

            seed = class_id * 10000 + 5000 + j
            np.random.seed(seed)
            angle = float(np.random.uniform(0.0, 360.0))
            distance = float(np.random.uniform(1.5, 3.5))
            scene = scenes[(j + 2) % len(scenes)]
            placement_diff = float(np.random.uniform(0.1, 0.7))

            base_m = synthesize_canonical_motion(action_name, seed=seed)

            human_state = {"position": [0, 0, 0], "placement_difficulty": placement_diff}
            robot_vp = {"angle": angle, "distance": distance, "position": [0, 1.2, distance]}

            obs = pipeline.observe(
                scene_id=scene,
                human_state=human_state,
                robot_viewpoint=robot_vp,
                base_motion_seq=base_m,
                action_id=class_id,
            )

            rgb_file = f"{motion_id}_rgb.png"
            depth_file = f"{motion_id}_depth.npy"
            skel_file = f"{motion_id}_skel.npy"

            Image.fromarray(obs["rgb"]).save(test_dir / "rgb" / rgb_file)
            np.save(test_dir / "depth" / depth_file, obs["depth"])
            np.save(test_dir / "skeleton" / skel_file, obs["skeleton"])

            test_records.append({
                "motion_id": motion_id,
                "action_id": class_id,
                "action_label": action_name,
                "scene_id": scene,
                "rgb_path": f"rgb/{rgb_file}",
                "depth_path": f"depth/{depth_file}",
                "skeleton_path": f"skeleton/{skel_file}",
                "visible_ratio": round(obs["visible_ratio"], 4),
                "occlusion_ratio": round(obs["occlusion_ratio"], 4),
                "occlusion_level": obs["occlusion_level"],
                "missing_joints": obs["missing_joints"],
                "skeleton_source": "estimated",
                "camera_pose": obs["camera_pose"],
            })


    # 保存 label.json
    with open(train_dir / "label.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2)

    with open(test_dir / "label.json", "w", encoding="utf-8") as f:
        json.dump(test_records, f, indent=2)

    # 保存划分统计
    overlap_count = len(train_motion_ids.intersection(test_motion_ids))
    split_stats = {
        "dataset_name": "ACTIVEVIEW v11.4.2 Real Habitat RGB-D Perception Dataset",
        "num_categories": len(DEFAULT_ACTION_CATEGORIES),
        "total_train_samples": len(train_records),
        "total_test_samples": len(test_records),
        "train_motion_ids_count": len(train_motion_ids),
        "test_motion_ids_count": len(test_motion_ids),
        "overlap_count": overlap_count,
        "source": "estimated_rgbd",
    }
    with open(action_root / "split_statistics.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    # 保存遮挡统计
    tot_samples = len(train_records)
    occ_stats = {
        "total_train_samples": tot_samples,
        "easy_count": occlusion_counts["easy"],
        "easy_ratio": round(occlusion_counts["easy"] / tot_samples, 4),
        "medium_count": occlusion_counts["medium"],
        "medium_ratio": round(occlusion_counts["medium"] / tot_samples, 4),
        "hard_count": occlusion_counts["hard"],
        "hard_ratio": round(occlusion_counts["hard"] / tot_samples, 4),
    }
    with open(action_root / "occlusion_statistics.json", "w", encoding="utf-8") as f:
        json.dump(occ_stats, f, indent=2)

    logger.info("Successfully generated RGB-D Action Dataset: Train=%d, Test=%d, Overlap=%d", len(train_records), len(test_records), overlap_count)
    return split_stats


if __name__ == "__main__":
    generate_rgbd_action_dataset()
