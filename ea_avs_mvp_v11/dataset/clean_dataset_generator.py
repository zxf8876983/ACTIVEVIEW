#!/usr/bin/env python3
"""
Clean Perception ST-GCN Dataset Generator —— clean_dataset_generator.py (v11.5)
=============================================================================

职责：
    1. 读取 /home/zxf/WorkSpace/code/data/ActiveView/datasets/amass_split/ 中的 train.json 与 val.json；
    2. 在 100% 纯色背景 Studio 空旷环境 (scene_id='NONE') 下渲染 SMPL 仿真人体姿态；
    3. 调用真实视觉感知链路：RGB Camera -> Keypoint R-CNN -> VideoPose3D -> Skeleton Normalizer；
    4. 生成标准 5D 张量格式：(N, C=3, T=30, V=17, M=1) 及一维标签数组 (N,)；
    5. 严格隔离：只使用 Train 和 Validation 分割动作，绝不接触 Test 动作；
    6. 保存输出至 /home/zxf/WorkSpace/code/data/ActiveView/datasets/clean_perception_v11_5/。
"""

import json
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import habitat_sim
import numpy as np
import quaternion
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

# 保证包路径正确
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ea_avs_mvp_v11.core.paths import get_data_root, get_repo_root
from ea_avs_mvp_v11.perception.pose3d_estimator import VideoPose3DEstimator
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_dataset_generator")

URDF_PATH = "/home/zxf/WorkSpace/code/code/robot/habitat-lab/data/versioned_data/habitat_humanoids/neutral_0/neutral_0.urdf"

ACTION_CATEGORIES = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]
LABEL_TO_ID = {cat: idx for idx, cat in enumerate(ACTION_CATEGORIES)}


def load_split_manifest(split_name: str) -> List[Dict[str, Any]]:
    split_dir = get_data_root() / "datasets" / "amass_split"
    split_file = split_dir / f"{split_name}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_clean_perception_dataset(
    split_name: str = "train",
    samples_per_motion: int = 15,
    target_frames: int = 30,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    为指定 split 生成真实 Clean Perception 估计骨架数据集。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing VideoPose3DEstimator on %s for split: %s...", device, split_name)

    skel_def = get_skeleton_definition(backend="h36m_17")
    estimator = VideoPose3DEstimator(device=str(device), skel_def=skel_def)
    normalizer = SkeletonNormalizer(skel_def=skel_def)

    # 初始化 Habitat 纯色单色背景 (scene_id='NONE')
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"
    backend_cfg.enable_physics = True

    H, W = 512, 512
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [H, W]
    rgb_spec.position = [0.0, 0.0, 0.0]
    rgb_spec.hfov = 50.0

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    aom = sim.get_articulated_object_manager()
    art_obj = aom.add_articulated_object_from_urdf(URDF_PATH)

    manifest = load_split_manifest(split_name)
    logger.info("Split '%s' contains %d motion instances", split_name, len(manifest))

    all_samples = []
    all_labels = []
    metadata_list = []

    for motion_idx, rec in enumerate(manifest):
        abs_p = rec["absolute_path"]
        action_label = rec["action_label"]
        if action_label not in LABEL_TO_ID:
            logger.warning("Skipping unknown action category: %s", action_label)
            continue
        label_id = LABEL_TO_ID[action_label]

        with open(abs_p, "rb") as f:
            motion_data = pickle.load(f)

        joints_array = motion_data["pose_motion"]["joints_array"]  # (total_f, 216)
        total_f = joints_array.shape[0]

        # 采样生成多个不同的时间窗口与视点增强样本
        for sample_i in range(samples_per_motion):
            # 1. 选取 30 帧时序切片
            if total_f >= target_frames:
                start_f = (sample_i * (total_f - target_frames)) // max(1, samples_per_motion - 1)
                frame_indices = np.linspace(start_f, start_f + target_frames - 1, target_frames, dtype=int)
            else:
                frame_indices = np.linspace(0, total_f - 1, target_frames, dtype=int)

            # 2. 选取轻微视角扰动 (Frontal ± 25度, 距离 2.4m ~ 2.8m)
            angle_deg = np.random.uniform(-25.0, 25.0) if split_name == "train" else 0.0
            dist_m = np.random.uniform(2.4, 2.8) if split_name == "train" else 2.6
            rad = np.radians(angle_deg)

            rgb_frames = []
            for f_idx in frame_indices:
                q_joints = joints_array[f_idx]
                art_obj.joint_positions = q_joints

                # 动态受力面贴地
                art_obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                min_link_y = min(art_obj.get_link_scene_node(i).absolute_translation[1] for i in range(art_obj.num_links))
                max_link_y = max(art_obj.get_link_scene_node(i).absolute_translation[1] for i in range(art_obj.num_links))
                grounded_y = -min_link_y
                art_obj.translation = np.array([0.0, grounded_y, 0.0], dtype=np.float32)

                human_center_y = grounded_y + (max_link_y - min_link_y) * 0.50

                cam_pos = np.array([dist_m * np.sin(rad), human_center_y - 0.75, dist_m * np.cos(rad)], dtype=np.float32)
                target_pos = np.array([0.0, human_center_y, 0.0], dtype=np.float32)
                dir_vec = target_pos - (cam_pos + np.array([0.0, 0.75, 0.0]))
                dir_norm = dir_vec / np.linalg.norm(dir_vec)
                yaw = np.arctan2(-dir_norm[0], -dir_norm[2])
                pitch = np.arcsin(dir_norm[1])
                cam_rot = quaternion.from_rotation_vector([0, yaw, 0]) * quaternion.from_rotation_vector([pitch, 0, 0])

                agent_state = habitat_sim.AgentState()
                agent_state.position = cam_pos
                agent_state.rotation = cam_rot
                sim.get_agent(0).set_state(agent_state)

                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"][:, :, :3]
                rgb_frames.append(rgb)

            # 3. 通过 VideoPose3D 估计 3D 骨架序列 (T=30, V=17, 3)
            skels_3d, confs = estimator.estimate_sequence(rgb_frames)

            # 4. 骨架归一化 (Root Centering + Scale Normalization)
            norm_skel = normalizer.normalize_sequence(skels_3d, align_canonical=True)  # (30, 17, 3)

            # 5. 转为 ST-GCN 格式 (C=3, T=30, V=17, M=1)
            sample_tensor = np.transpose(norm_skel, (2, 0, 1))[:, :, :, np.newaxis]  # (3, 30, 17, 1)

            all_samples.append(sample_tensor)
            all_labels.append(label_id)
            metadata_list.append({
                "motion_id": rec["motion_id"],
                "action_label": action_label,
                "label_id": label_id,
                "sample_idx": sample_i,
                "split": split_name,
                "mean_confidence": float(np.mean(confs)),
            })

        logger.info("  [%02d/%02d] Processed motion %s (category: %s, %d samples generated)",
                    motion_idx + 1, len(manifest), rec["motion_id"], action_label, samples_per_motion)

    sim.close()

    data_arr = np.array(all_samples, dtype=np.float32)  # (N, C, T, V, M)
    labels_arr = np.array(all_labels, dtype=np.int64)   # (N,)

    logger.info("Generated %s dataset: data shape=%s, labels shape=%s", split_name, data_arr.shape, labels_arr.shape)
    return data_arr, labels_arr, metadata_list


def main():
    data_root = get_data_root()
    out_base = data_root / "datasets" / "clean_perception_v11_5"

    for split in ["train", "val"]:
        logger.info("==========================================================")
        logger.info("  Building Clean Perception Dataset: %s", split)
        logger.info("==========================================================")
        n_samples = 20 if split == "train" else 10
        data, labels, meta = build_clean_perception_dataset(split_name=split, samples_per_motion=n_samples)

        split_out = out_base / split
        split_out.mkdir(parents=True, exist_ok=True)

        np.save(split_out / "data.npy", data)
        np.save(split_out / "labels.npy", labels)
        with open(split_out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Saved %s dataset to %s (data: %s, labels: %s)", split, split_out, data.shape, labels.shape)


if __name__ == "__main__":
    main()
