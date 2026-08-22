#!/usr/bin/env python3
"""
ST-GCN 动作感知数据集全自动构建引擎 —— build_action_dataset.py
============================================================

职责：
    1. 扫描 AMASS 动作库 (覆盖 6 大动作类别: standing, walking, sitting, bending, reaching, fall_related)；
    2. 生成 Clean Perception 训练集 (train/clean_perception)；
    3. 生成 Clean Perception 测试集 (test/clean_perception)；
    4. 生成 Habitat Perception 多视角测试集 (test/habitat_perception)；
    5. 严格保证：所有数据均通过同一个 Pose3DEstimator (MediaPipe3D) 提取；
    6. 将规范数据保存至 `/home/zxf/WorkSpace/code/data/ActiveView/datasets/action/`。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from ea_avs_mvp_v7.core.paths import get_data_root
from ea_avs_mvp_v10.core.paths import get_repo_root
from ea_avs_mvp_v10.perception.pose3d_estimator import create_pose3d_estimator
from ea_avs_mvp_v10.perception.skeleton_definition import get_skeleton_definition
from tools.dataset_generation.amass_renderer import AMASSCleanRenderer
from tools.dataset_generation.habitat_renderer import HabitatPerceptionRenderer
from tools.dataset_generation.pose_extraction import SequencePoseExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_action_dataset")


ACTION_NAME_TO_ID = {
    "standing": 0,
    "walking": 1,
    "sitting": 2,
    "bending": 3,
    "reaching": 4,
    "fall_related": 5,
}


def build_action_dataset(
    output_dir: Optional[Path] = None,
    num_frames: int = 30,
    samples_per_action_train: int = 15,
    samples_per_action_test: int = 5,
) -> Dict[str, Any]:
    skel_def = get_skeleton_definition()
    data_root = output_dir or (get_data_root() / "datasets" / "action")
    data_root.mkdir(parents=True, exist_ok=True)

    train_clean_dir = data_root / "train" / "clean_perception"
    test_clean_dir = data_root / "test" / "clean_perception"
    test_habitat_dir = data_root / "test" / "habitat_perception"
    meta_dir = data_root / "metadata"

    for d in [train_clean_dir, test_clean_dir, test_habitat_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 保存动作类别映射
    with open(meta_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(ACTION_NAME_TO_ID, f, indent=2)

    clean_renderer = AMASSCleanRenderer(fps=30.0)
    habitat_renderer = HabitatPerceptionRenderer(fps=30.0)
    extractor = SequencePoseExtractor(skel_def=skel_def)

    # 动作资产库
    motion_mgr = clean_renderer.motion_mgr
    actions_dict = {}
    for action_name in ACTION_NAME_TO_ID.keys():
        m_list = motion_mgr.get_motions_by_class(action_name)
        if not m_list:
            m_list = [f"{action_name}_default"]
        actions_dict[action_name] = m_list

    logger.info(">>> 1. Building Clean Perception Training Dataset...")
    train_data = []
    train_labels = []

    for action_name, action_id in ACTION_NAME_TO_ID.items():
        motion_ids = actions_dict.get(action_name, [f"{action_name}_default"])
        for s_idx in range(samples_per_action_train):
            m_id = motion_ids[s_idx % len(motion_ids)]
            # 多角度微调增加训练数据多样性
            angle = (s_idx * 24.0) % 360.0
            dist = 1.8 + 0.1 * (s_idx % 5)
            rgb_seq = clean_renderer.render_motion_sequence(
                motion_id=m_id,
                num_frames=num_frames,
                viewpoint_distance=dist,
                camera_angle_deg=angle,
            )
            skel_seq, _ = extractor.extract_and_normalize(rgb_seq) # (T, V, 3)
            # 格式化为 (C, T, V, 1) -> C=3, T=num_frames, V=33, M=1
            tensor_seq = np.transpose(skel_seq, (2, 0, 1))[..., np.newaxis]
            train_data.append(tensor_seq)
            train_labels.append(action_id)

    train_data_np = np.array(train_data, dtype=np.float32) # (N_train, 3, T, V, 1)
    train_labels_np = np.array(train_labels, dtype=np.int64)

    np.save(train_clean_dir / "data.npy", train_data_np)
    np.save(train_clean_dir / "labels.npy", train_labels_np)
    logger.info("Saved Train Clean Dataset: Data %s, Labels %s", train_data_np.shape, train_labels_np.shape)

    logger.info(">>> 2. Building Clean Perception Test Dataset (Oracle Baseline)...")
    test_clean_data = []
    test_clean_labels = []

    for action_name, action_id in ACTION_NAME_TO_ID.items():
        motion_ids = actions_dict.get(action_name, [f"{action_name}_default"])
        for s_idx in range(samples_per_action_test):
            m_id = motion_ids[(s_idx + samples_per_action_train) % len(motion_ids)]
            angle = (s_idx * 72.0) % 360.0
            rgb_seq = clean_renderer.render_motion_sequence(
                motion_id=m_id,
                num_frames=num_frames,
                viewpoint_distance=2.0,
                camera_angle_deg=angle,
            )
            skel_seq, _ = extractor.extract_and_normalize(rgb_seq)
            tensor_seq = np.transpose(skel_seq, (2, 0, 1))[..., np.newaxis]
            test_clean_data.append(tensor_seq)
            test_clean_labels.append(action_id)

    test_clean_data_np = np.array(test_clean_data, dtype=np.float32)
    test_clean_labels_np = np.array(test_clean_labels, dtype=np.int64)

    np.save(test_clean_dir / "data.npy", test_clean_data_np)
    np.save(test_clean_dir / "labels.npy", test_clean_labels_np)
    logger.info("Saved Test Clean Dataset: Data %s, Labels %s", test_clean_data_np.shape, test_clean_labels_np.shape)

    logger.info(">>> 3. Building Habitat Perception Multi-View Test Dataset...")
    test_hab_data = []
    test_hab_labels = []
    viewpoints_meta = []

    # 4 个观察半径 x 8 个水平方位角
    test_viewpoints = []
    for r in [1.5, 2.0, 2.5]:
        for ang in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            rad = np.radians(ang)
            test_viewpoints.append({
                "view_id": f"vp_r{r:.1f}_a{int(ang):03d}",
                "radius": r,
                "angle_deg": ang,
                "position": [float(r * np.sin(rad)), 0.0, float(r * np.cos(rad))],
                "yaw_deg": float(ang + 180.0),
            })

    for action_name, action_id in ACTION_NAME_TO_ID.items():
        motion_ids = actions_dict.get(action_name, [f"{action_name}_default"])
        m_id = motion_ids[0]
        # 对该动作渲染所有视点
        vp_render_dict = habitat_renderer.render_multiview_sequences(
            motion_id=m_id,
            viewpoints=test_viewpoints[:samples_per_action_test * 4], # 每个动作采样多个视点
            num_frames=num_frames,
        )

        for v_id, vp_info in vp_render_dict.items():
            rgb_seq = vp_info["rgb_frames"]
            skel_seq, conf_seq = extractor.extract_and_normalize(rgb_seq)
            tensor_seq = np.transpose(skel_seq, (2, 0, 1))[..., np.newaxis]
            test_hab_data.append(tensor_seq)
            test_hab_labels.append(action_id)
            viewpoints_meta.append({
                "action": action_name,
                "action_id": action_id,
                "view_id": v_id,
                "viewpoint": vp_info["viewpoint"],
                "mean_confidence": float(np.mean(conf_seq)),
            })

    test_hab_data_np = np.array(test_hab_data, dtype=np.float32)
    test_hab_labels_np = np.array(test_hab_labels, dtype=np.int64)

    np.save(test_habitat_dir / "data.npy", test_hab_data_np)
    np.save(test_habitat_dir / "labels.npy", test_hab_labels_np)
    with open(test_habitat_dir / "viewpoints.json", "w", encoding="utf-8") as f:
        json.dump(viewpoints_meta, f, indent=2)
    logger.info("Saved Test Habitat Dataset: Data %s, Labels %s", test_hab_data_np.shape, test_hab_labels_np.shape)

    manifest_info = {
        "num_classes": len(ACTION_NAME_TO_ID),
        "actions": list(ACTION_NAME_TO_ID.keys()),
        "time_steps": num_frames,
        "joint_num": skel_def.joint_num,
        "pose_estimator": "mediapipe_33",
        "train_clean_samples": len(train_data_np),
        "test_clean_samples": len(test_clean_data_np),
        "test_habitat_samples": len(test_hab_data_np),
    }
    with open(meta_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    logger.info(">>> Action Perception Dataset successfully built at: %s", data_root)
    return manifest_info


def main():
    parser = argparse.ArgumentParser(description="Build ST-GCN Action Recognition Datasets")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_frames", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--train_samples", type=int, default=15, help="Train samples per action")
    parser.add_argument("--test_samples", type=int, default=5, help="Test samples per action")
    args = parser.parse_args()

    out_d = Path(args.output_dir) if args.output_dir else None
    build_action_dataset(
        output_dir=out_d,
        num_frames=args.num_frames,
        samples_per_action_train=args.train_samples,
        samples_per_action_test=args.test_samples,
    )


if __name__ == "__main__":
    main()
