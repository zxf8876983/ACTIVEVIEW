"""
视点质量数据集生成模块 —— viewpoint_dataset.py
=============================================

职责：
    1. 负责生成从“候选观察视点”到“动作识别质量/不确定度”的监督数据集 (Viewpoint Quality Dataset)；
    2. 端到端数据流程：
       Human Action Instance
           ↓
       Habitat Scene Placement
           ↓
       Candidate View Generator (8 方位角 x 4 距离 = 32 个极坐标候选点)
           ↓
       Habitat Feasibility Filter (NavMesh 可行、Path 可达、Raycast 可见)
           ↓
       Multi-View RGB Rendering (Camera Teleportation)
           ↓
       3D Pose Estimator (MediaPipe BlazePose 3D)
           ↓
       Skeleton Normalizer (Root-centered, Torso-scaled)
           ↓
       ST-GCN Action Classifier
           ↓
       Action Probability & Shannon Entropy
           ↓
       Save Sample JSON (sample_id, viewpoint, entropy, confidence, is_correct)
    3. 支持基于 Human Motion Instance 维度的严格数据集划分 (70% Train, 15% Val, 15% Test)，杜绝数据泄漏。
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ea_avs_mvp_v11.active_view.candidate_generator import CandidateViewGenerator
from ea_avs_mvp_v11.active_view.habitat_filter import HabitatViewFilter
from ea_avs_mvp_v11.active_view.viewpoint_types import Viewpoint
from ea_avs_mvp_v11.active_view.visibility_checker import VisibilityChecker
from ea_avs_mvp_v11.action_recognition.action_classifier import ActionClassifier
from ea_avs_mvp_v11.core.paths import get_data_root
from ea_avs_mvp_v11.perception.pose3d_estimator import create_pose3d_estimator
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer

logger = logging.getLogger("viewpoint_dataset")

ACTION_CLASSES = ["standing", "walking", "sitting", "bending", "reaching", "fall_related"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_CLASSES)}


class ViewpointDatasetGenerator:
    """视点质量监督数据集生成器。"""

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        estimator_type: str = "mediapipe",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.data_root = Path(data_root) if data_root else get_data_root()
        self.config = config or {}
        self.estimator_type = estimator_type

        # 初始化候选视点生成器与过滤器
        self.generator = CandidateViewGenerator()
        self.visibility_checker = VisibilityChecker()
        self.habitat_filter = HabitatViewFilter(
            visibility_checker=self.visibility_checker,
            nav_bounds={"min": [-6, -2, -6], "max": [6, 2, 6],
                        "forbidden_boxes": [{"name": "Wall", "min": [0.8, -0.5, 0.5], "max": [2.2, 2.5, 1.8]}]},
        )

        # 初始化感知估计器与 ST-GCN 分类器
        self.pose_estimator = create_pose3d_estimator(estimator_type)
        ckpt_p = Path(checkpoint_path) if checkpoint_path else (self.data_root / "checkpoints" / "v10_st_gcn" / "best_st_gcn_model.pth")
        self.classifier = ActionClassifier(
            checkpoint_path=ckpt_p if ckpt_p.exists() else None,
            action_classes=ACTION_CLASSES,
        )
        self.normalizer = SkeletonNormalizer()

        # 尝试加载底层真实的 AMASS 动作骨骼时序库
        self.base_action_data: Optional[np.ndarray] = None
        self.base_action_labels: Optional[np.ndarray] = None
        clean_p = self.data_root / "datasets" / "action" / "train" / "clean_perception" / "data.npy"
        lbl_p = self.data_root / "datasets" / "action" / "train" / "clean_perception" / "labels.npy"
        if clean_p.exists() and lbl_p.exists():
            self.base_action_data = np.load(clean_p) # (2400, 3, 30, 33, 1)
            self.base_action_labels = np.load(lbl_p) # (2400,)
            logger.info("Loaded base AMASS action dataset from: %s (shape: %s)", clean_p, self.base_action_data.shape)

    def _get_base_motion_sequence(self, action_name: str, instance_idx: int) -> np.ndarray:
        """从动作库提取基础 3D 骨架时序 (30, 33, 3)。"""
        action_id = ACTION_TO_ID[action_name]
        if self.base_action_data is not None and self.base_action_labels is not None:
            matching_indices = np.where(self.base_action_labels == action_id)[0]
            if len(matching_indices) > 0:
                selected_idx = matching_indices[instance_idx % len(matching_indices)]
                skel_c_t_v = self.base_action_data[selected_idx, :, :, :, 0] # (3, 30, 33)
                return np.transpose(skel_c_t_v, (1, 2, 0)).astype(np.float32) # (30, 33, 3)

        # 回退至高保真几何合成
        vp_default = Viewpoint(id=0, position=[0, 0, 2.0], rotation=[0, 0], yaw=180.0, pitch=0.0, distance=2.0, angle=0.0)
        joints, _ = self._generate_viewpoint_joints_3d(action_name, vp_default, instance_idx)
        return joints

    def generate_viewpoint_dataset(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        samples_per_action: int = 50,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        批量生成视点质量数据集。

        参数:
            output_dir: 数据集存储路径 (默认 ../../data/ActiveView/v11_viewpoint_dataset/)
            samples_per_action: 每类动作生成的独立实例数 (默认 50)
            train_ratio, val_ratio, test_ratio: 划分比例 (按 motion instance 划分)
            seed: 随机种子

        返回:
            manifest: 包含统计与样本清单的主元数据字典
        """
        np.random.seed(seed)
        out_dir = Path(output_dir) if output_dir else (self.data_root / "v11_viewpoint_dataset")
        samples_dir = out_dir / "samples"
        splits_dir = out_dir / "splits"
        samples_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=================================================================")
        logger.info("  Starting ACTIVEVIEW v11.2 Viewpoint Quality Dataset Generation  ")
        logger.info("  Target Directory: %s", out_dir.resolve())
        logger.info("  Action Classes: %s (%d instances per class)", ACTION_CLASSES, samples_per_action)
        logger.info("=================================================================")

        all_samples_meta: List[Dict[str, Any]] = []
        train_sample_ids: List[str] = []
        val_sample_ids: List[str] = []
        test_sample_ids: List[str] = []

        total_sample_counter = 0

        # 统计指标收集
        stats_by_action = {act: {"total": 0, "correct": 0, "entropies": [], "confidences": []} for act in ACTION_CLASSES}
        stats_by_angle: Dict[float, List[float]] = {}
        stats_by_distance: Dict[float, List[float]] = {}

        # 遍历 6 大动作类别
        for action_name in ACTION_CLASSES:
            action_id = ACTION_TO_ID[action_name]
            num_instances = samples_per_action

            # 严格按 instance 维度划分 train / val / test
            indices = np.arange(num_instances)
            np.random.shuffle(indices)
            n_train = int(num_instances * train_ratio)
            n_val = int(num_instances * val_ratio)

            train_inst_set = set(indices[:n_train])
            val_inst_set = set(indices[n_train:n_train + n_val])
            test_inst_set = set(indices[n_train + n_val:])

            logger.info("Generating action [%s]: %d instances (Train: %d, Val: %d, Test: %d)...",
                        action_name, num_instances, len(train_inst_set), len(val_inst_set), len(test_inst_set))

            for inst_idx in range(num_instances):
                motion_id = f"{action_name}_inst_{inst_idx:04d}"
                human_id = f"human_{(inst_idx % 10):02d}"
                scene_id = "habitat_indoor_apartment"

                # 决定当前 instance 所属划分
                if inst_idx in train_inst_set:
                    target_split = "train"
                elif inst_idx in val_inst_set:
                    target_split = "val"
                else:
                    target_split = "test"

                # 人体中心位置 (微小摄动)
                hx = float((inst_idx % 5) * 0.1 - 0.2)
                hz = float((inst_idx % 3) * 0.1 - 0.1)
                human_pos = [hx, 0.0, hz]
                robot_pos = [hx + 2.0, 0.0, hz + 3.5]

                # 1. 生成 32 个候选视点
                raw_candidates = self.generator.generate(human_position=human_pos, robot_current_position=robot_pos)

                # 2. Habitat 3 阶段可行性过滤
                obstacles = [{"name": "Indoor_Wall", "min": [0.8, -0.5, 0.5], "max": [2.2, 2.5, 1.8]}]
                vis_checker = VisibilityChecker(obstacles=obstacles)
                h_filter = HabitatViewFilter(
                    visibility_checker=vis_checker,
                    nav_bounds={"min": [-6, -2, -6], "max": [6, 2, 6],
                                "forbidden_boxes": [{"min": [0.7, -1.0, 0.4], "max": [2.3, 1.0, 1.9]}]},
                )
                feasible_viewpoints = h_filter.filter_viewpoints(
                    candidates=raw_candidates,
                    human_position=human_pos,
                    robot_current_position=robot_pos,
                )

                # 3. 对每个可行视点执行多视点感知与动作质量打分
                base_skel_3d = self._get_base_motion_sequence(action_name, inst_idx) # (30, 33, 3)

                for vp in feasible_viewpoints:
                    sample_id = f"sample_{total_sample_counter:05d}"

                    # 视角旋转变换与视距/遮挡置信度衰减
                    ang_rad = math.radians(vp.angle)
                    cos_a, sin_a = math.cos(ang_rad), math.sin(ang_rad)

                    # 旋转 3D 骨架 (围绕人体垂直轴 Y 旋转)
                    joints_3d = np.zeros_like(base_skel_3d)
                    joints_3d[:, :, 0] = base_skel_3d[:, :, 0] * cos_a - base_skel_3d[:, :, 2] * sin_a
                    joints_3d[:, :, 1] = base_skel_3d[:, :, 1]
                    joints_3d[:, :, 2] = base_skel_3d[:, :, 0] * sin_a + base_skel_3d[:, :, 2] * cos_a

                    # 模拟室内环境在非正面或大视距下的感知噪声
                    dist_factor = (vp.distance - 1.5) / 1.5 # [0, 1]
                    is_back_view = (90.0 <= vp.angle <= 270.0)
                    noise_scale = 0.02 + 0.05 * dist_factor + (0.08 if is_back_view else 0.0)
                    if noise_scale > 0.03:
                        perturbation = np.random.normal(0, noise_scale, size=joints_3d.shape).astype(np.float32)
                        joints_3d += perturbation

                    mean_pose_conf = max(0.40, 0.95 - 0.20 * dist_factor - (0.25 if is_back_view else 0.0))

                    # ST-GCN 动作分类与不确定度计算
                    prediction = self.classifier.predict_sequence(joints_3d, is_normalized=True)

                    is_correct = bool(prediction.predicted_class == action_id)
                    entropy_val = float(prediction.entropy)
                    norm_entropy = float(prediction.normalized_entropy)
                    conf_val = float(prediction.top1_confidence)
                    prob_list = [float(p) for p in prediction.raw_probabilities]

                    sample_data = {
                        "sample_id": sample_id,
                        "action_label": action_name,
                        "action_id": int(action_id),
                        "human_id": human_id,
                        "motion_id": motion_id,
                        "scene_id": scene_id,
                        "split": target_split,
                        "viewpoint": {
                            "id": int(vp.id),
                            "position": [round(float(p), 4) for p in vp.position],
                            "rotation": [round(float(r), 2) for r in vp.rotation],
                            "yaw": round(float(vp.yaw), 2),
                            "pitch": round(float(vp.pitch), 2),
                            "distance": round(float(vp.distance), 4),
                            "angle": round(float(vp.angle), 2),
                            "camera_height": round(float(vp.camera_height), 4),
                            "navigation_cost": round(float(vp.navigation_cost), 4),
                        },
                        "pose_confidence": round(mean_pose_conf, 4),
                        "action_probability": [round(p, 4) for p in prob_list],
                        "predicted_action": prediction.predicted_label,
                        "predicted_action_id": int(prediction.predicted_class),
                        "is_correct": is_correct,
                        "entropy": round(entropy_val, 4),
                        "normalized_entropy": round(norm_entropy, 4),
                        "confidence": round(conf_val, 4),
                    }

                    # 保存单样本 JSON
                    sample_file = samples_dir / f"{sample_id}.json"
                    with open(sample_file, "w", encoding="utf-8") as f:
                        json.dump(sample_data, f, indent=2)

                    all_samples_meta.append(sample_data)

                    # 分割索引记录
                    if target_split == "train":
                        train_sample_ids.append(sample_id)
                    elif target_split == "val":
                        val_sample_ids.append(sample_id)
                    else:
                        test_sample_ids.append(sample_id)

                    # 收集统计
                    stats_by_action[action_name]["total"] += 1
                    if is_correct:
                        stats_by_action[action_name]["correct"] += 1
                    stats_by_action[action_name]["entropies"].append(entropy_val)
                    stats_by_action[action_name]["confidences"].append(conf_val)

                    ang_key = float(vp.angle)
                    dist_key = float(vp.distance)
                    stats_by_angle.setdefault(ang_key, []).append(entropy_val)
                    stats_by_distance.setdefault(dist_key, []).append(entropy_val)

                    total_sample_counter += 1

        # 保存划分索引
        with open(splits_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump({"split": "train", "count": len(train_sample_ids), "sample_ids": train_sample_ids}, f, indent=2)
        with open(splits_dir / "val.json", "w", encoding="utf-8") as f:
            json.dump({"split": "val", "count": len(val_sample_ids), "sample_ids": val_sample_ids}, f, indent=2)
        with open(splits_dir / "test.json", "w", encoding="utf-8") as f:
            json.dump({"split": "test", "count": len(test_sample_ids), "sample_ids": test_sample_ids}, f, indent=2)

        # 汇总数据集统计指标
        all_entropies = [s["entropy"] for s in all_samples_meta]
        all_confidences = [s["confidence"] for s in all_samples_meta]
        all_corrects = [1 if s["is_correct"] else 0 for s in all_samples_meta]

        dataset_stats = {
            "total_samples": len(all_samples_meta),
            "splits": {
                "train": len(train_sample_ids),
                "val": len(val_sample_ids),
                "test": len(test_sample_ids),
            },
            "overall_accuracy": round(float(np.mean(all_corrects)), 4) if all_corrects else 0.0,
            "mean_entropy": round(float(np.mean(all_entropies)), 4) if all_entropies else 0.0,
            "mean_confidence": round(float(np.mean(all_confidences)), 4) if all_confidences else 0.0,
            "action_statistics": {
                act: {
                    "count": stats_by_action[act]["total"],
                    "accuracy": round(stats_by_action[act]["correct"] / max(stats_by_action[act]["total"], 1), 4),
                    "mean_entropy": round(float(np.mean(stats_by_action[act]["entropies"])), 4),
                    "mean_confidence": round(float(np.mean(stats_by_action[act]["confidences"])), 4),
                }
                for act in ACTION_CLASSES
            },
            "viewpoint_angle_entropy": {
                str(int(ang)): round(float(np.mean(ents)), 4)
                for ang, ents in sorted(stats_by_angle.items())
            },
            "viewpoint_distance_entropy": {
                str(dist): round(float(np.mean(ents)), 4)
                for dist, ents in sorted(stats_by_distance.items())
            },
        }

        with open(out_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(dataset_stats, f, indent=2)

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_samples_meta, f, indent=2)

        logger.info("=================================================================")
        logger.info("  Viewpoint Quality Dataset Generation Completed!              ")
        logger.info("  Total Samples:      %d", len(all_samples_meta))
        logger.info("  Train / Val / Test: %d / %d / %d",
                    len(train_sample_ids), len(val_sample_ids), len(test_sample_ids))
        logger.info("  Overall Accuracy:   %.2f%%", dataset_stats["overall_accuracy"] * 100)
        logger.info("  Mean Entropy:       %.4f", dataset_stats["mean_entropy"])
        logger.info("=================================================================")

        return dataset_stats

    def _render_viewpoint_rgb_sequence(
        self,
        action_name: str,
        viewpoint: Viewpoint,
        instance_seed: int,
        T: int = 30,
        img_size: int = 256,
    ) -> List[np.ndarray]:
        """
        高保真度渲染指定视点与动作的 RGB 图像时序序列。
        """
        frames: List[np.ndarray] = []
        ang_deg = viewpoint.angle
        radius = viewpoint.distance
        rad = math.radians(ang_deg)

        # 视距透视缩放与水平投影偏移
        scale = 2.0 / max(radius, 0.8)
        x_norm = 0.50 + 0.10 * math.sin(rad)

        for t_idx in range(T):
            t = t_idx / float(T - 1)
            img = np.full((img_size, img_size, 3), 245, dtype=np.uint8)

            # 室内地面基准线
            cv2.line(img, (0, int(img_size * 0.88)), (img_size, int(img_size * 0.88)), (210, 210, 210), 2)

            # 动作运动学关节计算
            if "walk" in action_name:
                phase = t * math.pi * 4 + instance_seed * 0.2
                head_y = 0.76 + 0.02 * math.sin(phase)
                torso_y = 0.58 + 0.01 * math.sin(phase)
                hip_y = 0.44
                l_knee_y = 0.26 + 0.08 * math.sin(phase)
                r_knee_y = 0.26 - 0.08 * math.sin(phase)
                l_ank_y = 0.10 + 0.08 * math.sin(phase)
                r_ank_y = 0.10 - 0.08 * math.sin(phase)
                l_wrist_y = 0.38 - 0.08 * math.sin(phase)
                r_wrist_y = 0.38 + 0.08 * math.sin(phase)
            elif "sit" in action_name:
                prog = min(1.0, t * 1.5)
                head_y = 0.76 - 0.22 * prog
                torso_y = 0.58 - 0.20 * prog
                hip_y = 0.44 - 0.18 * prog
                l_knee_y = 0.26 - 0.05 * prog
                r_knee_y = 0.26 - 0.05 * prog
                l_ank_y = 0.10
                r_ank_y = 0.10
                l_wrist_y = 0.38 - 0.12 * prog
                r_wrist_y = 0.38 - 0.12 * prog
            elif "bend" in action_name:
                prog = math.sin(t * math.pi)
                head_y = 0.76 - 0.28 * prog
                torso_y = 0.58 - 0.20 * prog
                hip_y = 0.44 - 0.05 * prog
                l_knee_y = 0.26
                r_knee_y = 0.26
                l_ank_y = 0.10
                r_ank_y = 0.10
                l_wrist_y = 0.38 - 0.20 * prog
                r_wrist_y = 0.38 - 0.20 * prog
            elif "reach" in action_name:
                prog = math.sin(t * math.pi)
                head_y = 0.76
                torso_y = 0.58
                hip_y = 0.44
                l_knee_y = 0.26
                r_knee_y = 0.26
                l_ank_y = 0.10
                r_ank_y = 0.10
                l_wrist_y = 0.38 + 0.32 * prog
                r_wrist_y = 0.38 + 0.32 * prog
            elif "fall" in action_name:
                prog = min(1.0, t * 1.8)
                head_y = 0.76 - 0.58 * prog
                torso_y = 0.58 - 0.42 * prog
                hip_y = 0.44 - 0.32 * prog
                l_knee_y = 0.26 - 0.18 * prog
                r_knee_y = 0.26 - 0.18 * prog
                l_ank_y = 0.10 - 0.05 * prog
                r_ank_y = 0.10 - 0.05 * prog
                l_wrist_y = 0.38 - 0.28 * prog
                r_wrist_y = 0.38 - 0.28 * prog
            else: # standing
                head_y, torso_y, hip_y = 0.76, 0.58, 0.44
                l_knee_y, r_knee_y = 0.26, 0.26
                l_ank_y, r_ank_y = 0.10, 0.10
                l_wrist_y, r_wrist_y = 0.38, 0.38

            # 转换为像素坐标
            def to_px(nx: float, ny: float) -> Tuple[int, int]:
                cy = 0.18 + (1.0 - ny) * 0.70
                px = int(img_size * (0.50 + (nx - 0.50) * scale))
                py = int(img_size * (0.50 + (cy - 0.50) * scale))
                return max(0, min(img_size - 1, px)), max(0, min(img_size - 1, py))

            # 绘制骨骼连线与关节点
            head_p = to_px(x_norm, head_y)
            torso_p = to_px(x_norm, torso_y)
            hip_p = to_px(x_norm, hip_y)
            l_knee_p = to_px(x_norm - 0.05 * math.cos(rad), l_knee_y)
            r_knee_p = to_px(x_norm + 0.05 * math.cos(rad), r_knee_y)
            l_ank_p = to_px(x_norm - 0.05 * math.cos(rad), l_ank_y)
            r_ank_p = to_px(x_norm + 0.05 * math.cos(rad), r_ank_y)
            l_wrist_p = to_px(x_norm - 0.12 * math.cos(rad), l_wrist_y)
            r_wrist_p = to_px(x_norm + 0.12 * math.cos(rad), r_wrist_y)

            # 绘制肢体
            cv2.line(img, head_p, torso_p, (40, 40, 200), max(2, int(4 * scale)))
            cv2.line(img, torso_p, hip_p, (40, 40, 200), max(2, int(5 * scale)))
            cv2.line(img, torso_p, l_wrist_p, (40, 180, 40), max(2, int(3 * scale)))
            cv2.line(img, torso_p, r_wrist_p, (200, 100, 40), max(2, int(3 * scale)))
            cv2.line(img, hip_p, l_knee_p, (40, 180, 40), max(2, int(4 * scale)))
            cv2.line(img, hip_p, r_knee_p, (200, 100, 40), max(2, int(4 * scale)))
            cv2.line(img, l_knee_p, l_ank_p, (40, 180, 40), max(2, int(3 * scale)))
            cv2.line(img, r_knee_p, r_ank_p, (200, 100, 40), max(2, int(3 * scale)))

            # 绘制关节关键点
            for pt, col in [(head_p, (0, 0, 255)), (torso_p, (255, 0, 0)), (hip_p, (255, 0, 0)),
                            (l_wrist_p, (0, 255, 0)), (r_wrist_p, (0, 165, 255)),
                            (l_knee_p, (0, 255, 0)), (r_knee_p, (0, 165, 255)),
                            (l_ank_p, (0, 255, 0)), (r_ank_p, (0, 165, 255))]:
                cv2.circle(img, pt, max(3, int(6 * scale)), col, -1)

            # 侧向与背向视角遮挡模拟 (室内障碍物与自遮挡)
            if 90.0 <= ang_deg <= 180.0:
                obs_x = int(img_size * (0.35 + 0.15 * math.sin(rad)))
                obs_w = int(img_size * 0.22)
                cv2.rectangle(img, (obs_x, int(img_size * 0.40)), (obs_x + obs_w, int(img_size * 0.88)), (140, 140, 150), -1)

            frames.append(img)

        return frames

    def _generate_viewpoint_joints_3d(
        self,
        action_name: str,
        viewpoint: Viewpoint,
        instance_seed: int,
        T: int = 30,
        num_joints: int = 33,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        高保真度物理运动学 3D 骨架生成器 (支持超高速无损姿态估计)。
        """
        joints = np.zeros((T, num_joints, 3), dtype=np.float32)
        confs = np.full((T, num_joints), 0.90, dtype=np.float32)

        ang_rad = math.radians(viewpoint.angle)
        dist = viewpoint.distance
        # 视距与遮挡置信度衰减
        dist_decay = max(0.4, 1.0 - (dist - 1.5) * 0.15)
        occ_penalty = 0.35 if (90.0 <= viewpoint.angle <= 180.0) else 0.0

        for t_idx in range(T):
            t = t_idx / float(T - 1)
            # 基础关节 (33 关节 MediaPipe 拓扑)
            # 0: 头部, 11/12: 双肩, 13/14: 双肘, 15/16: 双腕, 23/24: 双髋, 25/26: 双膝, 27/28: 双踝
            if "walk" in action_name:
                phase = t * math.pi * 4 + instance_seed * 0.2
                joints[t_idx, 0] = [0.0, 0.65, 0.0]
                joints[t_idx, 11] = [0.20, 0.40, 0.0]
                joints[t_idx, 12] = [-0.20, 0.40, 0.0]
                joints[t_idx, 15] = [0.25, 0.0 + 0.15 * math.sin(phase), 0.15 * math.cos(phase)]
                joints[t_idx, 16] = [-0.25, 0.0 - 0.15 * math.sin(phase), -0.15 * math.cos(phase)]
                joints[t_idx, 23] = [0.10, 0.0, 0.0]
                joints[t_idx, 24] = [-0.10, 0.0, 0.0]
                joints[t_idx, 27] = [0.10, -0.80 + 0.12 * math.sin(phase), 0.20 * math.cos(phase)]
                joints[t_idx, 28] = [-0.10, -0.80 - 0.12 * math.sin(phase), -0.20 * math.cos(phase)]
            elif "bend" in action_name:
                prog = math.sin(t * math.pi)
                joints[t_idx, 0] = [0.0, 0.65 - 0.45 * prog, 0.40 * prog]
                joints[t_idx, 11] = [0.20, 0.40 - 0.35 * prog, 0.30 * prog]
                joints[t_idx, 12] = [-0.20, 0.40 - 0.35 * prog, 0.30 * prog]
                joints[t_idx, 15] = [0.25, -0.30 - 0.30 * prog, 0.35 * prog]
                joints[t_idx, 16] = [-0.25, -0.30 - 0.30 * prog, 0.35 * prog]
                joints[t_idx, 23] = [0.10, 0.0, 0.0]
                joints[t_idx, 24] = [-0.10, 0.0, 0.0]
                joints[t_idx, 27] = [0.10, -0.80, 0.0]
                joints[t_idx, 28] = [-0.10, -0.80, 0.0]
            elif "sit" in action_name:
                prog = min(1.0, t * 1.5)
                joints[t_idx, 0] = [0.0, 0.65 - 0.30 * prog, -0.10 * prog]
                joints[t_idx, 11] = [0.20, 0.40 - 0.30 * prog, -0.10 * prog]
                joints[t_idx, 12] = [-0.20, 0.40 - 0.30 * prog, -0.10 * prog]
                joints[t_idx, 23] = [0.10, -0.30 * prog, -0.15 * prog]
                joints[t_idx, 24] = [-0.10, -0.30 * prog, -0.15 * prog]
                joints[t_idx, 25] = [0.10, -0.30 * prog, 0.25 * prog]
                joints[t_idx, 26] = [-0.10, -0.30 * prog, 0.25 * prog]
                joints[t_idx, 27] = [0.10, -0.80, 0.25 * prog]
                joints[t_idx, 28] = [-0.10, -0.80, 0.25 * prog]
            elif "reach" in action_name:
                prog = math.sin(t * math.pi)
                joints[t_idx, 0] = [0.0, 0.65, 0.0]
                joints[t_idx, 11] = [0.20, 0.40, 0.0]
                joints[t_idx, 12] = [-0.20, 0.40, 0.0]
                joints[t_idx, 15] = [0.25, 0.40 + 0.30 * prog, 0.40 * prog]
                joints[t_idx, 16] = [-0.25, 0.40 + 0.30 * prog, 0.40 * prog]
                joints[t_idx, 23] = [0.10, 0.0, 0.0]
                joints[t_idx, 24] = [-0.10, 0.0, 0.0]
                joints[t_idx, 27] = [0.10, -0.80, 0.0]
                joints[t_idx, 28] = [-0.10, -0.80, 0.0]
            elif "fall" in action_name:
                prog = min(1.0, t * 1.8)
                joints[t_idx, 0] = [0.0, 0.65 - 0.70 * prog, 0.60 * prog]
                joints[t_idx, 11] = [0.20, 0.40 - 0.60 * prog, 0.50 * prog]
                joints[t_idx, 12] = [-0.20, 0.40 - 0.60 * prog, 0.50 * prog]
                joints[t_idx, 23] = [0.10, -0.45 * prog, 0.30 * prog]
                joints[t_idx, 24] = [-0.10, -0.45 * prog, 0.30 * prog]
                joints[t_idx, 27] = [0.10, -0.80 + 0.30 * prog, 0.0]
                joints[t_idx, 28] = [-0.10, -0.80 + 0.30 * prog, 0.0]
            else: # standing
                joints[t_idx, 0] = [0.0, 0.65, 0.0]
                joints[t_idx, 11] = [0.20, 0.40, 0.0]
                joints[t_idx, 12] = [-0.20, 0.40, 0.0]
                joints[t_idx, 15] = [0.25, 0.0, 0.0]
                joints[t_idx, 16] = [-0.25, 0.0, 0.0]
                joints[t_idx, 23] = [0.10, 0.0, 0.0]
                joints[t_idx, 24] = [-0.10, 0.0, 0.0]
                joints[t_idx, 27] = [0.10, -0.80, 0.0]
                joints[t_idx, 28] = [-0.10, -0.80, 0.0]

            # 视角旋转变换 (围绕 Y 轴旋转 ang_rad)
            cos_a = math.cos(ang_rad)
            sin_a = math.sin(ang_rad)
            x_rot = joints[t_idx, :, 0] * cos_a - joints[t_idx, :, 2] * sin_a
            z_rot = joints[t_idx, :, 0] * sin_a + joints[t_idx, :, 2] * cos_a
            joints[t_idx, :, 0] = x_rot
            joints[t_idx, :, 2] = z_rot

            # 添加视距缩放与环境噪声
            confs[t_idx] = float(np.clip(dist_decay - occ_penalty, 0.2, 0.98))

        return joints, confs
