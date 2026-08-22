"""
Phase 2 感知数据集处理流水线 —— perception_dataset.py
===================================================

职责：
    1. 批量读取 Phase 1 生成的 RGB-D 样本；
    2. 执行端到端真实机器人感知流水线：
       RGB-D -> 2D Pose (COCO-17) -> Depth 逆投影 -> 3D 骨架生成 -> 3D 归一化 -> 坐标健康度校验；
    3. 规范化保存估计结果至 `datasets/v10/perception/`：
       - pose2d/
       - pose3d_camera/
       - pose3d_world/
       - normalized_pose3d/
       - confidence/
       - metadata/
    4. 生成并维护 `perception_manifest.json` 元数据索引；
    5. 严格隔离：感知流水线全过程绝不接触或读取 GT Skeleton 真值。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ea_avs_mvp_v10.core.paths import get_data_root, get_v10_dataset_root
from ea_avs_mvp_v10.core.types import CameraIntrinsics, CameraPose, V10Sample
from ea_avs_mvp_v10.perception.coordinate_validator import CoordinateValidator, ValidationResult
from ea_avs_mvp_v10.perception.depth_projection import DepthProjector
from ea_avs_mvp_v10.perception.pose_estimator import BasePoseEstimator, TorchvisionPoseEstimator
from ea_avs_mvp_v10.perception.skeleton_converter import EstimatedSkeleton3D, SkeletonConverter
from ea_avs_mvp_v10.perception.skeleton_normalizer import SkeletonNormalizer

logger = logging.getLogger(__name__)


class V10PerceptionPipeline:
    """v10.0 单样本与批量 RGB-D 3D 骨架感知估计器。"""

    def __init__(
        self,
        pose_estimator: Optional[BasePoseEstimator] = None,
        depth_projector: Optional[DepthProjector] = None,
        skeleton_converter: Optional[SkeletonConverter] = None,
        skeleton_normalizer: Optional[SkeletonNormalizer] = None,
        coordinate_validator: Optional[CoordinateValidator] = None,
        dataset_root: Optional[Union[str, Path]] = None,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root else get_v10_dataset_root()
        self.pose_estimator = pose_estimator or TorchvisionPoseEstimator()
        self.depth_projector = depth_projector or DepthProjector()
        self.skeleton_converter = skeleton_converter or SkeletonConverter()
        self.skeleton_normalizer = skeleton_normalizer or SkeletonNormalizer()
        self.coordinate_validator = coordinate_validator or CoordinateValidator()

        # 初始化输出子目录
        self.perception_root = self.dataset_root / "perception"
        self.dirs = {
            "pose2d": self.perception_root / "pose2d",
            "pose3d_camera": self.perception_root / "pose3d_camera",
            "pose3d_world": self.perception_root / "pose3d_world",
            "normalized_pose3d": self.perception_root / "normalized_pose3d",
            "confidence": self.perception_root / "confidence",
            "visualization": self.perception_root / "visualization",
            "metadata": self.perception_root / "metadata",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def process_sample(
        self,
        sample: V10Sample,
        rgb_image: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        save_outputs: bool = True,
    ) -> Tuple[EstimatedSkeleton3D, Dict[str, Any]]:
        """
        处理单张 RGB-D 样本，估计 3D 姿态并可选持久化。

        Args:
            sample: Phase 1 V10Sample 元数据
            rgb_image: 显式传入的 RGB (可选，为空则从 rgb_path 读取)
            depth_map: 显式传入的 Depth (可选，为空则从 depth_path 读取)
            save_outputs: 是否保存到 perception/ 目录

        Returns:
            (EstimatedSkeleton3D, sample_record)
        """
        # 1. 加载图像数据
        if rgb_image is None:
            rgb_path = self.dataset_root / sample.rgb_path
            rgb_arr = np.array(Image.open(rgb_path))
        else:
            rgb_arr = rgb_image

        if depth_map is None:
            depth_path = self.dataset_root / sample.depth_path
            depth_arr = np.load(depth_path)
        else:
            depth_arr = depth_map

        intrinsics = sample.camera_pose.intrinsics
        cam_pose = sample.camera_pose

        # 2. Step 1: RGB -> 2D Pose (COCO-17)
        pose2d_res = self.pose_estimator.estimate_pose2d(rgb_arr)

        # 3. Step 2: Depth Back-Projection -> 3D Camera/World Pose
        depth_res = self.depth_projector.project_2d_to_3d(
            keypoints_2d=pose2d_res.keypoints,
            depth_map=depth_arr,
            intrinsics=intrinsics,
            camera_pose=cam_pose,
        )

        # 4. Step 3: Skeleton Fusion (COCO-17)
        skeleton3d = self.skeleton_converter.convert_and_fuse(
            pose2d=pose2d_res,
            depth_res=depth_res,
        )

        # 5. Step 4: Skeleton Normalization (Root centered & Scale normalized)
        skeleton3d = self.skeleton_normalizer.normalize(skeleton3d)

        # 6. Step 5: Coordinate Sanity Check
        val_res = self.coordinate_validator.validate(skeleton3d)

        # 7. 构造元数据记录
        record = {
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "motion_id": sample.motion_id,
            "action_label": sample.action_label,
            "frame_idx": sample.frame_idx,
            "view_id": sample.view_id,
            "joint_format": skeleton3d.joint_format,
            "person_detected": bool(pose2d_res.person_score >= 0.5),
            "person_score": float(pose2d_res.person_score),
            "mean_confidence": float(np.mean(skeleton3d.perception_confidence)),
            "num_valid_joints": int(np.sum(~skeleton3d.uncertainty_mask)),
            "validation_status": val_res.status,
            "validation_reasons": val_res.reasons,
            "part_confidence": skeleton3d.part_confidence,
            "pose2d_path": f"perception/pose2d/{sample.sample_id}.json",
            "pose3d_camera_path": f"perception/pose3d_camera/{sample.sample_id}.json",
            "pose3d_world_path": f"perception/pose3d_world/{sample.sample_id}.json",
            "normalized_pose3d_path": f"perception/normalized_pose3d/{sample.sample_id}.json",
            "confidence_path": f"perception/confidence/{sample.sample_id}.json",
        }

        # 8. 分级独立持久化各坐标系产物
        if save_outputs:
            p2d_file = self.dirs["pose2d"] / f"{sample.sample_id}.json"
            p3d_cam_file = self.dirs["pose3d_camera"] / f"{sample.sample_id}.json"
            p3d_world_file = self.dirs["pose3d_world"] / f"{sample.sample_id}.json"
            norm_p3d_file = self.dirs["normalized_pose3d"] / f"{sample.sample_id}.json"
            conf_file = self.dirs["confidence"] / f"{sample.sample_id}.json"

            with open(p2d_file, "w", encoding="utf-8") as f:
                json.dump(pose2d_res.to_dict(), f, indent=2)

            with open(p3d_cam_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_id": sample.sample_id,
                    "joint_format": skeleton3d.joint_format,
                    "coordinate_system": "camera_frame_xyz_meters",
                    "joints_3d_camera": skeleton3d.joints_3d_camera.tolist(),
                    "joint_names": skeleton3d.joint_names,
                }, f, indent=2)

            with open(p3d_world_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_id": sample.sample_id,
                    "joint_format": skeleton3d.joint_format,
                    "coordinate_system": "world_frame_xyz_meters",
                    "joints_3d_world": skeleton3d.joints_3d_world.tolist(),
                    "joint_names": skeleton3d.joint_names,
                }, f, indent=2)

            with open(norm_p3d_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_id": sample.sample_id,
                    "joint_format": skeleton3d.joint_format,
                    "coordinate_system": "root_centered_scale_normalized",
                    "joints_3d_normalized": skeleton3d.joints_3d_normalized.tolist() if skeleton3d.joints_3d_normalized is not None else None,
                    "joint_names": skeleton3d.joint_names,
                }, f, indent=2)

            with open(conf_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_id": sample.sample_id,
                    "joint_format": skeleton3d.joint_format,
                    "perception_confidence": skeleton3d.perception_confidence.tolist(),
                    "uncertainty_mask": skeleton3d.uncertainty_mask.tolist(),
                    "part_confidence": skeleton3d.part_confidence,
                    "validation": val_res.to_dict(),
                }, f, indent=2)

        return skeleton3d, record

    def process_dataset(
        self,
        samples_manifest_path: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """批量处理 Phase 1 数据集并生成全量估计骨架清单。"""
        manifest_p = Path(samples_manifest_path) if samples_manifest_path else (self.dataset_root / "metadata" / "samples.json")
        if not manifest_p.exists():
            raise FileNotFoundError(f"Phase 1 dataset manifest not found at: {manifest_p}")

        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        raw_samples = manifest_data.get("samples", [])
        if max_samples:
            raw_samples = raw_samples[:max_samples]

        self.coordinate_validator.reset_stats()
        logger.info("Processing %d samples with Perception Pipeline (COCO-17 native)...", len(raw_samples))
        records = []

        for idx, s_dict in enumerate(raw_samples):
            sample = V10Sample.from_dict(s_dict)
            _, rec = self.process_sample(sample, save_outputs=True)
            records.append(rec)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(raw_samples):
                logger.info("Perception processed [%d/%d] samples.", idx + 1, len(raw_samples))

        val_summary = self.coordinate_validator.get_summary()
        logger.info("Validation Summary: Total=%d, Valid=%d, Warning=%d, Invalid=%d (Pass rate: %.1f%%)",
                    val_summary["total_checked"], val_summary["valid_count"],
                    val_summary["warning_count"], val_summary["invalid_count"],
                    val_summary["pass_rate"] * 100.0)

        # 保存 perception manifest
        out_manifest_p = self.dirs["metadata"] / "perception_manifest.json"
        with open(out_manifest_p, "w", encoding="utf-8") as f:
            json.dump({
                "version": "10.0.0",
                "phase": "phase_2_perception",
                "joint_format": "COCO17",
                "total_processed": len(records),
                "validation_summary": val_summary,
                "records": records,
            }, f, indent=2)

        logger.info("Saved perception dataset manifest to: %s", out_manifest_p)
        return records
