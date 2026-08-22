"""
Phase 2 感知数据集处理流水线 —— perception_dataset.py
===================================================

职责：
    1. 批量读取 Phase 1 生成的 RGB-D 样本；
    2. 执行端到端成熟 RGB-D 骨架感知流水线：
       RGB-D Observation -> RGB-D Skeleton Extractor -> 3D Skeleton -> Normalizer -> Validator；
    3. 规范化保存估计结果至 `datasets/v10/perception/`：
       - skeleton_raw/ (包含 camera 与 world 3D 骨架)
       - skeleton_normalized/ (ST-GCN 专用根节点去中心与尺度归一化骨架)
       - confidence/ (逐关节置信度、不确定性掩码与坐标健康度校验结果)
       - metadata/ (包含 perception_manifest.json)
       - visualization/
    4. 严格隔离：全过程绝不读取或使用仿真真值 (Ground Truth Skeleton)。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ea_avs_mvp_v11.core.paths import get_data_root, get_v10_dataset_root
from ea_avs_mvp_v11.core.types import CameraIntrinsics, CameraPose, V10Sample
from ea_avs_mvp_v11.perception.coordinate_validator import CoordinateValidator, ValidationResult
from ea_avs_mvp_v11.perception.rgbd_skeleton_extractor import (
    BaseRGBDSkeletonExtractor,
    RGBDSkeletonExtractor,
)
from ea_avs_mvp_v11.perception.skeleton_converter import EstimatedSkeleton3D
from ea_avs_mvp_v11.perception.skeleton_normalizer import SkeletonNormalizer

logger = logging.getLogger(__name__)


class V10PerceptionPipeline:
    """v10.0 RGB-D 3D 人体骨架感知流水线。"""

    def __init__(
        self,
        extractor: Optional[BaseRGBDSkeletonExtractor] = None,
        skeleton_normalizer: Optional[SkeletonNormalizer] = None,
        coordinate_validator: Optional[CoordinateValidator] = None,
        dataset_root: Optional[Union[str, Path]] = None,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root else get_v10_dataset_root()
        self.extractor = extractor or RGBDSkeletonExtractor(backend="mediapipe")
        self.skeleton_normalizer = skeleton_normalizer or SkeletonNormalizer()
        self.coordinate_validator = coordinate_validator or CoordinateValidator()

        # 初始化输出子目录 (按最新规范定义)
        self.perception_root = self.dataset_root / "perception"
        self.dirs = {
            "skeleton_raw": self.perception_root / "skeleton_raw",
            "skeleton_normalized": self.perception_root / "skeleton_normalized",
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
        处理单张 RGB-D 样本，提取 3D 骨架、归一化、健康度校验并持久化。
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

        cam_pose = sample.camera_pose

        # 2. RGB-D Skeleton Extractor 推理
        skeleton3d = self.extractor.extract(
            rgb=rgb_arr,
            depth=depth_arr,
            camera_pose=cam_pose,
        )

        # 3. 3D 骨架归一化 (Root Center & Scale Normalization)
        skeleton3d = self.skeleton_normalizer.normalize(skeleton3d)

        # 4. 3D 坐标合理性与健康度校验
        val_res = self.coordinate_validator.validate(skeleton3d)

        # 5. 构造元数据记录
        record = {
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "motion_id": sample.motion_id,
            "action_label": sample.action_label,
            "frame_idx": sample.frame_idx,
            "view_id": sample.view_id,
            "extractor_name": "MediaPipeBlazePose3D",
            "joint_format": skeleton3d.joint_format,
            "num_joints": len(skeleton3d.joint_names),
            "mean_confidence": float(np.mean(skeleton3d.perception_confidence)),
            "num_valid_joints": int(np.sum(~skeleton3d.uncertainty_mask)),
            "validation_status": val_res.status,
            "validation_reasons": val_res.reasons,
            "part_confidence": skeleton3d.part_confidence,
            "skeleton_raw_path": f"perception/skeleton_raw/{sample.sample_id}.json",
            "skeleton_normalized_path": f"perception/skeleton_normalized/{sample.sample_id}.json",
            "confidence_path": f"perception/confidence/{sample.sample_id}.json",
        }

        # 6. 分级持久化产物
        if save_outputs:
            raw_file = self.dirs["skeleton_raw"] / f"{sample.sample_id}.json"
            norm_file = self.dirs["skeleton_normalized"] / f"{sample.sample_id}.json"
            conf_file = self.dirs["confidence"] / f"{sample.sample_id}.json"

            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_id": sample.sample_id,
                    "extractor": "MediaPipeBlazePose3D",
                    "joint_format": skeleton3d.joint_format,
                    "coordinate_system": "camera_frame_xyz_meters (X:right, Y:up, Z:forward)",
                    "joints_3d_camera": skeleton3d.joints_3d_camera.tolist(),
                    "joints_3d_world": skeleton3d.joints_3d_world.tolist(),
                    "joints_2d": skeleton3d.joints_2d.tolist(),
                    "joint_names": skeleton3d.joint_names,
                }, f, indent=2)

            with open(norm_file, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_id": sample.sample_id,
                    "joint_format": skeleton3d.joint_format,
                    "coordinate_system": "root_centered_scale_normalized (ST-GCN input)",
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
        logger.info("Processing %d samples with RGB-D Skeleton Extractor...", len(raw_samples))
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
                "extractor_name": "MediaPipeBlazePose3D",
                "model_version": "0.10.14",
                "joint_format": "MediaPipe33",
                "total_processed": len(records),
                "validation_summary": val_summary,
                "records": records,
            }, f, indent=2)

        logger.info("Saved perception dataset manifest to: %s", out_manifest_p)
        return records
