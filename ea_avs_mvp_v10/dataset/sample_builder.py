"""
v10 样本持久化构建器 —— sample_builder.py
=======================================

职责：
    1. 将单帧 RGB、Depth、CameraPose、GT 骨骼及 Action 标签组装并持久化写入 v10 标准数据集目录结构；
    2. 生成规范的 V10Sample 元数据对象；
    3. 维护 metadata/samples.json 样本索引清单。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ea_avs_mvp_v10.core.paths import init_v10_dataset_dirs
from ea_avs_mvp_v10.core.types import CameraPose, V10Sample
from ea_avs_mvp_v10.sensors.rgbd_capture import RGBDCapture

logger = logging.getLogger(__name__)


class V10SampleBuilder:
    """v10 样本写入与数据集构建器。"""

    def __init__(self, dataset_root: Optional[Union[str, Path]] = None):
        self.dirs = init_v10_dataset_dirs(dataset_root)
        self.root = self.dirs["root"]

    def build_and_save_sample(
        self,
        sample_id: str,
        scene_id: str,
        motion_id: str,
        action_label: str,
        frame_idx: int,
        view_id: str,
        rgb_array: np.ndarray,
        depth_array: np.ndarray,
        camera_pose: CameraPose,
        gt_skeleton: Optional[Dict[str, List[float]]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        save_visual_depth: bool = True,
    ) -> V10Sample:
        """
        持久化保存单个样本的所有模态数据与元数据。

        Returns:
            sample: V10Sample 实例
        """
        # 1. 保存 RGB (PNG)
        rgb_filename = f"{sample_id}.png"
        rgb_file_path = self.dirs["raw_rgb"] / rgb_filename
        RGBDCapture.save_rgb_image(rgb_array, rgb_file_path)
        rel_rgb_path = str(Path("raw") / "rgb" / rgb_filename)

        # 2. 保存 Depth (NPY)
        depth_filename = f"{sample_id}.npy"
        depth_file_path = self.dirs["raw_depth"] / depth_filename
        RGBDCapture.save_depth_array(depth_array, depth_file_path)
        rel_depth_path = str(Path("raw") / "depth" / depth_filename)

        # 可选：保存 Depth 可视化图
        if save_visual_depth:
            depth_vis_p = self.dirs["raw_depth"] / f"{sample_id}_vis.png"
            RGBDCapture.save_depth_visual(depth_array, depth_vis_p)

        # 3. 保存 CameraPose (JSON)
        cam_pose_filename = f"{sample_id}.json"
        cam_pose_path = self.dirs["raw_camera_pose"] / cam_pose_filename
        RGBDCapture.save_camera_pose(camera_pose, cam_pose_path)

        # 4. 保存 Scene & Action Meta
        scene_meta_p = self.dirs["raw_scene_meta"] / f"{sample_id}.json"
        scene_meta_data = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "motion_id": motion_id,
            "frame_idx": frame_idx,
            "view_id": view_id,
            "extra": extra_metadata or {},
        }
        with open(scene_meta_p, "w", encoding="utf-8") as f:
            json.dump(scene_meta_data, f, indent=2, ensure_ascii=False)

        # 5. 保存 Ground Truth (Action & Skeleton)
        act_gt_p = self.dirs["gt_action"] / f"{sample_id}.json"
        with open(act_gt_p, "w", encoding="utf-8") as f:
            json.dump({"sample_id": sample_id, "action_label": action_label, "motion_id": motion_id}, f, indent=2)

        rel_gt_skel_path = None
        if gt_skeleton is not None:
            skel_filename = f"{sample_id}.json"
            skel_p = self.dirs["gt_skeleton"] / skel_filename
            with open(skel_p, "w", encoding="utf-8") as f:
                json.dump({
                    "sample_id": sample_id,
                    "comment": "GT is only used for supervision/evaluation/Oracle. Never enter model forward pass.",
                    "joints_3d": {k: [round(float(c), 4) for c in v] for k, v in gt_skeleton.items()},
                }, f, indent=2, ensure_ascii=False)
            rel_gt_skel_path = str(Path("ground_truth") / "skeleton" / skel_filename)

        # 6. 构造标准样本对象
        sample = V10Sample(
            sample_id=sample_id,
            scene_id=scene_id,
            motion_id=motion_id,
            action_label=action_label,
            frame_idx=frame_idx,
            view_id=view_id,
            camera_pose=camera_pose,
            rgb_path=rel_rgb_path,
            depth_path=rel_depth_path,
            gt_skeleton_path=rel_gt_skel_path,
            metadata=extra_metadata or {},
        )

        return sample

    def save_dataset_manifest(self, samples: List[V10Sample], manifest_name: str = "samples.json") -> Path:
        """保存数据集样本清单至 metadata 目录。"""
        manifest_p = self.dirs["metadata"] / manifest_name
        data = {
            "dataset_version": "10.0.0",
            "phase": "Phase 1: Habitat RGB-D Dataset Generation",
            "total_samples": len(samples),
            "samples": [s.to_dict() for s in samples],
        }
        with open(manifest_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Dataset manifest saved (%d samples) to: %s", len(samples), manifest_p)
        return manifest_p
