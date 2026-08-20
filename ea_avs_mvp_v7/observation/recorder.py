"""
观测数据持久化记录器 —— recorder.py
==================================

职责：
    1. 保存单帧 RGB 图像 (rgb/frame_000000.png) 与 Depth 深度图 (depth/frame_000000.npy)；
    2. 保存单帧人体 3D GT 姿态 (human_pose/frame_000000.json)；
    3. 序列化保存规范的 Episode metadata.json；
    4. 统一使用相对数据路径管理。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.core.paths import get_data_root, to_relative_data_path
from .metadata import FrameMetadata

logger = logging.getLogger(__name__)


class ObservationRecorder:
    """观测数据落盘记录器。"""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else get_data_root() / "runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_frame(
        self,
        target_dir: Path,
        frame_idx: int,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        metadata: FrameMetadata,
        human_pose_gt: Optional[Dict[str, Any]] = None,
    ) -> FrameMetadata:
        """保存单帧数据、深度图与 GT 人体姿态。"""
        rgb_dir = target_dir / "rgb"
        depth_dir = target_dir / "depth"
        pose_dir = target_dir / "human_pose"

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        fname = f"frame_{frame_idx:06d}"

        # 1. 保存 RGB
        if rgb is not None:
            rgb_path = rgb_dir / f"{fname}.png"
            img = Image.fromarray(rgb)
            img.save(rgb_path)
            metadata.rgb_relative_path = to_relative_data_path(rgb_path)

        # 2. 保存 Depth
        if depth is not None:
            depth_path = depth_dir / f"{fname}.npy"
            np.save(depth_path, depth.astype(np.float32))
            metadata.depth_relative_path = to_relative_data_path(depth_path)

        # 3. 保存 Human Pose GT
        if human_pose_gt is not None:
            pose_path = pose_dir / f"{fname}.json"
            pose_data = {
                "frame_id": frame_idx,
                "timestamp": metadata.timestamp,
                "human_pose_gt": human_pose_gt,
            }
            with open(pose_path, "w", encoding="utf-8") as f:
                json.dump(pose_data, f, indent=2, ensure_ascii=False)

        return metadata

    def record_episode_metadata(
        self,
        target_dir: Path,
        metadata_dict: Dict[str, Any],
    ) -> Path:
        """保存 Episode 的顶层 metadata.json。"""
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_path = target_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        return meta_path
