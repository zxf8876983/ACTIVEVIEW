"""
观测数据持久化记录器 —— recorder.py
==================================

职责：
    1. 保存单帧 RGB 图像 (.png) 与 Depth 深度图 (.npy)；
    2. 序列化保存 FrameMetadata (.json)；
    3. 输出整段观测序列汇总 sequence_summary.json；
    4. 统一使用相对数据路径管理。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ea_avs_mvp_v7.core.paths import get_data_root, to_relative_data_path
from .metadata import FrameMetadata, SequenceMetadata

logger = logging.getLogger(__name__)


class ObservationRecorder:
    """观测数据落盘记录器。"""

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else get_data_root() / "runs" / "v70_observations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record_frame(
        self,
        target_dir: Path,
        frame_idx: int,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        metadata: FrameMetadata,
    ) -> FrameMetadata:
        """保存单帧数据与元数据。"""
        rgb_dir = target_dir / "rgb"
        depth_dir = target_dir / "depth"
        meta_dir = target_dir / "metadata"

        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{frame_idx:04d}"

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

        # 3. 保存 Frame JSON
        meta_path = meta_dir / f"{fname}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        return metadata

    def record_sequence_summary(
        self,
        target_dir: Path,
        sequence_meta: SequenceMetadata,
    ) -> Path:
        """保存序列汇总信息。"""
        target_dir.mkdir(parents=True, exist_ok=True)
        summary_path = target_dir / "sequence_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(sequence_meta.to_dict(), f, indent=2, ensure_ascii=False)
        return summary_path
