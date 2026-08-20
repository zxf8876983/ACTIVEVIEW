"""
v9.0 数据集与动作元数据加载工具 —— v9_dataset_loader.py
======================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ea_avs_mvp_v8.core.types import CandidateViewpoint
from ea_avs_mvp_v9.core.paths import get_data_root

logger = logging.getLogger(__name__)


def load_v8_episode(episode_dir: Union[str, Path]) -> Dict[str, Any]:
    """读取已生成的 v8 episode 目录中的结构化元数据。"""
    p = Path(episode_dir) if Path(episode_dir).is_absolute() else get_data_root() / episode_dir

    candidate_views_file = p / "candidate_views.json"
    metadata_file = p / "metadata.json"

    candidates: List[CandidateViewpoint] = []
    if candidate_views_file.exists():
        with open(candidate_views_file, "r", encoding="utf-8") as f:
            raw_vps = json.load(f)
            for v in raw_vps:
                candidates.append(CandidateViewpoint(
                    viewpoint_id=v["viewpoint_id"],
                    position=v["position"],
                    yaw_deg=v.get("yaw_deg", v.get("yaw", 0.0)),
                    radius=v.get("radius", 2.0),
                    angle_deg=v.get("angle_deg", 0.0),
                    camera_height=v.get("camera_height", 1.2),
                    ground_height=v.get("ground_height", -1.60),
                    camera_pose=v.get("camera_pose"),
                    feasible=v.get("feasible", True),
                    reason=v.get("reason", "valid"),
                    metadata=v.get("metadata", {}),
                ))

    meta = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "episode_dir": str(p),
        "candidates": candidates,
        "metadata": meta,
        "human_position": meta.get("human_position", [1.5, -1.60, 4.0]),
        "scene_id": meta.get("scene_id", "apartment_1"),
    }


def save_action_metadata(
    output_dir: Union[str, Path],
    action_info: Dict[str, Any],
) -> Path:
    """持久化 action.json 动作元数据。"""
    p = Path(output_dir) if Path(output_dir).is_absolute() else get_data_root() / output_dir
    p.mkdir(parents=True, exist_ok=True)
    out_file = p / "action.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(action_info, f, indent=2, ensure_ascii=False)
    return out_file
