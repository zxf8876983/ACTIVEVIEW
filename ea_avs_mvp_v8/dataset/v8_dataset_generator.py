"""
v8 视点数据集生成器 —— v8_dataset_generator.py
=============================================

职责：
    1. 整合 HumanPlacement, ViewpointGenerator, ConstraintChecker 与 VisibilityEvaluator；
    2. 生成符合规范的 v8 视点数据集结构：
       data/ActiveView/v8/episode_xxx/
       ├── rgb/
       ├── depth/
       ├── human_pose/
       ├── candidate_views.json
       ├── visibility.json
       └── metadata.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ea_avs_mvp_v8.core.paths import get_data_root, to_relative_data_path
from ea_avs_mvp_v8.core.types import CandidateViewpoint, ViewpointQuality, HumanPlacementPose
from ea_avs_mvp_v8.evaluation.view_metrics import summarize_viewpoint_qualities

logger = logging.getLogger(__name__)


def save_v8_viewpoint_dataset(
    output_dir: Union[str, Path],
    scene_id: str,
    episode_id: str,
    human_pose: HumanPlacementPose,
    action_info: Dict[str, Any],
    robot_start_pos: List[float],
    candidate_views: List[CandidateViewpoint],
    view_qualities: List[ViewpointQuality],
) -> Path:
    """持久化保存 v8 标准视点数据集与元数据。"""
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. 保存 candidate_views.json
    views_json_path = out_p / "candidate_views.json"
    views_data = [vp.to_dict() for vp in candidate_views]
    with open(views_json_path, "w", encoding="utf-8") as f:
        json.dump(views_data, f, indent=2, ensure_ascii=False)

    # 2. 保存 visibility.json
    vis_json_path = out_p / "visibility.json"
    qualities_data = [vq.to_dict() for vq in view_qualities]
    summary = summarize_viewpoint_qualities(view_qualities)
    with open(vis_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "viewpoint_evaluations": qualities_data,
        }, f, indent=2, ensure_ascii=False)

    # 3. 保存 metadata.json
    meta_json_path = out_p / "metadata.json"
    metadata = {
        "scene_id": scene_id,
        "episode_id": episode_id,
        "human": {
            "avatar": action_info.get("avatar", "neutral_0"),
            "action_class": action_info.get("action_class", "fall_related"),
            "action_label": action_info.get("action_label", "fall to the ground"),
            "motion_id": action_info.get("motion_id", "fall_related_3522"),
            "position": [float(x) for x in human_pose.position],
            "yaw_deg": float(human_pose.yaw_deg),
        },
        "robot": {
            "initial_position": [float(x) for x in robot_start_pos],
        },
        "candidate_views": {
            "total_candidates": len(candidate_views),
            "feasible_candidates": sum(1 for v in candidate_views if v.feasible),
            "views_file": to_relative_data_path(views_json_path),
        },
        "visibility_summary": summary,
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Saved v8 viewpoint dataset to: %s", out_p)
    return out_p
