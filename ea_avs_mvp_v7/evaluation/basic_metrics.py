"""
基础数据集统计指标 —— basic_metrics.py
======================================

功能：
    1. 统计 Episode 生成的帧完整率、RGB/Depth 文件存在性与 3D 关节坐标覆盖率；
    2. 辅助数据集质量校验。
"""

from pathlib import Path
from typing import Any, Dict

from ea_avs_mvp_v7.core.episode import Episode
from ea_avs_mvp_v7.core.paths import from_relative_data_path


def compute_episode_statistics(episode: Episode) -> Dict[str, Any]:
    """计算单个 Episode 的质量与完整性指标。"""
    num_frames = len(episode.frames)
    rgb_valid_count = 0
    depth_valid_count = 0
    gt_keypoint_counts = []

    for f in episode.frames:
        if f.rgb_relative_path:
            p = from_relative_data_path(f.rgb_relative_path)
            if p.exists() and p.stat().st_size > 0:
                rgb_valid_count += 1

        if f.depth_relative_path:
            p = from_relative_data_path(f.depth_relative_path)
            if p.exists() and p.stat().st_size > 0:
                depth_valid_count += 1

        gt_keypoint_counts.append(len(f.human_pose_gt_world))

    avg_kpts = sum(gt_keypoint_counts) / max(1, len(gt_keypoint_counts))

    return {
        "episode_id": episode.episode_id,
        "action_class": episode.action_class,
        "total_frames": num_frames,
        "rgb_valid_ratio": rgb_valid_count / max(1, num_frames),
        "depth_valid_ratio": depth_valid_count / max(1, num_frames),
        "avg_gt_keypoints_per_frame": avg_kpts,
        "is_complete": (rgb_valid_count == num_frames and depth_valid_count == num_frames and num_frames > 0),
    }
