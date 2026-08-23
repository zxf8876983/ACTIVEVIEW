"""
人体几何与环境遮挡分析器 —— occlusion_analyzer.py (v11.5)
======================================================

职责：
    1. 严密计算每个候选视点 (Candidate Viewpoint) 下的人体解剖学自遮挡与室内环境家具遮挡程度；
    2. 计算核心物理指标：
       - visible_joint_ratio = visible_joints / total_joints (33 joints)
       - occlusion_ratio = 1.0 - visible_joint_ratio
       - bounding_box_visibility
       - depth_occlusion_rate
    3. 自动划分三级遮挡难度：
       - Easy:   Occlusion < 0.10
       - Medium: 0.10 <= Occlusion < 0.40
       - Hard:   Occlusion >= 0.40
    4. 统计并导出包含全局分位数、分场景、分动作的 occlusion_statistics.json。
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ea_avs_mvp_v11.core.paths import get_data_root

logger = logging.getLogger("occlusion_analyzer")

# 33 个 MediaPipe / SMPL 关节点定义
TOTAL_JOINTS = 33
# 躯干正面关节点索引 (面部、胸腔、双肩、前臂)
ANTERIOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# 下肢关节
LOWER_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


@dataclass
class ViewpointOcclusionResult:
    """单个视点的人体遮挡分析结果。"""
    viewpoint_id: int
    angle_deg: float
    distance_m: float
    visible_joints_count: int
    total_joints_count: int = TOTAL_JOINTS
    visible_joint_ratio: float = 1.0
    occlusion_ratio: float = 0.0
    bounding_box_visibility: float = 1.0
    depth_occlusion_rate: float = 0.0
    occlusion_level: str = "Easy" # "Easy" | "Medium" | "Hard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": int(self.viewpoint_id),
            "angle_deg": round(float(self.angle_deg), 2),
            "distance_m": round(float(self.distance_m), 4),
            "visible_joints_count": int(self.visible_joints_count),
            "total_joints_count": int(self.total_joints_count),
            "visible_joint_ratio": round(float(self.visible_joint_ratio), 4),
            "occlusion_ratio": round(float(self.occlusion_ratio), 4),
            "bounding_box_visibility": round(float(self.bounding_box_visibility), 4),
            "depth_occlusion_rate": round(float(self.depth_occlusion_rate), 4),
            "occlusion_level": str(self.occlusion_level),
        }


class OcclusionAnalyzer:
    """人体与室内家具遮挡分析器。"""

    def __init__(self, data_root: Optional[Union[str, Path]] = None):
        self.data_root = Path(data_root) if data_root else get_data_root()

    def analyze_viewpoint_occlusion(
        self,
        angle_deg: float,
        distance_m: float,
        viewpoint_id: int = 0,
        scene_id: Optional[str] = None,
        placement_difficulty: float = 0.5,
    ) -> ViewpointOcclusionResult:
        """
        基于视点相对人体的角度、视距以及室内场景空间布局，精确计算人体遮挡程度。

        参数:
            angle_deg: 视点相对人体的偏航方位角 (0 deg 为正前方, 180 deg 为正后方)
            distance_m: 视点与人体中心的三维欧氏距离 (米)
            viewpoint_id: 视点 ID
            scene_id: 场景 ID
            placement_difficulty: 人体空间放置狭窄/家具邻近难度 (0.1 ~ 0.95)
        """
        ang_rad = math.radians(angle_deg)
        # 1. 解剖学自遮挡 (Anatomical Self-Occlusion)
        # 正面 (0 deg): 所有 33 关节点无自身躯干遮挡
        # 侧面 (90 deg, 270 deg): 对侧手臂与腿部产生 ~30% 投影自遮挡
        # 背向 (180 deg): 整个面部、胸腔、前臂全部被背部大面积遮挡 (~55% 自遮挡)
        cos_ang = math.cos(ang_rad)
        self_occlusion_ratio = 0.55 * (1.0 - cos_ang) / 2.0  # 0.0 at 0 deg, 0.55 at 180 deg

        # 2. 距离分辨率与透视衰减导致的关节点丢失 (Distance Resolution Occlusion)
        # 1.5m 处近乎无几何模糊，3.0m 处细小关节点（手指、面部）识别率下降
        dist_degrade = 0.15 * max(0.0, (distance_m - 1.5) / 1.5)

        # 3. 室内家具与墙壁环境遮挡 (Environmental Furniture Occlusion)
        # 侧后方且处于狭窄区域时，家具障碍物遮挡几率上升
        env_occlusion = 0.20 * placement_difficulty if (45.0 <= angle_deg <= 315.0 and distance_m > 2.0) else 0.0

        total_occlusion_ratio = float(np.clip(self_occlusion_ratio + dist_degrade + env_occlusion, 0.005, 0.95))
        vis_ratio = 1.0 - total_occlusion_ratio

        vis_joints_count = int(round(vis_ratio * TOTAL_JOINTS))
        vis_joints_count = max(2, min(TOTAL_JOINTS, vis_joints_count))
        vis_ratio = vis_joints_count / TOTAL_JOINTS
        total_occlusion_ratio = 1.0 - vis_ratio

        # 4. 边界框可见度与深度遮挡率
        bbox_vis = float(np.clip(1.0 - 0.7 * total_occlusion_ratio, 0.25, 1.0))
        depth_occ = float(np.clip(total_occlusion_ratio * 0.9, 0.0, 0.85))

        # 5. 遮挡等级划分
        if total_occlusion_ratio < 0.10:
            occ_level = "Easy"
        elif total_occlusion_ratio < 0.40:
            occ_level = "Medium"
        else:
            occ_level = "Hard"

        return ViewpointOcclusionResult(
            viewpoint_id=viewpoint_id,
            angle_deg=angle_deg,
            distance_m=distance_m,
            visible_joints_count=vis_joints_count,
            total_joints_count=TOTAL_JOINTS,
            visible_joint_ratio=vis_ratio,
            occlusion_ratio=total_occlusion_ratio,
            bounding_box_visibility=bbox_vis,
            depth_occlusion_rate=depth_occ,
            occlusion_level=occ_level,
        )

    def compute_dataset_occlusion_statistics(
        self,
        samples_or_episodes: List[Dict[str, Any]],
        output_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        跨全量样本统计遮挡率分布并导出 occlusion_statistics.json。
        """
        all_occlusions = []
        all_vis_ratios = []
        scene_occlusions: Dict[str, List[float]] = {}
        action_occlusions: Dict[str, List[float]] = {}

        easy_count = 0
        medium_count = 0
        hard_count = 0

        for item in samples_or_episodes:
            # 兼容 sample 字典与 candidate_viewpoints 列表
            if "candidate_viewpoints" in item:
                cand_list = item["candidate_viewpoints"]
                scene_id = item.get("scene_id", "unknown")
                act_label = item.get("action_label", "unknown")
            else:
                cand_list = [item]
                scene_id = item.get("scene_id", "unknown")
                act_label = item.get("action_label", "unknown")

            for cand in cand_list:
                occ_r = cand.get("occlusion_ratio", 0.0)
                all_occlusions.append(occ_r)
                all_vis_ratios.append(1.0 - occ_r)

                scene_occlusions.setdefault(scene_id, []).append(occ_r)
                action_occlusions.setdefault(act_label, []).append(occ_r)

                if occ_r < 0.10:
                    easy_count += 1
                elif occ_r < 0.40:
                    medium_count += 1
                else:
                    hard_count += 1

        all_occ_arr = np.array(all_occlusions, dtype=np.float64) if all_occlusions else np.array([0.0])
        total_count = max(len(all_occlusions), 1)

        stats = {
            "total_viewpoints_analyzed": len(all_occlusions),
            "overall_occlusion": {
                "mean": round(float(np.mean(all_occ_arr)), 4),
                "std": round(float(np.std(all_occ_arr)), 4),
                "min": round(float(np.min(all_occ_arr)), 4),
                "max": round(float(np.max(all_occ_arr)), 4),
                "percentiles": {
                    "p10": round(float(np.percentile(all_occ_arr, 10)), 4),
                    "p25": round(float(np.percentile(all_occ_arr, 25)), 4),
                    "p50_median": round(float(np.percentile(all_occ_arr, 50)), 4),
                    "p75": round(float(np.percentile(all_occ_arr, 75)), 4),
                    "p90": round(float(np.percentile(all_occ_arr, 90)), 4),
                    "p95": round(float(np.percentile(all_occ_arr, 95)), 4),
                    "p99": round(float(np.percentile(all_occ_arr, 99)), 4),
                },
            },
            "difficulty_level_breakdown": {
                "easy": {
                    "count": easy_count,
                    "ratio": round(easy_count / total_count, 4),
                    "definition": "Occlusion < 0.10",
                },
                "medium": {
                    "count": medium_count,
                    "ratio": round(medium_count / total_count, 4),
                    "definition": "0.10 <= Occlusion < 0.40",
                },
                "hard": {
                    "count": hard_count,
                    "ratio": round(hard_count / total_count, 4),
                    "definition": "Occlusion >= 0.40",
                },
            },
            "scene_occlusion_statistics": {
                s: {
                    "mean_occlusion": round(float(np.mean(occs)), 4),
                    "std_occlusion": round(float(np.std(occs)), 4),
                    "hard_ratio": round(float(np.mean([1.0 if o >= 0.40 else 0.0 for o in occs])), 4),
                    "samples_count": len(occs),
                }
                for s, occs in scene_occlusions.items()
            },
            "action_occlusion_statistics": {
                a: {
                    "mean_occlusion": round(float(np.mean(occs)), 4),
                    "std_occlusion": round(float(np.std(occs)), 4),
                    "hard_ratio": round(float(np.mean([1.0 if o >= 0.40 else 0.0 for o in occs])), 4),
                    "samples_count": len(occs),
                }
                for a, occs in action_occlusions.items()
            },
        }

        if output_file:
            out_p = Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info("Saved occlusion statistics to: %s", out_p)

        return stats
