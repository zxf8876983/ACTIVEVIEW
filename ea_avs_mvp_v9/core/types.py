"""
v9.0 核心数据结构与类型定义 —— types.py
=======================================

包含：
    1. ActionClass: 动作类别枚举 (FALL, SITTING, STANDING, BENDING, REACHING)；
    2. ActionEmbedding: 动作嵌入与先验属性结构；
    3. ViewFeature: 视角观测特征结构 (几何、姿态与身体分区覆盖率)；
    4. ActionViewpointScore: 动作条件视角打分结构 (Q_geom, Delta_Q, Q_action)；
    5. ActionSelectionReport: 动作感知视点评测汇总报表。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionClass(str, Enum):
    """支持的标准动作分类。"""
    FALL = "fall"
    SITTING = "sitting"
    STANDING = "standing"
    BENDING = "bending"
    REACHING = "reaching"


@dataclass
class ActionEmbedding:
    """动作表示与观测先验数据结构。"""
    action_name: str
    action_class: ActionClass
    vector: List[float]                  # One-hot 或连续特征向量
    critical_regions: List[str]          # 该动作关键观察身体部位
    preferred_angle_range: List[float]   # 推荐视角偏角区间 [min_deg, max_deg]
    optimal_distance: float              # 推荐最优观测距离 (米)
    region_weights: Dict[str, float] = field(default_factory=dict)
    aspect_weight: float = 0.25
    distance_weight: float = 0.15
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_class": self.action_class.value,
            "vector": [float(x) for x in self.vector],
            "critical_regions": self.critical_regions,
            "preferred_angle_range": [float(x) for x in self.preferred_angle_range],
            "optimal_distance": float(self.optimal_distance),
            "region_weights": self.region_weights,
            "aspect_weight": float(self.aspect_weight),
            "distance_weight": float(self.distance_weight),
            "metadata": self.metadata,
        }


@dataclass
class ViewFeature:
    """视点多维特征提取结果。"""
    viewpoint_id: str
    distance: float                      # 到人体中心距离 (米)
    viewing_angle_deg: float             # 相对人体正面的观测偏角 (度)
    pose_coverage: float                 # 全身 16 关键点覆盖率 [0.0, 1.0]
    visibility_loss_ratio: float         # 姿态未观测损失比例 [0.0, 1.0]
    projected_area_ratio: float          # 人体投影面积占比 [0.0, 1.0]
    body_part_visibilities: Dict[str, float] = field(default_factory=dict)  # 7 大身体关键解剖部位可见性
    region_coverages: Dict[str, float] = field(default_factory=dict)        # 兼容旧字段
    feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "distance": float(self.distance),
            "viewing_angle_deg": float(self.viewing_angle_deg),
            "pose_coverage": float(self.pose_coverage),
            "visibility_loss_ratio": float(self.visibility_loss_ratio),
            "projected_area_ratio": float(self.projected_area_ratio),
            "body_part_visibilities": {k: float(v) for k, v in self.body_part_visibilities.items()},
            "region_coverages": {k: float(v) for k, v in self.region_coverages.items()},
            "feasible": bool(self.feasible),
            "metadata": self.metadata,
        }


@dataclass
class ActionViewpointScore:
    """动作感知视点综合打分结果。"""
    viewpoint_id: str
    action_name: str
    geometry_score: float                # 基础几何得分 Q_geom(v)
    action_delta: float                  # 动作适配增益 Delta_Q(a, v)
    total_score: float                   # 综合评分 Q(v|a)
    region_score: float                  # 关键区域匹配得分
    aspect_score: float                  # 视角朝向偏好得分
    distance_score: float                # 距离适配得分
    evaluation_mode: str = "oracle"      # oracle / estimated
    pose_source: str = "oracle"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "action_name": self.action_name,
            "geometry_score": float(self.geometry_score),
            "action_delta": float(self.action_delta),
            "total_score": float(self.total_score),
            "region_score": float(self.region_score),
            "aspect_score": float(self.aspect_score),
            "distance_score": float(self.distance_score),
            "evaluation_mode": self.evaluation_mode,
            "pose_source": self.pose_source,
            "metadata": self.metadata,
        }
