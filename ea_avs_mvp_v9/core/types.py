"""
v9.1 核心数据结构定义 —— types.py
==================================

基于当前感知质量的人体主动视角选择 (Perception-aware Active View Selection)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ActionClass(str, Enum):
    """人体动作类别 (仅作为实验分组、性能分析与数据统计元数据，严禁输入模型)。"""
    FALL = "fall"
    SITTING = "sitting"
    STANDING = "standing"
    BENDING = "bending"
    REACHING = "reaching"


@dataclass
class ActionEmbedding:
    """动作先验嵌入 (仅供 v9.0 规则基线对比使用)。"""
    action_class: Optional[ActionClass] = None
    action_name: Optional[str] = None
    region_weights: Dict[str, float] = field(default_factory=dict)
    critical_regions: List[str] = field(default_factory=list)
    preferred_angle_range: Any = (0.0, 360.0)
    optimal_distance: float = 2.0
    aspect_weight: float = 0.25
    distance_weight: float = 0.15
    vector: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionViewpointScore:
    """动作条件化视点得分 (仅供 v9.0 规则基线对比使用)。"""
    viewpoint_id: str
    action_class: Optional[ActionClass] = None
    action_name: Optional[str] = None
    total_score: float = 0.0
    geometry_score: float = 0.0
    action_delta: float = 0.0
    visibility_score: float = 0.0
    angle_alignment_score: float = 0.0
    region_score: float = 0.0
    aspect_score: float = 0.0
    distance_score: float = 0.0
    evaluation_mode: str = "oracle"
    pose_source: str = "oracle"
    metadata: Dict[str, Any] = field(default_factory=dict)
    feasible: bool = True


@dataclass
class ObservationState:
    """当前视角下由感知模块估计的人体不完整观测状态 (Estimated Observation State)。"""
    estimated_joints_3d: Dict[str, List[float]]  # 16 骨骼关节点 3D 估计坐标 (p_est = p_gt + epsilon)
    joint_confidences: Dict[str, float]          # 16 关节点估计置信度 [0.0, 1.0]
    body_part_confidences: Dict[str, float]      # 7 大解剖部位可见性置信度 (head, torso, pelvis, hands, legs)
    missing_joint_names: List[str] = field(default_factory=list)  # 缺失关节点列表
    viewpoint_id: str = "current_robot_view"    # 当前观测视点 ID
    mean_confidence: float = 0.0                # 平均关节置信度
    missing_joint_count: int = 0                # 缺失/不可见关节点数量
    completeness_score: float = 1.0             # 观测完整度得分 [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "mean_confidence": round(self.mean_confidence, 3),
            "missing_joint_count": self.missing_joint_count,
            "completeness_score": round(self.completeness_score, 3),
            "missing_joint_names": self.missing_joint_names,
            "joint_confidences": {k: round(v, 3) for k, v in self.joint_confidences.items()},
            "body_part_confidences": {k: round(v, 3) for k, v in self.body_part_confidences.items()},
        }


@dataclass
class ViewFeature:
    """候选视点多维几何特征描述子 (严禁包含未来真实观测状态)。"""
    viewpoint_id: str
    distance: float
    viewing_angle_deg: float
    pose_coverage: float
    visibility_loss_ratio: float
    projected_area_ratio: float
    region_coverages: Dict[str, float] = field(default_factory=dict)
    body_part_visibilities: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "distance": round(self.distance, 3),
            "viewing_angle_deg": round(self.viewing_angle_deg, 1),
            "pose_coverage": round(self.pose_coverage, 3),
            "visibility_loss_ratio": round(self.visibility_loss_ratio, 3),
            "projected_area_ratio": round(self.projected_area_ratio, 4),
            "region_coverages": {k: round(v, 3) for k, v in self.region_coverages.items()},
            "body_part_visibilities": {k: round(v, 3) for k, v in self.body_part_visibilities.items()},
            "feasible": self.feasible,
        }


@dataclass
class InformationGainScore:
    """视角迁移带来的信息增益 (Information Gain)。"""
    viewpoint_id: str
    gain: float
    confidence_before: float
    confidence_after: float
    missing_recovered_count: int
    quality_before: float
    quality_after: float
    feasible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "gain": round(self.gain, 3),
            "confidence_before": round(self.confidence_before, 3),
            "confidence_after": round(self.confidence_after, 3),
            "missing_recovered_count": self.missing_recovered_count,
            "quality_before": round(self.quality_before, 3),
            "quality_after": round(self.quality_after, 3),
            "feasible": self.feasible,
        }


@dataclass
class OracleViewpointResult:
    """Oracle 理论上限视点结果 (仅用于性能上限评估，严禁输入模型)。"""
    best_viewpoint_id: str
    oracle_visibility_score: float
    oracle_quality_score: float
    oracle_information_gain: float
    oracle_joints_visible_count: int
    oracle_body_parts_visibility: Dict[str, float] = field(default_factory=dict)
