"""
统一主动视点数据结构定义 —— viewpoint_types.py
=============================================

职责：
    1. 提供 ACTIVEVIEW v11.0 统一的 Viewpoint 数据结构；
    2. 包含空间三维坐标、方位角、视距、机器人相机姿态 (yaw, pitch)、导航代价与状态标志；
    3. 支持序列化字典导出与规范化访问。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Viewpoint:
    """标准主动候选视点数据类。"""
    id: int                                 # 视点唯一标识 ID
    position: List[float]                   # 机器人底盘三维坐标 [x, y, z] (米)
    rotation: List[float]                   # 机器人旋转 [yaw_deg, pitch_deg] (度)
    yaw: float                              # 水平朝向角 (度, 0° 表示朝向 +Z, 逆时针为正, 自动朝向人体)
    pitch: float                            # 俯仰角 (度, 0° 为平视, 负值为低头看人体)
    distance: float                         # 视点到人体的欧氏距离 (米)
    angle: float                            # 围绕人体中心的水平圆周角 θ (度, [0, 360))
    camera_height: float = 1.25             # 机器人相机相对地面的垂直高度 (米)
    navigation_cost: float = 0.0            # 从机器人当前位置到该视点的测地线/欧氏导航代价 (米)
    is_navigable: bool = True               # 是否在 NavMesh 可行域内
    is_reachable: bool = True               # 机器人是否具备有效导航路径可到达
    is_visible: bool = True                 # 视点与人体之间是否无遮挡通视
    metadata: Dict[str, Any] = field(default_factory=dict) # 扩展元数据 (如 view_id, radius, angle_deg 等)

    @property
    def is_feasible(self) -> bool:
        """综合可行性：必须同时满足可行走、可到达、且视线可见。"""
        return self.is_navigable and self.is_reachable and self.is_visible

    @property
    def camera_position(self) -> List[float]:
        """获取相机传感器在世界坐标系下的三维坐标 [x, y + camera_height, z]。"""
        return [
            float(self.position[0]),
            float(self.position[1] + self.camera_height),
            float(self.position[2]),
        ]

    def to_dict(self) -> Dict[str, Any]:
        """导出为结构化字典。"""
        return {
            "id": int(self.id),
            "position": [round(float(p), 4) for p in self.position],
            "camera_position": [round(float(cp), 4) for cp in self.camera_position],
            "rotation": [round(float(r), 2) for r in self.rotation],
            "yaw": round(float(self.yaw), 2),
            "pitch": round(float(self.pitch), 2),
            "distance": round(float(self.distance), 4),
            "angle": round(float(self.angle), 2),
            "camera_height": round(float(self.camera_height), 4),
            "navigation_cost": round(float(self.navigation_cost), 4),
            "is_navigable": bool(self.is_navigable),
            "is_reachable": bool(self.is_reachable),
            "is_visible": bool(self.is_visible),
            "is_feasible": bool(self.is_feasible),
            "metadata": self.metadata,
        }
