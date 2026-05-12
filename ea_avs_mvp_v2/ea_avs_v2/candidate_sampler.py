"""
候选视角采样模块 —— candidate_sampler.py
==========================================

功能：
    围绕人体目标生成候选观察位姿，并通过导航网格进行可达性过滤。

数据结构：
    CandidateView —— 核心数据结构，包含位置、朝向、距离、预测评分、真实评分等。
                    v2.0 新增 pred_score、true_score、selected_by 字段。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .geometry import compute_look_at_yaw


@dataclass
class CandidateView:
    """候选观察位姿数据结构。

    v2.0 新增字段：
        pred_score: 移动前预测评分（由 PredictiveEvaluator 计算）
        true_score: 到达后真实评分（由 TrueEvaluator 计算）
        selected_by: 被哪些策略选中（用于调试和可视化）

    字段说明：
        candidate_id:   候选点编号（current_view 使用 -1）
        position:       三维位置，shape=(3,)
        yaw:            朝向角（弧度），朝向人体
        geodesic_distance: 从起点到候选点的测地距离（米）
        euclidean_distance_to_human: 到人体的欧氏距离（米）
        is_valid:       是否有效
        invalid_reason: 无效原因
        pred_score:     预测评分字典
        true_score:     真实评分字典
        selected_by:    选中此位姿的策略名称列表
    """
    candidate_id: int
    position: np.ndarray
    yaw: float
    geodesic_distance: float
    euclidean_distance_to_human: float
    is_valid: bool
    invalid_reason: str = ""
    pred_score: dict = field(default_factory=dict)
    true_score: dict = field(default_factory=dict)
    selected_by: list = field(default_factory=list)


class CandidateSampler:
    """候选视角采样器 —— 生成并过滤候选观察位姿。"""

    def __init__(self, config: dict):
        """初始化采样器。

        参数：
            config: 需要 "candidate_sampling" 配置段。
        """
        self.config = config

    def sample(
        self,
        human_pos: np.ndarray,
        robot_pos: np.ndarray,
        runner,
    ) -> List[CandidateView]:
        """围绕人体采样候选位姿，并用导航网格过滤。

        参数：
            human_pos: 人体基座位置，shape=(3,)。
            robot_pos: 机器人起始位置，shape=(3,)。
            runner: HabitatRunner 实例。

        返回：
            CandidateView 列表（包含有效和无效点）。
        """
        sampling_cfg = self.config["candidate_sampling"]
        radii = sampling_cfg["radii"]
        angles_deg = sampling_cfg["angles_deg"]
        min_dist = sampling_cfg["min_distance_to_human"]
        max_dist = sampling_cfg["max_distance_to_human"]
        max_geo = sampling_cfg["max_geodesic_distance"]

        candidates: List[CandidateView] = []
        candidate_id = 0
        angles_rad = np.deg2rad(angles_deg)

        for r in radii:
            for theta in angles_rad:
                # 计算理想位置（X-Z 平面）
                cx = human_pos[0] + r * np.cos(theta)
                cz = human_pos[2] + r * np.sin(theta)
                cy = human_pos[1]
                candidate = np.array([cx, cy, cz], dtype=np.float32)

                # 吸附到导航网格
                snapped = runner.snap_point(candidate)

                # 检查导航性
                if not runner.is_navigable(snapped):
                    candidates.append(CandidateView(
                        candidate_id=candidate_id,
                        position=snapped,
                        yaw=0.0,
                        geodesic_distance=float("inf"),
                        euclidean_distance_to_human=float("inf"),
                        is_valid=False,
                        invalid_reason="not_navigable",
                    ))
                    candidate_id += 1
                    continue

                # 计算距离
                euclidean_dist = float(np.linalg.norm(snapped - human_pos))
                geo_dist = runner.geodesic_distance(robot_pos, snapped)

                # 欧氏距离过滤
                if euclidean_dist < min_dist or euclidean_dist > max_dist:
                    candidates.append(CandidateView(
                        candidate_id=candidate_id,
                        position=snapped,
                        yaw=0.0,
                        geodesic_distance=geo_dist,
                        euclidean_distance_to_human=euclidean_dist,
                        is_valid=False,
                        invalid_reason="out_of_distance_range",
                    ))
                    candidate_id += 1
                    continue

                # 测地距离过滤
                if geo_dist == float("inf") or geo_dist > max_geo:
                    reason = ("no_geodesic_path" if geo_dist == float("inf")
                              else "geodesic_distance_too_large")
                    candidates.append(CandidateView(
                        candidate_id=candidate_id,
                        position=snapped,
                        yaw=0.0,
                        geodesic_distance=geo_dist,
                        euclidean_distance_to_human=euclidean_dist,
                        is_valid=False,
                        invalid_reason=reason,
                    ))
                    candidate_id += 1
                    continue

                # 计算朝向（看向人体）
                yaw = compute_look_at_yaw(snapped, human_pos)

                # 有效候选点
                candidates.append(CandidateView(
                    candidate_id=candidate_id,
                    position=snapped,
                    yaw=yaw,
                    geodesic_distance=geo_dist,
                    euclidean_distance_to_human=euclidean_dist,
                    is_valid=True,
                ))
                candidate_id += 1

        return candidates
