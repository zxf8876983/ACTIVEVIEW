"""
候选视角采样模块 —— candidate_sampler.py
==========================================

功能：
    围绕人体目标生成一批候选观察点，并使用导航网格（navmesh）进行可达性过滤。

MVP0.1 的采样策略：
    - 以人体位置为中心，在多个半径和角度上均匀采样
    - 对每个采样点进行 navmesh 可达性检查
    - 计算测地距离并过滤掉距离过远的点
    - 计算每个有效候选点的朝向（yaw），使其"看向"人体目标

数据结构：
    CandidateView —— 表示一个候选视角，包含位置、朝向、距离、评分等信息
    CandidateSampler —— 采样器类，负责生成和过滤候选视角

评分不在本模块进行，评分由 evaluator.py 中的 ViewpointEvaluator 完成。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .geometry import compute_look_at_yaw


@dataclass
class CandidateView:
    """
    候选视角数据结构 —— 表示机器人可以移动到一个观察位置和朝向。

    这是 MVP0.1 中最核心的数据结构之一，贯穿整个实验流程：
    1. CandidateSampler 生成和过滤候选点
    2. ViewpointEvaluator 对每个候选点计算评分
    3. 各个 Policy 从候选点中选择最佳视角

    字段说明：
        candidate_id: int
            候选点编号。有效候选点从 0 开始递增编号；无效候选点也会获得编号
            但 is_valid=False。初始视角（current_view）编号为 -1。
        
        position: np.ndarray, shape=(3,)
            候选点在导航网格上的三维坐标（已通过 snap_point 吸附到 navmesh）。
        
        yaw: float
            机器人的朝向角（弧度制），使得相机指向人体目标方向。
            通过 compute_look_at_yaw() 计算。
        
        geodesic_distance: float
            从机器人起始位置到候选点的测地距离（米）。
            如果不可达则为 float("inf")。
        
        euclidean_distance_to_human: float
            候选点到人体目标的欧氏直线距离（米）。
        
        is_valid: bool
            候选点是否有效。有效条件：
            1. 在导航网格上（is_navigable）
            2. 到人体的欧氏距离在 [min, max] 范围内
            3. 测地距离可达且不超过 max_geodesic_distance
        
        invalid_reason: str
            如果 is_valid=False，记录失效原因。用于调试输出。
            可能的值： "not_navigable", "out_of_distance_range",
                      "no_geodesic_path", "geodesic_distance_too_large"
        
        score: Dict
            视角评分结果字典，由 ViewpointEvaluator.score_view() 填写。
            包含 S_kp, S_center, S_dist, C_move, Q 等评分字段。
            初始为空字典 {}。
    """
    candidate_id: int
    position: np.ndarray  # 三维坐标，shape=(3,)
    yaw: float             # 朝向角（弧度）
    geodesic_distance: float  # 测地距离（米）
    euclidean_distance_to_human: float  # 到人体的欧氏距离（米）
    is_valid: bool         # 是否有效
    invalid_reason: str = ""  # 无效原因（仅当 is_valid=False 时有意义）
    score: Dict = field(default_factory=dict)  # 评分结果


class CandidateSampler:
    """
    候选视角采样器 —— 围绕人体目标生成并过滤候选观察点。

    采样策略：
        以人体位置为中心，在配置指定的多个半径和角度上放置候选点，
        然后通过导航网格进行三步过滤：
        1. 导航性过滤：点必须在 navmesh 上
        2. 距离过滤：到人体的欧氏距离必须在 [min, max] 区间
        3. 路径过滤：从机器人起点到候选点的测地路径必须存在且距离不超过上限
    """

    def __init__(self, config: dict):
        """
        初始化采样器。

        参数：
            config: 配置字典，需要包含 "candidate_sampling" 字段：
                - radii: [float] —— 采样半径列表（米），如 [1.5, 2.0, 2.5]
                - angles_deg: [float] —— 采样角度列表（度），如 [-135, -90, ..., 180]
                - min_distance_to_human: float —— 候选点到人体的最小欧氏距离
                - max_distance_to_human: float —— 候选点到人体的最大欧氏距离
                - max_geodesic_distance: float —— 最大允许测地距离
        """
        self.config = config

    def sample(
        self,
        human_pos: np.ndarray,
        robot_pos: np.ndarray,
        runner,
    ) -> List[CandidateView]:
        """
        围绕人体目标采样候选视角，并过滤不可达点。

        参数：
            human_pos: 人体脚底中心位置，shape=(3,)。
                       这是采样中心点。
            robot_pos: 机器人起始位置，shape=(3,)。
                       用于计算测地距离。
            runner: HabitatRunner 实例。
                   用于导航性检查、测地距离计算等。

        返回：
            CandidateView 列表，同时包含有效和无效候选点。
            主脚本通过 c.is_valid 过滤有效候选点。

        采样数量：
            设 radii 有 R 个值，angles_deg 有 A 个值，
            则最多生成 R × A 个候选点。
            默认配置：3 个半径 × 8 个角度 = 24 个候选点。

        过滤步骤：
            1. 对每个 (r, θ) 计算理想位置
            2. snap_point —— 吸附到导航网格
            3. is_navigable —— 检查导航性
            4. 计算到人体的欧氏距离，检查是否在 [min, max] 范围
            5. geodesic_distance —— 计算测地距离，检查是否可达且不超过上限
            6. 通过 compute_look_at_yaw 计算朝向
        """
        sampling_cfg = self.config["candidate_sampling"]
        radii = sampling_cfg["radii"]               # 采样半径列表
        angles_deg = sampling_cfg["angles_deg"]     # 采样角度列表（度）
        min_dist = sampling_cfg["min_distance_to_human"]  # 最小欧氏距离
        max_dist = sampling_cfg["max_distance_to_human"]  # 最大欧氏距离
        max_geo = sampling_cfg["max_geodesic_distance"]   # 最大测地距离

        candidates: List[CandidateView] = []
        candidate_id = 0
        angles_rad = np.deg2rad(angles_deg)  # 角度转弧度

        # ---------- 双重循环：半径 × 角度 ----------
        for r in radii:
            for theta in angles_rad:
                # 1. 计算理想位置（在 X-Z 平面上绕人体旋转）
                cx = human_pos[0] + r * np.cos(theta)
                cz = human_pos[2] + r * np.sin(theta)
                cy = human_pos[1]  # Y 坐标保持与人体基座相同
                candidate = np.array([cx, cy, cz], dtype=np.float32)

                # 2. 吸附到导航网格
                snapped = runner.snap_point(candidate)

                # ---------- 过滤步骤 1：导航性检查 ----------
                if not runner.is_navigable(snapped):
                    candidates.append(CandidateView(
                        candidate_id=candidate_id,
                        position=snapped,
                        yaw=0.0,
                        geodesic_distance=float("inf"),
                        euclidean_distance_to_human=float("inf"),
                        is_valid=False,
                        invalid_reason="not_navigable",  # 点不在导航网格上
                    ))
                    candidate_id += 1
                    continue

                # 计算距离
                euclidean_dist = float(np.linalg.norm(snapped - human_pos))
                geo_dist = runner.geodesic_distance(robot_pos, snapped)

                # ---------- 过滤步骤 2：欧氏距离检查 ----------
                if euclidean_dist < min_dist or euclidean_dist > max_dist:
                    candidates.append(CandidateView(
                        candidate_id=candidate_id,
                        position=snapped,
                        yaw=0.0,
                        geodesic_distance=geo_dist,
                        euclidean_distance_to_human=euclidean_dist,
                        is_valid=False,
                        invalid_reason="out_of_distance_range",  # 距离超出范围
                    ))
                    candidate_id += 1
                    continue

                # ---------- 过滤步骤 3：测地距离检查 ----------
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

                # ---------- 计算朝向：看向人体目标 ----------
                yaw = compute_look_at_yaw(snapped, human_pos)

                # ---------- 所有过滤通过 → 有效候选点 ----------
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
