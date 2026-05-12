"""
视角质量评估模块 —— evaluator.py
==================================

功能：
    对某个候选视角计算骨架关键点可见性和最终视角质量分数。

评分函数（核心）：
    Q(v) = 0.60 × S_kp + 0.20 × S_center + 0.20 × S_dist - 0.20 × C_move

    其中：
    - S_kp:     关键点可见率（Keypoint Visibility Score），0~1
    - S_center: 居中度评分（Centering Score），0~1
    - S_dist:   距离评分（Distance Score），0~1
    - C_move:   运动代价（Movement Cost），0~1，从总分中扣除

评分流程：
    1. 对骨架每个关键点调用 angle_in_camera_fov 判断可见性
    2. 按身体分组（torso/lower_body/head）计算各组可见率
    3. 加权计算 S_kp
    4. 计算所有关键点的平均水平偏角得到 S_center
    5. 用高斯函数计算距离评分 S_dist
    6. 归一化测地距离得到运动代价 C_move
    7. 加权组合得到最终评分 Q
"""

from typing import Dict, List
import numpy as np

from .geometry import angle_in_camera_fov, gaussian_score, normalize_angle
from .skeleton import KEYPOINT_GROUPS


class ViewpointEvaluator:
    """
    视角质量评估器 —— 计算某个视角下人体骨架的可见性和质量评分。

    评分函数设计说明：
        - Q 值范围通常在 [-0.2, 1.0] 之间（因为 C_move 可能使总分为负）
        - S_kp 权重最大（0.6），强调关键点可见性是首要目标
        - C_move 作为惩罚项（0.2），鼓励选择移动代价小的视角
        - 所有子分数都在 [0, 1] 范围内，保证可比性
    """

    def __init__(self, config: dict):
        """
        初始化评估器。

        参数：
            config: 配置字典，需要包含以下字段：
                - camera: 相机参数（hfov_deg, vfov_deg, camera_height）
                - visibility: 可见性参数（min_depth, max_depth）
                - score_weights: 身体部位权重（torso, lower_body, head）
                - distance_score: 距离评分参数（optimal_distance, sigma）
                - view_score: 最终评分权重（w_kp, w_center, w_dist, w_move）
                - candidate_sampling: 需要 max_geodesic_distance
        """
        self.config = config
        self.camera_cfg = config["camera"]
        self.vis_cfg = config["visibility"]
        self.score_weights = config["score_weights"]
        self.dist_cfg = config["distance_score"]
        self.view_score_cfg = config["view_score"]

    def score_view(
        self,
        view_pos: np.ndarray,
        view_yaw: float,
        robot_start_pos: np.ndarray,
        human_base_pos: np.ndarray,
        human_skeleton: Dict[str, np.ndarray],
        geodesic_distance: float,
    ) -> dict:
        """
        对指定视角计算完整的评分。

        参数：
            view_pos: 候选视角的机器人位置，shape=(3,)。
            view_yaw: 候选视角的机器人朝向（弧度制）。
            robot_start_pos: 机器人起始位置，shape=(3,)。
                             用于计算 C_move（但传入的是已计算的 geodesic_distance）。
            human_base_pos: 人体基座位置，shape=(3,)。
                            用于计算 S_dist（到人体的距离）。
            human_skeleton: 人体骨架字典，键为关键点名称，值为三维坐标。
            geodesic_distance: 从起始位置到候选视角的测地距离（米）。
                               对于初始视角（current_view），此值为 0.0。

        返回：
            字典，包含以下字段：
            - S_kp: float —— 加权关键点可见率
            - S_center: float —— 居中度评分
            - S_dist: float —— 距离评分
            - C_move: float —— 运动代价
            - Q: float —— 最终综合评分
            - torso_visibility: float —— 躯干可见率（0~1）
            - lower_body_visibility: float —— 下肢可见率（0~1）
            - head_visibility: float —— 头部可见率（0~1）
            - visible_keypoints: list[str] —— 可见关键点名称列表
            - invisible_keypoints: list[str] —— 不可见关键点名称列表
        """
        # =====================================================================
        # Step 1：判断每个关键点是否在相机 FOV 内
        # =====================================================================
        visible_kps = []      # 可见关键点列表
        invisible_kps = []    # 不可见关键点列表
        horizontal_angles = []  # 所有关键点的水平偏角（用于 S_center）

        for name, pos in human_skeleton.items():
            result = angle_in_camera_fov(
                camera_base_pos=view_pos,
                camera_yaw=view_yaw,
                point=pos,
                hfov_deg=self.camera_cfg["hfov_deg"],
                vfov_deg=self.camera_cfg["vfov_deg"],
                camera_height=self.camera_cfg["camera_height"],
                min_depth=self.vis_cfg["min_depth"],
                max_depth=self.vis_cfg["max_depth"],
            )
            if result["in_fov"]:
                visible_kps.append(name)
            else:
                invisible_kps.append(name)
            horizontal_angles.append(result["horizontal_angle"])

        # =====================================================================
        # Step 2：计算分组可见率
        # =====================================================================
        def group_vis(group_name: str) -> float:
            """计算指定身体分组的可见率。
            
            公式：可见率 = 组内可见关键点数量 / 组内关键点总数
            
            例如 torso 组有 6 个关键点，5 个可见，则可见率为 5/6 ≈ 0.833
            """
            group_kps = KEYPOINT_GROUPS[group_name]
            if len(group_kps) == 0:
                return 0.0
            visible_count = sum(1 for kp in group_kps if kp in visible_kps)
            return visible_count / len(group_kps)

        torso_vis = group_vis("torso")               # 躯干可见率
        lower_body_vis = group_vis("lower_body")     # 下肢可见率
        head_vis = group_vis("head")                 # 头部可见率

        # =====================================================================
        # Step 3：计算加权关键点可见率 S_kp
        # =====================================================================
        # S_kp = 0.4 × V_torso + 0.4 × V_lower + 0.2 × V_head
        # 躯干和下肢各占 0.4（最重要），头部占 0.2
        S_kp = (
            self.score_weights["torso"] * torso_vis
            + self.score_weights["lower_body"] * lower_body_vis
            + self.score_weights["head"] * head_vis
        )

        # =====================================================================
        # Step 4：计算居中度评分 S_center
        # =====================================================================
        # 原理：所有关键点的平均水平偏角越小，说明人体越居中
        # S_center = 1 - clamp(mean(|θ|) / (HFOV/2), 0, 1)
        # 当人体完美居中时 θ≈0，S_center≈1.0
        # 当人体偏到 FOV 边缘时 θ≈HFOV/2，S_center≈0.0
        hfov_half_rad = np.deg2rad(self.camera_cfg["hfov_deg"] / 2.0)
        if len(horizontal_angles) > 0:
            mean_abs_angle = float(np.mean(np.abs(horizontal_angles)))
            center_error = mean_abs_angle / hfov_half_rad
            S_center = 1.0 - min(max(center_error, 0.0), 1.0)
        else:
            S_center = 0.0  # 没有关键点时居中度评分为 0

        # =====================================================================
        # Step 5：计算距离评分 S_dist
        # =====================================================================
        # 使用高斯评分：最优距离 2.0 米，sigma=0.7
        # 距离越接近 2.0 米，S_dist 越高
        dist_to_human = float(np.linalg.norm(view_pos - human_base_pos))
        S_dist = gaussian_score(
            dist_to_human,
            self.dist_cfg["optimal_distance"],  # 默认 2.0 米
            self.dist_cfg["sigma"],             # 默认 0.7
        )

        # =====================================================================
        # Step 6：计算运动代价 C_move
        # =====================================================================
        # C_move = clamp(geodesic_distance / max_geodesic_distance, 0, 1)
        # 如果 geodesic_distance = 0（初始视角），C_move = 0
        # 如果 geodesic_distance = max_geo（上限），C_move = 1
        max_geo = self.config["candidate_sampling"]["max_geodesic_distance"]
        if geodesic_distance == 0.0:
            C_move = 0.0
        else:
            C_move = min(max(geodesic_distance / max_geo, 0.0), 1.0)

        # =====================================================================
        # Step 7：计算最终评分 Q
        # =====================================================================
        # Q = 0.60 × S_kp + 0.20 × S_center + 0.20 × S_dist - 0.20 × C_move
        #
        # 注意 C_move 是惩罚项（用减号），其他三项是奖励项（用加号）
        # 设计理念：好的视角应该能看到更多关键点、人体在画面中央、
        #           距离适中、且移动代价不要太大
        ws = self.view_score_cfg
        Q = (
            ws["w_kp"] * S_kp           # +0.60 × 关键点可见率
            + ws["w_center"] * S_center # +0.20 × 居中度
            + ws["w_dist"] * S_dist     # +0.20 × 距离评分
            - ws["w_move"] * C_move     # -0.20 × 运动代价
        )

        return {
            "S_kp": S_kp,
            "S_center": S_center,
            "S_dist": S_dist,
            "C_move": C_move,
            "Q": Q,
            "torso_visibility": torso_vis,
            "lower_body_visibility": lower_body_vis,
            "head_visibility": head_vis,
            "visible_keypoints": visible_kps,
            "invisible_keypoints": invisible_kps,
        }
