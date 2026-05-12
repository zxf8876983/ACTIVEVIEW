"""
几何计算工具模块 —— geometry.py
=================================

功能：
    提供 MVP0.1 所需的所有几何计算函数，包括：
    - 角度归一化（将任意角度映射到 [-π, π] 范围）
    - 朝向角（yaw）与方向向量的相互转换
    - 计算从一个位置看向目标位置的 yaw 角
    - 判断三维点是否在相机视场角（FOV）内
    - 高斯形状评分函数

坐标系约定：
    - Y 轴向上（符合 Habitat-Sim 的坐标系）
    - yaw = 0 表示朝向 +Z 方向
    - yaw 为正表示绕 Y 轴逆时针旋转（朝向 +X 方向旋转）

本模块不依赖 Habitat API，可以独立测试和复用。
"""

import numpy as np


def normalize_angle(angle: float) -> float:
    """
    将任意角度归一化到 [-π, π] 范围内。

    参数：
        angle: 输入角度（弧度制），可以是任意大小。

    返回：
        归一化后的角度，范围在 [-π, π] 之间。

    数学原理：
        使用 arctan2(sin(θ), cos(θ)) 的特性来保证结果始终在 [-π, π] 内。
        这比取模运算更稳定，能正确处理接近 ±π 的边界情况。

    示例：
        >>> normalize_angle(3.0 * np.pi)  # 约等于 -3.1416
        -3.141592653589793
    """
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def yaw_to_forward(yaw: float) -> np.ndarray:
    """
    将 yaw 朝向角转换为三维前向向量。

    参数：
        yaw: 朝向角（弧度制）。

    返回：
        shape=(3,) 的 numpy 数组，表示前向单位向量，Y 分量始终为 0。

    坐标系约定：
        - yaw = 0 → 朝向 +Z 方向，前向向量为 (0, 0, 1)
        - yaw = π/2 → 朝向 +X 方向，前向向量为 (1, 0, 0)
        - 前向向量在 X-Z 平面上，Y 分量为 0（因为机器人在地面上运动）

    数学公式：
        forward = (sin(yaw), 0, cos(yaw))

    用途：
        将 Habitat 中 agent 的旋转角转换为方向向量，用于设置 agent 朝向。
    """
    return np.array([np.sin(yaw), 0.0, np.cos(yaw)])


def compute_look_at_yaw(source_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """
    计算从源位置看向目标位置所需的 yaw 朝向角。

    参数：
        source_pos: 源位置的三维坐标，shape=(3,)，如 (xₛ, yₛ, zₛ)。
        target_pos: 目标位置的三维坐标，shape=(3,)，如 (xₜ, yₜ, zₜ)。

    返回：
        yaw 角（弧度制），使得从 source 朝向 target 时正好"看向"目标。

    数学原理：
        忽略 Y 轴高度差，仅计算 X-Z 平面上的方向角：
            dx = xₜ - xₛ
            dz = zₜ - zₛ
            yaw = arctan2(dx, dz)

        使用 arctan2 而非 arctan 是因为 arctan2 能正确处理 dx=0 和四个象限的情况。

    用途：
        机器人需要始终"面向"人体目标，每个候选视角的 yaw 都用此函数计算。
    """
    dx = target_pos[0] - source_pos[0]
    dz = target_pos[2] - source_pos[2]
    return float(np.arctan2(dx, dz))


def angle_in_camera_fov(
    camera_base_pos: np.ndarray,
    camera_yaw: float,
    point: np.ndarray,
    hfov_deg: float,
    vfov_deg: float,
    camera_height: float,
    min_depth: float,
    max_depth: float,
) -> dict:
    """
    判断一个三维点是否在相机的视场角（FOV）范围内。

    这是 MVP0.1 中判断"关键点是否可见"的核心函数。
    判断条件同时考虑了水平角度、垂直角度和深度距离三个维度。

    参数：
        camera_base_pos: 机器人基座位置，shape=(3,)，即机器人在地面上的坐标。
        camera_yaw: 相机朝向角（弧度制）。
        point: 待判断的三维点坐标，shape=(3,)。
        hfov_deg: 水平视场角（度），如 90°。
        vfov_deg: 垂直视场角（度），如 60°。
        camera_height: 相机距离机器人基座的高度（米），如 1.2m。
        min_depth: 最小有效深度（米），太近无法对焦。
        max_depth: 最大有效深度（米），太远看不清。

    返回：
        字典，包含以下字段：
        - in_fov: bool，是否在视场角内
        - horizontal_angle: float，水平方向夹角（弧度），相对于相机中心方向
        - vertical_angle: float，垂直方向夹角（弧度），相对于相机中心方向
        - distance: float，点到相机的欧氏距离（米）
        - depth: float，同 distance，深度值

    实现步骤：
        1. 计算相机光心位置：camera_base_pos + (0, camera_height, 0)
        2. 计算从相机到目标点的向量
        3. 分解水平距离（X-Z 平面投影距离）和欧氏距离
        4. 计算水平夹角：相对于相机朝向的偏转角
        5. 计算垂直夹角：相对于水平面的俯仰角
        6. 综合判断：水平角 ≤ HFOV/2 且 垂直角 ≤ VFOV/2 且在深度范围内

    注意：
        当前版本（MVP0.1）只做 FOV 判断，不做遮挡判断。
        遮挡判断将在 MVP0.3 中通过 depth 或 ray casting 实现。
    """
    # ---------- 1. 计算相机光心位置 ----------
    # 相机安装在机器人顶部，高度为 camera_height
    camera_pos = camera_base_pos + np.array([0.0, camera_height, 0.0])

    # ---------- 2. 从相机指向目标点的向量 ----------
    vec = point - camera_pos

    # ---------- 3. 计算距离 ----------
    horizontal_dist = np.sqrt(vec[0]**2 + vec[2]**2)  # X-Z 平面距离
    distance = float(np.linalg.norm(vec))               # 欧氏距离

    # ---------- 4. 计算水平夹角 ----------
    # 点的水平朝向角
    point_yaw = float(np.arctan2(vec[0], vec[2]))
    # 相对于相机朝向的偏差角
    horizontal_angle = normalize_angle(point_yaw - camera_yaw)

    # ---------- 5. 计算垂直夹角 ----------
    # 使用 atan2(高度差, 水平距离) 得到俯仰角
    vertical_angle = float(np.arctan2(vec[1], horizontal_dist))

    # ---------- 6. 综合判断 ----------
    hfov_rad = np.deg2rad(hfov_deg / 2.0)  # 水平半视场角（弧度）
    vfov_rad = np.deg2rad(vfov_deg / 2.0)  # 垂直半视场角（弧度）

    in_h = abs(horizontal_angle) <= hfov_rad    # 水平方向在 FOV 内
    in_v = abs(vertical_angle) <= vfov_rad      # 垂直方向在 FOV 内
    in_d = min_depth <= distance <= max_depth   # 距离在有效范围内
    in_fov = bool(in_h and in_v and in_d)       # 三者同时满足才认为可见

    return {
        "in_fov": in_fov,
        "horizontal_angle": horizontal_angle,
        "vertical_angle": vertical_angle,
        "distance": distance,
        "depth": distance,  # MVP0.1 中 depth 等同于欧氏距离
    }


def gaussian_score(value: float, optimal: float, sigma: float) -> float:
    """
    计算高斯形状的评分，分值范围在 (0, 1] 之间。

    参数：
        value: 输入值（如相机到人体的实际距离）。
        optimal: 最优值（高斯分布的中心，如最佳观察距离 2.0 米）。
        sigma: 高斯分布的标准差，控制评分的衰减速度。

    返回：
        评分值，范围在 (0, 1] 之间。当 value == optimal 时达到最大值 1.0。

    数学公式：
        score = exp(-(value - optimal)² / (2 * σ²))

    用途：
        MVP0.1 中用此函数计算"距离得分" S_dist ——
        当机器人距离人体 2.0 米时得分最高，偏离越远得分越低。

    示例：
        >>> gaussian_score(2.0, 2.0, 0.7)  # 恰好在最优距离
        1.0
        >>> gaussian_score(3.0, 2.0, 0.7)  # 偏离 1 米
        0.361...
    """
    return float(np.exp(-((value - optimal) ** 2) / (2 * sigma ** 2)))
