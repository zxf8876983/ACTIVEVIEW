"""
EA-AVS-MVP v5.0 包初始化
=========================

中文名称：面向老人动作感知的移动机器人主动视角选择 —— v5.0 真实 Humanoid 版本

v5.0 相比 v4.0 的核心升级：
    1. Habitat 场景中真实存在可渲染、可控制、具有完整人体表面的 Humanoid
    2. 机器人 RGB-D 相机真实观察该人体（含环境遮挡与人体自遮挡）
    3. 保留 GT-State 支路：从 Humanoid 实际关节状态建立 GT skeleton，接入 v4.0 NBV
    4. 坚持 v4.0 的 pred/true 信息边界：策略选择阶段不渲染候选点未来图像

延续 v2/v3/v4 的重要约束：
    - 策略只能使用 pred_score（GT skeleton + 已知地图几何，禁止未来 RGB/depth）
    - true_score 只能在渲染后计算
    - Ours 必须允许选择 current_view（不移动）
    - Oracle 仅在评估阶段使用 depth 口径 true_score

本版本不是：动作识别、人体姿态估计、RL、ROS、多步 NBV。真实视觉前端（RGB-D → 2D
pose → estimated state）留待后续版本。
"""

__version__ = "5.0.0"