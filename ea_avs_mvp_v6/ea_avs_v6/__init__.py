"""
EA-AVS-MVP v6.0 包初始化
=========================

中文名称：面向老人动作感知的移动机器人主动视角选择 —— v6.0 估计状态版本

v6.0 相比 v5.0 的核心升级：
    1. Current RGB-D -> 2D Pose -> Depth Lifting -> EstimatedHumanState
    2. 估计状态驱动的局部候选视角采样与预测评分（零 GT 泄漏）
    3. 纯视觉前端与 Proxy 骨架构建（template completion）
    4. 三支路系统对照：EstimatedState-Ours（主方法）、GTState-Ours（特权基线）、Oracle-NBV（离线上界）

核心科研约束：
    - 在线决策阶段（EstimatedState-Ours）严禁读取任何 Humanoid GT（位置/朝向/骨架/语义分割）
    - 严禁在选择候选点前渲染未来观测（RGB/Depth/Semantic）
    - 状态估计失效时安全停在原地（stay），严禁静默回退 GT
"""

__version__ = "6.0.0"
