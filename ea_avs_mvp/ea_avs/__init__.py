"""
EA-AVS-MVP0.1: Elderly Action Active View Selection - Minimal Viable Prototype
================================================================================

中文名称：面向老人动作感知的移动机器人主动视角选择 —— 最小可运行版本

本包实现了在 Habitat 室内场景中，给定一个抽象人体骨架目标和移动机器人初始视角，
围绕人体目标采样候选观察点，并选择一个使人体关键点更完整可见的观察视角的核心闭环。

项目阶段：
    MVP0.1 —— 只做几何骨架目标的主动视角选择，不做强化学习、不做动作识别、
              不接真实 humanoid、不接 ROS/Unity。

依赖：
    - habitat-sim >= 0.3.0
    - numpy
    - PyYAML
    - Pillow
"""
