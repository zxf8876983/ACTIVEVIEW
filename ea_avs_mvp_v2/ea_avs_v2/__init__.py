"""
EA-AVS-MVP v2.0 包初始化
=========================

中文名称：面向老人动作感知的移动机器人主动视角选择 —— v2.0 规范化版本

v2.0 核心改进：
    v0.1 中"预测评分"和"真实评估"没有严格区分。
    v2.0 严格分离：
        - PredictiveEvaluator: 移动前只能用几何 FOV 预测评分（不可渲染图像）
        - TrueEvaluator: 策略选定并渲染后才能进行真实评估
        - OursPolicy: 只能使用 pred_score，必须允许"不移动"

依赖：
    - habitat-sim >= 0.3.0
    - numpy
    - PyYAML
    - Pillow
"""

__version__ = "2.0.0"
