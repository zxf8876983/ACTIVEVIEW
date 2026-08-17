"""
EA-AVS-MVP v4.0 包初始化
=========================

中文名称：面向老人动作感知的移动机器人主动视角选择 —— v4.0 遮挡感知版本

v4.0 相比 v3.0 的核心升级：
    1. Habitat 场景几何 ray casting（需开启 physics）
    2. 基于射线的关键点环境遮挡判断：visible = in_fov AND (NOT occluded)
    3. 遮挡感知动作关键部位评分 S_action_occ_pred 参与 Q_pred
    4. 模块化消融策略与 Oracle 离线上界

延续 v2.0/v3.0 的重要约束：
    - 选择阶段禁止使用候选点未来 RGB / depth / Q_true
    - 已知地图静态几何 ray casting 允许在 pred 阶段使用
    - 策略只能使用 pred_score
    - true_score 只能在渲染后计算
    - Ours 必须允许选择 current_view（不移动）
"""

__version__ = "4.0.0"