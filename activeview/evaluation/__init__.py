"""文件用途：提供统一的 ActiveView 评估接口。

主要输入：预测行、动作标签和 FULL/MOVING 协议。
主要输出：Accuracy、Macro-F1、regret 与 trajectory 摘要。
项目角色：独立评估层，被训练后诊断和最终 runner 复用。
"""
