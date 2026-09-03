"""文件用途：暴露离线 episode 与 utility label 生成工具。

主要输入：BABEL 动作记录、Habitat 离线观测和冻结识别结果。
主要输出：策略 episode 与 utility supervision 文件。
项目角色：负责数据生成，不负责数据划分或模型训练。
"""
