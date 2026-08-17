# EA-AVS-MVP v4.0 代码生成指导文档

## 0. 文档用途

本文档用于指导大模型在当前 GitHub 仓库中生成 `ea_avs_mvp_v4` 版本代码。

v4.0 不推翻 v3.0，而是在 v3.0 已实现的以下能力基础上继续演进：

1. 多姿态人体骨架：`standing / sitting / lying_fallen / bending`；
2. 人体朝向建模：`human_yaw`；
3. 动作关键部位评分：`S_action_part_pred`；
4. 人体相对观察方向评分：`S_orient_pred`；
5. v2.0 延续下来的 `pred / true` 严格分离；
6. Ours 只能使用 `Q_pred`，不能偷看候选点未来图像。

v4.0 的核心新增能力是：

> **让机器人不仅知道“关键点是否在相机视野中”，还要判断“关键点是否真的会被墙、桌子、沙发、柜子等环境物体遮挡”。**

因此