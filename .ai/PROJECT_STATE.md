# ACTIVEVIEW Project State

Last Updated: 2026-08-19  
Active Branch: main  
Active Scientific Code Baseline: 02bc45e  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：ACTIVEVIEW v6.0 实现与系统验证已完成 (Implemented & Validated)。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> 未来版本`），各版本保留独立目录与设计文档。核心功能全链路跑通前暂不重构为单一 `src/`。

---

## 2. Active Development Version
- **Version**: v6.0 (6.0.0)
- **Active Development Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`
- **Implementation Status**: IMPLEMENTED & VALIDATED
- **Development Code Directory**: `ea_avs_mvp_v6/`

---

## 3. Latest Stable Implementation
- **Version**: v5.0 (5.0.0)
- **Stable Code Directory**: `ea_avs_mvp_v5/`
- **Stable Version Specification**: `EA_AVS_MVP50_Code_Generation_Document.md`
- **Active Scientific Code Baseline**: `02bc45e`

---

## 4. Current Development Goal (v6.0)
从 **GT-State Active Perception** 进入 **Estimated-State Active Perception**：
在 v5.0 真实 Humanoid RGB-D 仿真基础上，构建“当前视角 RGB-D -> 2D 人体姿态检测 -> 深度逆投影 3D Lifting -> 人体状态估计 (EstimatedHumanState: 位置、朝向、尺度、观测/缺失关节、Proxy 骨架) -> 估计状态候选视角采样与预测评分 -> EstimatedState-Ours 单步 NBV 决策”的完整视觉闭环，全程杜绝在线决策阶段对 Humanoid GT 的读取与信息泄漏，并系统评估估计误差对主动观察位姿选择质量的影响。

---

## 5. Version Evolution
- **v1.0 (`ea_avs_mvp/`)**: 几何版候选视角采样与可达性过滤，基于抽象骨架计算 FOV 可见性与移动代价，对比 Fixed/Random/Nearest/Ours。
- **v2.0 (`ea_avs_mvp_v2/`)**: 严格确立 `Pred`（决策前预测评分）与 `True`（到达后真实评估）信息边界，严禁选择阶段偷看未来图像，引入原地保留（stay）机制。
- **v3.0 (`ea_avs_mvp_v3/`)**: 引入多姿态骨架（standing, sitting, lying_fallen, bending）、人体朝向建模（`human_yaw`）与动作关键部位加权评分（`S_action_part`）。
- **v4.0 (`ea_avs_mvp_v4/`)**: 引入物理引擎 Ray Casting 实现环境障碍物遮挡检测，提出遮挡感知评分（`S_action_occ`）、消融策略体系与同口径 Oracle 离线上界。
- **v5.0 (`ea_avs_mvp_v5/`)**: 接入真实 Habitat KinematicHumanoid（`neutral_0`）与 RGB-D/语义渲染，由 URDF link 派生 15 关节 GT 骨架，实现 5 类遮挡判定与决策闭环（前一稳定基线）。
- **v6.0 (`ea_avs_mvp_v6/`, 已实现并验证)**: 构建当前 RGB-D 状态估计前端（2D Pose + Depth Lifting + Orientation/Position/Scale Estimator + Proxy Skeleton），驱动 Estimated-State NBV 选择并与 GT-State/Oracle 严格对照。

---

## 6. Current v6 Implemented Capabilities & Verified Results
- **2D Pose Backend 适配与集成**：成功接入 TorchVision Keypoint R-CNN 与离线 Mock 后端，提供 COCO-17 到 EA-AVS 15 的转换与颈部/骨盆中点派生。
- **Depth Lifting 与 3D 关节估计**：实现稳健局部 5x5 邻域深度采样与相机/世界坐标 3D 逆投影，平均关节 3D 估计误差 ~10cm-20cm。
- **人体状态估计器 (HumanStateEstimator)**：统一封装位置、朝向、尺度、可见/缺失关节划分及 Proxy 骨架补全，严格杜绝 GT 参数泄漏。
- **估计状态 NBV 路径闭环**：实现以估计位置为中心的候选点采样及基于估计状态的预测评分器，支持 stay 与失效安全兜底。
- **No-GT-Leakage 审计与测试**：通过 3/3 严格 AST 语法树检查与函数签名校验。
- **三支路评测与指标统计**：20 个完整 Episode 闭环评测已跑通，成功量化状态估计带来的 Estimation Gap 与 Oracle Gap。

---

## 7. Important Scientific Constraints
1. **One-shot Active Re-observation**：定位为单步主动观察位姿重选择，非多步强化学习。
2. **严禁 GT 信息泄漏到在线决策**：`EstimatedState-Ours` 在线决策路径禁止传入或读取 `gt_human_pos`, `gt_human_yaw`, `gt_skeleton`, `humanoid_manager`, `semantic_mask`。
3. **严禁未来观测泄漏**：策略决策只能使用预测评分（`Q_pred_est` / `Q_pred_gt`），选择前绝对禁止渲染候选点未来 RGB/Depth/Semantic。
4. **状态无效安全兜底**：状态估计失败时，`EstimatedState-Ours` 必须安全停留在当前位姿（`stay`），严禁静默回退到 GT 状态。
5. **三支路独立命名**：必须清晰区分 `EstimatedState-Ours`（主方法）、`GTState-Ours`（特权基线）、`Oracle-NBV`（离线上界）。
6. **主实验场景常量**：v6.0 评测主变量聚焦于状态估计引入的误差传播，动作场景统一设为 standing 常量。

---

## 8. Current Stable Point
- **Active Branch**: `main`
- **Active Scientific Code Baseline**: `02bc45e` - `refactor(v5.0): final closure - current-vs-candidate competition, invalid-occlusion validity, geometry cause counts`
- **Recent Key Scientific Commits**:
  - `1a4b0e5`: docs(v6.0): add estimated-state NBV development specification
  - `02bc45e`: v5.0 final closure (current-vs-candidate competition, invalid-occlusion validity, closure unit test)
  - `aab699a`: v5.0 closure fix (unknown validity, target-surface occlusion, semantic invisible, oracle same-aperture gap)
  - `46a615e`: v5.0 rigor fix round 3 (object IDs, 5-way self-occlusion, semantic mask, depth coverage)
  - `869653b`: v5.0 rigor fixes for humanoid visibility/yaw/GT
  - `76e0e7e`: Initial v5.0 implementation

---

## 9. Expected Next Phase
- **后续版本演进 (v7+)**：计划在 Estimated-State 基础上引入动作假设不确定性建模（Action Hypothesis & Uncertainty）、缺失证据恢复评分（Evidence Recovery Score）与下游动作识别网络增益验证。
- **工程重构计划**：在全链路科学闭环验证完成后，再统一进行单主线 `src/` 代码重构。

---

## 10. Historical Versions (Read-Only)
- `ea_avs_mvp/` (v1.0), `ea_avs_mvp_v2/` (v2.0), `ea_avs_mvp_v3/` (v3.0), `ea_avs_mvp_v4/` (v4.0), `ea_avs_mvp_v5/` (v5.0) 为历史及稳定参考版本，保持只读，不参与当前开发修改。
