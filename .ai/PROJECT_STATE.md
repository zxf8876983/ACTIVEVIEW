# ACTIVEVIEW Project State

Last Updated: 2026-08-20  
Active Branch: main  
Active Scientific Code Baseline: 56088e6  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：ACTIVEVIEW v6.0 第一轮 Scientific Rigor Fix 已完成并全量验证 (Implemented & Scientifically Validated)。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> 未来版本`），各版本保留独立目录与设计文档。核心功能全链路跑通前暂不重构为单一 `src/`。

---

## 2. Active Development Version
- **Version**: v6.0 (6.0.0)
- **Active Development Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`
- **Implementation Status**: IMPLEMENTED & RIGOROUSLY VALIDATED
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
- **v6.0 (`ea_avs_mvp_v6/`, 已实现并通过科研严谨性修复)**: 构建当前 RGB-D 状态估计前端，完成 3D 双侧解剖推导、GT-free 几何遮挡检测、解耦候选池 (Est-Pool / GT-Pool / Shared-Pool) 以及同口径 Oracle 评测体系。

---

## 6. Current v6 Implemented Capabilities & Verified Results
- **GT-free 几何光路遮挡检测**：实现 `cast_ray_to_estimated_point`，支持 `0.12m` 容差带与 3 态分类（`clear`, `estimated_geometric_blocked`, `unknown`），严格过滤 `valid=False` 关键点进入 `visible_occ`。
- **Yaw 坐标与前向向量统一**：对齐 Habitat KinematicHumanoid URDF 原生坐标系与项目全局 $+Z$ 约定，8 方向校准实测中位数绝对误差降至 **0.97°**，平均绝对误差 **3.03°**。
- **3D Bilateral 中点解剖推导**：移除 neck / pelvis 在 2D midpoint 像素上的深度采样，直接由 3D 左右肩与左右髋中点几何推导。
- **尺度感知与 6 级基座位置估计**：尺度估计前置，基座位置优先级为双踝 -> 单踝 -> 骨盆 -> 髋中点 -> 躯干均值 -> 可见关节中位数，统一读取 `POSE_SKELETONS["standing"]`。
- **解耦候选池与三协议评估**：
  - 协议 A (Shared-Pool 离线分析): 决策一致率 50%，$Q_{true}$ Gap 0.070。
  - 协议 B (Candidate Shift 分析): 平均采样中心偏移 0.417m。
  - 协议 C (端到端系统级主实验): `EstimatedState-Ours` ($Q_{true}=0.488$) 对比 `GTState-Ours` ($Q_{true}=0.561$)，同口径 Oracle Gap 分别为 0.031 (EstPool) 与 0.023 (GTPool)。
- **三重零 GT 泄漏防御**：通过 Guard A (AST 语法树禁词扫描)、Guard B (Runner Humanoid 访问阻断)、Guard C (未来候选观测运行时 Sentinel 拦截)。

---

## 7. Important Scientific Constraints
1. **One-shot Active Re-observation**：定位为单步主动观察位姿重选择，非多步强化学习。
2. **严禁 GT 信息泄漏到在线决策**：`EstimatedState-Ours` 在线决策路径禁止传入或读取 `gt_human_pos`, `gt_human_yaw`, `gt_skeleton`, `humanoid_manager`, `semantic_mask`。
3. **严禁未来观测泄漏**：策略决策只能使用预测评分（`Q_pred_est` / `Q_pred_gt`），选择前绝对禁止渲染候选点未来 RGB/Depth/Semantic。
4. **状态无效安全兜底**：状态估计失败时，`EstimatedState-Ours` 必须安全停留在当前位姿（`stay`），严禁静默回退到 GT 状态。
5. **三支路独立命名**：必须清晰区分 `EstimatedState-Ours`（主方法）、`GTState-Ours`（特权基线）、`Oracle-GTPool` / `Oracle-EstPool`（离线上界）。
6. **主实验场景常量**：v6.0 评测主变量聚焦于状态估计引入的误差传播，动作场景统一设为 standing 常量。

---

## 8. Current Stable Point
- **Active Branch**: `main`
- **Active Scientific Code Baseline**: `02bc45e` - `refactor(v5.0): final closure - current-vs-candidate competition, invalid-occlusion validity, geometry cause counts`
- **Recent Key Scientific Commits**:
  - `56088e6`: feat(v6.0): implement estimated-state active view selection architecture and validation suite
  - `1a4b0e5`: docs(v6.0): add estimated-state NBV development specification
  - `02bc45e`: v5.0 final closure (current-vs-candidate competition, invalid-occlusion validity, closure unit test)

---

## 9. Expected Next Phase
- **后续版本演进 (v7+)**：计划在 Estimated-State 基础上引入动作假设不确定性建模（Action Hypothesis & Uncertainty）、缺失证据恢复评分（Evidence Recovery Score）与下游动作识别网络增益验证。
- **工程重构计划**：在全链路科学闭环验证完成后，再统一进行单主线 `src/` 代码重构。

---

## 10. Historical Versions (Read-Only)
- `ea_avs_mvp/` (v1.0), `ea_avs_mvp_v2/` (v2.0), `ea_avs_mvp_v3/` (v3.0), `ea_avs_mvp_v4/` (v4.0), `ea_avs_mvp_v5/` (v5.0) 为历史及稳定参考版本，保持只读，不参与当前开发修改。
