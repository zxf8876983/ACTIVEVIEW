# ACTIVEVIEW Project State

Last Updated: 2026-08-20  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：EA-AVS-MVP v6.0 最终封口修复已完成，已正式封版定稿 (EA-AVS-MVP v6.0 — CLOSED / FINALIZED)。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> 未来版本`），各版本保留独立目录与设计文档。核心功能全链路跑通前暂不重构为单一 `src/`。

---

## 2. Active Development Version
- **Version**: v6.0 (6.0.0 — CLOSED / FINALIZED)
- **Active Development Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`
- **Implementation Status**: CLOSED / FINALIZED & RIGOROUSLY VALIDATED
- **Development Code Directory**: `ea_avs_mvp_v6/`

---

## 3. Latest Stable Implementation
- **Version**: v6.0 (6.0.0)
- **Code Directory**: `ea_avs_mvp_v6/`
- **Specification Document**: `EA_AVS_MVP60_Code_Generation_Document.md`
- **Previous Stable Baseline**: v5.0 (`ea_avs_mvp_v5/`)

---

## 4. Scientific Definition & Architecture (v6.0)
ACTIVEVIEW v6.0 构建了由当前 RGB-D 驱动的 Estimated-State 单步主动观察位姿选择框架。
在线 `EstimatedState-Ours` 仅使用视觉估计人体状态、导航信息和严格的静态场景几何；当静态场景射线能力不可用时采用 fail-closed 策略（`valid=False / unknown`），不允许静默退化到 full-collision raycast，因此绝不会使用真实 Humanoid articulated-body 碰撞几何。GT-State 和同候选池 Oracle 仅作为特权基线与离线评估参考。

---

## 5. Version Evolution
- **v1.0 (`ea_avs_mvp/`)**: 几何版候选视角采样与可达性过滤，基于抽象骨架计算 FOV 可见性与移动代价，对比 Fixed/Random/Nearest/Ours。
- **v2.0 (`ea_avs_mvp_v2/`)**: 严格确立 `Pred`（决策前预测评分）与 `True`（到达后真实评估）信息边界，严禁选择阶段偷看未来图像，引入原地保留（stay）机制。
- **v3.0 (`ea_avs_mvp_v3/`)**: 引入多姿态骨架（standing, sitting, lying_fallen, bending）、人体朝向建模（`human_yaw`）与动作关键部位加权评分（`S_action_part`）。
- **v4.0 (`ea_avs_mvp_v4/`)**: 引入物理引擎 Ray Casting 实现环境障碍物遮挡检测，提出遮挡感知评分（`S_action_occ`）、消融策略体系与同口径 Oracle 离线上界。
- **v5.0 (`ea_avs_mvp_v5/`)**: 接入真实 Habitat KinematicHumanoid（`neutral_0`）与 RGB-D/语义渲染，由 URDF link 派生 15 关节 GT 骨架，实现 5 类遮挡判定与决策闭环。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: 
  1. Static-Scene-Only 射线检测彻底消除对真实 Humanoid 铰接碰撞体的隐式 GT 泄漏；
  2. Estimated-State 移除 full-collision 降级，缺失 static API 时严格 fail-closed (`valid=False / unknown`)；
  3. `resolve_stage_id` 禁止猜测默认值，缺失 stage_id 时严格 fail-fast；
  4. AST Guard A2 静态严禁 `cast_ray_to_estimated_point` 出现 `runner.cast_ray` 调用；
  5. 修复 Shared-Pool GT 支路打分一致性，确保在同一 GT-centered 几何空间严谨对比 GT vs Est 评分；
  6. 建立 Oracle Candidate-Pool Identity Guard，跨池返回 `pool_mismatch`，上界破坏返回 `oracle_not_upper_bound`，杜绝 `max(0)` 掩盖错误；
  7. 恢复 `strict_gt_skeleton=True` 特权基线与评测约束；
  8. 建立 17 项纯 Python 严谨性单元测试套件与 4 项零 GT 防护测试。

---

## 6. Current v6 Implemented Capabilities & Verified Results
- **Fail-Closed Static-Scene-Only 射线检测**：实现 `HabitatRunner.cast_ray_static_scene` 与 `resolve_stage_id`，仅以 `stage_id` 为遮挡物，自动穿透/忽略 Humanoid 动态碰撞体；当 static API 缺失时严格返回 `valid=False/unknown`，绝不回退至 `cast_ray`。
- **Yaw 坐标与前向向量统一**：对齐 Habitat KinematicHumanoid URDF 原生坐标系与项目全局 $+Z$ 约定，8 方向校准实测中位数绝对误差降至 **0.97°**，平均绝对误差 **3.03°**，输出 PASS exit code 0。
- **3D Bilateral 中点解剖推导**：移除 neck / pelvis 在 2D midpoint 像素上的深度采样，直接由 3D 左右肩与左右髋中点几何推导。
- **尺度感知与 6 级基座位置估计**：尺度估计前置，基座位置优先级为双踝 -> 单踝 -> 骨盆 -> 髋中点 -> 躯干均值 -> 可见关节中位数，统一读取 `POSE_SKELETONS["standing"]`。
- **解耦候选池与三协议评估**（20 episodes 实测）：
  - 协议 A (Shared-Pool 离线分析): 决策一致率 45%，位置平均偏差 1.267m，$Q_{true}$ Gap 0.087。
  - 协议 B (Candidate Shift 分析): 平均采样中心偏移 0.322m (XZ 偏移 0.289m)。
  - 协议 C (端到端系统级主实验): `EstimatedState-Ours` ($Q_{true}=0.467$) 对比 `GTState-Ours` ($Q_{true}=0.553$)，端到端 Gap 0.086；同口径 Oracle Gap 分别为 0.052 (EstPool) 与 0.027 (GTPool)。
  - Oracle 跨池冲突与上界破坏计数均为 0 (`oracle_pool_mismatch_count: 0`, `oracle_not_upper_bound_count: 0`)。
- **四重零 GT 泄漏防御**：通过 Guard A1 (AST 语法树禁词扫描)、Guard A2 (AST 禁调 generic cast_ray)、Guard B (Runner Humanoid 访问阻断)、Guard C (未来候选观测运行时 Sentinel 拦截)。

---

## 7. Important Scientific Constraints
1. **One-shot Active Re-observation**：定位为单步主动观察位姿重选择，非多步强化学习。
2. **严禁 GT 信息泄漏到在线决策**：`EstimatedState-Ours` 在线决策路径禁止传入或读取 `gt_human_pos`, `gt_human_yaw`, `gt_skeleton`, `humanoid_manager`, `semantic_mask`。
3. **严禁未来观测泄漏**：策略决策只能使用预测评分（`Q_pred_est` / `Q_pred_gt`），选择前绝对禁止渲染候选点未来 RGB/Depth/Semantic。
4. **状态无效安全兜底**：状态估计失败或 static raycast 能力缺失时，`EstimatedState-Ours` 必须安全停留在当前位姿（`stay`），严禁静默回退到 GT 状态或 full collision raycast。
5. **三支路独立命名**：必须清晰区分 `EstimatedState-Ours`（主方法）、`GTState-Ours`（特权基线）、`Oracle-GTPool` / `Oracle-EstPool`（离线上界）。
6. **主实验场景常量**：v6.0 评测主变量聚焦于状态估计引入的误差传播，动作场景统一设为 standing 常量。

---

## 8. Current Stable Point
- **Active Branch**: `main`
- **Status**: EA-AVS-MVP v6.0 — CLOSED / FINALIZED

---

## 9. Expected Next Phase
- **后续版本演进 (v7+)**：计划在 Estimated-State 基础上引入动作假设不确定性建模（Action Hypothesis & Uncertainty）、缺失证据恢复评分（Evidence Recovery Score）与下游动作识别网络增益验证。
- **工程重构计划**：在全链路科学闭环验证完成后，再统一进行单主线 `src/` 代码重构。

---

## 10. Historical Versions (Read-Only)
- `ea_avs_mvp/` (v1.0), `ea_avs_mvp_v2/` (v2.0), `ea_avs_mvp_v3/` (v3.0), `ea_avs_mvp_v4/` (v4.0), `ea_avs_mvp_v5/` (v5.0) 为历史及稳定参考版本，保持只读，不参与当前开发修改。
