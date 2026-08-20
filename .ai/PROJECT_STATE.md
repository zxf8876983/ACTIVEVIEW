# ACTIVEVIEW Project State

Last Updated: 2026-08-21  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v9.0 — Action-conditioned Active View Scoring**（初版研发完成，建立动作感知主动视角选择算法基线）。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> v7 -> v8 -> v9`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v9.0 (9.0.0 — Action-conditioned Active View Scoring)
- **Title**: Action-conditioned Active View Scoring
- **Active Code Directory**: `ea_avs_mvp_v9/`
- **Active Specification**: `EA_AVS_MVP90_Code_Generation_Document.md`
- **Previous Specification**: `EA_AVS_MVP80_Code_Generation_Document.md` (v8.2 COMPLETED & FROZEN)
- **Inherited Assets & Infrastructure from v7 & v8**:
  - Habitat 室内仿真环境生命周期管理 (`ea_avs_mvp_v7.environment.habitat_env`, `ea_avs_mvp_v8.environment.env_adapter`)
  - KinematicHumanoid 人形模型加载与 16 关节 3D GT 提取 (`ea_avs_mvp_v7.human`)
  - 局部极坐标网格视点生成与三阶约束检查 (`ea_avs_mvp_v8.viewpoint`, `ea_avs_mvp_v8.constraints`)
  - 基础几何质量评价 Q_geom(v) (`ea_avs_mvp_v8.evaluation.view_quality`)
  - 移动机器人底盘与相机同步外参挂载 (`ea_avs_mvp_v8.robot`)

---

## 3. Latest Stable Implementations
- **v9.0 (`ea_avs_mvp_v9/`, ACTIVE / COMPLETED INITIAL RELEASE)**: 动作感知主动视角选择、身体分区特征提取与四大基线定量对比基线平台。
- **v8.2 (`ea_avs_mvp_v8/`, CLOSED / FINALIZED & FROZEN)**: 局部主动视点规划、三阶空间约束管道与透明科学质量评价基准平台。
- **v7.0 (`ea_avs_mvp_v7/`, CLOSED / FINALIZED)**: 拟人化室内主动感知仿真与动作数据集生成平台。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: 严格防泄漏 Estimated-State 物理光线追踪主动视角闭环。

---

## 4. v9.0 Research Objectives & Boundaries

### 核心假设与科研边界 (Core Assumptions & Scope):
*The optimal observation viewpoint is not only determined by geometry, but also depends on the current human activity state: $Q(v \mid A) \neq Q(v)$.*
1. **Human State & Action State Known**：假设人体位置与当前动作标签（如 `fall`, `sitting`, `standing`, `bending`, `reaching`）已由外部系统或上游模块提供；
2. **Action-conditioned Scoring**：在 v8 几何质量评价 $Q_{\text{geom}}(v)$ 基础上，引入动作关键解剖区域覆盖、推荐视角偏角区间与专属距离适配度，构建 $Q(v \mid a) = w_{\text{geom}} \cdot Q_{\text{geom}}(v) + w_{\text{act}} \cdot \Delta Q(a, v)$；
3. **Four Baseline Comparison**：统一横向对比 `Random`, `Nearest`, `Geometry Best (v8)`, `Action-conditioned (v9 Ours)`。

### 明确排除在 v9.0 之外的非目标 (Explicitly NOT in v9.0):
- ❌ 不训练 动作识别/分类神经网络 (No Action Recognition Training)
- ❌ 不实现 人体全局搜索与环境探索 (No Human Search / No Global Exploration)
- ❌ 不实现 深度强化学习策略网络 (No RL / Learned policy network)
- ❌ 不实现 全局路径规划与闭环避障控制 (Global Navigation)
- ❌ 不修改 `ea_avs_mvp_v7/`、`ea_avs_mvp_v8/` 及历史版本代码
