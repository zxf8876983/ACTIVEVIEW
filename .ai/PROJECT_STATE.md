# ACTIVEVIEW Project State

Last Updated: 2026-08-21  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v9.0 — Action-conditioned Active View Scoring**（已完成全面收尾与科研实验闭环，建立动作感知可解释性基线）。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> v7 -> v8 -> v9`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v9.0 (9.0.0 — Action-conditioned Active View Scoring)
- **Title**: Action-conditioned Active View Scoring Baseline
- **Active Code Directory**: `ea_avs_mvp_v9/`
- **Active Specification**: `EA_AVS_MVP90_Code_Generation_Document.md`
- **Previous Specification**: `EA_AVS_MVP80_Code_Generation_Document.md` (v8.2 COMPLETED & FROZEN)
- **Inherited Assets & Infrastructure from v7 & v8**:
  - Habitat 室内仿真环境生命周期管理 (`ea_avs_mvp_v7.environment.habitat_env`, `ea_avs_mvp_v8.environment.env_adapter`)
  - KinematicHumanoid 人形模型加载与 16 关节 3D GT 提取 (`ea_avs_mvp_v7.human`)
  - 局部极坐标网格视点生成与三阶空间硬约束检查 (`ea_avs_mvp_v8.viewpoint`, `ea_avs_mvp_v8.constraints`)
  - 基础几何质量评价 Q_geom(v) (`ea_avs_mvp_v8.evaluation.view_quality`)
  - 移动机器人底盘与相机外参同步挂载 (`ea_avs_mvp_v8.robot`)

---

## 3. Latest Stable Implementations
- **v9.0 (`ea_avs_mvp_v9/`, ACTIVE / COMPLETED & VALIDATED BASELINE)**: 动作感知主动视角打分、解耦动作先验配置、身体分区特征提取、四大基线横向评测、多动作视角迁移实验与权重敏感性消融验证。
- **v8.2 (`ea_avs_mvp_v8/`, CLOSED / FINALIZED & FROZEN)**: 局部主动视点规划、三阶空间约束管道与透明科学质量评价基准平台。
- **v7.0 (`ea_avs_mvp_v7/`, CLOSED / FINALIZED)**: 拟人化室内主动感知仿真与动作数据集生成平台。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: 严格防泄漏 Estimated-State 物理光线追踪主动视角闭环。

---

## 4. v9.0 Research Objectives & Experimental Proof

### 核心科研结论与实验闭环 (Core Scientific Proof):
*实验严格证实科学假设：$Q(v \mid A) \neq Q(v)$，即动作状态驱动观察视角的自适应迁移。*
1. **Action Prior Decoupling**：动作先验（关键部位、偏好偏角、最优距离）完全解耦至 `configs/action_prior.yaml`，禁止在 Python 代码中硬编码动作规则；
2. **Multi-Action Comparison Experiment** (`scripts/run_action_comparison.py`)：在同一场景和人体位置下，`SITTING` 与 `BENDING` 动作的最优视点由纯正面（$1^\circ$）迁移至侧前方（$44^\circ$），实现视角得分提升；
3. **Weight Sensitivity Ablation** (`scripts/run_weight_ablation.py`)：评测 $(0.8/0.2, 0.6/0.4, 0.4/0.6, 0.2/0.8)$ 权重区间，证实随着 $w_{\text{act}}$ 提升，动作感知能够自适应扩展观测距离与调整观察方位。

### 明确排除在 v9.0 之外的非目标 (Explicitly NOT in v9.0):
- ❌ 不训练 动作识别/分类神经网络 (No Action Recognition Training)
- ❌ 不实现 人体全局搜索与环境探索 (No Human Search / No Global Exploration)
- ❌ 不实现 深度强化学习与端到端策略网络 (No RL / End-to-end policy network)
- ❌ 不实现 全局路径规划与闭环避障控制 (Global Navigation)
- ❌ 不修改 `ea_avs_mvp_v7/`、`ea_avs_mvp_v8/` 及历史版本代码
