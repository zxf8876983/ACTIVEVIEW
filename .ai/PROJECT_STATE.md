# ACTIVEVIEW Project State

Last Updated: 2026-08-22  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v9.1 — Human-state-aware Learnable Active View Selection**（重构完成并完成闭环验证，建立人体物理姿态驱动的主动视角学习打分网络）。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v9.0`: 动作感知主动视角规则打分基准（COMPLETED & RETAINED AS BASELINE）。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> v7 -> v8 -> v9`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v9.1 (9.1.0 — Human-state-aware Learnable Active View Selection)
- **Title**: Human-state-aware Learnable Active View Selection
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
- **v9.1 (`ea_avs_mvp_v9/`, ACTIVE / COMPLETED & VALIDATED)**: 人体状态感知学习型视点打分网络 (`Q(v | H)`)、7 大身体关键部位可见性特征提取、Pairwise Ranking 训练管道、5 大基线对比及空间遮挡鲁棒性实验。
- **v9.0 (`ea_avs_mvp_v9/`, CLOSED BASELINE)**: 动作感知主动视角启发式规则打分基线。
- **v8.2 (`ea_avs_mvp_v8/`, CLOSED / FINALIZED & FROZEN)**: 局部主动视点规划、三阶空间约束管道与透明科学质量评价基准平台。
- **v7.0 (`ea_avs_mvp_v7/`, CLOSED / FINALIZED)**: 拟人化室内主动感知仿真与动作数据集生成平台。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: 严格防泄漏 Estimated-State 物理光线追踪主动视角闭环。

---

## 4. v9.1 Research Objectives & Scientific Results

### 核心科研定位与结论 (Core Scientific Results):
1. **纯人体状态感知打分 $\hat{Q}(v \mid H)$**：网络输入仅包含人体 16 骨骼关键点相对坐标及偏航朝向角 (49d) 与 13d 视点几何及 7 大身体关键解剖部位可见性特征，**完全排除 Action Label 作为模型输入**；
2. **7 大身体解剖部位可见性特征**：显式计算并输出 `head`, `torso`, `pelvis`, `left_hand`, `right_hand`, `left_leg`, `right_leg` 的视锥投影与遮挡覆盖率；
3. **训练与基线验证**：Pairwise Ranking Loss 训练下验证集 Top-1 选点准确率达 **95.0%**，目标效用保持率达 **100.0%**；
4. **空间遮挡鲁棒性**：在不同初始站位和遮挡条件下，模型自适应规避被遮挡方向，选择全局与解剖部位可见性最大化的视点。

### 明确排除在 v9.1 之外的非目标 (Explicitly NOT in v9.1):
- ❌ 动作标签作为模型输入 (No Action Label in Model Input)
- ❌ 训练动作分类/识别神经网络 (No Action Recognition Training)
- ❌ 估计器/姿态检测器 (GT state used in simulation)
- ❌ 人体全局搜索与环境探索 (No Human Search / No Global Exploration)
- ❌ 深度强化学习与端到端机器人策略网络 (No RL / End-to-end policy network)
- ❌ 全局路径规划与闭环避障控制 (Global Navigation)
- ❌ 修改 `ea_avs_mvp_v7/`、`ea_avs_mvp_v8/` 及历史版本代码
