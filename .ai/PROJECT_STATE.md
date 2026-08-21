# ACTIVEVIEW Project State

Last Updated: 2026-08-22  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v9.1 — Perception-aware Active View Selection**（基于当前感知质量的人体主动视角选择，彻底解耦 GT 人体真值与动作标签输入，以信息增益 Gain 为核心学习目标，完成 5 大退化场景与 Oracle 理论上限实验闭环）。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v9.0`: 任务先验规则主动视角打分基准（COMPLETED & RETAINED AS BASELINE）。
- **架构组织**：处于版本化递进过渡期（`v1 -> v2 -> v3 -> v4 -> v5 -> v6 -> v7 -> v8 -> v9`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v9.1 (9.1.0 — Perception-aware Active View Selection)
- **Title**: Perception-aware Active View Selection
- **Active Code Directory**: `ea_avs_mvp_v9/`
- **Active Specification**: `EA_AVS_MVP90_Code_Generation_Document.md`
- **Inherited Assets & Infrastructure from v7 & v8**:
  - Habitat 室内仿真环境生命周期管理 (`ea_avs_mvp_v7.environment.habitat_env`, `ea_avs_mvp_v8.environment.env_adapter`)
  - KinematicHumanoid 人形模型加载 (`ea_avs_mvp_v7.human`)
  - 局部极坐标网格视点生成与三阶空间硬约束检查 (`ea_avs_mvp_v8.viewpoint`, `ea_avs_mvp_v8.constraints`)
  - 基础几何质量评价 Q_geom(v) (`ea_avs_mvp_v8.evaluation.view_quality`)
  - 移动机器人底盘与相机外参同步挂载 (`ea_avs_mvp_v8.robot`)

---

## 3. Latest Stable Implementations
- **v9.1 (`ea_avs_mvp_v9/`, ACTIVE / COMPLETED & VALIDATED)**:
  - 当前不完整观测状态编码器 (`ObservationEncoder`, 71d)；
  - 感知质量驱动的信息增益学习网络 (`PerceptionAwareViewScorer`, $\hat{G}(v \mid O_t)$)；
  - 统一感知数据抽象基类与视觉误差模拟器 (`BaseObservationProvider`, `ObservationSimulator`)；
  - Oracle 理论上限评测器 (`OracleViewEvaluator`)；
  - 5 大感知退化基准评测 (Scenario A~E) 与完整实验报告套件。
- **v9.0 (`ea_avs_mvp_v9/`, CLOSED BASELINE)**: 任务先验主动视角启发式规则打分基准。
- **v8.2 (`ea_avs_mvp_v8/`, CLOSED / FINALIZED & FROZEN)**: 局部主动视点规划、三阶空间约束管道与透明科学质量评价基准平台。
- **v7.0 (`ea_avs_mvp_v7/`, CLOSED / FINALIZED)**: 拟人化室内主动感知仿真与动作数据集生成平台。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: 严格防泄漏 Estimated-State 物理光线追踪主动视角闭环。

---

## 4. v9.1 Research Objectives & Scientific Results

### 核心科研结论 (Core Scientific Findings):
1. **基于当前感知质量的信息增益目标 $\hat{G}(v \mid O_t)$**：输入严格为 71 维当前不完整观测状态（估计坐标 + 关节置信度 + 7 大部位可见置信度），彻底移除了对 SMPL-X GT 姿态真值与 Action Label 的依赖；
2. **5 大感知退化场景基准实验 (Scenario A~E)**：实验证实**当前观测质量越差（自遮挡 $\rightarrow$ 家具遮挡 $\rightarrow$ 严重噪声 $\rightarrow$ 肢体缺失），模型主动选点获得的信息增益与质量提升越显著**；
3. **6 大方法对比与 Oracle 理论上限**：Perception-aware 选点在关节平均置信度改善与缺失关节恢复上全面优于 Random、Nearest、Geometry v8 与 Rule v9.0，达到 Oracle 上限的 89.8%；
4. **统一接口预留**：规范实现 `BaseObservationProvider`，为 v10+ 接入端到端 RGB 姿态估计器提供无缝迁移接口。
