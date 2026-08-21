# ACTIVEVIEW Project State

Last Updated: 2026-08-22  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v10.0 Phase 1 — Habitat RGB-D Dataset Generation**（RGB-D 驱动的动作识别主动视角选择：阶段 1 仿真多视角 RGB-D 数据集生成底座构建）。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v9.0`: 任务先验规则主动视角打分基准（COMPLETED & RETAINED AS BASELINE）；
  - `v9.1`: 感知质量驱动主动视角选择与退化基准（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：处于版本化递进过渡期（`v1 -> ... -> v9 -> v10`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v10.0 (10.0.0 Phase 1 — Habitat RGB-D Dataset Generation)
- **Title**: RGB-D Driven Task-aware Active View Selection for Human Action Recognition
- **Active Code Directory**: `ea_avs_mvp_v10/`
- **Active Specifications**:
  - `ACTIVEVIEW_V10.0_Research_Development_Document.md`
  - `ACTIVEVIEW_V10.0_Code_Generation_Specification.md`
- **Inherited Assets & Infrastructure from v7, v8, v9**:
  - Habitat 室内仿真环境生命周期管理 (`ea_avs_mvp_v7.environment.habitat_env`, `ea_avs_mvp_v8.environment.env_adapter`)
  - KinematicHumanoid 人形模型加载与动作驱动 (`ea_avs_mvp_v7.human`)
  - MotionPlayer 动作回放控制器与标准 AMASS 动作库 (`ea_avs_mvp_v7.motion`)
  - 局部极坐标网格视点生成 (`ea_avs_mvp_v8.viewpoint`, `ea_avs_mvp_v10.viewpoint`)
  - 统一数据根目录：`/home/zxf/WorkSpace/code/data/ActiveView/`

---

## 3. Latest Stable Implementations
- **v10.0 Phase 1 (`ea_avs_mvp_v10/`, ACTIVE / COMPLETED & VERIFIED)**:
  - 针孔相机与 RGB-D 传感器数据采集器 (`RGBDCapture`)；
  - 6 大核心动作类别资产与标签管理器 (`MotionManager`: standing, walking, sitting, bending, reaching, falling)；
  - 多视角极坐标候选视点生成器 (`CandidateViewpointGeneratorV10`)；
  - 多模态样本构建器与数据集生成引擎 (`V10SampleBuilder`, `V10DatasetGenerator`)；
  - 标准演示与多模态可视化套件 (`examples/v10_phase1_demo/`)。
- **v9.1 (`ea_avs_mvp_v9/`, CLOSED / FINALIZED)**: 感知质量与信息增益驱动的主动视角规划基准。
- **v8.2 (`ea_avs_mvp_v8/`, CLOSED / FINALIZED)**: 局部主动视点规划与三阶空间硬约束基准。
- **v7.0 (`ea_avs_mvp_v7/`, CLOSED / FINALIZED)**: 拟人化室内主动感知仿真与动作数据集生成平台。
- **v6.0 (`ea_avs_mvp_v6/`, CLOSED / FINALIZED)**: Estimated-State 物理光线追踪主动视角闭环。

---

## 4. Phase 1 Scientific & Engineering Achievements
1. **真实多模态观测数据闭环**：成功在 Habitat 中针对 6 大动作类别生成同步 RGB (PNG)、高精度 Depth (NPY / Vis PNG)、相机外参/位姿 (JSON)、动作标签 (JSON) 与 GT 骨骼 (JSON, 仅供 Oracle 理论上限与离线评测)；
2. **多视角采样与元数据索引**：实现多半径、多方位的极坐标采样，并自动生成 `metadata/samples.json` 数据集索引清单；
3. **严格隔离与无泄漏**：Phase 1 仅提供数据生成底座，未提前实现 ST-GCN 或主动选点网络，GT 姿态不进入任何在线模型前向输入。
