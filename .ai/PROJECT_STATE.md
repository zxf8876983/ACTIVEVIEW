# ACTIVEVIEW Project State

Last Updated: 2026-08-22  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v11.0 Phase 2 (v11.2, `ea_avs_mvp_v11/`) — Viewpoint Quality Dataset Generation**
  - **v10.0 (Perception & Action Recognition Foundation)**: `FROZEN` (只读基准，统一 RGB 驱动 3D 姿态估计、ST-GCN 动作分类与 Shannon 熵不确定度量化体系全面固化)；
  - **v11.1 (Candidate Generation & Filtering)**: `COMPLETED & CLOSED` (实现人体中心极坐标 32 候选点生成、Face Human Yaw 朝向计算、Habitat 3阶段过滤器与自包含闭环架构)；
  - **v11.2 (Viewpoint Quality Dataset Generation)**: `COMPLETED & CLOSED` (实现 `ViewpointDatasetGenerator`，批量生成 300 动作实例 $\times$ 28 视点 = 8,400 视点质量样本，严格实例互斥划分 70% Train / 15% Val / 15% Test，生成角度-距离熵热力图可视化与 176/176 单元测试回归)；
  - **v11.3 (Utility Predictor)**: `READY TO START` (待用户明确指令后进入)；
  - **v11.4 (Active Selection Policy)**: `PLANNED`。
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v8.0` ~ `v8.2`: 局部主动视角规划与几何质量基准（COMPLETED / FINALIZED & FROZEN，保持只读）；
  - `v9.0`: 任务先验规则主动视角打分基准（COMPLETED & RETAINED AS BASELINE）；
  - `v9.1`: 感知质量驱动主动视角选择与退化基准（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：处于版本化递进过渡期（`v1 -> ... -> v9 -> v10`），各版本保留独立目录与设计文档。

---

## 2. Active Development Version
- **Version**: v10.0 (10.0.0 Phase 3 — Action Recognition & Uncertainty)
- **Title**: RGB-D Driven Task-aware Active View Selection for Human Action Recognition
- **Active Code Directory**: `ea_avs_mvp_v10/`
- **Active Specifications**:
  - `ACTIVEVIEW_V10.0_Research_Development_Document.md`
  - `ACTIVEVIEW_V10.0_Code_Generation_Specification.md`
  - `PHASE3_ACTION_RECOGNITION_REPORT.md`
- **Inherited Assets & Infrastructure from v7, v8, v9**:
  - Habitat 室内仿真环境生命周期管理 (`ea_avs_mvp_v7.environment.habitat_env`, `ea_avs_mvp_v8.environment.env_adapter`)
  - KinematicHumanoid 人形模型加载与动作驱动 (`ea_avs_mvp_v7.human`)
  - MotionPlayer 动作回放控制器与标准 AMASS 动作库 (`ea_avs_mvp_v7.motion`)
  - 局部极坐标网格视点生成与约束过滤 (`ea_avs_mvp_v8.viewpoint`, `ea_avs_mvp_v8.constraints`)
  - 统一数据根目录：`/home/zxf/WorkSpace/code/data/ActiveView/`

---

## 3. Latest Stable Implementations
- **v10.0 Phase 3 (`ea_avs_mvp_v10/action_recognition/`, `tools/dataset_generation/`, ACTIVE / CLOSED)**:
  - 统一 3D 姿态估计器规范与 MediaPipe / Mock 实现 (`ea_avs_mvp_v10/perception/pose3d_estimator.py`)；
  - 动态时空图拓扑与 33 关节空间划分邻接矩阵 (`ea_avs_mvp_v10/action_recognition/graph.py`)；
  - 9 层时空图卷积残差动作分类网络 (`ea_avs_mvp_v10/action_recognition/st_gcn_model.py`)；
  - 动作分类与 Shannon 熵 / 归一化不确定度 / 决策余量计算引擎 (`ea_avs_mvp_v10/action_recognition/action_classifier.py`)；
  - PyTorch 动作数据集加载器与训练引擎 (`action_dataset.py`, `trainer.py`)；
  - Clean Perception (Oracle: 87.50% Acc, 0.6048 Ent) vs Habitat Perception (46.88% Acc, 0.9200 Ent) 双基准评测 (`evaluate_benchmarks.py`)；
  - 视点不确定度定量分析（验证了遮挡视角熵激增与正面视角高确信度，为 Phase 4 Active View 奠定实验依据）。
- **v10.0 Phase 2.1 (`ea_avs_mvp_v10/evaluation/`, `ea_avs_mvp_v10/perception/`, FROZEN)**:
  - 统一骨架拓扑规范 (`configs/skeleton_definition.json`) 与 3-tier 几何一致性评价指标 (Absolute/Relative MPJPE, PCK@5cm/10cm/15cm, PA-MPJPE)；
  - 10 样本多视角动作科研验证集 (`ea_avs_mvp_v10/examples/v10_phase2_validation/`) 与主报告 (`PHASE2_FINAL_VALIDATION_REPORT.md`)。
- **v10.0 Phase 1 (`ea_avs_mvp_v10/dataset/`, FROZEN)**:
  - 真实 RGB-D 多视角数据集采集底座与 6 大动作资产规范化。
- **v9.1 / v8.2 / v7.0 (FROZEN)**: 历史版本全部测试通过并保持只读。

---

## 4. Current Work State
- **当前分支**: `main`
- **当前正在执行的任务**: Phase 3 实施完成，全部 118 单元测试通过，等待用户指示下一步 (Phase 4)。
- **下一步行动建议**:
  1. 保持 Phase 1 ~ Phase 3 代码冻结；
  2. 启动 Phase 4: Active View Selection Policy (基于 ST-GCN 信息熵不确定度与几何可见性融合的主动视角规划算法)。
