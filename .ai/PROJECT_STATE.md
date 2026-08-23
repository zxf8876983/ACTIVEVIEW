# ACTIVEVIEW Project State

Last Updated: 2026-08-22  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v11.5 (`ea_avs_mvp_v11/`) — Perception-aware Active View Benchmark Reconstruction**
  - **v11.5 (Perception-aware Active View Benchmark Reconstruction)**: `COMPLETED & VALIDATED` (端到端真实光学感知驱动、AMASS 实例级互斥划分、Studio 纯净感知训练与冻结 ST-GCN、HM3D 真实室内场景 5 大策略闭环评测与严格防泄漏代码审查已闭环)；
  - **v11.3 (Utility Predictor & Multi-Scene Distribution Upgrade)**: `COMPLETED & CLOSED` (多场景多位置数据集升级，重新训练 Utility Predictor MLP)；
  - **v10.0 (Perception & Action Recognition Foundation)**: `FROZEN` (只读基准，统一 RGB 驱动 3D 姿态估计、ST-GCN 动作分类与 Shannon 熵不确定度量化体系全面固化)；
- **历史基线**：
  - `v1.0` ~ `v6.0`: Active View 几何与物理遮挡评测探索（CLOSED / FINALIZED，保持只读）；
  - `v7.0` ~ `v9.1`: 拟人化主动感知仿真环境与动作数据集平台（COMPLETED / FINALIZED & FROZEN，保持只读）。
- **架构组织**：独立版本目录架构（`ea_avs_mvp_v11/`）。

---

## 2. Active Development Version
- **Version**: v11.5 (Perception-aware Active View Benchmark)
- **Title**: Perception-aware Active View Benchmark Reconstruction
- **Active Code Directory**: `ea_avs_mvp_v11/`
- **Active Specifications & Reports**:
  - `V11.5_PERCEPTION_AWARE_ACTIVE_VIEW_REPORT.md`

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
