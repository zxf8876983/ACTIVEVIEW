# ACTIVEVIEW Project State

Last Updated: 2026-08-24  
Active Branch: main  
Target Audience: Coding Agents (Codex, DeepSeek, Claude Code, Gemini) & Researchers

> **Runtime Git State Note**: 当前 Repository HEAD 属于运行时 Git 状态，不在本文件中静态保存。代码模型若需获取当前真实仓库状态，请直接查询 Git (`git rev-parse --short HEAD` / `git status --short`)。

---

## 1. Current Stage
- **项目阶段**：**v11.5 (`ea_avs_mvp_v11/`) — Perception-aware Active View Benchmark Reconstruction**
  - **v11.5 (Perception-aware Active View Benchmark Reconstruction)**: `COMPLETED & VALIDATED` (端到端真实光学感知驱动、AMASS 实例级互斥划分、Studio 纯净感知训练与冻结 ST-GCN、HM3D 真实室内场景 5 大策略闭环评测、分级姿态质量分析与严格防泄漏代码审查已闭环)；
  - **v11.4 / v11.4.2 (Real Optical RGB-D Sensor Pipeline & VideoPose3D Lifting)**: `COMPLETED & MERGED` (验证 Keypoint R-CNN + VideoPose3D 3D 提升链路、消除绿地漂浮并实现 100% 严密物理射线贴地、合并至 `ea_avs_mvp_v11/` 统一版本)；
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
  - `V11.4.2_REAL_HABITAT_SENSOR_PIPELINE_REPORT.md`

- **Core Capabilities in `ea_avs_mvp_v11/`**:
  - `dataset/dataset_split.py`: AMASS 实例级分层抽样拆分（Train 70%, Val 15%, Test 15%）；
  - `dataset/clean_dataset_generator.py`: Studio 纯净环境光学估计骨架数据集生成；
  - `action_recognition/train_st_gcn_v11_5.py`: 基于估计骨架训练并冻结 ST-GCN 分类网络；
  - `active_view/habitat_active_view_benchmark.py`: Habitat HM3D 场景下 5 大主动视角策略闭环基准评测；
  - `evaluation/pose_quality_analysis.py`: 视点遮挡/偏角 $\to$ 关节点置信度 $\to$ 动作不确定度与准确率因果分层分析；
  - `perception/pose3d_estimator.py`: 统一 Keypoint R-CNN + VideoPose3D 3D HPE 估计器与工厂；
  - `perception/skeleton_normalizer.py`: 骨盆根节点去中心化、躯干尺度归一化与标准正向参考系对齐。

---

## 3. Physical & Runtime Data Boundary
- 遵循统一外部数据路径机制，大型数据集、权重与评测产物输出至外部数据根目录：
  `../../data/ActiveView/` (可通过环境变量 `ACTIVEVIEW_DATA_ROOT` 显式覆盖)。
