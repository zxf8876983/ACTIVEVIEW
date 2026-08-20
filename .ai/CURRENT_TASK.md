# ACTIVEVIEW Current Task

Status: IN_PROGRESS

## Task
ACTIVEVIEW v7.0 Development Preparation

## Objective
根据 `EA_AVS_MVP70_Code_Generation_Document.md` 建立 Humanoid-driven Active Perception Simulation Environment 的开发准备：
1. 确立从动作数据到机器人观测的核心链路：
   ```text
   AMASS motion
         ↓
   Habitat humanoid
         ↓
   Indoor scene
         ↓
   Robot RGB-D observation
   ```
2. 明确 v7.0 作为基础设施与实验仿真平台的定位，为后续动作感知主动视角研究提供受控、逼真的评估基准；
3. 基于已下载校验的 17 条 AMASS 动作资产（`motion_asset_manifest.json`），准备接入 Habitat 官方 Humanoid 动作回放与室内场景仿真。

## Explicit Prohibitions & Non-Goals (明确禁止)
- ❌ **不实现 NBV 算法** (No NBV algorithm design)
- ❌ **不实现 Action-aware utility** (No Action-aware utility optimization)
- ❌ **不实现 Evidence Recovery** (No Evidence Recovery mechanism)
- ❌ **不实现 强化学习 (RL)** (No Reinforcement Learning)
- ❌ **不实现 多步规划 (Multi-step Planning)** (No sequential viewpoint planning)
- ❌ **不实现 动作识别模型训练 (HAR)** (No action recognition training)
- ❌ **不修改 v6.0 及历史版本代码** (Keep v1-v6 strictly read-only)
- ❌ **不将 AMASS / BABEL 原始大型数据提交至 Git**

## Current Version Status
- **Active Code Version**: v6.0 (6.0.0 — CLOSED / FINALIZED, Read-Only)
- **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`
- **Current Phase**: v7.0 Development Preparation
- **Motion Assets Status**: AMASS 5 个子库 7,862 个 `.npz` 文件已就绪，17 条 feasibility 动作索引校验完成 (17/17 PASS)

## Validation Plan
1. 上下文规范一致性检查：确认 `.ai/PROJECT_STATE.md`、`.ai/CURRENT_TASK.md` 与 `EA_AVS_MVP70_Code_Generation_Document.md` 范围完全一致；
2. 历史与工具套件回归测试：
   - `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (14 项 PASS)
   - `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (17 项 PASS)
   - `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (4 项 PASS)
