# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (v7.0: AMASS-Humanoid-Habitat-RGBD pipeline verification completed)

## Objective
严格根据最新提交的 `EA_AVS_MVP70_Code_Generation_Document.md` 建立解耦、严谨、面向动作驱动的拟人化主动感知仿真环境：
1. **核心流水线落地与闭环验证**：
   ```text
   BABEL Action Annotation
           ↓
   AMASS Motion Asset
           ↓
   Motion Normalization (NormalizedMotion)
           ↓
   Habitat Motion Conversion (SMPL-X PKL)
           ↓
   KinematicHumanoid (neutral_0)
           ↓
   Habitat Indoor Scene (apartment_1.glb)
           ↓
   Robot RGB-D Observation (RGB + Depth + 3D Pose GT)
           ↓
   Episode Dataset (runs/episode_xxx)
   ```
2. **已完成模块强化与独立验证工具 (`ea_avs_mvp_v7/`)**：
   - `motion/joint_mapping.py`: 规范 SMPL-X 54 关节映射、162 维 Rodrigues $\rightarrow$ 216 维四元数转换、四元数合法性校验器；
   - `scripts/verify_motion_pipeline.py`: 独立验证 AMASS `.npz` $\rightarrow$ Habitat `.pkl` 转换链路 (`PASS: v7 Motion Pipeline Verified`)；
   - `scripts/verify_episode_output.py`: 校验 `rgb/`, `depth/`, `metadata.json` 数据结构完整性 (`PASS: Episode Structure and Metadata Verified`)；
   - `scripts/generate_visual_demo.py`: 生成摔倒动作动态感知可视化产物 (`PASS: Visual Demonstration Generated`)；
   - `scripts/run_smoke_test.py`: 10 帧 `fall_related` 端到端最小闭环测试 (`PASS: ACTIVEVIEW v7.0 Humanoid Pipeline Verified`)；
   - `tests/`: 纯 Python 单元测试扩展至 25/25 全部 PASS。

## Explicit Prohibitions & Non-Goals (严格遵守)
- ❌ 不实现 NBV 视角选择算法
- ❌ 不实现 Action-aware utility
- ❌ 不实现 Evidence Recovery
- ❌ 不实现 强化学习 (RL)
- ❌ 不实现 多步规划 (Multi-step Planning)
- ❌ 不训练 动作识别模型 (HAR)
- ❌ 不修改 v6.0 及历史版本代码
- ❌ 不将 AMASS / BABEL 原始大型数据提交至 Git

## Current Version Status
- **Active Development Version**: v7.0 (7.0.0 — COMPLETED)
- **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`
- **Previous Stable Baseline**: v6.0 (6.0.0 — CLOSED / FINALIZED, Read-Only)

## Validation Plan Execution Results
- [x] v7.0 纯 Python 单元测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 25/25)
- [x] Motion Pipeline 独立验证: `python -m ea_avs_mvp_v7.scripts.verify_motion_pipeline --action fall_related` (PASS)
- [x] Episode 输出结构与 Metadata 校验: `python -m ea_avs_mvp_v7.scripts.verify_episode_output` (PASS)
- [x] 动态感知可视化 Demo 生成: `python -m ea_avs_mvp_v7.scripts.generate_visual_demo --action fall_related --num-frames 10` (PASS, 0.495m dynamic drop)
- [x] Habitat 端到端最小闭环 Smoke Test: `python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 10` (PASS, exit code 0)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
