# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (v7.0 final validation completed)

## Objective
严格根据最新提交的 `EA_AVS_MVP70_Code_Generation_Document.md` 建立解耦、严谨、面向动作驱动的拟人化主动感知仿真环境，并完成科研实验闭环终期打磨与验收：
1. **科研实验基础链路闭环全量通过**：
   ```text
   AMASS Motion (Schema A/B)
           ↓
   Habitat Motion Format (MotionConverterSMPLX)
           ↓
   Humanoid 动作播放 (KinematicHumanoid neutral_0)
           ↓
   Robot RGB-D Observation (RGB + Depth + 3D Pose GT)
           ↓
   Episode Dataset (data/ActiveView/runs/ & visualizations/v7_demo/)
   ```
2. **已完成终期打磨清单 (`ea_avs_mvp_v7/`)**：
   - `human/action_metrics.py`: 多维动力学动作真实性评价模块 (`ActionMotionMetrics`)，可显著区分静态站立与动态跌倒；
   - `scripts/run_v7_demo.py`: 统一科研演示入口，分离 `robot_pose` 与 `camera_pose`，输出包含每帧 16 关键点 GT 与 `motion_metrics` 的标准化 `metadata.json`，并生成 `v7_demo.mp4`；
   - `scripts/create_video.py`: 独立 RGB 序列转 MP4 视频工具；
   - `scripts/validate_three_actions.py`: 批量验证 3 类典型动作 (`standing`, `sitting`, `fall_related`) 并生成 `validation_report.json`；
   - `tests/`: 组织为 `unit/`, `integration/`, `smoke/` 三层结构，扩展至 22 项纯 Python 单元/集成测试全量 PASS。

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
- **Active Development Version**: v7.0 (7.0.0 — COMPLETED / FINALIZED)
- **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`
- **Previous Stable Baseline**: v6.0 (6.0.0 — CLOSED / FINALIZED, Read-Only)

## Validation Plan Execution Results
- [x] v7.0 纯 Python 单元/集成/冒烟测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 22/22)
- [x] Motion Pipeline 独立验证: `python -m ea_avs_mvp_v7.scripts.verify_motion_pipeline --action fall_related` (PASS)
- [x] 统一演示入口与视频生成: `python -m ea_avs_mvp_v7.scripts.run_v7_demo --action fall_related --num-frames 15` (PASS)
- [x] 独立视频转换工具: `python -m ea_avs_mvp_v7.scripts.create_video` (PASS)
- [x] 3 类典型动作批量验证: `python -m ea_avs_mvp_v7.scripts.validate_three_actions` (PASS, standing/sitting/fall_related)
- [x] Habitat 端到端最小闭环 Smoke Test: `python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 10` (PASS)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
