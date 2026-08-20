# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (v7.0 humanoid coordinate alignment fixed)

## Objective
严格落实 Habitat 坐标系规范 (x: 左右, y: 高度 >= 0.0m, z: 前后)，完成空间坐标对齐、地面高度校验与正面对准：
1. **空间坐标与对准修复**：
   - 严禁负值 Y 人体高度设置，默认统一为地面高度基准 `Y = 0.0m`；
   - 调试与演示默认对准：
     - Humanoid: `[0.0, 0.0, 0.0]`, `yaw = 180.0 deg` (面向 +Z 轴)
     - Robot: `[0.0, 0.0, 2.5]`, `yaw = 0.0 deg` (面向 -Z 轴，正对 Humanoid 前方)
   - 增加 `[V7 Spatial Debug]` 标准空间日志输出；
   - 新增 `scripts/debug_humanoid_pose.py` 生成 `visualizations/humanoid_debug/rgb.png`；
   - 在 `HumanoidAgent` 中增加负高度预警 (`pos[1] < -0.5m`)。
2. **完整闭环全量通过**：
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
- [x] 独立空间姿态与坐标调试: `python -m ea_avs_mvp_v7.scripts.debug_humanoid_pose` (PASS)
- [x] Humanoid 可见性独立调试: `python -m ea_avs_mvp_v7.scripts.debug_humanoid_visibility` (PASS)
- [x] 极简 Humanoid 独立渲染 Demo: `python -m ea_avs_mvp_v7.scripts.render_humanoid_only_demo` (PASS)
- [x] 统一演示入口与视频生成: `python -m ea_avs_mvp_v7.scripts.run_v7_demo --action fall_related --num-frames 10` (PASS)
- [x] 3 类动作全量闭环: `python -m ea_avs_mvp_v7.scripts.validate_three_actions` (PASS)
- [x] Habitat 端到端最小闭环 Smoke Test: `python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 10` (PASS)
- [x] v7.0 纯 Python 单元/集成/冒烟测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 22/22)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
