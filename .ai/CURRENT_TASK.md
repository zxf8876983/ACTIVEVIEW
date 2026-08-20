# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (v7.0 Humanoid visibility debugging completed)

## Objective
定位并修复 v7.0 中 Humanoid 实体在 Habitat 场景中的可见性与相机对准问题，完成端到端清晰渲染验收：
1. **Humanoid 渲染可见性与场景对准修复**：
   - 场景地面与开阔空间坐标修正：`apartment_1.glb` 室内地面高度 $Y \approx -1.60\text{ m}$，将基座定位在开阔无遮挡区域 (`[1.5, -1.60, 4.0]`) 与相机视锥中心 (`[1.5, -1.60, 6.8]`, yaw=0.0 deg)；
   - 强化 `HumanoidAgent.load()` 物理与渲染实体注册断言，增加 `set_visibility(True)` 控制；
   - 在 `RGBDSensor` 中提供 `check_object_in_view` 视锥检测；
   - 增加独立调试脚本 `scripts/debug_humanoid_visibility.py` 与极简渲染验证脚本 `scripts/render_humanoid_only_demo.py`。
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
- [x] Humanoid 可见性独立调试: `python -m ea_avs_mvp_v7.scripts.debug_humanoid_visibility` (PASS)
- [x] 极简 Humanoid 独立渲染 Demo: `python -m ea_avs_mvp_v7.scripts.render_humanoid_only_demo` (PASS, 84.8% non-black)
- [x] 统一演示入口与视频生成: `python -m ea_avs_mvp_v7.scripts.run_v7_demo --action fall_related --num-frames 10` (PASS)
- [x] 3 类动作全量闭环: `python -m ea_avs_mvp_v7.scripts.validate_three_actions` (PASS)
- [x] Habitat 端到端最小闭环 Smoke Test: `python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 10` (PASS)
- [x] v7.0 纯 Python 单元/集成/冒烟测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 22/22)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
