# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (Humanoid Floor & Motion Grounding Completed)

## Objective
完成 Habitat 室内场景、KinematicHumanoid 人形模型与 AMASS 动捕动作的全链路空间几何对齐与地面贴合：
1. **场景物理高程标定与对齐**：
   - 明确 `apartment_1.glb` 物理地面标高 $Y_{\text{floor}} \approx -1.60\text{m}$（天花板 $Y_{\text{ceiling}} \approx +1.14\text{m}$），严禁盲目假设 $Y=0.0\text{m}$ 为地面；
   - 规范人体与机器人基座对齐（开阔客厅）：
     - Humanoid: `[1.5, -1.60, 4.0]`, `yaw = 0.0 deg`
     - Robot: `[1.5, -1.60, 6.8]`, `yaw = 0.0 deg`（面向 -Z 轴，相机高度 $-1.60 + 1.20 = -0.40\text{m}$）
2. **AMASS 动态位移与骨盆基座相对补偿机制**：
   - 解决 AMASS 全局绝对原点与 Habitat `KinematicHumanoid` 自带 $+0.9\text{m}$ 变换的叠加问题；
   - 在 `apply_motion_frame` 中实现相对位移补偿：
     $$\text{mat}[:3, 3] = [T_x(t) - T_x(0), T_y(t) - 0.90, T_z(t) - T_z(0)]$$
   - 确保 `standing` 笔直站立、`sitting` 正常坐下、`fall_related` 完整摔倒触地（高度下落达 $0.523\text{m}$，全身落地 $Y=-1.44\text{m}$）。
3. **完整闭环全量通过**：
   ```text
   AMASS Motion (Schema A/B)
           ↓
   Habitat Motion Format (MotionConverterSMPLX)
           ↓
   Humanoid 动作播放 (KinematicHumanoid neutral_0 with relative offset)
           ↓
   Robot RGB-D Observation (RGB + Depth + 3D Pose GT)
           ↓
   Episode Dataset (data/ActiveView/runs/ & visualizations/v7_demo/)
   ```

## Key Technical Findings & Gotchas (核心技术要点记录)
1. **场景坐标系原点 $\neq$ 地面**：
   - 3D 资产（如 Habitat test scenes / Gibson / Matterport3D）的原点通常为建模坐标系原点。在 `apartment_1.glb` 中，$Y=0.0\text{m}$ 处于距离物理地面 $1.60\text{m}$ 的半空；若硬编码 $Y=0.0\text{m}$，人体将悬挂在天花板附近。
2. **KinematicHumanoid 与 AMASS 变换叠加**：
   - `KinematicHumanoid` 的 `base_transformation` 内部已经包含 $+0.90\text{m}$ 的腰骨盆高度偏置；
   - AMASS 的平移数据初始帧通常也是 $\approx 0.90\text{m}$；
   - 若直接传入 AMASS 绝对平移，$0.9 + 0.9 = 1.8\text{m}$ 会导致严重上浮；若直接全部清零平移，则会导致下蹲、坐下、摔倒等垂向下沉动作失效（腰部被死锁在站立高度）；
   - **正确做法**：采用相对位移补偿 $T_y(t) - 0.90$，保留动作内部真实的动态下沉与位移。

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
- [x] Humanoid 地面着地独立验证: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.debug_humanoid_grounding` (PASS)
- [x] 独立空间姿态与坐标调试: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.debug_humanoid_pose` (PASS)
- [x] Humanoid 可见性独立调试: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.debug_humanoid_visibility` (PASS)
- [x] 极简 Humanoid 独立渲染 Demo: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.render_humanoid_only_demo` (PASS)
- [x] 统一演示入口与视频生成: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.run_v7_demo --action fall_related --num-frames 15` (PASS)
- [x] 3 类动作全量闭环: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.validate_three_actions` (PASS)
- [x] Habitat 端到端最小闭环 Smoke Test: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 10` (PASS)
- [x] v7.0 纯 Python 单元/集成/冒烟测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 22/22)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
