# ACTIVEVIEW Current Task

Status: IN_PROGRESS

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (v7.0 motion-humanoid-observation pipeline refinement completed)

## Objective
严格根据最新提交的 `EA_AVS_MVP70_Code_Generation_Document.md` 建立解耦、严谨、面向动作驱动的拟人化主动感知仿真环境：
1. **核心流水线落地**：
   ```text
   BABEL Action Annotation
           ↓
   AMASS Motion Asset
           ↓
   Motion Normalization (NormalizedMotion)
           ↓
   Humanoid Motion Conversion (SMPL-X PKL)
           ↓
   Habitat Humanoid Playback (KinematicHumanoid)
           ↓
   Robot RGB-D Observation (RGB + Depth + 3D Pose GT)
           ↓
   Episode Dataset
   ```
2. **已完成模块强化与完善 (`ea_avs_mvp_v7/`)**：
   - `motion/joint_mapping.py`: 显式规范 SMPL-X 54 关节层级、切片索引与四元数归一化校验器；
   - `motion/motion_converter.py`: 严格校验 162 维关节角输入与 216 维四元数输出；
   - `human/humanoid_agent.py`: 完善 `get_gt_joint_positions` 与 `get_joint_summary`，保证 16 关键点 3D 世界坐标输出；
   - `human/human_state.py`: 完整封装 `position`, `orientation_yaw_rad`, `joint_positions_3d_world`, `action_class`, `action_label`, `frame_id`, `timestamp`；
   - `dataset/episode_generator.py` & `observation/recorder.py`: 规范 `episode_xxx/` 目录结构 (`rgb/frame_000000.png`, `depth/frame_000000.npy`, `metadata.json`)；
   - `scripts/run_smoke_test.py`: 消除静默 fallback，严格执行 8 项验收标准并输出 `[v7.0 Acceptance Report]`；
   - `tests/`: 编写 `test_motion_loader.py`, `test_motion_converter.py`, `test_humanoid_state.py`, `test_episode.py`, `test_paths.py`, `test_robot.py`, `test_observation.py` (共 19 项单元测试全部 PASS)。

3. **验证结果**：
   - Smoke Test (`run_smoke_test.py`): 100.0% RGB/Depth 有效率，16 个 3D 关节真值，输出 `PASS: v7.0 Humanoid-driven Active Perception Environment Verified`。
   - 纯 Python 单元测试: 19/19 全部 PASS。
   - 历史版本回归测试: v6.0 (17/17 PASS), No-GT-Leakage (4/4 PASS), Motion Assets (14/14 PASS)。

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
- **Active Development Version**: v7.0 (7.0.0 — In Progress)
- **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`
- **Previous Stable Baseline**: v6.0 (6.0.0 — CLOSED / FINALIZED, Read-Only)

## Validation Plan Execution Results
- [x] v7.0 纯 Python 单元测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 19/19)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
- [x] AMASS 动作批量转换: `python -m ea_avs_mvp_v7.scripts.convert_motions --action fall_related` (PASS)
- [x] Habitat 端到端最小闭环 Smoke Test: `python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 5` (PASS, exit code 0)
