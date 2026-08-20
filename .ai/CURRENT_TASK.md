# ACTIVEVIEW Current Task

Status: IN_PROGRESS

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (ACTIVEVIEW v7.0 implementation in progress)

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
   Humanoid Agent (KinematicHumanoid)
           ↓
   Habitat Indoor Scene (apartment_1.glb)
           ↓
   Robot RGB-D Observation (RGB + Depth + 3D Pose GT)
           ↓
   Episode Dataset
   ```
2. **已完成模块化架构重构 (`ea_avs_mvp_v7/`)**：
   - `configs/`: `habitat_config.yaml`, `humanoid_config.yaml`, `motion_config.yaml`, `sensor_config.yaml`
   - `core/`: `paths.py`, `episode.py`, `config.py` (统一数据根目录解析，支持 `ACTIVEVIEW_DATA_ROOT` / `../../data/ActiveView`)
   - `environment/`: `scene_manager.py`, `habitat_env.py` (Habitat 仿真环境生命周期封装，严禁 SLAM/导航规划)
   - `robot/`: `robot_agent.py`, `rgbd_sensor.py` (移动机器人底盘位姿与 RGB-D 相机内外参提取)
   - `human/`: `action_state.py`, `human_state.py`, `humanoid_agent.py` (Humanoid 实体加载与 16 关节 3D GT 提取)
   - `motion/`: `amass_loader.py`, `motion_converter.py`, `motion_player.py` (解耦 AMASS 数据读取、Schema A/B 标准化与 Habitat SMPL-X 转换回放)
   - `observation/`: `metadata.py`, `recorder.py` (RGB/Depth/JSON 落盘记录器)
   - `dataset/`: `episode_generator.py` (可复现的多动作多视角 Episode 生成器)
   - `evaluation/`: `basic_metrics.py` (数据完整性与真值覆盖率校验)
   - `scripts/`: `run_smoke_test.py`, `convert_motions.py`, `generate_dataset.py`
   - `tests/`: 纯 Python 单元测试套件 (`test_paths.py`, `test_motion.py`, `test_human.py`, `test_robot.py`, `test_episode.py`, `test_observation.py`)
3. **验证结果**：
   - Smoke Test (`run_smoke_test.py`): 100.0% RGB/Depth 有效率，平均 16 个 3D 关节真值，PASS。
   - 数据集批量生成 (`generate_dataset.py`): 成功生成多动作多视点 Episode 数据集及 catalog。
   - 纯 Python 单元测试: 16/16 全部 PASS。

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
- [x] v7.0 纯 Python 单元测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 16/16)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
- [x] AMASS 动作批量转换: `python -m ea_avs_mvp_v7.scripts.convert_motions --action fall_related` (PASS)
- [x] Habitat 端到端最小闭环 Smoke Test: `python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 5` (PASS, exit code 0)
- [x] Multi-action Episode 生成: `python -m ea_avs_mvp_v7.scripts.generate_dataset` (PASS)
