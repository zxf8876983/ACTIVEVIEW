# ACTIVEVIEW Current Task

Status: IN_PROGRESS

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (v7.0 development started)

## Objective
根据 `EA_AVS_MVP70_Code_Generation_Document.md` 建立面向动作驱动的拟人化主动感知仿真环境：
1. **核心流水线落地**：
   ```text
   AMASS / BABEL Motion
           ↓
   Humanoid Motion Conversion (SMPL-X PKL)
           ↓
   Habitat Indoor Scene (apartment_1.glb)
           ↓
   Robot Camera Observation (RGB-D)
           ↓
   RGB-D Dataset Generation
   ```
2. **已完成第一阶段最小闭环验证 (`run_humanoid_sim_smoke.py`)**：
   - 成功加载 Habitat 室内场景 (`apartment_1.glb`)
   - 成功加载 KinematicHumanoid (`neutral_0`)
   - 成功转换并加载 AMASS fall-related 跌倒动作 (`fall_related_3522.pkl`)
   - 成功回放动作并在仿真环境中驱动 Humanoid
   - 移动机器人相机成功采集 RGB-D 观测并提取 16 个核心关节 3D Ground-Truth
   - 观测数据已完整保存至外部数据目录 (`../../data/ActiveView/runs/v70_smoke_test/`)。
3. **完成 17 条 Feasibility 动作资产全量转换**：覆盖 standing, sitting, bending, reaching, fall_related 5 类老人动作。

## Code Structure (ea_avs_mvp_v7/)
```text
ea_avs_mvp_v7/
├── configs/
│   └── v70_humanoid_sim.yaml
├── humanoid/
│   ├── __init__.py
│   ├── humanoid_loader.py
│   └── humanoid_agent.py
├── motion/
│   ├── __init__.py
│   ├── motion_converter.py
│   └── motion_player.py
├── simulation/
│   ├── __init__.py
│   ├── scene_loader.py
│   ├── robot_sensor.py
│   └── observation_generator.py
├── scripts/
│   ├── convert_amass_motions.py
│   ├── run_humanoid_sim_smoke.py
│   └── generate_observation_dataset.py
└── tests/
    ├── __init__.py
    └── test_v70_pure_python.py
```

## Explicit Prohibitions & Non-Goals (严格遵守)
- ❌ 不实现 NBV 算法
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
- [x] v7.0 纯 Python 单元测试: `python3 -m unittest ea_avs_mvp_v7.tests.test_v70_pure_python` (PASS, 5/5)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
- [x] AMASS 动作批量转换: `python -m ea_avs_mvp_v7.scripts.convert_amass_motions` (PASS, 17/17 converted)
- [x] Habitat 端到端最小闭环 Smoke Test: `python -m ea_avs_mvp_v7.scripts.run_humanoid_sim_smoke` (PASS, exit code 0)
