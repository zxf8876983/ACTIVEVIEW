# ACTIVEVIEW Current Task

Status: COMPLETED (v7.0 finalization completed)

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (v7.0 Finalized & Frozen)

## Objective
完成 ACTIVEVIEW v7.0 版本的最终收尾与工程冻结（v7.0 Finalization & Freezing）：
1. **统一 v7 Demo 配置**：
   - 固化 `configs/v7_demo.yaml`，统一管理 `scene`, `human`, `robot`, `camera`, `simulation` 全部参数；
   - `run_v7_demo.py` 不再包含大量硬编码参数，所有实验参数通过 YAML 读取并支持 CLI 参数覆盖。
2. **固化 Humanoid 参数配置**：
   - 确认 `ground_offset`, `scale`, `rotation`, `avatar_name`, `floor_height`, `base_pelvis_offset` 统一在 `configs/humanoid.yaml` 中配置管理；
   - 杜绝散落在代码中的魔法数值。
3. **完善 v7 主入口与辅助工具**：
   - `run_v7_demo.py` 作为主入口，一键完成场景加载、实体创建、动作播放、RGB-D 采集、Episode 落盘与视频生成；
   - 保留规范辅助工具：`verify_v7_pipeline.py`, `validate_three_actions.py`, `create_video.py`, `convert_motions.py`, `generate_dataset.py`, `run_smoke_test.py`。
4. **Human Placement 接口规范化**：
   - `human/human_spawn.py` 提供 `sample_human_position()`，为未来 v8.0 预留接口，严禁包含任何主动选点、碰撞优化或自动导航算法。
5. **完善 Episode 数据格式**：
   - 统一输出结构：`data/ActiveView/runs/episode_xxx/`（`rgb/`, `depth/`, `human_pose/`, `metadata.json`）；
   - 逐帧持久化 16 关键点 3D 世界坐标真值，保证未来 v8 可直接读取复现。
6. **完善测试套件**：
   - 保留并组织 `tests/unit/`, `tests/integration/`, `tests/smoke/` 30 项测试，包含 `test_humanoid_loading.py`, `test_motion_pipeline.py`, `test_episode_generation.py` 等。
7. **README 与上下文最终整理**：
   - 全面更新 `ea_avs_mvp_v7/README.md`，明确声明完成内容与严格排除的非目标。

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
- **Active Development Version**: v7.0 (7.0.0 — COMPLETED / FINALIZED & FROZEN)
- **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`
- **Next Research Stage**: v8.0 Active View Foundation
- **Previous Stable Baseline**: v6.0 (6.0.0 — CLOSED / FINALIZED, Read-Only)

## Validation Plan Execution Results
- [x] 统一综合 Demo 演示: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.run_v7_demo` (PASS)
- [x] 流水线全量验证: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.verify_v7_pipeline` (PASS)
- [x] 3 类动作全量闭环: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.validate_three_actions` (PASS)
- [x] Habitat 端到端最小闭环 Smoke Test: `conda run -n habitat python -m ea_avs_mvp_v7.scripts.run_smoke_test --action fall_related --num-frames 10` (PASS)
- [x] v7.0 纯 Python 单元/集成/冒烟测试: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] Motion Assets 单元测试: `python3 -m unittest tools.motion_assets.tests.test_motion_assets` (PASS, 14/14)
- [x] v6.0 纯 Python 回归测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] v6.0 No-GT-Leakage 守卫测试: `PYTHONPATH=ea_avs_mvp_v6 python3 ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
