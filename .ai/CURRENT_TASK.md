# ACTIVEVIEW Current Task

Status: COMPLETED (v7.0 FINALIZED)

## Task
ACTIVEVIEW v7.0 Development — Humanoid-driven Active Perception Simulation Environment (Finalized & Frozen)

## Objective
完成 ACTIVEVIEW v7.0 版本的最终收尾与冻结（Finalization & Freezing）：
1. **Humanoid 配置固化**：
   - 固化 `configs/humanoid.yaml`（`avatar_name: neutral_0`, `ground_offset: 0.0`, `scale: 1.0`, `default_yaw: 0.0`, `assets_root`, `floor_height: -1.60`）；
   - 消除散落在代码中的硬编码数值。
2. **Demo 配置固化**：
   - 建立 `configs/v7_demo.yaml` 统一管理场景、机器人、Humanoid 放置与输出路径；
   - `run_v7_demo.py` 统一读取配置并支持 CLI 参数一键重载。
3. **统一 Demo 入口与视频自动生成**：
   - 保留规范化主入口 `python -m ea_avs_mvp_v7.scripts.run_v7_demo`；
   - 自动生成 RGB、Depth、Human Pose GT、`metadata.json` 与 `visualizations/v7_final_demo/v7_demo.mp4`。
4. **Episode 数据格式规范化**：
   - 统一输出结构：`data/ActiveView/runs/episode_xxx/`（`rgb/`, `depth/`, `human_pose/`, `metadata.json`）；
   - 逐帧持久化 16 关节 3D GT 姿态至 `human_pose/frame_000000.json`。
5. **端到端流水线全量验证**：
   - 新增 `scripts/verify_v7_pipeline.py`，一键校验 `AMASS -> MotionConverter -> Habitat Motion -> Humanoid -> RGB-D -> GT Pose -> Episode` 全链路。
6. **Human Placement 接口解耦**：
   - 新增 `human/human_spawn.py` 提供 `sample_human_position()` 物理地面放置接口（预留 v8 扩展，不包含任何主动选择算法）。
7. **代码与脚本清理**：
   - 清理所有单次调试脚本，规范保留核心工具。
8. **测试套件整理**：
   - 组织 `tests/unit/`, `tests/integration/`, `tests/smoke/` 30 项测试，全部 PASS。

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
- **Active Development Version**: v7.0 (7.0.0 — FINALIZED & FROZEN)
- **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`
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
