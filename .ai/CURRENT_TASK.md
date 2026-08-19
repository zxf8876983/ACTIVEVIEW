# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v6.0 Final Closure Fix & Verification

## Objective
对 `ea_avs_mvp_v6/` 完成最后一轮 Closure Fix：
1. 消除 Estimated-State raycast 对真实 Humanoid collision geometry 的隐式 GT 泄漏（实现 `cast_ray_static_scene`，仅以 `stage_id` 为遮挡物）；
2. 修复 Shared-Pool GT 分支未正确打分的问题（GT-centered 同一几何空间严格对比 GT vs Est 评分）；
3. 让 Oracle 真正具备 candidate-pool identity guard（`pool_mismatch` 与 `oracle_not_upper_bound`），彻底杜绝用 `max(0)` 掩盖错误；
4. 恢复 `strict_gt_skeleton=True` 特权基线与实验评测约束；
5. 补充 15 项纯 Python 单元测试套件 (`test_v60_pure_python.py`) 与射线调试验证脚本 (`debug_static_scene_raycast.py`)；
6. 完善朝向校准退出码与指标统计。

## Active Development Version
- **Version**: v6.0 (6.0.0)
- **Code Directory**: `ea_avs_mvp_v6/`
- **Active Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`

## Starting Stable Version
- **Version**: v5.0 (5.0.0)
- **Stable Code**: `ea_avs_mvp_v5/`
- **Active Scientific Code Baseline**: `02bc45e`

## Required Work Completed
1. **Static-Scene-Only Raycast (`habitat_runner.py` & `raycast_utils.py` & `occlusion.py`)**:
   - `cast_ray_static_scene`: 仅过滤 `hit.object_id == stage_id`，自动忽略 Humanoid 动态/铰接碰撞体。
   - `cast_ray_to_estimated_point`: 3 态分类（`clear`, `static_scene_blocked`, `unknown`）。
   - `debug_static_scene_raycast.py`: 实测全碰撞射线击中 Humanoid (dist=2.02m)，静态射线穿透并击中后方墙体 (dist=2.31m, source=stage)。
2. **Shared-Pool GT 分支评分修复 (`run_mvp60_estimated_state.py`)**:
   - `sel_shared_gt` 严格复用 `GTState-Ours`（同一 GT-centered 几何空间和同一 GT scoring）。
   - 记录 `shared_pool_selected_candidate_id_gt`, `shared_pool_selected_candidate_id_est`, `shared_pool_q_pred_gt_selected`, `shared_pool_q_pred_est_selected`。
3. **Oracle Candidate-Pool Identity Guard (`oracle_policy.py` & `candidate_sampler.py`)**:
   - `CandidateView` 增加 `pool_id` 属性 (`"gt_pool"`, `"est_pool"`, `"shared_pool"`)。
   - `compute_oracle_gap` 严格检查 `oracle_view.pool_id == selected_view.pool_id`（否则返回 `pool_mismatch`）。
   - 同 pool 下若 `selected_q > oracle_q` 严格返回 `oracle_not_upper_bound`，杜绝 `max(0)` 静默掩盖错误。
4. **Strict GT Skeleton 恢复 (`run_mvp60_estimated_state.py` & `compare_gt_estimated_state.py`)**:
   - 默认启用 `strict_gt_skeleton: true`，主实验与对比脚本 fail-fast 校验 15 关节完整性。
5. **Orientation Calibration 退出码 (`calibrate_estimated_orientation.py`)**:
   - 基于配置阈值（`min_valid_rate: 0.75, max_mean: 20.0°, max_median: 15.0°, max_error: 45.0°`）输出 PASS (exit code 0)。
6. **Pose Backend Max People (`pose_backend.py`)**:
   - `TorchvisionPoseBackend` / `MockPoseBackend` 规范支持 `max_people: int = 1` 截断。
7. **Pure Python 15 项测试套件与 No-GT 防护 (`test_v60_pure_python.py` & `test_no_gt_leakage.py`)**:
   - 15/15 纯 Python 测试全量通过。
   - 3/3 No-GT-Leakage 静态与运行时防护通过。
8. **20-Episode 端到端主实验验证 (`outputs/mvp60_closure_validation/`)**:
   - 实验完整运行无报错，数据及图表落盘成功。

## Do Not Change (Strict Invariants Maintained)
- `ea_avs_mvp/` ~ `ea_avs_mvp_v5/` 历史实现保持只读，Git diff 为 0 行。
- 在线决策阶段（`EstimatedState-Ours`）严禁接收或读取任何 Humanoid GT 变量。
- 严禁在选择前调用渲染器获取未来候选点的 RGB/Depth/Semantic 观测。
- 状态估计失效时安全停留在当前位姿（`stay`），严禁静默回退使用 GT。

## Validation Plan Execution Results
- [x] 语法与编译检查: `python -m compileall ea_avs_mvp_v6` (PASS)
- [x] 纯 Python 15 项单元与逻辑测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 15/15)
- [x] No-GT-Leakage 静态与运行时隔离测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 3/3)
- [x] Static Raycast 静态场景射线检测验证: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/debug_static_scene_raycast.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml` (PASS)
- [x] 朝向校准验证: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/calibrate_estimated_orientation.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml` (PASS, exit code 0)
- [x] 主功能 20 Episodes 闭环运行: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/run_mvp60_estimated_state.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml --episodes 20 --output_dir outputs/mvp60_closure_validation` (PASS, code 0)
