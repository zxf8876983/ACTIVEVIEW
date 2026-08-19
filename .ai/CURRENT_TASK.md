# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v6.0 Final Fail-Closed Hardening & Closure

## Objective
对 `ea_avs_mvp_v6/` 完成最终封口修复 (Fail-Closed Hardening & Closure)：
1. 彻底删除 Estimated-State 路径中对 generic full-collision `runner.cast_ray` 的静默 fallback，改为严格 fail-closed (`valid=False`, `occlusion_source="unknown"`)；
2. 消除 Habitat static `stage_id` 的猜测默认值，使用 `resolve_stage_id`，缺失时严格 fail-fast 抛出 `RuntimeError`；
3. 增加 Guard A2 AST 静态硬断言，禁止 `cast_ray_to_estimated_point` 内部出现任何 `runner.cast_ray` 调用；
4. 补充 TEST 16（No Static API Fail-Closed）与 TEST 17（Missing stage_id Fail-Fast）纯 Python 测试；
5. 最终验证全套测试与 20 episodes 主实验，正式完成 EA-AVS-MVP v6.0 封版定稿。

## Active Development Version
- **Version**: v6.0 (6.0.0 — CLOSED / FINALIZED)
- **Code Directory**: `ea_avs_mvp_v6/`
- **Active Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`

## Starting Stable Version
- **Version**: v5.0 (5.0.0)
- **Stable Code**: `ea_avs_mvp_v5/`
- **Active Scientific Code Baseline**: `02bc45e`

## Required Work Completed
1. **Estimated Raycast Fail-Closed Hardening (`raycast_utils.py`)**:
   - `cast_ray_to_estimated_point` 完全删除 `runner.cast_ray` fallback。
   - 当 `runner` 缺少 `cast_ray_static_scene` 时返回 `valid=False`, `occlusion_source="unknown"`, `failure_reason="static_scene_raycast_unavailable"`。
   - 当调用抛出异常时返回 `valid=False`, `occlusion_source="unknown"`, `failure_reason="static_scene_raycast_error:..."`。
2. **Explicit Stage ID Resolution & Fail-Fast (`habitat_runner.py`)**:
   - 增加 `resolve_stage_id(habitat_sim_module)`，缺失 `stage_id` 属性时抛出 `RuntimeError`。
   - `HabitatRunner.__init__` 初始化时解析并缓存 `self._stage_id`，提供 `stage_id` 属性。
   - `cast_ray_static_scene` 直接使用 `self._stage_id`，禁止使用任何默认猜测值。
3. **AST Guard A2 Hard Assertion (`test_no_gt_leakage.py`)**:
   - 扫描 `cast_ray_to_estimated_point` 函数语法树，若出现 `.cast_ray` 调用直接 fail，并断言存在 `.cast_ray_static_scene` 调用。
4. **Pure Python Hardening Tests (`test_v60_pure_python.py`)**:
   - 增加 TEST 16: `test_16_estimated_raycast_without_static_api_fails_closed` (PASS)。
   - 增加 TEST 17: `test_17_missing_stage_id_fails_fast` (PASS)。
5. **Static Raycast Debug Script (`debug_static_scene_raycast.py`)**:
   - 打印 `[HabitatRunner] Resolved stage_id: 0`。
   - 验证 full collision 命中 Humanoid (dist=2.02m, obj_id=12) 而 static ray 击中 stage (dist=2.31m, obj_id=0, source=stage)，输出 PASS (exit code 0)。
6. **Orientation Calibration (`calibrate_estimated_orientation.py`)**:
   - 8 方向校准实测中位数误差 0.97°，平均误差 3.03°，输出 PASS (exit code 0)。
7. **20-Episode Final Validation Experiment (`outputs/mvp60_final_validation/`)**:
   - 运行完成且无任何 static API failure，数据与图表落盘成功。

## Do Not Change (Strict Invariants Maintained)
- `ea_avs_mvp/` ~ `ea_avs_mvp_v5/` 历史实现保持只读，Git diff 为 0 行。
- 在线决策阶段（`EstimatedState-Ours`）严禁接收或读取任何 Humanoid GT 变量。
- 严禁在选择前调用渲染器获取未来候选点的 RGB/Depth/Semantic 观测。
- 状态估计失效或 static raycast 能力缺失时安全停留在当前位姿（`stay`），严禁静默回退使用 GT 或 full collision raycast。

## Validation Plan Execution Results
- [x] 语法与编译检查: `python -m compileall ea_avs_mvp_v6` (PASS)
- [x] 纯 Python 17 项单元与逻辑测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 17/17)
- [x] No-GT-Leakage 静态与运行时隔离测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 4/4)
- [x] Static Raycast 静态场景射线检测验证: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/debug_static_scene_raycast.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml` (PASS, exit code 0)
- [x] 朝向校准验证: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/calibrate_estimated_orientation.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml` (PASS, exit code 0)
- [x] 主功能 20 Episodes 闭环运行: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/run_mvp60_estimated_state.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml --episodes 20 --output_dir outputs/mvp60_final_validation` (PASS, code 0)
