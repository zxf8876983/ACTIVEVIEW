# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
ACTIVEVIEW v6.0 Scientific Rigor Fix & Multi-Protocol Validation

## Objective
对 `ea_avs_mvp_v6/` 执行第一轮 Scientific Rigor Fix：
1. 修复 Estimated-State 遮挡预测自洽性与 `visible_occ` 严格有效性过滤（GT-free 几何光路遮挡检测）；
2. 统一人体 Yaw 坐标系约定与前向向量，完成 8 方向校准；
3. 解耦候选池（Est-Pool / GT-Pool / Shared-Pool），建立同口径 Oracle-GTPool 与 Oracle-EstPool 上界与 Gap 计算体系；
4. 调整状态估计流水线顺序（尺度感知前置）并建立 6 级下肢/骨盆基座位置估计优先级；
5. 采用 3D Bilateral 中点解剖推导 Neck 与 Pelvis，消除 2D midpoint 深度采样误差；
6. 完善 AST 禁词、Runner 隔离与未来观测拦截三重零 GT 泄漏防御；
7. 完善异常边界处理与有限性校验。

## Active Development Version
- **Version**: v6.0 (6.0.0)
- **Code Directory**: `ea_avs_mvp_v6/`
- **Active Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`

## Starting Stable Version
- **Version**: v5.0 (5.0.0)
- **Stable Code**: `ea_avs_mvp_v5/`
- **Active Scientific Code Baseline**: `02bc45e`

## Required Work Completed
1. **Core Modules Rigor Updates**:
   - `raycast_utils.py`: 增加 `cast_ray_to_estimated_point`（GT-free 3 态分类，0.12m 容差带）。
   - `occlusion.py`: 增加 `compute_estimated_keypoint_occlusion` 与 `compute_estimated_occlusion_stats`。
   - `estimated_predictive_evaluator.py`: 严格条件 `in_fov and occ_valid and not is_occ` 过滤 `visible_occ`，失效返回 15 点。
   - `depth_lifter.py`: 3D 双侧解剖推导 neck/pelvis，移除 2D midpoint 深度采样，增加 `min_valid_depth_pixels`。
   - `human_state_estimator.py`: 尺度估计前置，6 级基座位置估计优先级，读取 `POSE_SKELETONS["standing"]`。
   - `humanoid_manager.py`: 对齐 Habitat KinematicHumanoid URDF 原生正面与全局 $+Z$ 坐标系。
   - `orientation_estimator.py`: 统一解剖左右肩与前向向量推导公式。
   - `oracle_policy.py`: 区分 Oracle-GTPool 与 Oracle-EstPool，计算同口径 Gap，拒绝跨池伪负 Gap。
   - `skeleton_completion.py` & `pose_backend.py`: 未知类型显式抛出 `ValueError`。
   - `state_validation.py`: 增加有限性与数值合理范围校验。
   - `metrics.py`: 扩展三协议指标记录与 Summary 汇总统计。
   - `configs/mvp60_estimated_state.yaml`: 规范配置参数字段。
2. **Validation & Test Scripts**:
   - `scripts/test_v60_pure_python.py`: 覆盖 10 项纯 Python 科学严谨性测试 (PASS, 8/8 test methods)。
   - `scripts/test_no_gt_leakage.py`: 实现 Guard A、B、C 三重静态与运行时防护断言 (PASS, 3/3)。
   - `scripts/calibrate_estimated_orientation.py`: 实测 8 方向校准中位数误差 0.97°，平均误差 3.03° (PASS)。
   - `scripts/compare_gt_estimated_state.py`: 8 episode 状态对比批处理运行通过 (PASS)。
   - `scripts/run_mvp60_estimated_state.py`: 20 episode 闭环实验运行完成，数据落盘至 `outputs/mvp60_rigor_validation/` (PASS)。

## Do Not Change (Strict Invariants Maintained)
- `ea_avs_mvp/` ~ `ea_avs_mvp_v5/` 历史实现保持只读，Git diff 为 0 行。
- 在线决策阶段（`EstimatedState-Ours`）严禁接收或读取任何 Humanoid GT 变量。
- 严禁在选择前调用渲染器获取未来候选点的 RGB/Depth/Semantic 观测。
- 状态估计失效时安全停留在当前位姿（`stay`），严禁静默回退使用 GT。

## Validation Plan Execution Results
- [x] 语法与编译检查: `python -m compileall ea_avs_mvp_v6` (PASS)
- [x] 纯 Python 单元与逻辑测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 8/8)
- [x] No-GT-Leakage 静态与运行时隔离测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 3/3)
- [x] 朝向校准验证: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/calibrate_estimated_orientation.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml` (PASS, median error 0.97°)
- [x] 状态对比批处理: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/compare_gt_estimated_state.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml --episodes 8` (PASS)
- [x] 主功能 20 Episodes 闭环运行: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/run_mvp60_estimated_state.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml --episodes 20 --output_dir outputs/mvp60_rigor_validation` (PASS, code 0)
