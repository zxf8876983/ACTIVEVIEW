# ACTIVEVIEW Current Task

Status: COMPLETED

## Task
Implement ACTIVEVIEW v6.0: Estimated-State Active Observation Pose Selection

## Objective
按照当前仓库中的 `EA_AVS_MVP60_Code_Generation_Document.md` 规范，实现从“当前 RGB-D 观测估计人体状态”到“驱动主动观察位姿选择”的完整科研代码实现 `ea_avs_mvp_v6/`，严格保护 Pred/True 与零 GT 泄漏信息边界，并建立与 GT-State baseline 及 Oracle 离线上界的系统对照。

## Active Development Version
- **Version**: v6.0 (6.0.0)
- **Code Directory**: `ea_avs_mvp_v6/`
- **Active Specification**: `EA_AVS_MVP60_Code_Generation_Document.md`

## Starting Stable Version
- **Version**: v5.0 (5.0.0)
- **Stable Code**: `ea_avs_mvp_v5/`
- **Active Scientific Code Baseline**: `02bc45e`

## Required Work Completed
1. **Perception Frontend & Adapters**:
   - `pose_backend.py`: 支持通用 COCO-17 关键点检测器抽象、TorchVision KeypointRCNN 后端适配器与离线 Mock 后端。
   - `keypoint_schema.py`: 统一 COCO-17 到 EA-AVS 15-keypoint schema 的映射与颈部/骨盆中点派生关系。
   - `depth_lifter.py`: 稳健局部 5x5 邻域深度采样、MAD 波动过滤与相机/世界坐标 3D 逆投影。
2. **State Estimation Core**:
   - `orientation_estimator.py`: 基于双侧解剖学对称对（肩/髋）的鲁棒躯干朝向（yaw）估计。
   - `skeleton_completion.py`: 基于观测 3D 关节、估计朝向、中心和尺度的 Proxy 全身骨架构建（杜绝 GT 补全）。
   - `estimated_human_state.py` & `state_validation.py`: 状态数据结构与有效性验证规则。
   - `human_state_estimator.py`: 统一封装 Current RGB-D -> EstimatedHumanState 前端（无任何 GT 入参）。
3. **Estimated-State NBV Pipeline**:
   - `candidate_sampler.py`: 以 `estimated_human_position` 为中心的候选观察位姿采样。
   - `estimated_predictive_evaluator.py`: 基于估计状态的候选预测评分器（零 GT 泄漏）。
   - `predictive_evaluator.py`: GT 状态预测评分器（供 GTState-Ours 特权基线使用）。
   - `policies.py`: `EstimatedStateOursPolicy`（支持 current-vs-candidate 竞优、stay 及感知失效安全兜底）、`GTStateOursPolicy` 及基线策略。
   - `oracle_policy.py`: 离线同口径 Oracle-NBV 评估与 Oracle Gap / Estimation Gap 计算。
   - `metrics.py`: 状态估计误差与三支路比较指标落盘工具。
   - `visualization.py`: 2D/3D 姿态检测与候选视角可视化输出。
4. **Validation & Debug Scripts**:
   - `scripts/smoke_test_pose_backend.py` (PASS)
   - `scripts/debug_current_rgbd_pose.py` (PASS)
   - `scripts/debug_depth_lifting.py` (PASS)
   - `scripts/calibrate_estimated_orientation.py` (PASS)
   - `scripts/compare_gt_estimated_state.py` (PASS)
   - `scripts/test_no_gt_leakage.py` (PASS, 3/3)
   - `scripts/test_v60_pure_python.py` (PASS, 7/7)
   - `scripts/run_mvp60_estimated_state.py` (PASS, 20 episodes)

## Do Not Change (Strict Invariants Maintained)
- `ea_avs_mvp/` ~ `ea_avs_mvp_v5/` 历史实现保持只读，Git diff 为 0 行。
- 在线决策阶段（`EstimatedState-Ours`）严禁接收或读取任何 Humanoid GT 变量。
- 严禁在选择前调用渲染器获取未来候选点的 RGB/Depth/Semantic 观测。
- 状态估计失效时安全停留在当前位姿（`stay`），严禁静默回退使用 GT。

## Validation Plan Execution Results
- [x] 语法与编译检查: `python -m compileall ea_avs_mvp_v6` (PASS)
- [x] 纯 Python 单元与逻辑测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_v60_pure_python.py` (PASS, 7/7)
- [x] No-GT-Leakage 静态与运行时隔离测试: `PYTHONPATH=ea_avs_mvp_v6 python ea_avs_mvp_v6/scripts/test_no_gt_leakage.py` (PASS, 3/3)
- [x] 视觉前端与状态估计 Debug 验证: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/debug_depth_lifting.py` (PASS)
- [x] 朝向校准验证: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/calibrate_estimated_orientation.py` (PASS)
- [x] 状态对比批处理: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/compare_gt_estimated_state.py --episodes 5` (PASS)
- [x] 主功能 20 Episodes 闭环运行: `PYTHONPATH=ea_avs_mvp_v6 /home/zxf/anaconda3/envs/habitat/bin/python ea_avs_mvp_v6/scripts/run_mvp60_estimated_state.py --config ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml --episodes 20` (PASS, code 0)
