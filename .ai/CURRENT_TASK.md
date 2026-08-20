# ACTIVEVIEW Current Task

Status: COMPLETED (v8.2 Local Active View Planning Final Baseline Finalized & Frozen)

## Current Phase
v8.2 — Local Active View Planning Final Baseline (Frozen, ready for v9 Action-aware Active View Selection)

## Accomplished in v8.2 Final Baseline
1. **科学指标命名与遮挡评价修正** (`evaluation/view_quality.py`)：
   - 指标严谨更名：`visibility_loss_ratio`（基于未被观测关节比率计算）取代易被误解为物理硬遮挡的命名；
   - 保留兼容字段 `occlusion_ratio`，并在 metadata 附带显式科学说明：`"estimated from missing visible joints rather than explicit physical occlusion"`；
   - 透明打分函数：$Q(v) = w_1 \cdot \text{human\_visibility} + w_2 \cdot \text{pose\_coverage} - w_3 \cdot \text{distance\_penalty} - w_4 \cdot \text{visibility\_loss\_penalty}$。
2. **明确 Constraint 与 Quality 的职责划分**：
   - **Constraint (布尔判定)**：回答“是否可观察”，包括 `navmesh_valid`、`line_of_sight_valid`、`human_visible`（$A_{proj} \ge 1.5\%$）；
   - **Quality (连续得分)**：回答“观察质量好坏”，输出 $Q(v) \in [0.0, 1.0]$。
3. **最终科研实验报表标准化** (`scripts/run_v8_demo.py`)：
   - 生成 `visualizations/v8_demo/view_selection_report.json`；
   - 包含 `strategy`, `evaluation_mode`, `total_candidates`, `navmesh_valid`, `line_of_sight_valid`, `human_visible`, `feasible_candidates`, `selected_view_id`, `best_score`, `distance`, `pose_coverage`, `visibility_loss_ratio`。
4. **Baseline 接口与实验配置固化** (`evaluation/baseline_strategies.py`, `configs/v8_experiment.yaml`)：
   - 固化 `select_view(strategy)` 支持 `random`、`nearest`、`geometry_best`；
   - 新增 `configs/v8_experiment.yaml` 将全套实验参数配置化。
5. **文档完备性与非目标承诺** (`ea_avs_mvp_v8/README.md`)：
   - 建立 `ea_avs_mvp_v8/README.md`，明确阐述 v8 能做什么与不能做什么；
   - 明确声明：*"v8 assumes coarse human localization is available and focuses on local viewpoint optimization."*

## Validation Plan Execution Results
- [x] Syntax & Compile Check: `python -m compileall ea_avs_mvp_v8` (PASS)
- [x] v8.2 Unit Tests: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7 Regression Tests: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] v8.2 End-to-End Demo: `conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo` (PASS, view_selection_report.json verified, best_view_rgb.png verified)
