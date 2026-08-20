# ACTIVEVIEW Current Task

Status: COMPLETED (v8.1 Local Active View Planning Baseline Finalized & Frozen)

## Current Phase
v8.1 — Local Active View Planning Baseline (Frozen, ready for v9 Action-aware Active View Selection)

## Accomplished in v8.1
1. **View Quality 增强与遮挡评估** (`evaluation/view_quality.py`)：
   - 科学评价公式：$Q(v) = w_1 \cdot \text{human\_visibility} + w_2 \cdot \text{pose\_coverage} - w_3 \cdot \text{distance\_penalty} - w_4 \cdot \text{occlusion\_penalty}$；
   - 权重配置：$w_1=0.40, w_2=0.30, w_3=0.15, w_4=0.15$；
   - 动态输出 `occlusion_ratio` 遮挡比率 $[0.0, 1.0]$。
2. **Oracle / Estimated 信息边界显式标识** (`core/types.py`, `configs/`, `evaluation/`)：
   - 支持 `evaluation_mode: "oracle"` (默认) 与 `"estimated"`；
   - 支持 `pose_source: "oracle"` (真值骨骼) 与 `"estimated"`；
   - 所有输出 JSON (`candidate_statistics.json`, `best_view.json`, `candidate_views.json`, `metadata.json`, `view_selection_report.json`) 均包含 `"evaluation_mode": "oracle"`。
3. **科研实验报表输出** (`scripts/run_v8_demo.py`)：
   - 新增输出 `visualizations/v8_demo/view_selection_report.json`；
   - 持续生成 `candidate_statistics.json`、`candidate_views.json`、`best_view.json`、`metadata.json` 与 `best_view_rgb.png`。
4. **Baseline 视角选择策略接口冻结** (`evaluation/baseline_strategies.py`)：
   - 统一接口 `select_view(strategy)`，支持 `Random View`、`Nearest View`、`Geometry Best View`。
5. **严禁事项与非目标恪守**：
   - ❌ 无全局搜索 / SLAM / 全局路径规划
   - ❌ 无人体检测 / 定位学习
   - ❌ 无强化学习 / 深度学习 view policy
   - ❌ 保持 `ea_avs_mvp_v7/` 及历史版本完全只读

## Validation Plan Execution Results
- [x] Syntax & Compile Check: `python -m compileall ea_avs_mvp_v8` (PASS)
- [x] v8.1 Baseline Strategies 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_baseline_strategies.py` (PASS, 4/4)
- [x] v8.1 View Quality 单元测试: `python3 -m unittest ea_avs_mvp_v8/tests/unit/test_view_quality.py` (PASS, 4/4)
- [x] v8.1 全量单元测试集: `python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"` (PASS, 17/17)
- [x] v7 回归测试集: `python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"` (PASS, 30/30)
- [x] v8.1 端到端演示入口: `conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo` (PASS, report generated, best_view_rgb.png verified)
