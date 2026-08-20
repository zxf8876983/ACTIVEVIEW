# ACTIVEVIEW v8.2: Local Active View Planning Baseline

> **ACTIVEVIEW v8.2 Final Baseline Specification & Execution Guide**  
> *v8 assumes coarse human localization is available and focuses on local viewpoint optimization.*

---

## 1. 科研定位与核心假设 (Scientific Scope & Assumptions)

ACTIVEVIEW v8.2 致力于建立可靠、可复现、严谨的 **Local Active View Planning Baseline（局部主动视点规划与几何质量基准）**。

### 核心科研假设 (Core Assumptions):
1. **Human location is known**: 假设目标老人/人体粗糙世界坐标 $[x, y, z]$ 已由外部系统（如固定监控、环境传感器或先验定位模块）获得；
2. **Robot is already in human vicinity**: 移动机器人底盘已导航就位至人体周围局部空间 ($1.5\text{m} \sim 3.0\text{m}$)；
3. **Local Viewpoint Optimization**: 在已知人体周围空间约束下，通过极坐标网格候选采样、三阶硬约束过滤与透明多维观测质量打分，寻找几何意义上最优的主动观察视角。

### 明确排除的非目标 (Explicit Non-Goals):
- ✗ **No Human Search**: 不进行全场景人体搜索或盲目探索；
- ✗ **No Global Navigation**: 不包含 SLAM、全局路径规划与避障控制器；
- ✗ **No Action Recognition**: 不训练动作识别/分类神经网络模型；
- ✗ **No Learned View Policy**: 不包含深度强化学习 (RL) 或复杂 Policy Network（留待 v9 动作感知主动视角选择研究）。

---

## 2. v8 完成的核心能力 (What v8 Solves)

- ✓ **Local candidate viewpoint generation**: 围绕人体的局部极坐标网格采样（半径 $1.5\text{m} \sim 3.0\text{m}$，角度 $0^\circ \sim 360^\circ$）；
- ✓ **Navigation feasibility checking**: 基于 Habitat Pathfinder NavMesh 的底盘可行域与路径可达性校验 (`NavigationConstraint`)；
- ✓ **Line-of-sight checking**: 基于物理光线追踪 (Ray Cast) 剔除隔墙阻挡视点 (`LineOfSightConstraint`)；
- ✓ **Human visibility evaluation**: 相机水平 FOV 视锥与人体投影面积占比 ($A_{proj} \ge 1.5\%$) 判定 (`HumanVisibilityConstraint`)；
- ✓ **Geometry-based viewpoint ranking**: 透明科学打分函数 $Q(v) = w_1 \cdot \text{vis} + w_2 \cdot \text{cov} - w_3 \cdot \text{dist} - w_4 \cdot \text{loss}$ (`ViewQualityEvaluator`)；
- ✓ **Standard baseline comparison**: 统一接口 `select_view(strategy)` 支持 `Random View`, `Nearest View`, `Geometry Best View`。

---

## 3. 核心接口与模块结构 (Architecture Map)

```text
ea_avs_mvp_v8/
├── configs/
│   ├── v8_demo.yaml            # 默认 Demo 配置文件
│   ├── viewpoint.yaml          # 视点采样与约束配置
│   ├── humanoid.yaml           # 人形模型配置
│   └── v8_experiment.yaml      # 标准实验参数配置文件
├── core/
│   ├── config.py               # 统一配置容器与加载器
│   ├── paths.py                # 物理数据路径解析
│   └── types.py                # CandidateViewpoint, ViewpointQuality 数据结构
├── viewpoint/
│   └── viewpoint_generator.py  # 局部极坐标规则视点生成器
├── constraints/
│   ├── navigation_constraint.py       # NavMesh 可行走性检查
│   ├── line_of_sight_constraint.py    # RayCast 通视性检查
│   ├── human_visibility_constraint.py # FOV 视锥与投影面积检查
│   └── constraint_checker.py          # 综合三阶约束过滤管道
├── evaluation/
│   ├── view_quality.py         # 综合质量评分 Q(v) 与排序器
│   ├── baseline_strategies.py  # Random, Nearest, Geometry-best 基线选择策略
│   └── view_metrics.py         # 实验指标汇总统计工具
├── robot/
│   └── robot_adapter.py        # Habitat Agent 与 Camera 外参同步挂载器
└── scripts/
    ├── run_v8_demo.py          # 端到端 Demo 演示脚本
    └── debug_view_render.py    # 单视点最小渲染验证脚本
```

---

## 4. 快速运行与验证 (Quick Start)

### 1. 运行纯 Python 单元测试 (无需仿真物理引擎):
```bash
python3 -m unittest discover -s ea_avs_mvp_v8/tests -p "test_*.py"
```

### 2. 运行 v8.2 端到端演示:
```bash
conda run -n habitat python -m ea_avs_mvp_v8.scripts.run_v8_demo --strategy geometry_best
```

### 3. 查看输出成果 (`data/ActiveView/visualizations/v8_demo/`):
- `view_selection_report.json`: 论文核心统计报表（包含策略、模式、各阶段有效候选数、最优得分与未观测比例）；
- `candidate_statistics.json`: 逐级约束过滤统计；
- `candidate_views.json`: 全量候选视点数据及可行性标记；
- `best_view.json`: 选定视点外参、三维坐标及覆盖指标；
- `best_view_rgb.png`: 选定视点高质量渲染图像。
