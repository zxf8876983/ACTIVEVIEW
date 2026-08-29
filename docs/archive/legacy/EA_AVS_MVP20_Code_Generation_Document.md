# EA-AVS-MVP v2.0 代码生成指导文档

## 0. 文档用途

本文档用于指导大模型在 GitHub 仓库中生成 `ea_avs_mvp_v2` 版本代码。

v2.0 的目标不是增加复杂算法，而是把主动视角选择实验做得更严谨：

> 移动前只能预测，移动后才能评价；策略只能看 `pred` 指标，实验结果必须报告 `true` 指标。

v2.0 仍然不做真实 humanoid、不做动作识别模型、不做强化学习、不接 ROS/Unity。它是在 v0.1 几何闭环基础上的规范化版本，重点解决 v0.1 中“预测评分”和“真实评估”没有严格区分的问题。

---

## 1. v2.0 的研究定位

v2.0 研究的是：

> 在已知地图、机器人当前位姿和人体目标位置条件下，移动机器人如何从局部可达候选观察位姿中选择一个最可能提升人体关键点可见性的下一观察位姿。

本文中的“观察位姿”不是单纯图像视角，而是机器人相机的位置和朝向：

```text
view = (x, y, z, yaw)
```

其中：

- `(x, y, z)` 表示机器人底盘位置；
- `yaw` 表示机器人相机朝向；
- 相机高度由配置文件 `camera.camera_height` 指定。

---

## 2. v2.0 与 v0.1 的核心区别

| 项目 | v0.1 | v2.0 |
|---|---|---|
| 主要目标 | 跑通几何版候选视角选择 | 严格区分预测评分与真实评估 |
| 策略选择依据 | `S_kp`、`Q` | `S_kp_pred`、`Q_pred` |
| 到达后评估 | 没有严格区分 | 使用 `S_kp_true`、`Q_true` |
| 是否允许 Ours 不移动 | 早期版本不明确 | 必须允许选择 current view |
| 是否允许偷看候选点图像 | 语义不够明确 | 选择阶段禁止使用候选点图像 |
| 图像用途 | 调试为主 | 只能在选择后用于评估和可视化 |

---

## 3. v2.0 研究边界

v2.0 默认机器人已经具备以下基础能力：

1. 已知地图；
2. 自定位；
3. 基础导航；
4. 能够判断候选点是否可达；
5. 能够估计从当前点到候选点的路径距离；
6. 能够在目标观察位姿获取图像。

在 Habitat 中对应为：

| 真实机器人能力 | Habitat 对应实现 |
|---|---|
| 已知地图 | scene + navmesh |
| 可达区域 | pathfinder / navmesh |
| 路径距离 | geodesic distance |
| 当前位姿 | agent state |
| 指定位姿观察 | render_at(position, yaw) |

v2.0 不研究：

1. SLAM 建图；
2. 机器人定位；
3. 底层路径规划算法；
4. 局部避障控制；
5. 前进、后退、左转、右转等低层动作序列；
6. 强化学习；
7. 真实 humanoid；
8. OpenPose、RTMPose、YOLOv8-pose 等真实姿态估计；
9. 动作识别模型训练；
10. 老人情绪识别。

---

## 4. v2.0 必须遵守的强约束

### 4.1 选择阶段禁止信息泄漏

候选观察位姿选择阶段禁止使用：

1. 候选点未来 RGB 图像；
2. 候选点未来 depth 图像；
3. 候选点真实关键点可见率；
4. 候选点真实动作识别结果；
5. 到达候选点后才能获得的任何信息。

选择阶段只允许使用：

1. 机器人当前位姿；
2. 人体目标位置；
3. 抽象人体骨架先验；
4. 相机参数；
5. navmesh 可达性；
6. geodesic distance；
7. 候选位姿与人体之间的几何关系。

### 4.2 评估阶段才允许渲染

只有在策略已经选出观察位姿之后，才允许调用：

```python
runner.render_at(selected.position, selected.yaw)
```

渲染图像只能用于：

1. 保存可视化图片；
2. 到达后真实评估；
3. 调试策略结果。

不能用于策略选择。

### 4.3 Ours 必须允许“不移动”

`OursPolicy` 必须把 `current_view` 和所有有效候选点一起比较。

如果 `current_view.pred_score["Q_pred"]` 最高，则返回 `current_view`。

---

## 5. v2.0 项目目录结构

请新建以下目录：

```text
ea_avs_mvp_v2/
├── configs/
│   └── mvp20_visibility.yaml
├── scripts/
│   └── run_mvp20_visibility.py
├── ea_avs_v2/
│   ├── __init__.py
│   ├── config.py
│   ├── habitat_runner.py
│   ├── skeleton.py
│   ├── geometry.py
│   ├── candidate_sampler.py
│   ├── predictive_evaluator.py
│   ├── true_evaluator.py
│   ├── policies.py
│   ├── metrics.py
│   └── visualization.py
└── outputs/
```

说明：

- `predictive_evaluator.py` 只负责移动前预测评分；
- `true_evaluator.py` 只负责到达后真实评估；
- `policies.py` 只能使用 `pred_score`；
- `metrics.py` 必须同时输出 pred 和 true 指标。

---

## 6. 配置文件模板

文件路径：

```text
ea_avs_mvp_v2/configs/mvp20_visibility.yaml
```

配置模板：

```yaml
project:
  name: "EA-AVS-MVP2.0"
  seed: 42

habitat:
  scene_path: "PLEASE_SET_SCENE_PATH.glb"
  navmesh_path: "PLEASE_SET_NAVMESH_PATH.navmesh"
  use_habitat_lab: false

camera:
  width: 640
  height: 480
  hfov_deg: 90
  vfov_deg: 60
  camera_height: 1.2

episode:
  num_episodes: 20
  min_robot_human_distance: 1.5
  max_robot_human_distance: 4.0
  max_sampling_tries: 100

human:
  pose_type: "standing"
  base_height_offset: 0.0

candidate_sampling:
  radii: [1.5, 2.0, 2.5]
  angles_deg: [-135, -90, -45, 0, 45, 90, 135, 180]
  min_distance_to_human: 1.2
  max_distance_to_human: 3.0
  max_geodesic_distance: 6.0

visibility:
  min_depth: 0.5
  max_depth: 5.0

score_weights:
  torso: 0.4
  lower_body: 0.4
  head: 0.2

predictive_score:
  w_kp_pred: 0.60
  w_center_pred: 0.20
  w_dist_pred: 0.20
  w_move: 0.20

true_score:
  w_kp_true: 0.70
  w_center_true: 0.15
  w_dist_true: 0.15

movement:
  allow_stay: true
  stay_margin: 0.0

distance_score:
  optimal_distance: 2.0
  sigma: 0.7

output:
  save_images: true
  save_candidate_json: true
  save_csv: true
```

---

## 7. 核心数据结构

### 7.1 CandidateView

定义在：

```text
ea_avs_mvp_v2/ea_avs_v2/candidate_sampler.py
```

建议实现：

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class CandidateView:
    candidate_id: int
    position: np.ndarray
    yaw: float
    geodesic_distance: float
    euclidean_distance_to_human: float
    is_valid: bool
    invalid_reason: str = ""
    pred_score: dict = field(default_factory=dict)
    true_score: dict = field(default_factory=dict)
    selected_by: list = field(default_factory=list)
```

字段说明：

| 字段 | 含义 |
|---|---|
| `candidate_id` | 候选位姿编号，current view 使用 -1 |
| `position` | 候选位姿位置，shape=(3,) |
| `yaw` | 候选位姿朝向，单位 rad |
| `geodesic_distance` | 当前点到候选点的测地距离 |
| `euclidean_distance_to_human` | 候选点到人体目标的欧氏距离 |
| `is_valid` | 候选位姿是否有效 |
| `invalid_reason` | 无效原因 |
| `pred_score` | 移动前预测评分 |
| `true_score` | 到达后真实评分 |
| `selected_by` | 被哪些策略选中 |

---

## 8. 文件级实现要求

## 8.1 `config.py`

功能：读取并校验 YAML 配置。

必须实现：

```python
def load_config(path: str) -> dict:
    pass
```

要求：

1. 使用 `yaml.safe_load`；
2. 检查配置文件存在；
3. 检查必要字段；
4. 不写实验逻辑。

必要字段包括：

```text
habitat.scene_path
camera.width
camera.height
candidate_sampling.radii
candidate_sampling.angles_deg
predictive_score.w_kp_pred
true_score.w_kp_true
```

---

## 8.2 `skeleton.py`

功能：生成抽象人体 3D 骨架。

MVP2.0 只要求支持 `standing`。

必须实现：

```python
def get_standing_skeleton(human_base_pos: np.ndarray) -> dict:
    pass

def get_skeleton(human_base_pos: np.ndarray, pose_type: str = "standing") -> dict:
    pass
```

必须定义：

```python
SKELETON_STANDING = {
    "head":           [0.00, 1.60, 0.00],
    "neck":           [0.00, 1.40, 0.00],
    "pelvis":         [0.00, 0.95, 0.00],
    "left_shoulder":  [-0.22, 1.35, 0.00],
    "right_shoulder": [ 0.22, 1.35, 0.00],
    "left_hip":       [-0.16, 0.90, 0.00],
    "right_hip":      [ 0.16, 0.90, 0.00],
    "left_knee":      [-0.16, 0.50, 0.00],
    "right_knee":     [ 0.16, 0.50, 0.00],
    "left_ankle":     [-0.16, 0.10, 0.00],
    "right_ankle":    [ 0.16, 0.10, 0.00],
}
```

必须定义关键点分组：

```python
KEYPOINT_GROUPS = {
    "torso": ["neck", "pelvis", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "lower_body": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
    "head": ["head"],
}
```

---

## 8.3 `geometry.py`

功能：提供基础几何计算。

必须实现：

```python
def normalize_angle(angle: float) -> float:
    pass

def yaw_to_forward(yaw: float) -> np.ndarray:
    pass

def compute_look_at_yaw(source_pos: np.ndarray, target_pos: np.ndarray) -> float:
    pass

def angle_in_camera_fov(
    camera_base_pos: np.ndarray,
    camera_yaw: float,
    point: np.ndarray,
    hfov_deg: float,
    vfov_deg: float,
    camera_height: float,
    min_depth: float,
    max_depth: float,
) -> dict:
    pass

def gaussian_score(value: float, optimal: float, sigma: float) -> float:
    pass
```

坐标约定：

- Y 轴向上；
- yaw = 0 表示朝向 +Z；
- yaw 正方向朝 +X 旋转；
- 相机位置 = robot_base_pos + `[0, camera_height, 0]`。

---

## 8.4 `habitat_runner.py`

功能：封装 Habitat-Sim。

必须实现：

```python
class HabitatRunner:
    def __init__(self, config: dict):
        pass

    def sample_navigable_point(self) -> np.ndarray:
        pass

    def is_navigable(self, point: np.ndarray) -> bool:
        pass

    def snap_point(self, point: np.ndarray) -> np.ndarray:
        pass

    def geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        pass

    def render_at(self, position: np.ndarray, yaw: float) -> dict:
        pass

    def close(self):
        pass
```

重要约束：

- `render_at()` 不能在候选点评分阶段调用；
- `render_at()` 只能在策略选择后用于保存图像和计算 true score。

---

## 8.5 `candidate_sampler.py`

功能：采样并过滤候选观察位姿。

必须实现：

```python
class CandidateSampler:
    def __init__(self, config: dict):
        pass

    def sample(self, human_pos: np.ndarray, robot_pos: np.ndarray, runner: HabitatRunner) -> list[CandidateView]:
        pass
```

采样流程：

1. 以人体位置为中心；
2. 按 `radii` 和 `angles_deg` 生成候选点；
3. 使用 `runner.snap_point()` 吸附到 navmesh；
4. 使用 `runner.is_navigable()` 判断是否可达；
5. 使用 `runner.geodesic_distance()` 计算路径距离；
6. 过滤距离过近、过远、无路径、移动距离过大的点；
7. 使用 `compute_look_at_yaw()` 使候选点朝向人体。

---

## 8.6 `predictive_evaluator.py`

功能：移动前预测候选观察位姿质量。

这是 v2.0 的核心模块。

必须实现：

```python
class PredictiveEvaluator:
    def __init__(self, config: dict):
        pass

    def score_view_pred(
        self,
        view_pos: np.ndarray,
        view_yaw: float,
        robot_start_pos: np.ndarray,
        human_base_pos: np.ndarray,
        human_skeleton: dict,
        geodesic_distance: float,
    ) -> dict:
        pass
```

输出字段必须包含：

```python
{
    "S_kp_pred": float,
    "S_center_pred": float,
    "S_dist_pred": float,
    "C_move": float,
    "Q_pred": float,
    "torso_visibility_pred": float,
    "lower_body_visibility_pred": float,
    "head_visibility_pred": float,
    "visible_keypoints_pred": list[str],
    "invisible_keypoints_pred": list[str],
}
```

预测评分函数：

```text
Q_pred = w_kp_pred * S_kp_pred
       + w_center_pred * S_center_pred
       + w_dist_pred * S_dist_pred
       - w_move * C_move
```

禁止事项：

- 不允许调用 `render_at()`；
- 不允许使用候选点 RGB；
- 不允许使用候选点 depth；
- 不允许使用候选点真实关键点结果。

---

## 8.7 `true_evaluator.py`

功能：策略选择完成后，对选中视角进行真实评估。

必须实现：

```python
class TrueEvaluator:
    def __init__(self, config: dict):
        pass

    def score_view_true(
        self,
        obs: dict,
        view_pos: np.ndarray,
        view_yaw: float,
        human_base_pos: np.ndarray,
        human_skeleton: dict,
    ) -> dict:
        pass
```

v2.0 简化说明：

由于抽象 skeleton 没有真实渲染进图像，`true_evaluator.py` 第一版可以仍然使用几何 FOV 计算 true 指标，但必须满足两个要求：

1. 必须在策略选定并调用 `render_at()` 后计算；
2. 字段名必须和 pred 分开。

输出字段必须包含：

```python
{
    "S_kp_true": float,
    "S_center_true": float,
    "S_dist_true": float,
    "Q_true": float,
    "torso_visibility_true": float,
    "lower_body_visibility_true": float,
    "head_visibility_true": float,
}
```

---

## 8.8 `policies.py`

功能：实现策略。

必须实现：

```python
class FixedPolicy:
    name = "Fixed"
    def select(self, current_view, candidates):
        pass

class RandomPolicy:
    name = "Random"
    def select(self, current_view, candidates):
        pass

class NearestPolicy:
    name = "Nearest"
    def select(self, current_view, candidates):
        pass

class OursPolicy:
    name = "Ours"
    def select(self, current_view, candidates):
        pass
```

策略逻辑：

1. Fixed：返回 current view；
2. Random：随机选有效候选点，无候选则返回 current view；
3. Nearest：选 geodesic distance 最短候选点，无候选则返回 current view；
4. Ours：比较 current view 和所有有效候选点，选择 `Q_pred` 最大者。

重要约束：

- Ours 只能使用 `pred_score`；
- Ours 不能使用 `true_score`；
- Ours 必须允许不移动。

---

## 8.9 `metrics.py`

功能：输出 CSV 与 JSONL。

必须实现：

```python
class MetricsWriter:
    def __init__(self, output_dir: str):
        pass

    def write_metric_row(self, row: dict):
        pass

    def write_episode_summary(self, summary: dict):
        pass

    def close(self):
        pass
```

`metrics.csv` 字段必须包含：

```text
episode_id
scene_id
policy
status
num_candidates
selected_is_current
human_x
human_y
human_z
robot_start_x
robot_start_y
robot_start_z
selected_x
selected_y
selected_z
selected_yaw
geodesic_distance
S_kp_pred
S_center_pred
S_dist_pred
C_move
Q_pred
torso_visibility_pred
lower_body_visibility_pred
head_visibility_pred
S_kp_true
S_center_true
S_dist_true
Q_true
torso_visibility_true
lower_body_visibility_true
head_visibility_true
pred_true_gap
visibility_gain_pred
visibility_gain_true
```

字段解释：

- `selected_is_current`：策略是否选择不移动；
- `pred_true_gap = Q_true - Q_pred`；
- `visibility_gain_pred = S_kp_pred_selected - S_kp_pred_current`；
- `visibility_gain_true = S_kp_true_selected - S_kp_true_current`。

---

## 8.10 `visualization.py`

功能：保存图像和候选点调试 JSON。

必须实现：

```python
def save_rgb_image(rgb: np.ndarray, path: str):
    pass

def save_candidate_debug_json(candidates: list, path: str):
    pass
```

输出要求：

```text
outputs/
├── metrics.csv
├── episodes.jsonl
├── images/
│   ├── ep_000_current.png
│   ├── ep_000_fixed.png
│   ├── ep_000_random.png
│   ├── ep_000_nearest.png
│   └── ep_000_ours.png
└── debug/
    └── ep_000_candidates.json
```

---

## 8.11 `run_mvp20_visibility.py`

功能：v2.0 主入口。

运行命令：

```bash
python scripts/run_mvp20_visibility.py \
  --config configs/mvp20_visibility.yaml \
  --episodes 20 \
  --output-dir outputs/mvp20_test_run
```

主流程：

```text
1. 读取配置
2. 初始化 HabitatRunner
3. 初始化 CandidateSampler
4. 初始化 PredictiveEvaluator
5. 初始化 TrueEvaluator
6. 初始化四种策略
7. 对每个 episode：
   7.1 采样人体位置
   7.2 生成 skeleton
   7.3 采样机器人起点
   7.4 构造 current_view
   7.5 对 current_view 计算 pred_score
   7.6 采样候选点
   7.7 对候选点计算 pred_score
   7.8 策略根据 pred_score 选择位姿
   7.9 对 current_view 和各策略选中位姿渲染图像
   7.10 对渲染后的位姿计算 true_score
   7.11 写 metrics.csv
   7.12 写 episodes.jsonl
   7.13 保存 candidates debug json
8. 关闭文件和模拟器
```

关键约束：

- 7.7 之前不能调用候选点 `render_at()`；
- 策略选择只能用 `pred_score`；
- `true_score` 只能在 7.9 之后计算；
- 单个 episode 失败不能中断整体实验。

---

## 9. v2.0 主流程伪代码

```python
def run_one_episode(...):
    human_pos = sample_valid_human_position(runner, config)
    skeleton = get_skeleton(human_pos, config["human"]["pose_type"])

    robot_start_pos = sample_robot_start_position_around_human(runner, human_pos, config)
    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)

    current_view = CandidateView(
        candidate_id=-1,
        position=robot_start_pos,
        yaw=robot_start_yaw,
        geodesic_distance=0.0,
        euclidean_distance_to_human=distance(robot_start_pos, human_pos),
        is_valid=True,
    )

    current_view.pred_score = pred_evaluator.score_view_pred(...)

    candidates = sampler.sample(human_pos, robot_start_pos, runner)
    valid_candidates = [c for c in candidates if c.is_valid]

    for cand in valid_candidates:
        cand.pred_score = pred_evaluator.score_view_pred(...)

    selected_by_policy = {}
    for policy in policies:
        selected_by_policy[policy.name] = policy.select(current_view, candidates)

    obs_current = runner.render_at(current_view.position, current_view.yaw)
    current_view.true_score = true_evaluator.score_view_true(...)
    save current image

    for policy_name, selected in selected_by_policy.items():
        obs = runner.render_at(selected.position, selected.yaw)
        selected.true_score = true_evaluator.score_view_true(...)
        save image
        write metric row

    save debug json
    write episode summary
```

---

## 10. 验收标准

### 10.1 运行验收

必须能运行：

```bash
python scripts/run_mvp20_visibility.py \
  --config configs/mvp20_visibility.yaml \
  --episodes 20 \
  --output-dir outputs/mvp20_test_run
```

### 10.2 文件验收

必须生成：

```text
outputs/mvp20_test_run/metrics.csv
outputs/mvp20_test_run/episodes.jsonl
outputs/mvp20_test_run/images/
outputs/mvp20_test_run/debug/
```

### 10.3 指标验收

成功 episode 不少于 14 个。

`metrics.csv` 行数不少于：

```text
14 episodes × 4 policies = 56 rows
```

### 10.4 逻辑验收

必须满足：

1. `Ours` 的选择只依赖 `Q_pred`；
2. `Q_true` 不参与策略选择；
3. `Ours` 可以选择 current view；
4. 至少保存 current 图像；
5. `metrics.csv` 中同时存在 pred 和 true 指标；
6. debug JSON 中保存每个候选点的 `pred_score`。

---

## 11. v2.0 当前局限与后续版本

v2.0 仍有以下限制：

1. 人体没有真实渲染；
2. true evaluation 仍然可能是几何评估；
3. 没有真实动作识别；
4. 没有人体朝向；
5. 没有动作关键部位动态权重；
6. 没有家具遮挡 ray casting；
7. 没有 Habitat humanoid。

后续版本建议：

| 后续版本 | 目标 |
|---|---|
| v2.1 | 多姿态 skeleton：standing/sitting/lying/bending |
| v2.2 | 加入 ray casting 遮挡预测 |
| v2.3 | 加入人体朝向与侧前方视角偏好 |
| v2.4 | 加入动作关键部位权重 |
| v3.0 | 接入 Habitat humanoid |

---

## 12. 给大模型的最终生成提示词

```text
请根据 EA_AVS_MVP20_Code_Generation_Document.md 实现 ea_avs_mvp_v2。

最重要的原则：
1. 选择阶段禁止使用候选点未来图像。
2. 选择阶段只使用 pred_score。
3. true_score 只能在策略选择完成并渲染该位姿后计算。
4. OursPolicy 必须允许选择 current view，即允许不移动。
5. 所有代码必须有详细中文注释。
6. 不要实现强化学习、动作识别、真实 humanoid、ROS、Unity。

请严格创建以下目录：
ea_avs_mvp_v2/
├── configs/mvp20_visibility.yaml
├── scripts/run_mvp20_visibility.py
├── ea_avs_v2/__init__.py
├── ea_avs_v2/config.py
├── ea_avs_v2/habitat_runner.py
├── ea_avs_v2/skeleton.py
├── ea_avs_v2/geometry.py
├── ea_avs_v2/candidate_sampler.py
├── ea_avs_v2/predictive_evaluator.py
├── ea_avs_v2/true_evaluator.py
├── ea_avs_v2/policies.py
├── ea_avs_v2/metrics.py
└── ea_avs_v2/visualization.py

验收命令：
python scripts/run_mvp20_visibility.py --config configs/mvp20_visibility.yaml --episodes 20 --output-dir outputs/mvp20_test_run

验收文件：
- metrics.csv
- episodes.jsonl
- images/
- debug/

metrics.csv 必须同时包含：
- S_kp_pred, Q_pred
- S_kp_true, Q_true
- pred_true_gap
- visibility_gain_pred
- visibility_gain_true
```

---

## 13. 一句话总结

EA-AVS-MVP v2.0 的核心不是增加复杂算法，而是让实验逻辑更严谨：

> 候选点选择阶段只能使用预测信息，不能偷看未来图像；策略选择后才允许渲染和真实评估。
