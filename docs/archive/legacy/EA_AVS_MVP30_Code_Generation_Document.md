# EA-AVS-MVP v3.0 代码生成指导文档

## 0. 文档用途

本文档用于指导大模型在 GitHub 仓库中生成 `ea_avs_mvp_v3` 版本代码。

v3.0 不是推翻 v2.0，而是在 v2.0 的“预测评分 pred / 到达后评估 true 分离”基础上，加入更贴近老人动作主动感知任务的三个核心能力：

1. **多姿态抽象人体骨架**：不再只有 standing，而是支持 standing、sitting、lying_fallen、bending。
2. **人体朝向建模**：每个 episode 具有 `human_yaw`，候选视角需要判断自己处于人体正面、侧面、背面还是侧前方。
3. **动作关键部位驱动评分**：不同动作使用不同身体部位权重，不再只优化一般关键点可见性。

一句话概括：

> v3.0 的目标是把 v2.0 的几何主动视角选择升级为“动作感知导向 + 人体朝向感知”的下一最佳观察位姿选择。

---

## 1. v3.0 的研究定位

v3.0 研究的是：

> 在已知地图、机器人当前位姿、人体目标位置、人体朝向和动作姿态先验的条件下，移动机器人如何从局部可达候选观察位姿中选择一个最有利于老人动作关键部位可见性的下一观察位姿。

v3.0 仍然是 one-shot 主动重观测，不做多步迭代导航。

流程是：

```text
当前位姿
  ↓
已知人体位置 + 姿态类型 + 人体朝向
  ↓
生成局部可达候选观察位姿
  ↓
移动前预测 Q_pred
  ↓
选择最佳位姿或保持当前视角
  ↓
渲染选中位姿
  ↓
计算 Q_true 进行评估
```

---

## 2. v3.0 与 v2.0 的核心区别

| 项目 | v2.0 | v3.0 |
|---|---|---|
| 人体姿态 | 仅 standing | standing / sitting / lying_fallen / bending |
| 人体朝向 | 无 | 新增 `human_yaw` |
| 骨架生成 | 只做平移 | 支持相对坐标旋转到人体朝向 |
| 评分核心 | 一般关键点可见性 `S_kp_pred` | 动作关键部位可见性 `S_action_part_pred` |
| 观察方向 | 不区分正面/背面/侧面 | 新增 `S_orient_pred` |
| 策略选择 | `Q_pred` | 包含动作部位和朝向的 `Q_pred` |
| 任务属性 | 几何视角选择 | 动作感知导向视角选择 |

v3.0 继续保留 v2.0 的重要约束：

1. 选择阶段禁止使用候选点未来图像；
2. 策略只能使用 `pred_score`；
3. `true_score` 只能在策略选择完成并渲染后计算；
4. Ours 必须允许选择 current view，即允许不移动。

---

## 3. v3.0 不做什么

v3.0 仍然不做以下内容：

1. 不接入 Habitat humanoid；
2. 不导入 AMASS / SMPL-X；
3. 不训练动作识别模型；
4. 不使用 OpenPose / RTMPose / YOLOv8-pose；
5. 不做强化学习；
6. 不做 ROS / Unity；
7. 不做真实机器人控制；
8. 不做多步迭代主动感知；
9. 不做老人情绪识别；
10. 不做大规模数据集生成。

v3.0 仍然是一个轻量级仿真实验框架。

---

## 4. v3.0 项目目录结构

请新建以下目录：

```text
ea_avs_mvp_v3/
├── configs/
│   └── mvp30_action_aware.yaml
├── scripts/
│   └── run_mvp30_action_aware.py
├── ea_avs_v3/
│   ├── __init__.py
│   ├── config.py
│   ├── habitat_runner.py
│   ├── geometry.py
│   ├── skeleton.py
│   ├── action_pose_library.py
│   ├── orientation.py
│   ├── action_part_weights.py
│   ├── candidate_sampler.py
│   ├── predictive_evaluator.py
│   ├── true_evaluator.py
│   ├── policies.py
│   ├── metrics.py
│   └── visualization.py
└── outputs/
```

说明：

- `action_pose_library.py`：定义多姿态骨架相对坐标；
- `orientation.py`：处理人体朝向和候选视角相对人体方向；
- `action_part_weights.py`：根据姿态类型返回动作关键部位权重；
- `predictive_evaluator.py`：新增 `S_action_part_pred` 和 `S_orient_pred`；
- `true_evaluator.py`：新增 `S_action_part_true` 和 `S_orient_true`；
- 其他模块可以从 v2.0 迁移，但必须适配 v3.0 字段。

---

## 5. 配置文件设计

文件路径：

```text
ea_avs_mvp_v3/configs/mvp30_action_aware.yaml
```

配置模板：

```yaml
project:
  name: "EA-AVS-MVP3.0"
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
  randomize_pose_type: true
  randomize_human_yaw: true

human:
  pose_types: ["standing", "sitting", "lying_fallen", "bending"]
  default_pose_type: "standing"
  base_height_offset: 0.0
  yaw_candidates_deg: [0, 45, 90, 135, 180, -135, -90, -45]

candidate_sampling:
  radii: [1.5, 2.0, 2.5]
  angles_deg: [-135, -90, -45, 0, 45, 90, 135, 180]
  min_distance_to_human: 1.2
  max_distance_to_human: 3.0
  max_geodesic_distance: 6.0

visibility:
  min_depth: 0.5
  max_depth: 5.0

# 默认关键点组权重，仅作为 fallback。
score_weights:
  torso: 0.4
  lower_body: 0.4
  head: 0.1
  arms: 0.1

# 不同姿态/动作状态对应的动作关键部位权重。
action_part_weights:
  standing:
    torso: 0.35
    lower_body: 0.35
    head: 0.20
    arms: 0.10
  sitting:
    torso: 0.40
    lower_body: 0.45
    head: 0.10
    arms: 0.05
  lying_fallen:
    torso: 0.40
    lower_body: 0.35
    head: 0.20
    arms: 0.05
  bending:
    torso: 0.45
    lower_body: 0.35
    head: 0.10
    arms: 0.10

# 候选视角相对人体朝向的评分配置。
orientation_score:
  preferred_angle_deg: 45
  sigma_deg: 35
  front_score: 0.8
  side_score: 0.7
  back_score: 0.2

predictive_score:
  w_action_part_pred: 0.45
  w_orient_pred: 0.20
  w_center_pred: 0.15
  w_dist_pred: 0.15
  w_move: 0.15

true_score:
  w_action_part_true: 0.50
  w_orient_true: 0.20
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

## 6. 核心数据结构

### 6.1 CandidateView

定义在：

```text
ea_avs_mvp_v3/ea_avs_v3/candidate_sampler.py
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

与 v2.0 相同，但 v3.0 的 `pred_score` 和 `true_score` 要包含动作和朝向相关字段。

### 6.2 EpisodeState

建议新增一个轻量结构用于记录 episode 级别的人体状态：

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class EpisodeState:
    episode_id: int
    pose_type: str
    human_pos: np.ndarray
    human_yaw: float
    robot_start_pos: np.ndarray
    robot_start_yaw: float
```

这个结构不是必须，但推荐实现，便于主流程传参。

---

## 7. 文件级实现说明

## 7.1 `config.py`

功能：读取并校验 YAML 配置。

必须实现：

```python
def load_config(path: str) -> dict:
    pass
```

必须校验字段：

```text
habitat.scene_path
camera.width
camera.height
human.pose_types
candidate_sampling.radii
candidate_sampling.angles_deg
action_part_weights
orientation_score.preferred_angle_deg
predictive_score.w_action_part_pred
true_score.w_action_part_true
```

要求：

1. 使用 `yaml.safe_load`；
2. 文件不存在时抛出 `FileNotFoundError`；
3. 必要字段缺失时抛出 `ValueError`；
4. 不写实验逻辑。

---

## 7.2 `action_pose_library.py`

功能：定义多种姿态的骨架相对坐标。

必须支持四种姿态：

```text
standing
sitting
lying_fallen
bending
```

建议定义：

```python
POSE_SKELETONS = {
    "standing": {...},
    "sitting": {...},
    "lying_fallen": {...},
    "bending": {...},
}
```

关键点至少包含：

```text
head
neck
pelvis
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
```

注意：

1. 相对坐标以人体局部坐标系定义；
2. 局部坐标系中，人体正面方向建议定义为 +Z；
3. 后续由 `skeleton.py` 根据 `human_yaw` 旋转到世界坐标系；
4. 这些姿态不要求生物力学完全真实，只要能表达动作状态差异。

建议简化姿态：

- standing：直立；
- sitting：髋部降低，膝盖前伸，小腿向下；
- lying_fallen：人体沿地面方向展开，关键点高度较低；
- bending：头和躯干前倾，髋部和膝部仍接近站立。

---

## 7.3 `skeleton.py`

功能：根据人体位置、姿态类型和人体朝向生成世界坐标骨架。

必须定义关键点分组：

```python
KEYPOINT_GROUPS = {
    "torso": ["neck", "pelvis", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "lower_body": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
    "head": ["head"],
    "arms": ["left_elbow", "right_elbow", "left_wrist", "right_wrist"],
}
```

必须实现：

```python
def rotate_local_point_by_yaw(local_point: np.ndarray, yaw: float) -> np.ndarray:
    pass


def get_skeleton(
    human_base_pos: np.ndarray,
    pose_type: str,
    human_yaw: float,
) -> dict:
    pass
```

实现逻辑：

1. 从 `POSE_SKELETONS[pose_type]` 读取局部坐标；
2. 对每个局部关键点绕 Y 轴旋转 `human_yaw`；
3. 加上 `human_base_pos` 得到世界坐标；
4. 返回 `{keypoint_name: world_pos}`。

---

## 7.4 `orientation.py`

功能：计算候选观察位姿相对人体朝向关系。

必须实现：

```python
def human_yaw_to_forward(human_yaw: float) -> np.ndarray:
    pass


def compute_view_direction_from_human(
    human_pos: np.ndarray,
    view_pos: np.ndarray,
) -> np.ndarray:
    pass


def compute_relative_view_angle(
    human_pos: np.ndarray,
    human_yaw: float,
    view_pos: np.ndarray,
) -> float:
    pass


def compute_orientation_score(
    relative_angle: float,
    config: dict,
) -> float:
    pass
```

角度定义：

- `human_yaw` 表示人体正面方向；
- `relative_angle = 0` 表示候选点位于人体正前方；
- `relative_angle = pi` 或 `-pi` 表示候选点位于人体正后方；
- `relative_angle ≈ ±pi/2` 表示候选点位于人体侧方。

推荐评分：

```text
侧前方 30°–60° 得分最高。
正前方次之。
侧方可接受。
背面较差。
```

可以用高斯函数实现：

```text
S_orient = exp(-((abs(relative_angle) - preferred_angle)^2) / (2 * sigma^2))
```

其中：

```text
preferred_angle = 45°
sigma = 35°
```

---

## 7.5 `action_part_weights.py`

功能：根据姿态类型返回动作关键部位权重。

必须实现：

```python
def get_action_part_weights(pose_type: str, config: dict) -> dict:
    pass
```

输出示例：

```python
{
    "torso": 0.40,
    "lower_body": 0.45,
    "head": 0.10,
    "arms": 0.05,
}
```

要求：

1. 优先读取 `config["action_part_weights"][pose_type]`；
2. 如果不存在，退回 `config["score_weights"]`；
3. 权重和应接近 1；
4. 如果权重和不是 1，可以归一化。

---

## 7.6 `geometry.py`

功能：保留 v2.0 几何工具函数。

必须实现：

```python
def normalize_angle(angle: float) -> float:
    pass


def yaw_to_forward(yaw: float) -> np.ndarray:
    pass


def compute_look_at_yaw(source_pos: np.ndarray, target_pos: np.ndarray) -> float:
    pass


def angle_in_camera_fov(...):
    pass


def gaussian_score(value: float, optimal: float, sigma: float) -> float:
    pass
```

坐标约定保持 v2.0：

- Y 轴向上；
- yaw = 0 表示朝向 +Z；
- yaw 正方向朝 +X 旋转。

---

## 7.7 `habitat_runner.py`

功能：可以从 v2.0 迁移。

必须保留：

```python
class HabitatRunner:
    def sample_navigable_point(self) -> np.ndarray: ...
    def is_navigable(self, point: np.ndarray) -> bool: ...
    def snap_point(self, point: np.ndarray) -> np.ndarray: ...
    def geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> float: ...
    def render_at(self, position: np.ndarray, yaw: float) -> dict: ...
    def close(self): ...
```

重要约束：

- `render_at()` 仍然只能在策略选择后调用；
- 不要在 `predictive_evaluator.py` 中调用 `render_at()`。

---

## 7.8 `candidate_sampler.py`

功能：可以从 v2.0 迁移。

必须实现：

```python
class CandidateSampler:
    def sample(self, human_pos: np.ndarray, robot_pos: np.ndarray, runner) -> list[CandidateView]:
        pass
```

要求：

1. 候选点围绕 `human_pos` 采样；
2. 使用 navmesh 过滤；
3. 使用 geodesic distance 过滤；
4. yaw 指向人体；
5. 不考虑人体朝向评分，人体朝向评分在 evaluator 中计算。

---

## 7.9 `predictive_evaluator.py`

功能：v3.0 核心模块，移动前预测动作感知导向视角质量。

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
        human_yaw: float,
        pose_type: str,
        human_skeleton: dict,
        geodesic_distance: float,
    ) -> dict:
        pass
```

必须输出字段：

```python
{
    "S_action_part_pred": float,
    "S_kp_pred": float,
    "S_orient_pred": float,
    "S_center_pred": float,
    "S_dist_pred": float,
    "C_move": float,
    "Q_pred": float,
    "relative_view_angle": float,
    "torso_visibility_pred": float,
    "lower_body_visibility_pred": float,
    "head_visibility_pred": float,
    "arms_visibility_pred": float,
    "visible_keypoints_pred": list[str],
    "invisible_keypoints_pred": list[str],
}
```

### 7.9.1 关键点可见性

同 v2.0，使用 FOV 几何判断每个关键点是否可见。

### 7.9.2 分组可见率

计算：

```text
torso_visibility_pred
lower_body_visibility_pred
head_visibility_pred
arms_visibility_pred
```

### 7.9.3 动作关键部位可见性

根据 `pose_type` 获取动作部位权重：

```python
weights = get_action_part_weights(pose_type, config)
```

计算：

```text
S_action_part_pred =
    weights["torso"] * torso_visibility_pred
  + weights["lower_body"] * lower_body_visibility_pred
  + weights["head"] * head_visibility_pred
  + weights["arms"] * arms_visibility_pred
```

### 7.9.4 朝向评分

调用：

```python
relative_angle = compute_relative_view_angle(human_base_pos, human_yaw, view_pos)
S_orient_pred = compute_orientation_score(relative_angle, config)
```

### 7.9.5 最终 Q_pred

```text
Q_pred = w_action_part_pred * S_action_part_pred
       + w_orient_pred * S_orient_pred
       + w_center_pred * S_center_pred
       + w_dist_pred * S_dist_pred
       - w_move * C_move
```

重要：

- 不允许使用候选点图像；
- 不允许调用 `render_at()`；
- 不允许使用 `true_score`。

---

## 7.10 `true_evaluator.py`

功能：策略选择并渲染后，计算真实评估指标。

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
        human_yaw: float,
        pose_type: str,
        human_skeleton: dict,
    ) -> dict:
        pass
```

v3.0 中，true evaluator 仍然可以使用几何 FOV 计算 true 指标，但必须在渲染之后调用。

必须输出：

```python
{
    "S_action_part_true": float,
    "S_kp_true": float,
    "S_orient_true": float,
    "S_center_true": float,
    "S_dist_true": float,
    "Q_true": float,
    "relative_view_angle_true": float,
    "torso_visibility_true": float,
    "lower_body_visibility_true": float,
    "head_visibility_true": float,
    "arms_visibility_true": float,
}
```

最终 Q_true：

```text
Q_true = w_action_part_true * S_action_part_true
       + w_orient_true * S_orient_true
       + w_center_true * S_center_true
       + w_dist_true * S_dist_true
```

注意：

- true 不包含移动代价，因为移动代价已经在 pred 的决策中考虑；
- 也可以额外输出 `C_move`，但不要重复计入 Q_true。

---

## 7.11 `policies.py`

可以从 v2.0 迁移。

必须保留四种策略：

```text
Fixed
Random
Nearest
Ours
```

要求：

1. Fixed：返回 current view；
2. Random：随机选有效候选点；
3. Nearest：选 geodesic distance 最短候选点；
4. Ours：在 current view 和候选点中选择 `Q_pred` 最大者。

Ours 只能使用：

```python
view.pred_score["Q_pred"]
```

不得使用 `true_score`。

---

## 7.12 `metrics.py`

v3.0 必须新增动作和朝向相关字段。

`metrics.csv` 必须包含：

```text
episode_id
scene_id
policy
status
num_candidates
selected_is_current
pose_type
human_yaw
relative_view_angle
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
S_action_part_pred
S_kp_pred
S_orient_pred
S_center_pred
S_dist_pred
C_move
Q_pred
torso_visibility_pred
lower_body_visibility_pred
head_visibility_pred
arms_visibility_pred
S_action_part_true
S_kp_true
S_orient_true
S_center_true
S_dist_true
Q_true
torso_visibility_true
lower_body_visibility_true
head_visibility_true
arms_visibility_true
pred_true_gap
action_part_gain_pred
action_part_gain_true
visibility_gain_pred
visibility_gain_true
```

字段说明：

- `pose_type`：当前动作/姿态类型；
- `human_yaw`：人体朝向；
- `relative_view_angle`：候选视角相对人体正面的角度；
- `S_action_part_pred`：动作关键部位预测可见性；
- `S_orient_pred`：朝向适配预测得分；
- `action_part_gain_true`：动作关键部位真实提升。

---

## 7.13 `visualization.py`

可以从 v2.0 迁移，但 debug JSON 必须保存：

```text
pose_type
human_yaw
relative_view_angle
S_action_part_pred
S_orient_pred
Q_pred
S_action_part_true
Q_true
selected_by
```

如果后续要画图，可以在 RGB 图像文件名中加入姿态类型：

```text
ep_000_bending_ours.png
```

---

## 7.14 `run_mvp30_action_aware.py`

v3.0 主入口。

运行命令：

```bash
python scripts/run_mvp30_action_aware.py \
  --config configs/mvp30_action_aware.yaml \
  --episodes 20 \
  --output-dir outputs/mvp30_test_run
```

主流程：

```text
1. 读取配置
2. 初始化 HabitatRunner / CandidateSampler / Evaluators / Policies
3. 对每个 episode：
   3.1 采样人体位置 human_pos
   3.2 采样 pose_type
   3.3 采样 human_yaw
   3.4 根据 human_pos + pose_type + human_yaw 生成 skeleton
   3.5 采样机器人初始位置 robot_start_pos
   3.6 构造 current_view
   3.7 对 current_view 计算 pred_score
   3.8 采样候选位姿
   3.9 对候选位姿计算 pred_score
   3.10 策略根据 pred_score 选择位姿
   3.11 渲染 current_view，计算 true_score
   3.12 渲染各策略选中位姿，计算 true_score
   3.13 写 metrics.csv
   3.14 写 episodes.jsonl
   3.15 保存 candidates debug json
4. 关闭资源
```

注意：

- 策略选择前禁止渲染候选点；
- 策略只能看 `Q_pred`；
- true score 只用于评估。

---

## 8. v3.0 主流程伪代码

```python
def run_one_episode(...):
    human_pos = sample_valid_human_position(runner, config)

    pose_type = sample_pose_type(config)
    human_yaw = sample_human_yaw(config)

    skeleton = get_skeleton(
        human_base_pos=human_pos,
        pose_type=pose_type,
        human_yaw=human_yaw,
    )

    robot_start_pos = sample_robot_start_position_around_human(...)
    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)

    current_view = CandidateView(...)

    current_view.pred_score = pred_evaluator.score_view_pred(
        view_pos=current_view.position,
        view_yaw=current_view.yaw,
        robot_start_pos=robot_start_pos,
        human_base_pos=human_pos,
        human_yaw=human_yaw,
        pose_type=pose_type,
        human_skeleton=skeleton,
        geodesic_distance=0.0,
    )

    candidates = sampler.sample(human_pos, robot_start_pos, runner)

    for cand in candidates:
        if cand.is_valid:
            cand.pred_score = pred_evaluator.score_view_pred(...)

    selected_by_policy = {}
    for policy in policies:
        selected_by_policy[policy.name] = policy.select(current_view, candidates)

    obs_current = runner.render_at(current_view.position, current_view.yaw)
    current_view.true_score = true_evaluator.score_view_true(...)

    for policy_name, selected in selected_by_policy.items():
        obs = runner.render_at(selected.position, selected.yaw)
        selected.true_score = true_evaluator.score_view_true(...)
        write_metric_row(...)

    write_episode_summary(...)
```

---

## 9. v3.0 验收标准

### 9.1 运行验收

必须能运行：

```bash
python scripts/run_mvp30_action_aware.py \
  --config configs/mvp30_action_aware.yaml \
  --episodes 20 \
  --output-dir outputs/mvp30_test_run
```

### 9.2 文件验收

必须生成：

```text
outputs/mvp30_test_run/metrics.csv
outputs/mvp30_test_run/episodes.jsonl
outputs/mvp30_test_run/images/
outputs/mvp30_test_run/debug/
```

### 9.3 字段验收

`metrics.csv` 必须包含：

```text
pose_type
human_yaw
relative_view_angle
S_action_part_pred
S_orient_pred
Q_pred
S_action_part_true
S_orient_true
Q_true
action_part_gain_pred
action_part_gain_true
```

### 9.4 逻辑验收

必须满足：

1. `pose_type` 至少覆盖 4 类姿态；
2. `human_yaw` 被实际用于 skeleton 旋转；
3. `S_orient_pred` 被实际用于 `Q_pred`；
4. `S_action_part_pred` 被实际用于 `Q_pred`；
5. Ours 只使用 `Q_pred`；
6. Ours 可以选择 current view；
7. true 指标不参与策略选择。

### 9.5 预期趋势

期望观察到：

```text
Ours-v3 的平均 S_action_part_true 高于 Fixed / Random / Nearest。
Ours-v3 在 sitting、lying_fallen、bending 等姿态上相比 visibility-only 更有优势。
```

---

## 10. v3.0 之后不建议立即做什么

v3.0 完成后，仍然不建议马上做：

1. 真实机器人；
2. 强化学习；
3. 大模型动作识别；
4. Unity 大工程；
5. 大规模真实数据采集。

更合理的下一步是：

| 后续版本 | 建议功能 |
|---|---|
| v3.1 | 加入 ray casting 遮挡预测 |
| v3.2 | 加入 visibility-only 与 orientation-only 消融 |
| v3.3 | 加入简单动作分类器，用 skeleton 可见性模拟动作识别质量 |
| v4.0 | 接入 Habitat humanoid 或真实多视角数据验证 |

---

## 11. 给大模型的最终提示词

```text
请根据 EA_AVS_MVP30_Code_Generation_Document.md 实现 ea_avs_mvp_v3。

v3.0 的核心目标：
在 v2.0 的 pred/true 分离基础上，加入多姿态 skeleton、人体朝向 human_yaw、动作关键部位权重和朝向感知评分，使主动视角选择从一般关键点可见性升级为动作感知导向的观察位姿选择。

重要约束：
1. 策略选择阶段禁止使用候选点未来图像。
2. 策略只能使用 pred_score，不能使用 true_score。
3. OursPolicy 必须允许选择 current view。
4. 不要实现强化学习、动作识别模型、真实 humanoid、ROS、Unity。
5. 所有代码必须有详细中文注释。

必须创建目录：
ea_avs_mvp_v3/
├── configs/mvp30_action_aware.yaml
├── scripts/run_mvp30_action_aware.py
├── ea_avs_v3/__init__.py
├── ea_avs_v3/config.py
├── ea_avs_v3/habitat_runner.py
├── ea_avs_v3/geometry.py
├── ea_avs_v3/skeleton.py
├── ea_avs_v3/action_pose_library.py
├── ea_avs_v3/orientation.py
├── ea_avs_v3/action_part_weights.py
├── ea_avs_v3/candidate_sampler.py
├── ea_avs_v3/predictive_evaluator.py
├── ea_avs_v3/true_evaluator.py
├── ea_avs_v3/policies.py
├── ea_avs_v3/metrics.py
└── ea_avs_v3/visualization.py

必须支持姿态：
standing, sitting, lying_fallen, bending

必须新增指标：
pose_type, human_yaw, relative_view_angle,
S_action_part_pred, S_orient_pred, Q_pred,
S_action_part_true, S_orient_true, Q_true,
action_part_gain_pred, action_part_gain_true

验收命令：
python scripts/run_mvp30_action_aware.py --config configs/mvp30_action_aware.yaml --episodes 20 --output-dir outputs/mvp30_test_run
```

---

## 12. 一句话总结

EA-AVS-MVP v3.0 的核心不是接入更复杂的真实人体模型，而是让视角选择真正服务于老人动作感知：

> 不只是看清“人”，而是看清“当前动作最关键的人体部位”，并优先选择更适合动作判断的人体相对观察方向。
