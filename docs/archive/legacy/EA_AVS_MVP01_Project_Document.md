# EA-AVS-MVP0.1 项目实现文档

## 0. 文档用途

本文档用于指导代码大模型实现一个**最小可运行版本**的移动机器人主动视角选择实验项目。

本项目不是完整养老机器人系统，也不是老人动作识别大模型项目。本项目第一阶段只验证一个核心流程：

> 在 Habitat 室内场景中，给定一个抽象人体骨架目标和一个移动机器人初始视角，系统能否围绕人体目标采样候选观察点，并选择一个使人体关键点更完整可见的观察视角。

该项目的核心目标是先把主动感知闭环跑通，后续再逐步加入真实 humanoid、遮挡判断、动作姿态和动作分类。

---

## 1. 项目名称

**EA-AVS-MVP0.1**

全称：

**Elderly Action Active View Selection - Minimal Viable Prototype 0.1**

中文名称：

**面向老人动作感知的移动机器人主动视角选择最小可运行版本**

---

## 2. MVP0.1 的核心目标

本版本只做以下事情：

1. 加载 Habitat 室内场景。
2. 在导航网格中采样一个人体目标位置。
3. 用一个抽象 3D 骨架表示人体目标。
4. 在人体目标周围采样一批候选观察点。
5. 用导航网格过滤不可达候选点。
6. 对每个候选视角计算几何可见性分数。
7. 比较四种策略：Fixed、Random、Nearest、Ours。
8. 输出 CSV、JSONL、候选点调试文件和渲染图像。
9. 验证 Ours 相比 Fixed 是否提升关键点可见率。

---

## 3. 本版本必须遵守的强约束

### 3.1 允许实现的内容

本版本允许实现：

- Habitat-Sim 或 Habitat-Lab simulator 的初始化；
- 室内 scene 加载；
- navmesh 可达性判断；
- agent 位姿设置；
- RGB / depth 图像渲染；
- 几何人体 skeleton；
- 候选视角采样；
- yaw 朝向计算；
- 视角几何评分；
- baseline policy 对比；
- CSV / JSONL / PNG 输出。

### 3.2 禁止实现的内容

本版本禁止实现：

1. 不做强化学习。
2. 不实现 PPO、DQN、SAC、A2C 等训练流程。
3. 不训练动作识别模型。
4. 不调用 OpenPose、RTMPose、MediaPipe Pose、YOLOv8-pose。
5. 不接 Unity。
6. 不接 ROS / ROS2。
7. 不导入 AMASS、SMPL-X、Motion-X 等真实人体动作数据。
8. 不实现真实 humanoid avatar。
9. 不实现真实跌倒动画。
10. 不注册新的 Habitat-Lab Task。
11. 不修改 Habitat-Lab 或 Habitat-Sim 官方源码。
12. 不写复杂多线程。
13. 不做大规模数据集生成。
14. 不引入大模型推理。
15. 不做老人情绪识别。

一句话：

> MVP0.1 只实现“几何骨架目标 + 候选视角采样 + 视角质量评分 + 策略对比”。

---

## 4. 项目目录结构

请严格按照以下目录结构实现：

```text
ea_avs_mvp/
├── configs/
│   └── mvp_visibility.yaml
├── scripts/
│   └── run_mvp_visibility.py
├── ea_avs/
│   ├── __init__.py
│   ├── config.py
│   ├── habitat_runner.py
│   ├── skeleton.py
│   ├── geometry.py
│   ├── candidate_sampler.py
│   ├── evaluator.py
│   ├── policies.py
│   ├── metrics.py
│   └── visualization.py
└── outputs/
```

---

## 5. 配置文件说明

文件路径：

```text
configs/mvp_visibility.yaml
```

### 5.1 配置文件完整模板

```yaml
project:
  name: "EA-AVS-MVP0.1"
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

distance_score:
  optimal_distance: 2.0
  sigma: 0.7

view_score:
  w_kp: 0.60
  w_center: 0.20
  w_dist: 0.20
  w_move: 0.20

output:
  save_images: true
  save_candidate_json: true
  save_csv: true
```

### 5.2 配置字段解释

| 字段 | 含义 |
|---|---|
| `project.seed` | 随机种子，保证实验可复现 |
| `habitat.scene_path` | Habitat 场景文件路径 |
| `habitat.navmesh_path` | navmesh 路径 |
| `camera.width` | 渲染图像宽度 |
| `camera.height` | 渲染图像高度 |
| `camera.hfov_deg` | 水平视场角 |
| `camera.vfov_deg` | 垂直视场角，MVP 中用于几何判断 |
| `camera.camera_height` | 移动机器人相机高度 |
| `episode.num_episodes` | 运行 episode 数量 |
| `candidate_sampling.radii` | 围绕人体采样候选点的半径 |
| `candidate_sampling.angles_deg` | 围绕人体采样候选点的角度 |
| `visibility.min_depth` | 关键点最小可见距离 |
| `visibility.max_depth` | 关键点最大可见距离 |
| `view_score` | 视角质量函数权重 |

---

## 6. 核心数据结构

### 6.1 `CandidateView`

定义位置：

```text
ea_avs/candidate_sampler.py
```

建议使用 `dataclass`：

```python
from dataclasses import dataclass, field
from typing import Dict, Optional
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
    score: Optional[Dict] = field(default_factory=dict)
```

字段说明：

| 字段 | 类型 | 含义 |
|---|---|---|
| `candidate_id` | int | 候选视角编号 |
| `position` | np.ndarray | 候选点 3D 坐标，shape=(3,) |
| `yaw` | float | 机器人朝向，单位 rad |
| `geodesic_distance` | float | 从机器人初始位置到候选点的测地距离 |
| `euclidean_distance_to_human` | float | 候选点到人体目标的欧氏距离 |
| `is_valid` | bool | 候选点是否有效 |
| `invalid_reason` | str | 无效原因 |
| `score` | dict | 视角评分结果 |

### 6.2 `ViewScore`

不一定需要单独定义类，可以使用 dict。

标准返回格式：

```python
{
    "S_kp": 0.75,
    "S_center": 0.83,
    "S_dist": 0.91,
    "C_move": 0.35,
    "Q": 0.68,
    "torso_visibility": 0.80,
    "lower_body_visibility": 0.67,
    "head_visibility": 1.00,
    "visible_keypoints": ["head", "neck", "pelvis"],
    "invisible_keypoints": ["left_ankle", "right_ankle"]
}
```

---

# 7. 文件级实现说明

## 7.1 `ea_avs/config.py`

### 功能

读取 YAML 配置文件，并返回 Python dict。

### 输入

- `path: str`：YAML 配置文件路径。

### 输出

- `config: dict`：配置字典。

### 必须实现的函数

```python
def load_config(path: str) -> dict:
    """
    Load YAML config file.

    Args:
        path: Path to yaml config.

    Returns:
        A dictionary containing config values.

    Raises:
        FileNotFoundError: if config file does not exist.
        ValueError: if required fields are missing.
    """
```

### 实现要求

1. 使用 `yaml.safe_load`。
2. 检查配置文件是否存在。
3. 检查必要字段是否存在：
   - `habitat.scene_path`
   - `camera.width`
   - `camera.height`
   - `candidate_sampling.radii`
   - `candidate_sampling.angles_deg`
4. 不要在该文件中初始化 Habitat。
5. 不要写实验逻辑。

---

## 7.2 `ea_avs/skeleton.py`

### 功能

生成抽象人体 3D 骨架关键点。

MVP0.1 中不使用真实 humanoid，只使用固定 standing skeleton。

### 输入

- `human_base_pos: np.ndarray`，人体底部位置，shape=(3,)。
- `pose_type: str`，目前只支持 `"standing"`。

### 输出

- `skeleton: dict[str, np.ndarray]`，关键点名称到 3D 世界坐标的映射。

### 必须实现的函数

```python
def get_standing_skeleton(human_base_pos: np.ndarray) -> dict:
    """
    Create a standing 3D skeleton.
    """
```

```python
def get_skeleton(human_base_pos: np.ndarray, pose_type: str = "standing") -> dict:
    """
    Return skeleton according to pose type.
    """
```

### 骨架定义

请使用以下相对坐标：

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

### 关键点分组

还需要定义：

```python
KEYPOINT_GROUPS = {
    "torso": [
        "neck", "pelvis",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip"
    ],
    "lower_body": [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ],
    "head": ["head"]
}
```

### 禁止事项

- 不要读取 SMPL-X。
- 不要导入 AMASS。
- 不要实现真实动作序列。
- 不要调用姿态估计模型。

---

## 7.3 `ea_avs/geometry.py`

### 功能

提供所有几何计算函数，包括 yaw 计算、角度归一化、FOV 判断、距离计算。

### 必须实现的函数

#### 7.3.1 `normalize_angle`

```python
def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi].
    """
```

#### 7.3.2 `yaw_to_forward`

```python
def yaw_to_forward(yaw: float) -> np.ndarray:
    """
    Convert yaw to forward vector in x-z plane.

    Coordinate assumption:
        - y is up.
        - yaw = 0 means facing +z direction.
        - positive yaw rotates toward +x.
    """
```

建议实现：

```python
forward = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
```

#### 7.3.3 `compute_look_at_yaw`

```python
def compute_look_at_yaw(source_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """
    Compute yaw angle so that source looks at target.
    """
```

建议实现：

```python
dx = target_pos[0] - source_pos[0]
dz = target_pos[2] - source_pos[2]
yaw = np.arctan2(dx, dz)
```

#### 7.3.4 `angle_in_camera_fov`

```python
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
    """
    Check whether a 3D point is inside the camera field of view.

    Returns:
        {
            "in_fov": bool,
            "horizontal_angle": float,
            "vertical_angle": float,
            "distance": float,
            "depth": float
        }
    """
```

### 实现逻辑

1. 相机位置：

```python
camera_pos = camera_base_pos + np.array([0.0, camera_height, 0.0])
```

2. 点相对相机向量：

```python
vec = point - camera_pos
```

3. 水平距离：

```python
horizontal_dist = sqrt(vec[0]**2 + vec[2]**2)
```

4. 深度距离：

```python
distance = norm(vec)
```

5. 计算点相对于相机朝向的水平角：

```python
point_yaw = atan2(vec[0], vec[2])
horizontal_angle = normalize_angle(point_yaw - camera_yaw)
```

6. 计算垂直角：

```python
vertical_angle = atan2(vec[1], horizontal_dist)
```

7. 判断：

```python
in_h = abs(horizontal_angle) <= radians(hfov_deg / 2)
in_v = abs(vertical_angle) <= radians(vfov_deg / 2)
in_d = min_depth <= distance <= max_depth
in_fov = in_h and in_v and in_d
```

#### 7.3.5 `gaussian_score`

```python
def gaussian_score(value: float, optimal: float, sigma: float) -> float:
    """
    Compute Gaussian-shaped score in [0, 1].
    """
```

---

## 7.4 `ea_avs/habitat_runner.py`

### 功能

封装 Habitat-Sim 或 Habitat-Lab 的初始化、导航点采样和图像渲染。

MVP0.1 中不在主脚本直接调用 Habitat API，所有 Habitat 相关逻辑都通过本文件封装。

### 核心类

```python
class HabitatRunner:
    def __init__(self, config: dict):
        ...
```

### 必须实现的方法

#### 7.4.1 `__init__`

```python
def __init__(self, config: dict):
    """
    Initialize Habitat simulator.

    Attributes:
        self.sim: Habitat simulator instance.
        self.pathfinder: Habitat pathfinder.
        self.config: config dict.
    """
```

#### 实现要求

1. 从配置中读取：
   - `scene_path`
   - `navmesh_path`
   - camera 参数
2. 初始化 simulator。
3. 加载 navmesh。
4. 暴露 `self.pathfinder`。
5. Habitat API 版本差异必须封装在本文件内。

#### 7.4.2 `sample_navigable_point`

```python
def sample_navigable_point(self) -> np.ndarray:
    """
    Sample a random navigable point from navmesh.
    """
```

优先使用：

```python
self.pathfinder.get_random_navigable_point()
```

#### 7.4.3 `is_navigable`

```python
def is_navigable(self, point: np.ndarray) -> bool:
    """
    Check whether a point is navigable.
    """
```

#### 7.4.4 `snap_point`

```python
def snap_point(self, point: np.ndarray) -> np.ndarray:
    """
    Snap point to nearest navigable point on navmesh.
    """
```

#### 7.4.5 `geodesic_distance`

```python
def geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> float:
    """
    Compute geodesic distance on navmesh.
    Return np.inf if no path exists.
    """
```

#### 7.4.6 `render_at`

```python
def render_at(self, position: np.ndarray, yaw: float) -> dict:
    """
    Render RGB/depth observation at a given robot position and yaw.

    Returns:
        {
            "rgb": np.ndarray,
            "depth": np.ndarray or None,
            "position": np.ndarray,
            "yaw": float
        }
    """
```

### MVP 简化

MVP0.1 的视角评分不依赖渲染图像，因此即使 `render_at` 只返回 RGB，也不影响主评分流程。但必须保存图像用于调试。

---

## 7.5 `ea_avs/candidate_sampler.py`

### 功能

围绕人体目标生成候选观察点，并过滤不可达点。

### 核心类

```python
class CandidateSampler:
    def __init__(self, config: dict):
        ...
```

### 必须实现的方法

```python
def sample(
    self,
    human_pos: np.ndarray,
    robot_pos: np.ndarray,
    runner: HabitatRunner,
) -> list[CandidateView]:
    """
    Sample valid candidate viewpoints around the human target.
    """
```

### 输入

- `human_pos`
- `robot_pos`
- `runner`
- `config`

### 输出

- `list[CandidateView]`

### 采样规则

从配置读取：

```yaml
candidate_sampling:
  radii: [1.5, 2.0, 2.5]
  angles_deg: [-135, -90, -45, 0, 45, 90, 135, 180]
```

对每个 `r, theta`：

```python
candidate_x = human_pos[0] + r * cos(theta)
candidate_z = human_pos[2] + r * sin(theta)
candidate_y = human_pos[1]
candidate = np.array([candidate_x, candidate_y, candidate_z])
```

然后：

1. `snapped = runner.snap_point(candidate)`
2. 判断 `runner.is_navigable(snapped)`
3. 计算 `geo = runner.geodesic_distance(robot_pos, snapped)`
4. 计算 `euclidean_distance_to_human`
5. 过滤：
   - `euclidean_distance_to_human < min_distance_to_human`
   - `euclidean_distance_to_human > max_distance_to_human`
   - `geo == inf`
   - `geo > max_geodesic_distance`
6. 计算 yaw：

```python
yaw = compute_look_at_yaw(snapped, human_pos)
```

7. 保存有效 CandidateView。

### 注意事项

- 无效候选点可以不返回，但建议在 debug JSON 中保存无效原因。
- 如果所有候选点无效，主脚本应写 failed episode，而不是崩溃。
- 不要在本文件中计算视角质量分数。

---

## 7.6 `ea_avs/evaluator.py`

### 功能

计算某个视角下的骨架关键点可见性和最终视角质量分数。

### 核心类

```python
class ViewpointEvaluator:
    def __init__(self, config: dict):
        ...
```

### 必须实现的方法

```python
def score_view(
    self,
    view_pos: np.ndarray,
    view_yaw: float,
    robot_start_pos: np.ndarray,
    human_base_pos: np.ndarray,
    human_skeleton: dict,
    geodesic_distance: float,
) -> dict:
    """
    Score a viewpoint.
    """
```

### 评分流程

#### Step 1：判断每个关键点是否在 FOV 内

对 skeleton 中每个 keypoint 调用：

```python
angle_in_camera_fov(...)
```

得到：

- 是否在视野内；
- 水平角；
- 垂直角；
- 距离。

#### Step 2：计算分组可见性

使用 `KEYPOINT_GROUPS`：

```python
torso_visibility = visible torso keypoints / total torso keypoints
lower_body_visibility = visible lower_body keypoints / total lower_body keypoints
head_visibility = visible head keypoints / total head keypoints
```

#### Step 3：计算加权关键点可见性

```python
S_kp = (
    torso_weight * torso_visibility
    + lower_body_weight * lower_body_visibility
    + head_weight * head_visibility
)
```

默认权重：

```yaml
score_weights:
  torso: 0.4
  lower_body: 0.4
  head: 0.2
```

#### Step 4：计算居中得分 `S_center`

对所有关键点的水平角取平均绝对值：

```python
mean_abs_angle = mean(abs(horizontal_angle_to_visible_or_all_joints))
center_error = mean_abs_angle / radians(hfov_deg / 2)
S_center = 1 - clamp(center_error, 0, 1)
```

建议使用所有关键点，而不仅是可见关键点，避免不可见点被忽略。

#### Step 5：计算距离得分 `S_dist`

候选视角到人体目标的欧氏距离：

```python
dist_to_human = np.linalg.norm(view_pos - human_base_pos)
```

使用高斯得分：

```python
S_dist = exp(-((dist_to_human - optimal_distance) ** 2) / (2 * sigma ** 2))
```

默认：

```yaml
distance_score:
  optimal_distance: 2.0
  sigma: 0.7
```

#### Step 6：计算移动代价 `C_move`

```python
C_move = geodesic_distance / max_geodesic_distance
C_move = clamp(C_move, 0, 1)
```

如果是 Fixed 视角：

```python
geodesic_distance = 0
C_move = 0
```

#### Step 7：计算最终得分 `Q`

```python
Q = (
    w_kp * S_kp
    + w_center * S_center
    + w_dist * S_dist
    - w_move * C_move
)
```

默认：

```yaml
view_score:
  w_kp: 0.60
  w_center: 0.20
  w_dist: 0.20
  w_move: 0.20
```

### 输出字段

必须返回：

```python
{
    "S_kp": float,
    "S_center": float,
    "S_dist": float,
    "C_move": float,
    "Q": float,
    "torso_visibility": float,
    "lower_body_visibility": float,
    "head_visibility": float,
    "visible_keypoints": list[str],
    "invisible_keypoints": list[str],
}
```

---

## 7.7 `ea_avs/policies.py`

### 功能

实现不同视角选择策略。

### 输入

- 当前视角 `current_view`
- 候选视角列表 `candidates`

### 输出

- 被选中的 `CandidateView` 或当前视角对象。

### 7.7.1 FixedPolicy

```python
class FixedPolicy:
    name = "Fixed"

    def select(self, current_view, candidates):
        return current_view
```

要求：

- 不使用 candidates。
- 返回 current_view。
- current_view 需要包含 position、yaw、geodesic_distance=0、score。

### 7.7.2 RandomPolicy

```python
class RandomPolicy:
    name = "Random"

    def __init__(self, seed: int = 42):
        ...

    def select(self, current_view, candidates):
        ...
```

要求：

- 从有效 candidates 中随机选一个。
- 使用固定 seed 以保证可复现。
- 如果 candidates 为空，返回 current_view 并标记 fallback。

### 7.7.3 NearestPolicy

```python
class NearestPolicy:
    name = "Nearest"

    def select(self, current_view, candidates):
        ...
```

要求：

- 选择 `geodesic_distance` 最小的候选点。
- 如果 candidates 为空，返回 current_view。

### 7.7.4 OursPolicy

```python
class OursPolicy:
    name = "Ours"

    def select(self, current_view, candidates):
        ...
```

要求：

- 选择 `candidate.score["Q"]` 最大的候选点。
- 如果 candidates 为空，返回 current_view。
- 不要在策略里重新计算分数，分数应由 evaluator 预先写入 candidate.score。

---

## 7.8 `ea_avs/metrics.py`

### 功能

负责结果记录、CSV 写入、JSONL 写入。

### 必须实现的类

```python
class MetricsWriter:
    def __init__(self, output_dir: str):
        ...

    def write_metric_row(self, row: dict):
        ...

    def write_episode_summary(self, summary: dict):
        ...

    def close(self):
        ...
```

### `metrics.csv` 字段

必须包含以下字段：

```text
episode_id
scene_id
policy
status
num_candidates
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
S_kp
S_center
S_dist
C_move
Q
torso_visibility
lower_body_visibility
head_visibility
```

每个成功 episode 应输出 4 行：

```text
Fixed
Random
Nearest
Ours
```

### `episodes.jsonl` 字段

成功 episode 示例：

```json
{
  "episode_id": 0,
  "scene_id": "example_scene",
  "status": "success",
  "human_base_pos": [0.0, 0.0, 0.0],
  "robot_start_pos": [1.0, 0.0, 2.0],
  "valid_candidate_count": 18,
  "fixed_score": 0.41,
  "ours_score": 0.82,
  "ours_improved": true
}
```

失败 episode 示例：

```json
{
  "episode_id": 3,
  "scene_id": "example_scene",
  "status": "failed",
  "reason": "no_valid_candidates",
  "human_base_pos": [0.0, 0.0, 0.0],
  "robot_start_pos": [1.0, 0.0, 2.0],
  "valid_candidate_count": 0
}
```

---

## 7.9 `ea_avs/visualization.py`

### 功能

保存渲染图像和调试图像。

### 必须实现的函数

```python
def save_rgb_image(rgb: np.ndarray, path: str):
    """
    Save RGB image to png.
    """
```

```python
def draw_skeleton_projection_placeholder(rgb: np.ndarray, score: dict) -> np.ndarray:
    """
    Optional placeholder.
    MVP0.1 does not require real 2D projection drawing.
    This function can simply return the original rgb.
    """
```

```python
def save_candidate_debug_json(candidates: list, path: str):
    """
    Save candidate positions, distances, scores, and validity into JSON.
    """
```

### 输出图片命名

每个 episode 至少保存：

```text
ep_000_current.png
ep_000_fixed.png
ep_000_random.png
ep_000_nearest.png
ep_000_ours.png
```

如果 Fixed 和 current 相同，也仍然保存一张 fixed 图，方便后续自动对比。

---

## 7.10 `scripts/run_mvp_visibility.py`

### 功能

项目主入口，负责串联所有模块并运行实验。

### 命令行接口

必须支持：

```bash
python scripts/run_mvp_visibility.py \
  --config configs/mvp_visibility.yaml \
  --episodes 20 \
  --output-dir outputs/test_run
```

### 参数说明

| 参数 | 含义 |
|---|---|
| `--config` | YAML 配置路径 |
| `--episodes` | 覆盖配置中的 episode 数量 |
| `--output-dir` | 输出目录 |

### 主流程伪代码

```python
def main():
    args = parse_args()
    config = load_config(args.config)

    if args.episodes is not None:
        config["episode"]["num_episodes"] = args.episodes

    set_random_seed(config["project"]["seed"])

    runner = HabitatRunner(config)
    sampler = CandidateSampler(config)
    evaluator = ViewpointEvaluator(config)
    metrics_writer = MetricsWriter(args.output_dir)

    policies = [
        FixedPolicy(),
        RandomPolicy(seed=config["project"]["seed"]),
        NearestPolicy(),
        OursPolicy(),
    ]

    for episode_id in range(num_episodes):
        try:
            run_one_episode(...)
        except Exception as e:
            write_failed_episode(...)
            continue

    metrics_writer.close()
```

### 单个 episode 详细流程

```python
def run_one_episode(episode_id, config, runner, sampler, evaluator, policies, output_dir):
    human_pos = sample_valid_human_position(runner, config)

    skeleton = get_skeleton(
        human_base_pos=human_pos,
        pose_type=config["human"]["pose_type"]
    )

    robot_start_pos = sample_robot_start_position_around_human(
        runner=runner,
        human_pos=human_pos,
        config=config
    )

    robot_start_yaw = compute_look_at_yaw(robot_start_pos, human_pos)

    current_view = CandidateView(
        candidate_id=-1,
        position=robot_start_pos,
        yaw=robot_start_yaw,
        geodesic_distance=0.0,
        euclidean_distance_to_human=euclidean_distance(robot_start_pos, human_pos),
        is_valid=True,
        invalid_reason="",
        score={}
    )

    current_view.score = evaluator.score_view(
        view_pos=current_view.position,
        view_yaw=current_view.yaw,
        robot_start_pos=robot_start_pos,
        human_base_pos=human_pos,
        human_skeleton=skeleton,
        geodesic_distance=0.0,
    )

    candidates = sampler.sample(
        human_pos=human_pos,
        robot_pos=robot_start_pos,
        runner=runner,
    )

    if len(candidates) == 0:
        write_failed_episode(...)
        return

    for cand in candidates:
        cand.score = evaluator.score_view(
            view_pos=cand.position,
            view_yaw=cand.yaw,
            robot_start_pos=robot_start_pos,
            human_base_pos=human_pos,
            human_skeleton=skeleton,
            geodesic_distance=cand.geodesic_distance,
        )

    for policy in policies:
        selected = policy.select(current_view, candidates)
        obs = runner.render_at(selected.position, selected.yaw)
        save_rgb_image(obs["rgb"], image_path)
        write_metric_row(...)

    save_candidate_debug_json(candidates, debug_path)
    write_episode_summary(...)
```

### 辅助函数要求

#### `sample_valid_human_position`

```python
def sample_valid_human_position(runner: HabitatRunner, config: dict) -> np.ndarray:
    """
    Sample a navigable point as human base position.

    Requirements:
        - Use runner.sample_navigable_point().
        - Try at most max_sampling_tries.
        - Return a valid point.
        - Raise RuntimeError if failed.
    """
```

#### `sample_robot_start_position_around_human`

```python
def sample_robot_start_position_around_human(
    runner: HabitatRunner,
    human_pos: np.ndarray,
    config: dict
) -> np.ndarray:
    """
    Sample a navigable robot start point around human.

    Requirements:
        - Distance to human should be within [min_robot_human_distance, max_robot_human_distance].
        - Point must be navigable.
        - Geodesic path from robot to human should exist.
        - Try at most max_sampling_tries.
    """
```

---

## 8. 评分函数数学定义

### 8.1 关键点可见性

给定一组关键点 \(K\)，每个关键点 \(k_i\) 的可见性：

\[
vis(k_i)=
\begin{cases}
1, & \text{if keypoint is inside camera FOV and valid depth} \\
0, & \text{otherwise}
\end{cases}
\]

分组可见率：

\[
V_g=\frac{1}{|K_g|}\sum_{k_i \in K_g} vis(k_i)
\]

加权关键点可见率：

\[
S_{kp}=0.4V_{torso}+0.4V_{lower}+0.2V_{head}
\]

### 8.2 居中得分

\[
S_{center}=1-\text{clip}\left(
\frac{\text{mean}(|\theta_i|)}{HFOV/2},0,1
\right)
\]

其中 \(\theta_i\) 是关键点相对于相机中心方向的水平角。

### 8.3 距离得分

\[
S_{dist}=\exp\left(
-\frac{(d-d_0)^2}{2\sigma^2}
\right)
\]

默认：

\[
d_0=2.0,\quad \sigma=0.7
\]

### 8.4 移动代价

\[
C_{move}=\text{clip}\left(
\frac{d_{geo}}{d_{max}},0,1
\right)
\]

默认：

\[
d_{max}=6.0
\]

### 8.5 最终视角质量

\[
Q(v)=0.60S_{kp}+0.20S_{center}+0.20S_{dist}-0.20C_{move}
\]

---

## 9. 输出文件要求

运行命令：

```bash
python scripts/run_mvp_visibility.py \
  --config configs/mvp_visibility.yaml \
  --episodes 20 \
  --output-dir outputs/test_run
```

运行完成后必须生成：

```text
outputs/test_run/
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

## 10. 验收标准

### 10.1 运行验收

必须能运行：

```bash
python scripts/run_mvp_visibility.py \
  --config configs/mvp_visibility.yaml \
  --episodes 20 \
  --output-dir outputs/test_run
```

### 10.2 文件验收

必须生成：

```text
metrics.csv
episodes.jsonl
images/
debug/
```

### 10.3 数据验收

20 个 episodes，每个成功 episode 有 4 个 policy 结果。

预期：

```text
metrics.csv 行数 >= 56
```

原因：

- 20 个 episodes；
- 成功率至少 70%；
- 每个成功 episode 4 行；
- 20 × 0.7 × 4 = 56。

### 10.4 指标验收

必须满足：

```text
mean(S_kp_ours) > mean(S_kp_fixed)
mean(Q_ours) > mean(Q_fixed)
```

如果不满足，需要检查：

1. 候选点 yaw 是否朝向人体；
2. FOV 判断是否正确；
3. 机器人起始位置是否本来就总是最优；
4. 候选点是否都无效；
5. `S_center` 是否计算反了；
6. `C_move` 权重是否过大。

### 10.5 健壮性验收

不允许出现：

- 程序因某个 episode 失败而整体崩溃；
- `metrics.csv` 为空；
- 所有 candidate 都无效；
- 所有 `S_kp` 都是 0；
- 所有 `S_kp` 都是 1；
- 图片保存失败但程序不提示；
- 输出目录没有创建。

---

## 11. 常见错误和处理方式

### 11.1 没有有效候选点

处理方式：

1. 当前 episode 写入 `episodes.jsonl`：
   - `status = failed`
   - `reason = no_valid_candidates`
2. 不写四种 policy 的 metrics 行。
3. 继续下一个 episode。

### 11.2 Habitat 渲染失败

处理方式：

1. 记录 warning。
2. 对该 policy 的图像保存跳过。
3. metrics 仍然写入。
4. 不要让整个程序崩溃。

### 11.3 geodesic distance 无法计算

处理方式：

1. 该候选点标记无效。
2. `invalid_reason = "no_geodesic_path"`。
3. 不参与策略选择。

### 11.4 YAML 配置缺字段

处理方式：

1. 抛出明确错误。
2. 错误信息说明缺失字段名称。
3. 不要静默使用 None。

---

## 12. 后续版本路线

### MVP0.2：多静态姿态

加入：

```text
standing
sitting
lying_fallen
bending
```

仍然使用几何 skeleton，不用 humanoid。

### MVP0.3：遮挡判断

加入 depth 或 ray casting 判断关键点是否被墙体、家具遮挡。

关键点可见性从：

```text
in_fov
```

升级为：

```text
in_fov and not_occluded
```

### MVP0.4：轻量动作分类

输入：

```text
2D keypoints + visibility mask
```

输出：

```text
standing / sitting / lying_fallen / bending
```

模型：

```text
MLP 或 LSTM
```

### MVP1.0：接入 Habitat humanoid

替换几何 skeleton 为 Habitat humanoid：

1. 读取 humanoid joint transform；
2. 获取 humanoid mask；
3. 使用真实 humanoid avatar；
4. 使用短 motion 片段；
5. 生成更真实的机器人第一视角图像。

---

## 13. 给代码大模型的执行提示词

下面这段可以直接复制给代码大模型：

```text
请根据以下项目文档实现 EA-AVS-MVP0.1。

项目目标：
实现一个基于 Habitat 的最小可运行主动视角选择实验。不要做完整老人动作识别系统，只做几何骨架目标的主动视角选择。

严格禁止：
1. 不要实现强化学习。
2. 不要训练动作识别模型。
3. 不要接入 OpenPose、RTMPose、MediaPipe、YOLOv8-pose。
4. 不要接入 Unity。
5. 不要接入 ROS。
6. 不要实现真实 humanoid。
7. 不要导入 AMASS、SMPL-X 动作序列。
8. 不要修改 Habitat 官方源码。

必须实现的目录：
ea_avs_mvp/
├── configs/mvp_visibility.yaml
├── scripts/run_mvp_visibility.py
├── ea_avs/__init__.py
├── ea_avs/config.py
├── ea_avs/habitat_runner.py
├── ea_avs/skeleton.py
├── ea_avs/geometry.py
├── ea_avs/candidate_sampler.py
├── ea_avs/evaluator.py
├── ea_avs/policies.py
├── ea_avs/metrics.py
└── ea_avs/visualization.py

必须实现的核心功能：
1. 读取 YAML 配置。
2. 初始化 Habitat-Sim 或封装 Habitat simulator。
3. 从 navmesh 采样 human position。
4. 用 standing skeleton 表示人体目标。
5. 采样 robot start position，并让 yaw 朝向人体。
6. 围绕人体按 radii 和 angles 采样候选观察点。
7. 对候选点进行 snap_point、is_navigable、geodesic_distance 过滤。
8. 对 current view 和所有 candidate view 计算视角质量。
9. 实现 Fixed、Random、Nearest、Ours 四种策略。
10. 渲染并保存每种策略选中视角的 RGB 图像。
11. 输出 metrics.csv、episodes.jsonl 和 candidates debug json。
12. 程序遇到单个 episode 失败时不要崩溃，应记录失败并继续。

核心评分函数：
Q(v) = 0.60*S_kp + 0.20*S_center + 0.20*S_dist - 0.20*C_move

其中：
- S_kp = 0.4*torso_visibility + 0.4*lower_body_visibility + 0.2*head_visibility
- S_center = 1 - normalized mean horizontal angle error
- S_dist = Gaussian distance score with optimal distance 2.0m and sigma 0.7
- C_move = normalized geodesic movement distance

运行命令：
python scripts/run_mvp_visibility.py --config configs/mvp_visibility.yaml --episodes 20 --output-dir outputs/test_run

验收标准：
1. 程序能跑完。
2. 生成 metrics.csv。
3. 生成 episodes.jsonl。
4. 生成 images/ 和 debug/。
5. 成功 episode 不少于 14 个。
6. metrics.csv 行数不少于 56。
7. mean(S_kp_ours) > mean(S_kp_fixed)。
8. mean(Q_ours) > mean(Q_fixed)。

优先保证流程跑通，不要扩展额外功能。
```

---

## 14. 一句话总结

EA-AVS-MVP0.1 的目标不是证明最终创新，而是先验证：

> Habitat 中能否建立“人体目标—机器人视角—候选观察点—可见性评分—主动选择”的最小闭环。

只要这个闭环跑通，后面再逐步加入遮挡、动作姿态、动作分类和 humanoid，研究路线才不会跑偏。
