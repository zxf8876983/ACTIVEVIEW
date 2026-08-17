# EA-AVS-MVP v4.0 代码生成指导文档

## 0. 文档用途

本文档用于指导大模型在当前 GitHub 仓库中生成 `ea_avs_mvp_v4` 版本代码。

v4.0 不推翻 v3.0，而是在 v3.0 已实现的以下能力基础上继续演进：

1. 多姿态人体骨架：`standing / sitting / lying_fallen / bending`；
2. 人体朝向建模：`human_yaw`；
3. 动作关键部位评分：`S_action_part_pred`；
4. 人体相对观察方向评分：`S_orient_pred`；
5. v2.0 延续下来的 `pred / true` 严格分离；
6. Ours 只能使用 `Q_pred`，不能偷看候选点未来图像。

v4.0 的核心新增能力是：

> **让机器人不仅知道“关键点是否在相机视野中”，还要判断“关键点是否真的会被墙、桌子、沙发、柜子等环境物体遮挡”。**

因此，v4.0 的主题定义为：

> **遮挡感知的动作导向主动观察位姿选择**

英文建议：

> **Occlusion-Aware Action-Oriented Active Observation Pose Selection**

v4.0 重点增加四类能力：

1. Habitat 场景几何 ray casting；
2. 基于射线的关键点遮挡判断；
3. 遮挡感知动作关键部位评分；
4. 消融策略与 Oracle 上界。

---

## 1. v1.0 → v4.0 的演进关系

### v1.0：几何主动视角选择

解决：

> 机器人应该去哪儿看？

核心：

- 局部候选观察位姿生成；
- navmesh 可达性过滤；
- geodesic distance；
- FOV 关键点可见性；
- Fixed / Random / Nearest / Ours。

### v2.0：预测与真实评估严格分离

解决：

> 机器人不能提前看到候选点未来图像，应该如何科学地进行选择？

核心：

- `pred_score` 与 `true_score` 分离；
- 候选点选择阶段禁止调用 `render_at()`；
- Ours 只能使用 `Q_pred`；
- Ours 允许选择 current view，即允许不移动。

### v3.0：动作感知导向

解决：

> 什么样的观察方向真正有利于动作判断？

核心：

- 多姿态 skeleton；
- `human_yaw`；
- 动作关键部位权重；
- `S_action_part_pred`；
- `S_orient_pred`。

### v4.0：遮挡感知

解决：

> 这些对动作判断重要的身体部位，实际上能不能被机器人看到？

核心：

- ray casting；
- 环境遮挡；
- 遮挡后的动作关键部位可见性；
- 消融实验；
- Oracle upper bound。

---

## 2. v4.0 的研究定位

v4.0 研究的是：

> 在已知地图、机器人当前位姿、人体位置、人体朝向和动作姿态先验条件下，移动机器人如何结合候选位姿可达性、动作关键部位、人体相对观察方向和室内环境遮挡，从局部候选观察位姿中选择最有利于动作感知的下一观察位姿。

v4.0 仍然采用 one-shot 主动重观测：

```text
当前状态
  ↓
生成局部可达候选观察位姿
  ↓
预测候选位姿的动作关键部位 + 朝向 + 遮挡质量
  ↓
选择最佳观察位姿或保持当前位姿
  ↓
移动/渲染到选中位姿
  ↓
计算 true 指标
```

v4.0 不研究多步 POMDP / RL 导航。

---

## 3. v4.0 与 v3.0 的核心区别

| 项目 | v3.0 | v4.0 |
|---|---|---|
| FOV 可见性 | 有 | 有 |
| 人体朝向 | 有 | 有 |
| 动作关键部位 | 有 | 有 |
| 环境遮挡 | 无 | 新增 |
| Ray casting | 无 | 新增 |
| 遮挡感知动作部位得分 | 无 | 新增 |
| 消融策略 | 基础 baseline | 新增模块化消融 |
| Oracle 上界 | 无 | 新增 |

v4.0 的关键原则：

```text
in_fov ≠ visible
```

真正的关键点可见性应该满足：

```text
visible = in_fov AND not_occluded
```

---

## 4. v4.0 不做什么

v4.0 仍然不做：

1. 不接 Habitat humanoid；
2. 不使用 AMASS / SMPL-X；
3. 不训练真实动作识别网络；
4. 不使用 OpenPose / RTMPose / YOLO-pose；
5. 不做强化学习；
6. 不做 ROS / ROS2；
7. 不做 Unity；
8. 不做真实机器人控制；
9. 不做多步主动感知；
10. 不做老人情绪识别。

v4.0 的目标仍然是：

> 把“观察位姿选择方法”做扎实，而不是扩展成完整机器人系统。

---

## 5. 项目目录结构

请新建：

```text
ea_avs_mvp_v4/
├── configs/
│   └── mvp40_occlusion_aware.yaml
├── scripts/
│   └── run_mvp40_occlusion_aware.py
├── ea_avs_v4/
│   ├── __init__.py
│   ├── config.py
│   ├── habitat_runner.py
│   ├── geometry.py
│   ├── action_pose_library.py
│   ├── skeleton.py
│   ├── orientation.py
│   ├── action_part_weights.py
│   ├── candidate_sampler.py
│   ├── raycast_utils.py
│   ├── occlusion.py
│   ├── predictive_evaluator.py
│   ├── true_evaluator.py
│   ├── policies.py
│   ├── ablation_policies.py
│   ├── oracle_policy.py
│   ├── metrics.py
│   └── visualization.py
└── outputs/
```

新增核心文件：

```text
raycast_utils.py
occlusion.py
ablation_policies.py
oracle_policy.py
```

---

## 6. 配置文件

文件：

```text
ea_avs_mvp_v4/configs/mvp40_occlusion_aware.yaml
```

建议模板：

```yaml
project:
  name: "EA-AVS-MVP4.0"
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

occlusion:
  enabled: true
  ray_epsilon: 0.05
  target_tolerance: 0.08
  min_hit_distance: 0.05

score_weights:
  torso: 0.40
  lower_body: 0.40
  head: 0.10
  arms: 0.10

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

orientation_score:
  preferred_angle_deg: 45
  sigma_deg: 35

predictive_score:
  w_action_occ_pred: 0.50
  w_orient_pred: 0.15
  w_center_pred: 0.10
  w_dist_pred: 0.10
  w_move: 0.15

true_score:
  w_action_occ_true: 0.60
  w_orient_true: 0.15
  w_center_true: 0.10
  w_dist_true: 0.15

movement:
  allow_stay: true
  stay_margin: 0.0

oracle:
  enabled: true

output:
  save_images: true
  save_candidate_json: true
  save_csv: true
```

---

## 7. 核心数据结构 CandidateView

可沿用 v3.0：

```python
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

v4.0 不需要改变 CandidateView 的主体结构，但 `pred_score` / `true_score` 中必须新增遮挡字段。

---

## 8. `raycast_utils.py`

### 功能

封装 Habitat 场景几何射线检测。

必须实现：

```python
def cast_ray_to_point(
    runner,
    ray_origin: np.ndarray,
    target_point: np.ndarray,
    ray_epsilon: float = 0.05,
    target_tolerance: float = 0.08,
) -> dict:
    pass
```

输出：

```python
{
    "hit": bool,
    "occluded": bool,
    "hit_distance": float,
    "target_distance": float,
    "clearance": float,
}
```

推荐判断逻辑：

```text
target_distance = ||target_point - ray_origin||

如果射线在目标点之前击中场景 mesh：
    hit_distance < target_distance - target_tolerance
则：
    occluded = True
否则：
    occluded = False
```

### 重要要求

1. 使用 Habitat-Sim 官方 ray casting / ray query API；
2. Habitat API 版本差异必须封装在 `raycast_utils.py` 或 `habitat_runner.py` 中；
3. 不允许把 ray casting 写散在 evaluator 内；
4. 射线起点应从真实相机位置发出，而不是机器人底盘点。

相机位置：

```python
camera_pos = view_pos + np.array([0.0, camera_height, 0.0])
```

---

## 9. `occlusion.py`

### 功能

对整个人体骨架进行环境遮挡分析。

必须实现：

```python
def compute_keypoint_occlusion(
    runner,
    view_pos: np.ndarray,
    human_skeleton: dict,
    camera_height: float,
    config: dict,
) -> dict:
    pass
```

推荐输出：

```python
{
    "head": {
        "occluded": False,
        "hit_distance": 2.10,
        "target_distance": 2.14,
    },
    "left_knee": {
        "occluded": True,
        "hit_distance": 1.20,
        "target_distance": 2.00,
    },
}
```

还应实现统计函数：

```python
def compute_occlusion_rate(
    occlusion_result: dict,
    keypoint_names: list,
) -> float:
    pass
```

定义：

```text
occlusion_rate = 被环境遮挡关键点数 / 目标关键点总数
```

---

## 10. 遮挡感知关键点可见性

v3.0 使用：

```text
visible = in_fov
```

v4.0 必须改成：

```text
visible_occ = in_fov AND (NOT occluded)
```

对于每个关键点，需要同时保存：

```text
in_fov
occluded
visible_after_occlusion
```

重要：

> **环境遮挡应该直接影响动作关键部位可见性，而不是只作为一个与关键点可见性无关的附加惩罚项。**

---

## 11. `predictive_evaluator.py`

v4.0 的核心预测评估器。

必须实现：

```python
class PredictiveEvaluator:
    def score_view_pred(
        self,
        runner,
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

### 11.1 允许的信息

v4.0 选择阶段可以使用：

- 已知场景静态几何；
- ray casting；
- navmesh；
- 当前人体位置；
- 当前人体骨架先验；
- human_yaw；
- pose_type；
- camera model；
- geodesic distance。

### 11.2 禁止的信息

仍然禁止：

- 候选点未来 RGB；
- 候选点未来 depth image；
- 候选点真实人体检测结果；
- 候选点真实动作识别结果；
- `Q_true`。

### 11.3 必须输出

```python
{
    "S_action_occ_pred": float,
    "S_action_part_pred": float,
    "S_kp_occ_pred": float,
    "S_kp_pred": float,
    "S_orient_pred": float,
    "S_center_pred": float,
    "S_dist_pred": float,
    "C_move": float,
    "Q_pred": float,
    "occlusion_rate_pred": float,
    "occluded_keypoint_count_pred": int,
    "relative_view_angle": float,
    "torso_visibility_occ_pred": float,
    "lower_body_visibility_occ_pred": float,
    "head_visibility_occ_pred": float,
    "arms_visibility_occ_pred": float,
    "visible_keypoints_occ_pred": list[str],
    "occluded_keypoints_pred": list[str],
}
```

### 11.4 `S_action_occ_pred`

动作关键部位遮挡感知可见性：

```text
S_action_occ_pred =
  w_torso(action) * torso_visibility_occ_pred
+ w_lower(action) * lower_body_visibility_occ_pred
+ w_head(action) * head_visibility_occ_pred
+ w_arms(action) * arms_visibility_occ_pred
```

### 11.5 最终 `Q_pred`

推荐：

```text
Q_pred =
    w_action_occ_pred * S_action_occ_pred
  + w_orient_pred * S_orient_pred
  + w_center_pred * S_center_pred
  + w_dist_pred * S_dist_pred
  - w_move * C_move
```

不要重复再加一个很大的 `-w_occ * occlusion_rate`，否则遮挡可能被双重计权。

`occlusion_rate_pred` 主要用于分析和消融。

---

## 12. `true_evaluator.py`

v4.0 的 true evaluator 仍然必须在策略选择之后执行。

必须实现：

```python
class TrueEvaluator:
    def score_view_true(
        self,
        runner,
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

必须输出：

```python
{
    "S_action_occ_true": float,
    "S_action_part_true": float,
    "S_kp_occ_true": float,
    "S_kp_true": float,
    "S_orient_true": float,
    "S_center_true": float,
    "S_dist_true": float,
    "Q_true": float,
    "occlusion_rate_true": float,
    "occluded_keypoint_count_true": int,
    "torso_visibility_occ_true": float,
    "lower_body_visibility_occ_true": float,
    "head_visibility_occ_true": float,
    "arms_visibility_occ_true": float,
}
```

### 注意

v4.0 仍然没有真实 humanoid，因此 true 仍是仿真几何层面的真实评价。

但是相比 v3.0，true 应至少真实考虑 Habitat 场景 mesh 遮挡。

---

## 13. `habitat_runner.py`

从 v3.0 迁移，并增加 ray casting 相关统一接口。

建议新增：

```python
def cast_ray(
    self,
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
):
    pass
```

作用：

> 隔离 Habitat-Sim 不同版本 ray API，其他模块不要直接依赖 Habitat 版本细节。

---

## 14. 消融策略 `ablation_policies.py`

v4.0 需要正式加入模块消融。

必须至少实现以下策略：

### 14.1 VisibilityOnlyPolicy

只使用普通关键点/FOV 可见性：

```text
Q = S_kp_pred
```

用途：证明动作关键部位建模是否有效。

### 14.2 ActionPartOnlyPolicy

只使用：

```text
Q = S_action_part_pred
```

不考虑 orientation 和 occlusion。

### 14.3 OrientationOnlyPolicy

只使用：

```text
Q = S_orient_pred
```

### 14.4 OcclusionOnlyPolicy

只使用遮挡后的总体关键点可见性：

```text
Q = S_kp_occ_pred
```

### 14.5 ActionOrientationPolicy

模拟 v3.0：

```text
动作关键部位 + orientation + distance + movement
```

不使用环境遮挡。

### 14.6 FullOursPolicy

完整 v4.0：

```text
动作关键部位遮挡感知可见性
+ orientation
+ center
+ distance
- movement cost
```

---

## 15. Oracle 上界 `oracle_policy.py`

Oracle 只用于离线仿真性能上界。

必须明确：

> **Oracle 不是可部署策略，绝不能作为 Ours 的输入或训练标签在线使用。**

建议实现：

```python
class OraclePolicy:
    name = "Oracle"

    def select(self, current_view, candidates):
        # 使用 true_score 的 Q_true 选择最大者
        pass
```

### Oracle 调用顺序

普通策略必须先完成选择。

然后实验框架可以离线对所有候选位姿计算 true_score，用于：

```text
Oracle = argmax Q_true(candidate)
```

Oracle 只能在 evaluation phase 运行。

必须输出：

```text
oracle_Q_true
oracle_gap
```

定义：

```text
oracle_gap = oracle_Q_true - ours_Q_true
```

---

## 16. `policies.py`

保留：

```text
Fixed
Random
Nearest
Ours / FullOurs
```

普通 Ours 仍然只能使用 `Q_pred`。

必须允许：

```text
current_view
```

参加比较，即机器人可以选择不移动。

---

## 17. 主脚本 `run_mvp40_occlusion_aware.py`

运行命令：

```bash
python scripts/run_mvp40_occlusion_aware.py \
  --config configs/mvp40_occlusion_aware.yaml \
  --episodes 20 \
  --output-dir outputs/mvp40_test_run
```

主流程建议：

```text
1. 读取配置
2. 初始化 HabitatRunner
3. 初始化 CandidateSampler
4. 初始化 PredictiveEvaluator
5. 初始化 TrueEvaluator
6. 初始化 baseline / ablation policies
7. 初始化 Oracle（仅评估）

每个 episode：
8. 采样 human_pos
9. 采样 pose_type
10. 采样 human_yaw
11. 生成 skeleton
12. 采样 robot_start_pos
13. 构造 current_view
14. current_view 计算 pred_score（含 ray casting）
15. 生成候选位姿
16. 候选点计算 pred_score（允许静态地图 ray casting，不允许未来 RGB/depth）
17. Fixed / Random / Nearest / 消融 / Ours 根据 pred_score 完成选择
18. 渲染 current_view 和各普通策略选中的位姿
19. 计算 true_score
20. 为 Oracle 离线评估所有有效候选位姿 true_score
21. Oracle 根据 Q_true 选上界位姿
22. 写 metrics.csv
23. 写 episodes.jsonl
24. 保存候选点 debug JSON
```

最重要的约束：

> **步骤 17 之前禁止使用候选点 RGB / depth / Q_true。**

静态场景 ray casting 被视为“已知地图几何信息”，允许用于 pred 阶段。

---

## 18. Metrics 设计

`metrics.csv` 必须保留 v3.0 字段，并增加：

```text
occlusion_rate_pred
occlusion_rate_true
occluded_keypoint_count_pred
occluded_keypoint_count_true
S_action_occ_pred
S_action_occ_true
S_kp_occ_pred
S_kp_occ_true
torso_visibility_occ_pred
lower_body_visibility_occ_pred
head_visibility_occ_pred
arms_visibility_occ_pred
torso_visibility_occ_true
lower_body_visibility_occ_true
head_visibility_occ_true
arms_visibility_occ_true
occlusion_gain_true
oracle_Q_true
oracle_gap
```

建议：

```text
occlusion_gain_true =
S_action_occ_true(selected) - S_action_occ_true(current)
```

---

## 19. Debug JSON

每个候选点必须保存：

```text
candidate_id
position
yaw
is_valid
invalid_reason
geodesic_distance
pred_score
selected_by
```

其中 `pred_score` 至少包含：

```text
Q_pred
S_action_occ_pred
S_action_part_pred
S_orient_pred
occlusion_rate_pred
occluded_keypoints_pred
visible_keypoints_occ_pred
```

Oracle evaluation 可以单独保存：

```text
true_score
```

但要清楚区分 offline evaluation。

---

## 20. v4.0 需要修正的概念

### 20.1 不要把“FOV 内”称为真实可见

统一术语：

```text
in_fov：在视场角内
occluded：被环境遮挡
visible_after_occlusion：FOV 内且未遮挡
```

### 20.2 不要把 Oracle 当成可部署算法

论文和代码中必须标注：

```text
Oracle-NBV (offline upper bound)
```

### 20.3 不要使用未来图像进行候选选择

即使 Habitat 能通过 `render_at(candidate)` 直接获得候选图像，也绝对不能在 Ours 选择阶段这么做。

---

## 21. 验收标准

### 21.1 运行验收

```bash
python scripts/run_mvp40_occlusion_aware.py \
  --config configs/mvp40_occlusion_aware.yaml \
  --episodes 20 \
  --output-dir outputs/mvp40_test_run
```

### 21.2 文件验收

必须生成：

```text
outputs/mvp40_test_run/metrics.csv
outputs/mvp40_test_run/episodes.jsonl
outputs/mvp40_test_run/images/
outputs/mvp40_test_run/debug/
```

### 21.3 功能验收

必须满足：

1. ray casting 实际参与关键点遮挡判断；
2. `visible = in_fov and not_occluded`；
3. `S_action_occ_pred` 实际参与 `Q_pred`；
4. Ours 只使用 pred 信息；
5. Oracle 只在 evaluation phase 运行；
6. 至少有一类消融策略；
7. `metrics.csv` 中含遮挡和 Oracle 指标；
8. 单 episode 失败不能中断整体运行。

### 21.4 预期实验趋势

合理预期：

```text
Fixed / Random / Nearest
    <
VisibilityOnly / ActionPartOnly
    <
ActionOrientation（v3-like）
    <=
Full Ours v4
    <=
Oracle
```

不同场景未必每个 episode 都满足，但整体均值应体现趋势。

---

## 22. 推荐的消融实验

建议最终至少报告：

```text
A. Fixed
B. Random
C. Nearest
D. Visibility Only
E. Action Part Only
F. Action + Orientation (v3.0)
G. Action + Orientation + Occlusion (v4.0 Full)
H. Oracle upper bound
```

核心比较：

### 动作关键部位贡献

```text
Visibility Only vs Action Part Only
```

### 朝向贡献

```text
Action Part Only vs Action + Orientation
```

### 遮挡贡献

```text
Action + Orientation vs Full v4
```

---

## 23. v4.0 之后的建议

完成 v4.0 后，不建议立即扩展到 RL。

推荐下一步：

```text
v5.0：真实视觉层验证
```

可能包括：

1. Habitat humanoid；
2. 真实 RGB-D 人体图像；
3. 图像关键点检测；
4. 简单动作分类器；
5. 真实多视角数据验证。

v5.0 才开始回答：

> 几何与遮挡意义上的好视角，是否真的能提高视觉动作识别准确率？

---

## 24. 给大模型的最终代码生成提示词

```text
请根据 EA_AVS_MVP40_Code_Generation_Document.md 实现 ea_avs_mvp_v4。

总体目标：
在 v3.0 的动作关键部位 + 人体朝向主动视角选择基础上，加入 Habitat 静态场景 ray casting，使关键点可见性从“仅在 FOV 内”升级为“在 FOV 内且不被环境遮挡”，并增加模块消融策略和 Oracle offline upper bound。

最重要的原则：
1. Ours 选择阶段禁止使用候选点未来 RGB/depth。
2. 已知地图静态几何 ray casting 可以用于 pred 阶段。
3. 关键点真实预测可见性定义为 in_fov AND not_occluded。
4. S_action_occ_pred 必须参与 Q_pred。
5. Ours 只能使用 pred_score。
6. Oracle 只能在 evaluation phase 使用 true_score。
7. 所有代码必须有详细中文注释。
8. 不实现强化学习、真实 humanoid、ROS、Unity、真实动作识别模型。

必须创建：
ea_avs_mvp_v4/
├── configs/mvp40_occlusion_aware.yaml
├── scripts/run_mvp40_occlusion_aware.py
├── ea_avs_v4/__init__.py
├── ea_avs_v4/config.py
├── ea_avs_v4/habitat_runner.py
├── ea_avs_v4/geometry.py
├── ea_avs_v4/action_pose_library.py
├── ea_avs_v4/skeleton.py
├── ea_avs_v4/orientation.py
├── ea_avs_v4/action_part_weights.py
├── ea_avs_v4/candidate_sampler.py
├── ea_avs_v4/raycast_utils.py
├── ea_avs_v4/occlusion.py
├── ea_avs_v4/predictive_evaluator.py
├── ea_avs_v4/true_evaluator.py
├── ea_avs_v4/policies.py
├── ea_avs_v4/ablation_policies.py
├── ea_avs_v4/oracle_policy.py
├── ea_avs_v4/metrics.py
└── ea_avs_v4/visualization.py

验收命令：
python scripts/run_mvp40_occlusion_aware.py --config configs/mvp40_occlusion_aware.yaml --episodes 20 --output-dir outputs/mvp40_test_run

必须新增核心指标：
S_action_occ_pred
S_action_occ_true
occlusion_rate_pred
occlusion_rate_true
occluded_keypoint_count_pred
occluded_keypoint_count_true
occlusion_gain_true
oracle_Q_true
oracle_gap
```

---

## 25. 一句话总结

EA-AVS-MVP v4.0 的核心问题是：

> **机器人不仅要知道“哪个方向更适合看动作”，还必须知道“从那个方向是否真的能看到动作关键部位”。**

因此 v4.0 的核心升级就是：

> **Action-aware + Orientation-aware + Occlusion-aware Observation Pose Selection。**
