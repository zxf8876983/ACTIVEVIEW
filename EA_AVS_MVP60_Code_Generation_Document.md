# EA-AVS-MVP v6.0 代码生成指导文档

## 0. 文档用途

本文档用于指导大模型在现有 `ACTIVEVIEW` 项目中开发 `ea_avs_mvp_v6`。

v5.0 已经完成了从“抽象人体骨架”到“真实 Habitat Humanoid”的过渡：场景中存在真实 articulated Humanoid，机器人能够获得包含人体的 RGB / Depth / Semantic 观测，并且现有 GT-State one-shot NBV 已经具备 target-surface-aware 环境/人体遮挡建模、current-vs-candidate 决策和严格的信息边界。

v6.0 的核心任务不再是继续完善 Humanoid，而是解决当前系统最大的部署性缺口：

> **在线 NBV 选择不能继续直接读取 Humanoid GT position / yaw / skeleton，而应只根据当前机器人 RGB-D 观测估计人体状态，再用 Estimated Human State 完成候选观察位姿选择。**

一句话概括：

> **v6.0 = v5.0 的真实 Humanoid RGB-D 仿真基础 + 当前 RGB-D 人体状态估计前端 + Estimated-State one-shot NBV。**

v6.0 是从 **GT-State Active Perception** 进入 **Estimated-State Active Perception** 的关键版本。

---

# 1. v1.0-v6.0 的研究演进位置

项目主线：

```text
v1.0
几何候选观察位姿 + 可达性 + 移动代价
        ↓
v2.0
Pred / True 分离，禁止候选未来图像泄漏
        ↓
v3.0
动作关键部位 + 人体朝向
        ↓
v4.0
环境遮挡 Ray Casting + Occlusion-aware NBV
        ↓
v5.0
真实 Habitat Humanoid + RGB-D/Semantic + Humanoid GT skeleton
+ target-surface-aware self-occlusion
        ↓
v6.0
Current RGB-D
→ 2D Pose
→ Depth Lifting
→ Estimated Human State
→ Estimated-State NBV
        ↓
后续版本
Current action hypothesis / uncertainty
+ missing-evidence recovery
+ actual action-recognition gain
```

v6.0 不是最终动作识别系统。

它只解决：

> **机器人如何从“当前真正看到的 RGB-D”得到 NBV 所需要的人体几何状态。**

---

# 2. v6.0 的核心科学与工程问题

v5.0 在线选择器仍然可以直接使用：

```text
GT human position
GT human yaw
GT 3D skeleton
known pose type (standing-only experiment)
```

这条链路适合验证 NBV 几何机制，但不具备真实部署意义。

v6.0 必须变为：

```text
Current RGB
    ↓
2D human pose estimator
    ↓
2D keypoints + confidence
    +
Current Depth
    ↓
keypoint depth sampling
    ↓
3D lifting
    ↓
Observed 3D joints
    ↓
human center / body scale / body orientation
visible joints / missing joints
proxy full skeleton
    ↓
EstimatedHumanState
    ↓
Candidate generation
    ↓
Estimated-State Predictive Evaluator
    ↓
Estimated-State Ours
```

因此 v6.0 主要回答：

1. 当前 RGB 中是否能够检测到 Humanoid 的 2D 人体关键点；
2. RGB 与 Depth 是否能够在同一相机模型下正确对齐；
3. 可见关键点是否能够被稳定提升到相机/世界 3D 坐标；
4. 是否能够仅依赖当前 RGB-D 估计 human position / body yaw / visible-missing joints；
5. 部分关键点缺失时，是否能够构造一个明确标注置信度和来源的 proxy skeleton；
6. 使用 Estimated Human State 后，NBV 是否仍能正确运行；
7. Estimated-State NBV 相比 GT-State NBV 损失多少；
8. 整条 Estimated-State 在线链路是否完全不读取 Humanoid GT。

---

# 3. v6.0 的研究边界

## 3.1 本版本必须做

```text
当前 RGB-D 获取
2D pose inference
2D keypoint confidence filtering
visible / missing joint 判断
depth sampling
2D→3D lifting
camera→world transform
estimated human center
estimated body yaw
estimated body scale
partial observed 3D skeleton
proxy full skeleton / template completion
EstimatedHumanState
Estimated-State candidate sampling
Estimated-State predictive evaluation
Estimated-State Ours
GT-State vs Estimated-State 对比
state estimation accuracy evaluation
no-GT-leakage audit
```

## 3.2 本版本明确不做

v6.0 不做：

1. 不训练新的 2D Pose 网络；
2. 不提出新的 pose-estimation SOTA 方法；
3. 不训练动作识别 HAR 网络；
4. 不从当前图像预测动作类别；
5. 不加入动作不确定性建模；
6. 不正式加入 missing-evidence recovery 新评分项；
7. 不加入时序动作识别；
8. 不做多帧 tracking；
9. 不做多步 NBV / RL；
10. 不做 SLAM / 定位 / 路径规划算法创新；
11. 不做 ROS；
12. 不做真实机器人；
13. 不更换 Humanoid 基础设施；
14. 不继续重构 v5.0 self-occlusion；
15. 不把 Semantic Humanoid mask 作为 Estimated-State 在线输入；
16. 不把 Humanoid GT skeleton/yaw/position 作为 Estimated-State fallback。

本版本的研究边界是：

> **用现成视觉人体姿态模型建立轻量、可解释的 Current RGB-D → Estimated Human State 前端，并验证它能否替代 GT State 驱动现有 one-shot NBV。**

---

# 4. v6.0 最重要的信息边界

这是本版本最高优先级约束。

## 4.1 Estimated-State Ours 在选择前允许使用

```text
current RGB
current Depth
current robot pose
camera intrinsics / extrinsics
known map
navmesh
current RGB-D 估计出的人体状态
scene ray casting
```

## 4.2 Estimated-State Ours 在选择前禁止使用

```text
Humanoid GT position
Humanoid GT yaw
Humanoid GT skeleton / joint transforms
Humanoid semantic mask
Humanoid semantic id pixels
future candidate RGB
future candidate Depth
future candidate Semantic
future candidate detected keypoints
future candidate action result
Q_true
Oracle result
```

## 4.3 GT 允许出现的位置

GT 只允许用于：

```text
simulator scenario construction
state-estimation accuracy evaluation
GT-State privileged baseline
post-selection true evaluation
Oracle-NBV offline upper bound
visual/debug comparison
```

必须明确区分：

> **GT 可用于实验评价，但不能进入 Estimated-State Ours 的在线输入。**

---

# 5. v6.0 三条实验支路必须分开命名

v6.0 中禁止把不同“上界”混在一起。

## 5.1 EstimatedState-Ours —— 本版本主方法

```text
Current RGB-D
→ Estimated Human State
→ Q_pred_est
→ select NBV
```

这是 v6.0 的主要可部署方向。

## 5.2 GTState-Ours —— privileged-state baseline

```text
Humanoid GT state
→ v5.0 Q_pred_gt
→ select NBV
```

它不是部署策略，只用于回答：

> **如果人体状态估计完全准确，现有 NBV 能做到多好？**

不要把它叫 Oracle。

## 5.3 Oracle-NBV —— future-view offline upper bound

```text
selection finished
→ render all candidate future observations
→ Q_true
→ select best true candidate
```

它回答：

> **如果未来候选真实观测全部已知，理论上可以选到多好？**

因此三者必须写为：

```text
EstimatedState-Ours
GTState-Ours
Oracle-NBV
```

禁止概念混淆。

---

# 6. v6.0 为什么主实验继续 standing-only

v6.0 建议继续只使用：

```text
standing
```

原因：

本版本的实验变量应尽可能只增加一个：

> **GT human state → Estimated human state**

如果同时加入：

```text
sitting
bending
lying_fallen
action classifier
action uncertainty
new action files
```

将很难判断性能变化到底来自：

```text
pose estimation
3D lifting
orientation estimation
action recognition
motion resource
```

中的哪一部分。

因此 v6.0 主实验中：

```text
pose_type = "standing"
```

可以作为实验场景常量使用。

这不是逐 episode 偷读 GT action label，因为整个 v6.0 benchmark 只包含 standing。

动作类别估计留到后续版本。

---

# 7. 2D Pose Estimator 的定位

v6.0 不研究 2D pose 网络本身。

应采用成熟现成模型，并通过统一 adapter 接入。

建议设计：

```python
class PoseBackend:
    def infer(self, rgb: np.ndarray) -> list[Pose2DDetection]:
        ...
```

后端可以是：

```text
RTMPose / MMPose
YOLO-Pose
其它成熟 COCO-keypoint detector
```

但代码不能和某一个具体库深度耦合。

实现前必须检查本机实际 Python/CUDA/torch 环境和已安装 API，不允许根据旧博客编造接口。

v6.0 推荐只选择一个默认 backend 把主流程跑通，同时保留 adapter 接口。

---

# 8. 建议统一使用 COCO-17 作为 2D 输入 schema

常见 2D pose 后端输出 COCO-17：

```text
nose
left_eye / right_eye
left_ear / right_ear
left_shoulder / right_shoulder
left_elbow / right_elbow
left_wrist / right_wrist
left_hip / right_hip
left_knee / right_knee
left_ankle / right_ankle
```

v5.0 NBV 使用的是 15-keypoint schema：

```text
head
neck
pelvis
left_shoulder / right_shoulder
left_elbow / right_elbow
left_wrist / right_wrist
left_hip / right_hip
left_knee / right_knee
left_ankle / right_ankle
```

v6.0 增加 `keypoint_schema.py` 做统一映射。

建议：

```text
head   = nose 或可靠 head proxy
neck   = midpoint(left_shoulder, right_shoulder)
pelvis = midpoint(left_hip, right_hip)
```

neck / pelvis 必须标记：

```text
source = derived_2d
```

不要伪装成 detector 直接预测点。

---

# 9. 2D 检测结果数据结构

建议新增：

```python
@dataclass
class Keypoint2D:
    name: str
    u: float
    v: float
    confidence: float
    detected: bool
    source: str

@dataclass
class Pose2DDetection:
    keypoints: dict[str, Keypoint2D]
    bbox_xyxy: tuple[float, float, float, float] | None
    score: float
    backend_name: str
```

如果 pose backend 检出多人：

v6.0 当前仿真只有一个 Humanoid，选择规则可以是：

```text
highest valid pose score
+ reasonable bbox size
```

或者最大 bbox / 最高平均 keypoint confidence。

但禁止使用 semantic id=100 去告诉在线前端“哪个检测框是真人”。

Semantic 只能评估。

---

# 10. visible / missing joint 定义

每个 2D keypoint 至少需要两个层级的状态：

## 10.1 2D visible

```text
confidence >= min_keypoint_confidence
AND
pixel inside image
```

## 10.2 3D observable

```text
2D visible
AND
depth sampling valid
AND
depth within camera range
AND
depth local consistency acceptable
```

因此：

```text
2D detected != 3D observable
```

最终 EstimatedHumanState 应明确记录：

```text
visible_2d_keypoints
observable_3d_keypoints
missing_keypoints
```

不要只保留一个 `visible=True/False`。

---

# 11. Depth sampling

重点新增：

```text
depth_lifter.py
```

对于每一个高置信度 2D keypoint `(u, v)`：

1. 首先检查中心像素 depth；
2. 中心无效时使用小邻域，例如 3×3 / 5×5；
3. 只使用 `depth > 0` 且在合法范围的像素；
4. 使用 robust median，而不是最小值；
5. 输出局部 depth spread / MAD 等稳定性指标；
6. depth 波动过大时应标记该 joint 3D 不可靠；
7. 禁止用 Humanoid semantic mask 帮助选择人体 depth。

建议结构：

```python
@dataclass
class DepthSample:
    valid: bool
    depth_m: float | None
    valid_pixel_count: int
    patch_size: int
    spread: float | None
    reason: str | None
```

---

# 12. 2D → 3D lifting

使用相机内参：

\[
\mathbf{X}_c = d K^{-1}
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
\]

其中：

```text
(u,v) = 当前 RGB keypoint 像素
 d    = 当前 aligned depth
 K    = Habitat camera intrinsics
```

然后通过当前相机外参：

\[
\mathbf{X}_w = T_{wc}\mathbf{X}_c
\]

得到世界坐标关键点。

必须复用 v5.0 已校准的相机坐标约定，不允许单独再写另一套 yaw/+Z/-Z 公式。

推荐新增统一接口：

```python
def backproject_keypoint_to_world(
    u,
    v,
    depth,
    intrinsics,
    camera_pose,
) -> np.ndarray:
    ...
```

---

# 13. Estimated 3D joint 数据结构

建议：

```python
@dataclass
class EstimatedJoint3D:
    name: str
    position_world: np.ndarray | None
    position_camera: np.ndarray | None
    confidence_2d: float
    depth_valid: bool
    observable_3d: bool
    source: str
    uncertainty: float | None
```

`source` 至少区分：

```text
observed_rgbd
derived_rgbd
template_completion
missing
```

这样后续 NBV 和实验分析可以知道某个 joint 是真正看到的，还是模板补出来的。

---

# 14. Estimated Human Position

v6.0 candidate sampling 不能再使用 GT human position。

必须从当前 RGB-D 估计：

```text
estimated_human_position
```

推荐优先级：

```text
pelvis 3D 可用
→ pelvis

否则 left/right hip 中心可用
→ hip midpoint

否则 torso joints 多数可用
→ robust torso center

否则全体可靠 3D joints
→ robust median/center
```

输出：

```text
human_position_world
human_position_confidence
human_position_source
```

如果无法得到可靠人体位置：

> **Estimated-State policy 必须失败/安全 stay，不允许偷偷回退 GT human position。**

---

# 15. Estimated Body Scale

需要估计人体尺度，支持 canonical skeleton 对齐。

可使用：

```text
shoulder width
hip width
shoulder-to-hip torso length
```

多个尺度指标可用时采用 robust aggregation。

输出：

```text
body_scale
body_scale_confidence
```

如果尺度无法估计，可使用一个固定 generic adult template scale 作为明确的低置信度 fallback，但必须记录：

```text
scale_source = default_template
```

不能使用 Humanoid GT body scale。

---

# 16. Estimated Human Yaw

v6.0 的关键模块之一是：

```text
orientation_estimator.py
```

优先使用可靠的左右肩/髋 3D 几何关系。

例如：

```text
right_shoulder → left_shoulder
right_hip → left_hip
```

构造 body lateral axis，再结合世界竖直轴得到 body forward axis。

需要特别注意：

1. 左/右必须使用 detector 的 anatomical labels；
2. shoulder 与 hip 两组结果应做一致性检查；
3. 若只剩一组 bilateral pair，降低 yaw confidence；
4. 若左右关系退化或深度不可靠，则 yaw invalid；
5. forward-axis 的符号/固定偏移只能通过一次性 calibration 确认，不能逐 episode 读取 GT yaw 修正；
6. 允许在开发阶段使用 GT yaw 评估误差和确定固定 convention，但部署路径不得读取 GT。

输出：

```text
estimated_yaw_rad
yaw_confidence
yaw_source
```

---

# 17. Orientation Calibration

新增：

```text
scripts/calibrate_estimated_orientation.py
```

流程：

```text
多个人体 yaw 场景
→ render current RGB-D
→ Estimated yaw
→ compare GT yaw (evaluation only)
→ determine one fixed sign / offset convention
→ write config
```

配置例如：

```yaml
human_state_estimation:
  orientation:
    yaw_offset_deg: 0.0
    forward_sign: 1
```

一旦确定后：

```text
Estimated-State runtime
```

只能使用固定配置，不能再看 GT yaw。

---

# 18. Partial Skeleton 与 Proxy Full Skeleton 必须分开

当前 RGB-D 可能只看到人体一部分。

因此 v6.0 不应该假装获得完整真实 skeleton。

必须同时保存：

```text
observed_skeleton
proxy_full_skeleton
```

## observed_skeleton

只包含：

```text
真正由 current RGB-D 得到的 3D joints
```

## proxy_full_skeleton

用于后续候选几何预测。

构建原则：

```text
canonical standing skeleton
+ estimated center
+ estimated yaw
+ estimated body scale
+ observed joints 约束
```

缺失 joint 可以使用 template completion，但必须：

```text
source = template_completion
confidence < observed_rgbd
```

禁止：

```text
直接读取 Humanoid GT missing joint 坐标补全
```

---

# 19. Skeleton Completion 的定位

新增：

```text
skeleton_completion.py
```

v6.0 不训练神经网络进行 skeleton completion。

采用轻量、可解释方案即可：

```text
canonical standing skeleton
→ scale
→ rotate by estimated yaw
→ translate to estimated human center
→ replace available template joints with observed 3D joints
```

必要时可做简单刚性/相似变换拟合。

本版本重点是：

> **建立一个不使用 GT 的完整几何 proxy，供 NBV 预测未来视角。**

不要把 completion 精度包装成科研创新。

---

# 20. EstimatedHumanState 数据结构

建议新增：

```python
@dataclass
class EstimatedHumanState:
    valid: bool

    human_position_world: np.ndarray | None
    human_position_confidence: float

    human_yaw: float | None
    yaw_confidence: float

    body_scale: float | None
    body_scale_confidence: float

    joints: dict[str, EstimatedJoint3D]
    observed_skeleton: dict[str, np.ndarray]
    proxy_full_skeleton: dict[str, np.ndarray]

    visible_2d_keypoints: list[str]
    observable_3d_keypoints: list[str]
    missing_keypoints: list[str]

    pose_detection_score: float
    state_confidence: float

    failure_reason: str | None
```

同时提供：

```python
state.to_dict()
```

用于 debug JSON / metrics。

---

# 21. EstimatedHumanState 的有效性规则

配置建议：

```yaml
human_state_estimation:
  min_pose_score: 0.30
  min_keypoint_confidence: 0.30
  min_2d_keypoints: 6
  min_3d_keypoints: 4
  min_torso_keypoints_3d: 2
  require_human_position: true
  require_orientation: true
```

具体阈值后续根据 debug 数据调整。

重要原则：

> **状态无效时，不允许 GT fallback。**

若 state invalid：

```text
EstimatedState-Ours
→ safe stay / episode marked perception_invalid
```

同时记录失败原因。

可以让 GTState-Ours baseline 正常继续计算，但不能用于替 EstimatedState-Ours 做决定。

---

# 22. v6.0 的 State Estimator

新增：

```text
human_state_estimator.py
```

统一接口：

```python
class HumanStateEstimator:
    def estimate(
        self,
        rgb,
        depth,
        camera_state,
    ) -> EstimatedHumanState:
        ...
```

内部流程：

```text
RGB
→ PoseBackend
→ COCO17
→ EA-AVS 15-keypoint schema
→ confidence filter
→ DepthSampler
→ 3D lifting
→ human center
→ body scale
→ human yaw
→ observed skeleton
→ template completion
→ state validation
```

注意：

接口参数中禁止出现：

```text
gt_human_pos
gt_human_yaw
gt_skeleton
humanoid_manager
semantic mask
```

这是防止 GT 泄漏的结构性保护。

---

# 23. Candidate Sampling 必须改用 estimated human position

v5.0：

```python
sampler.sample(human_pos=gt_human_pos, ...)
```

v6.0 Estimated-State 路径必须：

```python
sampler.sample(
    human_pos=estimated_state.human_position_world,
    ...
)
```

如果 estimated human position 有误，candidate ring 自然会出现偏差。

这正是 v6.0 应评估的真实误差传播之一。

禁止为了让候选更漂亮而：

```text
estimated position + GT correction
```

---

# 24. Estimated-State Predictive Evaluator

建议新增：

```text
estimated_predictive_evaluator.py
```

或在 v5 evaluator 外加严格 adapter。

输入只能是：

```text
EstimatedHumanState
robot current state
candidate pose
known map/raycast
```

核心映射：

```text
human_base_pos
← estimated_state.human_position_world

human_yaw
← estimated_state.human_yaw

human_skeleton
← estimated_state.proxy_full_skeleton
```

v6.0 为了隔离“状态估计误差”的影响，建议保持 v5.0 Q_pred 主公式不变：

\[
Q_{pred}=w_aS_{action-occ}+w_oS_{orient}+w_cS_{center}+w_dS_{dist}-w_mC_{move}
\]

本版本不要同时新增新评分项。

这样实验可以清晰回答：

> **完全相同的 NBV 几何评分，在 GT State 和 Estimated State 下性能下降多少？**

---

# 25. v6.0 暂时不正式加入 Evidence Recovery Score

当前 RGB-D 已经可以得到：

```text
visible joints
missing joints
```

这为后续研究：

```text
S_recover(v)
```

提供了基础。

但是 v6.0 首版建议只记录：

```text
current_missing_keypoints
candidate_predicted_recovered_keypoints
missing_joint_recovery_debug
```

不要立即把它加入 Q_pred。

原因：

> v6.0 首先需要建立 GT-State → Estimated-State 的干净对照。

Evidence Recovery 可以作为 v6.x / 后续研究增强项单独做消融。

---

# 26. v6.0 项目目录建议

新建：

```text
ea_avs_mvp_v6/
├── configs/
│   └── mvp60_estimated_state.yaml
├── scripts/
│   ├── smoke_test_pose_backend.py
│   ├── debug_current_rgbd_pose.py
│   ├── debug_depth_lifting.py
│   ├── calibrate_estimated_orientation.py
│   ├── compare_gt_estimated_state.py
│   ├── test_no_gt_leakage.py
│   └── run_mvp60_estimated_state.py
├── ea_avs_v6/
│   ├── __init__.py
│   ├── config.py
│   ├── habitat_runner.py
│   ├── pose_backend.py
│   ├── keypoint_schema.py
│   ├── depth_lifter.py
│   ├── orientation_estimator.py
│   ├── skeleton_completion.py
│   ├── estimated_human_state.py
│   ├── human_state_estimator.py
│   ├── state_validation.py
│   ├── candidate_sampler.py
│   ├── predictive_evaluator.py
│   ├── estimated_predictive_evaluator.py
│   ├── true_evaluator.py
│   ├── policies.py
│   ├── oracle_policy.py
│   ├── metrics.py
│   └── visualization.py
└── outputs/
```

v5.0 原目录保持封版，不要原地继续修改。

优先：

```text
copy stable v5 modules
+ add perception adapters
```

而不是全部重写。

---

# 27. 配置文件建议

文件：

```text
ea_avs_mvp_v6/configs/mvp60_estimated_state.yaml
```

建议：

```yaml
project:
  name: "EA-AVS-MVP6.0"
  seed: 42

perception:
  pose_backend: "rtmpose"   # 可替换，实际实现前检查本机环境
  device: "cuda:0"
  model_path: null
  min_pose_score: 0.30
  min_keypoint_confidence: 0.30
  max_people: 1

human_state_estimation:
  min_2d_keypoints: 6
  min_3d_keypoints: 4
  min_torso_keypoints_3d: 2

  depth_patch_size: 5
  depth_min_m: 0.3
  depth_max_m: 8.0
  max_depth_spread_m: 0.5

  require_human_position: true
  require_orientation: true

  orientation:
    yaw_offset_deg: 0.0
    forward_sign: 1
    min_bilateral_pair_confidence: 0.30

  skeleton_completion:
    enabled: true
    template: "standing_generic"
    allow_gt_fallback: false

estimated_state_policy:
  invalid_state_behavior: "stay"

oracle:
  enabled: true
  min_depth_coverage: 0.8
```

v5.0 的 habitat / camera / humanoid / candidate / score 配置优先继承，不要重复发明不同坐标约定。

---

# 28. 运行主流程

v6.0 正式 episode 流程建议：

```text
[Simulator setup only]
Place Humanoid using GT scenario state
Sample robot initial pose
        ↓
Render CURRENT RGB-D only
        ↓
===============================
Estimated-State Online Path
===============================
Current RGB
→ PoseBackend
→ 2D keypoints

Current Depth
→ depth sampling

2D + Depth
→ observed 3D joints
→ estimated human position
→ estimated body yaw
→ estimated body scale
→ observed/missing joints
→ proxy full skeleton
→ EstimatedHumanState
        ↓
state.valid ?
    ↓ yes
sample candidates around ESTIMATED human position
        ↓
Q_pred_est using estimated state
        ↓
EstimatedState-Ours select
        ↓
ONLINE SELECTION END
===============================
        ↓
render selected / evaluation candidates
        ↓
Q_true
        ↓
GT state accuracy evaluation
GTState-Ours baseline comparison
Oracle-NBV offline upper bound
        ↓
metrics
```

---

# 29. 注意 current RGB-D 本身是允许的

v2-v5 的规则是：

> 禁止“未来候选观测”在选择前泄漏。

这不等于禁止 current observation。

v6.0 正确规则是：

```text
Current RGB-D：必须在选择前使用
Future candidate RGB-D：选择前绝对禁止使用
```

否则无法建立真实视觉 active perception。

---

# 30. Simulator Setup 与 Policy Input 必须区分

为了构建受控仿真实验，Simulator 仍可使用 GT：

```text
放置 Humanoid
设置 Humanoid yaw
设置 standing pose
采样初始场景
```

这些属于：

```text
scenario generation
```

不是 policy input。

建议所有 GT 变量统一命名：

```text
gt_human_pos
gt_human_yaw
gt_skeleton
```

所有估计变量统一命名：

```text
est_human_pos
est_human_yaw
est_skeleton
estimated_state
```

禁止使用含糊变量名 `human_pos` 在 v6 主脚本里混用两种来源。

---

# 31. 建议增加结构性 No-GT-Leakage Guard

新增：

```text
scripts/test_no_gt_leakage.py
```

至少静态/接口检查：

```text
HumanStateEstimator.estimate()
不得接收 gt_* 参数

EstimatedPredictiveEvaluator.score_view_pred()
不得接收 gt_* 参数

EstimatedStateOurs.select()
只能读取 pred_score_est
```

主脚本建议把 Estimated-State 路径单独封装：

```python
def run_estimated_state_selection(
    current_obs,
    robot_state,
    map_context,
):
    ...
```

函数作用域内不传入 GT 变量。

这是比靠注释更可靠的信息边界。

---

# 32. 2D Pose Accuracy 评价

Humanoid GT skeleton 可以投影到当前图像，作为离线评价参考。

但只能在 estimated selection 完成后或者独立 debug script 中使用。

建议指标：

```text
2D keypoint detection count
2D keypoint recall
mean 2D pixel error on matched keypoints
PCK-style normalized error（可选）
```

注意：

如果 GT projection 本身被人体表面/环境遮挡，它不一定应该作为“必须检测成功”的点。

因此 v6.0 首版更重要的是：

```text
detected / observable / missing 状态统计
```

而不是追求一个看起来很漂亮但定义含糊的 PCK。

---

# 33. 3D State Accuracy 评价

GT 只用于 evaluation。

至少计算：

## Human position error

\[
e_{pos}=\|\hat p_h-p_h^{GT}\|_2
\]

单位：m。

## Yaw error

使用 circular angular difference：

\[
e_{yaw}=|wrap(\hat\theta-\theta^{GT})|
\]

单位：degree。

## Observed joint 3D error

只比较：

```text
source = observed_rgbd / derived_rgbd
```

不要把 template completion 与真实观测混在同一个误差里。

建议：

```text
observed_joint_error_mean_m
observed_joint_error_median_m
```

## Completed joint error

单独统计：

```text
completed_joint_error_mean_m
```

用于判断 proxy skeleton 的几何合理性。

---

# 34. State Estimation Metrics

建议至少记录：

```text
pose_detection_success
pose_detection_score

num_2d_visible_keypoints
num_3d_observable_keypoints
num_missing_keypoints
num_template_completed_keypoints

depth_valid_keypoint_ratio

estimated_human_position_x/y/z
human_position_error_m
human_position_confidence

estimated_human_yaw_deg
yaw_error_deg
yaw_confidence

estimated_body_scale
body_scale_confidence

estimated_state_valid
estimated_state_failure_reason
estimated_state_confidence
```

---

# 35. NBV 对比指标

v6.0 必须比较：

```text
EstimatedState-Ours
GTState-Ours
Oracle-NBV
Fixed
Random
Nearest
```

必要时保留 v5 消融策略。

核心指标：

```text
selected candidate id
selected_is_current
Q_pred_est
Q_pred_gt
Q_true_selected

GT-vs-Estimated selected-view agreement
GT-vs-Estimated geodesic distance difference
GT-vs-Estimated Q_true gap

current→selected S_action_occ_true gain
current→selected S_kp_occ_true gain

Oracle gap
```

尤其建议增加：

```text
estimated_gt_policy_agreement
```

表示 EstimatedState-Ours 是否选择了与 GTState-Ours 相同的位姿。

但不能只看 agreement，因为：

> 不同候选也可能有几乎相同的真实质量。

因此还必须看 Q_true gap。

---

# 36. 按当前观测质量分层分析

Estimated-State 的误差很可能与当前人体可见程度强相关。

建议按：

```text
High visibility
Medium visibility
Low visibility / severe occlusion
```

分层。

可依据 current 观测中的：

```text
num_3d_observable_keypoints
pose confidence
depth-valid ratio
```

划分。

分析：

```text
state error
NBV selection error
Q_true degradation
```

这样可以回答：

> **什么时候视觉状态估计足以支撑 NBV，什么时候会成为系统瓶颈？**

---

# 37. v6.0 调试脚本

## 37.1 `smoke_test_pose_backend.py`

输入单张当前 RGB。

输出：

```text
person count
bbox
2D keypoint names
confidence
pose overlay image
```

验收：

```text
Habitat Humanoid 能被 pose backend 检出
```

## 37.2 `debug_current_rgbd_pose.py`

保存：

```text
RGB
Depth
2D pose overlay
per-keypoint confidence
visible/missing joint list
```

## 37.3 `debug_depth_lifting.py`

对每个 joint 输出：

```text
u,v
confidence
depth sample
depth validity
camera XYZ
world XYZ
GT world XYZ（evaluation only）
3D error
```

## 37.4 `calibrate_estimated_orientation.py`

多个 GT yaw 下运行估计器，得到：

```text
est yaw
GT yaw
error
fixed sign/offset recommendation
```

## 37.5 `compare_gt_estimated_state.py`

可视化：

```text
GT skeleton
Observed estimated skeleton
Proxy completed skeleton
```

同一世界坐标图中对比。

## 37.6 `test_no_gt_leakage.py`

检查 Estimated-State online path 不读取 GT。

---

# 38. v6.0 可视化要求

至少输出三类图：

## 图 A：Current RGB + 2D pose

用于证明视觉前端真的在工作。

## 图 B：3D skeleton comparison

```text
GT skeleton
Observed RGB-D skeleton
Completed proxy skeleton
```

颜色/标记区分 source。

## 图 C：NBV comparison

地图俯视：

```text
current
EstimatedState-Ours
GTState-Ours
Oracle
human GT position
human estimated position
```

注意：

这张图是 evaluation visualization，可以显示 GT；不能成为 online input。

---

# 39. v6.0 失败处理原则

任何视觉前端失败都必须显式记录。

例如：

```text
no_person_detected
insufficient_2d_keypoints
insufficient_depth_keypoints
human_position_invalid
human_yaw_invalid
state_confidence_too_low
```

Estimated-State 主策略行为：

```text
invalid_state_behavior = stay
```

即：

```text
保持 current view
```

并标记：

```text
selection_reason = perception_invalid_fallback
```

禁止：

```text
如果估计失败，就读取 gt_human_pos / gt_yaw / gt_skeleton 继续运行
```

这种做法会让 v6.0 实验失去意义。

---

# 40. v6.0 与 v5.0 的兼容要求

v5.0 已经封版。

v6.0 应：

```text
继承稳定模块
但不修改 ea_avs_mvp_v5/
```

建议复用/迁移：

```text
HabitatRunner
HumanoidManager
CandidateView
geometry
raycast_utils
occlusion
TrueEvaluator
OraclePolicy
metrics 基础结构
```

新增视觉状态估计层后，再提供 v6 adapter。

不要让 v6.0 的实验需求反向破坏 v5.0 可复现性。

---

# 41. v6.0 的最低功能验收

## Acceptance A：Pose Backend

必须：

```text
真实 Habitat current RGB
→ detector
→ 至少输出一个 Humanoid pose
```

## Acceptance B：RGB-D Lifting

必须：

```text
2D keypoint
→ aligned depth
→ world 3D joint
```

至少多个 torso/limb joint 有效。

## Acceptance C：Estimated Human State

必须输出：

```text
estimated position
estimated yaw
body scale
observed joints
missing joints
proxy full skeleton
state validity/confidence
```

且 GT 不进入 estimator 接口。

## Acceptance D：Estimated Candidate Sampling

candidate ring 中心必须是：

```text
estimated_human_position
```

不是 GT。

## Acceptance E：Estimated-State Q_pred

预测器必须使用：

```text
estimated position
yaw
proxy skeleton
```

且在线选择前不 render candidate。

## Acceptance F：EstimatedState-Ours

至少完成：

```text
current vs candidates
allow stay
invalid state safe fallback
```

## Acceptance G：GT Comparison

完成：

```text
state estimation error
EstimatedState-Ours vs GTState-Ours
```

## Acceptance H：No GT Leakage

通过专门审查/测试证明：

```text
EstimatedState-Ours online path
```

不读取 Humanoid GT。

---

# 42. 推荐阶段性开发顺序

不要一次性全部写完。

建议严格按以下顺序：

## Stage 1：Current RGB Pose

```text
pose backend adapter
+ 2D overlay
```

只证明能检出人。

## Stage 2：Depth Lifting

```text
2D keypoints
+ Depth
→ partial 3D skeleton
```

## Stage 3：Human State

```text
position
scale
yaw
visible/missing joints
```

## Stage 4：Skeleton Completion

```text
observed skeleton
→ proxy full skeleton
```

## Stage 5：GT vs Estimated Debug

验证：

```text
position error
yaw error
joint error
```

## Stage 6：Estimated Candidate Sampling

将 candidate center 从 GT 切为 Estimated。

## Stage 7：Estimated-State Q_pred

复用 v5 score，替换人体状态来源。

## Stage 8：EstimatedState-Ours

跑通完整 one-shot selection。

## Stage 9：GTState / Oracle 对比

最后才跑完整 benchmark。

---

# 43. v6.0 不应一开始追求的指标

不要一开始要求：

```text
动作识别准确率提升
SOTA pose estimation
SOTA 3D human reconstruction
复杂 SMPL fitting
完整人体 mesh reconstruction
长时序 tracking
多动作 benchmark
```

这些都会稀释 v6.0 的核心目标。

当前最重要的是：

> **GT-State NBV 能否被 Current-RGB-D Estimated-State NBV 稳定替代。**

---

# 44. v6.0 预期代码主流程伪代码

```python
# -------------------------
# Simulator setup (GT only for scene construction)
# -------------------------
gt_state = setup_humanoid_scenario(...)
robot_state = setup_robot_start(...)

# -------------------------
# Current observation
# -------------------------
current_obs = runner.render_at(
    robot_state.position,
    robot_state.yaw,
)

# -------------------------
# Estimated-State online path
# NO GT arguments below
# -------------------------
est_state = human_state_estimator.estimate(
    rgb=current_obs["rgb"],
    depth=current_obs["depth"],
    camera_state=runner.get_camera_state(),
)

if not est_state.valid:
    estimated_selected = current_view
    estimated_selection_reason = "perception_invalid_fallback"
else:
    candidates_est = sampler.sample(
        human_pos=est_state.human_position_world,
        robot_pos=robot_state.position,
        runner=runner,
    )

    pred_est = []
    for cand in candidates_est:
        cand.pred_score = estimated_predictive_evaluator.score_view_pred(
            runner=runner,
            candidate=cand,
            robot_start_pos=robot_state.position,
            estimated_state=est_state,
        )
        pred_est.append(cand)

    estimated_selected = estimated_policy.select(
        current_view_est,
        pred_est,
    )

# ==========================================
# ONLINE ESTIMATED-STATE SELECTION ENDS HERE
# ==========================================

# Evaluation-only paths can now use GT
state_metrics = compare_estimated_to_gt(est_state, gt_state)
gt_state_selected = run_gt_state_baseline(...)
true_metrics = evaluate_selected_views(...)
oracle = run_oracle_after_render(...)
```

---

# 45. 最低实验规模

首轮 smoke / integration：

```text
10 episodes
```

流程稳定后：

```text
50-100 episodes
```

至少覆盖不同：

```text
human yaw
robot start angle
robot-human distance
clear LOS
partial environment occlusion
```

v6.0 首版不需要立即追求大规模统计显著性，先验证系统链路和误差传播。

---

# 46. 关键科研分析建议

v6.0 最值得回答的不是：

> “Pose 模型有多准？”

而是：

> **人体状态估计误差如何影响主动观察位姿选择？**

推荐分析：

```text
position error vs selected-view Q_true loss
yaw error vs orientation-score loss
missing-joint ratio vs NBV policy disagreement
3D observable keypoint count vs Estimated/GT policy gap
```

这会自然形成后续论文中的误差传播分析。

---

# 47. 后续版本接口预留

v6.0 完成后，EstimatedHumanState 应可进一步加入：

```text
current_action_probabilities
current_action_confidence
action_uncertainty
joint importance
missing evidence importance
```

然后后续 NBV 可加入：

\[
S_{recover}(v)
\]

以及：

\[
\hat Q(v)=\sum_a P(a\mid O_t)\hat Q(v\mid a)
\]

但这些不属于 v6.0 首版验收条件。

---

# 48. v6.0 最终版本定位

完成后应准确描述为：

> **EA-AVS-MVP v6.0: Current RGB-D based Estimated Human State for one-shot active observation pose selection.**

中文：

> **EA-AVS-MVP v6.0：基于当前 RGB-D 人体状态估计的单步主动观察位姿选择框架。**

它相对于 v5.0 的本质变化只有一条：

```text
v5.0:
Humanoid GT State
→ NBV

v6.0:
Current RGB-D
→ Estimated Human State
→ NBV
```

---

# 49. v6.0 封版标准

只有同时满足以下条件，才能认为 v6.0 完成：

1. 当前 RGB 能稳定进入 pose backend；
2. 2D keypoints 能映射到项目 15-keypoint schema；
3. 当前 Depth 能完成关键点 3D lifting；
4. 能估计 human position；
5. 能估计 human yaw；
6. 能输出 observed / missing joints；
7. 能构造不使用 GT 的 proxy full skeleton；
8. Candidate sampling 使用 estimated human position；
9. Estimated predictive evaluator 使用 estimated state；
10. EstimatedState-Ours 支持 current-vs-candidate 和 stay；
11. perception invalid 时只 safe stay，不 GT fallback；
12. GTState-Ours 与 EstimatedState-Ours 明确分开；
13. Oracle-NBV 仍只在 evaluation phase 使用 future true observation；
14. candidate future RGB/depth/semantic 不在选择前 render；
15. Semantic Humanoid mask 不进入 Estimated-State online path；
16. 输出 state-estimation accuracy metrics；
17. 输出 Estimated vs GT NBV comparison metrics；
18. `test_no_gt_leakage.py` 通过；
19. 至少 10 episodes integration run 成功；
20. v5.0 目录保持不变、仍可独立复现。

---

# 50. 给代码生成大模型的最终要求

开发 v6.0 时必须遵循：

```text
先建立 perception frontend
再建立 EstimatedHumanState
再替换 NBV state source
最后做 GT/Oracle comparison
```

不要同时扩展动作识别、Evidence Recovery、时序模型或新 NBV 公式。

每完成一个 Stage 必须先运行对应 debug script 并报告：

```text
输入
输出
有效率
失败原因
和 GT 的 evaluation-only 差异
```

特别禁止：

```text
为了让 demo 跑通，在 state estimation 失败时静默读取 Humanoid GT。
```

v6.0 成功的核心标准不是“程序不报错”，而是：

> **EstimatedState-Ours 的在线选择链路在结构上真正只依赖当前 RGB-D，并能够量化它相对于 GT-State NBV 的性能损失。**
