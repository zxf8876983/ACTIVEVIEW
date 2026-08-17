# EA-AVS-MVP v5.0 代码生成指导文档

## 0. 文档用途

本文档用于指导大模型在现有 `ACTIVEVIEW` 项目中新增 `ea_avs_mvp_v5`，在 **不推翻 v4.0 主动观察位姿选择算法** 的前提下，解决当前最关键的仿真真实性缺口：

> **让 Habitat 场景中真正存在一个可渲染、可控制、具有完整人体表面的 Humanoid，使机器人相机能够获得真实包含人体的 RGB-D 观测。**

v5.0 的核心不是动作识别，也不是人体姿态估计，而是建立从“抽象骨架”到“真实可渲染人体”的桥梁。

一句话概括：

> **v5.0 = v4.0 的主动观察位姿选择框架 + Habitat Humanoid + 真实人体 RGB-D 渲染 + Humanoid GT 状态接口。**

---

# 1. v1.0-v5.0 的研究演进位置

项目主线如下：

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
真实 Habitat Humanoid + 人体表面 + RGB-D + Self-occlusion 基础
        ↓
后续版本
当前 RGB-D → Pose Estimator → Estimated Human State
        ↓
真实动作识别增益评价
```

v5.0 不是最终科研系统，但它是从“几何模拟”走向“真实视觉主动感知”的关键转折点。

---

# 2. v5.0 的研究目标

当前 v4.0 使用的是：

```text
GT human_pos
GT human_yaw
GT pose_type
手工定义完整 skeleton
        ↓
NBV Selector
```

v5.0 要变为：

```text
Habitat Indoor Scene
        +
Real Renderable Humanoid
        ↓
Robot Camera
        ↓
RGB + Depth containing the humanoid
```

同时保留一条 GT-State 支路：

```text
Humanoid internal pose / joint state
        ↓
GT skeleton / GT human orientation
        ↓
v4.0 NBV selector
```

因此 v5.0 主要解决两个问题：

1. **仿真场景中真正存在人体 mesh，而不是只有代码中的抽象关键点。**
2. **机器人相机真正能够看到该人体，并输出包含人体遮挡、自遮挡和姿态差异的 RGB-D。**

---

# 3. v5.0 不解决什么

本版本明确不做：

1. 不接入 OpenPose / RTMPose / YOLO-Pose；
2. 不从 RGB 自动估计 skeleton；
3. 不训练动作识别网络；
4. 不训练人体检测模型；
5. 不做强化学习；
6. 不做多步 NBV；
7. 不做 ROS；
8. 不做 Unity；
9. 不做真实机器人控制；
10. 不设计新的 v4.0 NBV 评分公式；
11. 不把“老人外观建模”作为本版本研究目标；
12. 不要求本版本直接生成高质量老人专用 Avatar。

v5.0 的研究边界是：

> **真实可渲染 Humanoid 仿真基础设施 + 与现有主动视角选择框架的集成。**

---

# 4. v5.0 开发前环境与工具准备

## 4.1 必需组件

### 4.1.1 Habitat-Sim with Bullet physics

Humanoid 属于 articulated agent，v4.0 ray casting 也已经依赖 physics，因此环境必须支持 Bullet physics。

建议不要直接破坏当前能运行的 Habitat 环境。

优先：

```bash
conda create --name <new_humanoid_env> --clone <current_habitat_env>
```

然后在新环境中确认 Habitat-Sim / Habitat-Lab 版本兼容。

建议首先保存环境快照：

```bash
conda list > habitat_env_before_v5.txt
pip freeze > habitat_pip_before_v5.txt
```

不要为了 Humanoid 功能盲目升级所有依赖。

---

## 4.2 Habitat-Lab

如果当前环境只有 `habitat-sim`，必须补充 `habitat-lab`。

v5.0 需要使用 Habitat-Lab 中的 Humanoid / ArticulatedAgent 支持，例如：

```text
KinematicHumanoid
HumanoidRearrangeController
Humanoid-related articulated agent utilities
```

大模型在实现时必须首先检查本机实际安装版本的 Habitat-Lab API，不允许根据旧版本博客自行编造接口名。

推荐代码兼容策略：

```python
try:
    import habitat
except ImportError:
    raise RuntimeError(
        "EA-AVS v5.0 requires habitat-lab for Humanoid support."
    )
```

---

## 4.3 Habitat 官方 Humanoid 资源包

必须准备 Habitat 官方 humanoid assets。

建议目录：

```text
<data_root>/habitat_humanoids/
├── neutral_0/
│   ├── neutral_0.ao_config.json
│   ├── neutral_0.glb
│   ├── neutral_0.urdf
│   └── neutral_0_motion_data_smplx.pkl
├── female_0/
├── male_0/
├── ...
├── standing_pose_smplx.pkl
├── walking_motion_processed_smplx.pkl
└── walk_motion/
```

官方资源包含多个预制 humanoid avatar，每个 avatar 至少包含：

```text
GLB       → 外观、纹理、skinning
URDF      → 人体骨架 / articulated structure
AO config → Habitat articulated object 配置
PKL       → Humanoid motion/controller 数据
```

v5.0 默认先使用：

```text
neutral_0
```

不要一开始同时测试 12 个 avatar。

---

## 4.4 下载工具

资源包可以通过 Hugging Face / Git 方式下载。

可提前安装：

```bash
git lfs install
```

或者使用 Hugging Face 官方下载方式。

`git-lfs` 只属于下载辅助工具，不是运行 Humanoid 的必要 Python 依赖。

---

# 5. v5.0 暂时不必安装的工具

使用 Habitat 官方预制 humanoid 时，以下工具 **当前不是必须项**：

```text
Blender
Unity
ROS
Isaac Sim
MuJoCo
PyTorch3D
OpenPose
RTMPose
SMPLify-X
VPoser
AMASS 全量数据集
```

特别说明：

## Blender

只有在后续需要：

```text
修改人体 mesh
修改纹理
重新 rigging
导出 GLB
制作自定义老人 Avatar
```

时才需要。

v5.0 核心阶段不安装 Blender 也可以完成。

---

# 6. 与 sitting / bending / lying_fallen 有关的额外资源

Habitat 官方 humanoid 资源适合优先验证：

```text
standing
walking
reaching
```

但我们的研究需要：

```text
standing
sitting
bending
lying_fallen
```

因此 v5.0 代码必须区分：

## Mode A：Official Humanoid Smoke Test

必须完成：

```text
standing
walking / official motion
```

用于证明：

```text
Humanoid 可以加载
Humanoid 可以被渲染
RGB 中有人
Depth 中有人
不同观察位姿能看到不同人体表面
```

## Mode B：Research Action Pose Extension

代码必须预留接口支持：

```text
sitting
bending
lying_fallen
```

动作资源可以后续从：

```text
AMASS motion clips
SMPL-X pose/motion files
自行准备的 Habitat-compatible processed motion
```

导入。

### 如果希望 v5.0 同期完成四类姿态，建议提前准备

```text
SMPL-X model / Python package
选择性 AMASS 动作片段
```

但不要一开始下载/处理整个 AMASS 数据集。

优先只准备少量能够覆盖：

```text
sit / sit-down
bend
fall / lie
get-up
```

的动作片段。

### VPoser

只有当我们决定自己生成静态自然姿态，或者进行 pose optimization / IK 时再安装。

如果直接使用真实 mocap clip，则 VPoser 不是 v5.0 必需项。

---

# 7. v5.0 核心原则

## 7.1 Humanoid 必须是真正的场景对象

不允许：

```text
只把旧 skeleton 画到 RGB 图像上
```

或者：

```text
在 OpenCV 中后处理粘一个人体图片
```

Humanoid 必须作为 Habitat 内部真实可渲染对象存在。

机器人相机必须通过 Habitat renderer 自然观察到人体。

---

## 7.2 RGB 和 Depth 必须来自同一相机姿态

对于任一观察位姿：

```python
obs = runner.render_at(position, yaw)
```

应同时得到：

```text
obs["rgb"]
obs["depth"]
```

并且 RGB / depth 中 Humanoid 的空间位置必须一致。

---

## 7.3 v5.0 保留 GT-State 模式

本版本不要急着删除 GT 信息。

v5.0 的选择器仍可以使用：

```text
GT humanoid position
GT humanoid orientation
GT humanoid joint state / skeleton
```

原因：

> v5.0 的目标是验证真实人体渲染与 NBV 框架能否正确集成，而不是同时解决视觉人体状态估计问题。

后续版本再加入：

```text
Rendered RGB-D
→ 2D pose estimator
→ estimated human state
```

最终才能比较：

```text
GT-State NBV
vs
Estimated-State NBV
```

---

# 8. v5.0 对 v4.0 的保护要求

v5.0 不允许重新设计 v4.0 以下模块：

```text
candidate_sampler
predictive_evaluator 的核心 NBV 思路
OursPolicy
ablation policies
Oracle protocol
pred / true information boundary
```

应优先采用：

```text
copy + adapter
```

而不是：

```text
rewrite everything
```

v4.0 原目录保持不动。

---

# 9. v5.0 项目目录结构

请新建：

```text
ea_avs_mvp_v5/
├── configs/
│   └── mvp50_humanoid.yaml
├── scripts/
│   ├── smoke_test_humanoid.py
│   ├── debug_humanoid_rgbd.py
│   └── run_mvp50_humanoid.py
├── ea_avs_v5/
│   ├── __init__.py
│   ├── config.py
│   ├── habitat_runner.py
│   ├── humanoid_assets.py
│   ├── humanoid_manager.py
│   ├── humanoid_motion.py
│   ├── humanoid_state.py
│   ├── humanoid_skeleton_adapter.py
│   ├── geometry.py
│   ├── candidate_sampler.py
│   ├── predictive_evaluator.py
│   ├── true_evaluator.py
│   ├── policies.py
│   ├── ablation_policies.py
│   ├── oracle_policy.py
│   ├── metrics.py
│   └── visualization.py
└── outputs/
```

重点新增：

```text
humanoid_assets.py
humanoid_manager.py
humanoid_motion.py
humanoid_state.py
humanoid_skeleton_adapter.py
```

---

# 10. 配置文件

文件：

```text
ea_avs_mvp_v5/configs/mvp50_humanoid.yaml
```

建议：

```yaml
project:
  name: "EA-AVS-MVP5.0"
  seed: 42

habitat:
  scene_path: "/ABS/PATH/apartment_1.glb"
  navmesh_path: "/ABS/PATH/apartment_1.navmesh"
  enable_physics: true

camera:
  width: 640
  height: 480
  hfov_deg: 90
  camera_height: 1.2
  clip_near: 0.01
  clip_far: 10.0

humanoid:
  enabled: true
  assets_root: "/ABS/PATH/habitat_humanoids"
  avatar_name: "neutral_0"

  # official / custom_motion
  motion_mode: "official"

  # v5.0 smoke test 必须支持 standing；walking 用于验证动画链路
  default_pose: "standing"
  supported_official_states:
    - standing
    - walking

  # 后续动作素材准备后开启
  research_pose_files:
    sitting: null
    bending: null
    lying_fallen: null

  semantic_id: 100

  # Humanoid 放置时与地面的高度修正，默认从资源/config/API推导
  base_height_offset: 0.0

  # 如果 motion / pose 不存在，禁止静默使用错误动作
  strict_motion_loading: true

episode:
  num_episodes: 20
  min_robot_human_distance: 1.5
  max_robot_human_distance: 4.0
  max_sampling_tries: 100
  randomize_human_yaw: true

candidate_sampling:
  radii: [1.5, 2.0, 2.5]
  angles_deg: [-135, -90, -45, 0, 45, 90, 135, 180]
  min_distance_to_human: 1.2
  max_distance_to_human: 3.0
  max_geodesic_distance: 6.0

visibility:
  min_depth: 0.5
  max_depth: 5.0

output:
  save_images: true
  save_depth: true
  save_debug_json: true
  save_video: false
```

v4.0 的动作权重、评分权重、occlusion 参数应复制到 v5 配置，不在本文重复设计。

---

# 11. `humanoid_assets.py`

功能：

> 只负责 Humanoid 资源路径发现与完整性检查。

不得在此文件执行 Habitat 仿真逻辑。

建议数据结构：

```python
from dataclasses import dataclass

@dataclass
class HumanoidAssetBundle:
    avatar_name: str
    root_dir: str
    ao_config_path: str
    glb_path: str
    urdf_path: str
    motion_data_path: str
```

必须实现：

```python
def resolve_humanoid_assets(config: dict) -> HumanoidAssetBundle:
    pass


def validate_humanoid_assets(bundle: HumanoidAssetBundle) -> None:
    pass
```

验证至少包括：

```text
AO config 是否存在
GLB 是否存在
URDF 是否存在
motion PKL 是否存在
```

缺失时必须明确报错，例如：

```text
Humanoid assets missing: neutral_0.glb
Please download ai-habitat/habitat_humanoids first.
```

禁止自动退化成 v4.0 抽象 skeleton render。

---

# 12. `humanoid_manager.py`

这是 v5.0 核心新增模块。

职责：

```text
加载 Humanoid
放入场景
设置位置
设置 yaw
设置 pose/motion
reset
读取 Humanoid object/state
删除/清理
```

建议接口：

```python
class HumanoidManager:
    def __init__(self, runner, config: dict):
        pass

    def load(self):
        pass

    def reset(self):
        pass

    def set_base_pose(
        self,
        position,
        yaw,
    ):
        pass

    def set_pose(self, pose_name: str):
        pass

    def step_motion(self, dt: float):
        pass

    def get_state(self):
        pass

    def close(self):
        pass
```

### 重要 API 原则

大模型必须：

1. 优先查看本机安装的 Habitat-Lab v0.3.x 源码；
2. 优先参考官方：

```text
humanoids_tutorial.ipynb
articulated_agents_tutorial.ipynb
test_humanoid.py
KinematicHumanoid
HumanoidRearrangeController
```

3. 不允许根据记忆编造不存在的 API；
4. 如果当前版本 API 与示例不同，应写兼容 wrapper；
5. 不要为了加载 Humanoid 把整个 EA-AVS 工程改写成强化学习 / Rearrange task。

---

# 13. plain Habitat-Sim vs RearrangeSim

v5.0 优先目标是：

> **把 Humanoid 嵌入我们当前已有的 Habitat-Sim 主流程。**

因此优先尝试：

```text
现有 HabitatRunner
+ KinematicHumanoid / articulated object API
```

而不是强行将项目整体迁移为：

```text
Habitat Rearrange Environment
```

只有当实际安装版本中 Humanoid API 确实必须依赖 RearrangeSim 时，才增加一个内部 adapter：

```text
HumanoidBackend
├── SimHumanoidBackend
└── RearrangeHumanoidBackend
```

但主研究代码对 backend 无感。

---

# 14. `humanoid_motion.py`

功能：统一控制 Humanoid 动作/姿态。

必须定义：

```python
class HumanoidMotionController:
    def set_state(self, state_name: str):
        pass

    def step(self, dt: float):
        pass
```

至少支持：

```text
standing
walking
```

如果配置请求：

```text
sitting
bending
lying_fallen
```

但对应文件不存在，在：

```yaml
strict_motion_loading: true
```

时必须：

```python
raise FileNotFoundError(...)
```

绝不能：

```text
sitting → standing
```

静默替代。

建议允许外部 motion provider：

```python
class MotionProvider:
    def load_pose(self, pose_name):
        ...

    def get_pose(self, frame_idx):
        ...
```

为后续 AMASS/SMPL-X 留接口。

---

# 15. `humanoid_state.py`

功能：保存 Humanoid 当前 GT 状态。

建议：

```python
@dataclass
class HumanoidState:
    base_position: np.ndarray
    base_yaw: float
    pose_name: str
    motion_frame: int | None
    semantic_id: int
```

后续可以扩展：

```text
joint_transforms
joint_positions_world
body_forward
```

---

# 16. `humanoid_skeleton_adapter.py`

这是 v5.0 与 v4.0 NBV 方法连接的关键模块。

目的：

> 将 Humanoid 当前真实关节/链接状态转换为 v4.0 evaluator 能使用的关键点字典。

输出形式仍保持：

```python
{
    "head": np.ndarray([x, y, z]),
    "neck": ...,
    "left_shoulder": ...,
    "right_shoulder": ...,
    "left_elbow": ...,
    "right_elbow": ...,
    "left_wrist": ...,
    "right_wrist": ...,
    "pelvis": ...,
    "left_hip": ...,
    "right_hip": ...,
    "left_knee": ...,
    "right_knee": ...,
    "left_ankle": ...,
    "right_ankle": ...,
}
```

必须实现类似：

```python
def get_humanoid_gt_skeleton(
    humanoid_manager: HumanoidManager,
) -> dict:
    pass
```

### 关键要求

优先从 Humanoid articulated structure / joint transforms 中获取。

不能继续使用：

```text
action_pose_library.py 中手工定义的 standing/sitting 坐标
```

作为 v5.0 的主要 GT skeleton。

旧 skeleton 只能作为：

```text
legacy/debug fallback
```

且必须显式标记。

---

# 17. Humanoid yaw / forward 方向

v5.0 不再手工假设人体朝向和 mesh 朝向一定一致。

必须增加一个校准测试：

```text
Humanoid base yaw = 0
↓
渲染正面 / 背面
↓
确认模型 forward 轴
```

新增：

```python
def get_humanoid_forward_vector(...):
    pass
```

并保证：

```text
human_yaw
S_orient_pred
Humanoid 实际面朝方向
```

三者一致。

如果 Habitat Humanoid 资产内部 forward convention 与 v4.0 `+Z` 不一致，只在 adapter 层修正，不要到处添加 magic `+pi`。

---

# 18. `habitat_runner.py` 的 v5.0 修改

保持 v4.0：

```text
RGB sensor
Depth sensor
Physics
Ray casting
Navmesh
render_at()
```

新增：

```python
def attach_humanoid_manager(self, manager):
    pass
```

或者直接由外部 `HumanoidManager` 持有 `runner.sim`。

核心要求：

```python
obs = runner.render_at(robot_pos, robot_yaw)
```

必须自然渲染出已经存在于场景中的 Humanoid。

不要在 `render_at()` 中临时创建/删除人体。

Humanoid 应在 episode 生命周期内持续存在。

---

# 19. RGB-D Humanoid 可见性验证

新增：

```text
scripts/debug_humanoid_rgbd.py
```

功能：

1. 加载 scene；
2. 创建 Humanoid；
3. 放在可导航位置；
4. 设为 standing；
5. 在人体周围 4 个方向放置相机；
6. 分别 render RGB + depth；
7. 保存结果。

输出：

```text
outputs/humanoid_rgbd_debug/
├── front_rgb.png
├── front_depth.npy
├── left_rgb.png
├── left_depth.npy
├── back_rgb.png
├── back_depth.npy
├── right_rgb.png
└── right_depth.npy
```

验收要求：

```text
front / side / back 图像中的人体外观明显不同
人体确实存在于 RGB
人体对应位置在 depth 中有合理深度
家具可以真实遮挡人体
人体自身表面产生 self-occlusion
```

---

# 20. semantic sensor：可选但强烈建议

v5.0 可以增加一个调试 semantic sensor，用于确认：

```text
哪些像素属于 Humanoid
```

如果当前 Habitat asset / API 支持 humanoid semantic id，可配置：

```yaml
humanoid:
  semantic_id: 100
```

semantic 只用于：

```text
调试
自动检查 RGB 中是否真的出现 humanoid
计算 humanoid pixel area
```

不能用于后续真实视觉算法输入。

最终真实视觉 front-end 仍然只能用 RGB-D。

---

# 21. `smoke_test_humanoid.py`

这是 v5.0 第一优先级测试。

运行：

```bash
python scripts/smoke_test_humanoid.py \
  --config configs/mvp50_humanoid.yaml
```

只做：

```text
初始化 Habitat
加载 avatar assets
创建 Humanoid
standing
render
walking motion step N frames
render
关闭
```

必须打印：

```text
Humanoid asset loaded: neutral_0
Humanoid object created
Standing pose applied
RGB contains humanoid: PASS / UNKNOWN
Depth available: PASS
Walking motion update: PASS
```

如果失败，需要明确告诉用户失败在哪个阶段。

---

# 22. v5.0 主实验流程

```text
Episode Start
    ↓
采样 human base position
    ↓
创建/重置 Humanoid
    ↓
设置 Humanoid base position + yaw
    ↓
设置 pose / motion state
    ↓
读取 Humanoid GT skeleton
    ↓
生成 robot current pose
    ↓
生成 candidate observation poses
    ↓
使用 v4.0 pred evaluator
  （GT Humanoid skeleton + map geometry）
    ↓
策略选择完成
    ↓
【Evaluation Phase】
    ↓
render current + selected/all candidates
    ↓
保存包含 Humanoid 的 RGB-D
    ↓
计算几何/depth true metrics
    ↓
保存 Humanoid GT state + RGB-D debug
```

注意：

v5.0 仍然保持：

```text
候选点未来 RGB / depth
不得进入在线策略选择
```

Humanoid 的存在不改变 v2.0 建立的信息边界。

---

# 23. v5.0 PredictiveEvaluator 输入

v5.0 的预测选择输入可以使用：

```text
robot current pose
Humanoid GT base position
Humanoid GT orientation
Humanoid GT skeleton
known scene geometry
navmesh
camera model
```

这是明确的：

> **GT-State Humanoid experiment**

论文/文档中不能伪装成真实视觉估计。

---

# 24. v5.0 TrueEvaluator 的变化

v4.0 true evaluator 已经开始利用 rendered depth。

v5.0 中最大的变化是：

> rendered depth 中真的存在 Humanoid mesh，而不是只存在环境。

因此 v5.0 要验证：

```text
ray cast predicted environment occlusion
vs
rendered RGB-D actual human surface visibility
```

但本版本仍然不要求：

```text
RGB pose detector true visibility
```

那属于后续 Estimated-State / Vision version。

---

# 25. Self-Occlusion 的处理边界

v5.0 的 Humanoid mesh 会自然产生：

```text
躯干挡手臂
双腿重叠
身体正背面差异
手臂挡躯干
```

这属于真实 render 层的 self-occlusion。

但是：

在线 PredictiveEvaluator 当前使用的 map ray casting 是否能识别 humanoid 本体，取决于 Humanoid articulated body 是否被当前 ray query 纳入 collision scene。

因此必须做一个专门测试：

```text
candidate → wrist ray
```

当 wrist 被 torso 挡住时：

```text
ray cast 是否先命中 torso？
```

### 两种情况

#### A. Habitat ray cast 能命中 Humanoid body

则 v5.0 predictor 可自然获得：

```text
environment occlusion + self-occlusion
```

#### B. Habitat ray cast 只处理环境 / collision mesh 不包含 humanoid skin

不要伪造 self-occlusion predictor。

此时：

```text
pred = environment occlusion
true = environment + rendered self-occlusion
```

并在 metrics 中显式记录。

后续版本再设计人体几何 self-occlusion predictor。

---

# 26. 新增 Metrics

建议新增：

```text
humanoid_enabled
humanoid_avatar_name
humanoid_pose_name
humanoid_motion_frame
humanoid_base_x
humanoid_base_y
humanoid_base_z
humanoid_yaw
humanoid_gt_skeleton_source
rgb_humanoid_visible
humanoid_pixel_count        # semantic 可用时
humanoid_depth_valid_ratio
humanoid_render_success
humanoid_self_occlusion_supported_pred
```

继续保留 v4.0：

```text
S_action_occ_pred
S_action_occ_true
Q_pred
Q_true
occlusion_rate_pred
occlusion_rate_true
oracle_gap
```

---

# 27. Debug JSON

每个 episode 保存：

```json
{
  "humanoid": {
    "avatar_name": "neutral_0",
    "pose_name": "standing",
    "base_position": [0, 0, 0],
    "base_yaw": 0.0,
    "skeleton_source": "habitat_humanoid_gt",
    "asset_paths": {
      "glb": "...",
      "urdf": "..."
    }
  },
  "camera": {},
  "current_view": {},
  "candidate_views": []
}
```

---

# 28. v5.0 三阶段验收

## Stage A：Humanoid Asset Smoke Test

必须完成：

```text
官方 neutral_0 成功加载
standing 成功
RGB 中存在 Humanoid
Depth 可输出
```

---

## Stage B：Humanoid Motion Test

必须完成：

```text
walking motion 能驱动 mesh
连续若干帧人体姿态发生变化
不会崩溃
```

注意：

walking 不属于最终核心老人动作，它只是验证 animation pipeline。

---

## Stage C：EA-AVS Integration

必须完成：

```text
Humanoid 放入 apartment
机器人 current + candidate views 正常生成
策略选择前不 render candidates
策略选择后 render
RGB-D 中确实包含 Humanoid
v4.0 Q_pred / Ours / Oracle 流程仍能正常运行
```

---

# 29. v5.0 最低验收指标

建议运行：

```bash
python scripts/run_mvp50_humanoid.py \
  --config configs/mvp50_humanoid.yaml \
  --episodes 10 \
  --output-dir outputs/mvp50_smoke_run
```

至少满足：

```text
Humanoid load success = 100%
Episode runtime success >= 80%
RGB render success = 100% of successful episodes
Depth render success = 100% of successful episodes
No candidate future RGB used before policy selection
OursPolicy still only uses Q_pred
```

v5.0 首轮不要一上来跑 1000 episodes。

---

# 30. 不允许的错误实现

以下实现均不接受：

## 错误 1

```text
保留抽象 skeleton，然后在 RGB 上画一个人体图标
```

不是 Humanoid。

## 错误 2

```text
render 之前把 candidate RGB 全部生成，然后选最好的
```

信息泄漏。

## 错误 3

```text
sitting motion 缺失时自动用 standing 代替
```

会污染实验。

## 错误 4

```text
为了加载 Humanoid 重写成 RL task
```

偏离研究目标。

## 错误 5

```text
使用 old action_pose_library 的手工 skeleton 作为 v5.0 主 GT
```

v5.0 必须优先从 Humanoid 实际状态获得 GT skeleton。

## 错误 6

```text
Humanoid 只在 third-person debug camera 可见，机器人 RGB 看不到
```

不能通过验收。

---

# 31. 代码兼容与错误处理

由于 Habitat-Lab 不同版本的 Humanoid API 可能存在差异，大模型必须实现版本检查。

建议启动时打印：

```text
Habitat-Sim version
Habitat-Lab version / source location
Physics enabled
Humanoid asset path
Selected Humanoid backend
```

如果关键 API 不存在：

```python
raise RuntimeError(
    "Installed Habitat-Lab does not expose the required Humanoid API. "
    "Please verify the Habitat-Lab/Habitat-Sim version pair."
)
```

禁止用 `except Exception: pass` 静默跳过 Humanoid。

---

# 32. v5.0 推荐开发顺序

大模型必须按下面顺序实现，而不是一次写完所有文件：

```text
Step 1
检查 Habitat 环境 / Humanoid API / assets

Step 2
smoke_test_humanoid.py
单独加载 humanoid

Step 3
debug_humanoid_rgbd.py
确认 robot camera RGB-D 有人体

Step 4
humanoid_state + skeleton_adapter
从实际 Humanoid 提取 GT state

Step 5
迁移 v4.0 candidate/pred/policy

Step 6
run_mvp50_humanoid.py
完整 one-shot NBV integration

Step 7
metrics/debug
```

如果 Step 2 没通过，不要继续编写 NBV 集成代码。

---

# 33. v5.0 成功后，下一阶段才做什么

v5.0 完成以后，下一阶段才开始：

```text
Rendered RGB-D
      ↓
2D Pose Estimator
      ↓
Visible / Missing joints
      ↓
Depth lifting
      ↓
Estimated 3D Skeleton
      ↓
Estimated human orientation
      ↓
Estimated-State NBV
```

再之后才是：

```text
Action classifier
↓
Accuracy / Macro-F1 / critical-action Recall
↓
真正证明 active view 提升动作识别
```

不要在 v5.0 混入这些内容。

---

# 34. 给大模型的最终执行提示词

```text
请根据 EA_AVS_MVP50_Code_Generation_Document.md 实现 ea_avs_mvp_v5。

当前仓库已经存在 ea_avs_mvp_v4，请优先复用 v4.0 中已经稳定的：
- candidate sampling
- navmesh/geodesic filtering
- predictive evaluator
- occlusion ray casting
- policies / ablations
- Oracle protocol
- metrics framework

v5.0 的唯一核心新增任务是：
1. 使用 Habitat 官方 Humanoid asset 创建真实可渲染人体；
2. 机器人 RGB-D camera 能够在 current/candidate observation pose 看到该人体；
3. 使用 Humanoid 实际 GT state/joint transforms 建立 GT skeleton adapter；
4. 保持 v4.0 one-shot NBV 信息边界，禁止策略选择阶段使用未来 candidate RGB/depth；
5. 完成 standing + official walking motion 的 smoke test；
6. 为 sitting/bending/lying_fallen 预留外部 motion/pose 接口，但缺少素材时必须明确报错，不能用 standing 静默替代。

实现前必须检查当前安装的 Habitat-Lab / Habitat-Sim Humanoid API，参考官方 humanoids tutorial、articulated agents tutorial、test_humanoid，以及 KinematicHumanoid / HumanoidRearrangeController 实际源码。禁止编造不存在的接口。

必须优先实现：
python scripts/smoke_test_humanoid.py --config configs/mvp50_humanoid.yaml

只有 smoke test 成功后，才继续完整 EA-AVS 集成。
```

---

# 35. 一句话总结

EA-AVS-MVP v5.0 的任务不是让 NBV 算法更复杂，而是第一次让“被观察的人”真正存在于仿真视觉世界中：

> **从代码中的抽象人体骨架，升级为 Habitat 中真实可渲染、具有表面、自遮挡和动作状态的 Humanoid，为后续 RGB-D 人体状态估计和真实动作识别验证建立基础。**
