# EA-AVS MVP7.0: Humanoid-driven Active Perception Simulation Environment

> **Status**: **FINALIZED & COMPLETED** (2026-08-20)  
> **Active Code Directory**: `ea_avs_mvp_v7/`  
> **Active Specification**: `EA_AVS_MVP70_Code_Generation_Document.md`

---

## 1. 概述 (Overview)

`ea_avs_mvp_v7` 是面向老人监护与拟人化动态人体的室内主动感知仿真环境与科研实验平台。
本版本旨在构建严谨解耦、数据可复现的完整仿真与数据生成闭环：

```text
BABEL Action Annotation (Elderly Feasibility Set: 17 clips / 5 categories)
        ↓
AMASS Human Motion (.npz / 5 Sub-datasets / 7,862 files)
        ↓
Motion Normalization (NormalizedMotion / Schema A & B)
        ↓
Humanoid Motion Conversion (SMPL-X Quaternions PKL)
        ↓
Habitat Humanoid Playback (KinematicHumanoid neutral_0 with Relative Translation Compensation)
        ↓
Habitat Indoor Scene (apartment_1.glb with Calibrated Floor Y = -1.60m)
        ↓
Robot RGB-D Observation (RGB + Depth + 16-Joint 3D Pose GT)
        ↓
Episode Dataset (data/ActiveView/runs/ & visualizations/v7_final_demo/)
```

---

## 2. v7.0 已实现与验证的核心能力 (Completed Features)

1. **Habitat 室内环境集成 (Habitat Indoor Simulation)**：
   - 物理地面标高严格校准（`apartment_1.glb` 物理地面 $Y_{\text{floor}} = -1.60\text{m}$，天花板 $Y_{\text{ceiling}} = +1.14\text{m}$）；
   - 场景资产生命周期管理，解耦实体定位与传感器采集。
2. **拟人化人体代理 (Humanoid Agent)**：
   - 实例化 `KinematicHumanoid` (`neutral_0.urdf`)；
   - 实现了**相对位移与垂向下沉动态补偿机制**（$\text{mat}[:3, 3] = [T_x(t)-T_x(0), T_y(t)-0.90, T_z(t)-T_z(0)]$），确保站立、坐下、摔倒触地真实下沉，杜绝半空悬浮；
   - 支持通过 `configs/humanoid.yaml` 统一配置资产与基座参数。
3. **AMASS/BABEL 动作资产驱动 (AMASS/BABEL Motion Pipeline)**：
   - 覆盖老人 5 类典型动作（`standing`, `sitting`, `bending`, `reaching`, `fall_related`）；
   - 自动转换并校验 54 关节单位四元数。
4. **移动机器人 RGB-D 观测 (Robot RGB-D Observation)**：
   - 严格解耦底盘位姿 `robot_pose` 与相机位姿 `camera_pose`（外参矩阵与内参矩阵）；
   - 生成对齐的 RGB 图像（PNG）与 Depth 深度图（NPY）。
5. **16 关节 3D 世界坐标真值提取 (Human Pose Ground Truth)**：
   - 提取并严格校验 16 个核心关节的 3D 世界坐标真值，逐帧落盘至 `human_pose/frame_000000.json`。
6. **标准化 Episode 数据集生成 (Episode Dataset Generation)**：
   - 统一输出结构化 Episode 目录与包含完整位姿、内参、时序 GT 的 `metadata.json`。
7. **多维动作真实性动力学评价 (Action Motion Metrics)**：
   - 计算质心下落高度差、躯干俯仰角、关节动能与动态判定标志。
8. **演示与视频自动化生成 (Visualization & Video)**：
   - 一键合成 1080p/720p MP4 演示视频至 `visualizations/v7_final_demo/v7_demo.mp4`。

---

## 3. 科学边界与非目标声明 (Scientific Boundary & Explicit Non-Goals)

本版本严格定位为**感知仿真环境与数据集平台**，明确排除以下内容：
- ❌ **不实现** Active View Selection (NBV / 主动视角选择算法)
- ❌ **不实现** 机器人自主导航与路径规划 (Robot Navigation / Path Planning)
- ❌ **不实现** 强化学习 (RL / Reinforcement Learning)
- ❌ **不实现** 多步规划与视点效用优化 (Multi-step Planning / View Utility)
- ❌ **不实现** 动作识别网络训练 (Action Recognition / HAR Models)
- ❌ **历史版本保护**：`ea_avs_mvp_v6/` 及更早版本保持只读；
- ❌ **物理数据隔离**：严禁将原始大型动捕与渲染大文件提交至 Git 仓库。

---

## 4. 目录架构 (Repository Architecture)

```text
ea_avs_mvp_v7/
├── configs/
│   ├── humanoid.yaml                # Humanoid 实体统一配置 (avatar, offset, floor, scale)
│   ├── v7_demo.yaml                 # 综合演示参数统一配置 (scene, human, robot, camera, demo)
│   ├── habitat_config.yaml          # Habitat 场景配置
│   ├── motion_config.yaml           # AMASS 资产清单配置
│   └── sensor_config.yaml           # RGB-D 相机配置
│
├── core/
│   ├── paths.py                     # 数据根目录解析 (../../data/ActiveView 或 ACTIVEVIEW_DATA_ROOT)
│   ├── episode.py                   # Episode / EpisodeFrame 数据结构
│   └── config.py                    # 统一配置加载器 (load_v7_config, load_demo_config)
│
├── environment/
│   ├── scene_manager.py             # 场景与 NavMesh 资产定位
│   └── habitat_env.py               # Habitat-Sim 生命周期管理
│
├── robot/
│   ├── robot_agent.py               # 移动机器人底盘位姿控制
│   └── rgbd_sensor.py               # 相机外参/内参矩阵与视锥计算
│
├── human/
│   ├── human_spawn.py               # 人体安全地面位置采样接口 (sample_human_position)
│   ├── humanoid_agent.py            # KinematicHumanoid 实例化、相对位移补偿与姿态驱动
│   ├── action_metrics.py            # 多维动力学动作真实性评价 (ActionMotionMetrics)
│   ├── keypoint_mapping.py          # 16 核心关节世界坐标映射与合法性校验
│   ├── human_state.py               # 人体空间状态结构
│   └── action_state.py              # BABEL 动作语义状态
│
├── motion/
│   ├── amass_loader.py              # AMASS npz 读取与标准化 (NormalizedMotion)
│   ├── motion_converter.py          # Habitat SMPL-X 关节/刚体四元数格式转换
│   ├── motion_player.py             # 动作序列单步步进与帧控制
│   └── joint_mapping.py             # 54 关节映射与四元数校验
│
├── observation/
│   ├── recorder.py                  # 观测数据落盘 (RGB / Depth / Human Pose / Metadata)
│   └── metadata.py                  # 单帧元数据与序列元数据结构
│
├── dataset/
│   └── episode_generator.py         # 多动作多视角 Episode 生成器
│
├── evaluation/
│   └── basic_metrics.py             # 数据集统计指标
│
├── scripts/
│   ├── run_v7_demo.py               # 【主演示入口】一键端到端演示并生成视频
│   ├── verify_v7_pipeline.py        # 【流水线验证】端到端全量闭环测试
│   ├── create_video.py              # 【视频工具】RGB 序列编码为 MP4
│   ├── validate_three_actions.py    # 3 类典型动作全量校验
│   ├── run_smoke_test.py            # 最小闭环 Smoke Test
│   ├── generate_dataset.py          # 数据集批量生成
│   └── convert_motions.py           # AMASS 动作批量转换
│
└── tests/
    ├── unit/                        # 纯 Python 单元测试
    ├── integration/                 # 仿真与资产集成测试
    └── smoke/                       # 冒烟测试
```

---

## 5. 快速上手与运行指南 (Execution Guide)

### 5.1 运行一键综合科研演示 (Unified Demonstration)
```bash
conda run -n habitat python -m ea_avs_mvp_v7.scripts.run_v7_demo
```
**输出产物**（位于 `data/ActiveView/visualizations/v7_final_demo/`）：
- `rgb/`: `frame_000000.png` ~ `frame_000014.png`
- `depth/`: `frame_000000.npy` ~ `frame_000014.npy`
- `human_pose/`: `frame_000000.json` ~ `frame_000014.json`
- `metadata.json`: 完整位姿、相机内外参、16 关节 GT 与动力学评价指标
- `v7_demo.mp4`: 高清 MP4 演示视频

### 5.2 运行端到端流水线全量验证 (Pipeline Verification)
```bash
conda run -n habitat python -m ea_avs_mvp_v7.scripts.verify_v7_pipeline
```
输出：`[V7 Pipeline Verification] ... Status: PASS`

### 5.3 运行纯 Python 单元与回归测试
```bash
python3 -m unittest discover -s ea_avs_mvp_v7/tests -p "test_*.py"
```
输出：`Ran 30 tests in 0.118s -> OK`

### 5.4 图像序列转 MP4 视频工具
```bash
python3 -m ea_avs_mvp_v7.scripts.create_video \
  --input data/ActiveView/visualizations/v7_final_demo/rgb \
  --output data/ActiveView/visualizations/v7_final_demo/v7_demo.mp4 \
  --fps 10.0
```

---

## 6. Episode 数据落盘格式规范 (Episode Dataset Schema)

```text
data/ActiveView/runs/
└── episode_xxx/
    ├── rgb/
    │   └── frame_000000.png
    ├── depth/
    │   └── frame_000000.npy
    ├── human_pose/
    │   └── frame_000000.json
    └── metadata.json
```

**`metadata.json` 格式**：
```json
{
  "scene_id": "apartment_1",
  "episode_id": "v7_demo_fall_related_3522",
  "human": {
    "avatar": "neutral_0",
    "motion_id": "fall_related_3522",
    "action_class": "fall_related",
    "action_label": "fall to the ground"
  },
  "robot": {
    "initial_pose": {
      "position": [1.5, -1.6, 6.8],
      "yaw_deg": 0.0,
      "rotation_quat": [0.0, 0.0, 0.0, 1.0]
    },
    "camera_pose": {
      "extrinsic": [[1.0, 0.0, 0.0, 1.5], [0.0, 1.0, 0.0, -0.4], [0.0, 0.0, 1.0, 6.8], [0.0, 0.0, 0.0, 1.0]],
      "intrinsic": {"width": 640, "height": 480, "fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0}
    }
  },
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "rgb_path": "runs/episode_xxx/rgb/frame_000000.png",
      "depth_path": "runs/episode_xxx/depth/frame_000000.npy",
      "human_pose_gt": {
        "pelvis": [1.52, -0.98, 4.0],
        "head": [1.51, -0.76, 4.0],
        "left_ankle": [1.45, -1.54, 3.94]
      }
    }
  ],
  "motion_metrics": {
    "height_change": 0.523,
    "torso_angle_change": 49.7,
    "joint_motion_energy": 0.879,
    "dynamic_motion": true
  }
}
```
