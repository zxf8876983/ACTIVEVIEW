# ACTIVEVIEW v10.0 Code Generation Specification

## 1. Purpose

本文档用于指导 v10.0 版本代码开发。开发必须同时遵循：

1. ACTIVEVIEW_V10.0_Research_Development_Document.md（研究目标与实验规范）
2. 本文档（工程实现规范）

v10.0不是重新设计项目，而是在v9.1代码基础上的扩展。

核心目标：

> 将v9.1中的Observation Quality Gain升级为Task-level Action Recognition Gain。

---

# 2. 总体开发原则

## 保留v9.1能力

必须复用：

- Habitat环境加载
- Humanoid生成
- Candidate viewpoint生成
- View descriptor
- Observation Provider接口思想
- Experiment框架

禁止重新实现已有能力。

---

# 3. v10.0总体Pipeline

```
Habitat RGB-D
      |
      v
2D Pose Estimator
      |
      v
Depth Back Projection
      |
      v
Estimated 3D Skeleton
      |
      v
ST-GCN
      |
      v
Action Probability
      |
      v
Active View Selector
      |
      v
Next Best View
```

---

# 4. 新增模块

建议目录：

```
v10/

├── perception/
│   ├── pose_estimator.py
│   ├── depth_projection.py
│   └── skeleton_processor.py
│
├── action/
│   ├── stgcn_model.py
│   ├── train_action.py
│   └── inference_action.py
│
├── active_view/
│   ├── action_view_scorer.py
│   └── gain_estimator.py
│
├── dataset/
│   ├── generate_v10_dataset.py
│   └── dataset_loader.py
│
└── experiments/
```

实际路径必须适配现有v9代码结构，不允许破坏已有版本。

---

# 5. 数据目录规范

所有运行数据保存：

```
/home/zxf/WorkSpace/code/data/ActiveView/
```

禁止将大型数据提交Github。

建议：

```
ActiveView/

├── datasets/
│   └── v10/
│       ├── raw/
│       │   ├── rgb/
│       │   ├── depth/
│       │   ├── camera_pose/
│       │   └── metadata/
│       │
│       ├── perception/
│       │   ├── pose2d/
│       │   ├── pose3d/
│       │   └── confidence/
│       │
│       ├── action/
│       │   ├── labels/
│       │   └── stgcn/
│       │
│       └── viewpoints/
```

---

# 6. GT使用规则

严格禁止：

模型forward输入：

- GT skeleton
- GT action
- GT visibility

GT只能用于：

1. 数据生成
2. Action label
3. Oracle baseline
4. Offline evaluation

---

# 7. Pose模块规范

输入：

RGB + Depth

流程：

RGB -> 2D keypoints

Depth -> 3D projection

输出：

```
(T,J,3) skeleton
+
confidence
```

推荐使用：

- RTMPose
- MediaPipe

不要训练新的pose模型。

---

# 8. Skeleton Processing

必须：

## Root normalization

以pelvis作为中心。

## Scale normalization

按照人体尺度归一化。

禁止：

camera-facing rotation normalization。

原因：视角变化是研究变量。

---

# 9. Action Recognition模块

采用：

ST-GCN

输入：

```
T x J x 3
```

输出：

```
action probability vector
```

动作类别固定：

1 standing
2 walking
3 sitting
4 bending
5 reaching
6 falling

---

# 10. ST-GCN训练规范

动作识别模型独立训练。

流程：

Habitat生成多视角动作数据

-> skeleton dataset

-> ST-GCN training

-> freeze ST-GCN

-> Active View training

Active View不是训练ST-GCN。

---

# 11. Active View Selector

输入：

Current observation:

- estimated skeleton
- joint confidence
- action probability
- current view

Candidate view:

- distance
- angle
- height
- geometry feature

输出：

candidate score

---

# 12. 优化目标

预测：

Action Gain

定义：

```
Gain = ActionConfidence_after - ActionConfidence_before
```

或：

```
Gain = Entropy_before - Entropy_after
```

---

# 13. 实验要求

Baseline：

1 Random
2 Nearest
3 Geometry(v8)
4 Perception(v9.1)
5 Action-aware(v10)
6 Oracle

指标：

- Top1 Accuracy
- Accuracy improvement
- Action entropy reduction
- Confidence gain

---

# 14. 开发阶段

## v10.0.0

完成：

RGB-D -> Skeleton -> ST-GCN -> View ranking

## v10.0.1

Action gain scoring

## v10.0.2

Multi-scene evaluation

---

# 15. 验收标准

必须满足：

1. Habitat能够生成RGB-D数据
2. Pose pipeline运行
3. ST-GCN训练和推理运行
4. Active View能够输出候选排序
5. 主动视角相比随机和固定视角提升动作识别性能

---

# 16. 禁止事项

禁止：

- SLAM
- Navigation
- RL
- 新Pose网络
- 新Action Recognition网络
- End-to-end RGB policy

v10.0只研究Task-driven Active View Selection。
