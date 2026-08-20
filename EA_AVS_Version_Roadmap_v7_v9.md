# ACTIVEVIEW v7.0-v9.0 Research Roadmap

## Overall Research Goal

面向独居老人异常动作感知的移动机器人主动视角选择方法。

核心科学问题：

> 在室内动态人体场景中，移动机器人如何主动选择更有信息量的观察视角，提高老人状态和异常动作感知能力？

v7-v9 不应该同时解决所有问题，而采用逐步递进路线。

---

# v7.0: Humanoid-driven Active Perception Simulation Environment

## 定位

v7.0 是实验基础平台版本，不提出主动视角优化算法。

目标：解决“机器人面对真实人体动作场景时如何获得可靠观测数据”的问题。

核心链路：

```
AMASS/BABEL Motion
        ↓
Humanoid Motion Conversion
        ↓
Habitat Indoor Scene
        ↓
Robot Camera Observation
        ↓
RGB-D + Human State Ground Truth
```

## 主要任务

### 1. 人体动作资产

使用：

- AMASS
- BABEL

重点老人相关动作：

- standing
- sitting
- bending
- reaching
- fall-related posture

重点验证：

- 跌倒
- 倒地状态

不追求动作类别数量。

### 2. Humanoid构建

完成：

- SMPL-X/AMASS动作加载
- Habitat humanoid motion conversion
- Humanoid动作播放

### 3. 主动感知数据生成

输出：

- RGB image
- Depth image
- Camera pose
- Human pose GT
- Action label
- Scene information

## v7.0 不做

- NBV优化
- 强化学习
- 动作识别网络
- 多步规划
- 真实机器人部署

## v7.0成功标准

一个室内场景中：

```
Humanoid
   ↓
播放fall动作
   ↓
机器人相机观察
   ↓
生成RGB-D数据
```

---

# v8.0: Action-aware Active View Selection

## 定位

在v7数据基础上，研究动作条件主动视角选择方法。

核心问题：

> 不同人体状态和动作下，机器人是否应该选择不同观察视角？

## 输入

机器人当前状态：

- robot pose
- camera observation
- environment information

人体状态：

- human location
- body pose
- orientation
- action hypothesis

## 核心方法

设计：

Action-conditioned View Utility

例如：

```
View utility
=
f(visibility,
action evidence,
body region,
uncertainty)
```

比较：

1. Action-agnostic NBV
2. Geometry-based NBV
3. Action-aware NBV

## 实验

不同动作：

- standing
- sitting
- bending
- fall

评价：

- action recognition accuracy
- visible body evidence
- viewpoint efficiency

## v8.0创新重点

从“看哪里”扩展为：

> “为了确认人体状态，机器人应该看哪里”。

---

# v9.0: Uncertainty-aware Multi-step Active Perception

## 定位

从一次视角选择扩展到连续主动感知过程。

## 核心问题

> 当一次观察无法确定老人状态时，机器人如何主动探索并减少不确定性？

## 研究内容

### 1. Observation Uncertainty

建模：

```
P(action | observation)
```

### 2. Evidence Recovery

研究：

- 当前视角缺失哪些关键证据
- 下一视角如何补充信息

### 3. Multi-step View Planning

流程：

```
Observe
 ↓
Estimate uncertainty
 ↓
Select next view
 ↓
Move
 ↓
Observe again
```

## 评价

比较：

- one-shot NBV
- random exploration
- uncertainty-aware active perception

指标：

- final recognition accuracy
- number of movements
- observation efficiency

---

# 总体演进

```
v6
基础主动视角框架

 ↓

v7
真实人体动作仿真环境

 ↓

v8
动作感知主动视角选择

 ↓

v9
不确定性驱动多步主动感知
```

## 开发原则

1. 每个版本只解决一个科学问题。
2. 后一个版本必须建立在前一个版本实验基础上。
3. 不提前引入后续版本概念。
4. v7保证数据真实性，v8研究主动选择，v9研究长期决策。
