# EA-AVS MVP7.0 Code Generation Document

## 1. Version Positioning

Version: v7.0

Title:

**Humanoid-driven Active Perception Simulation Environment**

v7.0 is an infrastructure and experimental platform construction version.

The goal is to transform ACTIVEVIEW from an abstract/static human perception environment into a realistic indoor elderly monitoring simulation environment driven by human motion data.

The core pipeline is:

```
BABEL Action Annotation
        ↓
AMASS Human Motion
        ↓
Habitat Humanoid Agent
        ↓
Indoor Robot Perception Simulation
        ↓
RGB-D Observation Dataset
```

v7.0 does NOT propose the final active view selection algorithm. It provides the experimental foundation for future action-aware active perception research.

---

# 2. Research Boundary

## v7.0 focuses on:

1. Realistic humanoid representation.
2. Human motion loading and playback.
3. Habitat indoor scene integration.
4. Robot-mounted camera observation generation.
5. Ground-truth action/state annotation generation.

## Explicitly NOT included:

- Action-aware view utility optimization.
- NBV algorithm design.
- Reinforcement learning.
- Multi-step active perception.
- Evidence recovery mechanism.
- Real robot deployment.
- Human action recognition model training.
- SLAM/navigation/path planning implementation.

These are future research stages.

---

# 3. Scientific Role of v7.0

The research question of v7.0 is:

> Can we build a controllable indoor simulation environment where a mobile robot observes realistic elderly human activities from different viewpoints?

The output is not a final method, but a reliable benchmark environment for studying:

"How should a robot choose informative viewpoints for elderly activity perception?"

---

# 4. Input Data

## AMASS Motion

Location:

```
../../data/ActiveView/datasets/amass/
```

Validated motion sources:

- BMLrub
- CMU
- EKUT
- EyesJapanDataset
- KIT

Supported formats:

Schema A:

```
trans
root_orient
poses
fps
```

Schema B (standard AMASS):

```
trans
poses
fps
```

For Schema B:

```
root_orient = poses[:, :3]
```

---

# 5. v7.0 Implementation Pipeline

## Stage 1: Motion Normalization

Input:

```
AMASS npz
```

Output:

```
normalized human motion representation
```

Required fields:

```
translation
root_rotation
body_pose
fps
num_frames
```

---

## Stage 2: AMASS to Habitat Humanoid Motion

Use Habitat official humanoid conversion tools whenever possible.

Pipeline:

```
AMASS Motion
      ↓
Habitat-compatible motion
      ↓
Humanoid playback
```

Do not build a new human simulator.

---

## Stage 3: Humanoid Agent in Indoor Scene

Create:

```
ea_avs_mvp_v7/
```

Suggested structure:

```
ea_avs_mvp_v7/
├── configs/
├── humanoid/
├── motion/
├── simulation/
├── scripts/
└── tests/
```

Responsibilities:

- Load humanoid asset.
- Load converted motion.
- Place humanoid in Habitat scene.
- Execute motion playback.

---

## Stage 4: Robot Observation Generation

Generate:

```
RGB image
Depth image
Camera pose
Human pose ground truth
Action label
Timestamp
```

The generated data will support later active perception experiments.

---

# 6. Initial Elderly Action Set

The objective is not comprehensive action recognition.

Only select representative elderly indoor activities:

Required first-stage actions:

1. Standing
2. Sitting
3. Fall-related / fallen posture

Optional extension:

4. Bending
5. Reaching

The priority is realistic elderly monitoring scenarios, especially abnormal states such as falls.

---

# 7. MVP7.0 Minimum Success Criteria

The smallest successful experiment:

```
One Habitat indoor scene
        ↓
One humanoid
        ↓
One AMASS motion (prefer fall)
        ↓
Motion playback
        ↓
Robot camera observation
        ↓
RGB-D + ground truth saved
```

---

# 8. Engineering Constraints

1. Never hard-code machine-specific paths.

Use:

```
../../data/ActiveView
```

or:

```
ACTIVEVIEW_DATA_ROOT
```

2. Never commit AMASS/BABEL raw data.

3. Keep v6 read-only.

4. Every module must document:

- input
- output
- dependencies
- execution command

5. Prefer small verifiable milestones.

---

# 9. Future Research Direction (Reference Only)

Future versions may extend:

## v8.0

Action-aware Active View Selection:

- action hypothesis
- action-conditioned viewpoint utility
- viewpoint comparison

## v9.0

Uncertainty-aware Multi-step Active Perception:

- observation uncertainty
- evidence recovery
- sequential viewpoint planning

These are NOT part of v7.0 implementation.

---

# 10. Development Principle

v7.0 is the bridge between simulation infrastructure and scientific algorithm research.

The priority is:

**build a reliable humanoid-driven perception environment first, then study intelligent viewpoint selection.**
