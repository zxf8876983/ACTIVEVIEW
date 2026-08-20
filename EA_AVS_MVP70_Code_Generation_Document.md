# EA-AVS MVP7.0 Code Generation Document

# Humanoid-driven Active Perception Simulation Environment

Version: MVP7.0

---

# 1. Version Objective

MVP7.0 is the bridge version between the previous ACTIVEVIEW active-view framework (v1-v6) and future intelligent active perception algorithms.

The goal is NOT to design NBV or viewpoint optimization algorithms.

The goal is to build a reliable simulation platform where a mobile robot observes realistic elderly human activities driven by real human motion data.

Main pipeline:

```
BABEL Action Annotation
        ↓
AMASS Motion Asset
        ↓
Motion Normalization
        ↓
Humanoid Agent
        ↓
Habitat Indoor Environment
        ↓
Robot RGB-D Observation
        ↓
Episode Dataset
```

---

# 2. Relationship with v1-v6

v1-v6 solved:

- active view simulation framework
- robot viewpoint representation
- observation evaluation
- candidate viewpoint reasoning

However, human representation was simplified.

v7.0 upgrades:

```
Static human state
        ↓
Motion-driven humanoid state
```

The purpose is to provide realistic human dynamics for future action-aware perception research.

---

# 3. Scientific Boundary

## Included in v7.0

- AMASS/BABEL motion integration
- Humanoid loading
- Human motion playback
- Habitat indoor scene simulation
- Robot sensor simulation
- RGB-D generation
- Human pose/action ground truth recording
- Episode dataset generation

## Forbidden in v7.0

Do NOT implement:

- NBV optimization
- Action-aware viewpoint utility
- RL policy
- Multi-step active perception
- Evidence recovery
- Action recognition network
- SLAM
- Navigation planner
- Collision avoidance
- Real robot deployment

These belong to future versions.

---

# 4. Engineering Goal

The final demonstration:

```
Habitat scene
      ↓
Humanoid loaded
      ↓
AMASS elderly motion played
      ↓
Robot camera observes human
      ↓
RGB-D + metadata saved
```

---

# 5. Directory Specification

Create:

```
ea_avs_mvp_v7/

├── configs/
│   ├── habitat_config.yaml
│   ├── humanoid_config.yaml
│   ├── motion_config.yaml
│   └── sensor_config.yaml
│
├── core/
│   ├── paths.py
│   ├── episode.py
│   └── config.py
│
├── environment/
│   ├── habitat_env.py
│   └── scene_manager.py
│
├── robot/
│   ├── robot_agent.py
│   └── rgbd_sensor.py
│
├── human/
│   ├── humanoid_agent.py
│   ├── human_state.py
│   └── action_state.py
│
├── motion/
│   ├── amass_loader.py
│   ├── motion_converter.py
│   └── motion_player.py
│
├── observation/
│   ├── recorder.py
│   └── metadata.py
│
├── dataset/
│   └── episode_generator.py
│
├── evaluation/
│
├── scripts/
│
└── tests/
```

Do not create an independent Habitat project.

---

# 6. Module Interface Requirements

## motion/amass_loader.py

Input:

```
AMASS npz path
```

Output:

```
NormalizedMotion
```

Fields:

```
translation
root_rotation
body_pose
fps
num_frames
```

Support:

Schema A:

```
trans
root_orient
poses
fps
```

Schema B:

```
trans
poses
fps
```

For Schema B:

```
root_rotation = poses[:, :3]
```

The rest of the system must not depend on AMASS raw format.

---

## human/humanoid_agent.py

Responsible for:

- loading humanoid asset
- placing humanoid
- applying motion
- exposing human state

Input:

```
Habitat simulator
Motion object
```

Output:

```
HumanState
```

Contains:

```
position
orientation
pose
frame_id
action_label
```

Do not implement a new humanoid engine.

---

## environment/habitat_env.py

Responsible for:

- simulator initialization
- scene loading
- simulation stepping

Forbidden:

- SLAM
- path planning
- navigation algorithm

---

## robot/rgbd_sensor.py

Input:

```
robot camera pose
Habitat observation
```

Output:

```
RGB
Depth
Camera pose
```

---

## dataset/episode_generator.py

Generate:

```
episode_id
scene_id
robot_pose
human_pose
action_label
motion_id
RGB
Depth
camera_pose
timestamp
```

---

# 7. Data Rules

External data root:

```
../../data/ActiveView
```

or:

```
ACTIVEVIEW_DATA_ROOT
```

Never hard-code machine paths.

Never commit:

- AMASS data
- BABEL raw annotations
- generated datasets

---

# 8. Elderly Action Scope

First milestone only:

1. Standing
2. Sitting
3. Fall-related posture

Optional:

4. Bending
5. Reaching

The purpose is elderly monitoring, not action recognition benchmark construction.

---

# 9. Development Milestones

## MVP7.0-M1 Humanoid Smoke Test

Must achieve:

```
scene loaded
+ humanoid loaded
+ one motion playback
```


## MVP7.0-M2 Observation Pipeline

Must generate:

```
RGB
Depth
Camera pose
```


## MVP7.0-M3 Episode Dataset

Must generate:

```
RGB sequence
Depth sequence
Human GT
Action label
Metadata
```

---

# 10. Testing Requirements

Every module must provide tests.

Minimum tests:

- AMASS loading test
- motion conversion test
- humanoid loading test
- Habitat scene loading test
- RGB-D generation test
- episode saving test

---

# 11. Future Direction

v8.0 (reference only):

Action-aware Active View Selection.

v9.0 (reference only):

Uncertainty-aware Multi-step Active Perception.

Neither is implemented in v7.0.

---

# 12. Final Acceptance Criteria

v7.0 is completed only when:

1. Habitat scene runs.
2. Humanoid appears correctly.
3. AMASS motion plays.
4. Robot RGB-D observation works.
5. Human state/action GT is saved.
6. Dataset generation is reproducible.

v7.0 success means a reliable experimental environment exists for future active perception algorithms.
