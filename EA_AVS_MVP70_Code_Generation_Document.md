# EA-AVS MVP7.0 Code Generation Document

## 1. Version Positioning

Version: v7.0

Title:

**Humanoid-driven Active Perception Simulation Environment**

v7.0 is the bridge version between previous abstract active-view simulation and future intelligent active perception research.

The purpose of v7.0 is NOT to propose a new NBV algorithm. The purpose is to construct a reliable, extensible, and reproducible experimental environment where a mobile robot can observe realistic human activities in indoor scenes.

Core pipeline:

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
Active Perception Dataset
```

---

# 2. Relationship With Previous Versions

v1-v6 mainly studied:

```
Robot viewpoint selection
        ↓
Observation quality
        ↓
Active perception framework
```

However, previous versions simplified human representation.

v7.0 introduces realistic dynamic humans:

```
Static human model
        ↓
Motion-driven humanoid
```

The purpose is to provide realistic human state changes for future action-aware active perception research.

---

# 3. Research Boundary

## v7.0 Includes

1. AMASS/BABEL motion asset integration.
2. Humanoid loading and motion playback.
3. Habitat indoor scene integration.
4. Robot-mounted sensor simulation.
5. RGB-D observation generation.
6. Human motion/action ground truth recording.
7. Dataset generation pipeline.

## v7.0 Does NOT Include

The following are future research topics:

- Action-aware viewpoint utility.
- NBV optimization algorithm.
- Learned viewpoint policy.
- Reinforcement learning.
- Multi-step active perception.
- Evidence recovery.
- Human action recognition network.
- SLAM.
- Navigation planning.
- Real robot deployment.

Do not implement these features in v7.0.

---

# 4. Scientific Goal

The scientific goal of v7.0 is:

> Build a controllable indoor simulation environment where a mobile robot observes realistic elderly human activities from different viewpoints.

The final output of v7.0 is an experimental platform, not a perception algorithm.

---

# 5. Project Directory Structure

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
│   └── registry.py
│
├── environment/
│   ├── habitat_env.py
│   └── scene_manager.py
│
├── robot/
│   ├── robot_agent.py
│   └── rgbd_camera.py
│
├── human/
│   ├── humanoid_agent.py
│   ├── motion_loader.py
│   └── action_state.py
│
├── observation/
│   ├── observation_recorder.py
│   └── dataset_writer.py
│
├── datasets/
│   └── episode_generator.py
│
├── evaluation/
│   └── basic_metrics.py
│
├── scripts/
│   ├── run_smoke_test.py
│   └── generate_dataset.py
│
├── tests/
│
└── README.md
```

Do not create a completely independent Habitat project.
The code must remain part of ACTIVEVIEW version evolution.

---

# 6. Module Responsibilities

## 6.1 core/

### paths.py

Purpose:

Unified data path management.

Input:

- repository location
- ACTIVEVIEW_DATA_ROOT(optional)

Output:

- dataset paths
- asset paths
- output paths

Never hard-code:

```
/home/zxf/...
```

---

### episode.py

Purpose:

Define one simulation episode.

An episode contains:

```
scene
robot initial state
human state
motion
observation sequence
metadata
```

---

# 7. Human Motion Pipeline

## 7.1 motion_loader.py

Input:

```
AMASS npz
```

Output:

Normalized motion object:

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
root_rotation = poses[:,0:3]
```

---

## 7.2 humanoid_agent.py

Responsibilities:

- Load humanoid asset.
- Initialize pose.
- Attach motion.
- Control playback.

Input:

```
Habitat simulator
Humanoid asset
Motion object
```

Output:

```
Dynamic humanoid in scene
```

Do not implement a new physics humanoid system.
Reuse Habitat-supported mechanisms.

---

# 8. Environment Pipeline

## habitat_env.py

Responsibilities:

- Initialize Habitat simulator.
- Load indoor scene.
- Manage simulation step.

Do NOT implement:

- SLAM
- map building
- navigation planner
- obstacle avoidance

These are assumed robot capabilities.

---

# 9. Robot Observation Pipeline

## rgbd_camera.py

Input:

```
Robot camera pose
Habitat environment
```

Output:

```
RGB image
Depth image
Camera parameters
```

---

## observation_recorder.py

Save:

```
RGB
Depth
Camera pose
Human pose GT
Action label
Timestamp
```

Example:

```
runs/
 └── episode_0001/
      ├── rgb/
      ├── depth/
      └── metadata.json
```

---

# 10. Elderly Action Set

The goal is elderly monitoring, not action recognition benchmark construction.

First-stage actions:

1. Standing
2. Sitting
3. Fall-related / fallen posture

Second-stage optional:

4. Bending
5. Reaching

Priority:

Fall-related motion has highest importance.

Do not add large-scale action categories.

---

# 11. Development Milestones

## MVP7.0-M1: Smoke Test

Requirement:

```
One Habitat scene
+
One humanoid
+
One AMASS motion
+
Motion playback
```

Success:

Humanoid can move correctly.

---

## MVP7.0-M2: Observation Generation

Requirement:

```
Humanoid motion
        ↓
Robot camera
        ↓
RGB-D output
```

Success:

Dataset files generated successfully.

---

## MVP7.0-M3: Dataset Pipeline

Generate:

```
RGB
Depth
Pose GT
Action label
Camera pose
```

Success:

Repeatable episode generation.

---

# 12. Engineering Constraints

1. Keep v6 read-only.

2. Use:

```
../../data/ActiveView
```

or:

```
ACTIVEVIEW_DATA_ROOT
```

3. Never commit:

- AMASS npz
- BABEL raw data
- generated datasets

4. Every module must provide:

- input
- output
- dependencies
- execution command

5. Prefer minimal runnable demonstrations before extension.

---

# 13. Future Direction (Reference Only)

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

These are not v7.0 tasks.

---

# 14. Final Acceptance Criteria

v7.0 is completed when:

1. Habitat indoor scene loads.
2. Humanoid loads successfully.
3. AMASS motion plays correctly.
4. Robot RGB-D observation is generated.
5. Human state/action ground truth is recorded.
6. Dataset generation can be repeated.

After completion, ACTIVEVIEW is ready for future action-aware active perception research.
