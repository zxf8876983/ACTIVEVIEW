# EA-AVS MVP7.0 Code Generation Document

## 1. Version Positioning

Version: v7.0

Goal: Complete the transition from simulated fixed-body experiments to a real humanoid-driven indoor active perception environment.

v7.0 is NOT the final active view optimization algorithm. The purpose of this version is to establish a reliable simulation pipeline:

```
AMASS Motion
    ↓
Humanoid Motion Representation
    ↓
Habitat Humanoid Agent
    ↓
RGB-D Observation Generation
    ↓
Action-conditioned Active Perception Environment
```

The output of v7.0 should provide the experimental foundation for later Active View Selection research.

---

# 2. Scientific Scope

## v7.0 focuses on:

1. Realistic indoor human body representation.
2. Human motion playback in Habitat.
3. Robot camera observation generation.
4. Dataset generation for active perception experiments.

## Explicitly excluded:

- NBV optimization algorithm design.
- Reinforcement learning.
- Multi-step navigation planning.
- Real robot deployment.
- Human action recognition network training.
- Complex multi-person interaction.
- Evidence Recovery mechanism.

These belong to later versions.

---

# 3. Input Assets

## 3.1 AMASS Motion

Location:

```
../../data/ActiveView/datasets/amass/
```

Current validated datasets:

- BMLrub
- CMU
- EKUT
- EyesJapanDataset
- KIT

Motion format:

```
.npz
```

Supported schema:

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
root_orient = poses[:, :3]
```

---

# 4. v7.0 Core Pipeline

## Stage 1: Motion Loading

Implement:

```
tools/motion/
    amass_loader.py
```

Input:

```
AMASS npz path
```

Output:

```
normalized motion dictionary
```

Required fields:

```
translation
root_rotation
body_pose
fps
num_frames
```

No AMASS-specific format should leak into later modules.

---

# Stage 2: Humanoid Motion Conversion

Use Habitat official humanoid utilities.

Goal:

```
AMASS motion
      ↓
Habitat humanoid motion format
```

Output:

```
assets/motions/habitat/
```

Do not implement a new human body simulator.

Reuse Habitat supported humanoid pipeline whenever possible.

---

# Stage 3: Humanoid Agent Loading

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

## humanoid/

Load humanoid URDF and articulation.

Input:

```
Habitat scene
Humanoid asset
```

Output:

```
interactive humanoid agent
```

---

# Stage 4: Indoor Scene Integration

Use Habitat existing scene system.

Do not implement:

- SLAM
- mapping
- navigation
- collision avoidance

These are assumed robot capabilities.

The experiment focuses on perception viewpoint selection.

---

# Stage 5: RGB-D Observation Generation

Camera should simulate robot-mounted sensor.

Generate:

```
RGB image
Depth image
Camera pose
Human pose ground truth
Action label
Timestamp
```

Output:

```
data/ActiveView/runs/
```

Example:

```
frame_000001.png
frame_000001_depth.png
metadata.json
```

---

# 5. Initial Motion Set

Only validate three actions first:

## Required

1. Standing
2. Sitting
3. Fall-related posture

Optional later:

4. Bending
5. Reaching

Reason:

The research target is elderly monitoring, not action recognition coverage.

---

# 6. First Milestone (MVP7.0)

The minimum successful demonstration:

```
One indoor Habitat scene
        ↓
One humanoid model
        ↓
One AMASS motion
        ↓
Motion playback
        ↓
Robot camera captures RGB-D
        ↓
Saved observation sequence
```

A single falling motion is preferred as the final smoke test.

---

# 7. Evaluation Criteria

v7.0 is successful if:

- Humanoid loads correctly.
- Motion conversion succeeds.
- Motion playback is stable.
- Camera observations are generated.
- Human ground truth is available.
- Dataset generation pipeline works repeatedly.

No accuracy metric is required at this stage.

---

# 8. Implementation Constraints

1. Never hard-code absolute machine paths.

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

4. Add unit tests for every new module.

5. Prefer small runnable scripts over large frameworks.

6. Every script must clearly document:

- input
- output
- dependencies
- execution command

---

# 9. Future Versions

After v7.0:

v8.0:

```
Action-aware Active View Selection
```

including:

- action hypothesis
- observation uncertainty
- viewpoint utility

v9.0:

```
multi-step active perception
```

---

# 10. Development Philosophy

v7.0 is an infrastructure milestone.

The goal is not to create a complete robot system.

The goal is to build a controllable simulation environment where the scientific question can be studied:

"How should a mobile robot choose informative viewpoints for elderly activity perception?"
