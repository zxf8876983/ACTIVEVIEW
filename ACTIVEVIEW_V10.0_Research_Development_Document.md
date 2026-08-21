# ACTIVEVIEW v10.0 Research Development Document

## Title

**RGB-D Driven Task-aware Active View Selection for Human Action Recognition**

## 1. Version Objective

v10.0 is the transition from perception-quality-driven active view selection to task-performance-driven active perception.

Previous versions:

- v8: geometry-based next-best-view selection.
- v9.0: task-prior heuristic view scoring.
- v9.1: perception-aware information gain selection based on estimated observation quality.

v10.0 research question:

> Given incomplete robot observations, can a robot actively select a viewpoint that maximizes downstream human action recognition performance?

The optimization target changes from "seeing better" to "recognizing better".

---

# 2. Research Boundary

## Included

- Active viewpoint selection policy.
- RGB-D based human perception pipeline.
- 3D skeleton extraction from robot observations.
- ST-GCN based action recognition evaluation.
- Action recognition gain based viewpoint scoring.

## Excluded

Not studied in v10.0:

- Robot navigation.
- SLAM.
- New pose estimation algorithms.
- New action recognition architectures.
- Real robot deployment.

Assumption:

The robot has already reached the local area around the human and can move to candidate viewpoints.

---

# 3. Core System Pipeline

```
Habitat Scene
    |
Human Motion
    |
RGB-D Observation
    |
2D Pose Estimator
    |
Depth Back Projection
    |
Estimated 3D Skeleton
    |
ST-GCN Action Recognition
    |
Action Probability / Uncertainty
    |
Active View Selector
    |
Next Best View
    |
Improved Action Recognition
```

---

# 4. Data and Information Boundary

Runtime data root must remain compatible with previous versions:

```
/home/zxf/WorkSpace/code/data/ActiveView/
```

The v10.0 document does not create an independent dataset root.

v10.0 inherits:

- Habitat scenes.
- Humanoid assets.
- Motion assets.
- Viewpoint generation infrastructure.
- Existing v7-v9 simulation pipeline.

---

# 5. Data Organization

Recommended extension:

```
ActiveView/

├── datasets/
│   ├── v7/
│   ├── v9/
│   └── v10/
│       ├── raw/
│       │   ├── rgb/
│       │   ├── depth/
│       │   ├── camera_pose/
│       │   └── scene_meta/
│       ├── perception/
│       │   ├── pose2d/
│       │   ├── pose3d/
│       │   └── confidence/
│       ├── action/
│       │   ├── labels/
│       │   └── stgcn_format/
│       ├── viewpoints/
│       └── metadata/
├── checkpoints/
└── experiments/
```

---

# 6. GT Information Rule

Ground truth is only allowed for:

1. Habitat simulation.
2. Action labels.
3. Oracle upper bound evaluation.

GT human pose cannot enter online viewpoint selection.

Online pipeline must use:

RGB-D -> estimated pose -> action recognition.

---

# 7. Human Pose Module

Input:

Robot RGB-D observation.

Pipeline:

```
RGB
 |
2D Pose Estimator
 |
2D keypoints

Depth
 |
Camera projection
 |
3D skeleton reconstruction
```

Recommended estimators:

- RTMPose.
- MediaPipe Pose.

No pose estimator training is performed.

---

# 8. Skeleton Representation

Input format:

```
T x J x 3
```

where:

- T: temporal frames.
- J: joints.
- XYZ: 3D coordinates.

Normalization:

Required:

- Root centering.
- Scale normalization.

Not allowed:

- Camera-facing rotation normalization.

Reason:

Viewpoint variation is the research variable.

---

# 9. Action Recognition Module

Model:

ST-GCN.

Purpose:

Provide task-level evaluation rather than proposing a new recognizer.

Input:

Estimated 3D skeleton sequence.

Output:

Action probability distribution.

---

# 10. Action Categories

Initial six classes:

1. standing
2. walking
3. sitting
4. bending
5. reaching
6. falling

Complex elderly activities are reserved for future versions.

---

# 11. Dataset Generation Protocol

Data generation:

1. Select Habitat scene.
2. Load humanoid.
3. Replay motion sequence.
4. Render RGB-D observations from candidate viewpoints.
5. Run pose estimation pipeline.
6. Generate skeleton/action samples.

Dataset split must be scene-level.

No random frame split is allowed.

Purpose:

Avoid environment leakage.

---

# 12. Active View Selection

Input:

Current observation:

- estimated 3D pose.
- joint confidence.
- action probability.
- camera state.

Candidate viewpoint:

- relative position.
- distance.
- angle.
- geometry features.

Output:

Ranking score of candidate viewpoints.

---

# 13. Optimization Objective

The objective is action recognition improvement.

Preferred metric:

Action uncertainty reduction.

Example:

```
Gain(v)=Entropy(before)-Entropy(after selecting v)
```

Alternative:

Confidence improvement.

The selected viewpoint maximizes expected action recognition gain.

---

# 14. Experimental Protocol

Baselines:

1. Random viewpoint.
2. Nearest viewpoint.
3. Geometry based viewpoint (v8).
4. Perception aware viewpoint (v9.1).
5. Action aware viewpoint (v10).
6. Oracle viewpoint.

Metrics:

Main:

- Top-1 action accuracy.
- Accuracy improvement.

Secondary:

- Entropy reduction.
- Confidence gain.
- Pose confidence improvement.

---

# 15. Ablation Studies

Required:

- Remove action uncertainty.
- Remove pose feature.
- Remove geometry feature.
- Compare GT pose Oracle.
- Compare estimated pose.

---

# 16. Acceptance Criteria

System level:

- Habitat RGB-D generation works.
- Estimated 3D pose pipeline works.
- ST-GCN training and inference works.
- Active viewpoint ranking works.

Research level:

- Active View improves action recognition accuracy.
- Improvement increases under occlusion.
- Performance approaches Oracle upper bound.

---

# 17. Future Extension

v10.1:

Action uncertainty based scoring refinement.

v10.2:

More scenes and activities.

v11:

Real-world perception and deployment.

---

## Final Principle

ACTIVEVIEW evolves from:

"Where can the robot see more?"

into:

"Where should the robot observe to understand human actions better?"
