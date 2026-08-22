# ACTIVEVIEW v11.0 Code Generation Specification

## 1. Version Objective

v11.0 implements the Active View Selection layer on top of the frozen v10.0 perception and action recognition pipeline.

v10.0 answers:

> Given an observation, how confident is the robot about human action recognition?

v11.0 answers:

> Given uncertainty and environmental constraints, where should the robot observe next to maximize action recognition performance?

The goal is **single-step active viewpoint selection**.

The robot observes the human from the current viewpoint, generates feasible candidate viewpoints, predicts the expected recognition utility of each candidate, selects one viewpoint, moves once, and performs final action recognition.

---

# 2. Research Boundary

## Included

- Candidate viewpoint generation.
- Habitat navigation feasibility filtering.
- Visibility checking.
- Viewpoint quality dataset generation.
- Candidate viewpoint scoring.
- Uncertainty-aware viewpoint selection.
- Navigation cost-aware optimization.

## Excluded

- Multi-step planning.
- Reinforcement learning navigation.
- SLAM.
- Real robot control.
- New pose estimation methods.
- New action recognition networks.

The robot is assumed to already operate in a local Habitat environment containing the human target.

---

# 3. System Architecture

```text
Current Observation
        |
        v
RGB
        |
3D Pose Estimator
        |
3D Skeleton
        |
ST-GCN
        |
Action Probability
        |
Current Uncertainty
        |
        v
Candidate View Generation
        |
Habitat Feasibility Filtering
        |
Viewpoint Utility Prediction
        |
Active View Selection
        |
Robot Navigation
        |
New Observation
        |
Final Action Recognition
```

---

# 4. Module 1: Candidate Viewpoint Generation

## Objective

Generate possible observation locations around the human.

Candidate generation must not directly assume that all viewpoints are reachable.

The process is:

```text
Human position
        |
Polar sampling
        |
Candidate viewpoints
        |
Habitat filtering
        |
Feasible viewpoints
```

---

## 4.1 Polar Sampling

Represent candidate viewpoint:

```
(position, yaw, distance)
```

Human-centered sampling:

```
x = x_h + r*cos(theta)
z = z_h + r*sin(theta)
```

Default configuration:

Angles:

```
0,45,90,135,180,225,270,315 degrees
```

Distances:

```
1.5m
2.0m
2.5m
3.0m
```

Total candidates:

```
8 x 4 = 32
```

The configuration must remain adjustable.

---

# 5. Module 1.2: Habitat Feasibility Filtering

Generated points must pass three constraints.

## Navigation Constraint

Check:

- point is on navigation mesh.
- point is located on valid ground.

Use Habitat navigation API.

---

## Reachability Constraint

Check shortest path:

```
current robot position
        |
        v
candidate viewpoint
```

Reject:

- unreachable points.
- excessive path cost.

---

## Visibility Constraint

Use ray casting.

Check:

```
robot viewpoint
        |
        v
human target
```

Reject viewpoints fully blocked by obstacles.

---

# 6. Candidate View Data Structure

Each viewpoint must contain:

```json
{
 "view_id": 1,
 "position": [x,y,z],
 "rotation": [yaw,pitch],
 "distance_to_human": 2.0,
 "angle_to_human": 90,
 "navigation_cost": 3.5,
 "visible": true
}
```

---

# 7. Module 2: Viewpoint Quality Dataset Generation

## Purpose

Build a mapping:

```
viewpoint
      |
      v
expected action recognition quality
```

This dataset is NOT used to train ST-GCN.

ST-GCN remains frozen from v10.0.

---

## Data Generation Pipeline

For each action sequence:

```text
AMASS motion
        |
Habitat human simulation
        |
Candidate viewpoint rendering
        |
RGB observation
        |
3D Pose Estimator
        |
ST-GCN
        |
Action probability
        |
Entropy label
```

---

## Stored Sample

```json
{
 "action": "walking",
 "viewpoint": {
   "angle":45,
   "distance":2.0
 },
 "probability": [],
 "entropy":0.32,
 "pose_confidence":0.85,
 "navigation_cost":2.4
}
```

---

# 8. Module 3: Viewpoint Utility Prediction

## Objective

Predict viewpoint value without physically moving to every candidate.

Incorrect approach:

```
Candidate 1 -> move -> test
Candidate 2 -> move -> test
```

This is oracle evaluation only.

Runtime policy must predict.

---

## Utility Function

Recommended objective:

```
Utility(v)=InformationGain(v)-lambda*NavigationCost(v)
```

where:

```
InformationGain = CurrentEntropy - PredictedFutureEntropy
```

---

## Predictor Input

Candidate features:

```
angle
 distance
 navigation cost
 visibility score
 pose confidence
 current entropy
```

Output:

```
predicted utility
```

Initial implementation may use MLP.

Do not introduce reinforcement learning in v11.0.

---

# 9. Module 4: Single-step Active View Policy

Runtime:

```text
Current observation
        |
Generate candidates
        |
Filter invalid viewpoints
        |
Predict utility
        |
Select maximum utility
        |
Navigate once
        |
Final recognition
```

Selection:

```
v* = argmax Utility(v)
```

---

# 10. Evaluation Protocol

Baselines:

1. Random viewpoint.
2. Nearest viewpoint.
3. Fixed frontal viewpoint.
4. Entropy-only selection.
5. Utility-based Active View.

Metrics:

Main:

- Top-1 action accuracy improvement.
- Entropy reduction.

Secondary:

- Navigation distance.
- Utility per meter.
- Pose confidence improvement.

---

# 11. Implementation Structure

Recommended structure:

```
ACTIVEVIEW/

active_view/

├── candidate_generator.py
├── habitat_filter.py
├── visibility_checker.py
├── viewpoint_dataset.py
├── utility_predictor.py
└── active_selector.py

experiments/

└── v11/

configs/

└── viewpoint_config.yaml
```

---

# 12. Development Stages

## v11.1 Candidate View Generation

Implement:

- polar sampling.
- navigation filtering.
- viewpoint metadata.

Acceptance:

Valid candidate viewpoints generated in Habitat.

---

## v11.2 View Quality Dataset

Implement:

- multi-view rendering.
- RGB perception.
- ST-GCN evaluation.
- entropy labels.

Acceptance:

Viewpoint quality dataset generated.

---

## v11.3 Utility Prediction

Implement:

- feature extraction.
- utility model.
- candidate ranking.

Acceptance:

Predicted ranking correlates with true viewpoint utility.

---

## v11.4 Active View Evaluation

Implement:

- single-step selection.
- navigation execution.
- final recognition comparison.

Acceptance:

Active View outperforms random/fixed baselines.

---

# 13. Development Constraints

Must:

- Keep v10.0 Pose Estimator unchanged.
- Keep v10.0 ST-GCN unchanged.
- Use estimated skeleton for online decision.
- Keep GT only for simulation labels and offline dataset generation.

Must not:

- Modify action recognition network.
- Train pose estimator.
- Add multi-step planning.
- Add reinforcement learning.

---

# Final Principle

v10.0:

"Understand what the robot sees."

v11.0:

"Decide where the robot should look."

ACTIVEVIEW becomes a complete active perception framework for human action recognition.
