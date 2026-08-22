# ACTIVEVIEW v11.0 Research Development Document

## Title

**Navigation-Constrained Active View Selection for Human Action Recognition**

---

# 1. Version Objective

v11.0 extends ACTIVEVIEW from passive action recognition evaluation to true robot active perception.

v10.0 establishes the perception and recognition foundation:

```
RGB
 ↓
3D Pose Estimator
 ↓
3D Skeleton
 ↓
ST-GCN
 ↓
Action Probability
 ↓
Uncertainty
```

v11.0 research question:

> Given a limited set of reachable viewpoints, can a robot predict the viewpoint that maximizes future human action recognition reliability while minimizing movement cost?

The optimization target changes from:

"recognize from the current observation"

into:

"actively choose where to observe next".

---

# 2. Research Boundary

## Included

- Candidate viewpoint generation.
- Habitat navigation feasibility filtering.
- Viewpoint quality dataset construction.
- Action recognition gain prediction.
- Single-step active viewpoint selection.
- Navigation cost-aware viewpoint ranking.

## Excluded

Not studied in v11.0:

- Multi-step long horizon planning.
- Reinforcement learning navigation.
- SLAM.
- Full robot motion planning.
- New pose estimation models.
- New action recognition networks.

Assumption:

The robot is already located near the human and can navigate to feasible local viewpoints.

---

# 3. Overall System Pipeline

```
Current Observation
        |
        ↓
3D Pose Estimator
        |
        ↓
ST-GCN
        |
        ↓
Current Action Uncertainty
        |
        ↓
Candidate Viewpoint Generation
        |
        ↓
Habitat Feasibility Filtering
        |
        ↓
Candidate View Set
        |
        ↓
Viewpoint Utility Prediction
        |
        ↓
Cost-aware View Ranking
        |
        ↓
Best Viewpoint
        |
        ↓
Robot Moves Once
        |
        ↓
Final Action Recognition
```

---

# 4. Relationship with v10.0

v10.0:

> Evaluate observation quality.

v11.0:

> Decide how to improve observation quality.

v10.0 modules are frozen:

- Pose estimator.
- Skeleton representation.
- ST-GCN.
- Uncertainty calculation.

v11.0 only adds active decision modules.

---

# 5. Candidate Viewpoint Generation

## Objective

Generate possible robot observation positions around the human.

The viewpoint space is represented using polar sampling:

```
position = human_position + (r*cos(theta), 0, r*sin(theta))
```

Parameters:

## Horizontal angle

Initial setting:

```
0,45,90,135,180,225,270,315 degrees
```

## Distance

Initial setting:

```
1.5m
2.0m
2.5m
3.0m
```

Total theoretical candidates:

```
8 × 4 = 32 viewpoints
```

Distance serves two purposes:

1. Increase feasible candidate density under indoor obstacles.
2. Provide movement-cost optimization.

---

# 6. Habitat Feasibility Filtering

Generated viewpoints cannot directly enter selection.

Filtering stages:

## 6.1 Navigation Validity

Check:

- On navigation mesh.
- Not inside obstacles.
- Ground contact.

## 6.2 Reachability

Use Habitat pathfinding:

Input:

```
robot current position
        ↓
candidate viewpoint
```

Remove unreachable viewpoints.

## 6.3 Visibility Constraint

Use ray casting:

Check:

```
robot viewpoint
        ↓
human body
```

Remove fully occluded viewpoints.

Output:

```
Theoretical candidates
        ↓
Feasible viewpoints
```

---

# 7. Viewpoint Quality Dataset

## Purpose

Build supervision for viewpoint utility prediction.

Important:

This dataset does NOT train ST-GCN.

ST-GCN remains frozen.

---

## Data Generation

For each motion sample:

```
Human motion
        ↓
Habitat scene
        ↓
Candidate viewpoint
        ↓
RGB rendering
        ↓
3D Pose Estimator
        ↓
ST-GCN
        ↓
Action probability
        ↓
Entropy label
```

Each viewpoint obtains:

```
viewpoint → recognition quality
```

---

# 8. Viewpoint Dataset Format

Each sample contains:

```json
{
 "action": "walking",
 "viewpoint": {
    "angle": 90,
    "distance": 2.0
 },
 "navigation_cost": 1.8,
 "pose_confidence": 0.85,
 "action_probability": [],
 "entropy": 0.32
}
```

---

# 9. Viewpoint Utility Prediction

Goal:

Predict future observation quality without moving the robot.

Input:

Current state:

- Current entropy.
- Current action probability.
- Current pose confidence.

Candidate viewpoint features:

- Angle.
- Distance.
- Navigation cost.
- Visibility score.

Output:

Predicted future quality:

```
Expected entropy
```

or:

```
Expected action recognition gain
```

---

# 10. Single-step Active View Policy

The first version uses single-step selection.

Decision rule:

```
Score(v)=Gain(v)-lambda*Cost(v)
```

where:

```
Gain(v)=H(current)-H(predicted viewpoint)
```

Select:

```
argmax Score(v)
```

Then:

```
Move once
 ↓
Capture observation
 ↓
Recognize action
```

---

# 11. Evaluation Protocol

Baselines:

1. Random viewpoint.
2. Nearest viewpoint.
3. Fixed front viewpoint.
4. Entropy-only selection.
5. Proposed gain-cost selection.

Metrics:

Main:

- Top-1 action accuracy improvement.

Secondary:

- Entropy reduction.
- Confidence improvement.
- Navigation distance.
- Utility efficiency.

---

# 12. Ablation Studies

Required:

- Without movement cost.
- Without uncertainty.
- Without viewpoint geometry features.
- Different candidate densities.
- Different distance ranges.

---

# 13. Implementation Structure

Suggested:

```
ActiveView/

├── viewpoint_generation/
│   ├── candidate_sampler.py
│   └── habitat_filter.py
│
├── viewpoint_dataset/
│   └── build_view_quality_dataset.py
│
├── viewpoint_policy/
│   ├── utility_predictor.py
│   └── selector.py
│
└── experiments/
```

---

# 14. Development Stages

## v11.1 Candidate Generation

Implement:

- polar sampling.
- navigation filtering.
- visibility filtering.

## v11.2 View Quality Analysis

Generate:

```
viewpoint → entropy map
```

Verify viewpoint influence.

## v11.3 Utility Predictor

Train lightweight predictor.

## v11.4 Active Selection

Implement:

single-step viewpoint selection.

---

# 15. Acceptance Criteria

System:

- Candidate viewpoints generated.
- Habitat filtering works.
- Viewpoint dataset generated.
- Utility predictor works.
- Robot selects viewpoint automatically.

Research:

- Active view improves action recognition.
- Gain-cost strategy outperforms random/fixed strategies.
- Improvement increases under occlusion.

---

# Final Principle

ACTIVEVIEW evolves as:

v10.0:

"How well can the robot understand human actions from an observation?"

v11.0:

"Where should the robot observe to understand human actions better?"
