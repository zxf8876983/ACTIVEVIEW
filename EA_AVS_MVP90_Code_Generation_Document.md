# EA-AVS MVP9.0 Code Generation Document

## Action-conditioned Active View Scoring

Version: v9.0
Status: Development Planning

---

# 1. Version Overview

## 1.1 Purpose

MVP9.0 is the first research-oriented algorithm version of ACTIVEVIEW. Previous versions mainly focused on building a reliable simulation environment and geometric active view baseline. MVP9.0 introduces action-conditioned viewpoint evaluation and establishes the foundation for future learnable active perception methods.

The core idea is:

> The optimal observation viewpoint is not only determined by geometry, but also depends on the current human activity state.

---

# 2. Relationship with Previous Versions

## v7 Dynamic Human Simulation

Goal:

Build a realistic indoor human motion simulation pipeline.

Capabilities:

- Habitat environment
- Humanoid embodiment
- SMPL-X/BABEL/AMASS motion integration
- RGB observation generation

## v8 Local Active View Planning Baseline

Goal:

Given known human location, select a geometrically good viewpoint.

Pipeline:

Human location

-> candidate viewpoint generation

-> feasibility filtering

-> geometry quality evaluation

-> best viewpoint selection

## v9 Action-conditioned Active View Scoring

Goal:

Extend geometry-only viewpoint selection to task-aware viewpoint selection.

---

# 3. Scientific Problem Definition

Given:

- Indoor environment E
- Human state H
- Action state A
- Robot state R
- Candidate viewpoints V

Find:

v* = argmax Q(v|H,A,E)

where Q is an action-conditioned observation quality function.

---

# 4. Research Boundary

## Included

MVP9.0 includes:

- Action label integration
- Action representation
- Action-conditioned scoring
- View ranking
- Baseline comparison

## Explicitly excluded

The following are NOT part of MVP9.0:

- Human detection
- Human localization
- SLAM
- Global exploration
- Path planning
- Reinforcement learning
- End-to-end policy learning
- Action recognition

Reason:

The research focus is active viewpoint selection after coarse human localization is available.

---

# 5. Core Hypothesis

Geometry-only viewpoint selection assumes:

Q(v)

However, elderly monitoring tasks have different observation requirements.

Therefore:

Q(v|a) != Q(v)

Different actions should produce different optimal viewpoints.

Examples:

## Fall

Important factors:

- whole body visibility
- body orientation
- ground relationship

## Sitting

Important factors:

- lower body visibility
- chair relationship

## Bending

Important factors:

- torso angle
- posture change

---

# 6. Overall Architecture

```
v8 Episode Data
      |
      v
Human State + Action State
      |
      v
Action Representation Module
      |
      v
Action-conditioned View Scoring
      |
      v
View Ranking
      |
      v
Best View Selection
```

---

# 7. Input Interface Definition

## 7.1 Human State Interface

Input:

- SMPL-X joints
- body pose
- orientation

Example:

```json
{
 "joints": [],
 "orientation": []
}
```

---

## 7.2 Action Interface

MVP9.0 uses explicit action labels from BABEL.

Example:

```json
{
 "action": "fall"
}
```

Supported initial actions:

- fall
- sitting
- standing
- bending
- reaching

---

## 7.3 Candidate View Interface

Inherited from v8.

Each viewpoint contains:

- position
- rotation
- distance
- visibility
- pose coverage

---

# 8. Action Representation Module

Directory:

```
action/
```

MVP9.0 uses simple interpretable representation.

Default:

one-hot encoding.

Example:

fall:

```
[1,0,0,0,0]
```

Future versions may replace this with:

- temporal encoder
- graph neural network
- transformer encoder

---

# 9. Action-conditioned View Scoring

Directory:

```
scoring/
```

Input:

- geometry feature
- visibility feature
- human feature
- action feature

Output:

view score.

General formulation:

Q(v|a)=Q_geometry(v)+DeltaQ(action,v)

---

# 10. Implementation Strategy

MVP9.0 does NOT directly introduce deep learning.

The first goal is to prove:

"Action state changes viewpoint preference."

Therefore use interpretable action-aware rules.

Example:

Fall:

increase whole-body visibility weight.

Sitting:

increase lower-body visibility weight.

Bending:

increase torso visibility weight.

---

# 11. Code Structure

```
ea_avs_mvp_v9/

├── action/
├── features/
├── scoring/
├── selection/
├── dataset/
├── evaluation/
├── visualization/
├── scripts/
└── tests/
```

---

# 12. Dataset Extension

Inherited from v8 episode format.

Additional files:

```
action.json
view_action_scores.json
```

Example:

```json
{
 "motion_id":"xxx",
 "action":"fall"
}
```

---

# 13. Baseline Comparison

Required comparison:

## Random View

Random feasible viewpoint.

## Nearest View

Closest feasible viewpoint.

## Geometry Best

v8 method.

## Action-conditioned View

MVP9.0 method.

---

# 14. Evaluation Metrics

Geometry metrics:

- visibility
- pose coverage
- distance
- visibility loss

Action-related metrics:

- action-specific body coverage
- task relevant observation quality
- ranking improvement over v8

---

# 15. Development Phases

## Phase 1

Integrate action labels.

Acceptance:

Action information can be loaded with episode.

## Phase 2

Implement action-conditioned scoring.

Acceptance:

Different actions produce different scores.

## Phase 3

Baseline comparison.

Acceptance:

Generate quantitative comparison.

## Phase 4

Visualization.

Acceptance:

Same human state with different actions generates different preferred viewpoints.

---

# 16. Acceptance Criteria

The implementation is accepted only when:

- v8 pipeline remains functional
- action information is correctly loaded
- scoring depends on action state
- baseline comparison is available
- visualization demonstrates viewpoint changes
- no navigation or human search modules are introduced

---

# 17. Future Extension Interface

## v9.1

Learnable action-conditioned view scorer.

## v9.2

Temporal action-aware active perception.

## v9.3

Uncertainty-aware viewpoint selection.

## v10

Closed-loop active perception system.

---

# 18. Git Rules

Do not commit:

- AMASS raw data
- BABEL raw annotations
- cache files
- generated videos

Commit only:

- source code
- configuration
- documentation
- lightweight metadata

---

# 19. Final Goal

MVP9.0 aims to establish the first action-conditioned active view scoring baseline and provide a stable foundation for future learning-based active perception algorithms.
