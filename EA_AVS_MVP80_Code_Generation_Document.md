# EA-AVS MVP v8.0 Code Generation Document

## 1. Version Overview

**Version:** v8.0

**Position:** Human-aware Active View Foundation

v8.0 is the first research-oriented version after the completion of v7.0 simulation infrastructure.

v7.0 solved:

- Habitat indoor simulation environment
- Humanoid loading and rendering
- AMASS/BABEL motion pipeline
- RGB-D observation generation
- Human pose ground truth generation
- Episode dataset generation

v8.0 does NOT redesign the environment. It builds the foundation for active perception research.

The goal is to transform the system from:

> fixed human + fixed robot observation

into:

> constrained human-aware observation space generation.

---

# 2. Research Objective

## 2.1 Main Objective

Establish a human-aware active view foundation framework for indoor elderly monitoring scenarios.

Given:

- indoor environment
- dynamic humanoid
- robot platform
- sensor configuration

v8 generates feasible candidate viewpoints that satisfy:

1. physical feasibility
2. navigation feasibility
3. human visibility requirements
4. observation quality requirements

---

# 3. Research Boundary

## 3.1 Included in v8

v8 includes:

- human placement
- robot candidate viewpoint generation
- collision constraint checking
- visibility evaluation
- viewpoint dataset generation

## 3.2 NOT included in v8

The following belong to later versions:

- learned viewpoint selection network
- reinforcement learning
- closed-loop robot navigation
- action-aware active view selection
- real robot deployment

---

# 4. Scientific Problem Definition

Let:

- E denote environment
- H denote human state
- R denote robot state
- V denote candidate viewpoints

The objective of v8 is to construct:

V={v1,v2,...,vn}

where each viewpoint satisfies:

## Spatial constraint

v ∈ FreeSpace(E)

## Robot feasibility constraint

Path(R,v)=True

## Human observation constraint

Visibility(H,v)>0

---

# 5. Overall Architecture

```
v7 Simulation Platform
        |
        v
Human Placement Module
        |
        v
Candidate View Generator
        |
        v
Constraint Checking
        |
        v
Visibility Evaluation
        |
        v
View Candidate Dataset
```

---

# 6. Module Design

## 6.1 Human Placement Module

### Goal

Remove fixed human position limitation from v7.

### Function

Generate valid human positions in indoor scenes.

### Requirements

Input:

- Habitat scene
- human model
- action type

Output:

- human position
- human orientation
- initial pose

### Constraints

Must avoid:

- wall penetration
- furniture collision
- floating position

---

## 6.2 Robot Candidate View Generator

### Goal

Generate possible observation locations.

Candidate viewpoints should consider:

- distance
- horizontal angle
- camera height
- robot orientation

Example:

```
        v2

   v1    H    v3

        v4
```

Output:

candidate viewpoint list.

---

## 6.3 Constraint Checking Module

Responsibilities:

- validate robot location
- validate path accessibility
- remove invalid candidates

The module only provides feasibility checking.

It does not perform optimal planning.

---

## 6.4 Visibility Evaluation Module

Purpose:

Provide baseline observation quality metrics.

Metrics include:

### Human visibility

Percentage of visible human pixels.

### Occlusion ratio

Degree of human body obstruction.

### Pose coverage

Visible human keypoint ratio.

### Distance score

Robot-human distance measurement.

Output:

view_quality.json

---

# 7. Data Generation Specification

v8 should generate:

```
data/ActiveView/v8/

episode_xxx/

├── rgb/
├── depth/
├── human_pose/
├── candidate_views.json
├── visibility.json
└── metadata.json
```

metadata example:

```json
{
 "scene_id":"",
 "human":{
   "action":"",
   "position":[]
 },
 "robot":{
   "position":[]
 },
 "candidate_views":[]
}
```

---

# 8. Code Structure Requirement

v8 should extend v7 without modifying historical versions.

Recommended structure:

```
ea_avs_mvp_v8/

├── placement/
├── viewpoint/
├── constraints/
├── visibility/
├── dataset/
├── evaluation/
├── configs/
├── scripts/
└── tests/
```

---

# 9. Development Order

## Phase 1

Create v8 project structure.

Verify v7 compatibility.

## Phase 2

Implement human placement.

Acceptance:

- generated humans are valid
- no obvious collision

## Phase 3

Implement viewpoint candidate generation.

Acceptance:

- multiple viewpoints generated
- viewpoint metadata saved

## Phase 4

Implement feasibility checking.

Acceptance:

- invalid viewpoints removed

## Phase 5

Implement visibility evaluation.

Acceptance:

- visibility score generated
- occlusion metric generated

## Phase 6

Generate v8 dataset.

---

# 10. Acceptance Criteria

## Environment

- Habitat loads successfully
- v7 Humanoid pipeline remains functional

## Human Placement

- success rate >=95%
- no severe collision
- reproducible with random seed

## Candidate View Generation

Each episode:

- generate >=16 candidate viewpoints

## Feasibility

- invalid viewpoints filtered
- reachable viewpoints recorded

## Visibility

Every viewpoint has:

- visibility score
- occlusion score
- pose coverage score

## Dataset

Each episode contains:

- RGB
- Depth
- Human pose
- View candidates
- Metadata

---

# 11. Interface Reserved for v9

v8 output will become input for v9.

Future pipeline:

```
Candidate Views
        |
        v
Action-aware View Scoring Network
        |
        v
Best View Selection
```

---

# 12. Version Completion Definition

v8 is completed when:

1. Human position is no longer manually fixed.
2. Robot candidate views can be generated automatically.
3. Invalid viewpoints can be removed.
4. Observation quality can be quantitatively evaluated.
5. Candidate viewpoint dataset can be generated.

After completion, ACTIVEVIEW enters v9:

Action-aware Active View Selection.

---

# 13. Git Commit Requirement

Do not commit:

- AMASS raw data
- BABEL raw data
- generated RGB-D datasets
- videos
- cache files

Commit:

- source code
- configuration
- documentation
- tests

Suggested commit message:

```
feat(v8): implement human-aware active view foundation
```
