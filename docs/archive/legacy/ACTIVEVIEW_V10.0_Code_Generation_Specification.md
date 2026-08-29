# ACTIVEVIEW v10.0 Code Generation Specification

## Update Notice

This document defines the engineering implementation constraints for v10.0. Development must follow this specification together with ACTIVEVIEW_V10.0_Research_Development_Document.md.

The most important additional constraint is that v10.0 development must be completed through four independent phases. Each phase has its own inputs, outputs and acceptance criteria. Later phases must not bypass unfinished earlier phases.

---

# 17. Phased Development Workflow

## Phase 1: RGB-D Dataset Generation

### Objective

Build the simulation data foundation.

### Pipeline

Habitat Scene

+

Humanoid

+

Motion Sequence

↓

RGB-D Rendering

↓

Dataset Metadata

### Output

Generate:

- RGB images
- Depth images
- Camera pose
- Scene information
- Action labels
- Optional GT annotations for Oracle evaluation

### Acceptance Criteria

Phase 1 is completed only when:

1. RGB-D samples can be generated automatically.
2. Multiple viewpoints can be generated.
3. Dataset metadata can be parsed without manual intervention.

No ST-GCN or Active View development is allowed before Phase 1 passes.

---

## Phase 2: Perception Pipeline

### Objective

Replace direct simulation state with robot-like observations.

### Pipeline

RGB

↓

2D Pose Estimator

+

Depth

↓

3D Projection

↓

Estimated 3D Skeleton

### Output

- 2D keypoints
- 3D skeleton
- confidence values

### Rules

GT skeleton cannot be used as model input.

GT is only allowed for:

- supervision
- Oracle baseline
- offline evaluation

### Acceptance Criteria

The system must visualize and validate estimated skeleton quality.

---

## Phase 3: Action Recognition Module

### Objective

Build the task evaluation model.

### Pipeline

3D Skeleton Sequence

↓

ST-GCN

↓

Action Probability

### Requirements

Action recognition training is independent from Active View training.

After ST-GCN reaches stable performance:

- freeze ST-GCN parameters
- use it as task evaluator

### Reason

The research contribution is Active View Selection, not a new action recognition network.

### Acceptance Criteria

The system must provide:

- action accuracy
- confusion matrix
- confidence distribution

---

## Phase 4: Action-aware Active View Selection

### Objective

Develop the core research module.

### Input

Current observation:

- estimated skeleton
- confidence
- action probability
- current viewpoint

Candidate viewpoint:

- distance
- angle
- height
- geometric descriptor

### Output

Ranking score of candidate viewpoints.

### Optimization Target

Maximize action uncertainty reduction:

Gain(v)=Entropy(before)-Entropy(after)

### Acceptance Criteria

The method must outperform:

- Random
- Nearest
- Geometry(v8)
- Perception(v9.1)

under action recognition metrics.

---

# 18. Development Isolation Rules

The following dependencies are forbidden:

- Active View training cannot use GT skeleton.
- Active View training cannot modify ST-GCN weights.
- ST-GCN cannot use future selected viewpoints.
- Dataset generation cannot bypass perception simulation when evaluating online performance.

The purpose is to isolate the contribution of active viewpoint selection.
