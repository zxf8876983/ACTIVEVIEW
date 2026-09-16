# Project Learnings

## [LRN-20260824-001] environment-verification

**Logged**: 2026-08-24
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
The required runtime is the activated `conda` environment named `habitat`; its activation was verified explicitly.

### Details
`conda activate habitat` resolves Python to `/home/zxf/anaconda3/envs/habitat/bin/python` and imports Habitat-Sim 0.3.3, Torch 2.6.0 and Torchvision 0.21.0. The remaining failure is not environment selection: CUDA reports zero devices and Habitat cannot create an EGL context.

### Suggested Action
Run the generator on a host with a working CUDA/EGL device or a functional X display.

### Metadata
- Source: user_feedback
- Related Files: `ea_avs_mvp_v11/scripts/generate_v11_5_offline_dataset.py`
- Tags: habitat, conda, egl, cuda

---

## [LRN-20260917-001] habitat-humanoid-chirality

**Logged**: 2026-09-17
**Priority**: high
**Status**: pending
**Area**: data/motion

### Summary
A world-to-world axis mapping used to feed a rotation-solver must be a *proper*
rotation (`det = +1`): a reflection cannot be fitted by proper joint rotations
and shows up as a body frame flipped ~180° while joint *positions* stay correct.

### Details
`PARAHOME_TO_HABITAT` was `[[0,1,0],[0,0,1],[-1,0,0]]` (`det = -1`). Forward/up
mapped correctly, but the skeleton was mirrored. `ParaHomeSkeletonRetargeter`
solves one proper rotation per joint; for joints with >=3 children spanning 3D
(`pelvis`, `spine3`) the least-squares fit resolved the contradiction by
reversing the body facing (`chest facing · recorded facing = -0.91…-0.99`) while
the arms/hands (positions) still reached the right places — rendered as "the face
and torso are turned against the arms". The proper rotation with the same
forward/up mapping is `[[0,-1,0],[0,0,1],[-1,0,0]]` (`+0.80…+0.93`).

Related conventions verified by rendering, not by docs:
- `male_0` faces **+Z** at rest (the `+Z`-side camera sees the face); Habitat's
  "character faces -Z" wording does not describe this asset.
- All 54 URDF joint origins have `rpy = 0`, so the link frames are world-aligned
  at rest and an FK built from `getJointInfo[14]` offsets matches Habitat's.
- Habitat's articulated link order equals `HABITAT_MALE_0_JOINT_ORDER`.

### Suggested Action
Whenever a retarget/mapping is added, assert `det(R) == +1` and unit-test the
axis images (forward/up/left). When a rendered humanoid looks "reversed", check
chirality before blaming the rig or the mesh. Joint positions alone cannot
determine the twist about a bone; do not expect head yaw or palm roll to be
recoverable from positions.

### Metadata
- Source: user_feedback
- Related Files: `activeview/data/motion/parahome_retarget.py`,
  `tests/unit/test_parahome_retarget.py`,
  `experiments/parahome_feasibility_v1/retarget_pose_fidelity/`
- Tags: habitat, parahome, retarget, chirality, reflection

---

## [LRN-20260917-002] cross-rig-joint-definitions

**Logged**: 2026-09-17
**Priority**: high
**Status**: pending
**Area**: data/motion

### Summary
Two humanoid rigs can use the same joint names for different anatomical joints.
Fitting those offsets as if they were the same bone produces large residuals and
deforms the skinned mesh; verify with an object-interaction metric, not with the
fitted bone error.

### Details
ParaHome `hip` is the hip *centre* (hips 178.3 deg apart) while the `male_0`
pelvis link sits ~10 cm above its hip joints (60.5 deg apart); ParaHome
`spine1`/`spine2` are L5/L4 (0.05-0.07 m bones) while the `male_0` spine1/spine2
are higher up the spine (0.125/0.164 m) with a ~20 deg lordotic rest offset.
Fitting the raw offsets gave a ~60 deg pelvis residual and straightened the
lordotic lumbar, which pushed the abdomen mesh forward (`+0.043 m` vs `+0.011 m`
at rest) - the visible "protruding belly". Fixes that worked:
- solve the pelvis from the derivation-invariant hip line (`left_hip -
  right_hip`) plus the spine axis instead of the two hip offsets;
- let the intermediate spine joints keep the rig's own rest curvature
  (`REST_SHAPE_JOINTS`) while the top spine joint still carries the recorded
  trunk bend solved in world space.
Object metrics improved everywhere (hand-to-cup 17->13 cm, hand-to-laptop min
21->17 cm, hand-to-pan 34->30 cm) while the anatomically matched bone fit stayed
~1.4 deg.

### Suggested Action
Before trusting a joint-direction fit, compare the *relative geometry* of each
joint's children in the two rigs (angles and lengths). Skip or re-derive the
constraints for joints whose definitions differ, and judge retargets by
hand-to-object / hand-to-head distances and by rendered frames rather than by the
fitted bone-angle error, which can look worse while the pose gets better.

### Metadata
- Source: user_feedback
- Related Files: `activeview/data/motion/parahome_retarget.py`
  (`REST_SHAPE_JOINTS`, `_constraints`),
  `experiments/parahome_feasibility_v1/retarget_pose_fidelity/`
- Tags: retarget, rig-mismatch, skinning, habitat, parahome

---

## [LRN-20260917-003] retarget-spacing-not-just-direction

**Logged**: 2026-09-17
**Priority**: medium
**Status**: pending
**Area**: data/motion

### Summary
A direction-only retarget matches bone orientations but not joint *spacing*; when
the two rigs disagree about joint spacing the limbs end up squeezed together.
Check the separation of paired joints (hips/knees/ankles, shoulders/wrists)
between the recording and the avatar.

### Details
The `male_0` hip joints are 12.5 cm apart while ParaHome's hips (hip-centre
convention) are 18.8-21.9 cm apart depending on the subject, so the avatar's legs
stayed ~8 cm narrower and looked pressed together. Scaling the lateral component
of the two hip offsets by the recorded/rig ratio (clamped) reproduces the
recorded stance (s3: hips/knees/ankles 12/12/12 -> 20/19/19 cm vs recorded
20/19/19) with only local mesh stretch in the hip crease.

### Suggested Action
For every retarget, compare paired-joint separations and limb lengths between the
source skeleton and the rig, not only bone directions. Per-subject constants
(like hip width) are safe to calibrate from the data; clamp the scale so a bad
fit cannot tear the mesh.

### Metadata
- Source: user_feedback
- Related Files: `activeview/data/motion/parahome_retarget.py`
  (`match_hip_width`, `_frame_offsets`)
- Tags: retarget, proportions, habitat, parahome

---
