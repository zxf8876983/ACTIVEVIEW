# ParaHome → Habitat retarget pose-fidelity fixes

Status: **fixed and validated in the real Habitat simulator (CUDA, RTX 4090)**

Four independent defects were found and fixed, in this order of importance:

1. **Coordinate mapping was a reflection** (`det = -1`), which made the solved
   body frame face backwards — the visible "face and torso turned against the
   arms".
2. **World-space twist loss** in the joint solver, which left the spine and head
   near the rest facing instead of following the recorded body yaw.
3. **Joint-definition mismatches** at the pelvis and the lumbar spine, which
   pushed the abdomen mesh forward — the visible "protruding belly".
4. **Hip-width mismatch**, which left the standing avatar's legs ~8 cm narrower
   than the recording, reading as "legs pressed together".

## Symptom

After the earlier joint-order fix removed the tilted head / straight-arm
artifact, the rendered humanoid still had its **face and torso turned against
its own arms**: the arms/hands reached to the right places but the body seemed
to face the opposite way, and the abdomen showed a candy-wrapper twist.

## Cause 1 (dominant): mirrored coordinate mapping

`PARAHOME_TO_HABITAT` was

```python
[[0, 1, 0], [0, 0, 1], [-1, 0, 0]]     # det = -1 -> reflection
```

It maps the forward and up axes correctly, but its determinant is -1, so the
skeleton is *mirrored*. Every solved joint rotation is a proper rotation, and a
proper rotation cannot reproduce a mirrored target configuration. For the joints
whose children span 3D (`pelvis`, `spine3`) the least-squares fit resolves the
contradiction by flipping the body frame by ~180°:

| mapping | chest facing · recorded facing (s78 frames 480/574/660) |
| --- | --- |
| `det = -1` (before) | **-0.99 / -0.91 / -0.84** |
| `det = +1` (fixed) | **+0.93 / +0.87 / +0.80** |

The arm and hand *positions* stay faithful either way (they are positions, not
frames), which is exactly why the body looked reversed while the reach looked
right.

## Cause 2: twist loss in the joint solver

`ParaHomeSkeletonRetargeter._solve_global_rotations` solved every joint
independently in world space:

```python
Rotation.align_vectors(target_direction, world_rest_direction)
```

A segment direction never constrains the rotation *about* that segment. For
joints with a single child (`spine1`, `spine2`, `neck`, collars, shoulders,
elbows, hips, knees, ankles) that freedom was resolved as a minimal rotation away
from the **world rest frame**, so the solved world orientation of the bone
ignored the twist the parent chain already carried. `spine3` and `pelvis` have
three children each and were unaffected, which is why the arms (driven by the
collar/shoulder chain through `spine3`) followed the body while the spine and
head did not.

## Cause 3: joint-definition mismatches (belly)

The two rigs name different anatomical joints the same way:

| joint | ParaHome | `male_0` |
| --- | --- | --- |
| `hip` / pelvis link | hip *centre*; the two hips are 178.3 deg apart | link sits ~10 cm **above** the hip joints; the hip offsets are 60.5 deg apart |
| `spine1` / `spine2` | lumbar vertebrae L5/L4, bones 0.05–0.07 m | links much higher up the spine (0.125 / 0.164 m) carrying a ~20 deg lordotic rest offset |

Fitting those raw offsets made the least-squares residual reach ~60 deg at the
pelvis and forced the rig's long lordotic lumbar onto the recorded short straight
bones. The abdomen mesh (weighted to `spine1`/`spine2`) then swung forward: the
measured abdomen offset along the body front was +0.043 m on s50 versus +0.011 m
in the rest pose, which reads as a protruding belly.

## Cause 4: standing legs pressed together

Joint *directions* are matched but joint *spacing* is not: the `male_0` hip
joints are 12.5 cm apart, while ParaHome's hips (hip-centre convention) are
18.8-21.9 cm apart depending on the subject (a per-subject constant).  The whole
leg chain, driven by the recorded thigh directions, therefore stayed ~8 cm
narrower than the recording:

| frame (s3) | recorded hip / knee / ankle | avatar before | avatar after |
| --- | --- | --- | --- |
| 340 | 20 / 18 / 19 cm | 12 / 11 / 12 cm | **20 / 18 / 20 cm** |
| 420 | 20 / 19 / 19 cm | 12 / 12 / 12 cm | **20 / 19 / 19 cm** |
| 500 | 20 / 31 / 36 cm | 12 / 22 / 27 cm | **20 / 30 / 35 cm** |

Two limits appear once the joint spacing is right:

1. The recorded stance (knees 19-20 cm) is narrower than the rig's **own rest
   stance** (knees 24.4 cm), so the thick `male_0` legs interpenetrate: measured
   knee-level mesh gap on s3 was +1.1 cm (frame 420) and -5.9 cm (frame 580).
2. Therefore the target separation never drops below the rig's rest knee
   separation.  After the rule the same frames give a knee joint separation of
   23 / 21 cm and a mesh gap of +5.0 / -2.0 cm — the legs read as a normal
   relaxed stance instead of touching.

This deliberately makes the stance up to ~5 cm wider than the recording (the
person's own legs are thinner than the model's).  Setting
`match_hip_width=False` restores the exact recorded spacing.

`renders/hip_width_matched_vs_rig.png` and `renders/stance_clearance_ab.png`
show the standing pose with and without the correction.

## Fix

**Hip width** (`match_hip_width=True`, default): the lateral component of the two
hip offsets is scaled by the recorded/rig separation ratio (clamped to 0.8-2.0 so
a bad fit cannot tear the mesh).  The legs simply translate outward, so the mesh
is only stretched in the hip crease, and the recorded stance is reproduced.

**Pelvis**: solved from the derivation-invariant hip line
(`left_hip - right_hip`) plus the spine axis instead of the two raw hip offsets,
so the ~60 deg residual no longer tilts the trunk (constraint residual 2.7 deg).

**Lumbar (`REST_SHAPE_JOINTS = ("spine1", "spine2")`)**: these joints keep the
rig's own rest curvature; the recorded lumbar bones no longer drive them. The
trunk's overall bend is preserved because `spine3` is still solved in world
space from the neck and collar directions.

Measured with the skinned mesh and the recorded objects (median hand-to-object
distance, source reference in brackets):

| sequence / object | before | after | source |
| --- | --- | --- | --- |
| s50 cup | 17 cm | **13 cm** | 17 cm |
| s78 laptop (min) | 21 cm | **17 cm** | 15 cm |
| s3 pan | 34 cm | **30 cm** | — |
| s3 gasstove | 47 cm | **39 cm** | — |
| abdomen offset on s50 (rest = +0.011) | +0.043 m | **+0.021 m** | — |

`renders/belly_fixed_vs_old_solver.png` shows the result: front/side of the
drinking and typing poses, before and after.

`PARAHOME_TO_HABITAT` is now the proper rotation with the same forward/up
mapping:

```python
[[0, -1, 0], [0, 0, 1], [-1, 0, 0]]    # det = +1
```

(forward `+X -> -Z`, up `+Z -> +Y`, left `+Y -> -X`, which is the chirality a
`-Z`-facing Habitat character needs), and `_solve_global_rotations` now walks the
rig top-down (`_hierarchy_order`) so each joint inherits its parent's solved
frame and only adds the minimal rotation required by its own segment(s):

```python
rotations[name] = parent_frame @ _align_vectors(rest, parent_frame.T @ target)
```

Multi-child joints keep the same least-squares solution as before, because the
alignment residual is invariant under the parent rotation
(`||Δ rest - Rᵀ t|| = ||R Δ rest - t||`). The fitted segment directions, the root
transform and the grounding path are untouched.

## Evidence (`validate_retarget_pose_fidelity.py`, `validation.json`)

Synthetic ground truth (whole-body yaw 180° + 20° head pitch, exact target
orientations known), orientation error in degrees:

| solver | spine1 | spine2 | spine3 | neck | head |
| --- | --- | --- | --- | --- | --- |
| pre-fix (world space) | 126.6 | 121.8 | 0.0 | 118.0 | 118.0 |
| shipped (hierarchical) | 0.0 | 0.0 | 0.0 | 11.3 | 11.3 |

The remaining 11.3° on `neck`/`head` is the twist about the neck→head bone,
which joint positions cannot observe (the head bone is tilted ~12° off the up
axis, so a minimal swing is the only positions-only choice).

Real sequences (150 frames each, `s78` / `s50`):

| metric | pre-fix solver | shipped solver |
| --- | --- | --- |
| `spine1` facing vs observed shoulder-line forward (deg) | 136.4 / 124.7 | 8.8 / 7.7 |
| `spine3` (chest) facing error (deg) | 25.1 / 53.0 | 25.1 / 53.0 |
| `head` facing error (deg) | 120.4 / 111.3 | 27.3 / 50.7 |
| head-vs-chest relative rotation, mean/max (deg) | 143.3/179.8 · 110.3/179.9 | 13.7/20.9 · 6.8/18.4 |
| head-vs-chest relative rotation, std (deg) | 41.5 / 46.0 | 3.4 / 3.8 |
| observed head-tip drift in the head frame, mean/max (deg) | 18.1/47.9 · 14.3/37.4 | 7.3/18.5 · 8.7/35.9 |
| segment-direction fit error, all pairs, mean/max (deg) | 6.84 / 61.40 | 9.47 / 61.98 |
| segment-direction fit error, **anatomically matched joints**, mean/max (deg) | 1.34 / 9.74 | 1.42 / 9.74 |

(150 frames each, `s78` / `s50`, both with the corrected proper mapping, so the
comparison isolates the solver change.)

The matched-joint row is the honest skeleton-fidelity metric: it is ~1.4 deg
mean either way. The all-pairs row also contains the joints that are *different
anatomical joints* in the two rigs (`pelvis`-`hip`, `spine1`-`spine2`,
`spine2`-`spine3`), where a direction difference is a definition mismatch rather
than an error; the abdomen fix deliberately stops matching those.

`renders/before_frame_*.png` / `after_frame_*.png` contain an offline before/after
render of the skinned `male_0.glb` (Habitat's skinning reproduced with linear
blend skinning; the GLB rest skeleton matches the URDF rest skeleton up to one
constant root offset). Arrows show the `pelvis`/`spine1`/`spine3`/`head` facing.

## Cross-sequence chirality check (generalization)

Body facing · recorded facing, mean over 25 frames per sequence, for the shipped
proper mapping and for the pre-fix reflection (same solver, only the mapping
differs):

| sequence | proper (`det = +1`) | reflection (`det = -1`, before) |
| --- | --- | --- |
| s3 | **+0.84** | -0.90 |
| s50 | **+0.59** | -0.59 |
| s78 | **+0.90** | -0.95 |
| s99 | **+0.98** | -0.99 |
| s121 | **+0.95** | -0.99 |
| s162 | **+0.93** | -0.98 |

The reflection reverses the body in every sequence; the proper mapping never
does. (s50 mixes many activities including turns, hence the lower but clearly
positive value.)

## Habitat sample videos (real simulator)

Rendered on CUDA (RTX 4090, driver 550.54.14, `torch 2.6.0+cu124`) with the
clean `scene_id=NONE` replay path (floor + scanned chair/laptop/cup/kettle,
grounded `male_0`).

The default camera is a **single, completely static robot-eye view**: the robot
platform stands on the floor and **does not move** - its position *and* camera
orientation are fixed for the whole clip, placed once from the centre of the
recorded trajectory (2.8 m away, 30 deg off the human's initial facing) with the
camera at **1.2 m** above the ground aimed at 1.0 m, so the floor the human
stands on stays in frame.  Because the placement uses the trajectory centre, the
human stays visible for the whole clip even in a 3.5-minute recording.
`--robot-anchor follow` restores the old moving platform and `--robot-aim track`
keeps the base fixed but pans the camera.  Grounding uses the **skinned mesh** (lowest deformed vertex on the
floor), not the URDF debug boxes used by `precompute_grounding_offsets`, which
could leave the visible feet of a seated pose ~15 cm in the air.

The legacy three body-relative cameras are still available with
`--camera body3`.  Static-view sheets: `renders/s158_sequence_stills.png`
(whole sequence), `renders/static_robot_s78.png`, `static_robot_s50.png`,
`static_robot_s3.png`:

```bash
# robot-eye view (default), shipped vs pre-fix solver
PYTHONPATH=. python experiments/parahome_feasibility_v1/retarget_pose_fidelity/render_retarget_sample_video.py \
    --parahome-root /home/zxf/MG08/robot/ParaHome --sequence s78 \
    --frame-start 445 --frame-end 700 --variant both

# whole sequence (s158, 6308 frames)
PYTHONPATH=. python experiments/parahome_feasibility_v1/retarget_pose_fidelity/render_retarget_sample_video.py \
    --parahome-root /home/zxf/MG08/robot/ParaHome --sequence s158 \
    --frame-start 0 --frame-end 100000 --panel-size 384 --quality 7

# optional: moving platform, panning camera, or the legacy three views
    ... --robot-anchor follow
    ... --robot-aim track
    ... --camera body3 --views front,side,top
```

- `videos/s78_retarget_fixed.mp4` — shipped retargeter, frames 445–700
  (open laptop → type on laptop → close laptop).
- `videos/s78_retarget_prefix.mp4` — same cameras, pre-fix world-space solver.
- `videos/s50_retarget_fixed.mp4` — frames 1890–2070 (drink from cup), and
  `videos/s50_retarget_prefix.mp4` for the same A/B.
- `videos/s3_retarget_fixed.mp4` — frames 300–620 (sprinkle salt, standing).
- `videos/s158_retarget_fixed.mp4` — a **randomly picked whole sequence**
  (s158, subject p29: 6,308 frames = 3.5 min, 27 annotated actions from moving a
  cup through cooking and putting a pot on the stove), rendered end to end in the
  robot-eye view at 384 px panels.  It verifies that the retarget
  stays stable and plausible over a full recording: standing stances keep the
  leg gap, the seated actions put the body on the chair, and the drinking /
  pouring / cutting poses keep the hands in front of the torso
  (`renders/s158_sequence_stills.png`; `renders/robot_view_samples.png` compares
  the shipped and pre-fix solvers plus the s50/s3 segments in the same view).
- `renders/habitat_{fixed,prefix}_front_top_zoom.png` — zoomed front/top
  comparison at frame 574; `renders/habitat_*_frame_0574_views.png` — full
  three-view frames.
- `renders/rig_rest_facing_{minus,plus}_z.png` — rest-pose probe used to confirm
  that `male_0` faces **-Z** at rest (the `-Z` side camera sees the face), which
  is what the arrows and the body-relative cameras use.

The additional samples (s50 drink-from-cup, s3 sprinkle-salt) reproduce the same
result: face, chest and hands share one facing in the fixed videos, while the
pre-fix solver still shows the twisted torso.

With both fixes the person sits on the chair facing the laptop, the head bows
down over the table (chest elevation -21...-28 deg, head -25...-30 deg) and both
hands are in front of the torso. With the pre-fix solver (same mapping) the
shoulders/arms keep the recorded yaw while the spine and head stay near the rest
facing, so the torso is visibly twisted against its own arms.

## Caveats and open findings

- The neck/head still only follow the chest frame plus the observable neck-bone
  swing: an independent head yaw (and the twist about a single bone) is not
  recoverable from joint positions, so e.g. turning the head left while the chest
  stays still cannot be reproduced. Using ParaHome's `body_joint_orientations`
  for that would need the official SMPL-X rest frames (A→B→C calibration).
- The fixes above are generic and independent of action or sequence; no
  action-specific or frame-specific correction was introduced.

## Reproduce

```bash
PYTHONPATH=. python experiments/parahome_feasibility_v1/retarget_pose_fidelity/validate_retarget_pose_fidelity.py
PYTHONPATH=. python experiments/parahome_feasibility_v1/retarget_pose_fidelity/render_retarget_pose_ab.py
python -m pytest -q tests/unit/test_parahome_retarget.py
```

`PARAHOME_ROOT` defaults to `/home/zxf/MG08/robot/ParaHome` in the two scripts.
