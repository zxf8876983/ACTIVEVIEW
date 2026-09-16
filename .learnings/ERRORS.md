# Errors

## [ERR-20260905-001] npz_content_comparator

**Logged**: 2026-09-05T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
NPZ equality check passed `equal_nan=True` to string arrays.

### Error
`TypeError: ufunc 'isnan' not supported for the input types`

### Context
- Read-only comparison of old and new worker-scheduling benchmark outputs.
- `np.array_equal(..., equal_nan=True)` is only valid for compatible numeric arrays.

### Suggested Fix
Use NaN-aware equality only for floating/complex arrays and ordinary
`np.array_equal` for strings, integers and booleans.

### Metadata
- Reproducible: yes
- Related Files: none

### Resolution
- **Resolved**: 2026-09-05T00:00:00+08:00
- **Notes**: Comparator was corrected without changing dataset files.

---

## [ERR-20260906-001] apply_patch_context

**Logged**: 2026-09-06T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
Planning-file patch used stale wording from compacted context.

### Error
`apply_patch verification failed: Failed to find expected lines`

### Context
- Attempted to append the scheduler-persistence plan to `task_plan.md`.

### Suggested Fix
Read the exact current tail before constructing context-sensitive patches.

### Metadata
- Reproducible: no
- Related Files: task_plan.md

### Resolution
- **Resolved**: 2026-09-06T00:00:00+08:00
- **Notes**: Re-read the file and applied a patch against exact current text.

---

## [ERR-20260906-002] mixed_file_patch_context

**Logged**: 2026-09-06T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
A multi-file patch referenced a `notes.md` block before switching file targets.

### Error
`apply_patch verification failed: Failed to find expected lines`

### Context
- Updating scheduler verification wording after code review.

### Suggested Fix
Use explicit `Update File` markers for each file and inspect both exact snippets.

### Metadata
- Reproducible: no
- Related Files: task_plan.md, notes.md

### Resolution
- **Resolved**: 2026-09-06T00:00:00+08:00
- **Notes**: Reapplied using explicit file sections and exact contexts.

---

## [ERR-20260906-003] completed_session_poll

**Logged**: 2026-09-06T00:00:00+08:00
**Priority**: low
**Status**: resolved
**Area**: infra

### Summary
Polling a finished generation session returned an unknown process ID.

### Error
`write_stdin failed: Unknown process id 35283`

### Context
- Checked runtime safety immediately before renaming the generator entry file.

### Suggested Fix
When a long session disappears, inspect process state and final dataset
manifests before classifying it as a failure.

### Metadata
- Reproducible: no
- Related Files: activeview/scripts/data/generate_hm3d_train_offline.py

### Resolution
- **Resolved**: 2026-09-06T00:00:00+08:00
- **Notes**: Verified all 21 scene manifests and dataset summary were complete.

---

## [ERR-20260917-001] swapped-camera-labels-misdiagnosis

**Logged**: 2026-09-17
**Priority**: high
**Status**: resolved
**Area**: data/motion

### Summary
A rest-pose facing probe had its two camera labels swapped, which produced the
wrong conclusion that `male_0` faces -Z and led to declaring the retarget fix
complete while the rendered humanoid still faced backwards.

### Error
No exception: a silently wrong measurement (`{"minus_z": [0, 0.2, +2.6]}` placed
the "minus_z" camera on the **+Z** side). Every downstream "face direction"
metric and arrow was then interpreted with the wrong sign.

### Context
- Diagnosing "face and torso turned against the arms" in the ParaHome -> Habitat
  retarget; the user rejected the first "fixed" claim after watching the sample
  video, which is what exposed the mistake.
- Real cause (see ERR/LEARN 2026-09-17 entries): the coordinate mapping
  `PARAHOME_TO_HABITAT` was a reflection (`det = -1`); a proper-rotation solver
  flips the body frame ~180° on mirrored targets while hand positions stay right.

### Suggested Fix
- Derive camera labels from the position vector itself (assert the sign), never
  from a name typed next to the literal.
- Cross-check a facing conclusion with an independent observable (the eye/head
  link offsets, the recorded shoulder-line direction, and the rendered image).
- Do not declare a rendering bug fixed without looking at a rendered frame that
  the user can reproduce; prefer an A/B render and an object/arm consistency
  check over derived metrics that share the same assumption as the fix.

### Metadata
- Reproducible: yes
- Related Files: `experiments/parahome_feasibility_v1/retarget_pose_fidelity/renders/rig_rest_facing_probe.png`

### Resolution
- **Resolved**: 2026-09-17T03:00:00+08:00
- **Notes**: Probe relabelled (face is at +Z), mapping replaced with the proper
  rotation, arrows/cameras updated, sample videos re-rendered and confirmed by
  the user-visible A/B comparison.

---

## [ERR-20260917-002] urdf-box-grounding

**Logged**: 2026-09-17
**Priority**: medium
**Status**: resolved
**Area**: data/motion

### Summary
Humanoid grounding used the URDF debug boxes instead of the skinned render mesh,
so seated poses left the visible feet ~15 cm above the floor.

### Error
No exception: `precompute_grounding_offsets` -> `humanoid_geometry_y_bounds`
computes bounds from the URDF `<visual><box>` primitives (bone placeholders), but
the rendered body is the skinned `male_0.glb`. Measured on s78 frame 574: the
foot joint sat 0.184 m above the floor while the debug-box grounding reported a
clean contact. The first fix attempts did not notice because the standing frames
happened to look grounded.

### Context
- Building the ParaHome -> Habitat replay/visualisation path; the user reported
  the human "must stand on the ground".
- The same mismatch explains why the character can look slightly floating or
  sunk in other AMASS/BABEL visualisations that use the same helper.

### Suggested Fix
Ground on the actual rendered geometry: deform the skinned mesh for the frame
and put its lowest vertex on the floor (`SkinnedHumanoid.deform` in
`render_retarget_pose_fidelity/render_retarget_pose_ab.py`), or verify with the
Habitat link transforms that the foot links sit at their expected few-cm height.
Do not trust bone-primitive bounds as a proxy for the visible body.

### Metadata
- Reproducible: yes
- Related Files: `activeview/data/motion/humanoid_grounding.py`,
  `activeview/data/motion/babel_clean_dataset_generator.py`,
  `experiments/parahome_feasibility_v1/retarget_pose_fidelity/render_retarget_sample_video.py`

### Resolution
- **Resolved**: 2026-09-17T04:15:00+08:00
- **Notes**: The replay renderer now computes per-frame grounding from the
  skinned mesh; verified min mesh y = 0.000 m and foot links 0.027-0.042 m on
  s3/s78/s158 frames.

---
