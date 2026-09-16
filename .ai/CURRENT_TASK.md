# ParaHome dataset: issues found and resolved — completed (2026-09-17)

Scope: the ParaHome feasibility audit and the ParaHome → Habitat `male_0`
replay path.  No reduced12/BABEL artifact, recognizer, frozen checkpoint or
Policy Test data was touched.

## A. Dataset audit (read-only, `experiments/parahome_feasibility_v1/`)

- 207 sequences, 38 subjects, 486.33 minutes, 5,476 annotations, 30 fps
  (inferred; no per-sequence fps field).
- 16 classes meet both the 30/15/8 and 50/20/10 coverage thresholds; with a 2 s
  HAR window 9/16 classes support two 1 s decision cycles and 5/16 support two
  2 s cycles.  Verdict: **CONDITIONAL PROMOTE** — taxonomy duration/coverage and
  a coordinate/gender/scene adapter must be addressed before ParaHome becomes the
  continuous Active HAR mainline.
- Dataset-level caveats recorded for later work: gender/shape metadata is only a
  subject id (no shape asset), object replay carries the dataset's own
  alignment/penetration warnings, and the released motion is SMPL-X fits
  (no RGB/depth ground truth in the audited subset).
- Data fields audited: `joint_positions.pkl` (T,73,3), `body_joint_orientations`
  (T,23,6), `hand_joint_orientations` (T,40,6), `head_tips` (crown, T,3),
  `body_global_transform`, `object_transformations`, `text_annotation.json`.

## B. Replay/retarget defects (all fixed, all render-verified)

1. **Joint order.** Habitat exposes `male_0` articulated state in
   `HABITAT_MALE_0_JOINT_ORDER`, not the URDF declaration order used by
   `MotionConverterSMPLX`; writing PyBullet order caused the tilted head and
   straight arm.  Verified against `get_link_name(i)`.
2. **Coordinate mapping was a reflection** (dominant).  `PARAHOME_TO_HABITAT`
   was `[[0,1,0],[0,0,1],[-1,0,0]]` (`det = -1`).  Every solved joint rotation
   is a *proper* rotation, and a proper rotation cannot fit a mirrored target, so
   for `pelvis`/`spine3` (three children spanning 3D) the least-squares fit
   flipped the body frame ~180 deg (`chest facing · recorded facing =
   -0.99/-0.91/-0.84` at s78 frames 480/574/660) while the arm/hand **positions**
   stayed faithful — the rendered "face and torso turned against the arms".
   Fixed with the proper rotation `[[0,-1,0],[0,0,1],[-1,0,0]]` (`det = +1`, same
   forward/up mapping); the same poses give `+0.93/+0.87/+0.80` and
   cross-sequence `+0.59…+0.98` for s3/s50/s78/s99/s121/s162.
3. **Twist loss in the solver.**  `_solve_global_rotations` solved each joint
   independently in world space, so single-child joints (`spine1`, `spine2`,
   `neck`, collars, shoulders, elbows, hips, knees, ankles) lost the parent twist
   while multi-child joints kept it.  Now solved top-down (`_hierarchy_order`):
   each joint inherits its parent frame and adds only its own segment swing;
   multi-child joints keep the identical least-squares solution (the residual is
   invariant under the parent rotation).  Synthetic whole-body yaw-180 deg error
   on `spine1`/`spine2`/`neck`/`head` drops 118–127 deg -> 0.0 deg (11.3 deg
   residual is the unobservable twist about the neck bone).
4. **Joint-definition mismatches (belly, then stance).**  The rigs use different
   anatomical joints under the same names: ParaHome `hip` is the hip *centre*
   (hips 178.3 deg apart) while the `male_0` pelvis link sits ~10 cm above its
   hip joints (60.5 deg apart); ParaHome `spine1`/`spine2` are L5/L4
   (0.05–0.07 m bones) while `male_0` spine1/spine2 are higher up the spine
   (0.125/0.164 m) with a ~20 deg lordotic rest offset; and joint *spacing* is
   never matched (rig hips 12.5 cm vs recorded 18.8–21.9 cm).
   - Pelvis now solved from the derivation-invariant hip line
     (`left_hip - right_hip`) + spine axis (constraint residual 2.7 deg instead
     of ~60 deg).
   - `REST_SHAPE_JOINTS = ("spine1", "spine2")` keep the rig's own rest
     curvature; `spine3` still carries the recorded trunk bend in world space.
     Abdomen offset returns from `+0.043 m` to `+0.021 m` (rest `+0.011 m`).
   - `match_hip_width=True` (default) scales the lateral hip-offset component by
     the recorded/rig ratio; the target never drops below the rig's own rest knee
     separation (24.4 cm) so the thick `male_0` legs cannot interpenetrate.
   - Object-interaction distance improves everywhere: s50 hand-cup 17 -> 13 cm,
     s78 hand-laptop min 21 -> 17 cm, s3 hand-pan 34 -> 30 cm (source references
     17 cm / 15 cm).  Trade-off documented: the stance can be ~5 cm wider than
     the recording because the recorded person's legs are thinner than the
     model's; `match_hip_width=False` restores the exact recorded spacing.

Verification: synthetic ground truth (exact Planted rotations), s78/s50 facing
and head-vs-chest metrics (150 frames each), object-distance metrics on
s50/s78/s3, and a whole randomly picked sequence (s158, 6,308 frames / 3.5 min,
27 actions) rendered end to end.  `compileall` clean;
`pytest -q tests/unit tests/integration` -> 81 passed, 5 skipped.

## C. Render conventions (verified by rendering, not by docs)

- `male_0` **faces +Z at rest** (the `+Z`-side camera sees the face).  The
  Habitat phrase "character faces -Z" does not describe this asset.
- All 54 URDF joint origins have `rpy = 0`, so the link frames are world-aligned
  at rest and the retargeter's forward kinematics matches Habitat's.
- Grounding must use the **skinned mesh** (lowest deformed vertex on the floor),
  not the URDF debug boxes used by `precompute_grounding_offsets`: the box
  grounding left seated poses' visible feet ~15 cm in the air.
- Sample renders use a **single static robot-eye camera**: the robot platform
  neither translates nor rotates; it is placed once from the centre of the
  recorded ground trajectory (2.8 m from the human, 30 deg off the initial
  facing) with the camera **1.2 m** above the floor aimed at 1.0 m, so the human
  stays in frame for the whole clip and the floor is visible.
  `--robot-anchor follow`, `--robot-aim track` and `--camera body3` restore the
  moving/panning/three-view variants.

## D. Remaining limitations

- The rotation *about* a single bone (twist) is not observable from joint
  positions: an independent head yaw and the palm/elbow roll cannot be recovered.
  Full recovery needs ParaHome's `body_joint_orientations` (6D) plus the official
  SMPL-X rest frames (A→B→C calibration).
- Rig proportions differ from the recorded subjects (spine levels, hip width,
  limb lengths, body build).  Only the `male_0` asset is available locally; a
  gender/shape-matched humanoid is still open work for the audit's CONDITIONAL
  PROMOTE verdict.
- Dataset taxonomy/timing coverage and the ParaHome gender/coordinate adapter
  remain the gating items before any continuous Active HAR use.

## E. Artifacts

- Code: `activeview/data/motion/parahome_retarget.py`
  (`PARAHOME_TO_HABITAT`, `_hierarchy_order`, `REST_SHAPE_JOINTS`,
  `_constraints`, `match_hip_width`, `_frame_offsets`).
- Tests: `tests/unit/test_parahome_retarget.py` (neutral pose, non-finite input,
  `det = +1` chirality, whole-body yaw-180 deg twist, hip-width/stance rule).
- Record: `experiments/parahome_feasibility_v1/retarget_pose_fidelity/`
  (README, `validation.json`, validators/renderers, static-robot videos and
  comparison stills).
- Detailed chronological entries: `.ai/RESEARCH_LOG.md` (five entries,
  2026-09-17); durable lessons: `.learnings/LEARNINGS.md`
  LRN-20260917-001/002/003 and `.learnings/ERRORS.md` ERR-20260917-001/002.
