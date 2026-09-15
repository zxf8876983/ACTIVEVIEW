# True-facing ST-GCN angle-prior re-audit

## Goal
Recompute the reduced12 relative-angle prior with the AMASS frame-0 body
facing convention and determine whether the historical angle-prior failure was
caused by placement-yaw misalignment.

## Phases
- [x] Reuse frozen Yaw8Fair/ST-GCN assets and verify Train-internal-Val and Moving-Val boundaries
- [x] Recompute internal and Moving angle bins with `R_scene_yaw @ R_AMASS_root_frame0`
- [x] Evaluate corrected angle priors against historical, visibility and oracle references
- [x] Validate reports and prepare the scoped commit

## Key Questions
1. Was the historical angle prior failure caused by using placement yaw instead of true body facing?
2. Do corrected internal angle preferences generalize to Moving Val?
3. Does a corrected angle prior beat the existing StaticPrior/visibility references?

## Decisions Made
- Corrected body-facing angle uses local +Z transformed by the AMASS frame-0
  root rotation and scene yaw; no placement or recognizer changes were made.
- Angle tables are frozen from raw-train-derived Yaw8 internal Val; Moving Val
  is evaluation-only.
- Final decision: KILL ST-GCN RELATIVE-ANGLE PRIOR. Corrected Q-margin and
  Q-accuracy correlations are not both stable, and corrected selectors remain
  below the StaticPrior baseline.

## Errors Encountered
- Policy rows omit raw motion `source_path`; resolved root yaw through the canonical raw-val manifest.
- Existing visibility cache stores NaN in inactive padding slots; finite checks are restricted to legal mask entries.
- Internal Yaw8 rows retain `start_frame/end_frame/fps` from the expanded manifest for AMASS root reconstruction.

## Status
**Completed** - CUDA read-only audit ran, reports were validated, and only this
task's files are ready for commit. Policy Test was not read.
