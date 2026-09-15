# Task Plan: PepperPose confidence–visibility complementarity audit

## Goal
Evaluate whether the frozen true-facing frame-0 pose-confidence angle prior adds complementary value to existing visibility selectors on Moving Val.

## Phases
- [x] Phase 1: Confirm frozen artifacts, schemas, and Train/Moving-Val row alignment
- [x] Phase 2: Implement rank-normalized fusion and complementarity diagnostics
- [x] Phase 3: Run CUDA-only Train-holdout lambda selection and Moving-Val evaluation
- [x] Phase 4: Verify reports, update project log, commit, and push

## Key Questions
1. Does confidence correct visibility selector errors on disagreement contexts?
2. Does rank-normalized fusion improve RGBGlobal or Frame0SceneVisibility?
3. Is any gain reproduced on the fixed high-occlusion subset?

## Decisions Made
- Reuse the prior internal true-facing confidence table; do not recompute angle sanity or confidence landscapes.
- Select lambda only on a deterministic 10% Policy-Train record holdout; Moving Val is evaluation-only.
- Preserve the existing candidate-only selector/action protocol and frozen Yaw8Fair recognizer.

## Errors Encountered
- Policy rows omit raw motion `source_path`; resolved root yaw through the canonical raw-val manifest.
- Existing visibility cache stores NaN in inactive padding slots; finite checks are restricted to legal mask entries.
- The prior helper's holdout function returns one index array; imported the strict rebaseline helper for `(fit_idx, holdout_idx)`.

## Status
**Completed** - CUDA diagnostic ran, all requested reports were validated, and the work is ready to commit.
