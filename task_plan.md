# Overnight candidate utility surrogate sweep

## Goal
Run the preregistered, read-only capacity audit of action-independent
candidate observation-quality surrogates on reduced12 Moving Val.

## Phases
- [x] Inventory frozen Yaw8 assets, legal candidate cache and available archives
- [x] Implement feature-manifold, PCA, stability and consensus surrogates
- [x] Audit matched clean-perception artifact availability; run only if exact data exists
- [x] Evaluate Moving Val selectors, correlations, high-occlusion subset and gates
- [x] Validate reports, update project log, commit and push

## Key Questions
1. Which action-independent observation-quality surrogate is closest to HAR utility?
2. Does matched clean-perception degradation explain candidate utility?
3. Is any surrogate strong enough to promote or should surrogate search stop?

## Decisions Made
- No angle prior, PepperPose prior, direct HAR utility predictor or future
  evidence predictor will be added.
- Only frozen Yaw8 encoder/shared head and existing Train/Moving-Val artifacts
  are used; Policy Test is excluded.
- F (matched clean perception) is conditional on exact reusable artifact
  availability; no large rendering job will be launched implicitly.

## Errors Encountered
- Policy rows omit raw motion `source_path`; resolved root yaw through the canonical raw-val manifest.
- Existing visibility cache stores NaN in inactive padding slots; finite checks are restricted to legal mask entries.
- Internal Yaw8 rows retain `start_frame/end_frame/fps` from the expanded manifest for AMASS root reconstruction.
- Exact per-candidate HM3D-vs-clean perception cache is unavailable: historical
  recovery reports zero exact Moving mappings, so F1/F2/F3 were explicitly
  marked unavailable and no rendering was attempted.
- The initial run correctly reproduced the fixed references, then was rerun
  after enabling the conditional RGBGlobal+surrogate fusion gate for the
  qualifying Record Consensus branch.

## Status
**Complete** - A–E evaluated on GPU over Moving Val; F unavailable by artifact
  gate; reports and conditional fusion result written.
