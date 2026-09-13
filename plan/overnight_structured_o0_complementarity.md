# Task Plan: Overnight structured O0 complementarity sweep

## Goal
Run the Train/Val-only reduced12 structured O0 representation and
candidate-conditioned complementarity audit without reading Policy Test or
generating perception data.

## Phases
- [x] Phase 1: Verify protocol, frozen cache provenance, intermediate feature capability, and CUDA
- [x] Phase 2: Implement resumable priors, structured branches, interactions, and diagnostics
- [x] Phase 3: Run Train/Val sweep and persist compact reports
- [x] Phase 4: Review leakage/metrics, stage only task files, commit, and push

## Key Questions
1. Does structured O0 representation improve pair-margin selection beyond the current 51.7% baseline and static pair priors?
2. Which body-part/time cues and candidate-conditioned interactions contribute useful complementary-view evidence?

## Decisions Made
- Keep the exact current/Stay + Stage-A legal candidate pool and fixed normalized MeanLogP fusion.
- Use frozen ST-GCN/shared-head features; any future-candidate evidence is restricted to Train targets or privileged Val references.
- Use 12 epochs, record-balanced 16 contexts per record, seed 42, and minimum Val utility loss for checkpoint selection.
- Run optional B3 only if the best structured B2 meets the specified 0.54 Accuracy and +2pp gates.

## Errors Encountered
- The first run exposed missing posterior concatenation for PartPool/Temporal/Motion branches; this was fixed locally and the completed run reused the frozen intermediate cache.
- Half-precision intermediate cache tensors are explicitly cast to float32 at model inputs.
- Geometry permutation indexing and direct-top1/pairwise diagnostic objectives were corrected before the final run.

## Status
**Completed** - all Train/Val branches and diagnostics ran on CUDA; protocol gate passed and no Policy Test data were read.
