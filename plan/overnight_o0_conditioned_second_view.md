# Task Plan: O0-conditioned complementary second-view sweep

## Goal
Run the frozen-recognizer, Train/Val-only O0-conditioned B2 selector sweep and persist auditable JSON/Markdown outputs without reading Policy Test.

## Phases
- [x] Phase 1: Verify protocol, cache provenance, geometry, and CUDA runtime
- [x] Phase 2: Implement pair-oracle, priors, learned B2 branches, and resumable runner
- [x] Phase 3: Run Train/Val diagnostics and persist all requested summaries
- [x] Phase 4: Review leakage/metrics, stage only task files, commit, and push

## Key Questions
1. Does normalized O0+candidate fusion expose complementarity beyond the historical raw candidate ranking oracle?
2. Can O0 feature/posterior plus legal candidate geometry predict complementary second-view utility without future-observation leakage?

## Decisions Made
- Reuse the validated reduced12 option caches, frozen ST-GCN feature cache, shared adapted head, and Stage-A legal candidate pools.
- Keep B2 as exactly O0 plus one non-stay legal candidate and evaluate with fixed MeanLogP.
- Train all learned branches on Train only with record-balanced 16-context sampling; select checkpoints by Val utility loss, never HAR accuracy.
- Skip conditional B3 unless the best learned B2 reaches the specified 0.50 accuracy gate.

## Errors Encountered
- The first run exposed an advanced-indexing shape error in pair target construction; it was fixed locally and the cached branches were rerun successfully.
- The final rerun added exact visibility-cache id/mask alignment validation and completed without error.

## Status
**Completed** - B2 and conditional B3 completed on CUDA. The best learned B2 branch reaches 0.517361 Acc / 0.504616 Macro-F1; learned B3 reaches 0.558036 / 0.548492, while the pair-margin oracle is 0.706647 / 0.701819. Leakage and cache-alignment audits passed; the task files are ready for a scoped commit.
