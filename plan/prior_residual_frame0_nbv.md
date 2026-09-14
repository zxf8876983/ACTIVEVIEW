# Task Plan: Static View Prior + Frame0 Residual NBV Audit

## Goal
Evaluate whether a Train-derived viewpoint prior plus a Frame0 RGB residual selector improves strict one-step NBV over prior-only and instance-only baselines.

## Phases
- [x] Phase 1: Inspect protocol, artifacts, selector architecture and CUDA runtime
- [x] Phase 2: Implement Train/Moving-Val audit and artifact writers
- [ ] Phase 3: Run audit on CUDA with Test locked
- [ ] Phase 4: Review metrics, update project context, commit and push

## Key Questions
1. Does Frame0 RGB predict residual utility beyond the static viewpoint prior?
2. Does the residual selector improve selected real-O1 recognition, especially under high occlusion?

## Decisions Made
- Reuse the frozen old-adaptive recognizer, strict current/Stay + Stage-A legal action set, existing DINO/geometry/visibility caches, and TaskUtilityPredictor architecture.
- Train residual outputs from Train-only GT margins; evaluate only Moving Val.
- Use fixed residual scales lambda=0.5 and 1.0; no Test access or hyperparameter sweep.

## Errors Encountered
- None yet.

## Status
**Currently in Phase 3** - running the minimal residual selector audit on CUDA.
