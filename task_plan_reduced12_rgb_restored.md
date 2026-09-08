# Task Plan: Restore reduced12 RGB-conditioned ActiveView

## Goal
Restore the pre-14-class formal Recognition-aware WM-E + Multi-positive JR + closed-loop H2 protocol for reduced12 using only Train/Val and visited s0/s1 RGB/DINO caches.

## Phases
- [x] Phase 1: Inspect protocol, artifacts, scripts, and runtime environment
- [x] Phase 2: Run integrity checks and stop on any mismatch
- [x] Phase 3: Generate/deduplicate visited s0/s1 RGB and DINO spatial caches
- [x] Phase 4: Train reduced12 formal WM-E and rebuild counterfactual cache
- [x] Phase 5: Train reduced12 formal Multi-positive JR and evaluate Val
- [x] Phase 6: Write results/analysis, verify no Test access, commit and push

## Key Questions
1. Are reduced12 Stage-B labels/log-probabilities aligned with frozen ST-GCN outputs?
2. Do reduced12 Train/Val assets and candidate metadata cover the required visited observations without Test?
3. Can the formal RGB-conditioned WM-E/JR scripts be reused without later experimental heads or losses?

## Decisions Made
- Preserve all existing reduced12 no-RGB results and checkpoints.
- Use a new runtime/results root for RGB-restored artifacts.
- Do not stage unrelated worktree files.

## Errors Encountered
- Initial multi-worker renderer sessions overlapped and were stopped by exact PID
  after external-session termination left orphan workers. Existing atomic output
  files were preserved; the complete set was then regenerated/validated with a
  single worker per scene.
- The first full NPZ decompression audit was stopped as too IO-heavy; a complete
  path-set audit plus deterministic schema samples then passed.

## Status
**Completed** - RGB/DINO caches, formal WM-E/JR Train runs, and Val benchmark
are complete. Test was not read. Only task-owned files are staged for commit;
pre-existing worktree changes remain untouched.
