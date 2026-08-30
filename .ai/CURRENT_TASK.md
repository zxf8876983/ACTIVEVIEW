# Current Task

## Stage D geometry-semantics correction and Val rerun complete

Stage C-v0 one-shot ranking is frozen. EXP011–EXP013 diagnostics indicate
moderate online utility predictability but strong Top-K proposal coverage, so
the approved sequential active view selection study exposed a geometry semantic
issue during post-run audit.

EXP014 and EXP015 were rerun with `test_used=false` after rebuilding the cache
with the Stage A-compatible radial-azimuth correction. The pre-fix outputs are
archived for traceability and are not used for the corrected results.

## Frozen

Stage A/B/C-v0 artifacts, ST-GCN checkpoint, motion split 589/197/194,
candidate pool, current-view protocol, Stage C-v0 proposal ordering and all
perception data remain frozen.

## Protocol boundaries

- Train and Val only; Test is locked.
- No Habitat rendering, RGB/depth, YOLO, VideoPose3D or ST-GCN retraining.
- No Stage C-v0 retraining, loss/sampler/threshold changes, Top-K sweep,
  exploration or body-yaw feature.
- EXP014 may use only visited s1 perception; unvisited p2/p3 perception is
  never a policy input.
- EXP015 performs no training and fails clearly if EXP014 Val output is absent.

## Preparation status

- [x] Added frozen-v0 Train/Val proposal inference helper.
- [x] Added navigation-only pairwise viewpoint geodesic builder.
- [x] Added Stage D second-step cache schema, s1 frozen ST-GCN reconstruction,
      s1-relative 11-D geometry and U2 supervision construction.
- [x] Added SequentialObservationRanker, Val trajectory evaluator and
      Fixed-first Second-Step Oracle analysis.
- [x] Added EXP014/EXP015 README, config, run scripts and registry entries.
- [x] Added focused Stage D unit tests and compile check.
- [x] Executed EXP014 Train→Val and EXP015 Val-only analysis under the frozen
      Stage A/B/C-v0 protocol.
- [x] Audited Stage D geometry semantics and identified the pre-fix azimuth
      mismatch.
- [x] Corrected the cache builder to use Stage A radial metadata without
      changing model, loss, protocol or perception artifacts.
- [x] Rebuild Stage D cache and rerun approved Val-only experiments.
- [ ] Await human scientific review before any follow-up experiment or Test.

## Current state

The pre-fix EXP014/EXP015 metrics are retained in their compact records but
are not used for the corrected final interpretation because Stage D sixth/seventh
geometry features used the wrong relative-azimuth semantics. The corrected run
loads existing semantic-region-v2 `azimuth_deg` metadata and applies
`candidate_azimuth - s1_azimuth` wrapped to [-180°, 180°). Corrected EXP014
reached Accuracy 0.658254, Macro-F1 0.610153, mean regret 1.422463, P90 regret
5.515663 and headroom 0.783313; its recorded decision is REJECT. EXP015 is an
analysis-only INCONCLUSIVE diagnostic. No Stage A/B/C-v0, perception, model or
loss artifact was changed, and the pre-fix runtime remains archived.
