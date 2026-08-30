# Current Task

## Stage D experiments completed

Stage C-v0 one-shot ranking is frozen. EXP011–EXP013 diagnostics indicate
moderate online utility predictability but strong Top-K proposal coverage, so
the approved sequential active view selection study has been executed on Val.

EXP014 (two-step sequential policy) and EXP015 (fixed-first budget/oracle
analysis) are **COMPLETED** with `test_used=false`. Their compact results and
controlled conclusions are in `experiments/stage_d/EXP014_two_step_sequential/`
and `experiments/stage_d/EXP015_budget_oracle_analysis/`.

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
- [x] Recorded compact metrics, provenance and Observation/Interpretation/
      Decision/Next notes.
- [ ] Await human scientific review before any follow-up experiment or Test.

## Current state

EXP014 selected epoch 24 and achieved Val Accuracy 0.664331, Macro-F1
0.615151, mean regret 1.397287, P90 regret 5.403128 and aggregate positive
headroom capture 0.783344. EXP015 found a 68.93% initial-Stay missed-move
rate under SafeOracle and a 0.467255 second-step action-match rate. Both
experiments used only Train/Val; no Test, Habitat, RGB, YOLO, VideoPose3D or
ST-GCN retraining was run. The current decision for both records is
**INCONCLUSIVE**, pending human review.
