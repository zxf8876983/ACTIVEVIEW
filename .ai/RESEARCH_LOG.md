# ACTIVEVIEW Research Log

## 2026-08-29 — Phase 0 initialization

- Stage C-v0 completed.
- Stage C-v0 failure analysis PASS.
- Phase -1 repository consolidation PASS.
- Phase 0 research infrastructure started and completed.
- No Stage C-v1 scientific experiment has started.
- No model training, data regeneration, or new Test evaluation was performed.

Future entries are append-only and must record: Experiment ID, hypothesis,
single core change, Val result, decision, important observation, and Git commit.

## 2026-08-29 — EXP001 created / PLANNED

- Experiment ID: `EXP001` (`gap_aware_ranking`).
- Hypothesis: emphasizing larger ground-truth utility gaps in the Set Ranker
  objective will reduce harmful viewpoint-selection regret on Val while
  preserving recognition performance.
- Single core change: add a stay-inclusive utility-gap-weighted pairwise
  ranking term; `lambda_gap=1.0`, `tau_gap=1.0`, `max_weight=10.0`.
- Status: `PLANNED`; no start authorization, Val result, decision, training,
  data regeneration, or Test evaluation exists yet.
- The accepted Stage A/B/C-v0 artifacts, features, ST-GCN checkpoint, split,
  architecture and sampler remain frozen.

- Frozen Stage C-v0 Set Ranker Val baseline metrics and analysis were generated
  read-only before start and hashed in the EXP001 manifest. This preparation
  does not constitute an experiment result; no model training or Test
  evaluation occurred.

## 2026-08-29 — Phase 0 lifecycle hardening

- No scientific experiment was created; the registry remains header-only.
- Start-time execution freeze now records the actual run commit and hashes for
  config, hypothesis and command; draft config edits before start are allowed.
- Frozen Stage A/B/C artifacts are re-hashed during validation, and controlled
  config fields enforce `test=false`, `test_locked=true`, and
  `test_authorized=false` before start.
- The final Test gate is canonical nested-manifest only and the lifecycle
  integration/rollback regressions are covered by the research test suite.
- Status: awaiting final human review. No training, data regeneration, or new
  Test evaluation was performed.
- Final-test authorization is separated from tracked final-candidate freezing:
  freeze, commit, then authorize externally under the runtime root.
- Experiment source/runtime references are portable relative paths; validators
  also re-hash the locked hypothesis and command files after start.
