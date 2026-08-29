# EXP011 — Future-Perception Teacher Upper Bound

## Status

COMPLETED — corrected 17-D schema, Train→Val only. The initial schema exposed
the GT-dependent `logp_true` field; it was removed before this run. Retention
as a scientific direction remains pending review.

## Scientific question

If actual future candidate perception is revealed, does candidate utility become
substantially easier to predict? This is a diagnostic teacher / upper-bound
experiment, not a deployable policy.

## Inputs

The teacher uses the frozen Stage C current 275-D feature, accepted 11-D
candidate geometry, and only future fields actually persisted by Stage B:

* candidate `predicted_label_id`, represented as a 16-D one-hot vector;
* candidate `entropy`.

Stage B does not persist a full candidate 16-D log-probability vector, pooled
ST-GCN feature or pose confidence. These fields are not fabricated. The
ground-truth-dependent `logp_true` is excluded because it is a direct
ingredient of the Stage B utility target. The ground-truth-derived `correct`
flag is also excluded from the teacher input.

## Model and protocol

`FuturePerceptionTeacherMLP` is a simple candidate-wise MLP. It uses the same
SmoothL1 plus stay-inclusive listwise cross-entropy objective, record-balanced
16 Episodes/record sampling, seed 42, AdamW (1e-3, 1e-4), 100 epochs maximum,
patience 10 and Val Macro-F1 checkpoint selection as Stage C-v2. Only Train is
used for fitting and Val for evaluation; Test is locked.

`diagnostic_only=true`, `deployable_policy=false`, and
`future_candidate_perception_used=true` are persisted in the teacher schema.
The future input is exactly 17-D: predicted-label one-hot plus entropy.

## Pre-registered interpretation

Accuracy at least 70% or mean-regret improvement at least 20% would support an
information-limited interpretation. Improvement below 2 percentage points and
below 5% mean-regret improvement would indicate that future perception alone
does not explain the plateau. These are diagnostic interpretations, not model
acceptance criteria.

## Val result

The corrected run selected epoch 41 by Val Macro-F1. Accuracy was `65.21%`,
Macro-F1 `61.99%`, mean regret `1.64995`, median regret `0.00102`, P90 regret
`6.53404`, and aggregate positive headroom capture `76.70%`. Candidate exact
hit was `41.58%` and SafeOracle action match was `36.08%`.

Relative to frozen v0 Val (Accuracy `64.91%`, Macro-F1 `59.80%`, mean regret
`1.45050`, P90 `5.60782`, headroom `77.80%`), this restricted future-feature
teacher improved classification but not regret or headroom. It is therefore
not evidence that the available future predicted-label/entropy fields alone
explain the policy plateau, and it must not be described as a complete
future-perception upper bound.

Runtime evidence is stored under
`ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v3/EXP011_future_perception_teacher/`.

## Run

`run.sh` builds the cache from frozen Stage B/C-v0 files, trains the teacher and
runs a Val-only evaluator. The recorded runtime is external to Git.
