# EXP011 — Future-Perception Teacher Upper Bound

## Status

PLANNED — cache builder, teacher MLP, Val evaluator and run script prepared;
training has not been run.

## Scientific question

If actual future candidate perception is revealed, does candidate utility become
substantially easier to predict? This is a diagnostic teacher / upper-bound
experiment, not a deployable policy.

## Inputs

The teacher uses the frozen Stage C current 275-D feature, accepted 11-D
candidate geometry, and only future fields actually persisted by Stage B:

* candidate `predicted_label_id`, represented as a 16-D one-hot vector;
* candidate `logp_true`;
* candidate `entropy`.

Stage B does not persist a full candidate 16-D log-probability vector, pooled
ST-GCN feature or pose confidence. These fields are not fabricated. The
ground-truth-derived `correct` flag is excluded from the teacher input.

## Model and protocol

`FuturePerceptionTeacherMLP` is a simple candidate-wise MLP. It uses the same
SmoothL1 plus stay-inclusive listwise cross-entropy objective, record-balanced
16 Episodes/record sampling, seed 42, AdamW (1e-3, 1e-4), 100 epochs maximum,
patience 10 and Val Macro-F1 checkpoint selection as Stage C-v2. Only Train is
used for fitting and Val for evaluation; Test is locked.

`diagnostic_only=true`, `deployable_policy=false`, and
`future_candidate_perception_used=true` are persisted in the teacher schema.

## Pre-registered interpretation

Accuracy at least 70% or mean-regret improvement at least 20% would support an
information-limited interpretation. Improvement below 2 percentage points and
below 5% mean-regret improvement would indicate that future perception alone
does not explain the plateau. These are diagnostic interpretations, not model
acceptance criteria.

## Run

`run.sh` builds the cache from frozen Stage B/C-v0 files, trains the teacher and
runs a Val-only evaluator. It has not been executed.
