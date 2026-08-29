# EXP013 — Top-K Reachability Audit

## Status

COMPLETED — post-hoc frozen-v0 Val audit; Test was not used.

## Scientific question

Does the frozen Stage C-v0 Set Ranker identify a useful candidate region even
when its Top-1 choice is wrong? Fixed K values are 1, 2, 3 and 5, with the
existing geodesic-then-viewpoint-ID tie-break. Candidate Top-K metrics are
reported separately from SafeOracle Stay episodes.

The audit reports CandidateOracle hit@K, SafeOracle move hit@K,
near-optimal hit@K (`epsilon=0.01`), positive-candidate recall@K and the
oracle regret of the best true candidate within the predicted Top-K.

## Interpretation boundary

This is a post-hoc Val diagnostic using frozen Stage C-v0 predictions and true
utility labels. Top-K oracle-best values are not online policy performance. A
large improvement from Top-1 to Top-3/Top-5 would be evidence for using the
policy as a sequential proposal mechanism, not an automatic design decision.

## Run

`run.sh` reads only frozen Val prediction and Stage B utility files and writes
`topk_reachability.json` under the external runtime root.

## Val result

| K | CandidateOracle hit | SafeOracle move hit | Near-optimal hit | Positive-candidate recall | Mean Top-K regret |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.66% | 33.66% | 52.86% | 77.33% | 0.88692 |
| 2 | 57.35% | 55.23% | 73.80% | 89.44% | 0.36989 |
| 3 | 71.34% | 69.76% | 84.60% | 94.50% | 0.17940 |
| 5 | 87.16% | 86.30% | 94.44% | 98.31% | 0.04842 |

Top-5 reduces the move-only P90 regret to `0.000076`, but these are offline
proposal-set diagnostics rather than online policy performance. The runtime
report is
`ACTIVEVIEW_DATA_ROOT/experiments/stage_c_v3/EXP013_topk_reachability/topk_reachability.json`.
