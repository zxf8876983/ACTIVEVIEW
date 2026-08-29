# EXP013 — Top-K Reachability Audit

## Status

PLANNED — post-hoc Val audit prepared; no diagnostics have been executed.

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
`topk_reachability.json` under the external runtime root. It has not been run.
