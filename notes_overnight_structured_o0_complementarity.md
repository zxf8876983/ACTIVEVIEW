# Notes: Overnight structured O0 complementarity sweep

## Scope

The experiment will use the reduced12 eight-placement current-view O0
protocol: one complete, already acquired current-view observation followed by
one additional Stage-A legal candidate. Moving Val contains 10,080 contexts;
Policy Test, new perception generation, and recognizer fine-tuning are
forbidden.

## Prior reference

- Random B2 MeanLogP: 0.429762 Accuracy / 0.424594 Macro-F1.
- Current Feature+Posterior+Geometry selector: 0.517361 / 0.504616.
- Pair-margin oracle: 0.706647 / 0.701819.
- Pair AnyCorrect coverage: 7,123 / 10,080.

## Findings

- Protocol reproduction passed exactly: Random B2 0.429762/0.424594, current baseline 0.517361/0.504616, PairMargin Oracle 0.706647/0.701819.
- Best structured branch by Moving-Val selection was RawO0-SmallEncoder at 0.517460 Accuracy / 0.503193 Macro-F1, only +0.010 pp over the current branch.
- DirectTop1Ranker reached 0.518056/0.508831 but remains a diagnostic formulation, not a registered structured representation winner.
- Static and soft pair priors were 0.508234 and 0.505952, below the learned current baseline.
- Geometry shuffle caused a 9.325 pp drop for the best branch, while O0 shuffle caused a 1.052 pp drop; this indicates the selector still relies primarily on candidate geometry.
- Conditional B3 gate was not met (best Accuracy 0.517460, gain +0.010 pp), so B3 was correctly skipped.
