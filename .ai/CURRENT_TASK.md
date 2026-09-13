# Overnight structured O0 complementarity sweep — completed 2026-09-14

Completed the Train/Val-only reduced12 structured O0 complementarity sweep.
O0 is a complete already-observed current-view sequence; each B2 selector
chooses exactly one additional non-stay candidate from the Stage-A legal
candidate pool. Policy Test was not read, and no RGB/skeleton/DINO artifact,
recognizer, or formal checkpoint was modified.

Protocol gate reproduced Random B2 0.429762/0.424594, the historical current
selector 0.517361/0.504616, and PairMargin Oracle 0.706647/0.701819
(Accuracy/Macro-F1). Train used 46,324 contexts and Moving Val used 10,080;
all frozen ST-GCN intermediate features were cached outside Git on CUDA.

The best structured branch was RawO0-SmallEncoder at 0.517460 Accuracy /
0.503193 Macro-F1, only +0.010 pp Accuracy over the current selector.
DirectTop1Ranker reached 0.518056/0.508831 as a formulation diagnostic, but
does not change the structured-branch conclusion. Static and soft pair priors
were 0.508234 and 0.505952. O0 shuffle dropped 1.052 pp while geometry
shuffle dropped 9.325 pp, indicating candidate geometry remains dominant.
Body-part/time permutation and O0-correctness, confidence-quartile, and
low-visibility stratifications are recorded in the experiment directory.

The conditional B3 gate (best Accuracy >= 0.54 and gain >= +2 pp) was not met,
so B3 was skipped. The preregistered decision is **KILL STRUCTURED O0
REPRESENTATION** for this sweep; the result does not justify expanding this
representation family before an explicitly approved protocol change.

Report and implementation:
`experiments/reduced12_eight_placement_v1/overnight_structured_o0_complementarity/`
and
`activeview/scripts/experiments/run_reduced12_structured_o0_complementarity.py`.

Task status: **CLEAN**.
