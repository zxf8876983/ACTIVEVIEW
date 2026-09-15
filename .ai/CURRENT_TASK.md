# Overnight candidate utility surrogate sweep — completed

Implemented and ran
`activeview/scripts/eval/run_reduced12_overnight_candidate_surrogate_sweep.py`
on CUDA (`cuda:0`, RTX 4090) using the frozen Yaw8 encoder/shared head,
existing Yaw8Fair Train/Moving-Val caches and Stage-A legal candidate pool.
No Policy Test data, new RGB/skeleton generation, recognizer modification or
model training was used.

Moving Val contained 10,080 contexts and 68,702 legal candidate observations;
the Train manifold bank contained 17,696 Yaw8 skeleton observations. The
fixed references were reproduced: Random 0.426786/0.450423, StaticPrior
0.549901/0.571990, RGBGlobal 0.567361/0.582567,
Frame0SceneVisibility 0.583929/0.598955, GT-TrueLogP Oracle
0.760714/0.775961 and AnyCorrect Coverage 0.776190 (Accuracy/Macro-F1 where
applicable).

The strongest action-independent surrogate was privileged E Record Consensus:
0.608433 Accuracy, 0.624189 Macro-F1, within-context Spearman 0.399427 with
true-class log-probability and 0.343204 with GT margin. It therefore passed
the conditional 0.60 gate but remained a **WEAK SURROGATE** (below the 0.65
promotion threshold). The fixed Train-record-holdout RGBGlobal fusion selected
lambda=0.5 and reached 0.595833/0.612267, below Record Consensus alone.

Feature kNN was next (k10 0.593254/0.603653; k20 0.592361/0.602541), while
Mahalanobis, skeleton PCA, local perturbation stability and feature/skeleton
medoids were weaker. No simple surrogate met the >=0.65 promotion gate or
within-context rho >=0.50. The matched clean-perception F1/F2/F3 branch was
explicitly marked unavailable because the existing recovery audit has no exact
per-candidate HM3D-vs-clean cache and zero exact Moving mappings; no rendering
was attempted.

Reports are under
`experiments/reduced12_eight_placement_v1/overnight_candidate_surrogate_sweep/`.
The registered conclusion is to retain the negative result, stop scalar
surrogate expansion, and consider a low-cost candidate glimpse/active-probe
information structure rather than promoting a deployable scalar predictor.
