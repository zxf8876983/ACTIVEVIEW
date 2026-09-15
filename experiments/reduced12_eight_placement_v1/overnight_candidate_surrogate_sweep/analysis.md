# Overnight Candidate Utility Surrogate Sweep

Moving Val contains 10,080 contexts and 68,702 legal candidate observations. The frozen Yaw8 encoder/shared head and Stage-A legal candidate pool were reused; Policy Test was not read and no model or perception data was generated.

## Capacity table

| Method | Acc | Macro-F1 | Within rho(TrueLogP) | Within rho(Margin) | HighOcc Acc | Verdict |
|---|---:|---:|---:|---:|---:|---|
| StaticPrior | 0.549901 | 0.571990 | — | — | — | reference |
| RGBGlobal | 0.567361 | 0.582567 | — | — | — | reference |
| Frame0SceneVisibility | 0.583929 | 0.598955 | — | — | — | reference |
| A Feature-kNN k10 | 0.593254 | 0.603653 | 0.377299 | 0.182526 | 0.534270 | KILL |
| A Feature-kNN k20 | 0.592361 | 0.602541 | 0.377660 | 0.181343 | 0.531402 | KILL |
| B Global Mahalanobis | 0.509821 | 0.524887 | 0.162452 | 0.051087 | 0.460855 | KILL |
| B MinClass Mahalanobis | 0.532937 | 0.552239 | 0.206775 | 0.097099 | 0.477201 | KILL |
| C PCA64 | 0.452282 | 0.477136 | 0.087673 | 0.034714 | 0.379409 | KILL |
| C PCA128 | 0.459028 | 0.484320 | 0.097019 | 0.042337 | 0.382851 | KILL |
| D Perturbation Stability | 0.352976 | 0.359991 | -0.081396 | -0.030215 | 0.240608 | KILL |
| E Feature Medoid | 0.522619 | 0.547072 | 0.210867 | 0.189255 | 0.406653 | KILL |
| E Skeleton Medoid | 0.491468 | 0.519233 | 0.136864 | 0.091293 | 0.386579 | KILL |
| E Record Consensus | 0.608433 | 0.624189 | 0.399427 | 0.343204 | 0.542874 | WEAK SURROGATE |
| RGBGlobal + E Record Consensus (lambda=0.5) | 0.595833 | 0.612267 | 0.369486 | 0.250173 | 0.533983 | KILL |
| GT-TrueLogP Oracle | 0.760714 | 0.775961 | — | — | — | reference |
| F1 Matched Skeleton Error | unavailable | unavailable | — | — | — | unavailable |
| F2 Matched Feature Distance | unavailable | unavailable | — | — | — | unavailable |
| F3 Layerwise Feature Distance | unavailable | unavailable | — | — | — | unavailable |

Reference Random selector: Acc 0.426786, Macro-F1 0.450423; AnyCorrect Coverage: 0.776190 (7,824/10,080 contexts).

Best surrogate by Moving Accuracy: E Record Consensus (0.608433)
Strongest within-context absolute Spearman among A–E: 0.399427.

## Matched clean-perception branch

F1/F2/F3 were not scored: the existing recovery audit reports no exact per-candidate HM3D-vs-clean perception cache and a blocked Moving mapping. No clean rendering was attempted, so this branch is explicitly unavailable rather than approximated.

Final gate decision: **WEAK SURROGATE**.

## Scientific answers

1. A >=65% action-independent surrogate: no.
2. The table separates training-manifold membership (A/B), skeleton plausibility (C), local recognizer robustness (D), cross-view consensus (E), and matched perception error (F unavailable).
3. A future Frame0 predictor target is warranted only if a surrogate passes the hard gate; in this run the registered decision is WEAK SURROGATE.
4. If all A–E remain below 0.60, stop scalar-surrogate search and move to a low-cost candidate glimpse/active-probe information structure.

Flags: `policy_test_used=false`; `training_used=false`; `new_rgb_generated=false`; `new_skeleton_generated=false`; `frozen_stgcn_modified=false`; `deployable=false`.
