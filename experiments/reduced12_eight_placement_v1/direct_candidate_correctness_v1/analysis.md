# Direct Candidate Correctness Ranking

Moving Val contexts: 10080

## Protocol

The scorer uses only current Stage-C features/posterior, current viewpoint id, candidate geometry/id and an explicit stay flag. Candidate frozen-recognizer outputs and labels are Train supervision only.

## Moving Val

| Selector | Accuracy | Macro-F1 | Correct-candidate hit | Any-correct contexts | No-correct contexts | Move rate |
|---|---:|---:|---:|---:|---:|---:|
| S0-only | 0.254266 | 0.235500 | 0.000000 | 7153 | 2927 | 0.000000 |
| FrozenStageCv0 | 0.454266 | 0.444782 | 0.454266 | 7153 | 2927 | 1.000000 |
| Random candidate | 0.333829 | 0.331380 | 0.333829 | 7153 | 2927 | 1.000000 |
| OldBestOneShot | 0.458730 | 0.446342 | 0.452950 | 7153 | 2927 | 0.972024 |
| Direct-Correctness | 0.444940 | 0.431100 | 0.436549 | 7153 | 2927 | 0.964683 |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 1.000000 | 7153 | 2927 | 0.474008 |
| BestSingle Oracle | 0.727976 | 0.721495 | 0.736288 | 7153 | 2927 | 0.886310 |

Direct-Correctness Δ vs FrozenStageCv0: -0.009325 accuracy points; Δ Macro-F1: -0.013682.
FAILURE: Accuracy <= 49%; stop this direction.

## Leakage audit

No GT label, future candidate observation/recognizer output, candidate correctness, or hard predicted action is used by the inference scorer. Test was not read.

## Scientific decision

The direct-correctness target is task-aligned and preserves all correct candidates as positives. The result is a single no-tuning experiment; no second version or additional method is started automatically.

Train convergence: selected epoch 7 with best Val loss 1.044068; corresponding Train loss 1.101332.
No-correct-candidate fallback was used for 15578 Train and 4514 Val contexts; fallback supervision used the maximum true-class margin candidate.
