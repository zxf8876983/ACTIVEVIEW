# Reduced12 H2 verification bottleneck 2×2 decomposition

Train/Val only. No policy Test was read; no RGB, DINO, skeleton, WM-E, JR, or ST-GCN artifact was modified.

## Action inference

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| S0 ST-GCN | 0.254266 | 0.235500 |
| H1 ST-GCN | 0.471528 | 0.463723 |
| Mean(s0,h1) | 0.438591 | 0.416272 |
| PredAction | 0.540476 | 0.558889 |

## H2 2×2

| Method | Accuracy | Macro-F1 | H2 move | STOP | ΔAcc vs Candidate Spatial |
|---|---:|---:|---:|---:|---:|
| FrozenStageCv0 | 0.454266 | 0.444782 | 1.000000 | 0.000000 | -1.726 pp |
| Top3-GTSequentialVerifier | 0.642163 | 0.634401 | 0.584722 | 0.415278 | +17.063 pp |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 0.000000 | 0.000000 | +25.675 pp |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 | 0.000000 | 1.000000 | +0.000 pp |
| PredAction-RealEvidence | 0.508730 | 0.507227 | 0.480357 | 0.519643 | +3.720 pp |
| SoftPredAction-RealEvidence | 0.500496 | 0.500219 | 0.414187 | 0.585813 | +2.897 pp |
| GTAction-RealEvidence | 0.642163 | 0.634401 | 0.584722 | 0.415278 | +17.063 pp |
| GTAction-PredUtility | 0.543254 | 0.531460 | 0.444940 | 0.555060 | +7.173 pp |
| PredAction-PredUtility | 0.487401 | 0.480897 | 0.370933 | 0.629067 | +1.587 pp |
| SoftPredAction-PredUtility | 0.480357 | 0.476540 | 0.457440 | 0.542560 | +0.883 pp |

## Gaps

Total verification gap: +15.476 pp
Action inference gap: +13.343 pp
Utility prediction gap: +9.891 pp
Deployable interaction gap: +2.133 pp

On H1-wrong / Top3-correct contexts (n=1720), PredAction accuracy=0.357558; PredAction-RealEvidence success when PredAction is correct=1.000000, when wrong=0.087783.

These gaps are diagnostic decompositions and are not assumed to be strictly additive because action and utility errors interact.

## Scientific decision

The action-inference gap is the larger component; prioritize temporal/sequential action evidence.
GT-action predicted utility remains below 60%; observed state and candidate geometry do not yet provide a strong utility predictor.

test_used=false; future_rank2_rank3_evidence_used_as_deployable_input=false.
