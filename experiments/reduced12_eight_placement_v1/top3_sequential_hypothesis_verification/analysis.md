# Reduced12 Top-3 sequential hypothesis verification

Train/Val only; policy Test was not read and no RGB/DINO/skeleton data was regenerated.

## Moving Val

| Method | Accuracy | Macro-F1 | H2 move rate | STOP rate | ΔAcc vs Candidate Spatial |
|---|---:|---:|---:|---:|---:|
| Proposal-H1 | 0.471528 | 0.463723 | 0.000000 | 1.000000 | +0.000 pp |
| Proposal-H1-H2Fixed | 0.420833 | 0.413982 | 1.000000 | 0.000000 | -5.069 pp |
| Proposal-H1-H2Fixed-Fusion | 0.474008 | 0.453795 | 1.000000 | 0.000000 | +0.248 pp |
| S0-only | 0.254266 | 0.235500 | 0.000000 | 0.000000 | -21.726 pp |
| Candidate-Conditioned Spatial | 0.471528 | 0.463723 | 0.000000 | 1.000000 | +0.000 pp |
| Top3-SequentialVerifier-H2Only | 0.480556 | 0.475919 | 0.422321 | 0.577679 | +0.903 pp |
| Top3-SequentialVerifier-Fusion | 0.465774 | 0.443171 | 0.422321 | 0.577679 | -0.575 pp |
| AnyCorrect Oracle | 0.728274 | 0.729637 | 0.000000 | 0.000000 | +25.675 pp |
| Real-GTMargin Oracle | 0.728274 | 0.722059 | 0.000000 | 0.000000 | +25.675 pp |
| Top2-AnyCorrect | 0.584623 | 0.576027 | 0.000000 | 0.000000 | +11.310 pp |
| Top3-AnyCorrect | 0.642163 | 0.633458 | 0.000000 | 0.000000 | +17.063 pp |
| Top3-GTSequentialVerifier | 0.642163 | 0.634401 | 0.584722 | 0.415278 | +17.063 pp |
| Top3-GTSequentialVerifier-Fusion | 0.521528 | 0.499620 | 0.584722 | 0.415278 | +5.000 pp |
| FrozenStageCv0 | 0.454266 | 0.444782 | 0.000000 | 0.000000 | -1.726 pp |

## Sequential diagnostics

H1 proposal Accuracy=0.471528; H1→rank2 fixed Accuracy=0.420833; learned verifier H2-only=0.480556; GT verifier ceiling=0.642163.
H1 entropy change (H1 - s0)=-0.182404; H1 true-class logp gain=1.714796.
On H1-wrong contexts with a correct Top-3 option, learned verifier selected a correct remaining action at 0.401744, STOP at 0.415698; net rescue-harm=91.
Verifier utility Spearman (mean within-context)=0.173413; selected GT-best=0.440873; selected GT-top2=0.737202.

## Scientific decision

The learned sequential verifier remains below 55% Moving Accuracy; this run does not establish a large deployable gain from shortlist verification.
The GT sequential ceiling is at least 60%, while the learned verifier is substantially lower, indicating that the Top-3 shortlist is useful but H2 hypothesis discrimination remains the bottleneck.
No subsequent method was started automatically.

test_used=false; future_rgb_used=false; future_dino_used=false; deployable_learned_verifier=true.
