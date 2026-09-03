# EXP054 analysis

EXP054 freezes the Stage-C-v0 first action and the first Joint-Revision choice,
then evaluates only the second step on the canonical 9,742 moving Val
episodes. Terminal recognition is always the archived real ST-GCN prediction.
The matched selector is trained on Train true-recognition inputs only; it is a
privileged ceiling diagnostic, not a deployable policy.

## Results

| Method | Moving Accuracy | Moving Macro-F1 | Full Accuracy | Full Macro-F1 |
|---|---:|---:|---:|---:|
| H1 real | 0.661568 | 0.632699 | 0.685780 | 0.643640 |
| Current WM + frozen JR | 0.675529 | 0.642612 | 0.695503 | 0.649220 |
| True future + frozen JR | 0.668651 | 0.637410 | 0.690713 | 0.645941 |
| True future + matched JR | 0.671115 | 0.660819 | 0.692429 | 0.665222 |
| True-future GT-label oracle | 0.837918 | 0.825241 | 0.808608 | 0.786767 |

The matched true-future selector changes 1,872 second actions relative to the
current rollout: 323 rescued, 366 harmful, net −43. Current and matched
second-action agreement with the GT oracle is 0.683330 and 0.689591,
respectively (Stay is represented as the explicit no-second-action outcome).

## Ceiling decomposition

On moving Accuracy, the current→GT gap is 0.162390. The matched selector
recovers −0.004414 (−2.72%); 0.166804 remains as decision/oracle headroom
(102.72% of the current→GT gap). The result is therefore **Case C — little
world-model headroom**: replacing imagined candidate recognition with true
recognition and retraining the selector did not improve trajectory Accuracy.
The true-future frozen-JR row is a domain-shift diagnostic and is not evidence
that real future observations are intrinsically worse.

No Test data, new perception, Habitat rendering or WM training were used.
