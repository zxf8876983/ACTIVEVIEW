# EXP050 analysis

The set-aware revision model was trained for the fixed 20 epochs on Train only.
At `ALL_LEGAL`, JOINT_REVISION obtained Accuracy 0.686566 and Macro-F1
0.646064, versus the independent CF_CORRECTNESS_MLP Accuracy 0.676628 and
Macro-F1 0.629705. Paired bootstrap (10,000, seed 42) gives ΔAccuracy
0.009938 (95% CI [0.005148, 0.014728]) and ΔMacro-F1 0.016359 (95% CI
[0.009153, 0.023854]); McNemar p=5.91e-05. This satisfies the preregistered
single-step gate for EXP051. Full M-wise metrics and provenance are in
`result.json` and `paired_statistics.json`.

No Test access or perception/model retraining occurred.
