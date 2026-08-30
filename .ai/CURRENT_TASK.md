# Current Task

## EXP022 and EXP023 completed

Stage C-v0 and corrected EXP014–EXP021 remain frozen. EXP022 trained only a
fixed 129-D raw executed-candidate utility regressor on Train and applied the
strict `predicted_U_exec > 0` gate once on Val. On 9,742 Val v0-Move episodes it
reached Accuracy `0.659898`, Macro-F1 `0.611687`, mean regret `1.416495`, P90
regret `5.494913` and headroom `0.782352`; it is **ACCEPTED as
research-direction evidence**, not deployment acceptance.

EXP023 trained a fixed contextual bandit in two Train-only phases: 20 epochs
of candidate-U2 SmoothL1 warm-start followed by 10 epochs of expected-reward
fine-tuning with entropy bonus `0.001`. It avoided EXP021's all-Stay collapse
and reached Val Accuracy `0.660470`, Macro-F1 `0.608566`, mean regret
`1.374664`, P90 `5.294162` and headroom `0.786731`; it is **ACCEPTED as
research-direction evidence**, not final-policy acceptance. Fixed-first oracle
remains Accuracy `0.771502` / mean regret `0.586204`.

## Protocol boundaries

- EXP022 and EXP023 used fixed Train→Val protocols; Test was not read or used.
- No EXP014 retraining, Habitat rendering, perception regeneration, ST-GCN
  retraining, or Stage A/B/C-v0/EXP014–EXP021 artifact modification was
  performed.
- No Val tuning or threshold/architecture search was performed. EXP024 must
  not be started automatically.

## Status

EXP022 and EXP023 are complete and recorded for human scientific review.
Await the next explicit research authorization.
