# Current Task

## EXP020 and EXP021 completed

Stage C-v0 and corrected EXP014–EXP019 remain frozen. EXP020 trained only a
fixed 129-D binary gate on the frozen EXP014 contextual candidate token plus
predicted utility. On 9,742 Val v0-Move episodes it reached Accuracy `0.661757`,
Macro-F1 `0.612956`, mean regret `1.453868`, P90 regret `5.635810` and
headroom `0.778625`; decision **INCONCLUSIVE** because regret worsened.

EXP021 trained a new contextual full-information bandit policy with fixed
Stay score zero and actions `{Stay,p2,p3}`. The 30-epoch expected-utility
objective collapsed to Stay for every Val v0-Move episode: Accuracy `0.649103`,
Macro-F1 `0.598042`, mean regret `1.450498`, P90 `5.607818`, headroom
`0.777965`; decision **REJECT**. Fixed-first oracle remained Accuracy
`0.771502` / mean regret `0.586204`.

## Protocol boundaries

- Both experiments used Train→Val only; Test was not read or used.
- No EXP014 retraining, Habitat rendering, perception regeneration, ST-GCN
  retraining, or Stage A/B/C-v0/EXP014–EXP019 artifact modification was
  performed.
- EXP020 and EXP021 are recorded for scientific review; no EXP022 should be
  started automatically.

## Status

EXP020 and EXP021 completed. Await human scientific review before authorizing
any next experiment.
