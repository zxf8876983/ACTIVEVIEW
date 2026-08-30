# EXP023 analysis

EXP023 tested the final rapid one-step contextual-bandit intervention. The
existing Stage-D contextual scorer was first warm-started by 20 Train-only
epochs of masked candidate-U2 SmoothL1 regression (Phase A), then fine-tuned
for 10 Train-only epochs with the fixed full-information expected-reward loss
and entropy bonus `beta=0.001` (Phase B, Adam lr `1e-4`). Stay retained a
fixed score of zero. No Val selection, Test access, perception, Habitat or
ST-GCN processing was performed.

Runtime evidence is under:

`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP023_warmstarted_contextual_bandit/`

## Train

| quantity | value |
|---|---:|
| episodes | 29,133 |
| Phase-A final SmoothL1 / utility MAE | 2.064839 / 2.364638 |
| Phase-B final loss | -0.659894 |
| Phase-B final mean expected reward | 0.659802 |
| Phase-B final entropy | 0.092387 |
| final mean policy probabilities (Stay/p2/p3) | 0.728921 / 0.161081 / 0.109998 |
| all-Stay collapse indicator | false |

The warm-start prevented EXP021's all-Stay collapse and retained positive
expected Train reward throughout Phase B.

## Val action quality (9,742 frozen-v0-Move episodes)

| action | count | rate |
|---|---:|---:|
| Stay | 6,903 | 0.708581 |
| p2 | 1,730 | 0.177582 |
| p3 | 1,109 | 0.113837 |

Exact fixed-first oracle action match was `4,518/9,742 = 0.463765`; binary
Move/Stay match was `5,302/9,742 = 0.544241`. Among episodes where both model
and oracle moved, candidate exact hit was `1,154/1,938 = 0.595459`. The
selected-action mean true utility was `0.108878` (offline diagnostic only).

For comparison, the offline diagnostic oracle selected Train Stay/p2/p3
`12,572/9,363/7,198` and Val `4,265/3,185/2,292`; its selected-action mean
true utilities were `1.413366` and `1.240903`.

## Val trajectory metrics

| Variant | Accuracy | Macro-F1 | Mean regret | Median | P90 | Headroom | Avg moves | Mean geodesic (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 | 2.522080 |
| EXP021 | 0.649103 | 0.598042 | 1.450498 | 0.005143 | 5.607818 | 0.777965 | 0.696504 | 2.200938 |
| EXP023 | 0.660470 | 0.608566 | 1.374664 | 0.003706 | 5.294162 | 0.786731 | 0.899478 | 2.597665 |
| Fixed-first Second-Step Oracle | 0.771502 | 0.725081 | 0.586204 | 0 | 1.699901 | 0.890887 | 1.088082 | 2.859578 |

Relative to EXP014, EXP023 gains `+0.002216` Accuracy and reduces mean regret
by `0.047799`, recovering `1.96%` of the fixed-first Accuracy gap and `5.72%`
of the regret gap. EXP023 exceeds EXP021 on both Accuracy and regret.

**Observation.** Supervised warm-start followed by expected-reward fine-tuning
selects p2/p3 on Val, keeps positive Train expected reward, and avoids the
EXP021 all-Stay solution. It produces a modest trajectory improvement over
EXP014 and a larger improvement over EXP021.

**Interpretation.** EXP021's collapse was substantially optimization/
initialization-related under its random start. The warm-started bandit route
has limited but reproducible evidence of usefulness under the frozen
representation and reward; the remaining oracle gap is still large.

**Decision: ACCEPT (research-direction evidence).** This is not final-policy
or Test acceptance, and it does not authorize PPO/DQN or EXP024.

**Next (human review only):** decide whether the modest warm-start bandit and
raw-utility-gate gains justify a separately preregistered follow-up.
