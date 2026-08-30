# EXP021 analysis

EXP021 trained a new contextual scorer with Stage-D encoders and a two-layer
Transformer. Stay was fixed at score zero; p2/p3 scores were trained by
minimizing the negative mean expected Train reward under
`softmax([0,q2,q3])`. The fixed protocol was 30 epochs, Adam, learning rate
`1e-3`, batch size `256`, seed `42`, with no entropy bonus. True U2 was used
only as the full-information Train reward and never as model input or Val
action selection. Stage C-v0's first action/p1 was frozen. No Test was read.

Runtime evidence is under:

`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP021_contextual_bandit/`

## Train

| quantity | value |
|---|---:|
| Train episodes | 29,133 |
| final mean expected Train reward | `-1.15e-12` |
| final policy loss | `1.15e-12` |
| epochs / batch / learning rate | 30 / 256 / 0.001 |
| entropy bonus | not used |

The policy converged to the degenerate deterministic solution Stay for every
training and validation context (expected reward is effectively zero).

## Val action quality (9,742 frozen-v0-Move episodes)

| action | count | rate |
|---|---:|---:|
| Stay | 9,742 | 1.000000 |
| p2 | 0 | 0.000000 |
| p3 | 0 | 0.000000 |

Selected-action mean true utility was `0.0`. Exact action match with the
fixed-first oracle was `4,265 / 9,742 = 0.437795`; binary Move/Stay match was
the same. Candidate-hit denominator was zero because EXP021 never moved.

For reference, the offline diagnostic oracle (argmax over `[0, true_U2(p2),
true_U2(p3)]`, never used as a model input) selected Train actions Stay/p2/p3
`12,572 / 9,363 / 7,198` with mean true utility `1.413366`, and Val actions
Stay/p2/p3 `4,265 / 3,185 / 2,292` with mean true utility `1.240903`.

## Val trajectory metrics

| Variant | Accuracy | Macro-F1 | Mean regret | Median | P90 | Headroom | Avg moves | Mean geodesic (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 | 0.864946 | 2.522080 |
| EXP019 | 0.656681 | 0.607936 | 1.429851 | 0.003841 | 5.532555 | 0.780325 | 0.917209 | 2.554839 |
| EXP020 | 0.661757 | 0.612956 | 1.453868 | 0.002727 | 5.635810 | 0.778625 | 0.973833 | 2.640947 |
| EXP021 | 0.649103 | 0.598042 | 1.450498 | 0.005143 | 5.607818 | 0.777965 | 0.696504 | 2.200938 |
| Fixed-first Second-Step Oracle | 0.771502 | 0.725081 | 0.586204 | 0.000000 | 1.699901 | 0.890887 | 1.088082 | 2.859578 |

Relative to EXP014, EXP021 accuracy gain was `-0.009151` and mean-regret
reduction was `-0.028035` (regret increased). Against the fixed-first oracle,
accuracy recovery was `-8.08%` and regret recovery was `-3.35%`.

## Controlled interpretation

**Observation.** Direct expected-utility optimization selected Stay for all
9,742 Val v0-Move episodes, reproducing the one-step Stage C-v0 trajectory
rather than exploiting second-step actions. It underperformed EXP014 and was
far below the fixed-first oracle.

**Interpretation.** This first contextual-bandit formulation did not discover
a useful Move policy under the fixed representation and objective. The
degenerate Stay solution is evidence against escalating directly to PPO or
DQN; it does not justify changing frozen utility semantics or using Val to
tune rewards.

**Decision: REJECT** as a useful EXP021 policy formulation under this fixed
protocol. The negative result is retained; no upstream artifact was changed.

**Next (human review only):** consider whether a constrained/joint action
objective is scientifically warranted, or treat the current observable
representation as the bottleneck. Do not start EXP022 automatically.
