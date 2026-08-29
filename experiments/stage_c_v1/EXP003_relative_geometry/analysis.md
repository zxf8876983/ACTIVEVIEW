# EXP003 Val Analysis — Relative Geometry Representation

## Protocol

- Split: Val only (13,987 Episodes; 197 held-out motion records).
- Test was not read for model selection or evaluation (`test_used=false`).
- Single scientific change: append five candidate-set-relative geometry
  features, changing candidate geometry from 11-D to 16-D.
- Frozen: Stage A/B/C-v0 artifacts, ST-GCN, current feature, Set Ranker,
  record-balanced sampler, loss, optimizer, split and decision rule.
- Best checkpoint: epoch 46, selected by Val Stage-C recognition Macro-F1.

## Val metrics

| Metric | Frozen C-v0 | EXP003 | Delta (EXP003 − v0) |
|---|---:|---:|---:|
| Accuracy | 0.6491027 | 0.6549653 | +0.0058626 (+0.586 pp) |
| Macro-F1 | 0.5980420 | 0.6048748 | +0.0068328 (+0.683 pp) |
| Mean regret | 1.4504977 | 1.4201176 | −0.0303800 (2.094% relative) |
| Median regret | 0.0051434 | 0.0045975 | −0.0005459 |
| P90 regret | 5.6078177 | 5.5006064 | −0.1072113 (1.912% relative) |
| Positive headroom capture | 0.7779653 | 0.7841998 | +0.0062345 (+0.623 pp) |
| C2 rate | 0.3174376 | 0.3300922 | +0.0126546 (+1.265 pp; worse) |

The preregistered primary target was at least 5% relative reduction in mean
regret (target ≤ 1.3779728). EXP003 reached a 2.094% reduction and is therefore
REJECTED under the registered criterion.

## Geometry-bias diagnostic

The diagnostic below was recomputed read-only from the EXP003 Val feature cache,
using the first 11 frozen geometry columns. It does not use labels, utilities,
or future perception as model inputs.

- The learned policy moved in 71.75% of Episodes, versus 85.35% for SafeOracle.
- Among moved Episodes, the policy strongly preferred closer viewpoints:
  candidate radius mean 1.610 m and Δradius mean −0.709 m, versus SafeOracle
  1.941 m and −0.286 m.
- Radius-direction rates were closer/same/farther = 77.09%/19.00%/3.91% for
  the policy, versus 55.01%/20.20%/24.79% for SafeOracle. Mean radius rank was
  0.141 versus 0.346 (0 = nearest, 1 = farthest).
- Geodesic rank was similar (0.467 policy vs 0.455 SafeOracle), while absolute
  selected geodesic distance was slightly larger (3.079 m vs 2.903 m). This
  does not indicate a strong shortest-path bias; the dominant bias is radial.
- Absolute azimuth was mildly shifted toward lateral/back views (mean/median
  78.0°/90° policy vs 72.0°/45° SafeOracle), without a single dominant angle
  bin.
- Per-Episode predicted-utility/radius Pearson correlation averaged −0.591
  (median −0.700), compared with −0.325 (median −0.424) for true utility.
  This exploratory comparison indicates that the predictor amplifies the
  data's existing preference for nearer candidates.
- In the P90 high-regret group (threshold 5.5006), 1,399 Episodes were high
  regret; 1,248 had a SafeOracle move. Their SafeOracle candidates had mean
  Δradius −0.124 m, radius rank 0.431 and farther rate 31.89%, suggesting that
  catastrophic cases more often require same/farther-radius candidates than
  the policy typically selects.

## Tooling note

The generic failure-analysis helper now accepts variant geometries and reads
only the first 11 frozen base columns. The values above remain an independent
calculation from the serialized cache; this tooling change does not alter model
metrics or any upstream accepted artifact.

## Observation / Interpretation / Decision / Next

### Observation

EXP003 improves Val Accuracy, Macro-F1, mean regret, P90 regret and headroom,
but worsens C2 rate. Mean-regret improvement is below the preregistered 5%
target. The policy retains a pronounced near-viewpoint selection bias.

### Interpretation

Relative set statistics provide a modest overall improvement, but do not remove
the learned near-radius preference. The high-regret geometry pattern is
consistent with missed same/farther-radius opportunities; this is evidence for
further diagnosis, not proof that another feature or model change is required.

### Decision

REJECT — the 5% mean-regret target was not reached. Positive secondary signals
and the radius-bias diagnostic are retained as evidence for separate follow-up
experiments, not as acceptance of EXP003.

### Next

1. Review the independently prepared EXP004–EXP007 protocols.
2. Use the repaired analyzer compatibility as a diagnostic aid without
   changing the frozen Stage C-v0 baseline.
