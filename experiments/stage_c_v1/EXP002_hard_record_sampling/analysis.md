# EXP002 Analysis

## Observation

- Val Accuracy: `0.6403088582254951`;
- Val Macro-F1: `0.5936060388261705`;
- mean regret: `1.459952538530646`;
- P90 regret: `5.6153066275874135`;
- positive headroom capture: `0.7731270769008732`;
- C2 rate: `0.31986844927432617`.

Relative to the frozen Stage C-v0 Val baseline, P90 regret changed from
`5.607817676663398` to `5.6153066275874135` rather than improving by the
required 5%. Mean regret, headroom and C2 also moved slightly in the wrong
direction. Macro-F1 decreased by `0.004435940647265824` (0.44 percentage
points).

## Interpretation

Hard motion records are strongly associated with catastrophic failures, but
simply increasing their training exposure does not improve Val tail regret.

The long-tail failure therefore appears more likely to reflect a
representation/discrimination limitation than simple under-exposure during
training. The hard-record concentration finding remains valid; this result
does not imply that hard records are unimportant.

## Decision

REJECT

## Next

EXP003 — Relative Geometry Representation.

## Protocol

Only Train was used for hard-record selection and only Val was evaluated. Test
was not used, and Stage A/B/C-v0 artifacts were not modified.
