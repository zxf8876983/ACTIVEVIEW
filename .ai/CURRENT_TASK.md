# Current Task

## H1 Disambiguation Potential — completed 2026-09-07

Implemented and ran the reduced14 Val-only H1 disambiguation diagnostic. All
legal archived candidates from each Val moving context were evaluated with the
frozen pretrained history-identity classifier, comparing current Stage-C H1,
seeded random, minimum-entropy, maximum-margin and label-conditioned
IdentityOracle selectors.

Results are recorded in
`experiments/reduced14_eight_placement_v1/h1_disambiguation/`.
Val contexts: 14,809; candidate hypotheses: 441,283. Frozen H1 identity
Accuracy/Macro-F1: 0.482815/0.496501. IdentityOracle:
0.899588/0.900413. Test was not read and no formal checkpoint was modified.

Status: CLEAN. No follow-up experiment is authorized automatically.
