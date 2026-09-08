# Reduced12 visited-history action belief estimator

Train/Val only. No policy Test data was read; terminal predictions use the selected real archived observation and frozen reduced12 ST-GCN.

## Belief-only Val Moving diagnostic

- S1-only Accuracy/F1: **0.454266 / 0.444782**.
- Learned belief Accuracy/F1: **0.547619 / 0.566860**.
- Mean entropy S1/learned: **0.402727 / 0.603012**.

## Moving Val strategy comparison

| Method | Accuracy | Macro-F1 | Positive-action hit | Stay rate |
|---|---:|---:|---:|---:|
| Frozen H1 | 0.454266 | 0.444782 | 0.454266 | 1.000000 |
| Imagined + Inferred JR | 0.535615 | 0.536320 | 0.535615 | 0.372520 |
| Real + Inferred JR | 0.598313 | 0.606791 | 0.598313 | 0.121925 |
| Imagined + LearnedBelief JR | 0.543452 | 0.544200 | 0.543452 | 0.438095 |
| Real + LearnedBelief JR | 0.585813 | 0.603225 | 0.585813 | 0.116667 |
| Imagined + GT JR | 0.712103 | 0.696786 | 0.712103 | 0.446925 |
| Real + GT JR | 0.908532 | 0.904704 | 0.908532 | 0.092956 |
| FixedH1-H2 Oracle | 0.911210 | 0.908047 | 0.911210 | 0.082242 |

## Full Val strategy comparison

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen H1 | 0.459331 | 0.449009 |
| Imagined + Inferred JR | 0.512098 | 0.506333 |
| Real + Inferred JR | 0.552767 | 0.554133 |
| Imagined + LearnedBelief JR | 0.517181 | 0.511896 |
| Real + LearnedBelief JR | 0.544659 | 0.551313 |
| Imagined + GT JR | 0.626577 | 0.611871 |
| Real + GT JR | 0.753990 | 0.748774 |
| FixedH1-H2 Oracle | 0.755727 | 0.751001 |

## Requested gaps

- Learned belief gain (Imagined+LearnedBelief − Imagined+Inferred): **0.784 pp Accuracy / 0.788 pp Macro-F1**.
- Remaining identity gap (Imagined+GT − Imagined+LearnedBelief): **16.865 pp Accuracy / 15.259 pp Macro-F1**.
- Real+LearnedBelief → Real+GT Accuracy gap: **32.272 pp**.

## Scientific judgment

- Learned visited-history belief does not produce a large identity recovery (+0.784 pp Moving Accuracy).
- Real+LearnedBelief remains 32.272 pp below Real+GT, so identity/selector limitations remain even with real candidate evidence.
- Learned visited-history belief does not produce a large identity recovery (+0.784 pp Moving Accuracy). Real+LearnedBelief remains 32.272 pp below Real+GT, so identity/selector limitations remain even with real candidate evidence. The remaining limitation should be interpreted against the separate WM candidate-evidence gap, not as evidence to change WM-E in this experiment.

## Leakage/protocol

- reduced12 taxonomy; ALL_LEGAL candidate set; visited viewpoints excluded.
- No ST-GCN/WM-E retraining; existing visited RGB/DINO and counterfactual caches reused.
- `test_used=false`; no Test artifact was read.
