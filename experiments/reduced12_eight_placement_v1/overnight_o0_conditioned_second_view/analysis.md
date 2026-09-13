Experiment:
O0-Conditioned Complementary Second-View Selection

Moving Val:
10080 contexts

Policy Test:
false

Observation protocol:
finite full-observation budget

O0:
actually acquired complete current-view observation

Second candidate:
selected before acquiring its observation

Future candidate observation used by selector:
false

GT action used by selector:
false

O0 frozen feature:
allowed

O0 soft posterior:
allowed

Hard predicted action as formal intermediate:
false

Recognizer:
frozen ST-GCN + frozen shared head

B2 fusion:
fixed MeanLogP unless explicitly stated in fusion ablation

Continuous robot/human time synchronization:
not modeled

## Headline

- Historical raw MeanLogP candidate ranking vs normalized pair ranking agreement: 0.914385. The current-view term cancels in the historical raw target, so it is not complementarity-aware.
- Historical raw oracle: 0.681647 Acc / 0.675099 Macro-F1.
- Pair-normalized TrueLogP oracle: 0.681052 Acc / 0.674501 Macro-F1.
- Pair-margin oracle: 0.706647 Acc / 0.701819 Macro-F1.
- Pair AnyCorrect coverage: 0.706647 (7123 contexts).
- Random B2 MeanLogP: 0.429762 Acc. Best learned branch (Feature+Posterior+Geometry-PairLogP): 0.517361 Acc.
- O0 shuffle accuracies: normal=0.517361, feature_shuffled=0.499802, posterior_shuffled=0.517560, both_shuffled=0.499306.
- Conditional B3 status: COMPLETED. When run, learned B3 accuracy is 0.558036.

## Decision

The pre-registered decision is based on the best learned B2 accuracy, its gain over Random B2, and the O0-information shuffle. Best learned accuracy is 0.517361; gain over Random B2 is +8.760 pp; normal minus both-shuffled is +1.806 pp. This run is reported as **KEEP O0-CONDITIONED SECOND-VIEW SELECTION** under the specified gates.

The pair oracle headroom is real when pair-margin substantially exceeds Random B2, but a low learned score means complementary second views remain difficult to predict from O0 plus geometry. No Test artifact was read and no new perception data was generated.
