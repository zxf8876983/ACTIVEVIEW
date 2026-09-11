# Reduced12 Learned H2 Sequential Selector

Moving Val contexts: 10080. The policy makes decisions only after chunks 0 and 1 (5 and 10 observed frames), then holds the final viewpoint through t30. This is a discrete-time view-switch approximation, not continuous robot motion.

## Policy

Input dimension is 282: accumulated MeanFeature (256), current soft posterior (12), current-view/decision/candidate IDs (3 normalized scalars), and existing candidate geometry (11). The scorer is a shared 256-hidden GELU MLP with a scalar output.
Training target is archived frozen-ST-GCN candidate true-class log-probability gain over the current view; future observations are targets only. Train examples: 160798.

## Moving Val t30

| Method | Accuracy | Macro-F1 | Move rate | Contexts |
|---|---:|---:|---:|---:|
| Stay | 0.311210 | 0.337410 | 0.000000 | 10080 |
| Random-H2 | 0.335813 | 0.367500 | 0.426538 | 10080 |
| Learned-H2 | 0.344345 | 0.373911 | 0.504564 | 10080 |
| Privileged-Oracle-H2 | 0.434325 | 0.473973 | 0.409673 | 10080 |
| Privileged-Oracle-Full | 0.487798 | 0.527362 | 0.290813 | 10080 |

Learned-H2 − Random-H2 = +0.853pp Accuracy / +0.641pp Macro-F1.
Privileged Oracle-H2 − Learned-H2 = +8.998pp Accuracy / +10.006pp Macro-F1.
Privileged Oracle-Full is 0.487798/0.527362; the Full oracle is retained only as a ceiling reference.

## Leakage sanity

The scorer inference tensor is assembled only from history features/posterior, viewpoint IDs, decision index and candidate geometry. It never receives GT labels, future candidate skeletons, future features/logits, or hard predicted actions. Policy Test was not read.

## Answers

1. Learned-H2 exceeds Random-H2 by +0.853pp Accuracy.
2. The remaining H2 headroom to the privileged H2 oracle is 8.998pp; this is the direct selector gap under the same frozen MeanFeature terminal recognizer.
3. The policy is intentionally a minimal regression scorer rather than RL, MCTS, beam search or a new recognizer. Further policy work should depend on whether this Val-only gap is scientifically meaningful.

## Flags

```text
policy_test_used=false
future_candidate_observation_in_policy_input=false
future_candidate_feature_or_logits_in_policy_input=false
gt_action_in_policy_input=false
recognizer_modified=false
new_data_generated=false
```
