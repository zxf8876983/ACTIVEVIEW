# EXP049 analysis

Frozen Val-only candidate scaling over 13,987 Stage-B trajectories (9,742
second-step eligible contexts). Legal prefixes are geometry ordered and exclude
the two visited viewpoints. At `ALL_LEGAL`, CF_CORRECTNESS_MLP reached
Accuracy 0.676628 / Macro-F1 0.629705 / mean regret 3.5199; the privileged
imagined-label oracle reached Accuracy 0.796525. Legal accuracy was not
monotonic across M (the entropy and confidence rules degraded as M grew), so
no M was selected from Val. See `result.json` for the complete table.

The regret reference is the unchanged Stage-B safe-oracle field used by the
canonical evaluator. Test, perception regeneration, WM-E retraining and
ST-GCN retraining were not used.
