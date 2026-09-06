# Reduced14 History Action Identity (Val)

This diagnostic trains two fixed two-layer MLPs on Train contexts and evaluates only the Val moving contexts. Formal WM-E, Joint Revision and ST-GCN checkpoints remain unchanged.

Train contexts: 44248; Val moving contexts: 14809.

## Frozen ST-GCN / history classifiers

Per-class `accuracy` is class recall (true positives divided by class support). Each row in `result.json` also contains the full 14x14 confusion matrix.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| S1_only_frozen_ST_GCN | 0.415085 | 0.396831 |
| Posterior_history_MLP | 0.464245 | 0.475459 |
| Feature_history_MLP | 0.484300 | 0.500795 |

## Privileged candidate selector

The selector compares Stay and legal candidates directly. Candidate recognition is archived `true_logp` and is used only for this offline privileged diagnostic; classifier inputs never contain true_logp.

| Selector | Positive action hit | Stay rate | Terminal Accuracy | Terminal Macro-F1 |
|---|---:|---:|---:|---:|
| S1_only_frozen_ST_GCN | 0.415085 | 0.982038 | 0.415085 | 0.396831 |
| Posterior_history_MLP | 0.430414 | 0.140995 | 0.430414 | 0.434256 |
| Feature_history_MLP | 0.473901 | 0.125532 | 0.473901 | 0.483283 |
| PrivilegedJR | 0.595584 | 0.108380 | 0.595584 | 0.604960 |
| GTLabelPrivilegedJR | 0.872037 | 0.081504 | 0.872037 | 0.869523 |
| SafeOracle | 0.875886 | 0.071848 | 0.875886 | 0.873737 |

## Interpretation

Posterior-history accuracy change versus S1-only: +0.049159; Feature-history change: +0.069215.
Feature-history privileged selector terminal-accuracy change versus S1-only: +0.058816.

If Feature-history materially exceeds Posterior-history, frozen ST-GCN features retain identity information that is absent from the final posterior and a learned belief refiner would be justified. If both remain close to S1-only, the next direction should be information-seeking/disambiguation viewpoint selection rather than another history classifier.

The privileged JR and GT-label Privileged JR rows are reused from the preceding Val-only diagnostic for comparison; they are not retrained by this script.

Leakage audit: `test_used=false`; no Test path is loaded; Val is not used for training; no formal checkpoint is modified.

Classes (14): walk, sit, stand up, bend, crawl, stumble, kneel, clap, throw, clean something, kick, knock, punch, touching face
