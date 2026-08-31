# EXP028 diagnostic figures

## `figures/margin_vs_exp027_accuracy.png`

- Purpose: show frozen EXP027 three-way imitation accuracy across preregistered oracle top-1 margin bins.
- Data: `result.json`, Val eligible episodes (n=9,742).
- Observation to inspect: accuracy rises only for the highest-margin bin and remains low in all lower bins.
- Caveat: one held-out Val split; no uncertainty interval or causal claim is implied.

## `figures/margin_vs_nn25_entropy.png`

- Purpose: show 25-nearest-Train-neighbor oracle-label entropy by the same fixed margin bins.
- Data: `result.json`, Train-only cosine index and Val labels.
- Observation to inspect: entropy remains high even at larger margins, although it declines in the `[2,+inf)` bin.
- Caveat: cosine NN is a fixed diagnostic, not a learned policy or a proof of intrinsic unpredictability.
