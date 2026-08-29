# EXP001 — Utility-Gap-Aware Ranking

## Status

PLANNED

## Question

Does utility-gap-aware ranking reduce harmful large-gap candidate mistakes?

## Hypothesis

Adding a stay-inclusive, utility-gap-weighted pairwise ranking term to the
existing Set Ranker objective will reduce large-gap Val regret while retaining
recognition quality.

## Baseline

Stage C-v0 `SetUtilityRanker` with SmoothL1 regression and stay-inclusive
listwise ranking. The frozen Val baseline is summarized in `baseline.json`.

## Single Change

Only add the utility-gap-weighted pairwise ranking loss. Stage A/B outputs,
features, architecture, sampler, split and ST-GCN remain unchanged.

## Frozen

- Stage A and Stage B accepted artifacts;
- Stage C-v0 feature schema and Set Ranker architecture;
- record-balanced sampler and 589/197/194 record split;
- frozen ST-GCN checkpoint and 32-view candidate protocol.

## Parameters

`lambda_gap = 1.0`, `tau_gap = 1.0`, `max_gap_weight = 10.0`.

The gap term is normalized over all valid ordered candidate pairs in each
training batch (global pair normalization, not per-Episode normalization).

## Acceptance

On Val, large-gap mean regret must decrease by at least 5%; at least one of C2
rate, p90 regret, or positive headroom capture must improve; and recognition
Macro-F1 may not drop by more than 0.5 percentage points. The analysis must
report the complete comparison and use `test_used: false`.

## Run

```bash
bash experiments/stage_c_v1/EXP001_gap_aware_ranking/run.sh
```

Do not run before human review. The script does not evaluate Test.

## Result

Pending.
