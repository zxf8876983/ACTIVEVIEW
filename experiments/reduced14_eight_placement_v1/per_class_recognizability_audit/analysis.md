# Reduced14 per-class recognizability and viewpoint-recoverability audit

Val-only audit using the frozen reduced14 ST-GCN and existing reduced14 eight-placement ActiveView Val utility/cache. No model was trained, no data was regenerated, and no policy Test artifact was opened.

## Complete per-class table

| Action | Dev Val Acc/F1 | ActiveView s0 Acc/F1 | Legal-view mean Acc | Random legal Acc | H0 CandidateOracle Acc | H0 SafeOracle Acc | AnyCorrect | Correct-view ratio | # correct legal views | Recoverability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| walk | 0.7000/0.6562 | 0.4981/0.2805 | 0.4833 | 0.4907 | 0.9346 | 0.9556 | 0.9574 | 0.4838 | 3.598 | 0.4648 |
| sit | 0.7333/0.7719 | 0.6691/0.4438 | 0.6731 | 0.6846 | 0.9728 | 0.9877 | 0.9877 | 0.6738 | 5.011 | 0.3031 |
| stand up | 0.6333/0.7037 | 0.4704/0.5449 | 0.4779 | 0.4519 | 0.8574 | 0.8765 | 0.8765 | 0.4747 | 3.557 | 0.4247 |
| bend | 0.6000/0.5538 | 0.2130/0.1990 | 0.2164 | 0.2210 | 0.6444 | 0.6722 | 0.6722 | 0.2175 | 1.611 | 0.4512 |
| crawl | 1.0000/0.9333 | 0.3630/0.4996 | 0.3692 | 0.3481 | 0.8494 | 0.8840 | 0.8852 | 0.3654 | 2.748 | 0.5358 |
| stumble | 0.6667/0.6667 | 0.1091/0.1640 | 0.0931 | 0.0995 | 0.3573 | 0.3937 | 0.3951 | 0.0911 | 0.693 | 0.2942 |
| kneel | 0.4000/0.5000 | 0.1370/0.1802 | 0.1303 | 0.0988 | 0.4753 | 0.5037 | 0.5037 | 0.1240 | 0.970 | 0.4049 |
| clap | 0.7000/0.6364 | 0.2037/0.2698 | 0.2188 | 0.2140 | 0.7233 | 0.7613 | 0.7613 | 0.2214 | 1.629 | 0.5473 |
| throw | 0.8333/0.8065 | 0.2599/0.2854 | 0.2785 | 0.2784 | 0.6907 | 0.7272 | 0.7278 | 0.2746 | 2.073 | 0.4488 |
| clean something | 0.4000/0.4138 | 0.1389/0.1826 | 0.1308 | 0.1370 | 0.4864 | 0.5290 | 0.5290 | 0.1310 | 0.974 | 0.3920 |
| kick | 0.8333/0.8333 | 0.4920/0.3516 | 0.4688 | 0.4617 | 0.9086 | 0.9302 | 0.9302 | 0.4706 | 3.490 | 0.4685 |
| knock | 0.8571/0.8571 | 0.0975/0.1546 | 0.1090 | 0.1111 | 0.4728 | 0.5074 | 0.5111 | 0.1091 | 0.811 | 0.3963 |
| punch | 0.7143/0.7143 | 0.2185/0.2706 | 0.2102 | 0.2123 | 0.6475 | 0.6901 | 0.6932 | 0.2129 | 1.565 | 0.4778 |
| touching face | 0.5333/0.5333 | 0.1167/0.1468 | 0.1184 | 0.1099 | 0.4735 | 0.5043 | 0.5049 | 0.1167 | 0.881 | 0.3944 |

`Dev Val Acc/F1` is the frozen ST-GCN development-Val class recall/F1 on 270 samples. `ActiveView s0` and all H0/legal-view quantities use the 19,440 records-only ActiveView Val contexts and their embedded frozen predictions. `AnyCorrect` is reported including the current s0 view; candidate-only values remain in `result.json`.

## Diagnostic groups

Thresholds are transparent distributional triage thresholds: clean-high = class-median F1 (0.6852), recoverability-high = class-median (0.4367), clean-low = 25th-percentile F1 (0.5745), and AnyCorrect-low = class median (0.7105). They are not deletion rules.

- **A — recommended retain:** crawl, throw, kick, punch
- **B — manual review:** walk, sit, stand up, stumble, clap, knock
- **C — potential removal candidates:** bend, kneel, clean something, touching face

At most four lowest clean-F1/AnyCorrect classes from group C are listed as potential candidates (not automatically removed): clean something, kneel, touching face, bend.

## Requested action checks

The following actions receive a focused metric readout because they involve hand/object interaction or ground-level posture. These are hypotheses for human inspection only; the audit does not add an artificial rule or establish H36M17 insufficiency from metrics alone.

| Action | Clean F1 | s0 Acc | H0 AnyCorrect | Recoverability |
|---|---:|---:|---:|---:|
| crawl | 0.9333 | 0.3630 | 0.8852 | 0.5358 |
| kneel | 0.5000 | 0.1370 | 0.5037 | 0.4049 |
| clap | 0.6364 | 0.2037 | 0.7613 | 0.5473 |
| throw | 0.8065 | 0.2599 | 0.7278 | 0.4488 |
| clean something | 0.4138 | 0.1389 | 0.5290 | 0.3920 |
| knock | 0.8571 | 0.0975 | 0.5111 | 0.3963 |
| touching face | 0.5333 | 0.1167 | 0.5049 | 0.3944 |

## Scientific interpretation

- A high clean F1 with a large positive viewpoint-recoverability gap supports retaining the action while treating viewpoint/occlusion as a plausible limiting factor.
- Mixed cases belong in manual review; a low ActiveView score alone is not evidence for deletion.
- Potential deletion candidates require both low clean recognizability and low H0 AnyCorrect, and should be checked independently of final-policy performance.
- This audit does not provide evidence by itself that reduced14 should be shortened to 10–12 classes; any taxonomy change requires a separate protocol decision.
- No Test score, Ours/final-policy score, or policy outcome was used to select a class group.

## Leakage/runtime flags

- `test_used = false`
- `training_performed = false`
- `taxonomy_modified = false`
- `data_regenerated = false`
