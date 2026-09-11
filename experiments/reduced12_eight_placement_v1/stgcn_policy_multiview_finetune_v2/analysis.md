# Reduced12 ST-GCN Policy-Train Fine-tuning v2

This is the final exposure/LR fairness rerun. The pretrained clean/BABEL reduced12 ST-GCN was fine-tuned end-to-end with 12-class cross entropy. Each record contributes 16 sampled observations per epoch; Moving Val legal-candidate global cross-entropy selects the checkpoint.

## Data and training

Policy Train contexts=46324, unique records=313, available observations=351499. Each epoch samples 5008 observations (16 per record), with unique observations per epoch=5008. Total optimizer steps=1185. Moving Val contexts=10080, legal observations=68702, all32 observations=322560.

## Matched Moving-Val results (Acc / global Macro-F1)

| Population | Frozen | Previous sparse fine-tune | New v2 fine-tune | New Δ vs Frozen |
|---|---:|---:|---:|---:|
| s0_current | 0.254266/0.235500 | 0.270337/0.281765 | 0.293254/0.305447 | +3.899pp/+6.995pp |
| s1 | 0.454266/0.444782 | 0.453075/0.473547 | 0.479167/0.501372 | +2.490pp/+5.659pp |
| legal_candidates_micro | 0.334474/0.330999 | 0.339452/0.365686 | 0.365171/0.389074 | +3.070pp/+5.807pp |
| all32 | 0.290858/0.284585 | 0.292699/0.307170 | 0.311257/0.325209 | +2.040pp/+4.062pp |

## Privileged ceiling

GT-TrueLogP Oracle=0.701587/0.715299; GT-Margin Oracle=0.760218/0.785369; Legal AnyCorrect coverage=0.760218 (7663/10080). AnyCorrect is coverage only.

## View generalization

FineTuned−Frozen gains: s0=+3.899pp, s1=+2.490pp, legal=+3.070pp, all32=+2.040pp; s1−legal=-0.580pp; s1−all32=+0.450pp.
Fixed-view mean/best/worst accuracy: Frozen=0.290858/r1_a6 (0.372421)/r4_a5 (0.191667); FineTuned=0.311257/r1_a7 (0.399702)/r4_a5 (0.200298).

## Training validity

Best epoch=15; epoch-1/5/10/15 train losses={'1': 4.565686503543129, '5': 2.9343841890745526, '10': 2.1475772691678396, '15': 1.9233115307892426}; legal-val selection criterion=Moving-Val legal candidate global cross-entropy.
The rerun provides limited +2–5pp domain benefit; do not add fine-tuning complexity without a separate decision.

## Flags

```text
policy_test_used=false
moving_val_used_for_checkpoint_selection_and_evaluation_only=true
new_rgb_or_skeleton_generated=false
pretrained_initialization=true
```
