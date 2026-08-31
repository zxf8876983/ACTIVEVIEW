# EXP027 analysis

EXP027 completed one fixed 30-epoch Train run and one Val evaluation. The
oracle labels exactly matched the frozen Fixed-first Second-Step Oracle.

| Variant | Accuracy | Macro-F1 | Mean regret | Median regret | P90 regret | Headroom |
|---|---:|---:|---:|---:|---:|---:|
| EXP014 | 0.658254 | 0.610153 | 1.422463 | 0.003526 | 5.515663 | 0.783313 |
| EXP023 | 0.660470 | 0.608566 | 1.374664 | 0.003706 | 5.294162 | 0.786731 |
| EXP025 | 0.659898 | 0.607331 | 1.378650 | 0.004016 | 5.330633 | 0.784692 |
| EXP027 BC | 0.657039 | 0.608384 | 1.486694 | 0.002731 | 5.751852 | 0.769306 |
| Fixed-first Oracle | 0.771502 | 0.725081 | 0.586204 | 0.000000 | 1.699901 | 0.890887 |

On 9,742 Val second-step episodes, EXP027 predicted Stay/p2/p3 counts
4,411/2,923/2,408; three-way imitation accuracy was 0.493841 and binary
Move/Stay accuracy was 0.644837 (balanced accuracy 0.641147). Conditional on
both policies moving, candidate exact hit was 2,203/3,674 = 0.599619.
Selected-action mean true utility was -0.051969. Harmful moves (selected true
utility ≤ 0) numbered 2,442, while 1,803 beneficial oracle moves were missed.

Relative to EXP014, accuracy change was -0.001215 and mean regret change was
+0.064232; fixed-first-oracle recovery was -0.010732 (accuracy) and -0.076808
(regret). Thus direct unweighted action cloning did not improve the frozen
policy under this representation, despite avoiding a severe Stay collapse.

True U2 was used only to form Train labels and offline Val diagnostics. RGB was
read only for visited s0/s1 keys; no future-candidate RGB/depth/skeleton or Test
data was accessed. Full provenance and diagnostics are in `result.json`.
