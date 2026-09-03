# EXP051-R2 analysis

## Protocol

The 9,742 v0-Move episodes were evaluated as a paired population. The 4,245
frozen v0-Stay episodes were then added unchanged to form the 13,987 full
population. H1 terminal is the real archived observation at `s1` or `v1`; H2
terminal is the real archived observation at the final visited view. Fused
predictions are the argmax of the mean probability over real visited views.
The archived `true_logp` arrays are the frozen ST-GCN outputs for those real
skeleton observations; WM-E outputs never enter final recognition.

## Recognition results

| population | H1 real Acc | H1 real Macro-F1 | H2 real Acc | H2 real Macro-F1 |
|---|---:|---:|---:|---:|
| moving subset (n=9,742) | 0.661568 | 0.632699 | 0.675529 | 0.642612 |
| full population (n=13,987) | 0.685780 | 0.643640 | 0.695503 | 0.649220 |

On the moving subset, H2−H1 is +0.013960 Accuracy and +0.009913
Macro-F1. Paired outcomes are 369 rescued versus 233 harmful (net +136);
McNemar exact two-sided p = 3.30e-08. A 10,000-replicate paired bootstrap
(seed 42) gives Accuracy Δ 95% CI [0.009033, 0.018887] and Macro-F1 Δ 95% CI
[0.002913, 0.017177].

## Real-view fusion

| population | H1 fused Acc | H1 fused Macro-F1 | H2 fused Acc | H2 fused Macro-F1 |
|---|---:|---:|---:|---:|
| moving subset | 0.599364 | 0.545336 | 0.607370 | 0.552557 |
| full population | 0.642454 | 0.574306 | 0.648030 | 0.579441 |

Moving-subset fused ΔAccuracy = +0.008007 and ΔMacro-F1 = +0.007220;
McNemar p = 2.39e-06 (174 rescued, 96 harmful).

## Recurrent WM-E fidelity

| history | Top-1 agreement | true-label logp Pearson | Spearman | KL(real‖imagined) | probability L1 |
|---|---:|---:|---:|---:|---:|
| h0 `[s0,s1]` (275,732 candidate views) | 0.596064 | 0.738312 | 0.767009 | 1.027688 | 0.905418 |
| h1 `[s1,v1]` (95,529 candidate views) | 0.498571 | 0.645801 | 0.676368 | 1.302844 | 1.066320 |

The recurrent history shows lower agreement/correlation and larger KL/L1 than
the initial history, consistent with a WM-E distribution shift. H2 nevertheless
improves real terminal recognition on this paired population, so the observed
result is descriptive evidence for Case A (closed-loop terminal gain), with a
simultaneous fidelity warning rather than a clean WM-E success claim.

## Validity

`test_used=false`, `training_performed=false`, Habitat rendering and perception
regeneration were false, and candidate/observation access remained frozen to
the R1 protocol. No H3 was run.
