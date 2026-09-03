# EXP052 — Diverse-History WM-E

## Protocol

WM-E-DH uses the frozen WM-E architecture, recognition-aware loss (pose + 0.10 ST-GCN KL), and DINO spatial history features. Train sampling is deterministic (seed 42): 4 distinct ordered real history pairs × 4 excluded candidate targets per context. No Joint Revision, ST-GCN, perception, Habitat or Test artifact was changed.

## Fidelity

| Model | History | Agreement | Pearson | Spearman | KL | L1 |
|---|---|---:|---:|---:|---:|---:|
| Original WM-E | h0 [s0,s1] | 0.596064 | 0.738313 | 0.767008 | 1.027694 | 0.905417 |
| WM-E-DH | h0 [s0,s1] | 0.316376 | 0.302647 | 0.346792 | 1.971956 | 1.408939 |
| Original WM-E | h1 [s1,v1] | 0.498571 | 0.645801 | 0.676367 | 1.302844 | 1.066320 |
| WM-E-DH | h1 [s1,v1] | 0.227858 | 0.240198 | 0.253896 | 2.136991 | 1.519875 |

WM-E-DH decreases both h0 and h1 fidelity rather than correcting recurrent shift.

## H2 real terminal recognition

| Population | Original H2 Acc/F1 | WM-E-DH H2 Acc/F1 |
|---|---:|---:|
| Moving (9,742) | 0.675529 / 0.642612 | 0.656025 / 0.613289 |
| Full (13,987) | 0.695503 / 0.649220 | 0.681919 / 0.624581 |

Second-step action changed in 2194 episodes; DH rescued 289, harmed 479, net -190.

## Decision

**Case C (with an h0 regression):** simple diverse-history sampling did not improve h1 fidelity and reduced initial-history fidelity. It provides no evidence that this intervention resolves recurrent distribution shift; a history-aware architecture or mixed canonical/recurrent training would require separate human authorization.

`test_used=false`; no H3 was run.
