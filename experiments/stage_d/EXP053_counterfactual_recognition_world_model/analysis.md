# EXP053 analysis

## Protocol

The model predicts future frozen ST-GCN recognition log-probabilities directly
from real visited skeleton, RGB, recognition and geometry tokens. Candidate
identity is never selected from true future recognition. Training used Train
only; Val was evaluated once after the final epoch. The existing Joint
Revision and first-step policy were frozen, and terminal recognition used only
the archived real skeleton observation.

## Results

The requested `conda` `habitat` environment was used, but CUDA/NVIDIA was not
available (`torch.cuda.is_available() == false`), so the fixed protocol ran on
CPU; no GPU result is claimed. EXP053 was trained for 15 epochs (124,520 samples: 116,532 canonical and
7,988 rollout-matched samples with complete existing RGB history cache). The
rollout selector identified 12,090 Train v1 transitions; 10,093 lacked an
approved cached v1 RGB embedding and were skipped rather than imputed.

| Model | History | Agreement | Pearson | Spearman | KL | L1 |
|---|---|---:|---:|---:|---:|---:|
| Original WM-E | h0 | 0.596064 | 0.738313 | 0.767008 | 1.027694 | 0.905417 |
| EXP053 | h0 | 0.435608 | 0.418994 | 0.479604 | 2.417143 | 1.144041 |
| Original WM-E | h1 | 0.498571 | 0.645801 | 0.676367 | 1.302844 | 1.066320 |
| EXP053 | h1 | 0.374180 | 0.397477 | 0.433348 | 2.601120 | 1.246926 |

| Population | Original H2 Acc/F1 | EXP053 H2 Acc/F1 |
|---|---:|---:|
| Moving (9,742) | 0.675529 / 0.642612 | 0.563231 / 0.513049 |
| Full (13,987) | 0.695503 / 0.649220 | 0.617287 / 0.550369 |

Compared with original WM-E + frozen Joint Revision, 5,979 moving trajectories
changed; 553 were rescued and 1,647 were harmful (net −1,094).

## Decision

**Observed Case C (partial run):** direct recognition-space prediction was
below the original WM-E on both h0 and h1 fidelity and reduced H2 recognition.
Because the approved Train RGB cache covered only 1,997 of 12,090 rollout v1
histories, this is a partial rapid result rather than a fully balanced
canonical/rollout mixture. It nevertheless provides no positive evidence that
replacing skeleton prediction with direct recognition prediction is beneficial
under the current representation and frozen Joint Revision.

`test_used=false`; no H3, Test, Habitat rendering or perception regeneration.
