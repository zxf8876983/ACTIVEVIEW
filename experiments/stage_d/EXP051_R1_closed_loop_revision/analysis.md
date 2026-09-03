# EXP051-R1 analysis

## Protocol and outcome

EXP050-R1 was retrained exactly once with the frozen architecture/configuration;
the runtime checkpoint SHA256 is recorded in the paired EXP050-R1 manifest.
Its canonical ALL_LEGAL H1 reproduction was Accuracy 0.685065 and Macro-F1
0.642002.

The Val all-view DINO spatial artifact covers 9,742 contexts × 32 viewpoints
(311,744 observations).  The legal rollout uses `VisitedObservationStore` and
recomputes WM-E and the candidate graph after the first real transition.

| protocol | Accuracy | Macro-F1 |
|---|---:|---:|
| H1 canonical trajectory | 0.685065 | 0.642002 |
| H2 terminal real observation | 0.675529 | 0.642612 |
| H2 fused real observations | 0.607370 | 0.552557 |

H2 executed a recurrent second step on 3,489 episodes; 1,766 action sequences
changed relative to H1 and candidate identity mismatch was zero.  Terminal
Accuracy decreased by 0.009536 while Macro-F1 increased by 0.000610, so this
run does not support a clear terminal improvement (descriptive Case D).

No Test data, Habitat rendering, perception regeneration, or substitute RGB
features were used.  Privileged H2 controls and paired rescue/harm statistics
were not run in this minimal legal rollout and are explicitly marked NOT_RUN.
