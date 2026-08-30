# EXP020 — Frozen EXP014 Contextual-Latent Executed Gate

EXP020 tests whether the frozen EXP014 contextual candidate token contains
enough information to predict the executed-candidate target
`1[true_U2(c_hat)>0]`.  Only a small `Linear(129,64) → GELU → Linear(64,1)`
gate is trained.  The 128-D token is extracted immediately before the frozen
EXP014 utility head, and the 129th input is the frozen EXP014 predicted
utility for `c_hat`.

Training uses Train only for 30 fixed epochs (Adam, learning rate 1e-3,
batch size 256, seed 42).  Val is evaluated exactly once with the fixed
`sigmoid(logit)>0.5` rule.  Stage C-v0's first decision, EXP014's candidate
ranking, Stage D cache, perception and ST-GCN are frozen.  True U2 is used
only as the Train target and offline diagnostics; it is never a model input.

Test is locked and is not read or used.
