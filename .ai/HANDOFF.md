# Handoff

Status: CLEAN

EXP016 preparation is complete. The new analysis-only code keeps the frozen
Stage C-v0 first decision and separates the second-step Stay/Move gate from
p2/p3 candidate identity. It accepts Val only and rejects Test at the CLI.

No real EXP016 Val analysis, training, Test access, Habitat rendering,
perception regeneration or upstream artifact modification was performed.
Await human code review and explicit authorization before running the real
Val command.

The preparation also now matches frozen EXP015 true-U2 tie behavior (cached
candidate order) and rejects any Stage B/v0 or second-step cache/prediction
episode-ID mismatch before analysis.
