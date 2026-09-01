# EXP039 — Deployable Belief-Conditioned View Risk

Train/Val-only class-conditioned CE and correctness heads. Legal planners use
the frozen ST-GCN belief (MAP, full soft belief, or fixed top-3); the GT-label
head is a privileged upper bound. True CE is evaluator-only and no unvisited
view output is used as an input.
