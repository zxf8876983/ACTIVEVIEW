# EXP050 — Joint Rollout Revision

Train-only set-aware Transformer revision policy over deterministic legal
candidate prefixes. The primary objective is listwise correctness-first
revision with a fixed candidate-correctness auxiliary term; the posterior head
is diagnostic only. Val is evaluated once after the fixed 20-epoch training
protocol.
