# EXP054 — H2 World-Model / Decision Ceiling Decomposition

The first Stage-C-v0 action and the first Joint Revision decision are frozen.
The Val moving population compares the original WM-E + frozen Joint Revision,
true future recognition + frozen Joint Revision, true future recognition + a
Train-matched Joint Revision, and a privileged GT-label action oracle. Terminal
recognition is always read from archived real skeleton/ST-GCN observations.
Test, H3, perception regeneration and WM training are not used.

The matched Joint Revision is the only model trained in this analysis: it uses
Train true-recognition inputs and is evaluated on Val solely as a privileged
ceiling diagnostic. The original WM-E, Stage C-v0 first action, terminal ST-GCN
recognition, and candidate graph remain frozen.
