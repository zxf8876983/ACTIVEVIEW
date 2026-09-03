# EXP053 — Full-History Counterfactual Recognition Model

Train-only direct prediction of a candidate's frozen ST-GCN recognition
log-probability from real visited observation history and a legal candidate
descriptor. The frozen Joint Revision, ST-GCN and Stage C-v0 first action are
used for the Val closed-loop audit. Terminal recognition is always taken from
the archived real observation. Test, H3, Habitat rendering and perception
regeneration are not used.

Runtime checkpoints and RGB caches remain outside Git under the configured
`ACTIVEVIEW_DATA_ROOT`/MG08 data roots.
