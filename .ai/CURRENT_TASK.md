# Current Task

## H1 Disambiguation Ranker — completed 2026-09-07

Implemented and ran the reduced14 Train/Val-only deployable H1
disambiguation-ranker experiment. The ranker uses only the frozen s0 ST-GCN
feature/posterior and 9-D candidate geometry; frozen history-identity margins
from real archived candidates are Train supervision targets. Val checkpoint
selection used history-identity Macro-F1.

Results are recorded in
`experiments/reduced14_eight_placement_v1/h1_disambiguation_ranker/`.
Train contexts: 44,248; Val moving contexts: 14,809. The selected ranker
reached identity Accuracy/Macro-F1 0.418732/0.428180, below Frozen H1
0.482815/0.496501. Test was not read and no formal WM-E, JR or ST-GCN
checkpoint was modified.

Status: CLEAN. No follow-up experiment is authorized automatically.
