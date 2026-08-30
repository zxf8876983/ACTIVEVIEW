# Handoff

Status: CLEAN

EXP018 executed-candidate gate alignment is complete as a Val-only,
no-training diagnostic and is **INCONCLUSIVE**. It froze the Stage C-v0 first
action/p1 and corrected EXP014 candidate ranking, then compared an
any-positive true-U2 gate with the true utility of the candidate that EXP014
would actually execute. There were 1,196 ranking-induced gate mismatches among
9,742 v0-Move episodes; the executed-candidate oracle reached 0.743119
Accuracy / 0.761339 mean regret.

Runtime result:
`ACTIVEVIEW_DATA_ROOT/experiments/stage_d/EXP018_executed_candidate_gate_alignment/`.
Compact experiment record:
`experiments/stage_d/EXP018_executed_candidate_gate_alignment/`.

Test remains locked and was not read. No training, perception regeneration,
Habitat rendering, ST-GCN retraining or Stage A/B/C-v0/EXP014/EXP015/EXP016/
EXP017 modification was performed. EXP019 has not been started; human review
is required before any next experiment.
