# Policy–Recognizer Coupling Audit — completed 2026-09-14

Completed a strict Frame0 single-step Train/Moving-Val-only audit on the
reduced12 eight-placement protocol. The action set was exactly current/Stay
plus the Stage-A legal candidate pool; each selected action was evaluated using
the selected real O1 skeleton only. Existing historical and retrained
old-adaptive-aware selectors, shared and old-adaptive heads, option caches and
frame-0 DINO/geometry caches were reused. No model was trained and Policy Test
was not read.

Moving Val contained 10,080 contexts (46,324 Train contexts were used only to
form old-adaptive static viewpoint and current/view-pair margin priors). The
adaptive-aware selector with the old adaptive head reached Accuracy/F1
0.526984/0.547002; the static prior reached 0.522917/0.546040 and the pair
prior 0.519940/0.543270. RGB shuffling lowered Accuracy by 4.256 pp and
geometry shuffling by 15.159 pp. The adaptive-aware gain over the best fixed
prior was only 0.407 pp, while old-adaptive versus shared on identical selected
views gained 1.865 pp (below the preregistered 2 pp threshold).

Decision: **MOSTLY FIXED VIEWPOINT PRIOR** with measurable but insufficient
instance-conditioned RGB signal. Preserve the 52.6984% result as a strict
Frame0 diagnostic, not as evidence for a standalone instance-conditioned NBV
main line. No follow-up experiment was started automatically.

Artifacts:
`experiments/reduced12_eight_placement_v1/policy_recognizer_coupling_audit/`

Task status: **CLEAN**; task-owned commit and push completed.
