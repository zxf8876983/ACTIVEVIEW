# Adaptive Head × Frame0 NBV Combination Audit — completed 2026-09-14

Completed the strict single-step Train/Moving-Val audit over 46,324 Train and
10,080 Moving-Val contexts. The action set was exactly current/Stay plus the
Stage-A legal candidate pool. The selector saw only current frame-0 DINO
context, legal geometry and the existing visibility auxiliary; terminal
recognition used the selected real O1 observation alone. Policy Test and new
perception artifacts were not read or generated.

The historical selector reproduction gate passed exactly: shared-head strict
Frame0 reached Accuracy/Macro-F1 0.506151/0.503722. The old adaptive head
reproduced its matched s1 performance at 0.530952/0.555852, but legal-candidate
accuracy fell to 0.369800, triggering the distribution-specialization flag.
The new record-balanced legal-candidate head reached only 0.488393/0.493137
on s1 and 0.364633/0.367584 on legal candidates.

Strict Frame0 results (Accuracy/Macro-F1) were: random shared
0.365079/0.364567, historical selector + shared 0.506151/0.503722, the same
selector + old adaptive 0.523115/0.542017, retrained old-adaptive selector
0.526984/0.547002, and retrained balanced-adaptive selector
0.493452/0.497652. Thus the old adaptive head can improve the matched
selector, while the balanced head does not generalize to legal candidates;
the balanced adaptive-aware branch is -1.270pp below the historical strict
baseline. Gain decomposition gives head gain -1.091pp, shared-selector gain
14.107pp, adaptive-selector gain 13.929pp and synergy -0.179pp.

Candidate-only GT-TrueLogP AnyCorrect coverage was 77.173% (shared), 78.542%
(old adaptive) and 74.712% (balanced); including Stay, GT-TrueLogP oracle
Accuracy was 75.317%, 77.480% and 74.345%, respectively. Candidate ranking
was highly correlated between heads (shared/old Spearman 0.908; shared/balanced
0.954), so the balanced head's failure is not a useful new utility landscape.

Decision: **KILL ADAPTIVE-HEAD × NBV COMBINATION** under the preregistered
<+1pp strict-selector criterion. Preserve the old head as a matched s1
recognition diagnostic only; do not promote it as a general NBV recognizer.
No follow-up experiment was started automatically.

Report/script:
`experiments/reduced12_eight_placement_v1/adaptive_head_frame0_nbv_audit/`
and
`activeview/scripts/experiments/run_reduced12_adaptive_head_frame0_nbv_audit.py`.

Task status: **CLEAN**.
