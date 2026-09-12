# Prefix-length causal mixed-view oracle sweep — completed 2026-09-13

Ran a Val-only, no-training reduced12 sweep for L=5, 8, 10, 12 and 15 using
the exact current/Stay plus Stage-A legal candidate pool. Stay is
current[0:30]; a candidate is current[0:L] + candidate[L:30]. The frozen
reduced12 ST-GCN and shared head were unchanged, no perception artifacts were
regenerated, and Policy Test was not read. This is a discrete-time view-switch
approximation rather than continuous navigation.

Moving-Val GT-TrueLogP oracle Accuracy/Macro-F1: L5 0.687599/0.671197,
L8 0.653571/0.633511, L10 0.633234/0.608946, L12 0.614087/0.585399,
L15 0.585317/0.554597. GT-Margin/AnyCorrect coverage was 0.720040, 0.684325,
0.664087, 0.646825 and 0.613591 respectively. FullView reference was
0.753175/0.755358. L8 is the longest prefix satisfying the preregistered
0.65 ceiling and <0.10 FullView drop; L10 is below the viability threshold.

High-occlusion (frame-0 Stay visibility bottom tertile) is best at L5
(0.534087), and motion evidence increases monotonically with L. Boundary
amplification remains about 8.0–9.0× across lengths. This supports retaining
the longer-prefix family with L8 as the next candidate length, but no selector
was trained automatically.

Report:
`experiments/reduced12_eight_placement_v1/short_prefix_length_oracle_sweep/`

Task status: **CLEAN**.
